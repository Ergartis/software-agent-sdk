"""A message sent while a TOOL is running must be accepted, and must land.

Companion to ``test_astep_releases_state_lock.py``, which covers the LLM network
call. That fix (``LocalConversation._released_state_lock_during_io`` applied at
``Agent.astep``) made the agent responsive while *thinking*. It was still
unresponsive while *doing*: ``_aexecute_actions`` runs the tool batch with the
run loop's state lock held, and ``send_message()`` needs that lock, so a message
sent during a tool call blocked for the tool's full duration.

The case this comes from: an agent looping on ``dockerd`` inside the ``terminal``
tool for 23 minutes. Every attempt to reach it was swallowed for those 23 minutes,
which read as "the agent went silent". The model was fine — the inbox was shut.
A long tool call is not an edge case; test suites, builds, installs and container
startups are all one tool invocation.

The fix is an inbox rather than releasing the lock, and the difference matters:
accepting the message must be instant, but INSERTING it has to wait for a safe
point in the event log. Appending it between an assistant ``tool_call`` and its
tool result would build a message sequence providers reject. So these tests assert
the two halves separately — accepted immediately, appended at the step boundary.
"""

import threading
import time
from collections.abc import Sequence
from typing import ClassVar

import pytest

from openhands.sdk.agent import Agent
from openhands.sdk.conversation import Conversation
from openhands.sdk.conversation.impl.local_conversation import LocalConversation
from openhands.sdk.llm import ImageContent, Message, MessageToolCall, TextContent
from openhands.sdk.testing import TestLLM
from openhands.sdk.tool import (
    Action,
    Observation,
    Tool,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)


class _ProbeAction(Action):
    command: str


class _ProbeObservation(Observation):
    result: str

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(text=self.result)]


class _SendingExecutor(ToolExecutor[_ProbeAction, _ProbeObservation]):
    """Stands in for any slow tool: a test run, a build, ``docker compose up``.

    While it "runs", it sends a message from another thread — which is how the
    agent-server does it — and records how long that call took to return.
    """

    def __init__(self, convo_box: list, probe: dict, messages: list[str]):
        self._convo_box = convo_box
        self._probe = probe
        self._messages = messages

    def __call__(self, action: _ProbeAction, conversation=None) -> _ProbeObservation:
        convo = self._convo_box[0]
        self._probe["ran"] = True

        def _send():
            started = time.monotonic()
            for text in self._messages:
                convo.send_message(text)
            self._probe["elapsed"] = time.monotonic() - started

        t = threading.Thread(target=_send)
        t.start()
        # A generous join: if send_message blocks on the lock this never returns
        # until the tool does, and the assertion below reports it as a timeout
        # rather than hanging the suite.
        t.join(timeout=5.0)
        self._probe["send_returned"] = not t.is_alive()

        # The message must NOT have been appended yet — mid-tool is exactly the
        # unsafe point, between the tool_call and its result.
        self._probe["events_during_tool"] = len(convo.state.events)
        return _ProbeObservation(result=f"ran {action.command}")


class _ProbeTool(ToolDefinition[_ProbeAction, _ProbeObservation]):
    name: ClassVar[str] = "probe_tool"

    @classmethod
    def create(
        cls, conv_state=None, *, executor: ToolExecutor, **params
    ) -> Sequence["_ProbeTool"]:
        return [
            cls(
                description="A tool that sends a message while it runs",
                action_type=_ProbeAction,
                observation_type=_ProbeObservation,
                executor=executor,
            )
        ]


def _conversation(
    probe: dict, convo_box: list, messages: list[str]
) -> LocalConversation:
    tool = _ProbeTool.create(executor=_SendingExecutor(convo_box, probe, messages))[0]
    register_tool("probe_tool", tool)

    llm = TestLLM.from_messages(
        [
            Message(
                role="assistant",
                content=[TextContent(text="")],
                tool_calls=[
                    MessageToolCall(
                        id="call_1",
                        name="probe_tool",
                        arguments='{"command": "sleep 1380"}',
                        origin="completion",
                    )
                ],
            ),
            Message(role="assistant", content=[TextContent(text="done")]),
        ]
    )
    agent = Agent(llm=llm, tools=[Tool(name="probe_tool")])
    convo = Conversation(agent=agent)
    assert isinstance(convo, LocalConversation)
    convo._ensure_agent_ready()
    convo_box.append(convo)
    convo.send_message("build it")
    return convo


def _user_texts(convo: LocalConversation) -> list[str]:
    return [
        c.text
        for e in convo.state.events
        if getattr(e, "llm_message", None) and e.llm_message.role == "user"
        for c in getattr(e.llm_message, "content", [])
        if isinstance(c, TextContent)
    ]


@pytest.mark.asyncio
async def test_send_message_during_a_tool_call_does_not_block():
    probe: dict = {}
    convo_box: list = []
    convo = _conversation(probe, convo_box, ["status update please?"])

    convo._step_holds_state_lock = True
    with convo._state:
        await convo.agent.astep(convo, on_event=convo._on_event)

    assert probe.get("ran"), (
        "the probe tool never ran; the LLM script produced no tool call"
    )
    assert probe.get("send_returned") is True, (
        "send_message() did not return while a tool was executing — it is still "
        "blocking on the run loop's state lock for the tool's full duration"
    )
    assert probe["elapsed"] < 0.5, (
        f"send_message() took {probe['elapsed']:.2f}s during a tool call; it must be "
        "accepted immediately, not held until the tool finishes"
    )


@pytest.mark.asyncio
async def test_message_is_not_inserted_mid_tool_but_lands_at_the_step_boundary():
    """The insertion point is why this is an inbox and not a lock release."""
    probe: dict = {}
    convo_box: list = []
    convo = _conversation(probe, convo_box, ["also add a /health endpoint"])

    before = len(convo.state.events)
    convo._step_holds_state_lock = True
    with convo._state:
        await convo.agent.astep(convo, on_event=convo._on_event)

    # Mid-tool the message must NOT be in the log: that point sits between the
    # assistant's tool_call and its tool result.
    assert probe["events_during_tool"] <= before + 1, (
        "the message was appended while the tool was still running, which puts a "
        "user message between a tool_call and its result"
    )
    assert "also add a /health endpoint" not in _user_texts(convo), (
        "message appeared before the step boundary"
    )

    # Draining is the run loop's job; do what the loop does at its next iteration.
    with convo._state:
        convo._drain_inbox_locked()

    assert "also add a /health endpoint" in _user_texts(convo), (
        "the message never reached the agent at all"
    )


@pytest.mark.asyncio
async def test_messages_are_drained_in_order():
    probe: dict = {}
    convo_box: list = []
    convo = _conversation(probe, convo_box, ["first", "second"])

    convo._step_holds_state_lock = True
    with convo._state:
        await convo.agent.astep(convo, on_event=convo._on_event)
    with convo._state:
        convo._drain_inbox_locked()

    texts = _user_texts(convo)
    assert texts.index("first") < texts.index("second")


def test_idle_conversation_keeps_todays_synchronous_behaviour():
    """Only the mid-step path changes. An idle send must still append immediately."""
    probe: dict = {}
    convo_box: list = []
    convo = _conversation(probe, convo_box, [])

    convo.send_message("hello")

    assert "hello" in _user_texts(convo), (
        "an idle send_message no longer appends synchronously — the inbox is "
        "swallowing messages that nothing will drain, because no run loop is active"
    )


def _scripted_conversation(
    probe: dict, convo_box: list, messages: list[str], script: list[Message], **kwargs
) -> LocalConversation:
    tool = _ProbeTool.create(executor=_SendingExecutor(convo_box, probe, messages))[0]
    register_tool("probe_tool", tool)
    agent = Agent(llm=TestLLM.from_messages(script), tools=[Tool(name="probe_tool")])
    convo = Conversation(agent=agent, **kwargs)
    assert isinstance(convo, LocalConversation)
    convo_box.append(convo)
    convo.send_message("build it")
    return convo


def _call(call_id: str, name: str, arguments: str) -> MessageToolCall:
    return MessageToolCall(
        id=call_id, name=name, arguments=arguments, origin="completion"
    )


@pytest.mark.asyncio
async def test_arun_acts_on_a_message_sent_during_a_step_that_finishes():
    """End to end: a message sent while a tool runs gets its own step.

    The step runs a tool (the message goes to the inbox) and then finishes. Upstream's
    mid-step rescan (#4194) compares ``last_user_message_id``, which an inbox message
    has not moved yet; it is the top-of-loop drain, ahead of the FINISHED check, that
    turns FINISHED back into a step for the new message. Guards that ordering.
    """
    probe: dict = {}
    convo_box: list = []
    script = [
        Message(
            role="assistant",
            content=[TextContent(text="")],
            tool_calls=[
                _call("call_1", "probe_tool", '{"command": "make"}'),
                _call("call_2", "finish", '{"message": "built"}'),
            ],
        ),
        Message(role="assistant", content=[TextContent(text="answered")]),
    ]
    convo = _scripted_conversation(probe, convo_box, ["one more thing"], script)

    await convo.arun()

    assert probe.get("send_returned") is True
    assert "one more thing" in _user_texts(convo)
    llm = convo.agent.llm
    assert isinstance(llm, TestLLM)
    assert llm.call_count == 2, (
        "the run ended on FINISHED with the inbox message unread instead of "
        f"taking another step for it (LLM calls: {llm.call_count})"
    )


@pytest.mark.asyncio
async def test_inbox_is_drained_when_the_run_loop_exits_from_the_bottom():
    """An iteration-cap exit must not leave an accepted message out of the log."""
    probe: dict = {}
    convo_box: list = []
    script = [
        Message(
            role="assistant",
            content=[TextContent(text="")],
            tool_calls=[_call("call_1", "probe_tool", '{"command": "make"}')],
        ),
    ]
    convo = _scripted_conversation(
        probe, convo_box, ["are you there?"], script, max_iteration_per_run=1
    )

    await convo.arun()

    assert probe.get("send_returned") is True
    assert "are you there?" in _user_texts(convo), (
        "the run loop exited with the message still in the inbox"
    )
    assert not convo._inbox_has_pending()
