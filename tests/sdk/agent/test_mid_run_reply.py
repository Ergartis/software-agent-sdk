"""A message from a person mid-run must actually get answered.

``require_finish_tool`` made a reply *possible*: prose stopped meaning "finished", so
the agent could speak without quitting. It did not make a reply *happen*. Two live runs
on 2026-08-20 (tasks eb38f9e2 and 79d8dcc4) proved the gap — the message was delivered
into the LLM's context as a proper user turn, the system prompt asked for an answer in
as many words, and across 84 tool calls the agent produced zero prose. Mid-task, with
tools available, acting outranks answering every time.

The fix is structural rather than another paragraph of prompt: the drain marks the
message ``expects_reply``, and the agent takes exactly one step with no tools, where
prose is the only thing the model can return.

What these tests pin down:
  * a drained message earns the flag; the loop's own user-role messages do not
  * the tool-free step happens, and happens exactly once
  * the reply is published as an agent MessageEvent and the run does NOT end
  * work resumes with tools on the very next step
"""

from unittest.mock import MagicMock

import pytest
from litellm.types.utils import ModelResponse
from pydantic import PrivateAttr

from openhands.sdk.agent import Agent
from openhands.sdk.agent.response_dispatch import reply_is_owed
from openhands.sdk.conversation import Conversation, LocalConversation
from openhands.sdk.conversation.state import ConversationExecutionStatus as S
from openhands.sdk.event import Event, MessageEvent
from openhands.sdk.llm import LLM, LLMResponse, Message, TextContent
from openhands.sdk.llm.utils.metrics import MetricsSnapshot, TokenUsage


def _response(text: str) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=[TextContent(text=text)]),
        metrics=MetricsSnapshot(
            model_name="test",
            accumulated_cost=0.0,
            max_budget_per_task=0.0,
            accumulated_token_usage=TokenUsage(model="test"),
        ),
        raw_response=MagicMock(spec=ModelResponse, id="r1"),
    )


class _RecordingLLM(LLM):
    """Records the `tools` argument of every completion call.

    That argument is the whole mechanism: an empty list is what leaves the model
    nothing to return but prose. Asserting on it is asserting on the guarantee.
    """

    _calls: list = PrivateAttr(default_factory=list)

    def __init__(self):
        super().__init__(model="test-model")
        self._calls = []

    @property
    def calls(self) -> list:
        return self._calls

    _messages: list = PrivateAttr(default_factory=list)

    def completion(  # type: ignore[override]
        self, *, messages, tools=None, **kwargs
    ) -> LLMResponse:
        self._calls.append(list(tools or []))
        self._messages.append(list(messages))
        return _response("So far I have cloned the repo and started the backend.")

    @property
    def sent(self) -> list:
        return self._messages


def _conversation() -> tuple[LocalConversation, list[Event], _RecordingLLM]:
    llm = _RecordingLLM()
    agent = Agent(
        llm=llm,
        tools=[],
        require_finish_tool=True,
        include_default_tools=["FinishTool"],
    )
    conversation = Conversation(agent=agent)
    assert isinstance(conversation, LocalConversation)
    conversation._ensure_agent_ready()
    conversation.send_message("Add a healthcheck endpoint")
    return conversation, [], llm


def _step(conversation: LocalConversation, events: list[Event]) -> None:
    def on_event(event: Event) -> None:
        conversation._on_event(event)
        events.append(event)

    conversation.agent.step(conversation, on_event=on_event)


def _deliver_via_inbox(conversation: LocalConversation, text: str) -> None:
    """A person messaging an agent busy INSIDE A TOOL: queued, drained at the
    boundary."""
    with conversation._inbox_lock:
        conversation._inbox.append(
            (Message(role="user", content=[TextContent(text=text)]), None)
        )
    with conversation._state:
        conversation._drain_inbox_locked()


def _deliver_direct(conversation: LocalConversation, text: str) -> None:
    """A person messaging an agent busy in its LLM CALL: send_message appends inline.

    This is the ordinary path and it is the one that shipped broken. ``send_message``
    only queues on the inbox when ``_step_holds_state_lock`` is true, and that flag is
    False for the whole LLM round-trip — which is most of a step's wall clock. A live
    run on 2026-08-21 absorbed the message, produced 18 more tool calls and never
    answered, while the inbox-only tests were green.
    """
    conversation.send_message(text)


# The default for tests that do not care which path was used.
_deliver_mid_run = _deliver_via_inbox


def test_a_drained_message_expects_a_reply():
    conversation, _, _ = _conversation()

    _deliver_mid_run(conversation, "What is the status?")

    last = conversation.state.events[-1]
    assert isinstance(last, MessageEvent)
    assert last.source == "user"
    assert last.expects_reply is True
    assert reply_is_owed(conversation.state)


def test_the_opening_task_does_not_expect_a_reply():
    """Otherwise every run would burn a turn talking before it started working."""
    conversation, _, _ = _conversation()

    opening = [
        e
        for e in conversation.state.events
        if isinstance(e, MessageEvent) and e.source == "user"
    ]
    assert opening, "the opening task should be in the log"
    assert all(e.expects_reply is False for e in opening)
    assert not reply_is_owed(conversation.state)


def test_the_agent_takes_one_tool_free_step_to_answer():
    conversation, events, llm = _conversation()
    _step(conversation, events)  # ordinary step: tools offered
    assert llm.calls[-1], "a normal step must offer the agent its tools"

    _deliver_mid_run(conversation, "What is the status?")
    _step(conversation, events)

    assert llm.calls[-1] == [], "the reply step must offer no tools at all"


def test_the_reply_is_published_and_the_run_does_not_end():
    conversation, events, _ = _conversation()
    _deliver_mid_run(conversation, "What is the status?")

    _step(conversation, events)

    replies = [e for e in events if isinstance(e, MessageEvent) and e.source == "agent"]
    assert len(replies) == 1
    assert "cloned the repo" in str(replies[0].llm_message.content)
    # The point of require_finish_tool: answering is not quitting.
    assert conversation.state.execution_status != S.FINISHED


def test_work_resumes_with_tools_on_the_next_step():
    """The reply must cost exactly one step. If the flag stuck, the agent would
    monologue until max_consecutive_prose_steps killed the run — a worse failure than
    the silence this replaces."""
    conversation, events, llm = _conversation()
    _deliver_mid_run(conversation, "What is the status?")

    _step(conversation, events)
    assert llm.calls[-1] == []
    assert not reply_is_owed(conversation.state), "the reply clears the debt"

    _step(conversation, events)
    assert llm.calls[-1], "the step after the reply gets its tools back"


def test_several_messages_are_answered_by_one_reply():
    """A human pressing send three times is owed an answer, not three."""
    conversation, events, llm = _conversation()
    for text in ("What is your plan?", "what is the update?", "still there?"):
        _deliver_mid_run(conversation, text)

    _step(conversation, events)

    assert llm.calls[-1] == []
    _step(conversation, events)
    assert llm.calls[-1], "one reply settles the whole backlog"


@pytest.mark.parametrize(
    "deliver",
    [_deliver_via_inbox, _deliver_direct],
    ids=["inbox (busy in a tool)", "direct (busy in the LLM call)"],
)
def test_both_delivery_paths_owe_a_reply(deliver):
    """Whichever way the message got in, the person is equally owed an answer.

    Parametrised deliberately: the first cut of this feature marked only the inbox
    path, every test passed, and the live run stayed silent.
    """
    conversation, events, llm = _conversation()
    _step(conversation, events)  # the conversation is underway
    assert llm.calls[-1], "sanity: a normal step offers tools"
    before = sum(
        1 for e in events if isinstance(e, MessageEvent) and e.source == "agent"
    )

    deliver(conversation, "What is the plan?")

    last = conversation.state.events[-1]
    assert isinstance(last, MessageEvent) and last.expects_reply is True
    assert reply_is_owed(conversation.state)

    _step(conversation, events)
    assert llm.calls[-1] == [], "the reply step must offer no tools"
    after = sum(
        1 for e in events if isinstance(e, MessageEvent) and e.source == "agent"
    )
    assert after - before == 1, "exactly one reply for the interruption"


def test_the_opening_task_is_not_an_interruption():
    """The first message IS the task, however it arrives. Answering it would spend a
    turn talking before any work began."""
    conversation, _, _ = _conversation()  # this sends the opening message

    opening = [
        e
        for e in conversation.state.events
        if isinstance(e, MessageEvent) and e.source == "user"
    ]
    assert len(opening) == 1
    assert opening[0].expects_reply is False
    assert not reply_is_owed(conversation.state)


def _bury_under_tool_traffic(conversation: LocalConversation) -> None:
    """Replay what a step already in flight does after a message is accepted.

    ``send_message`` appends the moment it arrives, but the running step has not
    finished: it still writes its own ActionEvent and ObservationEvent, which land
    AFTER the question. So the question is almost never the newest event by the time
    the next step looks at the log.
    """
    from openhands.sdk.event import ActionEvent
    from openhands.sdk.llm import TextContent as _TC
    from openhands.sdk.tool.builtins.finish import FinishAction

    with conversation._state:
        conversation._on_event(
            ActionEvent(
                source="agent",
                thought=[_TC(text="carrying on")],
                action=FinishAction(message="x"),
                tool_name="finish",
                tool_call_id="call-1",
                tool_call={
                    "id": "call-1",
                    "type": "function",
                    "name": "finish",
                    "arguments": "{}",
                    "origin": "completion",
                },
                llm_response_id="resp-1",
            )
        )


def test_a_question_buried_under_tool_traffic_is_still_owed():
    """THE REGRESSION THAT SHIPPED. reply_is_owed originally returned False at the first
    ActionEvent it met scanning back, so a question followed by the in-flight step's own
    tool events read as 'already handled'. Live run 909e38a3 on 2026-08-21: two
    questions, 24 tool calls, zero replies — while every unit test passed, because the
    test LLM only
    ever produced prose and so never wrote an ActionEvent after the message."""
    conversation, events, _ = _conversation()
    _step(conversation, events)
    conversation.send_message("What is the plan?")
    assert reply_is_owed(conversation.state), "sanity: owed the moment it arrives"

    _bury_under_tool_traffic(conversation)

    assert not isinstance(conversation.state.events[-1], MessageEvent), (
        "sanity: the question is no longer the newest event"
    )
    assert reply_is_owed(conversation.state), "still owed — working is not answering"


def test_the_agent_speaking_is_what_settles_the_debt():
    """Only an agent message clears it — and it stays clear while work continues."""
    conversation, events, _ = _conversation()
    _step(conversation, events)
    conversation.send_message("What is the plan?")
    _bury_under_tool_traffic(conversation)
    assert reply_is_owed(conversation.state)

    with conversation._state:  # the agent answers
        conversation._on_event(
            MessageEvent(
                source="agent",
                llm_message=Message(
                    role="assistant",
                    content=[TextContent(text="Cloned the repo; wiring the API now.")],
                ),
            )
        )
    assert not reply_is_owed(conversation.state), "the reply settles it"

    _bury_under_tool_traffic(conversation)  # more work afterwards
    assert not reply_is_owed(conversation.state), "and it stays settled"


def _text_of(message) -> str:
    c = message.content or []
    return " ".join(x.text for x in c if hasattr(x, "text"))


def test_the_reply_step_is_told_to_answer():
    """Taking the tools away makes speech the only option; not an ANSWER.

    Live run 6ea4eeb5 (2026-08-21): three forced steps, two empty responses and one
    line of unrelated narration. The single good answer that run produced came from an
    ordinary step where the model volunteered prose. An empty forced turn is worse than
    none — it burns a
    step and the SDK then has to nudge the model back into calling tools.
    """
    conversation, events, llm = _conversation()
    _step(conversation, events)
    conversation.send_message("What are the test cases you will run?")

    _step(conversation, events)

    assert llm.calls[-1] == [], "sanity: still a tool-free step"
    last = _text_of(llm.sent[-1][-1])
    assert "waiting for an answer" in last
    assert "Do not return an empty message" in last
    assert "does NOT end your turn" in last


def test_the_directive_never_reaches_the_event_log():
    """It steers one turn. Persisting it would put words in the human's mouth in the
    transcript, and would be re-sent on every later step once it is in the history."""
    conversation, events, _ = _conversation()
    _step(conversation, events)
    conversation.send_message("What are the test cases you will run?")
    _step(conversation, events)

    for e in conversation.state.events:
        if isinstance(e, MessageEvent):
            assert "waiting for an answer" not in str(e.llm_message.content), (
                "the directive leaked into the conversation history"
            )


def test_an_ordinary_step_is_not_given_the_directive():
    conversation, events, llm = _conversation()

    _step(conversation, events)

    assert llm.calls[-1], "sanity: tools offered"
    assert "waiting for an answer" not in _text_of(llm.sent[-1][-1])
