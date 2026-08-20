"""Prose must be able to be a REPLY, not only a resignation.

``_handle_content_response`` sets ``FINISHED`` on any text-content response, so "answer
the human" and "stop working" are the same statement. That is why an agent asked a
question mid-run goes silent instead of replying: answering *is* quitting. It is also
why the SDK's own instruction to "consult with the user first" cannot be used in a loop
that must keep going afterwards.

``require_finish_tool`` separates the two. Off by default — every existing behaviour is
unchanged — and ignored when the agent has no finish tool, so an agent is never trapped
in a conversation it has no way to end.

Pairs with the lock-free inbox in ``local_conversation.py``: the inbox is what lets a
message REACH a busy agent, and this is what lets the agent answer it without ending the
run. Neither is much use without the other.
"""

from unittest.mock import MagicMock

import pytest
from litellm.types.utils import ModelResponse
from pydantic import PrivateAttr

from openhands.sdk.agent import Agent
from openhands.sdk.conversation import Conversation, LocalConversation
from openhands.sdk.conversation.state import ConversationExecutionStatus as S
from openhands.sdk.event import Event, MessageEvent
from openhands.sdk.llm import LLM, LLMResponse, Message, TextContent
from openhands.sdk.llm.utils.metrics import MetricsSnapshot, TokenUsage


def _prose(text: str = "Which database should I use?") -> LLMResponse:
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


class _ProseLLM(LLM):
    """Answers every step with plain text — the monologue case."""

    _text: str = PrivateAttr(default="Which database should I use?")

    def __init__(self, text: str = "Which database should I use?"):
        super().__init__(model="test-model")
        self._text = text

    def completion(self, *, messages, tools=None, **kwargs) -> LLMResponse:  # type: ignore[override]
        return _prose(self._text)


def _conversation(
    *, require_finish_tool: bool, with_finish_tool: bool = True, **agent_kwargs
) -> tuple[LocalConversation, list[Event]]:
    agent = Agent(
        llm=_ProseLLM(),
        tools=[],
        require_finish_tool=require_finish_tool,
        include_default_tools=["FinishTool"] if with_finish_tool else [],
        **agent_kwargs,
    )
    conversation = Conversation(agent=agent)
    assert isinstance(conversation, LocalConversation)
    conversation._ensure_agent_ready()

    events: list[Event] = []
    conversation.send_message("Add a healthcheck endpoint")
    return conversation, events


def _step(conversation: LocalConversation, events: list[Event]) -> None:
    """One agent step, wired the way ``run()`` wires it.

    The conversation's own callback is what appends to ``state.events``, so a test
    passing only its own collector would leave the event log empty — and the chatter
    guard reads that log. Both callbacks run here: the real one, and the collector.
    """

    def on_event(event: Event) -> None:
        conversation._on_event(event)
        events.append(event)

    conversation.agent.step(conversation, on_event=on_event)


def test_prose_finishes_when_the_flag_is_off():
    """The default is unchanged: prose still ends the conversation."""
    conversation, events = _conversation(require_finish_tool=False)

    _step(conversation, events)

    assert conversation.state.execution_status == S.FINISHED


def test_prose_does_not_finish_when_the_flag_is_on():
    conversation, events = _conversation(require_finish_tool=True)

    _step(conversation, events)

    assert conversation.state.execution_status != S.FINISHED


def test_prose_is_still_recorded_as_a_message_event():
    """The answer must reach the human either way — this changes when the run ends,
    not whether the words are published."""
    conversation, events = _conversation(require_finish_tool=True)

    _step(conversation, events)

    messages = [e for e in events if isinstance(e, MessageEvent)]
    assert len(messages) == 1
    assert messages[0].source == "agent"
    assert "database" in str(messages[0].llm_message.content)


def test_chatter_guard_finishes_after_repeated_prose():
    """Without a guard a confused model monologues until the iteration budget dies.

    The count is of *consecutive* prose steps, so this only fires when the agent has
    genuinely stopped acting.
    """
    conversation, events = _conversation(require_finish_tool=True)

    for _ in range(3):
        assert conversation.state.execution_status != S.FINISHED
        _step(conversation, events)

    assert conversation.state.execution_status == S.FINISHED


def test_the_guard_is_configurable():
    conversation, events = _conversation(
        require_finish_tool=True, max_consecutive_prose_steps=1
    )

    _step(conversation, events)

    assert conversation.state.execution_status == S.FINISHED


def test_a_human_reply_resets_the_chatter_guard():
    """A conversation is not a monologue. If the human answered, the agent's next
    message is a reply to it, not the next line of a rant — so the count restarts, and
    a long back-and-forth cannot be cut off by a guard meant for a stuck model."""
    conversation, events = _conversation(require_finish_tool=True)

    _step(conversation, events)
    _step(conversation, events)
    conversation.send_message("Use Postgres")
    _step(conversation, events)

    assert conversation.state.execution_status != S.FINISHED


def test_flag_is_ignored_when_finish_tool_is_absent():
    """Never trap an agent with no exit: with no finish tool, prose is the only way out
    and must keep working as one."""
    conversation, events = _conversation(
        require_finish_tool=True, with_finish_tool=False
    )

    _step(conversation, events)

    assert conversation.state.execution_status == S.FINISHED


@pytest.mark.parametrize("flag", [True, False])
def test_the_flag_survives_serialization(flag: bool):
    """The platform sets this in the agent spec it POSTs; a field that does not
    round-trip would be silently ignored, which is how planMode was lost."""
    agent = Agent(llm=_ProseLLM(), tools=[], require_finish_tool=flag)

    restored = Agent.model_validate(agent.model_dump())

    assert restored.require_finish_tool is flag
