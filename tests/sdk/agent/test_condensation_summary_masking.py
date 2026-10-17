"""A condensation summary is durable model output: registered secrets are masked.

The summarizer is shown each forgotten action's arguments verbatim -- including a
token the model typed into a command, such as a git remote URL. Whatever it
copies into the summary is persisted and replayed to the agent on every later
step, so the agent masks it before emitting the condensation, exactly as it masks
its own messages (#4783).
"""

import pytest

from openhands.sdk.agent import Agent
from openhands.sdk.context.condenser.base import CondenserBase
from openhands.sdk.context.view import View
from openhands.sdk.conversation import Conversation, LocalConversation
from openhands.sdk.event.base import Event
from openhands.sdk.event.condenser import Condensation
from openhands.sdk.llm import LLM


SECRET_VALUE = "glpat-condenser-leak-0123456789"


class EchoingCondenser(CondenserBase):
    """Condenses on every step with a summary that repeats a secret."""

    summary: str | None

    def condense(self, view: View, agent_llm: LLM | None = None) -> View | Condensation:
        return Condensation(
            forgotten_event_ids=set(),
            summary=self.summary,
            summary_offset=0 if self.summary is not None else None,
            llm_response_id="summary-response",
        )


def _conversation(summary: str | None) -> tuple[Agent, LocalConversation]:
    agent = Agent(
        llm=LLM(model="test-model", usage_id="test-llm"),
        tools=[],
        condenser=EchoingCondenser(summary=summary),
    )
    conversation = Conversation(agent=agent)
    assert isinstance(conversation, LocalConversation)
    conversation._ensure_agent_ready()
    conversation.update_secrets({"GITLAB_TOKEN": SECRET_VALUE})
    return agent, conversation


def _emitted_condensation(events: list[Event]) -> Condensation:
    condensations = [e for e in events if isinstance(e, Condensation)]
    assert len(condensations) == 1
    return condensations[0]


SUMMARY = (
    "CURRENT_STATE:\n- Re-cloned with "
    f"`git clone https://oauth2:{SECRET_VALUE}@gitlab.com/acme/orders.git`"
)


def test_step_masks_secrets_in_the_condensation_summary() -> None:
    agent, conversation = _conversation(SUMMARY)
    events: list[Event] = []

    agent.step(conversation, on_event=events.append)

    summary = _emitted_condensation(events).summary
    assert summary is not None
    assert SECRET_VALUE not in summary
    assert "<secret-hidden>" in summary
    assert "gitlab.com/acme/orders.git" in summary


@pytest.mark.asyncio
async def test_astep_masks_secrets_in_the_condensation_summary() -> None:
    agent, conversation = _conversation(SUMMARY)
    events: list[Event] = []

    await agent.astep(conversation, on_event=events.append)

    summary = _emitted_condensation(events).summary
    assert summary is not None
    assert SECRET_VALUE not in summary
    assert "<secret-hidden>" in summary


def test_step_leaves_a_condensation_without_summary_alone() -> None:
    agent, conversation = _conversation(None)
    events: list[Event] = []

    agent.step(conversation, on_event=events.append)

    assert _emitted_condensation(events).summary is None
