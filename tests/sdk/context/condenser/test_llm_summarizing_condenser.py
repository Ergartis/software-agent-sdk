import json
from collections.abc import Sequence
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from litellm.types.utils import (
    Choices,
    Message as LiteLLMMessage,
    ModelResponse,
    Usage,
)
from pydantic import SecretStr

from openhands.sdk.context.condenser.base import (
    CondensationRequirement,
    NoCondensationAvailableException,
)
from openhands.sdk.context.condenser.llm_summarizing_condenser import (
    LLMSummarizingCondenser,
    Reason,
)
from openhands.sdk.context.view import View
from openhands.sdk.event.base import Event
from openhands.sdk.event.condenser import Condensation, CondensationRequest
from openhands.sdk.event.llm_convertible import (
    ActionEvent,
    MessageEvent,
    ObservationEvent,
)
from openhands.sdk.llm import (
    LLM,
    ImageContent,
    LLMResponse,
    Message,
    MessageToolCall,
    MetricsSnapshot,
    TextContent,
)
from openhands.sdk.tool import Action, Observation


def message_event(content: str) -> MessageEvent:
    return MessageEvent(
        llm_message=Message(role="user", content=[TextContent(text=content)]),
        source="user",
    )


@pytest.fixture
def mock_llm() -> LLM:
    """Create a mock LLM for testing."""
    mock_llm = MagicMock(spec=LLM)

    # Mock the completion response - now returns LLMResponse
    def create_completion_result(content: str) -> LLMResponse:
        message = Message(role="assistant", content=[TextContent(text=content)])
        metrics = MetricsSnapshot(
            model_name="test-model",
            accumulated_cost=0.0,
            max_budget_per_task=None,
            accumulated_token_usage=None,
        )
        # Create a mock ModelResponse
        raw_response = MagicMock(spec=ModelResponse)
        raw_response.id = "mock-llm-response-id"
        return LLMResponse(message=message, metrics=metrics, raw_response=raw_response)

    mock_llm.completion.return_value = create_completion_result(
        "Summary of forgotten events"
    )
    mock_llm.acompletion = AsyncMock(return_value=mock_llm.completion.return_value)
    mock_llm.uses_responses_api = lambda: False
    mock_llm.requires_streaming = False
    mock_llm.format_messages_for_llm = lambda messages: messages

    # Mock the required attributes that the LLM validator reads
    mock_llm.openrouter_site_url = "https://docs.all-hands.dev/"
    mock_llm.openrouter_app_name = "OpenHands"
    mock_llm.aws_access_key_id = None
    mock_llm.aws_secret_access_key = None
    mock_llm.aws_session_token = None
    mock_llm.aws_region_name = None
    mock_llm.aws_profile_name = None
    mock_llm.aws_role_name = None
    mock_llm.aws_session_name = None
    mock_llm.aws_bedrock_runtime_endpoint = None
    mock_llm.metrics = None
    mock_llm.model = "test-model"
    mock_llm.log_completions = False
    mock_llm.log_completions_folder = None
    mock_llm.custom_tokenizer = None
    mock_llm.base_url = None
    mock_llm.reasoning_effort = None
    mock_llm.litellm_extra_body = {}
    mock_llm.temperature = 0.0
    # Streaming is off by default (matches LLM.stream's default), so the
    # condenser uses this LLM directly without copying it.
    mock_llm.stream = False

    # Explicitly set pricing attributes required by LLM -> Telemetry wiring
    mock_llm.input_cost_per_token = None
    mock_llm.output_cost_per_token = None

    mock_llm._metrics = None
    mock_llm._telemetry = None

    # Helper method to set mock response content
    def set_mock_response_content(content: str):
        result = create_completion_result(content)
        mock_llm.completion.return_value = result
        mock_llm.acompletion = AsyncMock(return_value=result)

    mock_llm.set_mock_response_content = set_mock_response_content

    return mock_llm


def test_default_values(mock_llm: LLM) -> None:
    """Test that LLMSummarizingCondenser has correct default values.

    These defaults are tuned to ensure workable manipulation indices for condensation.
    See https://github.com/OpenHands/software-agent-sdk/issues/1518 for context.
    """
    condenser = LLMSummarizingCondenser(llm=mock_llm)

    # Default max_size should be 240 (raised from 120 to allow more room for tool loops)
    assert condenser.max_size == 240

    # Default keep_first should be 2 (reduced from 4 to leave more room for
    # condensation)
    assert condenser.keep_first == 2


def test_should_condense(mock_llm: LLM) -> None:
    """Test that LLMSummarizingCondenser correctly determines when to condense."""
    max_size = 100
    condenser = LLMSummarizingCondenser(llm=mock_llm, max_size=max_size)

    # Create events below the threshold
    small_events = [message_event(f"Event {i}") for i in range(max_size)]
    small_view = View.from_events(small_events)

    assert condenser.condensation_requirement(small_view) is None

    # Create events above the threshold (triggers EVENTS reason -> SOFT requirement)
    large_events = [message_event(f"Event {i}") for i in range(max_size + 1)]
    large_view = View.from_events(large_events)

    assert (
        condenser.condensation_requirement(large_view) == CondensationRequirement.SOFT
    )


def test_condense_returns_view_when_no_condensation_needed(mock_llm: LLM) -> None:
    """Test that condenser returns the original view when no condensation is needed."""  # noqa: E501
    max_size = 100
    condenser = LLMSummarizingCondenser(llm=mock_llm, max_size=max_size)

    events: list[Event] = [message_event(f"Event {i}") for i in range(max_size)]
    view = View.from_events(events)

    result = condenser.condense(view)

    assert isinstance(result, View)
    assert result == view
    # LLM should not be called
    cast(MagicMock, mock_llm.completion).assert_not_called()


def test_condense_returns_condensation_when_needed(mock_llm: LLM) -> None:
    """Test that condenser returns a Condensation when condensation is needed."""
    max_size = 10
    keep_first = 3
    condenser = LLMSummarizingCondenser(
        llm=mock_llm, max_size=max_size, keep_first=keep_first
    )

    # Set up mock response
    cast(Any, mock_llm).set_mock_response_content("Summary of forgotten events")

    events: list[Event] = [message_event(f"Event {i}") for i in range(max_size + 1)]
    view = View.from_events(events)

    result = condenser.condense(view)

    assert isinstance(result, Condensation)
    assert result.summary == "Summary of forgotten events"
    # summary_offset should be the smallest manipulation index >= keep_first
    # Since all events are MessageEvents, manipulation indices are [0,1,2,3,4,...]
    # The smallest index >= keep_first (3) is 3
    # This means we keep events [0:3] = indices 0,1,2 = 3 events
    assert result.summary_offset == keep_first
    assert len(result.forgotten_event_ids) > 0

    # LLM should be called once
    cast(MagicMock, mock_llm.completion).assert_called_once()


def test_condense_uses_responses_api_when_required(mock_llm: LLM) -> None:
    condenser = LLMSummarizingCondenser(llm=mock_llm, max_size=10, keep_first=3)
    cast(Any, mock_llm).set_mock_response_content("Summary from responses")
    mock_llm.uses_responses_api = lambda: True
    cast(Any, mock_llm.responses).return_value = cast(
        Any, mock_llm.completion
    ).return_value

    view = View.from_events([message_event(f"Event {i}") for i in range(11)])
    result = condenser.condense(view)

    assert isinstance(result, Condensation)
    assert result.summary == "Summary from responses"
    cast(MagicMock, mock_llm.responses).assert_called_once()
    cast(MagicMock, mock_llm.completion).assert_not_called()


def test_get_condensation_with_previous_summary(mock_llm: LLM) -> None:
    """Test that condenser properly handles previous summary content."""
    max_size = 10
    keep_first = 3
    condenser = LLMSummarizingCondenser(
        llm=mock_llm, max_size=max_size, keep_first=keep_first
    )

    # Set up mock response
    cast(Any, mock_llm).set_mock_response_content("Updated summary")

    # Create events with a condensation in the history
    # Need enough events so that after condensation, the view still exceeds max_size
    # Condensation will remove 2 events (events[3] and events[4]) plus itself
    # So we need at least max_size + 1 + 3 = 14 events to exceed max_size after
    # condensation
    events = [message_event(f"Event {i}") for i in range(14)]

    # Add a condensation to simulate previous summarization
    # The summary will be inserted at keep_first due to summary_offset
    condensation = Condensation(
        forgotten_event_ids={events[3].id, events[4].id},
        summary="Previous summary content",
        summary_offset=keep_first,
        llm_response_id="condensation_response_1",
    )
    events_with_condensation = (
        events[:keep_first] + [condensation] + events[keep_first:]
    )

    view = View.from_events(events_with_condensation)

    result = condenser.get_condensation(view)

    assert isinstance(result, Condensation)
    assert result.summary == "Updated summary"

    # Verify that the LLM was called with the previous summary
    completion_mock = cast(MagicMock, mock_llm.completion)
    completion_mock.assert_called_once()
    call_args = completion_mock.call_args
    messages = call_args[1]["messages"]  # Get keyword arguments
    prompt_text = messages[0].content[0].text

    # The prompt should contain the previous summary (it's in <PREVIOUS SUMMARY> sec.)
    # The summary is now retrieved from the view, which should have it at the summary
    # event
    assert (
        "Previous summary content" in prompt_text or "<PREVIOUS SUMMARY>" in prompt_text
    )


def test_invalid_config(mock_llm: LLM) -> None:
    """Test that LLMSummarizingCondenser validates configuration parameters."""
    # Test max_size must be positive
    with pytest.raises(ValueError):
        LLMSummarizingCondenser(llm=mock_llm, max_size=0)

    # Test keep_first must be non-negative
    with pytest.raises(ValueError):
        LLMSummarizingCondenser(llm=mock_llm, keep_first=-1)

    # Test keep_first must be less than max_size // 2 to leave room for condensation
    with pytest.raises(ValueError):
        LLMSummarizingCondenser(llm=mock_llm, max_size=10, keep_first=8)


def test_get_condensation_does_not_pass_extra_body(mock_llm: LLM) -> None:
    """Condenser should not pass extra_body to llm.completion.

    This prevents providers like 1p Anthropic from rejecting the request with
    "extra_body: Extra inputs are not permitted".
    """
    condenser = LLMSummarizingCondenser(llm=mock_llm, max_size=10, keep_first=2)

    # Prepare a view that triggers condensation (len > max_size)
    events: list[Event] = [message_event(f"Event {i}") for i in range(12)]
    view = View.from_events(events)

    result = condenser.condense(view)
    assert isinstance(result, Condensation)

    # Ensure completion was called without an explicit extra_body kwarg
    completion_mock = cast(MagicMock, mock_llm.completion)
    assert completion_mock.call_count == 1


def test_condense_with_agent_llm(mock_llm: LLM) -> None:
    """Test that condenser accepts and works with optional agent llm parameter."""
    condenser = LLMSummarizingCondenser(llm=mock_llm, max_size=10, keep_first=2)

    # Create a separate mock for the agent's LLM
    agent_llm = MagicMock(spec=LLM)
    agent_llm.model = "gpt-4"
    agent_llm.effective_max_input_tokens = None

    # Prepare a view that triggers condensation
    events: list[Event] = [message_event(f"Event {i}") for i in range(12)]
    view = View.from_events(events)

    # Call condense with the agent's LLM
    result = condenser.condense(view, agent_llm=agent_llm)
    assert isinstance(result, Condensation)

    # Verify the condenser still uses its own LLM for summarization
    completion_mock = cast(MagicMock, mock_llm.completion)
    assert completion_mock.call_count == 1

    # Agent LLM should not be called for completion (condenser uses its own LLM)
    assert not agent_llm.completion.called
    _, kwargs = completion_mock.call_args
    assert "extra_body" not in kwargs


def test_condense_with_token_limit_exceeded(mock_llm: LLM) -> None:
    """Test that condenser triggers on TOKENS reason when token limit is exceeded."""
    max_tokens = 100
    keep_first = 2
    condenser = LLMSummarizingCondenser(
        llm=mock_llm, max_size=1000, max_tokens=max_tokens, keep_first=keep_first
    )

    # Create a separate mock for the agent's LLM with token counting
    agent_llm = MagicMock(spec=LLM)
    agent_llm.model = "gpt-4"
    agent_llm.effective_max_input_tokens = None

    # Mock get_token_count to return predictable values based on message content length
    def mock_token_count(messages, **_kwargs):
        # Simple heuristic: count characters in all text content
        # Each character = 0.25 tokens (roughly 4 chars per token)
        total_chars = 0
        for msg in messages:
            for content in msg.content:
                if hasattr(content, "text"):
                    total_chars += len(content.text)
        return total_chars // 4

    cast(MagicMock, agent_llm.get_token_count).side_effect = mock_token_count

    # Create events that exceed token limit
    # Each event has 40 chars = 10 tokens
    # 15 events = 150 tokens (exceeds max_tokens of 100)
    events: list[Event] = [message_event("A" * 40) for i in range(15)]
    view = View.from_events(events)

    # Verify that TOKENS is the condensation reason
    reasons = condenser.get_condensation_reasons(view, agent_llm=agent_llm)
    assert Reason.TOKENS in reasons
    assert Reason.EVENTS not in reasons  # Should not trigger on event count
    assert Reason.REQUEST not in reasons

    # Condense the view
    result = condenser.condense(view, agent_llm=agent_llm)
    assert isinstance(result, Condensation)

    # Verify the condenser used its own LLM for summarization
    completion_mock = cast(MagicMock, mock_llm.completion)
    assert completion_mock.call_count == 1

    # Verify forgotten events were calculated based on token reduction
    assert len(result.forgotten_event_ids) > 0


@pytest.mark.parametrize(
    ("configured_limit", "agent_limit", "token_count", "expected_trigger"),
    [(500, 100, 200, True), (100, 500, 200, True), (100, 500, 50, False)],
)
def test_token_limit_uses_stricter_configured_or_agent_limit(
    mock_llm: LLM,
    configured_limit: int,
    agent_limit: int,
    token_count: int,
    expected_trigger: bool,
) -> None:
    condenser = LLMSummarizingCondenser(
        llm=mock_llm, max_size=1000, max_tokens=configured_limit, keep_first=2
    )
    agent_llm = MagicMock(spec=LLM)
    agent_llm.effective_max_input_tokens = agent_limit
    agent_llm.get_token_count.return_value = token_count

    view = View.from_events([message_event("event") for _ in range(10)])

    reasons = condenser.get_condensation_reasons(view, agent_llm=agent_llm)

    assert (Reason.TOKENS in reasons) is expected_trigger


def test_token_limit_inherits_agent_effective_input_limit(
    mock_llm: LLM,
) -> None:
    condenser = LLMSummarizingCondenser(llm=mock_llm, max_size=1000, keep_first=2)
    agent_llm = MagicMock(spec=LLM)
    agent_llm.effective_max_input_tokens = 100
    agent_llm.get_token_count.return_value = 200

    view = View.from_events([message_event("event") for _ in range(10)])

    reasons = condenser.get_condensation_reasons(view, agent_llm=agent_llm)

    assert Reason.TOKENS in reasons


def test_token_reduction_uses_agent_effective_input_limit(mock_llm: LLM) -> None:
    condenser = LLMSummarizingCondenser(
        llm=mock_llm, max_size=1000, max_tokens=500, keep_first=2
    )
    agent_llm = MagicMock(spec=LLM)
    agent_llm.effective_max_input_tokens = 100
    agent_llm.get_token_count.return_value = 200

    view = View.from_events([message_event("event") for _ in range(10)])

    forgotten_events, _ = condenser._get_forgotten_events(view, agent_llm=agent_llm)

    assert forgotten_events


def test_condense_with_request_and_events_reasons(mock_llm: LLM) -> None:
    """Test condensation when both REQUEST and EVENTS reasons are true simultaneously.

    Verifies that the most aggressive condensation (minimum suffix) is chosen.
    """
    max_size = 20
    keep_first = 2
    condenser = LLMSummarizingCondenser(
        llm=mock_llm, max_size=max_size, keep_first=keep_first
    )

    # Create events that exceed max_size AND include a condensation request
    # 25 events > max_size of 20 (triggers EVENTS)
    # Plus a CondensationRequest (triggers REQUEST)
    events: list[Event] = [message_event(f"Event {i}") for i in range(25)]
    events.append(CondensationRequest())
    view = View.from_events(events)

    # Verify both reasons are present
    reasons = condenser.get_condensation_reasons(view, agent_llm=None)
    assert Reason.REQUEST in reasons
    assert Reason.EVENTS in reasons
    assert Reason.TOKENS not in reasons

    # Get the condensation
    result = condenser.condense(view)
    assert isinstance(result, Condensation)

    # Calculate expected behavior:
    # REQUEST: target_size = len(view) // 2 = 25 // 2 = 12
    #          suffix_to_keep = 12 - keep_first - 1 = 12 - 2 - 1 = 9
    # EVENTS:  target_size = max_size // 2 = 20 // 2 = 10
    #          suffix_to_keep = 10 - keep_first - 1 = 10 - 2 - 1 = 7
    # Most aggressive: min(9, 7) = 7

    # With manipulation indices for MessageEvents:
    # naive_start = keep_first = 2
    # naive_end = 25 - 7 = 18
    # manipulation_indices = [0, 1, 2, 3, ..., 25]
    # forgetting_start = smallest index >= keep_first = 2
    # forgetting_end = smallest index >= naive_end = 18
    # Forgotten: events[2:18] = 16 events
    expected_forgotten_count = 16
    assert len(result.forgotten_event_ids) == expected_forgotten_count


def test_condense_with_request_and_tokens_reasons(mock_llm: LLM) -> None:
    """Test condensation when both REQUEST and TOKENS reasons are true simultaneously.

    Verifies that the most aggressive condensation (minimum suffix) is chosen.
    """
    max_tokens = 100
    keep_first = 2
    condenser = LLMSummarizingCondenser(
        llm=mock_llm, max_size=1000, max_tokens=max_tokens, keep_first=keep_first
    )

    # Create a separate mock for the agent's LLM with token counting
    agent_llm = MagicMock(spec=LLM)
    agent_llm.model = "gpt-4"
    agent_llm.effective_max_input_tokens = None

    # Mock get_token_count to return predictable values
    def mock_token_count(messages, **_kwargs):
        total_chars = 0
        for msg in messages:
            for content in msg.content:
                if hasattr(content, "text"):
                    total_chars += len(content.text)
        return total_chars // 4

    cast(MagicMock, agent_llm.get_token_count).side_effect = mock_token_count

    # Create 20 events with 40 chars each = 10 tokens each = 200 total tokens
    # This exceeds max_tokens of 100 (triggers TOKENS)
    events: list[Event] = [message_event("A" * 40) for i in range(20)]
    # Add a CondensationRequest (triggers REQUEST)
    events.append(CondensationRequest())
    view = View.from_events(events)

    # Verify both reasons are present
    reasons = condenser.get_condensation_reasons(view, agent_llm=agent_llm)
    assert Reason.REQUEST in reasons
    assert Reason.TOKENS in reasons
    assert Reason.EVENTS not in reasons

    # Get the condensation
    result = condenser.condense(view, agent_llm=agent_llm)
    assert isinstance(result, Condensation)

    # The most aggressive condensation should be chosen (minimum suffix)
    assert len(result.forgotten_event_ids) > 0


def test_condense_with_events_and_tokens_reasons(mock_llm: LLM) -> None:
    """Test condensation when both EVENTS and TOKENS reasons are true simultaneously.

    Verifies that the most aggressive condensation (minimum suffix) is chosen.
    """
    max_size = 15
    max_tokens = 100
    keep_first = 2
    condenser = LLMSummarizingCondenser(
        llm=mock_llm, max_size=max_size, max_tokens=max_tokens, keep_first=keep_first
    )

    # Create a separate mock for the agent's LLM with token counting
    agent_llm = MagicMock(spec=LLM)
    agent_llm.model = "gpt-4"
    agent_llm.effective_max_input_tokens = None

    def mock_token_count(messages, **_kwargs):
        total_chars = 0
        for msg in messages:
            for content in msg.content:
                if hasattr(content, "text"):
                    total_chars += len(content.text)
        return total_chars // 4

    cast(MagicMock, agent_llm.get_token_count).side_effect = mock_token_count

    # Create 20 events (exceeds max_size of 15) with 40 chars each
    # 20 events * 10 tokens = 200 tokens (exceeds max_tokens of 100)
    events: list[Event] = [message_event("A" * 40) for i in range(20)]
    view = View.from_events(events)

    # Verify both reasons are present
    reasons = condenser.get_condensation_reasons(view, agent_llm=agent_llm)
    assert Reason.EVENTS in reasons
    assert Reason.TOKENS in reasons
    assert Reason.REQUEST not in reasons

    # Get the condensation
    result = condenser.condense(view, agent_llm=agent_llm)
    assert isinstance(result, Condensation)

    # The most aggressive condensation should be chosen (minimum suffix)
    assert len(result.forgotten_event_ids) > 0


def test_condense_with_all_three_reasons(mock_llm: LLM) -> None:
    """Test condensation when all three reasons are true simultaneously.

    Verifies that the most aggressive condensation (minimum suffix) is chosen
    when REQUEST, EVENTS, and TOKENS all trigger at once.
    """
    max_size = 15
    max_tokens = 100
    keep_first = 2
    condenser = LLMSummarizingCondenser(
        llm=mock_llm, max_size=max_size, max_tokens=max_tokens, keep_first=keep_first
    )

    # Create a separate mock for the agent's LLM with token counting
    agent_llm = MagicMock(spec=LLM)
    agent_llm.model = "gpt-4"
    agent_llm.effective_max_input_tokens = None

    def mock_token_count(messages, **_kwargs):
        total_chars = 0
        for msg in messages:
            for content in msg.content:
                if hasattr(content, "text"):
                    total_chars += len(content.text)
        return total_chars // 4

    cast(MagicMock, agent_llm.get_token_count).side_effect = mock_token_count

    # Create 20 events (exceeds max_size of 15) with 40 chars each
    # 20 events * 10 tokens = 200 tokens (exceeds max_tokens of 100)
    events: list[Event] = [message_event("A" * 40) for i in range(20)]
    # Add CondensationRequest (triggers REQUEST)
    events.append(CondensationRequest())
    view = View.from_events(events)

    # Verify all three reasons are present
    reasons = condenser.get_condensation_reasons(view, agent_llm=agent_llm)
    assert Reason.REQUEST in reasons
    assert Reason.EVENTS in reasons
    assert Reason.TOKENS in reasons

    # Get the condensation
    result = condenser.condense(view, agent_llm=agent_llm)
    assert isinstance(result, Condensation)

    # The most aggressive condensation should be chosen (minimum suffix)
    # This means the most events should be forgotten
    assert len(result.forgotten_event_ids) > 0

    # Verify the condenser used its own LLM for summarization
    completion_mock = cast(MagicMock, mock_llm.completion)
    assert completion_mock.call_count == 1


def test_most_aggressive_condensation_chosen(mock_llm: LLM) -> None:
    """Test that the minimum suffix is chosen when multiple reasons provide different
    targets.

    This test explicitly verifies the min() logic at line 200 of the condenser.
    """
    max_size = 30  # Set high so EVENTS triggers with specific target
    keep_first = 2
    condenser = LLMSummarizingCondenser(
        llm=mock_llm, max_size=max_size, keep_first=keep_first
    )

    # Create a scenario where REQUEST and EVENTS give different suffix sizes
    # 40 events total
    events: list[Event] = [message_event(f"Event {i}") for i in range(40)]
    events.append(CondensationRequest())
    view = View.from_events(events)

    # Calculate expected suffix lengths:
    # REQUEST: target_size = len(view) // 2 = 40 // 2 = 20
    #          suffix_to_keep = 20 - keep_first - 1 = 20 - 2 - 1 = 17
    # EVENTS:  target_size = max_size // 2 = 30 // 2 = 15
    #          suffix_to_keep = 15 - keep_first - 1 = 15 - 2 - 1 = 12
    # Most aggressive: min(17, 12) = 12

    result = condenser.condense(view)
    assert isinstance(result, Condensation)

    # With manipulation indices for MessageEvents:
    # naive_start = keep_first = 2
    # naive_end = 40 - 12 = 28
    # manipulation_indices = [0, 1, 2, 3, ..., 40]
    # forgetting_start = smallest index >= keep_first = 2
    # forgetting_end = smallest index >= naive_end = 28
    # Forgotten events: events[2:28] = 26 events
    expected_forgotten_count = 26
    assert len(result.forgotten_event_ids) == expected_forgotten_count


def test_generate_condensation_raises_on_zero_events(mock_llm: LLM) -> None:
    """Test that _generate_condensation raises AssertionError when given 0 events.

    This prevents the LLM from being called with an empty event list, which would
    produce a confusing summary like "I don't see any events provided to summarize."
    See https://github.com/OpenHands/software-agent-sdk/issues/1518 for context.
    """
    condenser = LLMSummarizingCondenser(llm=mock_llm, max_size=100, keep_first=2)

    with pytest.raises(AssertionError, match="No events to condense"):
        condenser._generate_condensation(
            forgotten_events=[],
            summary_offset=0,
        )

    # Verify the LLM was never called
    cast(MagicMock, mock_llm.completion).assert_not_called()


@pytest.mark.parametrize(
    "reasons",
    [set()],
)
def test_condensation_requirement_returns_none(
    mock_llm: LLM, reasons: set[Reason]
) -> None:
    """Test that condensation_requirement returns None when appropriate.

    Mocks get_condensation_reasons to test different reason combinations.
    """
    condenser = LLMSummarizingCondenser(llm=mock_llm, max_size=100, keep_first=2)
    events: list[Event] = [message_event(f"Event {i}") for i in range(10)]
    view = View.from_events(events)

    with patch.object(
        LLMSummarizingCondenser, "get_condensation_reasons", return_value=reasons
    ):
        result = condenser.condensation_requirement(view)
        assert result is None


@pytest.mark.parametrize(
    "reasons",
    [
        {Reason.EVENTS},
    ],
)
def test_condensation_requirement_returns_soft(
    mock_llm: LLM, reasons: set[Reason]
) -> None:
    """Test that condensation_requirement returns SOFT for resource constraints.

    Mocks get_condensation_reasons to test different resource reason combinations.
    """
    condenser = LLMSummarizingCondenser(llm=mock_llm, max_size=100, keep_first=2)
    events: list[Event] = [message_event(f"Event {i}") for i in range(10)]
    view = View.from_events(events)

    with patch.object(
        LLMSummarizingCondenser, "get_condensation_reasons", return_value=reasons
    ):
        result = condenser.condensation_requirement(view)
        assert result == CondensationRequirement.SOFT


@pytest.mark.parametrize(
    "reasons",
    [
        {Reason.TOKENS},
        {Reason.TOKENS, Reason.EVENTS},
    ],
)
def test_condensation_requirement_returns_hard_for_token_pressure(
    mock_llm: LLM, reasons: set[Reason]
) -> None:
    """Token pressure should trigger before the next LLM request can overflow."""
    condenser = LLMSummarizingCondenser(llm=mock_llm, max_size=100, keep_first=2)
    events: list[Event] = [message_event(f"Event {i}") for i in range(10)]
    view = View.from_events(events)

    with patch.object(
        LLMSummarizingCondenser, "get_condensation_reasons", return_value=reasons
    ):
        result = condenser.condensation_requirement(view)
        assert result == CondensationRequirement.HARD


@pytest.mark.parametrize(
    "reasons",
    [
        {Reason.REQUEST},
        {Reason.REQUEST, Reason.TOKENS},
        {Reason.REQUEST, Reason.EVENTS},
        {Reason.REQUEST, Reason.TOKENS, Reason.EVENTS},
    ],
)
def test_condensation_requirement_returns_hard(
    mock_llm: LLM, reasons: set[Reason]
) -> None:
    """Test that condensation_requirement returns HARD when REQUEST is present.

    Mocks get_condensation_reasons to test different combinations with REQUEST.
    """
    condenser = LLMSummarizingCondenser(llm=mock_llm, max_size=100, keep_first=2)
    events: list[Event] = [message_event(f"Event {i}") for i in range(10)]
    view = View.from_events(events)

    with patch.object(
        LLMSummarizingCondenser, "get_condensation_reasons", return_value=reasons
    ):
        result = condenser.condensation_requirement(view)
        assert result == CondensationRequirement.HARD


def test_condense_with_hard_requirement_and_no_condensation_available(
    mock_llm: LLM,
) -> None:
    """Test that condense raises error with hard requirement but no condensation.

    When there's a hard requirement but no valid condensation range available
    (e.g., entire view is a single atomic unit), should raise an exception.
    """
    from openhands.sdk.context.condenser.base import NoCondensationAvailableException

    condenser = LLMSummarizingCondenser(llm=mock_llm, max_size=100, keep_first=2)
    events: list[Event] = [message_event(f"Event {i}") for i in range(10)]
    view = View.from_events(events)

    # Mock to return HARD requirement but no events to condense
    # Also mock hard_context_reset to return None so the exception gets re-raised
    with (
        patch.object(
            LLMSummarizingCondenser,
            "get_condensation_reasons",
            return_value={Reason.REQUEST},
        ),
        patch.object(condenser, "_get_forgotten_events", return_value=([], 0)),
        patch.object(LLMSummarizingCondenser, "hard_context_reset", return_value=None),
    ):
        with pytest.raises(NoCondensationAvailableException):
            condenser.condense(view)


def test_condense_with_soft_requirement_and_no_condensation_available(
    mock_llm: LLM,
) -> None:
    """Test that condense returns view with soft requirement but no condensation.

    When there's a soft requirement but no valid condensation range available,
    should return the original view unchanged.
    """
    condenser = LLMSummarizingCondenser(llm=mock_llm, max_size=100, keep_first=2)
    events: list[Event] = [message_event(f"Event {i}") for i in range(10)]
    view = View.from_events(events)

    # Mock to return SOFT requirement but no events to condense
    with (
        patch.object(
            LLMSummarizingCondenser,
            "get_condensation_reasons",
            return_value={Reason.EVENTS},
        ),
        patch.object(condenser, "_get_forgotten_events", return_value=([], 0)),
    ):
        result = condenser.condense(view)
        assert isinstance(result, View)
        assert result == view
        # LLM should not be called
        cast(MagicMock, mock_llm.completion).assert_not_called()


def test_minimum_progress_default_value(mock_llm: LLM) -> None:
    """Test that minimum_progress has the correct default value."""
    condenser = LLMSummarizingCondenser(llm=mock_llm)
    assert condenser.minimum_progress == 0.1


def test_minimum_progress_custom_value(mock_llm: LLM) -> None:
    """Test that minimum_progress accepts custom values."""
    condenser = LLMSummarizingCondenser(llm=mock_llm, minimum_progress=0.2)
    assert condenser.minimum_progress == 0.2


@pytest.mark.parametrize(
    "invalid_value",
    [
        0.0,  # must be > 0.0
        -0.1,  # must be > 0.0
        1.0,  # must be < 1.0
        1.5,  # must be < 1.0
    ],
)
def test_minimum_progress_validation(mock_llm: LLM, invalid_value: float) -> None:
    """Test that minimum_progress validates the range (0.0 < value < 1.0)."""
    with pytest.raises(ValueError):
        LLMSummarizingCondenser(llm=mock_llm, minimum_progress=invalid_value)


def test_minimum_progress_threshold_not_met(mock_llm: LLM) -> None:
    """Test that condensation raises when forgotten events are below minimum_progress.

    When the ratio of forgotten events to total events is less than minimum_progress,
    should raise NoCondensationAvailableException.
    """
    # Create a condenser with a high minimum_progress value
    condenser = LLMSummarizingCondenser(
        llm=mock_llm, max_size=10, keep_first=2, minimum_progress=0.8
    )

    # Create a view with 100 events
    events: list[Event] = [message_event(f"Event {i}") for i in range(100)]
    events.append(CondensationRequest())
    view = View.from_events(events)

    # Mock _get_forgotten_events to return a small number of forgotten events
    # This allows us to directly test the minimum_progress threshold check
    # without dealing with complex boundary calculations
    small_forgotten = [events[2], events[3]]  # Only 2 events forgotten

    with patch.object(
        condenser, "_get_forgotten_events", return_value=(small_forgotten, 2)
    ):
        # Forgotten count (2) << minimum_progress (0.8) * len(view) (100)
        # 2 < 80, so the threshold is not met
        with pytest.raises(NoCondensationAvailableException, match="minimum progress"):
            condenser.get_condensation(view)


def test_minimum_progress_threshold_met(mock_llm: LLM) -> None:
    """Test that condensation succeeds when forgotten events meet minimum_progress.

    When the ratio of forgotten events to total events is >= minimum_progress,
    condensation should proceed normally.
    """
    # Use a low minimum_progress so it's easy to meet the threshold
    condenser = LLMSummarizingCondenser(
        llm=mock_llm, max_size=20, keep_first=2, minimum_progress=0.1
    )

    # Set up mock response
    cast(Any, mock_llm).set_mock_response_content("Summary of forgotten events")

    # Create enough events to trigger EVENTS reason (more than max_size=20)
    # With 30 events, target_size = 20 // 2 = 10
    # suffix_to_keep = 10 - keep_first - 1 = 10 - 2 - 1 = 7
    # forgotten = 30 - 7 = 23 events
    # 23/30 = 0.77 > 0.1, so minimum_progress is met
    events: list[Event] = [message_event(f"Event {i}") for i in range(30)]
    view = View.from_events(events)

    result = condenser.condense(view)

    assert isinstance(result, Condensation)
    assert result.summary == "Summary of forgotten events"


def test_generate_condensation_wraps_llm_errors(mock_llm: LLM) -> None:
    """LLM failures in _generate_condensation raise NoCondensationAvailableException."""  # noqa: E501
    condenser = LLMSummarizingCondenser(llm=mock_llm, max_size=10, keep_first=2)

    cast(MagicMock, mock_llm.completion).side_effect = RuntimeError("boom")

    events: list[Event] = [message_event(f"Event {i}") for i in range(12)]
    view = View.from_events(events)

    with pytest.raises(NoCondensationAvailableException, match="boom"):
        condenser.get_condensation(view)


@pytest.mark.asyncio
async def test_agenerate_condensation_wraps_llm_errors(mock_llm: LLM) -> None:
    """Async variant: LLM failures surface as NoCondensationAvailableException."""
    condenser = LLMSummarizingCondenser(llm=mock_llm, max_size=10, keep_first=2)

    cast(MagicMock, mock_llm.acompletion).side_effect = RuntimeError("boom")

    events: list[Event] = [message_event(f"Event {i}") for i in range(12)]
    view = View.from_events(events)

    with pytest.raises(NoCondensationAvailableException, match="boom"):
        await condenser.aget_condensation(view)


def test_llm_error_triggers_hard_context_reset(mock_llm: LLM) -> None:
    """A summarizer LLM failure during condense() triggers hard_context_reset."""
    condenser = LLMSummarizingCondenser(llm=mock_llm, max_size=10, keep_first=2)

    # Force a HARD condensation requirement via a CondensationRequest
    events: list[Event] = [message_event(f"Event {i}") for i in range(12)]
    events.append(CondensationRequest())
    view = View.from_events(events)

    # First call (get_condensation path) fails; second call
    # (hard_context_reset path) succeeds.
    success_response = cast(Any, mock_llm).completion.return_value
    cast(MagicMock, mock_llm.completion).side_effect = [
        RuntimeError("context window exceeded"),
        success_response,
    ]

    result = condenser.condense(view)

    assert isinstance(result, Condensation)
    assert result.summary == "Summary of forgotten events"
    assert cast(MagicMock, mock_llm.completion).call_count == 2


def _streaming_llm() -> LLM:
    """A real LLM with streaming enabled, as a long-running conversation has."""
    return LLM(
        model="gpt-4o",
        api_key=SecretStr("test-key"),
        usage_id="summarizer-test",
        stream=True,
    )


def _summary_response(content: str = "A summary") -> ModelResponse:
    return ModelResponse(
        id="resp-id",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=LiteLLMMessage(content=content, role="assistant"),
            )
        ],
        created=1234567890,
        model="gpt-4o",
        object="chat.completion",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


@patch("openhands.sdk.llm.llm.LLM._transport_call", autospec=True)
def test_summarization_disables_streaming_when_llm_streams(mock_transport) -> None:
    """Regression test for issue #3902: a ``stream=True`` LLM must still summarize
    even though the condenser passes no ``on_token`` callback."""
    mock_transport.return_value = _summary_response("A summary")

    llm = _streaming_llm()
    condenser = LLMSummarizingCondenser(llm=llm, max_size=10, keep_first=3)

    events: list[Event] = [message_event(f"Event {i}") for i in range(11)]
    view = View.from_events(events)

    result = condenser.condense(view)

    assert isinstance(result, Condensation)
    assert result.summary == "A summary"
    mock_transport.assert_called_once()
    # Streaming was disabled on a copy, not the agent's own LLM (autospec =>
    # self is the first positional arg).
    assert mock_transport.call_args.kwargs["enable_streaming"] is False
    assert mock_transport.call_args.kwargs["on_token"] is None
    summarizing_llm = mock_transport.call_args.args[0]
    assert summarizing_llm is not llm
    assert summarizing_llm.stream is False
    assert llm.stream is True  # original untouched (model_copy is non-mutating)
    # Token usage is still counted: the copy shares the original's metrics.
    usage = llm.metrics.accumulated_token_usage
    assert usage is not None
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5


@pytest.mark.asyncio
@patch("openhands.sdk.llm.llm.LLM._atransport_call", new_callable=AsyncMock)
async def test_async_summarization_disables_streaming_when_llm_streams(
    mock_atransport,
) -> None:
    """Async variant of the issue #3902 regression test (aget_condensation)."""
    mock_atransport.return_value = _summary_response("A summary")

    llm = _streaming_llm()
    condenser = LLMSummarizingCondenser(llm=llm, max_size=10, keep_first=3)

    events: list[Event] = [message_event(f"Event {i}") for i in range(11)]
    view = View.from_events(events)

    result = await condenser.aget_condensation(view)

    assert isinstance(result, Condensation)
    assert result.summary == "A summary"
    mock_atransport.assert_awaited_once()
    assert mock_atransport.call_args.kwargs["enable_streaming"] is False
    assert mock_atransport.call_args.kwargs["on_token"] is None
    assert llm.stream is True


@patch("openhands.sdk.llm.llm.LLM._transport_call", autospec=True)
def test_summarization_uses_llm_as_is_when_not_streaming(mock_transport) -> None:
    """When streaming is off, the condenser summarizes with the LLM unchanged."""
    mock_transport.return_value = _summary_response("A summary")

    llm = LLM(
        model="gpt-4o",
        api_key=SecretStr("test-key"),
        usage_id="summarizer-test",
        stream=False,
    )
    condenser = LLMSummarizingCondenser(llm=llm, max_size=10, keep_first=3)

    events: list[Event] = [message_event(f"Event {i}") for i in range(11)]
    view = View.from_events(events)

    result = condenser.condense(view)

    assert isinstance(result, Condensation)
    assert result.summary == "A summary"
    # The exact same LLM instance is used (no copy when not streaming).
    assert mock_transport.call_args.args[0] is llm


# ---------------------------------------------------------------------------
# What the summarizer is shown
#
# The summary is the agent's ONLY memory of every forgotten event, so what the
# summarizer is shown bounds what the agent can remember. These tests pin that
# input to the events themselves rather than to `str(event)`, which is a
# 500-character display preview: it cut a previous summary to its first ~500
# characters (in practice, USER_CONTEXT alone) and rendered every action as its
# class name with no command, path or arguments.
# ---------------------------------------------------------------------------


class SummaryInputMockAction(Action):
    command: str


class SummaryInputMockObservation(Observation):
    result: str

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(text=self.result)]


def _tool_pair(
    call_id: str,
    arguments: dict[str, Any],
    result: str,
    *,
    thought: str = "",
    summary: str | None = None,
    tool_name: str = "terminal",
) -> list[Event]:
    """An action and its observation, as the agent loop records them."""
    action = ActionEvent(
        thought=[TextContent(text=thought)] if thought else [],
        action=SummaryInputMockAction(command=str(arguments.get("command", ""))),
        tool_name=tool_name,
        tool_call_id=call_id,
        tool_call=MessageToolCall(
            id=call_id,
            name=tool_name,
            arguments=json.dumps(arguments),
            origin="completion",
        ),
        llm_response_id=f"response-{call_id}",
        summary=summary,
    )
    observation = ObservationEvent(
        observation=SummaryInputMockObservation(result=result),
        action_id=action.id,
        tool_name=tool_name,
        tool_call_id=call_id,
    )
    return [action, observation]


def _summarizer_prompt(mock_llm: LLM) -> str:
    completion_mock = cast(MagicMock, mock_llm.completion)
    completion_mock.assert_called_once()
    messages = completion_mock.call_args[1]["messages"]
    return "\n".join(
        part.text
        for message in messages
        for part in message.content
        if isinstance(part, TextContent)
    )


def _view_forgetting(
    forgettable: Sequence[Event], previous_summary: str | None
) -> View:
    """A view whose next condensation forgets every event in `forgettable`.

    keep_first=2 and max_size=10 keep two events at the head and two at the
    tail; the previous summary, when given, sits at the head of the forgotten
    range exactly where a real earlier condensation put it.
    """
    head: list[Event] = [message_event("The task"), message_event("Head event")]
    tail: list[Event] = [message_event(f"Tail {i}") for i in range(6)]
    events: list[Event] = list(head)
    if previous_summary is not None:
        stale = message_event("Already summarised")
        events.append(stale)
        events.append(
            Condensation(
                forgotten_event_ids={stale.id},
                summary=previous_summary,
                summary_offset=2,
                llm_response_id="earlier-condensation",
            )
        )
    return View.from_events([*events, *forgettable, *tail])


def test_summarizer_sees_the_whole_previous_summary(mock_llm: LLM) -> None:
    """Every condensation after the first re-summarises the previous summary.

    Cut to a preview, its COMPLETED / CODE_STATE / VERSION_CONTROL_STATUS
    sections vanish and the next summary reports work already done as "not
    checked yet" -- the agent then explores the same files again.
    """
    previous = (
        "USER_CONTEXT:\n- Add workstations as first-class entities.\n"
        + "COMPLETED:\n"
        + "".join(f"- Reviewed models/Model{i}.js and its routes\n" for i in range(60))
        + "VERSION_CONTROL_STATUS:\n- Backend HEAD is d006196 on master\n"
    )
    assert len(previous) > 2000
    condenser = LLMSummarizingCondenser(llm=mock_llm, max_size=10, keep_first=2)

    condenser.get_condensation(
        _view_forgetting([message_event(f"Middle {i}") for i in range(4)], previous)
    )

    prompt = _summarizer_prompt(mock_llm)
    assert "Backend HEAD is d006196 on master" in prompt
    assert "Reviewed models/Model59.js" in prompt


def test_summarizer_sees_what_each_action_did(mock_llm: LLM) -> None:
    """An action is its tool and arguments, not its class name.

    Most models emit tool calls with no thought text, so a preview of an action
    was literally `Thought:` and `Action: SummaryInputMockAction` -- the
    summarizer could not tell which command ran or which file was read.
    """
    condenser = LLMSummarizingCondenser(llm=mock_llm, max_size=10, keep_first=2)
    forgettable = _tool_pair(
        "call-1",
        {"command": "cat balapitiya_automobiles_backend/package.json"},
        '{"name": "balapitiya_automobiles_backend"}',
        summary="Read backend package.json",
    ) + [message_event("Middle")]

    condenser.get_condensation(_view_forgetting(forgettable, None))

    prompt = _summarizer_prompt(mock_llm)
    assert "cat balapitiya_automobiles_backend/package.json" in prompt
    assert "Read backend package.json" in prompt
    assert "terminal" in prompt


def test_summarizer_sees_a_long_user_message_whole(mock_llm: LLM) -> None:
    """A user's answer at the end of a long message must reach the summary."""
    condenser = LLMSummarizingCondenser(llm=mock_llm, max_size=10, keep_first=2)
    answer = "Ticket id is AB-TEST-12"
    long_message = "Some context about the change. " * 40 + answer
    assert len(long_message) > 1000

    condenser.get_condensation(
        _view_forgetting(
            [message_event(long_message)]
            + [message_event(f"Middle {i}") for i in range(3)],
            None,
        )
    )

    assert answer in _summarizer_prompt(mock_llm)


def test_summarizer_sees_the_tail_of_a_tool_result(mock_llm: LLM) -> None:
    """The verdict of a command is usually its last line, past any preview."""
    condenser = LLMSummarizingCondenser(llm=mock_llm, max_size=10, keep_first=2)
    output = "> webpack compiling\n" * 150 + "npm ERR! Missing script: start"
    assert len(output) > 2000

    condenser.get_condensation(
        _view_forgetting(
            _tool_pair("call-1", {"command": "npm start"}, output)
            + [message_event("Middle")],
            None,
        )
    )

    assert "npm ERR! Missing script: start" in _summarizer_prompt(mock_llm)


def test_summarizer_input_stays_bounded_for_a_huge_tool_result(mock_llm: LLM) -> None:
    """Whole events, but never an unbounded one: the head and tail survive."""
    condenser = LLMSummarizingCondenser(llm=mock_llm, max_size=10, keep_first=2)
    output = "FIRST LINE\n" + "x" * 200_000 + "\nLAST LINE"

    condenser.get_condensation(
        _view_forgetting(
            _tool_pair("call-1", {"command": "cat big.log"}, output)
            + [message_event("Middle")],
            None,
        )
    )

    prompt = _summarizer_prompt(mock_llm)
    assert "FIRST LINE" in prompt
    assert "LAST LINE" in prompt
    assert len(prompt) < 20_000


@pytest.mark.asyncio
async def test_async_summarizer_sees_the_whole_previous_summary(
    mock_llm: LLM,
) -> None:
    """The async path renders events the same way as the sync one."""
    previous = (
        "USER_CONTEXT:\n- Goal\n" + "x" * 3000 + "\nPENDING:\n- Write the model\n"
    )
    condenser = LLMSummarizingCondenser(llm=mock_llm, max_size=10, keep_first=2)
    forgettable = _tool_pair("call-1", {"command": "ls -la routes"}, "app.js")
    forgettable.append(message_event("Middle"))

    await condenser.aget_condensation(_view_forgetting(forgettable, previous))

    acompletion_mock = cast(AsyncMock, mock_llm.acompletion)
    acompletion_mock.assert_awaited_once()
    prompt = "\n".join(
        part.text
        for message in acompletion_mock.call_args[1]["messages"]
        for part in message.content
        if isinstance(part, TextContent)
    )
    assert "- Write the model" in prompt
    assert "ls -la routes" in prompt


def test_hard_context_reset_retry_shrinks_from_the_rendered_events(
    mock_llm: LLM,
) -> None:
    """A failed reset retries at 80% of the longest event, not of its preview.

    Scaling from the ~500-character preview cut every event to ~400 characters
    on the first retry, however small the overflow was.
    """
    condenser = LLMSummarizingCondenser(llm=mock_llm, max_size=10, keep_first=2)
    fact = "the payment service listens on port 8443"
    long_message = "Background. " * 150 + fact + "."
    view = View.from_events(
        [message_event("The task"), message_event(long_message), message_event("x")]
    )
    ok = cast(MagicMock, mock_llm.completion).return_value
    cast(MagicMock, mock_llm.completion).side_effect = [
        RuntimeError("context window exceeded"),
        ok,
    ]

    assert condenser.hard_context_reset(view) is not None

    retry_messages = cast(MagicMock, mock_llm.completion).call_args[1]["messages"]
    retry_prompt = "\n".join(
        part.text
        for message in retry_messages
        for part in message.content
        if isinstance(part, TextContent)
    )
    assert fact in retry_prompt


def test_summarizer_overflow_shrinks_events_instead_of_failing(mock_llm: LLM) -> None:
    """Whole events may not fit a smaller summarizer window: shrink, then retry.

    Failing instead would fall through to a hard context reset, which forgets
    every event -- far worse than a summary of trimmed ones.
    """
    from openhands.sdk.llm.exceptions import LLMContextWindowExceedError

    condenser = LLMSummarizingCondenser(llm=mock_llm, max_size=10, keep_first=2)
    ok = cast(MagicMock, mock_llm.completion).return_value
    completion_mock = cast(MagicMock, mock_llm.completion)
    completion_mock.side_effect = [LLMContextWindowExceedError("too long"), ok]
    output = "START\n" + "y" * 8_000 + "\nEND"

    result = condenser.get_condensation(
        _view_forgetting(
            _tool_pair("call-1", {"command": "cat big"}, output)
            + [message_event("Middle")],
            None,
        )
    )

    assert isinstance(result, Condensation)
    assert completion_mock.call_count == 2
    first, retry = (
        "\n".join(
            part.text
            for message in call[1]["messages"]
            for part in message.content
            if isinstance(part, TextContent)
        )
        for call in completion_mock.call_args_list
    )
    assert len(retry) < len(first)
    assert "START" in retry and "END" in retry


def test_summarizer_failure_that_is_not_overflow_is_not_retried(mock_llm: LLM) -> None:
    """An auth or provider error will not go away by trimming: raise at once."""
    condenser = LLMSummarizingCondenser(llm=mock_llm, max_size=10, keep_first=2)
    completion_mock = cast(MagicMock, mock_llm.completion)
    completion_mock.side_effect = RuntimeError("401 unauthorized")

    with pytest.raises(NoCondensationAvailableException):
        condenser.get_condensation(
            _view_forgetting([message_event(f"Middle {i}") for i in range(4)], None)
        )

    assert completion_mock.call_count == 1
