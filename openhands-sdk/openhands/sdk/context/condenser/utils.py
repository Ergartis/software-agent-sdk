from collections.abc import Sequence

from openhands.sdk.event.base import LLMConvertibleEvent
from openhands.sdk.event.condenser import CondensationSummaryEvent
from openhands.sdk.event.llm_convertible import (
    ActionEvent,
    AgentErrorEvent,
    MessageEvent,
    ObservationEvent,
    UserRejectObservation,
)
from openhands.sdk.event.llm_convertible.system import SystemPromptEvent
from openhands.sdk.llm import LLM, content_to_str
from openhands.sdk.utils import maybe_truncate


def render_event_for_summary(
    event: LLMConvertibleEvent, max_length: int | None = None
) -> str:
    """Render one forgotten event as the summarizing LLM should see it.

    Not ``str(event)``: that is a display preview capped at ``N_CHAR_PREVIEW``
    (500) characters, and it renders an action as its class name alone. Fed to
    the summarizer it cut every previous summary to its first ~500 characters
    and hid which command ran, which file was read and what a long user message
    ended with -- so each condensation forgot the work the last one recorded,
    and the agent repeated it.

    This shows what the agent itself was shown: a previous summary whole, an
    action as its tool and arguments, a result, error or user message as its
    text. ``max_length`` caps the rendered event, keeping head and tail; event
    types not listed here keep their ``str()`` form.
    """
    if isinstance(event, CondensationSummaryEvent):
        text = f"PREVIOUS SUMMARY ({event.source})\n{event.summary}"
    elif isinstance(event, ActionEvent):
        lines = [f"ActionEvent ({event.source})", f"  Tool: {event.tool_name}"]
        if event.summary:
            lines.append(f"  Summary: {event.summary}")
        thought = " ".join(t.text for t in event.thought).strip()
        if thought:
            lines.append(f"  Thought: {thought}")
        lines.append(f"  Arguments: {event.tool_call.arguments}")
        if event.action is None:
            lines.append("  (not executed)")
        text = "\n".join(lines)
    elif isinstance(event, ObservationEvent):
        result = "".join(content_to_str(event.observation.to_llm_content))
        text = (
            f"ObservationEvent ({event.source})\n"
            f"  Tool: {event.tool_name}\n  Result: {result}"
        )
    elif isinstance(event, UserRejectObservation):
        text = (
            f"UserRejectObservation ({event.source})\n  Tool: {event.tool_name}\n"
            f"  Rejected by {event.rejection_source}: {event.rejection_reason}"
        )
    elif isinstance(event, AgentErrorEvent):
        text = (
            f"AgentErrorEvent ({event.source})\n  Tool: {event.tool_name}\n"
            f"  Error: {event.error}"
        )
    elif isinstance(event, MessageEvent):
        body = " ".join(content_to_str(event.llm_message.content)) or "[no text]"
        text = f"MessageEvent ({event.source})\n  {event.llm_message.role}: {body}"
        if event.activated_skills:
            text += f"\n  [Skills: {', '.join(event.activated_skills)}]"
    else:
        text = str(event)
    return maybe_truncate(text, truncate_after=max_length)


def get_total_token_count(
    events: Sequence[LLMConvertibleEvent],
    llm: LLM,
) -> int:
    """Calculate the total token count for a list of LLM convertible events.

    This function converts the events to LLM messages and uses the provided LLM
    to count the total number of tokens. This is useful for understanding how many
    tokens a sequence of events will consume in the context window. A view is
    expected to have one system prompt event; if multiple are present, only the
    first system prompt's tools are included.

    Args:
        events: List of LLM convertible events to count tokens for
        llm: The LLM instance to use for token counting (uses the litellm's token
            counting utilities)

    Returns:
        Total token count for all events converted to messages

    Example:
        >>> from openhands.sdk.llm import LLM
        >>> from openhands.sdk.event.llm_convertible import MessageEvent
        >>>
        >>> llm = LLM(model="gpt-4")
        >>> events = [
        ...     MessageEvent.from_text("Hello, how are you?", source="user"),
        ...     MessageEvent.from_text("I'm doing great!", source="agent"),
        ... ]
        >>> token_count = get_total_token_count(events, llm)
        >>> print(f"Total tokens: {token_count}")
    """
    messages = LLMConvertibleEvent.events_to_messages(list(events))
    tools = next(
        (event.tools for event in events if isinstance(event, SystemPromptEvent)),
        None,
    )
    return llm.get_token_count(
        messages,
        tools=tools or None,
        # Security-risk tokens are always included in real tool requests.
        add_security_risk_prediction=bool(tools),
    )


def get_shortest_prefix_above_token_count(
    events: Sequence[LLMConvertibleEvent],
    llm: LLM,
    token_count: int,
    base_events: Sequence[LLMConvertibleEvent] | None = None,
) -> int:
    """Find the length of the shortest prefix whose token count exceeds the target.

    This function performs a binary search to efficiently find the shortest prefix
    of events that, when converted to messages, has a total token count greater than
    the specified target token count.

    Args:
        events: List of LLM convertible events to search through
        llm: The LLM instance to use for token counting (uses the model's tokenizer)
        token_count: The target token count threshold

    Returns:
        The length of the shortest prefix that exceeds the token count.
        Returns 0 if no events are provided.
        Returns len(events) if all events combined don't exceed the token count.

    Example:
        >>> from openhands.sdk.llm import LLM
        >>> from openhands.sdk.event.llm_convertible import MessageEvent
        >>>
        >>> llm = LLM(model="gpt-4")
        >>> events = [
        ...     MessageEvent.from_text("Hi", source="user"),
        ...     MessageEvent.from_text("Hello", source="agent"),
        ...     MessageEvent.from_text("How are you?", source="user"),
        ...     MessageEvent.from_text("Great!", source="agent"),
        ... ]
        >>> prefix_len = get_shortest_prefix_above_token_count(events, llm, 20)
        >>> # prefix_len might be 2 if first 2 events exceed 20 tokens
    """
    if not events:
        return 0

    base_events = base_events or []
    base_tokens = get_total_token_count(base_events, llm) if base_events else 0

    # Check if all events combined don't exceed the token count
    total_tokens = get_total_token_count([*base_events, *events], llm) - base_tokens
    if total_tokens <= token_count:
        return len(events)

    # Binary search for the shortest prefix
    left, right = 1, len(events)

    while left < right:
        mid = (left + right) // 2
        prefix_tokens = (
            get_total_token_count([*base_events, *events[:mid]], llm) - base_tokens
        )

        if prefix_tokens > token_count:
            # This prefix exceeds the count, try to find a shorter one
            right = mid
        else:
            # This prefix doesn't exceed, we need a longer one
            left = mid + 1

    return left


def get_suffix_length_for_token_reduction(
    events: Sequence[LLMConvertibleEvent],
    llm: LLM,
    token_reduction: int,
    base_events: Sequence[LLMConvertibleEvent] | None = None,
) -> int:
    """Find how many suffix events can be kept while reducing tokens by target amount.

    This function determines the maximum number of events from the end of the list
    that can be retained while ensuring the total token count is reduced by at least
    the specified amount. It uses the get_shortest_prefix_above_token_count function
    to find the prefix that must be removed.

    Args:
        events: List of LLM convertible events
        llm: The LLM instance to use for token counting (uses the model's tokenizer)
        token_reduction: The minimum number of tokens to reduce by

    Returns:
        The number of events from the end that can be kept (suffix length).

    Example:
        >>> from openhands.sdk.llm import LLM
        >>> from openhands.sdk.event.llm_convertible import MessageEvent
        >>>
        >>> llm = LLM(model="gpt-4")
        >>> events = [
        ...     MessageEvent.from_text("Event 1", source="user"),
        ...     MessageEvent.from_text("Event 2", source="agent"),
        ...     MessageEvent.from_text("Event 3", source="user"),
        ...     MessageEvent.from_text("Event 4", source="agent"),
        ... ]
        >>> # Suppose total is 100 tokens, and we want to reduce by 40 tokens
        >>> suffix_len = get_suffix_length_for_token_reduction(events, llm, 40)
        >>> # suffix_len tells us how many events from the end we can keep
        >>> # If first 2 events = 45 tokens, suffix_len = 2 (keep last 2 events)
    """
    if not events:
        return 0

    if token_reduction <= 0:
        return len(events)

    # Find the shortest prefix that exceeds the token reduction target
    prefix_length = get_shortest_prefix_above_token_count(
        events,
        llm,
        token_reduction,
        base_events=base_events,
    )

    # The suffix length is what remains after removing the prefix
    suffix_length = len(events) - prefix_length

    return suffix_length
