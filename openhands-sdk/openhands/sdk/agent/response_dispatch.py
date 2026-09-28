"""Classify LLM responses and dispatch to type-specific handlers.

Contains:
  - ``LLMResponseType`` — enum for response classification.
  - ``classify_response`` — pure classifier function (no side effects).
  - ``ResponseDispatchMixin`` — handler methods mixed into ``Agent``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import ActionEvent, MessageEvent
from openhands.sdk.llm import LLMResponse, Message, TextContent
from openhands.sdk.logger import get_logger
from openhands.sdk.tool.builtins.finish import FinishTool


if TYPE_CHECKING:
    from openhands.sdk.conversation import (
        ConversationCallbackType,
        ConversationState,
        LocalConversation,
    )
    from openhands.sdk.critic.base import CriticBase, CriticResult
    from openhands.sdk.llm import (
        MessageToolCall,
        ReasoningItemModel,
        RedactedThinkingBlock,
        ThinkingBlock,
    )
    from openhands.sdk.security.analyzer import SecurityAnalyzerBase

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class LLMResponseType(StrEnum):
    """Mutually exclusive classification of an LLM response."""

    TOOL_CALLS = "tool_calls"
    CONTENT = "content"
    REASONING_ONLY = "reasoning_only"
    EMPTY = "empty"


def classify_response(message: Message) -> LLMResponseType:
    """Classify an LLM response message into exactly one type.

    Decision priority (first match wins):
      1. TOOL_CALLS  — message contains tool calls
      2. CONTENT     — message contains non-blank TextContent
      3. REASONING_ONLY — message has reasoning but no visible content
      4. EMPTY       — nothing useful

    This function is pure: no side effects, no logging, no mutation.
    """
    if message.tool_calls:
        return LLMResponseType.TOOL_CALLS

    if any(isinstance(c, TextContent) and c.text.strip() for c in message.content):
        return LLMResponseType.CONTENT

    if (
        message.responses_reasoning_item is not None
        or message.reasoning_content is not None
        or message.thinking_blocks
    ):
        return LLMResponseType.REASONING_ONLY

    return LLMResponseType.EMPTY


# ---------------------------------------------------------------------------
# Dispatch mixin
# ---------------------------------------------------------------------------


def _consecutive_agent_prose_steps(state: ConversationState) -> int:
    """Message-only agent turns since the agent last acted, or the human last spoke.

    Derived from the event log rather than counted into a field, for two reasons. The
    log is what gets persisted and replayed, so a resumed conversation gets the same
    answer as a live one — a counter would silently reset on resume. And ``state``
    is a
    frozen-ish pydantic model shared across threads; the run loop already treats its
    event list as the single source of truth for what has happened.

    A human message resets the count: a back-and-forth is a conversation, and the guard
    exists for a model talking to itself.
    """
    count = 0
    for event in reversed(state.events):
        if isinstance(event, MessageEvent):
            if event.source != "agent":
                break
            count += 1
        elif isinstance(event, ActionEvent):
            break
    return count


# Appended to the message list for the one tool-free step, and never written to the
# event log — it steers that single turn without becoming part of the conversation's
# history.
#
# Removing the tools is what makes speech the only possible output; this is what makes
# it an ANSWER. Without it, live run 6ea4eeb5 (2026-08-21) took three forced steps and
# produced two empty responses and one line of unrelated narration, while the single
# good answer that run produced came from an ordinary step where the model happened to
# volunteer prose. An empty turn is worse than none: it burns a step, and the SDK then
# has to nudge the model back into calling tools.
#
# Phrased as the supervisor speaking, not as a system rule, because the model is being
# asked to do something it is otherwise trained out of mid-task: stop and talk.
REPLY_DIRECTIVE = (
    "The person supervising this run is waiting for an answer to the message above.\n\n"
    "Reply to them now, in plain text. You have no tools this turn, so a tool call is "
    "not "
    "available to you — and you do not need one: answer from what you already know.\n"
    "- Say what you have finished, what you are doing right now, and what is left.\n"
    "- If they asked something specific, answer that first and directly.\n"
    "- If they changed what they want, say so in one line and follow the new "
    "instruction.\n"
    "- Do not describe your next tool call. Do not return an empty message.\n\n"
    "Replying does NOT end your turn or finish the task — you carry on working "
    "immediately "
    "afterwards, with your tools back."
)


# Appended, like REPLY_DIRECTIVE, only to the message list of a call whose history ends
# with the agent's OWN message — never written to the event log.
#
# require_finish_tool is what produces that history: prose no longer finishes the
# run, so the next step is sent the log as it stands, ending in the agent's words.
# Some models accept that as a continuation; Claude (the 4.6 family onwards) rejects
# the whole call - "This model does not support assistant message prefill. The
# conversation must end with a user message." - so a Claude run died on the step
# after the agent first spoke (live run a7666d29, 2026-09-28). Kept out of the log
# for the reason REPLY_DIRECTIVE is: logged, it would read as a human turn and reset
# the chatter guard.
CONTINUE_DIRECTIVE = (
    "Your message above has been shown to the person supervising this run. Carry on "
    "with the task using your tools. When the task is complete, call the finish tool "
    "with a summary. If you cannot go on without the person's answer, call the finish "
    "tool and say exactly what you need from them."
)


def ends_with_agent_message(messages: list[Message]) -> bool:
    """Whether a call would be sent a history whose last turn is the assistant's own."""
    return bool(messages) and messages[-1].role == "assistant"


def reply_is_owed(state: ConversationState) -> bool:
    """Whether a person is waiting on an answer the agent has not given yet.

    True when the most recent *conversational* turn is a ``MessageEvent`` a human sent
    mid-run (``expects_reply``) and the agent has not spoken since. Only messages count:
    the debt is settled by answering, not by carrying on working, so the tool events an
    in-flight step writes after the question do not clear it. One reply flips it False,
    so it cannot loop, and a backlog of questions costs one answer rather than one each.

    Derived from the log rather than a flag on the agent for the same reason as
    :func:`_consecutive_agent_prose_steps`: the log is what is persisted and replayed,
    so a resumed conversation reaches the same answer as a live one.

    Why this exists: ``require_finish_tool`` made a mid-run reply *possible* —
    prose stopped meaning "finished". It did not make one happen. Observed on
    2026-08-20 across two runs (tasks eb38f9e2 and 79d8dcc4): the message was
    delivered into the LLM's context as a proper user turn, the system prompt invited
    an answer in as many words, and the agent produced 84 tool calls and zero prose.
    Mid-task, with tools on the table, acting always outranks answering. Permission is
    not obligation; the caller turns this into obligation by taking the tools away for
    exactly one step.
    """
    for event in reversed(state.events):
        if not isinstance(event, MessageEvent):
            # Tool traffic is irrelevant here, and treating it as an answer is the bug
            # this comment exists to prevent. A message accepted mid-step is
            # appended the moment it arrives, and the step already in flight then
            # writes its own ActionEvent and ObservationEvent AFTER it — so the
            # question is almost never the newest event by the time the next step
            # looks. Stopping at the first
            # ActionEvent made this dead on every real run while the unit tests, whose
            # LLM only ever produced prose, stayed green.
            continue
        if event.source == "agent":
            return False  # the agent has spoken since; the debt is settled
        if event.source == "user":
            return event.expects_reply
    return False


class ResponseDispatchMixin:
    """Handler methods for each ``LLMResponseType``. Mixed into ``Agent``.

    Expects the host class (``Agent``) to provide the members declared in the
    ``TYPE_CHECKING`` block below.
    """

    # Declared for pyright — the actual implementations live on Agent.
    if TYPE_CHECKING:
        critic: CriticBase | None
        require_finish_tool: bool
        max_consecutive_prose_steps: int
        _tools: dict[str, object]

        def _get_action_event(
            self,
            tool_call: MessageToolCall,
            conversation: LocalConversation,
            llm_response_id: str,
            on_event: ConversationCallbackType,
            security_analyzer: SecurityAnalyzerBase | None = None,
            thought: list[TextContent] | None = None,
            reasoning_content: str | None = None,
            thinking_blocks: (
                list[ThinkingBlock | RedactedThinkingBlock] | None
            ) = None,
            responses_reasoning_item: ReasoningItemModel | None = None,
        ) -> ActionEvent | None: ...

        def _execute_actions(
            self,
            conversation: LocalConversation,
            action_events: list[ActionEvent],
            on_event: ConversationCallbackType,
        ) -> None: ...

        async def _aexecute_actions(
            self,
            conversation: LocalConversation,
            action_events: list[ActionEvent],
            on_event: ConversationCallbackType,
        ) -> None: ...

        def _requires_user_confirmation(
            self,
            state: ConversationState,
            action_events: list[ActionEvent],
        ) -> bool: ...

        def _maybe_emit_vllm_tokens(
            self,
            llm_response: LLMResponse,
            on_event: ConversationCallbackType,
        ) -> None: ...

        def _evaluate_with_critic(
            self,
            conversation: LocalConversation,
            event: ActionEvent | MessageEvent,
        ) -> CriticResult | None: ...

    def _handle_tool_calls(
        self,
        message: Message,
        llm_response: LLMResponse,
        conversation: LocalConversation,
        state: ConversationState,
        on_event: ConversationCallbackType,
    ) -> None:
        """Handle LLM response containing tool calls."""
        if not all(isinstance(c, TextContent) for c in message.content):
            logger.warning(
                "LLM returned tool calls but message content is not all "
                "TextContent - ignoring non-text content"
            )

        thought_content = [c for c in message.content if isinstance(c, TextContent)]

        action_events: list[ActionEvent] = []
        assert message.tool_calls, "classify_response guarantees tool_calls"
        for i, tool_call in enumerate(message.tool_calls):
            action_event = self._get_action_event(
                tool_call,
                conversation=conversation,
                llm_response_id=llm_response.id,
                on_event=on_event,
                security_analyzer=state.security_analyzer,
                thought=thought_content if i == 0 else [],
                reasoning_content=(message.reasoning_content if i == 0 else None),
                thinking_blocks=(list(message.thinking_blocks) if i == 0 else []),
                responses_reasoning_item=(
                    message.responses_reasoning_item if i == 0 else None
                ),
            )
            if action_event is None:
                continue
            action_events.append(action_event)

        if self._requires_user_confirmation(state, action_events):
            return

        if action_events:
            self._execute_actions(conversation, action_events, on_event)

        self._maybe_emit_vllm_tokens(llm_response, on_event)

    async def _ahandle_tool_calls(
        self,
        message: Message,
        llm_response: LLMResponse,
        conversation: LocalConversation,
        state: ConversationState,
        on_event: ConversationCallbackType,
    ) -> None:
        """Async variant of :meth:`_handle_tool_calls`.

        Delegates tool execution to :meth:`_aexecute_actions` so each
        tool call runs in its own thread and multiple calls are scheduled
        concurrently via :func:`asyncio.gather`.
        """
        if not all(isinstance(c, TextContent) for c in message.content):
            logger.warning(
                "LLM returned tool calls but message content is not all "
                "TextContent - ignoring non-text content"
            )

        thought_content = [c for c in message.content if isinstance(c, TextContent)]

        action_events: list[ActionEvent] = []
        assert message.tool_calls, "classify_response guarantees tool_calls"
        for i, tool_call in enumerate(message.tool_calls):
            action_event = self._get_action_event(
                tool_call,
                conversation=conversation,
                llm_response_id=llm_response.id,
                on_event=on_event,
                security_analyzer=state.security_analyzer,
                thought=thought_content if i == 0 else [],
                reasoning_content=(message.reasoning_content if i == 0 else None),
                thinking_blocks=(list(message.thinking_blocks) if i == 0 else []),
                responses_reasoning_item=(
                    message.responses_reasoning_item if i == 0 else None
                ),
            )
            if action_event is None:
                continue
            action_events.append(action_event)

        if self._requires_user_confirmation(state, action_events):
            return

        if action_events:
            await self._aexecute_actions(conversation, action_events, on_event)

        self._maybe_emit_vllm_tokens(llm_response, on_event)

    def _handle_content_response(
        self,
        message: Message,
        llm_response: LLMResponse,
        conversation: LocalConversation,
        state: ConversationState,
        on_event: ConversationCallbackType,
    ) -> None:
        """Handle an LLM response with text content.

        By default this finishes the conversation: prose means the agent has nothing
        further to do and is handing back to the human.

        With ``require_finish_tool`` the two are separated — prose is a *reply*, and
        only an explicit finish tool call ends the run. That is what lets an agent
        answer a question mid-task and carry on working, which it otherwise cannot
        do: answering would be quitting.
        """
        self._emit_message_event(message, llm_response, conversation, on_event)
        self._maybe_emit_vllm_tokens(llm_response, on_event)

        # An agent with no finish tool has prose as its ONLY exit. Honouring the flag
        # there would leave it running until something else killed it.
        if not (self.require_finish_tool and self._has_finish_tool()):
            logger.debug("LLM produced a message response - awaits user input")
            state.execution_status = ConversationExecutionStatus.FINISHED
            return

        prose_steps = _consecutive_agent_prose_steps(state)
        if prose_steps >= self.max_consecutive_prose_steps:
            # A model that has stopped calling tools is stuck, not deliberating. Left
            # alone it monologues until the iteration budget dies, which costs a tenant
            # real money and reads as a hang.
            logger.info(
                "agent produced %d consecutive message-only responses "
                "(limit %d) - finishing",
                prose_steps,
                self.max_consecutive_prose_steps,
            )
            state.execution_status = ConversationExecutionStatus.FINISHED
            return

        logger.debug(
            "LLM produced a message response - keeps working (%d/%d prose steps)",
            prose_steps,
            self.max_consecutive_prose_steps,
        )

    def _has_finish_tool(self) -> bool:
        """Whether this agent has a way to end the conversation other than prose."""
        return FinishTool.name in self._tools

    def _handle_no_content_response(
        self,
        message: Message,
        llm_response: LLMResponse,
        conversation: LocalConversation,
        state: ConversationState,  # noqa: ARG002
        on_event: ConversationCallbackType,
        *,
        response_type: LLMResponseType,
    ) -> None:
        """Handle LLM response with no user-facing content.

        Covers both reasoning-only and empty responses. Emits the message
        event and sends corrective feedback so the model knows it must
        produce a tool call or user-facing content.
        """
        if response_type is LLMResponseType.EMPTY:
            logger.warning("LLM produced empty response - continuing agent loop")
        self._emit_message_event(message, llm_response, conversation, on_event)
        self._maybe_emit_vllm_tokens(llm_response, on_event)
        self._send_corrective_nudge(on_event)

    def _emit_message_event(
        self,
        message: Message,
        llm_response: LLMResponse,
        conversation: LocalConversation,
        on_event: ConversationCallbackType,
    ) -> MessageEvent:
        """Create and emit a MessageEvent, running critic if configured."""
        msg_event = MessageEvent(
            source="agent",
            llm_message=self._mask_secrets(message, conversation),
            llm_response_id=llm_response.id,
        )
        if self.critic is not None and self.critic.mode == "finish_and_message":
            critic_result = self._evaluate_with_critic(conversation, msg_event)
            if critic_result is not None:
                msg_event = msg_event.model_copy(
                    update={"critic_result": critic_result}
                )
        on_event(msg_event)
        return msg_event

    @staticmethod
    def _mask_secrets(message: Message, conversation: LocalConversation) -> Message:
        """Return ``message`` with registered secret values masked in its text.

        ``thinking_blocks`` and ``responses_reasoning_item`` are left alone:
        they are signed provider payloads, and rewriting them invalidates the
        signature replayed on the next request.
        """
        mask = conversation.state.secret_registry.mask_secrets_in_output
        reasoning = message.reasoning_content
        return message.model_copy(
            update={
                "content": [
                    part.model_copy(update={"text": mask(part.text)})
                    if isinstance(part, TextContent)
                    else part
                    for part in message.content
                ],
                "reasoning_content": mask(reasoning) if reasoning else reasoning,
            }
        )

    def _send_corrective_nudge(self, on_event: ConversationCallbackType) -> None:
        """Inject corrective feedback when no tool call and no content.

        The model still receives this as a user-role message, but the event
        source marks that it came from the framework rather than the human.
        """
        logger.warning(
            "LLM response contained no tool call and no content"
            " - sending corrective feedback"
        )
        nudge = MessageEvent(
            source="environment",
            llm_message=Message(
                role="user",
                content=[
                    TextContent(
                        text=(
                            "Your last response did not include a "
                            "function call or a message. Please "
                            "use a tool to proceed with the task."
                        )
                    )
                ],
            ),
        )
        on_event(nudge)
