"""A *blocking* command that goes silent forever must still hand control back.

Passing an explicit ``timeout`` marks a command blocking, which switches off the
ordinary
30-second no-change check on purpose: an agent that asks for ``timeout=600`` on a cold
compile is
saying "do not pester me". That exemption is right for a command that is quiet for a
*while* and
wrong for one that is quiet *forever*.

Task ``1f9a3848`` (2026-08-24) is the case that motivated this. An agent wrote a
Playwright script
ending in ``input("Press Enter to close browser...")``, ran it with a timeout, and the
run sat for
19 minutes holding its budget with the browser open. Nothing in the polling loop could
end it: the
only remaining net was a hard timeout the agent had itself set generously. A person
pressed Stop.

These tests drive the rule directly rather than a live terminal. That is not a shortcut
— the
end-to-end terminal tests in this directory do not run on every host (the tmux backend
returns
instantly with empty output on some, which makes
``test_long_running_command_follow_by_execute``
fail there too, unrelated to this change). A rule that decides when to interrupt a run
should not
be verifiable only where a terminal happens to work, so it lives in ``timeout_policy``
as a pure
function, the same choice the platform made for its own stall detector.
"""

from openhands.tools.terminal.constants import (
    BLOCKED_NO_OUTPUT_MESSAGE_TEMPLATE,
    BLOCKING_NO_CHANGE_TIMEOUT_SECONDS,
    NO_CHANGE_TIMEOUT_SECONDS,
)
from openhands.tools.terminal.timeout_policy import should_break_blocked_command


class TestTheRule:
    def test_a_blocking_command_silent_past_the_ceiling_is_broken(self):
        """The regression. This is the state task 1f9a3848 sat in for 19 minutes."""
        assert should_break_blocked_command(
            is_blocking=True,
            time_since_last_change=BLOCKING_NO_CHANGE_TIMEOUT_SECONDS,
        )

    def test_it_fires_at_the_boundary_and_not_before(self):
        assert not should_break_blocked_command(
            is_blocking=True,
            time_since_last_change=BLOCKING_NO_CHANGE_TIMEOUT_SECONDS - 0.01,
        )
        assert should_break_blocked_command(
            is_blocking=True,
            time_since_last_change=BLOCKING_NO_CHANGE_TIMEOUT_SECONDS + 0.01,
        )

    def test_a_blocking_command_that_is_merely_quiet_is_left_alone(self):
        """The exemption this must not break.

        A cold compile or a large install goes quiet for stretches and is still working.
        Interrupting it every 30 seconds is the noise ``is_blocking`` exists to prevent,
        so
        anything short of the ceiling must be left entirely alone.
        """
        for quiet_for in (0, 1, 30, 60, 120, 299):
            assert not should_break_blocked_command(
                is_blocking=True, time_since_last_change=quiet_for
            ), f"interrupted a working command after only {quiet_for}s"

    def test_non_blocking_commands_are_untouched(self):
        """Check 2 (the 30s rule) still owns these; this rule must never pre-empt it."""
        assert not should_break_blocked_command(
            is_blocking=False, time_since_last_change=10_000
        )

    def test_the_ceiling_can_be_disabled(self):
        assert not should_break_blocked_command(
            is_blocking=True,
            time_since_last_change=10_000,
            blocking_no_change_timeout_seconds=0,
        )


class TestTheThresholdItself:
    def test_it_is_far_above_the_ordinary_no_change_rule(self):
        """Otherwise it would defeat the ``is_blocking`` exemption it is built
        around."""
        assert BLOCKING_NO_CHANGE_TIMEOUT_SECONDS > NO_CHANGE_TIMEOUT_SECONDS * 5

    def test_it_lands_before_the_platform_tells_a_human(self):
        """Ergatis' TaskStallDetector notes a stalled run at 10 minutes and deliberately
        does
        not kill it. The agent should get its chance to self-correct before that note is
        the
        only thing that has happened."""
        assert BLOCKING_NO_CHANGE_TIMEOUT_SECONDS < 10 * 60


class TestWhatTheAgentIsTold:
    def test_it_actively_countermands_waiting(self):
        """The ordinary timeout message opens with "you may wait longer", which is the
        exactly
        wrong instinct when nothing is ever coming. It is not enough to omit that advice
        — the
        agent has the other message in context too, so this one has to overrule it."""
        assert "Do not simply wait longer" in BLOCKED_NO_OUTPUT_MESSAGE_TEMPLATE

    def test_it_names_the_cause_and_asks_for_a_different_approach(self):
        msg = BLOCKED_NO_OUTPUT_MESSAGE_TEMPLATE
        assert "stdin" in msg
        assert "no interactive user" in msg
        assert "C-c" in msg  # how to get out
        assert "CHANGE APPROACH" in msg  # and what to do instead
