"""Runtime-idle-aware terminal timeout policy."""

import os
from collections.abc import Mapping

from openhands.tools.terminal.constants import BLOCKING_NO_CHANGE_TIMEOUT_SECONDS


RUNTIME_IDLE_TIMEOUT_SECONDS_ENV = "OH_RUNTIME_IDLE_TIMEOUT_SECONDS"
MAX_FOREGROUND_TIMEOUT_RATIO = 0.9


def get_runtime_idle_timeout_seconds(
    env: Mapping[str, str] | None = None,
) -> float | None:
    """Return configured runtime idle timeout in seconds, if any.

    The runtime-api injects this value into managed runtime pods. Local and
    self-hosted deployments that do not have idle cleanup can leave it unset,
    which disables the foreground terminal timeout cap.
    """
    raw = (env or os.environ).get(RUNTIME_IDLE_TIMEOUT_SECONDS_ENV)
    if raw is None or raw.strip() == "":
        return None

    try:
        value = float(raw)
    except ValueError:
        return None

    if value <= 0:
        return None
    return value


def get_max_foreground_timeout_seconds(
    runtime_idle_timeout_seconds: float | None = None,
) -> float | None:
    """Return max safe foreground terminal timeout, or None when uncapped."""
    idle_timeout = (
        get_runtime_idle_timeout_seconds()
        if runtime_idle_timeout_seconds is None
        else runtime_idle_timeout_seconds
    )
    if idle_timeout is None:
        return None
    return idle_timeout * MAX_FOREGROUND_TIMEOUT_RATIO


def format_seconds(value: float) -> str:
    """Format second values compactly for user-facing observations."""
    return f"{value:g}"


def foreground_timeout_rejection_message(
    requested_timeout_seconds: float,
    runtime_idle_timeout_seconds: float,
) -> str:
    """Build the user-facing rejection for unsafe foreground timeouts."""
    max_timeout = get_max_foreground_timeout_seconds(runtime_idle_timeout_seconds)
    assert max_timeout is not None
    ratio_percent = int(MAX_FOREGROUND_TIMEOUT_RATIO * 100)
    return (
        "Refusing to run foreground terminal command with "
        f"timeout={format_seconds(requested_timeout_seconds)}s.\n\n"
        "This runtime is subject to idle cleanup after "
        f"{format_seconds(runtime_idle_timeout_seconds)}s. Foreground terminal "
        f"commands are capped at {ratio_percent}% of that threshold, currently "
        f"{format_seconds(max_timeout)}s, so commands can return control before "
        "the runtime is considered idle.\n\n"
        "For long-running work, use a shorter timeout, omit the timeout to use "
        "the terminal soft-timeout/polling flow, or start the command in the "
        "background and poll its log file."
    )


def foreground_timeout_rejection_for(
    *,
    command: str,
    is_input: bool,
    timeout: float | None,
    runtime_idle_timeout_seconds: float | None = None,
) -> str | None:
    """Return a rejection message if a terminal action is unsafe to run."""
    if is_input or not command.strip() or timeout is None:
        return None

    idle_timeout = (
        get_runtime_idle_timeout_seconds()
        if runtime_idle_timeout_seconds is None
        else runtime_idle_timeout_seconds
    )
    max_timeout = get_max_foreground_timeout_seconds(idle_timeout)
    if idle_timeout is None or max_timeout is None or timeout <= max_timeout:
        return None

    return foreground_timeout_rejection_message(timeout, idle_timeout)


def should_break_blocked_command(
    *,
    is_blocking: bool,
    time_since_last_change: float,
    blocking_no_change_timeout_seconds: float = BLOCKING_NO_CHANGE_TIMEOUT_SECONDS,
) -> bool:
    """Whether a *blocking* command has been silent long enough to hand control back.

    Passing an explicit ``timeout`` marks a command blocking, which switches off the
    ordinary
    30-second no-change check by design: an agent that asks for ``timeout=600`` on a
    cold
    compile is saying "do not pester me", and quiet phases in such commands are real.
    That
    exemption is right for a command that is quiet for a *while* and wrong for one that
    is quiet
    *forever* — a command blocked reading stdin in a sandbox with no interactive user
    produces
    new output never, so the hard timeout became the only remaining net, and an agent
    that chose
    a generous timeout had disabled every net it had.

    Observed on task ``1f9a3848`` (2026-08-24): an agent-written Playwright script
    ending in
    ``input("Press Enter to close browser...")`` held a live run for 19 minutes and was
    ended by
    a person, not by this loop.

    The rule keys on time since the last *change*, deliberately not on "produced no
    output at
    all": the motivating script printed a full page of results and only then blocked, so
    a
    "never produced anything" guard would miss the very case it exists for. What
    separates
    blocked from busy is not whether output ever appeared but how long ago the last byte
    did.

    Acting on this is safe because it **returns control rather than killing**: a
    genuinely long
    quiet phase costs one round-trip — the agent sends ``''`` and carries on waiting —
    while a
    real block gets broken.

    Pure, so the rule is testable without a terminal, a clock, or a container.
    """
    if not is_blocking:
        return False  # check 2 (the 30s rule) already owns non-blocking commands
    if blocking_no_change_timeout_seconds <= 0:
        return False  # disabled
    return time_since_last_change >= blocking_no_change_timeout_seconds

