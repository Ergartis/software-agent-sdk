import re
from typing import Final


CMD_OUTPUT_PS1_BEGIN: Final[str] = "\n###PS1JSON###\n"
CMD_OUTPUT_PS1_END: Final[str] = "\n###PS1END###"
# Regex to match PS1 metadata blocks. Uses negative lookahead to handle corruption
# scenarios where concurrent output causes nested ###PS1JSON### markers. This ensures
# we match only the LAST ###PS1JSON### before each ###PS1END###.
CMD_OUTPUT_METADATA_PS1_REGEX: Final[re.Pattern[str]] = re.compile(
    rf"^{CMD_OUTPUT_PS1_BEGIN.strip()}((?:(?!{CMD_OUTPUT_PS1_BEGIN.strip()}).)*?){CMD_OUTPUT_PS1_END.strip()}",
    re.DOTALL | re.MULTILINE,
)

# Default max size for command output content
# to prevent too large observations from being saved in the stream
# This matches the default max_message_chars in LLM class
MAX_CMD_OUTPUT_SIZE: Final[int] = 30000


# Common timeout message that can be used across different timeout scenarios
TIMEOUT_MESSAGE_TEMPLATE: Final[str] = (
    "You may wait longer to see additional output by sending empty command '', "
    "send other commands to interact with the current process, send keys "
    '("C-c", "C-z", "C-d") '
    "to interrupt/kill the previous command before sending your new command, "
    "or use the timeout parameter in terminal for future commands."
)

# How long to wait with no new output before considering it a no-change timeout
NO_CHANGE_TIMEOUT_SECONDS: Final[int] = 30

# The same idea for a command the agent marked *blocking* by passing an explicit
# `timeout`.
#
# Passing a timeout disables the 30s check above on purpose: an agent that asks for
# `timeout=600` on a cold compile or a large install is saying "do not pester me",
# and quiet phases in those commands are real. But "quiet for a while" and "quiet
# forever" are different states, and only the first is legitimate — a command blocked
# reading stdin in a sandbox with no interactive user produces new output *never*, so
# the hard timeout was the only thing left to save it, and an agent that picked a
# generous timeout had effectively disabled every safety net it had.
#
# Observed 2026-08-24 on task 1f9a3848: an agent-written Playwright script ended with
# `input("Press Enter to close browser...")`. It sat for 19 minutes holding its
# budget, and the run only ended because a person pressed Stop.
#
# So: keep the 30s rule for non-blocking commands exactly as it was, and give blocking
# ones a far longer ceiling. Five minutes is comfortably above any quiet build phase
# and comfortably below the platform's 10-minute stall note, so the agent gets a
# chance to correct itself before a human is told the run went quiet.
BLOCKING_NO_CHANGE_TIMEOUT_SECONDS: Final[int] = 300

# What to tell the agent when a blocking command has gone silent. Deliberately
# different from TIMEOUT_MESSAGE_TEMPLATE: that one's first suggestion is "you may
# wait longer", which is the exactly wrong instinct here. Nothing is coming. Name the
# likely cause, and ask for a different approach rather than more patience.
BLOCKED_NO_OUTPUT_MESSAGE_TEMPLATE: Final[str] = (
    "This command has produced no new output for a long time, which usually means "
    "it is blocked waiting for something that will never arrive — most often input "
    "on stdin (input(), read, a confirmation prompt, a REPL, or a foreground "
    "process holding the terminal). Nothing is going to type here: this environment "
    "has no interactive user. "
    'Do not simply wait longer. Send "C-c" to interrupt it, then CHANGE APPROACH '
    "— re-run it non-interactively (remove interactive prompts, pass a "
    "--yes/--no-input style flag, redirect stdin from /dev/null), or start it in "
    "the background and poll its log file."
)
# How often to poll for new output in seconds
POLL_INTERVAL: Final[float] = 0.5
HISTORY_LIMIT: Final[int] = 10_000

TMUX_SOCKET_NAME: Final[str] = "openhands"

# Tmux session dimensions (columns x rows). Keep the viewport wide enough for
# common command output while leaving scrollback retention to HISTORY_LIMIT.
# Output wider than TMUX_SESSION_WIDTH columns will wrap; this is an accepted
# tradeoff to avoid the oversized 1000x1000 virtual terminal grid.
TMUX_SESSION_WIDTH: Final[int] = 256
TMUX_SESSION_HEIGHT: Final[int] = 200
