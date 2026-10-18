"""A command that outlives its soft timeout keeps running in pool mode.

Upstream's pool returned the pane to the pool after every call, and the next
checkout sent C-c to anything still running there. So the advice every timeout
observation gives — "send empty command '' to keep waiting" — killed the command
instead, and a follow-up landed on a fresh pane that answered "No previous
running command". An agent asked to wait out a quiet two-minute step re-ran it
forever (Ergatis task c0d1349c, 2026-10-17: 43 commands, never finished).
"""

import tempfile
import time

import pytest

from openhands.tools.terminal.definition import TerminalAction
from openhands.tools.terminal.impl import TerminalExecutor
from openhands.tools.terminal.terminal.factory import _is_tmux_available


pytestmark = pytest.mark.skipif(not _is_tmux_available(), reason="needs tmux")


@pytest.fixture
def executor():
    with tempfile.TemporaryDirectory() as work_dir:
        ex = TerminalExecutor(
            working_dir=work_dir,
            terminal_type="tmux",
            max_panes=3,
            no_change_timeout_seconds=1,
        )
        assert ex.is_pooled
        yield ex
        ex.close()


def _wait_for(ex: TerminalExecutor, marker: str, limit: float = 20.0) -> str:
    seen = ""
    deadline = time.time() + limit
    while time.time() < deadline:
        obs = ex(TerminalAction(command=""))
        seen += obs.text
        if marker in seen:
            return seen
    return seen


def test_empty_command_keeps_waiting_instead_of_killing(executor):
    first = executor(TerminalAction(command="sleep 4 && echo SLEEP-DONE-7f3"))
    assert first.metadata.exit_code == -1
    assert executor._held is not None

    seen = _wait_for(executor, "SLEEP-DONE-7f3")

    assert "SLEEP-DONE-7f3" in seen
    assert "No previous running command" not in seen
    assert executor._held is None


def test_new_command_runs_beside_the_held_one(executor):
    executor(TerminalAction(command="sleep 4 && echo LONG-DONE-91a"))

    quick = executor(TerminalAction(command="echo QUICK-42b"))

    assert "QUICK-42b" in quick.text
    assert quick.metadata.exit_code == 0
    assert "LONG-DONE-91a" in _wait_for(executor, "LONG-DONE-91a")


def test_ctrl_c_reaches_the_held_command(executor):
    executor(TerminalAction(command="sleep 60"))

    executor(TerminalAction(command="C-c", is_input=True))

    assert executor._held is None
    after = executor(TerminalAction(command=""))
    assert "No previous running command" in after.text


def test_follow_up_with_nothing_held_still_says_so(executor):
    obs = executor(TerminalAction(command=""))

    assert "No previous running command" in obs.text


def test_second_timeout_takes_the_slot_and_returns_the_first(executor):
    executor(TerminalAction(command="sleep 60"))
    first = executor._held
    executor(TerminalAction(command="sleep 3 && echo SECOND-DONE-c4d"))

    assert executor._held is not None
    assert executor._held is not first
    assert "SECOND-DONE-c4d" in _wait_for(executor, "SECOND-DONE-c4d")


def test_reset_releases_the_held_pane(executor):
    executor(TerminalAction(command="sleep 60"))

    executor(TerminalAction(command="", reset=True))

    assert executor._held is None
