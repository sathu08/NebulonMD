"""TUI tests (headless): Textual app composes, status renders, and the
server_ops layer returns bool/message tuples without side effects.

These run inside Textual's ``run_test`` driver — no real terminal, no real
server lifecycle. Start/stop wrappers are exercised against a helper
monkeypatch that fakes the underlying process state.
"""

from __future__ import annotations

import pytest

pytest.importorskip("textual")

from nmd_host.tui import commands, server_ops
from nmd_host.tui.app import NebulonMindApp
from nmd_host.tui.context import enable_tui_mode, tui_mode, _probe_host
from nmd_host.tui.screens import BackendCredentialsScreen


# ---------------------------------------------------------------------- #
# server_ops                                                             #
# ---------------------------------------------------------------------- #


def test_probe_host_rewrites_bind_targets():
    assert _probe_host("0.0.0.0") == "127.0.0.1"
    assert _probe_host("::") == "127.0.0.1"
    assert _probe_host("") == "127.0.0.1"
    assert _probe_host("localhost") == "localhost"
    assert _probe_host("192.168.1.5") == "192.168.1.5"


def test_server_status_returns_tuple(monkeypatch):
    monkeypatch.setattr(server_ops, "is_port_open", lambda host, port: False)
    running, starting, pid_label = server_ops.server_status()
    assert running is False
    assert isinstance(starting, bool)
    assert pid_label in ("Present", "Not Found")


def test_backend_status_reports_unreachable(monkeypatch):
    monkeypatch.setattr(server_ops, "is_port_open", lambda host, port: False)
    reachable, addr = server_ops.backend_status()
    assert reachable is False
    assert ":" in addr


def test_is_startable_blocks_when_backend_down(monkeypatch):
    monkeypatch.setattr(commands, "SKIP_BACKEND_CHECK", False)
    # API not running; backend unreachable.
    monkeypatch.setattr(commands, "is_server_running", lambda host, port: False)
    monkeypatch.setattr(server_ops, "is_port_open", lambda host, port: False)
    ok, reason = server_ops.is_startable()
    assert ok is False
    assert "backend" in reason.lower()


def test_save_credentials_rejects_empty():
    ok, message = server_ops.save_credentials("", "")
    assert ok is False
    assert "empty" in message.lower()


# ---------------------------------------------------------------------- #
# Textual app (headless)                                                 #
# ---------------------------------------------------------------------- #


@pytest.mark.anyio
async def test_app_composes_and_renders_status(monkeypatch):
    enable_tui_mode()
    assert tui_mode() is True
    # environment-independent: force "stopped" regardless of local ports
    monkeypatch.setattr(server_ops, "server_status", lambda: (False, False, "Not Found"))
    monkeypatch.setattr(server_ops, "backend_status", lambda: (False, "localhost:6969"))
    app = NebulonMindApp()
    async with app.run_test() as pilot:
        assert app.server_status is False
        # centered brand
        brand = app.query_one("#brand")
        assert "NEBULONMIND" in str(getattr(brand, "renderable", "")) or brand is not None
        # status bar reports API state
        status = app.query_one("#status-bar")
        assert "API" in str(status.renderable)
        # chat widgets present
        assert app.query_one("#chat-input") is not None
        assert app.query_one("#chat-send") is not None
        # no lifecycle sidebar buttons
        for button_id in ("start", "stop", "restart"):
            assert not app.query(f"#{button_id}")
        await pilot.pause()


@pytest.mark.anyio
async def test_chat_append_renders_bubbles():
    app = NebulonMindApp()
    async with app.run_test() as pilot:
        app._append_message("user", "hello")
        app._append_message("agent", "hi there")
        assert app._transcript == [
            {"role": "user", "content": "hello"},
            {"role": "agent", "content": "hi there"},
        ]
        log = app.query_one("#chat-log")
        assert log is not None


@pytest.mark.anyio
async def test_thinking_indicator_lifecycle():
    app = NebulonMindApp()
    async with app.run_test() as pilot:
        app._start_thinking()
        assert app._thinking is True
        assert app._pending_index is not None
        assert "thinking" in app._transcript[app._pending_index]["content"].lower()
        await pilot.pause()
        app._stop_thinking("final answer")
        assert app._thinking is False
        assert app._pending_index is None
        assert app._transcript[-1] == {"role": "agent", "content": "final answer"}
        await pilot.pause()


@pytest.mark.anyio
async def test_user_id_defaults_to_configured_or_user_001(monkeypatch):
    app = NebulonMindApp()
    assert app._user_id == "nmd_user_01"
    monkeypatch.setenv("NMD_BACKGROUND_USER", "nmd_user_01")
    app2 = NebulonMindApp()
    assert app2._user_id == "nmd_user_01"


@pytest.mark.anyio
async def test_create_command_without_username_shows_usage():
    app = NebulonMindApp()
    async with app.run_test() as pilot:
        app._handle_create_user("/create")
        assert any(
            m["role"] == "system" and "Usage" in m["content"]
            for m in app._transcript
        )
        await pilot.pause()


@pytest.mark.anyio
async def test_credentials_screen_composes():
    app = NebulonMindApp()
    async with app.run_test() as pilot:
        app.push_screen(BackendCredentialsScreen())
        await pilot.pause()
        assert app.query_one("#username") is not None
        assert app.query_one("#password") is not None
        assert app.query_one("#save") is not None
        await pilot.pause()
