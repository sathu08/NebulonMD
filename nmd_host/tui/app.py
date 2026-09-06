# nmd_host/tui/app.py
"""Interactive NebulonMind chat TUI (Textual).

An opencode-style chat interface: the NebulonMind brand sits centered at
the top, a thin status bar reports the API / backend state, and the main
area is a live agent chat. The server lifecycle is managed from the CLI
(``nebulonmind start|stop|restart``), not from inside the TUI.
"""

from __future__ import annotations

import os

import requests

from textual import events
from textual.binding import Binding
from textual.reactive import reactive

from textual.app import App, ComposeResult
from textual.widgets.option_list import Option
from textual.containers import Horizontal, Vertical, VerticalScroll

from textual.widgets import Button, Footer, Header, Input, Markdown, OptionList, Static

from nmd_host.utils.constants import (
    APP_NAME,
    BRAND_TAGLINE,
    DEFAULT_USERNAME,
    NEBULONMIND_BANNER,
    TUI_USAGE,
)

from . import commands, server_ops
from . import options as command_options
from .screens import BackendCredentialsScreen, ConfigKeyScreen, UsernameScreen

from .context import _probe_host, cfg, enable_tui_mode, setup_nebulonmind_paths


class CommandInput(Input):
    """Chat input that drives the command-menu box rendered above it.

    While the menu is visible the input takes over the up/down/tab/enter
    keys (exactly like other terminal bots): up/down move the highlight,
    tab completes the highlighted command, enter runs the highlighted
    command, escape hides the menu. Normal chat messages are untouched.
    """

    def on_key(self, event: events.Key) -> None:
        menu = self._menu()
        if menu is not None and menu.display:
            if event.key == "escape":
                menu.display = False
                palette = self._palette()
                if palette is not None:
                    palette.display = False
                event.stop()
                return
            if event.key in ("up", "down"):
                (menu.action_cursor_up if event.key == "up" else menu.action_cursor_down)()
                event.stop()
                return
            if event.key == "tab":
                self._complete_from_menu(menu)
                event.stop()
                return
            if event.key == "enter":
                menu.action_select()
                event.stop()
                return
        # otherwise: default behaviour (submit, arrows, etc.)

    def _palette(self) -> Vertical | None:
        try:
            return self.app.query_one("#command-palette", Vertical)
        except Exception:
            return None

    def _menu(self) -> OptionList | None:
        try:
            return self.app.query_one("#command-menu", OptionList)
        except Exception:
            return None

    def _complete_from_menu(self, menu: OptionList) -> None:
        command = _highlighted_command(menu)
        if command:
            self.value = command
            self.cursor_position = len(command)
            self.action_end()


def _highlighted_command(menu: OptionList) -> str | None:
    index = menu.highlighted
    if index is None:
        return None
    try:
        option_id = menu._options[index].id
    except (IndexError, AttributeError):
        return None
    if not option_id or option_id.startswith(":header:"):
        return None
    return option_id


def _valid_username(username: str) -> bool:
    """A usable NebulonMind username: non-empty and not command-like.

    ``"/create"`` / ``"/setup"`` are slash commands, never usernames — a
    mistyped command must never register a bogus user or leak into the cfg.
    """
    username = (username or "").strip()
    return bool(username) and not username.startswith("/")


class NebulonMindApp(App):

    TITLE = APP_NAME

    CSS = """
    Screen {
        background: #0b0f14;
    }

    #brand {
        height: 5;
        content-align: center middle;
        text-style: bold;
        color: #00e5ff;
    }

    #brand-subtitle {
        height: 0.5;
        content-align: center middle;
        color: #546e7a;
    }

    #status-bar {
        height: 3;
        border: solid #263238;
        padding: 0 2;
        content-align: center middle;
    }

    #chat-log-container {
        width: 100%;
        height: 1fr;
        border: solid #263238;
        padding: 1;
    }

    #chat-log {
        width: 100%;
    }

    #command-palette {
        display: none;
        margin-top: 1;
        width: 100%;
        border: round #00e5ff;
        background: #0f1a22;
    }

    #command-hint {
        height: 1;
        padding: 0 2;
        color: #546e7a;
        text-style: bold;
        background: #12202b;
    }

    #command-menu {
        width: 100%;
        height: auto;
        background: transparent;
        padding: 0 0;
    }

    #command-menu .option-list--option {
        padding: 0 2;
    }

    #command-menu .option-list--option-highlighted {
        background: #123b4d;
        color: #00e5ff;
        text-style: bold;
    }

    #command-menu .option-list--option-disabled {
        color: #7aa3b0;
        text-style: bold;
    }

    #cfg-status {
        height: 1;
        color: #546e7a;
    }

    #cfg-list {
        height: 1fr;
        width: 100%;
        border: solid #263238;
    }

    #chat-input-row {
        height: 3;
        width: 100%;
        margin-top: 1;
    }

    #chat-input {
        width: 1fr;
    }

    #chat-send {
        width: 12;
        margin-left: 1;
    }

    .user-msg {
        color: #4dd0e1;
    }

    .agent-msg {
        color: #ffd54f;
    }

    .muted {
        color: #546e7a;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "stop_or_quit", "Stop LLM / Quit"),
    ]

    server_status = reactive(False)

    def __init__(self, user_id: str = "") -> None:
        super().__init__()
        self._user_id = (
            user_id or os.environ.get("NMD_BACKGROUND_USER", "") or DEFAULT_USERNAME
        ).strip()
        if not _valid_username(self._user_id):
            self._user_id = DEFAULT_USERNAME
        self._transcript: list = []
        self._pending_index: int | None = None
        self._thinking = False
        self._thinking_timer = None
        self._agent_run_id = 0
        self._agent_worker = None

    # ======================================================
    # Lifecycle
    # ======================================================

    def on_mount(self) -> None:
        self.refresh_status()
        self.set_interval(3, self.refresh_status)

    def _refresh_subtitle(self) -> None:
        try:
            subtitle = self.query_one("#brand-subtitle", Static)
        except Exception:
            return
        subtitle.update(BRAND_TAGLINE)

    def action_stop_or_quit(self) -> None:
        """Esc while the LLM is responding cancels the request; otherwise quit."""
        if self._thinking:
            self._agent_run_id += 1
            worker = self._agent_worker
            self._agent_worker = None
            if worker is not None:
                worker.cancel()
            self._stop_thinking("_(stopped — no answer)_")
            self.send_button.disabled = False
            self.notify("Agent response cancelled.", severity="warning")
            self.query_one("#chat-input", Input).focus()
            return
        self.exit()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("✦\n" + NEBULONMIND_BANNER, id="brand")
        yield Static(BRAND_TAGLINE, id="brand-subtitle")
        yield Static("", id="status-bar")
        yield VerticalScroll(Markdown("", id="chat-log"), id="chat-log-container")
        with Vertical(id="command-palette"):
            yield Static(
                "Commands — ↑/↓ navigate  •  Tab complete  •  Enter run  •  Esc close",
                id="command-hint",
            )
            yield OptionList(id="command-menu")
        with Horizontal(id="chat-input-row"):
            yield CommandInput(
                placeholder="Message the NebulonMind agent (or type / to see commands)...",
                id="chat-input",
            )
            yield Button("Send", id="chat-send", variant="primary")
        yield Footer()

    # ======================================================
    # Status bar
    # ======================================================

    def refresh_status(self) -> None:
        running, starting, pid_label = server_ops.server_status()
        self.server_status = running
        reachable, backend_addr = server_ops.backend_status()
        if running:
            status = f"[bold green]● API RUNNING[/bold green]  •  {commands.HOST}:{commands.PORT}  •  [bold cyan]http://{_probe_host(commands.HOST)}:{commands.PORT}/api/NebulonMind/dashboard/[/bold cyan]"
        elif starting:
            status = "[bold yellow]● API STARTING...[/bold yellow]"
        else:
            status = "[bold red]● API STOPPED[/bold red]  •  start with `nebulonmind start`"
        backend = (
            f"[green]backend up[/green] {backend_addr}"
            if reachable
            else f"[red]backend down[/red] {backend_addr}"
        )
        user = self._user_id
        try:
            bar = self.query_one("#status-bar", Static)
        except Exception:
            return
        bar.update(
            f"{status}  •  {backend}  •  [bold cyan]user: {user}[/bold cyan]"
        )

    # ======================================================
    # Live agent chat (opencode-style)
    # ======================================================

    def _chat_url(self) -> str:
        return f"http://{_probe_host(commands.HOST)}:{commands.PORT}/api/NebulonMind/agent/chat"

    def _render_log(self) -> None:
        log = self.query_one("#chat-log", Markdown)
        lines = []
        for msg in self._transcript:
            if msg["role"] == "user":
                lines.append(f"### 👤 You\n\n{msg['content']}\n")
            elif msg["role"] == "system":
                lines.append(f"### ⚙ System\n\n{msg['content']}\n")
            else:
                lines.append(f"### 🤖 NebulonMind\n\n{msg['content']}\n")
        log.update("\n\n".join(lines))
        container = self.query_one("#chat-log-container", VerticalScroll)
        container.scroll_end(animate=True)

    def _append_message(self, role: str, content: str) -> None:
        self._transcript.append({"role": role, "content": content})
        self._render_log()

    def _start_thinking(self) -> None:
        """Append a live 'processing' placeholder and animate it."""
        self._pending_index = len(self._transcript)
        self._transcript.append(
            {"role": "agent", "content": "_NebulonMind is thinking_", "pending": True}
        )
        self._dash_count = 1
        self._thinking = True
        self._render_log()
        self._thinking_timer = self.set_interval(0.4, self._tick_thinking)

    def _tick_thinking(self) -> None:
        if not self._thinking or self._pending_index is None:
            return
        dots = "." * (self._dash_count % 4)
        self._dash_count += 1
        self._transcript[self._pending_index]["content"] = f"_NebulonMind is thinking{dots}_"
        self._render_log()

    def _stop_thinking(self, final_content: str) -> None:
        self._thinking = False
        if self._thinking_timer is not None:
            self._thinking_timer.stop()
            self._thinking_timer = None
        if self._pending_index is not None and self._pending_index < len(self._transcript):
            self._transcript[self._pending_index] = {
                "role": "agent",
                "content": final_content,
            }
        self._pending_index = None
        self._render_log()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "chat-input":
            self.send_message()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "chat-input":
            self._update_command_menu(event.value)

    # ------------------------------------------------------------------ #
    # Command menu (dropdown box above the chat input)                   #
    # ------------------------------------------------------------------ #

    def _command_menu(self) -> OptionList | None:
        try:
            return self.query_one("#command-menu", OptionList)
        except Exception:
            return None

    def _palette(self) -> Vertical | None:
        try:
            return self.query_one("#command-palette", Vertical)
        except Exception:
            return None

    def _update_command_menu(self, value: str) -> None:
        menu = self._command_menu()
        if menu is None:
            return
        value = value or ""
        if not value.startswith("/"):
            self._hide_command_menu()
            return

        query = value.lstrip("/").split(" ", 1)[0].lower()
        sections = command_options.categories_for_query(query)

        menu.clear_options()
        first_enabled: int | None = None
        index = 0
        for category, options in sections:
            menu.add_option(
                Option(f"[{category.upper()}][/]", id=f":header:{category}", disabled=True)
            )
            index += 1
            for option in options:
                if first_enabled is None:
                    first_enabled = index
                menu.add_option(
                    Option(
                        f"{option.command:<12} {option.description}",
                        id=option.command,
                    )
                )
                index += 1

        if first_enabled is not None:
            try:
                menu.highlighted = first_enabled
            except Exception:
                pass

        palette = self._palette()
        if sections:
            if palette is not None:
                palette.display = True
            menu.display = True
        else:
            self._hide_command_menu()

    def _hide_command_menu(self) -> None:
        menu = self._command_menu()
        if menu is not None:
            menu.display = False
        palette = self._palette()
        if palette is not None:
            palette.display = False

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        command = event.option.id
        event.stop()
        self._run_command(command)

    def _run_command(self, command: str) -> None:
        self._hide_command_menu()
        input_widget = self.query_one("#chat-input", CommandInput)
        typed = input_widget.value
        input_widget.value = ""
        option, args, _ = command_options.resolve(typed)
        if option is not None and option.command == command:
            self._dispatch(option.command, args)
        else:
            self._dispatch(command, args)

    def _dispatch(self, command: str, args: str = "") -> None:
        """Run a slash ``command`` (canonical spelling) with optional ``args``."""
        name = (command or "").lower().lstrip("/").strip()
        if name == "create":
            if args:
                self._handle_create_user(args)
            else:
                self._open_create_popup()
        elif name == "setup":
            if args:
                self._handle_setup_user(args)
            else:
                self._open_setup_popup(args)
        elif name == "settings":
            self._open_config_editor()
        elif name == "credentials":
            self._open_credentials()
        elif name == "start":
            self._handle_server_op("start")
        elif name == "stop":
            self._handle_server_op("stop")
        elif name == "restart":
            self._handle_server_op("restart")
        elif name == "status":
            self._dispatch_status()
        elif name == "whoami":
            self._dispatch_whoami()
        elif name == "clear":
            self._dispatch_clear()
        elif name == "help":
            self._dispatch_help()
        else:
            self._append_message(
                "system",
                f"Unknown command: `/{name}` — type `/help` to see all commands.",
            )

    # ---------- command handlers ----------

    def _handle_server_op(self, action: str) -> None:
        self._append_message("user", f"/{action}")
        try:
            if action == "start":
                ok, message = server_ops.start_server()
            elif action == "stop":
                ok, message = server_ops.stop_server()
            else:
                ok, message = server_ops.restart_server()
        except Exception as exc:  # noqa: BLE001 - surface platform errors gracefully
            ok, message = False, f"/{action} failed: {exc}"
        self._append_message("agent", f"_({message})_" if not ok else message)
        self.refresh_status()

    def _dispatch_status(self) -> None:
        running, starting, pid_label = server_ops.server_status()
        reachable, backend_addr = server_ops.backend_status()
        state = "RUNNING" if running else ("STARTING" if starting else "STOPPED")
        lines = (
            f"API: **{state}**  •  {commands.HOST}:{commands.PORT}  •  pid {pid_label}\n"
            f"Backend: {'**up**' if reachable else '**down**'}  •  {backend_addr}\n"
            f"Chatting as: **{self._user_id}**"
        )
        self._append_message("agent", lines)

    def _dispatch_whoami(self) -> None:
        self._append_message("agent", f"Currently chatting as `{self._user_id}`.")

    def _dispatch_clear(self) -> None:
        self._transcript.clear()
        self._render_log()
        self.notify("Chat cleared.", severity="information")

    def _dispatch_help(self) -> None:
        lines = ["## Available commands\n"]
        for category, options in command_options.grouped_options():
            lines.append(f"### {category}\n")
            for option in options:
                lines.append(f"- `{option.help}` — {option.description}")
        self._append_message("agent", "\n".join(lines))

    def _open_credentials(self) -> None:
        self.push_screen(BackendCredentialsScreen())

    def send_message(self) -> None:
        if self._thinking:
            self.notify("NebulonMind is still processing your previous message.", severity="warning")
            return
        input_widget = self.query_one("#chat-input", Input)
        text = input_widget.value.strip()
        if not text:
            return
        input_widget.value = ""
        option, args, _ = command_options.resolve(text)
        if option is not None:
            self._dispatch(option.command, args)
            return
        if text.startswith("/"):
            self._append_message(
                "system",
                f"Unknown command: `{text}` — type `/help` to see all commands.",
            )
            return
        if not server_ops.server_status()[0]:
            self.notify("NebulonMind server is not running.", severity="warning")
            self._append_message("user", text)
            self._append_message(
                "agent",
                "_(server offline — start it with `nebulonmind start`)_",
            )
            return
        self._append_message("user", text)
        self.send_button.disabled = True
        self._start_thinking()
        self._agent_run_id += 1
        run_id = self._agent_run_id
        self._agent_worker = self.run_worker(
            self._call_agent(text, run_id), thread=True
        )

    def _handle_create_user(self, username: str) -> None:
        """Register the user ``username`` via the API."""
        username = (username or "").strip()
        if not _valid_username(username):
            self._append_message("system", "Usage: `/create <username>` — e.g. `/create nmd_user_01`")
            return
        self._append_message("user", f"/create {username}")
        self.send_button.disabled = True
        self.run_worker(self._create_user(username), exclusive=True, thread=True)

    def _handle_setup_user(self, username: str) -> None:
        """Switch to an existing user ``username`` (never auto-creates)."""
        username = (username or "").strip()
        if not _valid_username(username):
            self._append_message("system", "Usage: `/setup <username>` — e.g. `/setup nmd_user_01`")
            return
        self._append_message("user", f"/setup {username}")
        self.send_button.disabled = True
        self.run_worker(self._setup_user(username), exclusive=True, thread=True)

    async def _create_user(self, username: str) -> None:
        if not _valid_username(username):
            self.call_from_thread(self._fail_user, "invalid username")
            return
        try:
            resp = requests.post(
                f"http://{_probe_host(commands.HOST)}:{commands.PORT}/api/NebulonMind/user/create_user",
                json={"username": username},
                timeout=(5, 30),
            )
            body = resp.json()
            if resp.ok and body.get("success"):
                created = bool(body.get("data", {}).get("created"))
                self.call_from_thread(
                    self._finish_create_user, username, created
                )
            else:
                detail = body.get("message") or body.get("detail") or f"HTTP {resp.status_code}"
                self.call_from_thread(self._fail_user, f"could not create user: {detail}")
        except requests.RequestException as exc:
            self.call_from_thread(self._fail_user, f"could not reach API: {exc}")

    def _finish_create_user(self, username: str, created: bool) -> None:
        self._user_id = username
        cfg.set_background_user(username)
        self._append_message(
            "agent",
            f"User `{username}` "
            + ("created. " if created else "already existed. ")
            + f"Now chatting as `{username}` (saved to nebulonmind.cfg).",
        )
        self.refresh_status()
        self._refresh_subtitle()
        self._end_user_flow()

    def _fail_user(self, message: str) -> None:
        self._append_message("agent", f"_({message})_")
        self._end_user_flow()

    def _end_user_flow(self) -> None:
        self.send_button.disabled = False
        self.query_one("#chat-input", Input).focus()

    @property
    def send_button(self) -> Button:
        return self.query_one("#chat-send", Button)

    # ------------------------------------------------------------------ #
    # /create and /setup modal flows                                     #
    # ------------------------------------------------------------------ #

    def _open_config_editor(self) -> None:
        """Open the nebulonmind.cfg editor (reads/writes through the API)."""
        if not server_ops.server_status()[0]:
            self.notify("NebulonMind server is not running.", severity="warning")
            return
        self.push_screen(ConfigKeyScreen())

    def _open_create_popup(self) -> None:
        screen = UsernameScreen(
            title="Create a new user",
            hint=(
                "Enter a NEW NebulonMind username (e.g. `nmd_user_01`). "
                "It gets its own opaque user_id in NebulonDB."
            ),
            submit_label="Create",
        )
        self.push_screen(screen, self._on_create_result)

    def _open_setup_popup(self, initial: str = "") -> None:
        screen = UsernameScreen(
            title="Switch to an existing user",
            hint=(
                "Enter a username that is already registered. "
                "To create a NEW user, use `/create` instead."
            ),
            submit_label="Switch",
            initial=initial,
        )
        self.push_screen(screen, self._on_setup_result)

    def _on_create_result(self, username: str | None) -> None:
        if not username:
            self.query_one("#chat-input", Input).focus()
            return
        self._handle_create_user(username)

    def _on_setup_result(self, username: str | None) -> None:
        if not username:
            self.query_one("#chat-input", Input).focus()
            return
        self._handle_setup_user(username)

    async def _setup_user(self, username: str) -> None:
        """Switch to an existing registered username (never auto-creates)."""
        if not _valid_username(username):
            self.call_from_thread(self._fail_user, "invalid username")
            return
        try:
            resp = requests.post(
                f"http://{_probe_host(commands.HOST)}:{commands.PORT}/api/NebulonMind/user/setup",
                json={"username": username},
                timeout=(5, 30),
            )
            body = resp.json()
            if resp.ok and body.get("success"):
                self.call_from_thread(self._finish_setup_user, username)
            else:
                detail = body.get("message") or body.get("detail") or f"HTTP {resp.status_code}"
                self.call_from_thread(
                    self._fail_user,
                    f"user `{username}` is not registered: {detail} — use `/create {username}` to make it",
                )
        except requests.RequestException as exc:
            self.call_from_thread(self._fail_user, f"could not reach API: {exc}")

    def _finish_setup_user(self, username: str) -> None:
        self._user_id = username
        cfg.set_background_user(username)
        self.refresh_status()
        self._refresh_subtitle()
        self._append_message(
            "agent", f"Now chatting as existing user `{username}` (saved to nebulonmind.cfg)."
        )
        self._end_user_flow()

    async def _call_agent(self, text: str, run_id: int) -> None:
        prior = [
            {
                "role": "assistant" if m["role"] == "agent" else m["role"],
                "content": m["content"],
            }
            for m in self._transcript[:-1]
            if m.get("role") in ("user", "agent") and not m.get("pending")
        ]
        payload = {"text": text, "messages": prior}
        answer = None
        try:
            resp = requests.post(
                self._chat_url(),
                json=payload,
                params={"user_id": self._user_id},
                timeout=(5, 120),
            )
            body = resp.json()
            if resp.status_code != 200 or not body.get("success"):
                detail = body.get("message") or body.get("detail") or f"HTTP {resp.status_code}"
                answer = f"_(error: {detail})_"
            else:
                answer = body["data"]["answer"]
        except requests.RequestException as exc:
            answer = f"_(could not reach API: {exc})_"
        finally:
            # A newer message, or the user pressing Esc to stop, invalidates
            # this run_id — never let a stale worker touch the UI.
            if self._agent_run_id != run_id:
                return
            self.call_from_thread(self._stop_thinking, answer or "_(no answer)_")
            self.send_button.disabled = False
            self.query_one("#chat-input", Input).focus()


# ==========================================================
#         Main Entry Point (CLI + TUI dispatcher)
# ==========================================================

USAGE = TUI_USAGE


def main():
    import sys

    if len(sys.argv) < 2:
        # No command -> launch the interactive TUI.
        enable_tui_mode()
        setup_nebulonmind_paths()
        app = NebulonMindApp()
        app.run()
        return

    command = sys.argv[1].lower()

    if command in ("--help", "-h", "help"):
        print(USAGE)
        sys.exit(0)

    foreground = "--foreground" in sys.argv or "-f" in sys.argv
    force = "--force" in sys.argv or "-F" in sys.argv

    if command == "start":
        ok = commands.start_server(foreground=foreground)
        sys.exit(0 if ok else 1)
    elif command == "stop":
        ok = commands.stop_server(force=force)
        sys.exit(0 if ok else 1)
    elif command == "restart":
        ok = commands.restart_server(foreground=foreground, force=force)
        sys.exit(0 if ok else 1)
    else:
        print(f"Invalid command. Usage: nebulonmind {{start|stop|restart}} [--foreground|-f] [--force|-F]")
        sys.exit(1)
