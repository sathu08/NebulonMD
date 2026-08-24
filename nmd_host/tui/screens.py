# nmd_host/tui/screens.py
"""Modal screens for the NebulonMind terminal UI."""

from __future__ import annotations

from textual.screen import Screen
from textual.binding import Binding
from textual.app import ComposeResult

from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Input, Static
from textual.widgets.option_list import Option
from textual.widgets import OptionList

from . import server_ops
from .server_ops import save_credentials


# ==========================================================
# TUI SCREENS
# ==========================================================


class BackendCredentialsScreen(Screen):
    """Prompt for the NebulonDB backend credentials and persist them to .env."""

    BINDINGS = [
        Binding("escape", "cancel", "Back"),
        Binding("up", "focus_previous", "Up"),
        Binding("down", "focus_next", "Down"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="modal-container"):
            yield Static("NebulonDB Backend Credentials", id="modal-title")
            yield Static(
                "NebulonMind connects to the NebulonDB API on localhost:6969. "
                "Enter the credentials stored in NebulonDB's account hub.",
                id="modal-hint",
            )
            yield Input(placeholder="Username", id="username")
            yield Input(placeholder="Password", password=True, id="password")
            yield Input(placeholder="Confirm Password", password=True, id="confirm-password")
            with Horizontal(id="modal-buttons"):
                yield Button("Save", variant="success", id="save")
                yield Button("Cancel", variant="error", id="cancel")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.action_cancel()
            return
        if event.button.id == "save":
            username = self.query_one("#username", Input).value.strip()
            password = self.query_one("#password", Input).value
            confirm = self.query_one("#confirm-password", Input).value

            if not username:
                self.app.notify("Username cannot be empty.", severity="error")
                return
            if len(password) < 8:
                self.app.notify("Password must be at least 8 characters.", severity="error")
                return
            if password != confirm:
                self.app.notify("Passwords do not match.", severity="error")
                return

            ok, message = save_credentials(username, password)
            self.app.notify(message, severity="success" if ok else "error")
            if ok:
                self.app.pop_screen()
                self.app.call_after_refresh(self.app.refresh_status)

    def action_cancel(self) -> None:
        self.app.pop_screen()


class UsernameScreen(Screen):
    """Reusable modal: collect a username and dismiss with it (or ``None``).

    Used by ``/create`` (register a new username) and ``/setup`` (switch to
    an already-registered username). The submit/cancel buttons and Enter key
    both dismiss the screen; the app decides what to do with the value.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Back"),
        Binding("up", "focus_previous", "Up"),
        Binding("down", "focus_next", "Down"),
    ]

    def __init__(
        self,
        title: str = "Enter a username",
        hint: str = "",
        submit_label: str = "OK",
        initial: str = "",
    ) -> None:
        super().__init__()
        self._title_text = title
        self._hint_text = hint
        self._submit_label = submit_label
        self._initial = initial

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="modal-container"):
            yield Static(self._title_text, id="modal-title-username")
            yield Static(self._hint_text, id="modal-hint-username")
            yield Input(placeholder="Username", id="username", value=self._initial)
            with Horizontal(id="modal-buttons"):
                yield Button(self._submit_label, variant="primary", id="submit")
                yield Button("Cancel", variant="error", id="cancel")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        if event.button.id == "submit":
            self._submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "username":
            self._submit()

    def _submit(self) -> None:
        username = self.query_one("#username", Input).value.strip()
        if not username:
            self.app.notify("Username cannot be empty.", severity="error")
            return
        self.dismiss(username)

    def action_cancel(self) -> None:
        self.dismiss(None)


# ==========================================================
# CONFIG EDITOR (nebulonmind.cfg — via the API)
# ==========================================================


class ConfigKeyScreen(Screen):
    """Browse the grouped settings of ``nebulonmind.cfg``.

    Uses the API (``GET /api/NebulonMind/config/cfg``) so it shows the same
    grouped, typed view as the website. Selecting a setting opens a small
    editor to change it; edits are collected as a ``{section: {key: value}}``
    diff and written back through ``PUT /api/NebulonMind/config/cfg``.
    """

    BINDINGS = [
        Binding("escape", "close", "Back"),
        Binding("up", "focus_previous", "Up"),
        Binding("down", "focus_next", "Down"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._groups: list = []
        self._meta: dict = {}
        self._edits: dict = {}

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="modal-container"):
            yield Static("NebulonMind Settings", id="modal-title-username")
            yield Static(
                "Select a setting to edit. Changes are written to "
                "nebulonmind.cfg and need a server restart to take effect.",
                id="modal-hint-username",
            )
            yield Static("", id="cfg-status")
            yield OptionList(id="cfg-list")
            with Horizontal(id="modal-buttons"):
                yield Button("Edit", variant="primary", id="edit")
                yield Button("Save", variant="success", id="save")
                yield Button("Reload", variant="default", id="reload")
                yield Button("Close", variant="error", id="close")
        yield Footer()

    def on_mount(self) -> None:
        self._set_status("Loading configuration...")
        self.app.run_worker(self._load(), thread=True)

    def _set_status(self, text: str) -> None:
        try:
            self.query_one("#cfg-status", Static).update(text)
        except Exception:
            pass

    def _render_keys(self) -> None:
        lst = self.query_one("#cfg-list", OptionList)
        lst.clear_options()
        for group in self._groups:
            lst.add_option(Option(group["title"], id=f"__header__.{group['id']}"))
            for key_meta in group.get("keys", []):
                section = group["id"]
                key = key_meta["key"]
                label = key_meta.get("label", key)
                base = str(key_meta.get("value", ""))
                current = str(self._edits.get(section, {}).get(key, base))
                flag = " *" if section in self._edits and key in self._edits[section] else ""
                lst.add_option(
                    Option(
                        f"{label} = {current}{flag}",
                        id=f"{section}.{key}",
                    )
                )

    async def _load(self) -> None:
        ok, data = server_ops.fetch_config()
        self.app.call_from_thread(self._apply_data, ok, data)

    def _apply_data(self, ok: bool, data: dict) -> None:
        if not ok:
            self._groups = []
            self._meta = {}
            self._edits = {}
            self._set_status("(could not reach API — is the server running?)")
        else:
            self._groups = data.get("groups") or []
            self._meta = {}
            for group in self._groups:
                for key_meta in group.get("keys", []):
                    self._meta[(group["id"], key_meta["key"])] = key_meta
            path = data.get("config_path", "")
            self._set_status(f"Loaded from {path} — edits marked with *")
        self._render_keys()

    def _selected(self) -> str | None:
        lst = self.query_one("#cfg-list", OptionList)
        index = lst.highlighted
        if index is None:
            return None
        try:
            return lst._options[index].id
        except (IndexError, AttributeError):
            return None

    def _edit_selected(self) -> None:
        option_id = self._selected()
        if not option_id:
            self.app.notify("Select a setting first.", severity="warning")
            return
        if option_id.startswith("__header__."):
            self.app.notify("Select a setting inside the group.", severity="warning")
            return
        section, _, key = option_id.partition(".")
        if not section or not key:
            return
        meta = self._meta.get((section, key), {})
        base = str(meta.get("value", ""))
        current = str(self._edits.get(section, {}).get(key, base))
        title = meta.get("label", key)
        hint = meta.get("hint") or "Enter a new value for this setting."
        if meta.get("type") == "bool":
            hint += " Accepted: true / false."
        elif meta.get("type") == "int":
            hint += " Must be a whole number."
        elif meta.get("type") == "float":
            hint += " Must be a number."
        screen = ValueEditorScreen(
            title=f"Edit {title}  ({meta.get('type', 'str')})",
            hint=hint,
            initial=current,
        )
        self.app.push_screen(screen, lambda value, s=section, k=key: self._on_value(s, k, value))

    def _on_value(self, section: str, key: str, value: str | None) -> None:
        if value is None:
            return
        self._edits.setdefault(section, {})[key] = value
        self._render_keys()
        self._set_status("Unsaved change: *  •  Save to write nebulonmind.cfg")

    def _save(self) -> None:
        if not self._edits:
            self.app.notify("Nothing to save.", severity="warning")
            return
        self._set_status("Saving...")
        self.app.run_worker(self._save_thread(), thread=True)

    async def _save_thread(self) -> None:
        ok, message = server_ops.save_config(self._edits)
        self.app.call_from_thread(self._after_save, ok, message)

    def _after_save(self, ok: bool, message: str) -> None:
        self.app.notify(message, severity="success" if ok else "error")
        if ok:
            self.app.pop_screen()
            self.app.call_after_refresh(self.app.refresh_status)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "edit":
            self._edit_selected()
        elif event.button.id == "save":
            self._save()
        elif event.button.id == "reload":
            self.on_mount()
        elif event.button.id == "close":
            self.action_close()

    def on_option_list_option_selected(self, event) -> None:
        event.stop()
        self._edit_selected()

    def action_close(self) -> None:
        self.app.pop_screen()


class ValueEditorScreen(Screen):
    """Minimal modal that collects a single text value and dismisses with it."""

    BINDINGS = [
        Binding("escape", "cancel", "Back"),
        Binding("up", "focus_previous", "Up"),
        Binding("down", "focus_next", "Down"),
    ]

    def __init__(self, title: str = "Edit value", hint: str = "", initial: str = "") -> None:
        super().__init__()
        self._title_text = title
        self._hint_text = hint
        self._initial = initial

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="modal-container"):
            yield Static(self._title_text, id="modal-title-username")
            yield Static(self._hint_text, id="modal-hint-username")
            yield Input(value=self._initial, placeholder="New value", id="value")
            with Horizontal(id="modal-buttons"):
                yield Button("Save", variant="success", id="save")
                yield Button("Cancel", variant="error", id="cancel")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.action_cancel()
        elif event.button.id == "save":
            self._submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "value":
            self._submit()

    def _submit(self) -> None:
        value = self.query_one("#value", Input).value
        self.dismiss(value)

    def action_cancel(self) -> None:
        self.dismiss(None)
