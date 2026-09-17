# nmd_host/tui/options.py
"""Slash-command registry for the NebulonMind chat TUI.

Every ``/command`` the chat input understands is declared here as a single
:class:`CommandOption`, grouped into categories. The TUI builds its command
palette (the dropdown that appears the moment you type ``/``) straight from
this registry, so adding a command never touches any widget code -- declare
it here, add a handler to the app, and it shows up everywhere (palette and
the ``/help`` page).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandOption:
    """A slash command rendered in the TUI command palette.

    Attributes:
        command: The canonical command as typed, e.g. ``"/create"``.
        description: One-line help shown next to the command.
        category: Section header this command appears under in the palette.
        aliases: Accepted shortcut spellings, e.g. ``"/cr"``.
        args_hint: Template of the expected argument, e.g. ``"<username>"``.
    """

    command: str
    description: str
    category: str = "General"
    aliases: tuple[str, ...] = ()
    args_hint: str = ""

    @property
    def names(self) -> tuple[str, ...]:
        """Every accepted invocation, canonical spelling first."""
        return (self.command, *self.aliases)

    @property
    def help(self) -> str:
        """``/create <username>`` style signature."""
        return f"{self.command} {self.args_hint}".rstrip()

    def matches(self, query: str) -> bool:
        """True when ``query`` prefixes a name or appears in the description."""
        query = (query or "").strip().lower()
        if not query:
            return True
        if query in self.description.lower():
            return True
        return any(name.lstrip("/").startswith(query) for name in self.names)


# ---------------------------------------------------------------------- #
# Command registry                                                       #
# ---------------------------------------------------------------------- #

COMMAND_OPTIONS: tuple[CommandOption, ...] = (
    # ---- Chat ------------------------------------------------------------
    CommandOption("/clear", "Clear the chat transcript", category="Chat"),
    CommandOption("/whoami", "Show the active NebulonMind user", category="Chat"),
    CommandOption(
        "/help",
        "List every available / command",
        category="Chat",
        aliases=("/h", "/?"),
    ),
    # ---- Users -----------------------------------------------------------
    CommandOption(
        "/create",
        "Register a NEW username",
        category="Users",
        aliases=("/cr",),
        args_hint="<username>",
    ),
    CommandOption(
        "/setup",
        "Switch to an existing username",
        category="Users",
        aliases=("/set", "/se"),
        args_hint="[username]",
    ),
    # ---- Server ----------------------------------------------------------
    CommandOption("/start", "Start the NebulonMind API", category="Server"),
    CommandOption(
        "/stop", "Stop the NebulonMind API", category="Server", aliases=("/st",)
    ),
    CommandOption(
        "/restart",
        "Restart the NebulonMind API",
        category="Server",
        aliases=("/rs",),
    ),
    CommandOption(
        "/status", "Show API, backend and user status", category="Server"
    ),
    # ---- Config ----------------------------------------------------------
    CommandOption(
        "/settings", "Edit nebulonmd.cfg settings", category="Config"
    ),
    CommandOption(
        "/credentials",
        "Set NebulonDB backend credentials (.env)",
        category="Config",
        aliases=("/creds", "/env"),
    ),
)


# ---------------------------------------------------------------------- #
# Registry helpers                                                       #
# ---------------------------------------------------------------------- #

def ordered_categories() -> list[str]:
    """Category names in first-appearance order."""
    seen: list[str] = []
    for option in COMMAND_OPTIONS:
        if option.category not in seen:
            seen.append(option.category)
    return seen


def grouped_options() -> list[tuple[str, list[CommandOption]]]:
    """``[(category, [options, ...]), ...]`` preserving declaration order."""
    groups: dict[str, list[CommandOption]] = {}
    for option in COMMAND_OPTIONS:
        groups.setdefault(option.category, []).append(option)
    return [(category, groups[category]) for category in ordered_categories()]


def categories_for_query(query: str) -> list[tuple[str, list[CommandOption]]]:
    """Grouped options filtered by ``query`` (empty query returns everything)."""
    sections = []
    for category, options in grouped_options():
        matches = [option for option in options if option.matches(query)]
        if matches:
            sections.append((category, matches))
    return sections


def find(query: str) -> list[CommandOption]:
    """Every option matching ``query`` (used for palette completions)."""
    return [option for option in COMMAND_OPTIONS if option.matches(query)]


def resolve(text: str) -> tuple[CommandOption | None, str, str]:
    """Resolve typed text to ``(option, args, unmatched_text)``.

    ``"/cr nmd_user_01"`` -> ``(/create option, "nmd_user_01", "")``.
    A normal chat message returns ``(None, "", text)``.
    """
    text = (text or "").strip()
    head, _, rest = text.partition(" ")
    if not head.startswith("/"):
        return None, "", text
    token = head.lower()
    for option in COMMAND_OPTIONS:
        if any(name.lower() == token for name in option.names):
            return option, rest.strip(), ""
    return None, "", text


__all__ = [
    "CommandOption",
    "COMMAND_OPTIONS",
    "ordered_categories",
    "grouped_options",
    "categories_for_query",
    "find",
    "resolve",
]