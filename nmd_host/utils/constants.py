"""Shared constants.

Central home for the numeric/string constants several modules rely on, so a
magic value lives in exactly one place instead of being re-declared in each
file. Module-local domain tables (rule lists, prompt policies, category
maps, entity whitelists) deliberately stay in their owning modules for
cohesion — they are data specific to one subsystem, not shared defaults.
"""

from __future__ import annotations

import pyfiglet

# Time -----------------------------------------------------------------
SECONDS_PER_DAY = 86400.0
DEFAULT_TEMPORARY_TTL_SECONDS = 30 * 24 * 3600
GRACEFUL_SHUTDOWN_SECONDS_DEFAULT = 30

# Similarity / deduplication -------------------------------------------
SIZE_RATIO_FLOOR = 0.70

# Agent -----------------------------------------------------------------
MAX_TURN_RESULT_CHARS = 2000

# Background job schedules (5-field cron) --------------------------------
AUTO_DELETE_CRON_DEFAULT = "0 3 * * *"  # daily 03:00 — expired-memory sweep
MEMORY_CONSOLIDATION_CRON_DEFAULT = "0 2 * * *"  # nightly 02:00
WEEKLY_SUMMARY_CRON_DEFAULT = "0 9 * * 0"  # Sundays 09:00

# API / server ----------------------------------------------------------
API_HOST_DEFAULT = "0.0.0.0"
API_PORT_DEFAULT = 9696
NEBULONDB_API_HOST_DEFAULT = "localhost"
NEBULONDB_API_PORT_DEFAULT = 6969

# Context ----------------------------------------------------------------
CONTEXT_MAX_ITEMS_DEFAULT = 10
CONTEXT_MAX_CHARACTERS_DEFAULT = 6000

# Branding / TUI content -------------------------------------------------
APP_NAME = "NebulonMind"
BRAND_BANNER_TEXT = "NEBULONMIND"
BANNER_FONT = "smslant"
NEBULONMIND_BANNER = pyfiglet.figlet_format(BRAND_BANNER_TEXT, font=BANNER_FONT)
BRAND_TAGLINE = "Memory layer for AI assistants on NebulonDB"
DEFAULT_USERNAME = "nmd_user_01"

# Server lifecycle messages ---------------------------------------------
SERVER_GETTING_READY_MESSAGE = "Getting the server ready for you..."

TUI_USAGE = (
    "Usage: nebulonmind {start|stop|restart} [--foreground|-f] [--force|-F]\n"
    "       nebulonmind                          # launch the interactive chat TUI"
)

__all__ = [
    "API_HOST_DEFAULT",
    "API_PORT_DEFAULT",
    "APP_NAME",
    "AUTO_DELETE_CRON_DEFAULT",
    "BRAND_BANNER_TEXT",
    "BRAND_TAGLINE",
    "BANNER_FONT",
    "CONTEXT_MAX_CHARACTERS_DEFAULT",
    "CONTEXT_MAX_ITEMS_DEFAULT",
    "DEFAULT_TEMPORARY_TTL_SECONDS",
    "DEFAULT_USERNAME",
    "GRACEFUL_SHUTDOWN_SECONDS_DEFAULT",
    "MAX_TURN_RESULT_CHARS",
    "MEMORY_CONSOLIDATION_CRON_DEFAULT",
    "NEBULONDB_API_HOST_DEFAULT",
    "NEBULONDB_API_PORT_DEFAULT",
    "NEBULONMIND_BANNER",
    "SECONDS_PER_DAY",
    "SERVER_GETTING_READY_MESSAGE",
    "SIZE_RATIO_FLOOR",
    "TUI_USAGE",
    "WEEKLY_SUMMARY_CRON_DEFAULT",
]
