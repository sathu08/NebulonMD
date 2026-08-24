"""Shared utilities: constants, environment readers, time and text helpers.

Owned by no single module so duplicated ``def``/constants live in exactly
one place (previously re-declared across ``core``, ``lifecycle``,
``intelligence``, ``agent`` and ``agents``). Import submodules explicitly,
e.g. ``from nmd_host.utils.constants import SECONDS_PER_DAY``.
"""

from __future__ import annotations
