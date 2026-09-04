"""Closed jurisdiction vocabulary shared by serving contracts."""

from __future__ import annotations

from enum import StrEnum


class Jurisdiction(StrEnum):
    SA = "SA"
    KAWANEEN_DEMO = "KAWANEEN_DEMO"


__all__ = ["Jurisdiction"]
