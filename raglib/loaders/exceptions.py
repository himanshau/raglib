"""Custom exceptions used by raglib document loaders."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class InstructionError(RuntimeError):
    """Represents a user-actionable loader error with remediation guidance."""
