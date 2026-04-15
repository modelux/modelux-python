from __future__ import annotations

from typing import Optional

from ._types import BudgetInfo


class ModeluxError(Exception):
    """Base error for Modelux SDK."""

    def __init__(self, message: str, status: int, code: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


class BudgetExceededError(ModeluxError):
    """Raised when a request is blocked by budget enforcement (HTTP 402)."""

    def __init__(
        self,
        message: str,
        budget: BudgetInfo,
        retry_after: Optional[int] = None,
    ) -> None:
        super().__init__(message, 402, "budget_exceeded")
        self.budget = budget
        self.retry_after = retry_after
