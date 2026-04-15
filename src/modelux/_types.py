from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ModeluxMetadata:
    """Metadata extracted from X-Modelux-* response headers."""

    request_id: str = ""
    provider_used: str = ""
    model_used: str = ""
    cache_hit: bool = False
    cache_similarity: Optional[float] = None
    ab_variant: Optional[str] = None
    budget_name: Optional[str] = None
    budget_remaining: Optional[float] = None
    budget_action: Optional[Literal["block", "downgrade", "warn_only"]] = None
    budget_reset: Optional[str] = None


@dataclass
class BudgetInfo:
    """Budget details returned in a 402 error response."""

    name: str
    spend_usd: float
    cap_usd: float
    period: Literal["daily", "weekly", "monthly"]
    period_resets_at: str
