"""Token budgets.

Two enforcement points:

1. **Per-request prompt budget** -- before the LLM call, we estimate the
   prompt token count and reject if it exceeds
   ``settings.max_prompt_tokens_per_request``. Cheap guard against runaway
   prompts (e.g. a bad chunker dumping 100k tokens of context into one call).

2. **Per-tenant token budget** -- a sliding-window tally of tokens *actually
   consumed* (prompt + completion) by a tenant. Cheap guard against a
   runaway client burning $X/minute. Window is 60s by default; budget is
   configurable via settings (default: 1e6 tokens/minute = generous).

Token estimation is char-based: ``len(text) / 3`` for a mix of zh+en; this
is within 10-20% of tiktoken for our content and doesn't drag in another
dep. If precision matters later, swap in a real tokenizer.

Note:
    Per-request budget kicks in BEFORE the HTTP call to the LLM -- it
    protects against wasted API calls. Per-tenant budget is checked BEFORE
    the HTTP call too (so one bad request doesn't push a tenant miles over
    budget) and is UPDATED with the ACTUAL completion tokens from the
    response (so the tally is accurate after the fact).
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from .config import settings


def estimate_tokens(text: str) -> int:
    """Cheap char-based estimate. ~3 chars/token for mixed zh+en."""
    if not text:
        return 0
    return max(1, len(text) // 3)


class BudgetExceeded(RuntimeError):
    """Raised when a request would exceed either the per-request or per-tenant budget."""

    def __init__(self, scope: str, used: int, limit: int):
        super().__init__(f"{scope} budget exceeded: used={used} limit={limit}")
        self.scope = scope
        self.used = used
        self.limit = limit


@dataclass
class _TenantWindow:
    """Sliding window over (timestamp, tokens) for a single tenant."""
    samples: deque[tuple[float, int]] = field(default_factory=deque)

    def _trim(self, now: float, window_s: int) -> None:
        cutoff = now - window_s
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def total(self, now: float, window_s: int) -> int:
        self._trim(now, window_s)
        return sum(t for _, t in self.samples)

    def add(self, now: float, tokens: int) -> None:
        self.samples.append((now, tokens))


class TenantBudgetTracker:
    """Process-local per-tenant sliding budget.

    Like rate_limit._SlidingWindow but tracks *tokens* instead of call count.
    Swap for Redis when scaling horizontally.
    """

    def __init__(self):
        self._by_tenant: dict[str, _TenantWindow] = defaultdict(_TenantWindow)

    def check(self, tenant_id: str, would_add: int, limit: int, window_s: int,
              now: float | None = None) -> None:
        """Raise BudgetExceeded if (current + would_add) > limit."""
        if limit <= 0:
            return  # 0 disables
        now = now if now is not None else time.monotonic()
        total = self._by_tenant[tenant_id].total(now, window_s)
        if total + would_add > limit:
            raise BudgetExceeded(scope=f"tenant:{tenant_id}", used=total, limit=limit)

    def record(self, tenant_id: str, tokens: int, now: float | None = None) -> None:
        if tokens <= 0:
            return
        now = now if now is not None else time.monotonic()
        self._by_tenant[tenant_id].add(now, tokens)

    def reset(self) -> None:
        self._by_tenant.clear()


# Singleton tenant tracker. Single-process only.
_TENANT_TRACKER = TenantBudgetTracker()


def tenant_tracker() -> TenantBudgetTracker:
    return _TENANT_TRACKER


def check_prompt_budget(prompt_text: str) -> int:
    """Raise BudgetExceeded if the prompt exceeds the per-request cap. Returns the estimate."""
    limit = settings.max_prompt_tokens_per_request
    est = estimate_tokens(prompt_text)
    if limit > 0 and est > limit:
        raise BudgetExceeded(scope="prompt", used=est, limit=limit)
    return est


def reset_for_tests() -> None:
    _TENANT_TRACKER.reset()
