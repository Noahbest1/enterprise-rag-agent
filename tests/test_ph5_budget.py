"""PH5.5 token budget tests.

Covers:
- estimate_tokens: 0 for empty, scales roughly with length
- check_prompt_budget: raises when prompt exceeds per-request cap
- TenantBudgetTracker.check: raises when would-be add pushes over limit
- TenantBudgetTracker.record: actual tokens accumulate in-window
- Sliding window: old samples expire
- settings limit=0 disables
"""
from __future__ import annotations

import pytest

from rag.budget import (
    BudgetExceeded,
    TenantBudgetTracker,
    check_prompt_budget,
    estimate_tokens,
    reset_for_tests,
    tenant_tracker,
)
from rag.config import settings


@pytest.fixture(autouse=True)
def _wipe():
    reset_for_tests()
    yield
    reset_for_tests()


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_estimate_tokens_scales():
    short = estimate_tokens("hello")
    longer = estimate_tokens("hello world " * 100)
    assert longer > short
    assert longer >= 400  # 12 chars * 100 / 3 = 400


def test_check_prompt_budget_under_limit():
    # Default 8192 -> "short prompt" passes.
    est = check_prompt_budget("short prompt")
    assert est > 0


def test_check_prompt_budget_over_limit(monkeypatch):
    monkeypatch.setattr(settings, "max_prompt_tokens_per_request", 10)
    with pytest.raises(BudgetExceeded) as excinfo:
        check_prompt_budget("a" * 100)  # ~33 tokens
    assert excinfo.value.scope == "prompt"
    assert excinfo.value.limit == 10


def test_check_prompt_budget_zero_disables(monkeypatch):
    monkeypatch.setattr(settings, "max_prompt_tokens_per_request", 0)
    # Even a huge prompt should not raise when limit=0.
    check_prompt_budget("a" * 100_000)


def test_tenant_tracker_check_passes_under_limit():
    t = TenantBudgetTracker()
    t.check(tenant_id="acme", would_add=100, limit=1000, window_s=60, now=0.0)


def test_tenant_tracker_check_raises_when_would_exceed():
    t = TenantBudgetTracker()
    t.record("acme", tokens=900, now=0.0)
    with pytest.raises(BudgetExceeded) as excinfo:
        t.check(tenant_id="acme", would_add=200, limit=1000, window_s=60, now=1.0)
    assert excinfo.value.scope == "tenant:acme"


def test_tenant_tracker_window_expires():
    t = TenantBudgetTracker()
    t.record("acme", tokens=900, now=0.0)
    # 61s later, the old sample expired; full budget available.
    t.check(tenant_id="acme", would_add=900, limit=1000, window_s=60, now=61.0)


def test_tenant_tracker_limit_zero_disables():
    t = TenantBudgetTracker()
    # Should not raise despite monstrous would_add.
    t.check(tenant_id="acme", would_add=10**9, limit=0, window_s=60)


def test_tenant_tracker_isolation():
    t = TenantBudgetTracker()
    t.record("a", tokens=900, now=0.0)
    # Tenant b's window is independent.
    t.check(tenant_id="b", would_add=900, limit=1000, window_s=60, now=0.0)


def test_global_tracker_singleton():
    a = tenant_tracker()
    b = tenant_tracker()
    assert a is b
