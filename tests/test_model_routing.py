"""Model routing picks the right tier per task."""
from __future__ import annotations

from rag import model_routing
from rag.config import settings


def test_routing_enabled_by_default_post_sprint_a():
    # Sprint A flipped the default to True. Verify generate still stays on
    # the quality tier (qwen-plus) -- only simple tasks go turbo.
    assert settings.enable_model_routing is True
    assert model_routing.resolve_model("generate") == settings.qwen_model


def test_routing_can_be_disabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_model_routing", False, raising=False)
    assert model_routing.resolve_model("generate") == settings.qwen_model
    assert model_routing.resolve_model("rewrite") == settings.qwen_model


def test_routing_enabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_model_routing", True, raising=False)
    monkeypatch.setattr(settings, "qwen_model", "qwen-plus", raising=False)
    monkeypatch.setattr(settings, "qwen_turbo_model", "qwen-turbo", raising=False)

    # simple tasks → turbo
    assert model_routing.resolve_model("rewrite") == "qwen-turbo"
    assert model_routing.resolve_model("intent") == "qwen-turbo"
    assert model_routing.resolve_model("contextual") == "qwen-turbo"
    assert model_routing.resolve_model("judge") == "qwen-turbo"

    # quality-critical → plus
    assert model_routing.resolve_model("generate") == "qwen-plus"
    assert model_routing.resolve_model("default") == "qwen-plus"


def test_routing_unknown_task_defaults_to_plus(monkeypatch):
    monkeypatch.setattr(settings, "enable_model_routing", True, raising=False)
    monkeypatch.setattr(settings, "qwen_model", "qwen-plus", raising=False)
    assert model_routing.resolve_model("mystery") == "qwen-plus"  # type: ignore[arg-type]
