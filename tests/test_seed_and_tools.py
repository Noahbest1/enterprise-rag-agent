"""Seed data + agent DB-backed tools."""
from __future__ import annotations

import pytest

from agent.tools.orders import (
    find_recent_order_with_keyword,
    get_order,
    list_user_orders,
    track_package,
)


@pytest.fixture()
def seeded(seeded_db):
    return seeded_db


def test_seed_creates_expected_counts(seeded):
    from rag.db.models import Order, User
    assert seeded.query(User).count() == 2
    assert seeded.query(Order).count() == 6
    assert seeded.query(Order).filter_by(tenant="jd").count() == 3
    assert seeded.query(Order).filter_by(tenant="taobao").count() == 3


def test_get_order_known_id(seeded):
    o = get_order("JD20260420456")
    assert o is not None
    assert o["tenant"] == "jd"
    assert o["status"] == "delivered"
    titles = [it["title"] for it in o["items"]]
    assert any("iPhone 16 Pro" in t for t in titles)


def test_get_order_unknown(seeded):
    assert get_order("NOPE") is None


def test_list_user_orders_sorted_desc(seeded):
    rows = list_user_orders("jd-demo-user", "jd", limit=10)
    assert len(rows) == 3
    # placed_at should be non-increasing
    ts = [r["placed_at"] for r in rows]
    assert ts == sorted(ts, reverse=True)


def test_find_recent_by_keyword(seeded):
    hit = find_recent_order_with_keyword("jd-demo-user", "jd", "MacBook")
    assert hit is not None
    assert "MacBook" in hit["items"][0]["title"]


def test_track_package_known(seeded):
    info = track_package("SF7788990011")
    assert info is not None
    assert info["status"] == "shipped"
    assert info["timeline"], "timeline should not be empty"


def test_track_package_unknown(seeded):
    assert track_package("DOES-NOT-EXIST") is None
