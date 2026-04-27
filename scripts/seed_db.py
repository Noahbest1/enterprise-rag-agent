"""Seed mock users, orders, and a few Taobao/JD items into the DB.

Idempotent: running twice does not duplicate rows. Existing IDs are
refreshed (UPSERT-style via merge). Safe to run in dev and CI.

Usage:
    ./.venv/bin/python scripts/seed_db.py
    DATABASE_URL=postgresql://... ./.venv/bin/python scripts/seed_db.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.db.base import SessionLocal
from rag.db.models import Invoice, Order, OrderItem, User, UserAddress


JD_USER = "jd-demo-user"
TAOBAO_USER = "tb-demo-user"


def _yuan(y: int | float) -> int:
    return int(round(float(y) * 100))


USERS = [
    {"id": JD_USER, "tenant": "jd", "display_name": "demo-user (JD)", "phone": "13800001111"},
    {"id": TAOBAO_USER, "tenant": "taobao", "display_name": "demo-user (Taobao)", "phone": "13900002222"},
]


def _now():
    return datetime.now(timezone.utc)


ORDERS = [
    # ---- JD ----
    {
        "id": "JD20260418123",
        "user_id": JD_USER,
        "tenant": "jd",
        "status": "shipped",
        "tracking_no": "JDSF1234567890",
        "carrier": "京东快递",
        "placed_at": _now() - timedelta(days=5),
        "items": [
            {"sku": "100055667788", "title": "Apple MacBook Pro 14 M4 Pro 1TB 深空黑", "qty": 1, "unit_price": 18999},
        ],
    },
    {
        "id": "JD20260420456",
        "user_id": JD_USER,
        "tenant": "jd",
        "status": "delivered",
        "tracking_no": "JDSF9876543210",
        "carrier": "京东快递",
        "placed_at": _now() - timedelta(days=3),
        "items": [
            {"sku": "100012345671", "title": "iPhone 16 Pro 256GB 黑色钛金属", "qty": 1, "unit_price": 8999},
            {"sku": "100022334455", "title": "AirPods Pro 2 USB-C", "qty": 1, "unit_price": 1699},
        ],
    },
    {
        "id": "JD20260422789",
        "user_id": JD_USER,
        "tenant": "jd",
        "status": "placed",
        "tracking_no": None,
        "carrier": None,
        "placed_at": _now() - timedelta(hours=6),
        "items": [
            {"sku": "100077889900", "title": "美的空调 3 匹变频柜机", "qty": 1, "unit_price": 5999},
        ],
    },
    # ---- Taobao / Tmall ----
    {
        "id": "TB20260419001",
        "user_id": TAOBAO_USER,
        "tenant": "taobao",
        "status": "shipped",
        "tracking_no": "SF7788990011",
        "carrier": "顺丰速运",
        "placed_at": _now() - timedelta(days=4),
        "items": [
            {"sku": "TM_LANCOME_44556", "title": "兰蔻小黑瓶精华 50ml", "qty": 1, "unit_price": 1180},
        ],
    },
    {
        "id": "TB20260421002",
        "user_id": TAOBAO_USER,
        "tenant": "taobao",
        "status": "refunded",
        "tracking_no": "YTO44556677",
        "carrier": "圆通快递",
        "placed_at": _now() - timedelta(days=10),
        "items": [
            {"sku": "TM_UNIQ_00012", "title": "UNIQLO 轻型羽绒服(男)黑色 L 码", "qty": 1, "unit_price": 599},
        ],
    },
    {
        "id": "TB20260422003",
        "user_id": TAOBAO_USER,
        "tenant": "taobao",
        "status": "delivered",
        "tracking_no": "ZT11223344",
        "carrier": "中通快递",
        "placed_at": _now() - timedelta(days=2),
        "items": [
            {"sku": "TM_SSSS_11223", "title": "三只松鼠 每日坚果混合装 30 包", "qty": 2, "unit_price": 89},
            {"sku": "TM_NFSQ_33445", "title": "农夫山泉天然饮用水 550ml × 24 瓶", "qty": 1, "unit_price": 35},
        ],
    },
]


ADDRESSES = [
    # ---- JD user ----
    {
        "user_id": JD_USER, "label": "家",
        "recipient": "张小明", "phone": "13800001111",
        "province": "北京市", "city": "北京市", "district": "朝阳区",
        "line1": "朝阳门外大街 1 号 5 栋 302 室",
        "is_default": True,
    },
    {
        "user_id": JD_USER, "label": "公司",
        "recipient": "张小明", "phone": "13800001111",
        "province": "北京市", "city": "北京市", "district": "海淀区",
        "line1": "中关村大街 27 号 12 层 A 区",
        "is_default": False,
    },
    # ---- Taobao user ----
    {
        "user_id": TAOBAO_USER, "label": "家",
        "recipient": "李小红", "phone": "13900002222",
        "province": "上海市", "city": "上海市", "district": "浦东新区",
        "line1": "世纪大道 100 号 8 楼 805",
        "is_default": True,
    },
    {
        "user_id": TAOBAO_USER, "label": "父母家",
        "recipient": "李妈妈", "phone": "13900002223",
        "province": "江苏省", "city": "苏州市", "district": "姑苏区",
        "line1": "干将东路 200 号 3 幢 12-1",
        "is_default": False,
    },
]


INVOICES = [
    # JD20260420456 already issued (electronic)
    {
        "order_id": "JD20260420456",
        "tenant": "jd",
        "title": "个人",
        "tax_id": None,
        "invoice_type": "electronic",
        "amount_yuan": 10698.00,
        "status": "issued",
        # None => the runtime endpoint /invoice/{id}.pdf takes over (real PDF).
        "download_url": None,
        "issued_at_offset_days": -2,
    },
    # TB20260419001 requested but not yet issued
    {
        "order_id": "TB20260419001",
        "tenant": "taobao",
        "title": "上海某某科技有限公司",
        "tax_id": "91310000XYZ123456A",
        "invoice_type": "electronic",
        "amount_yuan": 1180.00,
        "status": "requested",
        "download_url": None,
        "issued_at_offset_days": None,
    },
]


def seed() -> dict:
    assert SessionLocal is not None
    inserted = {"users": 0, "orders": 0, "items": 0, "invoices": 0, "addresses": 0}
    with SessionLocal() as session:
        for u in USERS:
            existing = session.get(User, u["id"])
            if existing:
                existing.tenant = u["tenant"]
                existing.display_name = u["display_name"]
                existing.phone = u["phone"]
            else:
                session.add(User(**u))
                inserted["users"] += 1

        for raw in ORDERS:
            # copy so repeated seed() calls don't mutate the module-level list
            o = dict(raw)
            items = list(o.pop("items"))
            total = sum(_yuan(it["unit_price"]) * it["qty"] for it in items)
            existing = session.get(Order, o["id"])
            if existing:
                existing.status = o["status"]
                existing.tracking_no = o["tracking_no"]
                existing.carrier = o["carrier"]
                existing.total_cents = total
                for prev in list(existing.items):
                    session.delete(prev)
                for it in items:
                    existing.items.append(
                        OrderItem(
                            sku=it["sku"],
                            title=it["title"],
                            qty=it["qty"],
                            unit_price_cents=_yuan(it["unit_price"]),
                        )
                    )
            else:
                order = Order(total_cents=total, **o)
                for it in items:
                    order.items.append(
                        OrderItem(
                            sku=it["sku"],
                            title=it["title"],
                            qty=it["qty"],
                            unit_price_cents=_yuan(it["unit_price"]),
                        )
                    )
                session.add(order)
                inserted["orders"] += 1
                inserted["items"] += len(items)

        # The session has autoflush=False, so Orders/Users added above are
        # still pending. Dependent inserts (Invoice FK -> orders, UserAddress
        # FK -> users) won't find their parents until we flush.
        session.flush()

        # User addresses -- idempotent per (user_id, line1).
        for raw in ADDRESSES:
            addr = dict(raw)
            existing = session.query(UserAddress).filter(
                UserAddress.user_id == addr["user_id"],
                UserAddress.line1 == addr["line1"],
            ).first()
            if existing is not None:
                continue
            session.add(UserAddress(**addr))
            inserted["addresses"] += 1

        # Invoices -- idempotent by (order_id, status): skip if an invoice
        # for this order already exists in any state.
        for raw in INVOICES:
            inv = dict(raw)
            oid = inv["order_id"]
            existing_inv = session.query(Invoice).filter(Invoice.order_id == oid).first()
            if existing_inv is not None:
                continue
            offset = inv.pop("issued_at_offset_days")
            amount_y = inv.pop("amount_yuan")
            inv["amount_cents"] = _yuan(amount_y)
            inv["issued_at"] = _now() + timedelta(days=offset) if offset is not None else None
            session.add(Invoice(**inv))
            inserted["invoices"] += 1

        session.commit()
    return inserted


if __name__ == "__main__":
    out = seed()
    print(
        f"[seed] users inserted={out['users']} orders={out['orders']} "
        f"items={out['items']} invoices={out['invoices']} addresses={out['addresses']}"
    )
