"""Generate a real, downloadable PDF invoice.

Uses ``reportlab`` (pure Python, no OS deps) so the same build works on
macOS/Linux/CI without hunting for cairo / pango. The layout is deliberately
simple — title / buyer / seller / order summary / line-item table / amount
total / footer — close enough to a Chinese 电子发票 receipt to be
demo-convincing, without pretending to be a legally valid tax invoice.

Usage:
    from rag_api.invoice_pdf import render_invoice_pdf
    pdf_bytes = render_invoice_pdf(invoice_dict, order_dict)
"""
from __future__ import annotations

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# --- Chinese font registration --------------------------------------------

def _register_cjk_font() -> str:
    """Register a CJK-capable font. Returns the font name to use.

    Tries system fonts in order; falls back to reportlab's built-in CID font
    (``STSong-Light``) which requires no external file.
    """
    candidates = [
        ("/System/Library/Fonts/PingFang.ttc", "PingFang"),          # macOS
        ("/System/Library/Fonts/STHeiti Medium.ttc", "STHeiti"),     # macOS fallback
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "NotoSansCJK"),  # Linux
    ]
    for path, name in candidates:
        try:
            pdfmetrics.registerFont(TTFont(name, path))
            return name
        except Exception:
            continue
    # Last resort: CID builtin (no file needed, ships with reportlab).
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        return "STSong-Light"
    except Exception:
        return "Helvetica"  # ASCII only; will mojibake CJK but won't crash


_CJK_FONT = _register_cjk_font()


def _yuan(cents: int | None) -> str:
    if cents is None:
        return "¥0.00"
    return f"¥{cents / 100:,.2f}"


def _style(name: str, *, size: int, leading: int | None = None, bold: bool = False,
           align: int = 0, colour=colors.black) -> ParagraphStyle:
    base = getSampleStyleSheet()["Normal"]
    return ParagraphStyle(
        name=name,
        parent=base,
        fontName=_CJK_FONT,
        fontSize=size,
        leading=leading or (size + 4),
        textColor=colour,
        alignment=align,
        spaceAfter=4,
        fontWeight="bold" if bold else "normal",
    )


def render_invoice_pdf(invoice: dict, order: dict | None = None) -> bytes:
    """Produce a PDF as bytes.

    ``invoice`` keys expected: id, order_id, tenant, title, tax_id,
    invoice_type, amount_yuan OR amount_cents, status, requested_at, issued_at.
    ``order`` (optional): id, items list, total_cents.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=18 * mm, bottomMargin=15 * mm,
        leftMargin=18 * mm, rightMargin=18 * mm,
        title=f"Invoice-{invoice.get('id','?')}",
    )

    h1 = _style("h1", size=20, bold=True, align=1)  # centre
    h2 = _style("h2", size=12, bold=True)
    normal = _style("n", size=10)
    small = _style("small", size=9, colour=colors.grey)
    money = _style("money", size=16, bold=True, align=2, colour=colors.HexColor("#059669"))

    story: list = []

    tenant_label = {"jd": "京东商城", "taobao": "淘宝 / 天猫"}.get(invoice.get("tenant"), invoice.get("tenant", ""))
    story.append(Paragraph(f"{tenant_label}  电子发票 (Demo)", h1))
    story.append(Spacer(1, 4 * mm))

    # Status line
    status_colour = {
        "issued": "#059669", "requested": "#d97706", "cancelled": "#64748b",
    }.get(invoice.get("status", ""), "#555555")
    status_label = {
        "issued": "已开具", "requested": "处理中", "cancelled": "已取消",
    }.get(invoice.get("status", ""), invoice.get("status", "—"))
    story.append(Paragraph(
        f"<font color='{status_colour}'><b>状态:{status_label}</b></font> · "
        f"发票号 INV-{invoice.get('id','—')} · 类型:"
        f"{'电子发票' if invoice.get('invoice_type') == 'electronic' else '纸质发票'}",
        normal,
    ))
    story.append(Spacer(1, 6 * mm))

    # Header meta: buyer / order
    amount_yuan = invoice.get("amount_yuan")
    if amount_yuan is None and invoice.get("amount_cents") is not None:
        amount_yuan = invoice["amount_cents"] / 100.0

    header_rows = [
        ["购买方抬头", invoice.get("title") or "—"],
        ["纳税人识别号", invoice.get("tax_id") or "(个人免填)"],
        ["关联订单", invoice.get("order_id") or "—"],
        ["开票日期", _fmt_date(invoice.get("issued_at") or invoice.get("requested_at"))],
    ]
    t = Table(header_rows, colWidths=[32 * mm, 130 * mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), _CJK_FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8fafc")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#475569")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 6 * mm))

    # Items
    if order and order.get("items"):
        story.append(Paragraph("商品明细", h2))
        item_rows = [["商品", "数量", "单价", "小计"]]
        for it in order["items"]:
            unit_y = it.get("unit_price_yuan", 0) or 0
            qty = it.get("qty", 1)
            item_rows.append([
                it.get("title") or it.get("sku") or "?",
                str(qty),
                f"¥{unit_y:,.2f}",
                f"¥{unit_y * qty:,.2f}",
            ])
        it_tbl = Table(item_rows, colWidths=[90 * mm, 20 * mm, 26 * mm, 26 * mm])
        it_tbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), _CJK_FONT),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0a84ff")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(it_tbl)
        story.append(Spacer(1, 4 * mm))

    story.append(Paragraph(f"合计金额:{_yuan(int((amount_yuan or 0) * 100))}", money))
    story.append(Spacer(1, 10 * mm))

    story.append(Paragraph(
        "此为 demo 电子发票,仅用于系统展示,不具备任何税务效力。<br/>"
        "在真实生产环境中,电子发票由税控服务器签章下发;本服务仅负责用户流程。",
        small,
    ))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        f"生成时间:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        small,
    ))

    doc.build(story)
    return buf.getvalue()


def _fmt_date(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        # Handle both tz-aware and naive ISO strings.
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso[:19]
