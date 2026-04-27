"""PII redaction at ingest.

Scope we cover without an external model (presidio is an optional upgrade,
costs 1-2x RAM + boot time which we don't want during CLI ingest):

- CN mobile phone: 1[3-9]\\d{9}  → [PHONE]
- CN national ID card (18-digit incl. X): checksum-optional  → [ID]
- Bank card numbers (13-19 digit runs, Luhn-valid by default) → [BANKCARD]
- Email (any valid-shaped)                                    → [EMAIL]
- IPv4 address                                                 → [IP]

For each scrub we return a ``RedactReport`` so the pipeline can audit how
many redactions a doc triggered. High redaction counts are themselves a
signal (e.g. someone uploaded a user export CSV — maybe block it?).

English SSN / passport / credit cards and other locales are deliberately
not in the regex list; add presidio-analyzer if those matter.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


_MOBILE_CN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_ID_CN = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
_EMAIL = re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")
_IPV4 = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
_BANKCARD_CANDIDATE = re.compile(r"(?<!\d)\d{13,19}(?!\d)")


def _luhn_ok(s: str) -> bool:
    total, parity = 0, len(s) % 2
    for i, c in enumerate(s):
        n = int(c)
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


@dataclass
class RedactReport:
    phone: int = 0
    id_card: int = 0
    bank_card: int = 0
    email: int = 0
    ip: int = 0
    total: int = 0
    samples: list[str] = field(default_factory=list)  # first few original matches for audit

    def bump(self, kind: str, sample: str) -> None:
        setattr(self, kind, getattr(self, kind) + 1)
        self.total += 1
        if len(self.samples) < 5:
            self.samples.append(f"{kind}:{sample[:6]}…")


def redact_pii(text: str) -> tuple[str, RedactReport]:
    """Return (redacted_text, report).

    Order matters slightly: we do phone + ID first (highest-value to mask),
    then email + ip, then the bank-card pass (Luhn-gated to avoid false
    positives on long IDs like barcodes).
    """
    if not text:
        return text, RedactReport()
    report = RedactReport()

    def _sub(pattern: re.Pattern, kind: str, replacement: str) -> callable:
        def repl(m):
            report.bump(kind, m.group(0))
            return replacement
        return repl

    text = _MOBILE_CN.sub(_sub(_MOBILE_CN, "phone", "[PHONE]"), text)
    text = _ID_CN.sub(_sub(_ID_CN, "id_card", "[ID]"), text)
    text = _EMAIL.sub(_sub(_EMAIL, "email", "[EMAIL]"), text)
    text = _IPV4.sub(_sub(_IPV4, "ip", "[IP]"), text)

    # Bank cards last, with Luhn filter
    def _bank_repl(m):
        n = m.group(0)
        if _luhn_ok(n):
            report.bump("bank_card", n)
            return "[BANKCARD]"
        return n
    text = _BANKCARD_CANDIDATE.sub(_bank_repl, text)

    return text, report
