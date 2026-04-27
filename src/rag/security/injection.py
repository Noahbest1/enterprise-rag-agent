"""Simple prompt-injection detector + output filter.

Scope
-----
This is a first-line defence, not a complete security boundary. It covers
the common "copy-paste this jailbreak" cases that show up in support
traffic. Hardened LLM apps should also add:
  * Server-side system-prompt isolation (never let user text become system role)
  * Structured tool-call mediation (we already do: tools return structured
    data, the LLM just narrates)
  * Per-tenant rate limits (Day 7)

What we detect
--------------
- "Ignore previous / all instructions" family (EN + ZH)
- "You are now" / "forget your role" family
- Attempts to exfiltrate the system prompt ("print your system prompt")
- Attempts to force unsafe output ("act as DAN", "jailbreak")
- Obvious secret leakage patterns in assistant output (e.g. API keys)
"""
from __future__ import annotations

import re
from dataclasses import dataclass


_INJECTION_PATTERNS = [
    # English
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(the\s+)?above", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an|the)\b", re.IGNORECASE),
    re.compile(r"forget\s+(your|all)\s+(role|rules|instructions)", re.IGNORECASE),
    re.compile(r"(print|reveal|show\s+me|output)\s+(your|the)\s+(system\s+)?prompt", re.IGNORECASE),
    re.compile(r"\bDAN\b|\bjailbreak\b", re.IGNORECASE),
    re.compile(r"repeat\s+the\s+words\s+above", re.IGNORECASE),
    # Chinese -- allow a handful of filler chars between the trigger and the noun
    # so "忽略之前的所有指令" / "展示你的系统 prompt" both match.
    re.compile(r"忽略[^\n]{0,10}?(指令|规则|要求)"),
    re.compile(r"(不要|无视)[^\n]{0,10}?(指令|规则|要求)"),
    re.compile(r"(忘记|抛开)(你的|所有)?(身份|角色|规则)"),
    re.compile(r"(展示|打印|输出|告诉我)[^\n]{0,20}?(系统\s*)?(提示\s*词?|prompt)", re.IGNORECASE),
    re.compile(r"(现在|从现在起)你是一个?(?!客服|助手)"),
]


# Patterns that commonly leak from a successful injection
_OUTPUT_LEAK_PATTERNS = [
    re.compile(r"(sk|pk)-[A-Za-z0-9_-]{16,}"),
    re.compile(r"api[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9_-]{16,}", re.IGNORECASE),
    # Accept "BEGIN RSA PRIVATE KEY", "BEGIN OPENSSH PRIVATE KEY",
    # "BEGIN DSA PRIVATE KEY", or bare "BEGIN PRIVATE KEY".
    re.compile(r"BEGIN\s+(?:RSA\s+|DSA\s+|EC\s+|OPENSSH\s+)?PRIVATE\s+KEY"),
    re.compile(r"system\s+prompt:?", re.IGNORECASE),
]


@dataclass
class InjectionVerdict:
    is_injection: bool
    rule: str | None
    matched: str | None

    @property
    def blocked(self) -> bool:
        return self.is_injection


def detect_injection(text: str) -> InjectionVerdict:
    if not text:
        return InjectionVerdict(False, None, None)
    for p in _INJECTION_PATTERNS:
        m = p.search(text)
        if m:
            return InjectionVerdict(True, p.pattern, m.group(0))
    return InjectionVerdict(False, None, None)


def sanitize_output(text: str) -> tuple[str, list[str]]:
    """Replace suspected secret leaks with [REDACTED]. Returns (safe_text, hits_found)."""
    if not text:
        return text, []
    hits = []
    sanitized = text
    for p in _OUTPUT_LEAK_PATTERNS:
        def _repl(m):
            hits.append(m.group(0)[:80])
            return "[REDACTED]"
        sanitized = p.sub(_repl, sanitized)
    return sanitized, hits
