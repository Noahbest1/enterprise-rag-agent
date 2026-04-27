"""Rule-based complaint severity / topic classifier.

Why rules and not a fine-tuned BERT:
- Production Chinese customer-service teams use regex for exactly this step
  (fast, explainable, easy to add a phrase when a new abuse pattern shows up).
- Zero dependency + zero latency -- critical because this runs on EVERY
  complaint turn. LLM-based sentiment adds 300-700 ms per request.
- Upgrade path preserved: swap ``detect_severity`` / ``detect_topic`` with an
  LLM call behind the same signature; everything else in the specialist +
  SLA logic stays the same.

Severity rubric (empirical, tune per tenant):
    high   -- user is escalating: threats to go public, regulatory keywords,
              strong profanity, repeated ultimatums, multi-channel threats.
              Action: auto-assign to a human, 1h SLA.
    medium -- user is clearly unhappy and wants resolution, but civil:
              "不满意" / "退钱" / "换一单". 4h SLA.
    low    -- asking about a minor inconvenience, mild tone. 24h SLA.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


Severity = Literal["low", "medium", "high"]
Topic = Literal["delivery", "quality", "service", "refund", "price", "other"]


# --- severity keyword tables -----------------------------------------------

# "High" patterns: escalation threats, regulatory/media mentions, strong rage.
_HIGH_PATTERNS = [
    # regulatory / consumer-protection channels (严重投诉信号)
    r"12315", r"工商局", r"消协", r"消费者协会",
    r"市场监管", r"315", r"投诉电话",
    # media / public shaming
    r"黑猫投诉", r"微博曝光", r"发朋友圈", r"上电视",
    r"媒体曝光", r"向媒体", r"小红书曝",
    # legal
    r"打官司", r"起诉", r"报警", r"法院", r"律师函", r"法律途径",
    # ultimatums / rage markers
    r"马上(?:给我)?(?:退|换|处理)",
    r"必须(?:今天|立即|马上)",
    r"否则(?:我)?(?:投诉|曝光|退货|起诉)",
    r"再不(?:处理|解决|回复)",
    r"气死我了", r"太垃圾了", r"什么破玩意",
    # profanity (very minimal; production tunes per locale)
    r"傻[逼屄]", r"骗子", r"垃圾(?:客服|平台)", r"滚",
]

# "Medium": clearly unhappy, wants action, civil tone.
_MEDIUM_PATTERNS = [
    r"不满意", r"不满", r"失望", r"很差", r"质量差",
    r"(?:有|太)(?:多)?问题", r"(?:破|坏|裂|碎|缺)(?:了|损)",
    r"投诉", r"申诉", r"抱怨", r"差评",
    r"不合适", r"跟描述不符", r"货不对板", r"图文不符",
    r"要退(?:货|钱|款)", r"想退",
    r"太慢了", r"等了", r"(?:几|\d+)天(?:还)?没(?:到|发货|处理)",
    r"客服(?:态度)?(?:差|不好|垃圾|不理人)",
    r"能不能(?:快|处理)一下",
]

_HIGH_RE = re.compile("|".join(_HIGH_PATTERNS), flags=re.IGNORECASE)
_MEDIUM_RE = re.compile("|".join(_MEDIUM_PATTERNS), flags=re.IGNORECASE)


# --- topic keyword tables --------------------------------------------------

_TOPIC_RULES: list[tuple[str, Topic]] = [
    # Order matters: more-specific categories first.
    (r"(?:没到|没发货|物流|快递|运输|迟到|延误|未签收)", "delivery"),
    (r"(?:坏了|碎了|破损|损坏|质量|假货|次品|缺货|缺件|漏发|错发)", "quality"),
    (r"(?:退款|退钱|退货|refund|退不了|退不下来)", "refund"),
    (r"(?:客服(?:态度|不理)|人工|机器人|回复慢|不回复)", "service"),
    (r"(?:差价|降价|价保|价格不一|贵|虚假标价)", "price"),
]


@dataclass
class EmotionVerdict:
    severity: Severity
    topic: Topic
    matched_high: list[str]       # which high-signal patterns fired
    matched_medium: list[str]     # which medium-signal patterns fired
    matched_topic: str | None     # which topic pattern hit


def detect_severity(text: str) -> tuple[Severity, list[str], list[str]]:
    """Return (severity, high_hits, medium_hits)."""
    if not text or not text.strip():
        return "low", [], []
    high_hits = sorted(set(m.group(0) for m in _HIGH_RE.finditer(text)))
    if high_hits:
        return "high", high_hits, []
    medium_hits = sorted(set(m.group(0) for m in _MEDIUM_RE.finditer(text)))
    if medium_hits:
        return "medium", [], medium_hits
    return "low", [], []


def detect_topic(text: str) -> tuple[Topic, str | None]:
    if not text:
        return "other", None
    for pattern, topic in _TOPIC_RULES:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return topic, m.group(0)
    return "other", None


def classify(text: str) -> EmotionVerdict:
    severity, high_hits, medium_hits = detect_severity(text)
    topic, matched_topic = detect_topic(text)
    return EmotionVerdict(
        severity=severity,
        topic=topic,
        matched_high=high_hits,
        matched_medium=medium_hits,
        matched_topic=matched_topic,
    )
