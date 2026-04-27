"""Source-type prior for rerank score.

After the cross-encoder produces a sigmoid-normalized score in [0, 1], we
nudge it by a small ``prior`` based on source path and query intent. The
nudge is small (|delta| <= 0.06) so it only breaks ties; a truly relevant
release-notes chunk still beats a "how-to" doc when the cross-encoder
already scored it much higher.

Signals:
- Query intent: "how-to / tutorial" vs "version / release" vs neutral.
- Source type, inferred from source_path (no re-ingestion needed).
"""
from __future__ import annotations

import re


# Keywords that signal "user wants a tutorial/how-to/getting-started doc"
HOWTO_RE = re.compile(
    r"(how\s*(to|do|can)|set\s*up|configure|setup|install|tutorial|guide|step"
    r"|怎么|如何|怎样|教程|入门|步骤|开始|设置|配置)",
    re.IGNORECASE,
)

# Keywords that signal "user wants a specific release / version"
RELEASE_RE = re.compile(
    r"(which\s+version|what's\s+new|introduced|release\s+note|breaking"
    r"|哪个版本|新增|发布|更新|变更|release\s*\d|\bv\s*\d)",
    re.IGNORECASE,
)

# Keywords that signal "user has an error / troubleshooting question"
TROUBLE_RE = re.compile(
    r"(error|fail|issue|bug|crash|broken|can'?t|cannot|不能|报错|失败|问题|排错|排查)",
    re.IGNORECASE,
)


def infer_source_type(source_path: str) -> str:
    p = source_path.lower()
    if "release" in p or re.search(r"/v[-_.]?\d", p):
        return "release"
    if "faq" in p or "troubleshoot" in p:
        return "faq"
    if "getting-started" in p or "quickstart" in p or "getting_started" in p:
        return "getting_started"
    if "issue" in p or "github.com" in p:
        return "issue"
    return "doc"


def query_intent(query: str) -> str:
    if RELEASE_RE.search(query):
        return "release"
    if TROUBLE_RE.search(query):
        return "trouble"
    if HOWTO_RE.search(query):
        return "howto"
    return "neutral"


# Priors are (query_intent, source_type) -> score delta in [-0.06, +0.06].
_PRIOR_TABLE: dict[tuple[str, str], float] = {
    # How-to questions prefer getting_started and docs
    ("howto", "getting_started"): +0.06,
    ("howto", "doc"):             +0.03,
    ("howto", "release"):         -0.04,
    ("howto", "faq"):             +0.01,
    # Release questions prefer release notes; docs can mention features incidentally
    ("release", "release"):       +0.06,
    ("release", "doc"):           -0.02,
    ("release", "getting_started"): -0.03,
    # Troubleshooting prefers FAQ and issue trackers
    ("trouble", "faq"):           +0.05,
    ("trouble", "issue"):         +0.04,
    ("trouble", "doc"):           +0.01,
    # Neutral: mild bias toward canonical docs
    ("neutral", "getting_started"): +0.02,
    ("neutral", "doc"):           +0.01,
}


def score_prior(query: str, source_path: str) -> float:
    intent = query_intent(query)
    stype = infer_source_type(source_path)
    return _PRIOR_TABLE.get((intent, stype), 0.0)
