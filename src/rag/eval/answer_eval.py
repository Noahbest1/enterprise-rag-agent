"""Answer-layer metrics using LLM-as-Judge.

For each eval row we call the full RAG pipeline and then ask a judge LLM
four yes/no questions:

  - faithful:  does every claim appear in the cited chunks?
  - cites_correctly:  does the answer cite at least one correct source?
  - answers_question: does it address the user's question?
  - abstain_correct: if no relevant sources exist, did it abstain?

The judge is the same Qwen chat endpoint as generation. We aggregate the
four booleans across rows to produce the final metrics. This is cheap to
run (1 extra LLM call per row) and catches most regressions.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict

from ..answer.llm_client import LLMError, chat_once
from ..pipeline import answer_query


JUDGE_SYSTEM = """You are an evaluation judge for a RAG system.

You will see a user question, the answer given (with [n] citations), and the
text of each cited source. Decide four yes/no questions and return ONLY a
valid JSON object with exactly these boolean fields:

  "faithful"          -- every factual claim in the answer is supported by at
                         least one cited source. false if the answer adds facts
                         not in the sources.
  "cites_correctly"   -- the [n] citations point to sources that actually
                         support the claims next to them.
  "answers_question"  -- the answer addresses the user's question, not a
                         different question.
  "abstain_correct"   -- if the sources do NOT contain the answer, the system
                         correctly declined ("I don't have enough information"
                         or similar). If sources DO contain the answer, set
                         this to true (we only penalise wrong abstain).

Return ONLY the JSON. No explanation, no markdown.
"""


@dataclass
class AnswerRowResult:
    query: str
    kb_id: str
    answer: str
    abstained: bool
    faithful: bool
    cites_correctly: bool
    answers_question: bool
    abstain_correct: bool
    judge_raw: str


@dataclass
class AnswerMetrics:
    n: int
    faithful: float
    cites_correctly: float
    answers_question: float
    abstain_correct: float
    abstain_rate: float
    avg_latency_ms: float
    per_row: list[AnswerRowResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_judge(reply: str) -> dict:
    m = _JSON_RE.search(reply)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _judge_row(query: str, answer_text: str, hits: list, abstained: bool) -> tuple[dict, str]:
    sources_block = "\n\n".join(
        f"[{i + 1}] {h.title} ({h.source_id})\n{h.text[:1200]}"
        for i, h in enumerate(hits)
    )
    user = (
        f"Question: {query}\n\n"
        f"Answer given: {answer_text}\n\n"
        f"Abstained: {'yes' if abstained else 'no'}\n\n"
        f"Sources:\n{sources_block}\n\n"
        f"Return the JSON verdict."
    )
    try:
        reply = chat_once(system=JUDGE_SYSTEM, user=user, temperature=0.0, max_output_tokens=200)
    except LLMError as e:
        return {"_error": str(e)}, str(e)
    return _parse_judge(reply), reply


def evaluate_answers(rows: list[dict]) -> AnswerMetrics:
    per_row: list[AnswerRowResult] = []
    faithful = cites = answers = abstain_ok = 0
    abstained_n = 0
    latency_sum = 0

    for row in rows:
        ans = answer_query(row["query"], row["kb_id"])
        latency_sum += ans.latency_ms
        if ans.abstained:
            abstained_n += 1

        verdict, raw = _judge_row(
            query=row["query"],
            answer_text=ans.text,
            hits=ans.hits,
            abstained=ans.abstained,
        )
        f = bool(verdict.get("faithful"))
        c = bool(verdict.get("cites_correctly"))
        a = bool(verdict.get("answers_question"))
        ab = bool(verdict.get("abstain_correct"))
        per_row.append(
            AnswerRowResult(
                query=row["query"],
                kb_id=row["kb_id"],
                answer=ans.text,
                abstained=ans.abstained,
                faithful=f,
                cites_correctly=c,
                answers_question=a,
                abstain_correct=ab,
                judge_raw=raw,
            )
        )
        faithful += int(f)
        cites += int(c)
        answers += int(a)
        abstain_ok += int(ab)

    n = len(rows) or 1
    return AnswerMetrics(
        n=len(rows),
        faithful=faithful / n,
        cites_correctly=cites / n,
        answers_question=answers / n,
        abstain_correct=abstain_ok / n,
        abstain_rate=abstained_n / n,
        avg_latency_ms=latency_sum / n,
        per_row=per_row,
    )
