"""Logistics specialist -- queries track_package by tracking_no.

Picks up tracking_no from state.entities if a previous step (Order)
already resolved it, otherwise tries to find one in the query.
"""
from __future__ import annotations

import re

from ..state import AgentState
from ..tools.orders import track_package


TRACKING_RE = re.compile(r"[A-Z]{2,4}\d{8,16}", re.IGNORECASE)


def _current_step(state: AgentState):
    plan = state.get("plan") or []
    idx = state.get("current_step") or 0
    if 0 <= idx < len(plan):
        return plan[idx]
    return None


def _format_timeline(info: dict) -> str:
    lines = [
        f"物流单号 {info['tracking_no']}({info.get('carrier','?')}),当前状态:{info['status']}",
    ]
    if info.get("eta"):
        lines.append(f"预计送达:{info['eta'][:10]}")
    for ev in info.get("timeline", [])[-3:]:
        lines.append(f"  · {ev['ts'][:19].replace('T',' ')}  {ev['event']}")
    return "\n".join(lines)


def logistics_node(state: AgentState) -> dict:
    step = _current_step(state)
    if step is None:
        return {}

    entities = state.get("entities") or {}
    tracking_no = entities.get("last_tracking_no")
    carrier = entities.get("last_carrier")

    if not tracking_no:
        # Look in the query itself.
        m = TRACKING_RE.search(step["query"])
        if m:
            tracking_no = m.group(0)

    info = track_package(tracking_no, carrier) if tracking_no else None

    if not info:
        return {
            "step_results": {
                **(state.get("step_results") or {}),
                step["step_id"]: {
                    "agent": "logistics",
                    "query": step["query"],
                    "answer": "(未找到物流单号或没有可查的快递信息)",
                    "citations": [],
                    "abstain": True,
                },
            },
            "trace": [{"node": "logistics", "step_id": step["step_id"], "status": "no_match"}],
        }

    summary = _format_timeline(info)
    return {
        "step_results": {
            **(state.get("step_results") or {}),
            step["step_id"]: {
                "agent": "logistics",
                "query": step["query"],
                "answer": summary,
                "tracking_info": info,
                "citations": [],
                "abstain": False,
            },
        },
        "trace": [
            {"node": "logistics", "step_id": step["step_id"], "status": info["status"], "tracking_no": tracking_no}
        ],
    }
