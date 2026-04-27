"""In-process user-scoped event bus + SSE endpoint.

The admin dashboard posts a reply to a complaint. The reply handler calls
``publish_user_event(user_id, "complaint_reply", payload)``. Any live
``GET /users/{user_id}/events`` SSE connection the target user has open
receives the event and the AgentChat UI injects a system bubble.

This is a **single-process** bus -- fine for resume-scale where one API
container handles all traffic. A real multi-replica deploy would swap this
for Redis pub/sub (same publish/subscribe surface, swap the backing store
in a future step).
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse


log = logging.getLogger("user_events")

# user_id -> list of live asyncio.Queue subscribers.
_subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)

# Queue size per subscriber. 64 is generous for the expected bursts (typing
# indicators, admin replies). Overflow drops oldest-style would need deque,
# but a slow client blocking the producer is better than silently losing
# replies, so we use put_nowait + drop with a warning.
_QUEUE_MAX = 64

# Heartbeat cadence. Must be shorter than typical reverse-proxy idle timeout
# (nginx default 60s, cloudflare 100s). 15s gives plenty of margin.
_HEARTBEAT_SECONDS = 15.0


def publish_user_event(user_id: str, event_type: str, data: dict) -> int:
    """Push an event to every live subscriber for ``user_id``.

    Returns the number of subscribers notified. Zero means the user has no
    open AgentChat tab right now; the event is NOT buffered (the admin reply
    still persists in ``complaint_replies`` so a rehydrate on the next chat
    load can surface it -- scope for a future turn).
    """
    queues = _subscribers.get(user_id) or []
    delivered = 0
    for q in list(queues):
        try:
            q.put_nowait({"event": event_type, "data": data})
            delivered += 1
        except asyncio.QueueFull:  # pragma: no cover -- very slow consumer
            log.warning("user_events_queue_full", extra={"user_id": user_id})
    return delivered


def active_subscriber_count(user_id: str | None = None) -> int:
    """Test / ops helper. Total subscribers if user_id is None, else for that user."""
    if user_id is None:
        return sum(len(v) for v in _subscribers.values())
    return len(_subscribers.get(user_id) or [])


async def _subscribe(user_id: str) -> AsyncGenerator[dict, None]:
    q: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAX)
    _subscribers[user_id].append(q)
    try:
        # Immediately flush a "ready" so the client knows the channel is live.
        yield {"event": "ready", "data": {"user_id": user_id}}
        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=_HEARTBEAT_SECONDS)
                yield item
            except asyncio.TimeoutError:
                # Keep-alive comment keeps the connection warm without
                # polluting the event stream with heartbeat events the UI
                # has to ignore.
                yield {"event": "ping", "data": {}}
    finally:
        try:
            _subscribers[user_id].remove(q)
        except ValueError:  # pragma: no cover -- already removed
            pass
        if not _subscribers[user_id]:
            _subscribers.pop(user_id, None)


router = APIRouter()


@router.get("/users/{user_id}/events")
async def user_events_stream(user_id: str):
    """Long-lived SSE. The AgentChat UI opens this on mount keyed on its
    current user_id. Emits ``ready`` immediately, then ``complaint_reply``
    (or future events) as they happen, with ``ping`` every 15s to keep
    proxies from closing the connection."""
    async def _gen() -> AsyncGenerator[bytes, None]:
        try:
            async for item in _subscribe(user_id):
                event = item["event"]
                payload = json.dumps(item["data"], ensure_ascii=False)
                yield f"event: {event}\ndata: {payload}\n\n".encode("utf-8")
        except asyncio.CancelledError:  # pragma: no cover -- client disconnect
            raise

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
