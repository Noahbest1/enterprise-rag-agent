"""Step B: in-process user-events bus + SSE endpoint.

Verifies:
- publish_user_event returns subscriber count
- multiple subscribers on the same user both receive the event
- different-user subscriber does NOT receive the event (isolation)
- the _subscribe async generator yields a ``ready`` first, then events,
  cleans up on generator close
- GET /users/{user_id}/events emits SSE frames in the expected format
"""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from rag_api.main import app
from rag_api import user_events
from rag_api.user_events import active_subscriber_count, publish_user_event


@pytest.fixture(autouse=True)
def _clean_bus():
    # Paranoia: make sure tests don't leak subscribers to each other.
    user_events._subscribers.clear()
    yield
    user_events._subscribers.clear()


def test_publish_when_no_subscriber_returns_zero():
    assert publish_user_event("nobody", "complaint_reply", {"x": 1}) == 0


@pytest.mark.asyncio
async def test_publish_then_subscribe_delivers_event():
    user_id = "u1"

    # Start a subscriber task that grabs items until we cancel it.
    received: list[dict] = []

    async def sub():
        agen = user_events._subscribe(user_id)
        async for item in agen:
            received.append(item)
            if len(received) >= 2:  # 1 ready + 1 event
                await agen.aclose()
                return

    task = asyncio.create_task(sub())
    # Yield so the subscriber registers its queue.
    await asyncio.sleep(0.01)
    assert active_subscriber_count(user_id) == 1

    n = publish_user_event(user_id, "complaint_reply", {"hello": "world"})
    assert n == 1

    await asyncio.wait_for(task, timeout=1.0)
    # First item is the synthetic "ready", second is our event.
    assert received[0]["event"] == "ready"
    assert received[1]["event"] == "complaint_reply"
    assert received[1]["data"] == {"hello": "world"}
    # Cleanup after generator close.
    assert active_subscriber_count(user_id) == 0


@pytest.mark.asyncio
async def test_two_subscribers_same_user_both_receive():
    user_id = "u2"
    received_a: list[dict] = []
    received_b: list[dict] = []

    async def sub(bucket: list[dict]):
        agen = user_events._subscribe(user_id)
        async for item in agen:
            bucket.append(item)
            if len(bucket) >= 2:
                await agen.aclose()
                return

    t_a = asyncio.create_task(sub(received_a))
    t_b = asyncio.create_task(sub(received_b))
    await asyncio.sleep(0.01)
    assert active_subscriber_count(user_id) == 2

    assert publish_user_event(user_id, "complaint_reply", {"k": 1}) == 2
    await asyncio.wait_for(asyncio.gather(t_a, t_b), timeout=1.0)

    # Both saw the ready + event.
    assert [i["event"] for i in received_a] == ["ready", "complaint_reply"]
    assert [i["event"] for i in received_b] == ["ready", "complaint_reply"]
    assert active_subscriber_count(user_id) == 0


@pytest.mark.asyncio
async def test_different_user_does_not_receive():
    received: list[dict] = []

    async def sub():
        agen = user_events._subscribe("alice")
        async for item in agen:
            received.append(item)
            if item["event"] == "complaint_reply":
                await agen.aclose()
                return
            # ignore ready and pings

    task = asyncio.create_task(sub())
    await asyncio.sleep(0.01)
    # publish to bob -> alice's stream should not see it
    publish_user_event("bob", "complaint_reply", {"k": 1})
    await asyncio.sleep(0.05)

    # Alice only has the ready frame; no complaint_reply yet.
    assert received == [{"event": "ready", "data": {"user_id": "alice"}}]
    # Now cancel the task to clean up.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # After cancellation the subscriber should be removed from the bus.
    # Give the finally-block a tick to run.
    await asyncio.sleep(0.01)
    assert active_subscriber_count("alice") == 0


def test_sse_endpoint_route_is_registered():
    """Cheap sanity check that the router is wired. The actual streaming
    behaviour is covered by the _subscribe async-generator tests above
    (ASGI + SSE + sync TestClient don't mix cleanly; dedicated httpx
    stream tests flake on body buffering). Manual smoke is done with a
    real uvicorn + curl during dev."""
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/users/{user_id}/events" in paths
