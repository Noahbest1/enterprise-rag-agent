/**
 * Agent chat UI — shows the LangGraph Plan-and-Execute flow.
 *
 * Subscribes to POST /agent/chat (SSE) and renders each event as a distinct
 * message type so the user (or a watching ops person) can SEE the agent
 * think: which specialists fired, what each one returned, the final answer.
 *
 * Event mapping (from src/rag_api/agent_routes.py):
 *     agent_start       -> status chip "tenant X received"
 *     plan              -> blue "Plan" card with N-step list
 *     specialist_start  -> new specialist card in "running" state
 *     specialist_done   -> flip that card to "done" + attach trace
 *     answer            -> final assistant bubble + citations
 *     done              -> latency footer
 *     error             -> red error chip
 *
 * Image upload goes through POST /vision/ask separately (not through agent)
 * because vision_routes is a different code path that already composes
 * image descriptions into a RAG query. Keeping it out of /agent/chat means
 * we don't have to modify the agent graph.
 *
 * Multi-turn via thread_id in localStorage so "查我 iPhone 订单" -> "那能退吗?"
 * keeps the entity context (last_order_id, last_item_title).
 */
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import SessionSidebar from "./SessionSidebar";

const API_BASE = (import.meta as any).env?.VITE_API_BASE || "http://127.0.0.1:8008";

async function postAction(path: string, body: any): Promise<{ok: boolean; data: any; error?: string}> {
  try {
    const r = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) return { ok: false, data, error: data.detail || `HTTP ${r.status}` };
    return { ok: true, data };
  } catch (e: any) {
    return { ok: false, data: {}, error: String(e) };
  }
}

// Bottom-right toast stack. Mutations queue into ``_toastQueue``; the
// ``<ToastHost>`` component in AgentChat subscribes and renders them.
type ToastKind = "success" | "error" | "info";
type ToastItem = { id: number; kind: ToastKind; text: string };
let _toastId = 0;
const _toastSubs = new Set<(items: ToastItem[]) => void>();
let _toastItems: ToastItem[] = [];

function toast(text: string, kind: ToastKind = "info") {
  const item: ToastItem = { id: ++_toastId, kind, text };
  _toastItems = [..._toastItems, item];
  _toastSubs.forEach((s) => s(_toastItems));
  // Auto-dismiss after 4s
  setTimeout(() => {
    _toastItems = _toastItems.filter((t) => t.id !== item.id);
    _toastSubs.forEach((s) => s(_toastItems));
  }, 4000);
}

function ToastHost() {
  const [items, setItems] = useState<ToastItem[]>(_toastItems);
  useEffect(() => {
    _toastSubs.add(setItems);
    return () => { _toastSubs.delete(setItems); };
  }, []);
  return (
    <div style={S.toastHost}>
      {items.map((t) => (
        <div key={t.id} style={{
          ...S.toastItem,
          background: t.kind === "error" ? "#fee2e2" : t.kind === "success" ? "#d1fae5" : "#dbeafe",
          color: t.kind === "error" ? "#991b1b" : t.kind === "success" ? "#065f46" : "#1e3a8a",
          borderColor: t.kind === "error" ? "#fecaca" : t.kind === "success" ? "#a7f3d0" : "#bfdbfe",
        }}>
          {t.kind === "error" ? "❌ " : t.kind === "success" ? "✓ " : "ℹ "}
          {t.text}
        </div>
      ))}
    </div>
  );
}

type Tenant = "jd" | "taobao";

type PlanStep = {
  step_id: number;
  agent: string;
  query: string;
  depends_on?: number[];
};

type Citation = {
  n: number;
  source_id: string;
  title: string;
  source_path?: string;
  section_path?: string[];
  snippet?: string;
};

type SpecialistTrace = {
  agent: string;
  step_id: number;
  started_at?: number;
  finished_at?: number;
  state: "running" | "done" | "error";
  output?: any;       // whatever specialist_done returned
  raw?: any;          // full event payload
};

type AgentMessage =
  | { kind: "user"; text: string; image?: { name: string; preview: string } }
  | { kind: "agent_start"; tenant: string; user_id: string }
  | { kind: "plan"; steps: PlanStep[] }
  | { kind: "specialist"; trace: SpecialistTrace }
  | { kind: "answer"; text: string; citations: Citation[]; entities?: Record<string, any> }
  | { kind: "error"; detail: string }
  | { kind: "done"; latency_ms: number };

// Out-of-turn complaint thread state — admin replies arrive async via SSE,
// user replies are typed here. Each ComplaintThread groups all replies for
// one complaint so the UI shows "C-1 has 3 replies" as one card instead of
// 3 untethered bubbles.
type ThreadReply = {
  id: number | string;        // server id, or "tmp-..." for optimistic local
  at: number;                 // ms epoch (for sort)
  author_kind: "admin" | "user" | "system";
  author_label: string;
  content: string;
};

type ComplaintThread = {
  complaint_id: number;
  severity?: string;
  topic?: string;
  status?: string;
  escalated?: boolean;
  assigned_to?: string | null;
  replies: ThreadReply[];
};

type Turn = {
  userInput: string;
  image?: { name: string; preview: string; bytes?: File };
  stream: AgentMessage[];   // messages appended as SSE flows in
  streaming: boolean;
};

// localStorage helpers — keyed on thread_id so each session is isolated.
const turnsKey = (threadId: string) => `agent_chat_turns_${threadId}`;
const inboxSeenKey = (userId: string) => `agent_inbox_last_seen_${userId}`;

function loadTurns(threadId: string): Turn[] {
  try {
    const raw = localStorage.getItem(turnsKey(threadId));
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    // Drop any in-flight `streaming: true` flag from prior session — the SSE
    // stream that was driving it is long gone.
    return parsed.map((t: any) => ({ ...t, streaming: false }));
  } catch {
    return [];
  }
}

function saveTurns(threadId: string, turns: Turn[]) {
  try {
    // Strip non-serializable File objects from image payloads before persist;
    // the data-URL preview survives and that's enough for re-render.
    const slim = turns.map((t) => ({
      ...t,
      image: t.image
        ? { name: t.image.name, preview: t.image.preview }
        : undefined,
    }));
    localStorage.setItem(turnsKey(threadId), JSON.stringify(slim));
  } catch {
    // localStorage quota exceeded etc. — drop the persist silently;
    // in-memory state still works for this session.
  }
}

export default function AgentChat() {
  const [tenant, setTenant] = useState<Tenant>(
    (new URLSearchParams(window.location.search).get("tenant") as Tenant) || "jd",
  );
  const [query, setQuery] = useState("");
  const [threadId, setThreadId] = useState<string>(() => {
    const key = "agent_chat_thread_id";
    let tid = localStorage.getItem(key);
    if (!tid) {
      tid = `t-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      localStorage.setItem(key, tid);
    }
    return tid;
  });
  // Bump this number to force the SessionSidebar to re-fetch the list
  // (e.g. after agent_chat just touched a new session).
  const [sidebarRefresh, setSidebarRefresh] = useState<number>(0);
  // Bump to force the inbox-fetch effect to re-run without changing
  // tenant/threadId — used after submit-complaint so the new ticket
  // appears in the "本会话工单" panel immediately, not after a reload.
  const [inboxRefresh, setInboxRefresh] = useState<number>(0);
  // Globally accessible refresh trigger so deeply-nested ComplaintCard
  // (rendered inside specialistRenderer) can ask for an inbox refresh
  // without prop-drilling through 3 layers.
  (window as any).__bumpInbox = () => setInboxRefresh((n) => n + 1);
  // Restore prior conversation from localStorage so a reload doesn't wipe
  // the screen. Backend keeps full agent state via AsyncSqliteSaver(thread_id),
  // so the multi-turn entity carry-forward also keeps working.
  const [turns, setTurns] = useState<Turn[]>(() => loadTurns(threadId));
  const [busy, setBusy] = useState(false);
  const [showTrace, setShowTrace] = useState(true);
  const [pendingImage, setPendingImage] = useState<File | null>(null);
  const [pendingPreview, setPendingPreview] = useState<string | null>(null);
  const [threads, setThreads] = useState<ComplaintThread[]>([]);
  // Tickets are shown in a collapsible panel ABOVE the chat (not in the
  // main scroll) so the chat history stays clean. Auto-opens when an admin
  // reply lands; the unread counter resets when the user opens the panel.
  const [ticketsPanelOpen, setTicketsPanelOpen] = useState<boolean>(false);
  const [unreadAdminReplies, setUnreadAdminReplies] = useState<number>(0);
  const endRef = useRef<HTMLDivElement | null>(null);
  // Ref mirrors the current threadId so the long-lived SSE handler reads
  // the LATEST value, not the stale value captured when the effect first
  // ran. Without this, switching sessions left the SSE closure pointing
  // at the old thread_id, and admin replies on the active session got
  // mis-classified as "cross-session" (the user only saw them after a
  // forced re-fetch via session switch).
  const currentThreadIdRef = useRef<string>(threadId);
  useEffect(() => {
    currentThreadIdRef.current = threadId;
  }, [threadId]);

  // Resolve current user_id (used for SSE subscription + user-side reply
  // posting). Same convention as the backend default.
  const userId = tenant === "jd" ? "jd-demo-user" : "tb-demo-user";

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, threads]);

  // Helper used by both inbox-rehydrate and live SSE handler.
  function upsertReply(complaint_id: number, reply: ThreadReply, meta?: Partial<ComplaintThread>) {
    setThreads((prev) => {
      const idx = prev.findIndex((t) => t.complaint_id === complaint_id);
      if (idx === -1) {
        return [...prev, {
          complaint_id,
          ...(meta || {}),
          replies: [reply],
        }];
      }
      const t = prev[idx];
      // Dedupe by id (server pushes can race with local optimistic adds).
      if (t.replies.some((r) => r.id === reply.id)) return prev;
      const merged = [...t.replies, reply].sort((a, b) => a.at - b.at);
      const updated = { ...t, ...(meta || {}), replies: merged };
      return [...prev.slice(0, idx), updated, ...prev.slice(idx + 1)];
    });
  }

  // Persist turns whenever they change. Keyed on threadId, so "New session"
  // (which rotates threadId via newThread) gets a fresh empty store.
  useEffect(() => {
    saveTurns(threadId, turns);
  }, [turns, threadId]);

  // On mount + tenant + thread switch: rehydrate the user's complaint
  // threads SCOPED TO THE CURRENT SESSION. Tickets created elsewhere are
  // not pulled in, so each session feels self-contained. (NULL-thread_id
  // legacy complaints still appear in every session — server-side decision
  // to avoid orphaning old data.)
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const url = `${API_BASE}/users/${encodeURIComponent(userId)}/inbox`
          + `?limit=20&thread_id=${encodeURIComponent(threadId)}`;
        const r = await fetch(url);
        if (!r.ok || cancelled) return;
        const data = await r.json();
        const fresh: ComplaintThread[] = (data.items || []).map((c: any) => ({
          complaint_id: c.complaint_id,
          severity: c.severity,
          topic: c.topic,
          status: c.status,
          escalated: c.escalated,
          assigned_to: c.assigned_to,
          replies: (c.replies || []).map((r: any) => ({
            id: r.id,
            at: r.created_at ? new Date(r.created_at).getTime() : Date.now(),
            author_kind: r.author_kind || "admin",
            author_label: r.author_label,
            content: r.content,
          })).sort((a: ThreadReply, b: ThreadReply) => a.at - b.at),
        }))
        // Only show threads with at least one message — empty preview-only
        // complaints are noise (they appear because submit-button created
        // the row but admin hasn't engaged yet).
        .filter((t: ComplaintThread) => t.replies.length > 0);

        if (!cancelled && fresh.length) {
          setThreads((prev) => {
            // Merge: server data wins on metadata, but local optimistic
            // user replies (id starting with "tmp-") are preserved.
            const byId = new Map(fresh.map((t) => [t.complaint_id, t]));
            const merged: ComplaintThread[] = [];
            for (const t of fresh) {
              const localPending = prev.find((p) => p.complaint_id === t.complaint_id)
                ?.replies.filter((r) => typeof r.id === "string" && r.id.startsWith("tmp-")) || [];
              merged.push({
                ...t,
                replies: [...t.replies, ...localPending].sort((a, b) => a.at - b.at),
              });
            }
            // Keep any local-only threads (just in case) that aren't in server response
            for (const p of prev) {
              if (!byId.has(p.complaint_id)) merged.push(p);
            }
            return merged;
          });
        }
        localStorage.setItem(inboxSeenKey(userId), new Date().toISOString());
      } catch {
        // swallow; offline or backend-down shouldn't break the chat UI
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tenant, threadId, inboxRefresh]);

  async function postUserReply(complaintId: number, content: string) {
    const trimmed = content.trim();
    if (!trimmed) return false;
    // Optimistic insert so the user sees their bubble immediately.
    const tmpId = `tmp-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    upsertReply(complaintId, {
      id: tmpId,
      at: Date.now(),
      author_kind: "user",
      author_label: userId,
      content: trimmed,
    });
    try {
      const r = await fetch(`${API_BASE}/complaints/${complaintId}/user-reply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId, content: trimmed }),
      });
      if (!r.ok) {
        const data = await r.json().catch(() => ({}));
        toast(`回复失败:${data.detail || r.status}`, "error");
        // Roll back optimistic message.
        setThreads((prev) =>
          prev.map((t) =>
            t.complaint_id === complaintId
              ? { ...t, replies: t.replies.filter((rp) => rp.id !== tmpId) }
              : t
          )
        );
        return false;
      }
      const data = await r.json();
      // Replace tmp id with real server id.
      setThreads((prev) =>
        prev.map((t) =>
          t.complaint_id === complaintId
            ? {
                ...t,
                replies: t.replies.map((rp) =>
                  rp.id === tmpId
                    ? { ...rp, id: data.reply.id }
                    : rp
                ),
              }
            : t
        )
      );
      return true;
    } catch (e: any) {
      toast(`回复失败:${String(e)}`, "error");
      setThreads((prev) =>
        prev.map((t) =>
          t.complaint_id === complaintId
            ? { ...t, replies: t.replies.filter((rp) => rp.id !== tmpId) }
            : t
        )
      );
      return false;
    }
  }

  // Subscribe to /users/{user_id}/events so admin replies surface in the
  // matching ComplaintThread card in real time. Reconnects on tenant change;
  // EventSource retries on its own when the connection drops.
  useEffect(() => {
    const url = `${API_BASE}/users/${encodeURIComponent(userId)}/events`;
    const es = new EventSource(url);
    es.addEventListener("complaint_reply", (ev: MessageEvent) => {
      try {
        const d = JSON.parse(ev.data);
        // Filter by thread_id: only render the reply in the current
        // session if the complaint belongs to it. Read the LIVE threadId
        // from the ref (not the captured-at-mount value, otherwise every
        // reply on the active session looked like "cross-session" after
        // the user had switched at least once).
        const replyThreadId = d.thread_id ?? null;
        const liveThreadId = currentThreadIdRef.current;
        const isCurrentSession = replyThreadId && replyThreadId === liveThreadId;

        if (isCurrentSession) {
          upsertReply(d.complaint_id, {
            id: d.reply_id || `live-${Date.now()}`,
            at: d.created_at ? new Date(d.created_at).getTime() : Date.now(),
            author_kind: "admin",
            author_label: d.author_label,
            content: d.content,
          });
          toast(`客服 ${d.author_label} 回复了工单 C-${d.complaint_id}`, "success");
          setTicketsPanelOpen(true);
          setUnreadAdminReplies((c) => c + 1);
        } else if (replyThreadId) {
          // Reply belongs to another session the user has open elsewhere.
          toast(`客服 ${d.author_label} 回复了 C-${d.complaint_id}(在其他会话中,请切换查看)`, "info");
        } else {
          // Reply belongs to a deleted session (thread_id NULL after
          // session delete). Don't render in any session; just inform.
          toast(`客服 ${d.author_label} 回复了 C-${d.complaint_id}(原会话已删除,可在客服后台查看)`, "info");
        }
      } catch {}
    });
    es.onerror = () => { /* EventSource auto-retries; silence noise */ };
    return () => { es.close(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tenant]);

  function newThread() {
    // Mint a fresh thread_id and switch to it. The old thread's persisted
    // turns are KEPT in localStorage (keyed on the old threadId) so the
    // sidebar can let the user switch back. Server-side AsyncSqliteSaver
    // also keeps the old conversation. ("New session" = create a new
    // empty session, not delete the old one.)
    const tid = `t-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    localStorage.setItem("agent_chat_thread_id", tid);
    setThreadId(tid);
    setTurns([]);
    setThreads([]);
    setTicketsPanelOpen(false);
    setUnreadAdminReplies(0);
    setSidebarRefresh((n) => n + 1);
  }

  function selectSession(newThreadId: string) {
    if (newThreadId === threadId) return;
    localStorage.setItem("agent_chat_thread_id", newThreadId);
    setThreadId(newThreadId);
    // Restore the selected session's persisted turns from localStorage.
    setTurns(loadTurns(newThreadId));
    // Tickets and inbox auto-refresh via the [tenant, threadId] effect.
    setThreads([]);
    setTicketsPanelOpen(false);
    setUnreadAdminReplies(0);
  }

  function toggleTicketsPanel() {
    setTicketsPanelOpen((open) => {
      if (!open) setUnreadAdminReplies(0);  // opening = "marked as read"
      return !open;
    });
  }

  function onPickImage(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    setPendingImage(f);
    const reader = new FileReader();
    reader.onload = () => setPendingPreview(String(reader.result));
    reader.readAsDataURL(f);
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (busy) return;
    const q = query.trim();
    if (!q && !pendingImage) return;

    // Image path goes through /vision/ask (separate endpoint, composes
    // image description into a RAG query). Not streamed via /agent/chat
    // to keep the graph path simple.
    if (pendingImage) {
      await runVisionAsk(q || "请看这张图并基于知识库回答我", pendingImage);
      setPendingImage(null);
      setPendingPreview(null);
      setQuery("");
      return;
    }
    await runAgentChat(q);
    setQuery("");
  }

  async function runAgentChat(q: string) {
    const turn: Turn = { userInput: q, stream: [], streaming: true };
    setTurns((t) => [...t, turn]);
    setBusy(true);
    try {
      const resp = await fetch(`${API_BASE}/agent/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, tenant, thread_id: threadId }),
      });
      if (!resp.ok || !resp.body) {
        throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buf = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let sep = buf.indexOf("\n\n");
        while (sep !== -1) {
          const frame = buf.slice(0, sep);
          buf = buf.slice(sep + 2);
          applyEvent(parseFrame(frame));
          sep = buf.indexOf("\n\n");
        }
      }
    } catch (err: any) {
      appendToLastTurn({ kind: "error", detail: String(err) });
    } finally {
      setBusy(false);
      flipLastTurnStreamingOff();
      // Bump sidebar refresh: the just-sent message either created a new
      // session row (touch_session inserts) or updated last_msg_at on an
      // existing one. Either way the sidebar should re-fetch to show the
      // new ordering / new title.
      setSidebarRefresh((n) => n + 1);
    }
  }

  async function runVisionAsk(question: string, file: File) {
    // Show the user's image+text message first
    const preview = await new Promise<string>((resolve) => {
      const r = new FileReader();
      r.onload = () => resolve(String(r.result));
      r.readAsDataURL(file);
    });
    const turn: Turn = {
      userInput: question,
      image: { name: file.name, preview, bytes: file },
      stream: [],
      streaming: true,
    };
    setTurns((t) => [...t, turn]);
    setBusy(true);
    try {
      const form = new FormData();
      form.append("image", file);
      form.append("question", question);
      // /vision/ask expects kb_id; for the "customer service" scenario we
      // default to the tenant-matching demo KB.
      const kb_id = tenant === "jd" ? "jd_demo" : "taobao_demo";
      form.append("kb_id", kb_id);
      const resp = await fetch(`${API_BASE}/vision/ask`, { method: "POST", body: form });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
      const data = await resp.json();
      appendToLastTurn({
        kind: "answer",
        text: data.answer,
        citations: (data.citations || []).map((c: any) => ({
          n: c.n, title: c.title, source_id: c.source_id, snippet: c.snippet,
        })),
      });
      if (typeof data.latency_ms === "number") {
        appendToLastTurn({ kind: "done", latency_ms: data.latency_ms });
      }
    } catch (err: any) {
      appendToLastTurn({ kind: "error", detail: String(err) });
    } finally {
      setBusy(false);
      flipLastTurnStreamingOff();
    }
  }

  function appendToLastTurn(m: AgentMessage) {
    setTurns((all) => {
      if (!all.length) return all;
      const last = { ...all[all.length - 1], stream: [...all[all.length - 1].stream, m] };
      return [...all.slice(0, -1), last];
    });
  }

  function mutateSpecialist(agent: string, step_id: number, patch: Partial<SpecialistTrace>) {
    setTurns((all) => {
      if (!all.length) return all;
      const last = { ...all[all.length - 1] };
      last.stream = last.stream.map((m) => {
        if (m.kind === "specialist" && m.trace.agent === agent && m.trace.step_id === step_id) {
          return { ...m, trace: { ...m.trace, ...patch } } as AgentMessage;
        }
        return m;
      });
      return [...all.slice(0, -1), last];
    });
  }

  function flipLastTurnStreamingOff() {
    setTurns((all) => {
      if (!all.length) return all;
      const last = { ...all[all.length - 1], streaming: false };
      return [...all.slice(0, -1), last];
    });
  }

  function applyEvent(ev: { event: string; data: any } | null) {
    if (!ev) return;
    const { event, data } = ev;
    switch (event) {
      case "agent_start":
        appendToLastTurn({ kind: "agent_start", tenant: data.tenant, user_id: data.user_id });
        break;
      case "plan":
        appendToLastTurn({ kind: "plan", steps: data.plan || [] });
        break;
      case "specialist_start":
        appendToLastTurn({
          kind: "specialist",
          trace: {
            agent: data.agent,
            step_id: data.step_id,
            state: "running",
            started_at: Date.now(),
          },
        });
        break;
      case "specialist_done":
        mutateSpecialist(data.agent, data.step_id, {
          state: "done",
          finished_at: Date.now(),
          output: data.trace || data.output || data.result || null,
          raw: data,
        });
        break;
      case "answer":
        appendToLastTurn({
          kind: "answer",
          text: data.text || "",
          citations: (data.citations || []).map((c: any) => ({
            n: c.n, title: c.title, source_id: c.source_id, snippet: c.snippet,
          })),
          entities: data.entities,
        });
        break;
      case "done":
        appendToLastTurn({ kind: "done", latency_ms: data.total_latency_ms || 0 });
        break;
      case "error":
        appendToLastTurn({ kind: "error", detail: data.detail || "agent error" });
        break;
    }
  }

  async function sendFeedback(verdict: "up" | "down", turn: Turn) {
    const ans = turn.stream.find((m) => m.kind === "answer") as any;
    if (!ans) return;
    await fetch(`${API_BASE}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        trace_id: threadId,
        kb_id: tenant === "jd" ? "jd_demo" : "taobao_demo",
        query: turn.userInput,
        answer: ans.text,
        verdict,
      }),
    });
  }

  const latestTrace = useMemo(() => {
    for (let i = turns.length - 1; i >= 0; i--) {
      if (turns[i].stream.length) return turns[i];
    }
    return null;
  }, [turns]);

  return (
    <div style={S.root}>
      <header style={S.header}>
        <div style={S.brand}>
          <span style={{ fontWeight: 700 }}>🤖 Enterprise Agent</span>
          <span style={S.badge}>{tenant.toUpperCase()}</span>
          <span style={S.subtle}>thread: {threadId.slice(0, 12)}…</span>
        </div>
        <div style={S.controls}>
          <a href="/?view=admin" target="_blank" rel="noreferrer"
             style={{
               fontSize: 11, color: "#6b7280", textDecoration: "none",
               border: "1px solid #e5e7eb", padding: "4px 8px", borderRadius: 6,
               marginRight: 4,
             }}
             title="新标签页打开客服后台 — 演示双面板时用">
            🎧 客服后台 ↗
          </a>
          <label style={S.label}>Tenant:</label>
          <select value={tenant} onChange={(e) => setTenant(e.target.value as Tenant)} style={S.select}>
            <option value="jd">京东 JD</option>
            <option value="taobao">淘宝 Taobao</option>
          </select>
          <button onClick={newThread} style={S.ghostBtn}>New session</button>
          <label style={{ ...S.label, marginLeft: 8 }}>
            <input
              type="checkbox"
              checked={showTrace}
              onChange={(e) => setShowTrace(e.target.checked)}
              style={{ marginRight: 4 }}
            />
            Trace
          </label>
        </div>
      </header>

      <div style={S.body}>
        {/* Left: ChatGPT-style session list. Filters tickets/inbox by
            current thread_id; "New session" mints a fresh thread_id +
            switches to it without reloading the page. */}
        <SessionSidebar
          userId={userId}
          currentThreadId={threadId}
          onSelect={selectSession}
          onNewSession={newThread}
          refreshSignal={sidebarRefresh}
        />

        {/* Middle: ticket panel (collapsible) + main chat scroll. */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          {/* Ticket panel — separate from main chat scroll. Scoped to the
              current session via the inbox `?thread_id=` filter; old sessions
              don't bleed in. Auto-opens when admin reply arrives. */}
          {threads.length > 0 && (
            <div style={{
              borderBottom: "1px solid #e5e7eb",
              background: ticketsPanelOpen ? "#fafbff" : "#fff",
            }}>
              <button
                onClick={toggleTicketsPanel}
                style={{
                  width: "100%",
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  padding: "8px 18px",
                  border: 0,
                  background: "transparent",
                  fontSize: 13,
                  color: "#374151",
                  cursor: "pointer",
                  textAlign: "left",
                  fontFamily: "inherit",
                }}
              >
                <span>{ticketsPanelOpen ? "▾" : "▸"}</span>
                <span>🎫 本会话工单</span>
                <span style={{
                  fontSize: 11, color: "#6b7280",
                  background: "#f3f4f6", padding: "1px 8px", borderRadius: 10,
                }}>
                  {threads.length}
                </span>
                {unreadAdminReplies > 0 && (
                  <span style={{
                    fontSize: 11, color: "#fff",
                    background: "#dc2626", padding: "1px 8px", borderRadius: 10,
                    fontWeight: 500,
                  }}>
                    ⚠ {unreadAdminReplies} 条新回复
                  </span>
                )}
                <span style={{ marginLeft: "auto", fontSize: 11, color: "#9ca3af" }}>
                  {ticketsPanelOpen ? "点击收起" : "点击展开"}
                </span>
              </button>
              {ticketsPanelOpen && (
                <div style={{
                  padding: "0 18px 14px",
                  maxHeight: "40vh",
                  overflowY: "auto",
                  display: "flex",
                  flexDirection: "column",
                  gap: 10,
                }}>
                  {threads.map((t) => (
                    <ComplaintThreadCard
                      key={t.complaint_id}
                      thread={t}
                      currentUserLabel={userId}
                      onReply={(content) => postUserReply(t.complaint_id, content)}
                    />
                  ))}
                </div>
              )}
            </div>
          )}

        <main style={S.main}>
          {turns.length === 0 && (
            <div style={S.empty}>
              <h3 style={{ margin: 0 }}>企业级智能客服 — Agent 版</h3>
              <p style={{ color: "#666" }}>
                Plan-and-Execute · <b>9 specialists</b>(订单 / 物流 / 退货 / 投诉 / 发票 / 地址 / 推荐 / 商品 QA / 政策 QA)
                · 内部调 RAG · 多模态 · MCP tools.
                <br />
                试试这些:
              </p>
              <ul style={{ color: "#666" }}>
                <li>"查我最近的 iPhone 订单,想退货,退多少钱?"</li>
                <li>"PLUS 会员年费多少?运费上有啥优惠?"  <span style={{color:"#7c3aed"}}>(走 RAG)</span></li>
                <li>上传商品截图 + "这个在你们平台多少钱?能退吗?"</li>
                <li>"给我推荐几个类似 iPhone 的手机"</li>
                <li>"我要去 12315 投诉你们物流" <span style={{color:"#dc2626"}}>(触发投诉,需点按钮才落库)</span></li>
              </ul>
              <p style={{ color: "#888", fontSize: 12, marginTop: 8 }}>
                💡 客服回复(从 <a href="/?view=admin" style={{ color: "#2563eb" }}>客服后台</a> 推送)会显示在顶部「🎫 我的工单」面板里,有新消息会自动弹出。
              </p>
            </div>
          )}
          {turns.map((turn, i) => (
            <TurnView key={i} turn={turn} onFeedback={(v) => sendFeedback(v, turn)} />
          ))}
          <div ref={endRef} />
        </main>
        </div>{/* end of middle column wrapper */}

        {showTrace && (
          <aside style={S.traceAside}>
            <TracePanel turn={latestTrace} />
          </aside>
        )}
      </div>

      <form onSubmit={onSubmit} style={S.inputRow}>
        {pendingPreview && (
          <div style={S.imagePending}>
            <img src={pendingPreview} alt="pending" style={{ height: 40, borderRadius: 6 }} />
            <button type="button" onClick={() => { setPendingImage(null); setPendingPreview(null); }}
                    style={S.ghostBtn}>×</button>
          </div>
        )}
        <label style={S.imgBtn}>
          📎
          <input type="file" accept="image/*" onChange={onPickImage} style={{ display: "none" }} />
        </label>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={busy ? "agent working..." : pendingImage ? "Ask about the image..." : "Message the agent..."}
          disabled={busy}
          style={S.input}
        />
        <button type="submit" disabled={busy || (!query.trim() && !pendingImage)} style={S.sendBtn}>
          {busy ? "…" : "Send"}
        </button>
      </form>
      <ToastHost />
    </div>
  );
}

// ---------- subviews ----------

function TurnView({ turn, onFeedback }: { turn: Turn; onFeedback: (v: "up" | "down") => void }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <UserBubble turn={turn} />
      {turn.stream.map((m, i) => {
        switch (m.kind) {
          case "agent_start":
            return (
              <StatusChip key={i}>
                🏃 Agent received (tenant={m.tenant}, user={m.user_id})
              </StatusChip>
            );
          case "plan":
            return <PlanCard key={i} steps={m.steps} />;
          case "specialist":
            return <SpecialistCard key={i} trace={m.trace} />;
          case "answer":
            return <AnswerBubble key={i} text={m.text} citations={m.citations} entities={m.entities} onFeedback={onFeedback} />;
          case "error":
            return <ErrorChip key={i}>{m.detail}</ErrorChip>;
          case "done":
            return <DoneChip key={i} latency={m.latency_ms} />;
          default:
            return null;
        }
      })}
      {turn.streaming && !turn.stream.some((m) => m.kind === "answer") && (
        <StatusChip>
          <span style={{ opacity: 0.6 }}>…thinking</span>
        </StatusChip>
      )}
    </div>
  );
}

function UserBubble({ turn }: { turn: Turn }) {
  return (
    <div style={{ ...S.bubbleRow, justifyContent: "flex-end" }}>
      <div style={{ ...S.bubble, ...S.bubbleUser }}>
        {turn.image && (
          <img src={turn.image.preview} alt={turn.image.name}
               style={{ maxWidth: 200, borderRadius: 8, display: "block", marginBottom: 6 }} />
        )}
        <div style={S.text}>{turn.userInput}</div>
      </div>
    </div>
  );
}

function PlanCard({ steps }: { steps: PlanStep[] }) {
  return (
    <div style={S.planCard}>
      <div style={S.planHeader}>📋 Plan · {steps.length} step{steps.length === 1 ? "" : "s"}</div>
      <ol style={{ margin: 0, paddingLeft: 22 }}>
        {steps.map((s) => (
          <li key={s.step_id} style={{ marginBottom: 4 }}>
            <span style={S.planAgent}>{s.agent}</span>
            <span style={{ color: "#555" }}> — {s.query}</span>
            {s.depends_on && s.depends_on.length > 0 && (
              <span style={S.planDep}> (needs step {s.depends_on.join(", ")})</span>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}

function SpecialistCard({ trace }: { trace: SpecialistTrace }) {
  const [open, setOpen] = useState(false);
  const elapsed = trace.finished_at && trace.started_at
    ? `${trace.finished_at - trace.started_at} ms` : "…";
  const color = trace.state === "running" ? "#e4a94a" : trace.state === "error" ? "#c33" : "#2d9050";

  const fancy = specialistRenderer(trace.agent, trace.output);

  const summary = (() => {
    const o = trace.output;
    if (!o) return trace.state === "running" ? "…" : "(no output)";
    if (typeof o === "string") return o.slice(0, 80);
    if (o.answer) return String(o.answer).slice(0, 80);
    if (o.summary) return String(o.summary).slice(0, 80);
    try { return JSON.stringify(o).slice(0, 80) + "…"; } catch { return "(trace)"; }
  })();

  // RAG provenance badge — product_qa / policy_qa specialists hit /answer
  // internally. Showing the badge makes "Agent uses RAG as a tool" visible
  // to anyone watching the demo without us narrating it.
  const usesRag = trace.agent === "product_qa" || trace.agent === "policy_qa";

  return (
    <div style={S.spCard}>
      <div style={S.spHeader} onClick={() => setOpen(!open)}>
        <span style={{ ...S.spDot, background: color }} />
        <span style={S.spAgent}>{trace.agent}</span>
        {usesRag && (
          <span title="该 specialist 内部调用 RAG 检索 KB"
                style={{
                  fontSize: 10, padding: "1px 6px", borderRadius: 8,
                  background: "#ede9fe", color: "#7c3aed", fontWeight: 500,
                  border: "1px solid #ddd6fe",
                }}>
            📚 RAG
          </span>
        )}
        <span style={S.spStep}>step {trace.step_id}</span>
        <span style={S.spElapsed}>{elapsed}</span>
        {!fancy && <span style={S.spSummary}>{summary}</span>}
        <span style={S.spCaret}>{open ? "▾" : "▸"}</span>
      </div>
      {fancy && <div style={S.spFancy}>{fancy}</div>}
      {open && trace.raw && (
        <pre style={S.spTrace}>{JSON.stringify(trace.raw, null, 2)}</pre>
      )}
    </div>
  );
}

// ---------- agent-specific fancy renderers ----------

type OrderItem = { sku?: string; title: string; qty: number; unit_price_yuan?: number };
type OrderShape = {
  id: string;
  status?: string;
  total_yuan?: number;
  currency?: string;
  tracking_no?: string | null;
  carrier?: string | null;
  placed_at?: string;
  tenant?: string;
  user_id?: string;
  items?: OrderItem[];
};

function specialistRenderer(agent: string, output: any): JSX.Element | null {
  if (!output) return null;
  if (agent === "order" && Array.isArray(output.orders) && output.orders.length > 0) {
    return <OrderList orders={output.orders} />;
  }
  if (agent === "aftersale" && (output.eligibility || output.order_id)) {
    return <AfterSaleCard data={output} />;
  }
  if (agent === "recommend" && Array.isArray(output.items) && output.items.length > 0) {
    return <ProductStrip items={output.items} anchor={output.anchor} />;
  }
  if (agent === "logistics" && output.tracking_info) {
    return <LogisticsTimeline data={output.tracking_info} />;
  }
  if (agent === "invoice" && output.invoice) {
    return <InvoiceCard data={output} />;
  }
  if (agent === "complaint" && (output.complaint || output.preview)) {
    return <ComplaintCard data={output} />;
  }
  if (agent === "account") {
    return <AccountCard data={output} />;
  }
  return null;
}

function statusBadge(status?: string) {
  const s = (status || "").toLowerCase();
  const bg =
    s === "delivered" ? "#059669" :
    s === "shipped"   ? "#0a84ff" :
    s === "placed"    ? "#888" :
    s === "paid"      ? "#8b5cf6" :
    s === "refunded"  ? "#c77700" :
    s === "cancelled" ? "#c33" : "#555";
  const label =
    s === "delivered" ? "已签收" :
    s === "shipped"   ? "运输中" :
    s === "placed"    ? "已下单" :
    s === "paid"      ? "已付款" :
    s === "refunded"  ? "已退款" :
    s === "cancelled" ? "已取消" : (status || "—");
  return <span style={{ ...S.statusBadge, background: bg }}>{label}</span>;
}

function formatYuan(y?: number) {
  if (y == null) return "—";
  return `¥${y.toLocaleString("zh-CN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function OrderList({ orders }: { orders: OrderShape[] }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {orders.map((o) => <OrderCard key={o.id} order={o} />)}
    </div>
  );
}

function OrderCard({ order }: { order: OrderShape }) {
  const [submitted, setSubmitted] = useState<{ id?: number; status?: string } | null>(null);
  const placed = order.placed_at ? new Date(order.placed_at).toLocaleString("zh-CN", { hour12: false }) : "";
  // 7-day no-reason return eligibility: server is the source of truth, but
  // Status-aware action: each order state maps to a specific user-action,
  // mirroring real e-commerce flows. The 4 buckets:
  //  placed / paid  → 取消订单 (self-service, no shipment yet)
  //  shipped        → 联系客服拦截 (CS handles courier interception)
  //  delivered      → 此单退货 (7-day no-reason self-service)
  //  cancelled / refunded → no action
  const status = (order.status || "").toLowerCase();
  const action: "cancel" | "escalate" | "return" | null =
    status === "delivered" ? "return"
    : (status === "placed" || status === "paid") ? "cancel"
    : status === "shipped" ? "escalate"
    : null;

  async function onReturn() {
    const r = await postAction("/agent/actions/confirm-return",
      { order_id: order.id, reason: "用户从订单卡点击退货" });
    if (r.ok) {
      const rid = r.data?.request?.request_id;
      const refund = ((r.data?.eligibility?.refund_cents ?? 0) / 100).toFixed(2);
      setSubmitted({ id: rid, status: "pending" });
      toast(`订单 ${order.id} 已提交退货 R-${rid ?? "?"},预计退款 ¥${refund}`, "success");
    } else {
      toast(`订单 ${order.id} 退货失败:${r.error}`, "error");
    }
  }

  async function onCancelOrder() {
    if (!confirm(`确认取消订单 ${order.id}?未发货可全额退款。`)) return;
    const r = await postAction("/agent/actions/cancel-order", { order_id: order.id });
    if (r.ok) {
      const refund = ((r.data?.result?.refunded_cents ?? 0) / 100).toFixed(2);
      toast(`订单 ${order.id} 已取消,全额退款 ¥${refund}`, "success");
    } else {
      toast(`取消失败:${r.error}`, "error");
    }
  }

  async function onEscalate(topic: "delivery" | "refund", desc: string) {
    // Escalate to a human agent via the complaint flow. We leverage the
    // existing complaint tool — the order is in transit (shipped), or
    // past the 7-day window, both of which need CS intervention.
    const r = await postAction("/agent/actions/submit-complaint", {
      tenant: order.tenant || "jd",
      user_id: order.user_id,
      order_id: order.id,
      severity: "medium",
      topic,
      content: `${desc} (订单 ${order.id})`,
      would_escalate: true,
    });
    if (r.ok) {
      const cid = r.data?.complaint?.id;
      toast(`已转人工客服(C-${cid ?? "?"}),稍后会主动联系您`, "success");
    } else {
      toast(`转人工失败:${r.error}`, "error");
    }
  }

  async function onCancel() {
    if (!submitted?.id) return;
    const r = await postAction("/agent/actions/cancel-return-request",
      { request_id: submitted.id });
    if (r.ok) {
      setSubmitted({ ...submitted, status: "cancelled" });
      toast(`R-${submitted.id} 已取消`, "success");
    } else {
      toast(`取消失败:${r.error}`, "error");
    }
  }

  return (
    <div style={S.orderCard}>
      <div style={S.orderHead}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontFamily: "monospace", fontWeight: 600 }}>{order.id}</span>
          {statusBadge(order.status)}
        </div>
        <div style={{ fontSize: 13, color: "#666" }}>{formatYuan(order.total_yuan)}</div>
      </div>
      {order.items && order.items.length > 0 && (
        <div style={S.orderItems}>
          {order.items.map((it, i) => (
            <div key={i} style={S.orderItem}>
              <div style={{ fontSize: 13 }}>{it.title}</div>
              <div style={{ fontSize: 12, color: "#888" }}>
                ×{it.qty}{it.unit_price_yuan != null ? ` · ${formatYuan(it.unit_price_yuan)}` : ""}
              </div>
            </div>
          ))}
        </div>
      )}
      <div style={S.orderFoot}>
        {order.tracking_no && (
          <span>📦 {order.carrier || "物流"} · {order.tracking_no}</span>
        )}
        {placed && <span style={{ color: "#888" }}>下单 {placed}</span>}
      </div>
      {action && !submitted && (
        <div style={{ display: "flex", justifyContent: "flex-end" }}>
          {action === "return" && (
            <button style={{ ...S.ghostBtn, padding: "4px 10px", fontSize: 12, color: "#0a84ff" }}
                    onClick={onReturn}>
              此单退货
            </button>
          )}
          {action === "cancel" && (
            <button style={{ ...S.ghostBtn, padding: "4px 10px", fontSize: 12, color: "#dc2626", borderColor: "#fecaca" }}
                    onClick={onCancelOrder}>
              取消订单
            </button>
          )}
          {action === "escalate" && (
            <button style={{ ...S.ghostBtn, padding: "4px 10px", fontSize: 12, color: "#f97316", borderColor: "#fed7aa" }}
                    onClick={() => onEscalate("delivery", "用户希望中途拦截已发出的快递")}>
              联系客服拦截
            </button>
          )}
        </div>
      )}
      {submitted && submitted.status !== "cancelled" && (
        <div style={{ display: "flex", gap: 6, alignItems: "center", justifyContent: "flex-end" }}>
          <span style={{ fontSize: 11, color: "#065f46", background: "#d1fae5", padding: "2px 8px", borderRadius: 6 }}>
            ✓ 已提交 R-{submitted.id} · {submitted.status}
          </span>
          <button style={{ ...S.ghostBtn, padding: "4px 10px", fontSize: 12, color: "#c33", borderColor: "#fecaca" }}
                  onClick={onCancel}>
            取消
          </button>
        </div>
      )}
      {submitted && submitted.status === "cancelled" && (
        <div style={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: 8 }}>
          <span style={{ fontSize: 12, color: "#888" }}>
            退货 R-{submitted.id} 已取消
          </span>
          <button
            onClick={async () => {
              const r = await postAction("/agent/actions/reopen-return-request",
                { request_id: submitted.id });
              if (r.ok) {
                setSubmitted({ id: submitted.id, status: "pending" });
                toast(`R-${submitted.id} 已重新开启`, "success");
              } else {
                toast(`重新开启失败:${r.error}`, "error");
              }
            }}
            style={{
              padding: "3px 10px", borderRadius: 5, border: "1px solid #a7f3d0",
              background: "#fff", color: "#059669",
              fontSize: 11, cursor: "pointer",
            }}
          >
            🔓 重新开启
          </button>
        </div>
      )}
    </div>
  );
}

function AfterSaleCard({ data }: { data: any }) {
  const elig = data.eligibility || {};
  // Local override: when the user presses "确认提交" we inject a synthetic
  // request dict so the card flips to the "submitted" state without waiting
  // for a new SSE event.
  const [override, setOverride] = useState<{ request?: any }>({});
  const req = override.request ?? data.request;
  const kind = data.kind || "return";
  const kindLabel = ({ return: "退货", refund: "退款", exchange: "换货", price_protect: "保价" } as any)[kind] || kind;
  const refundYuan = elig.refund_cents != null ? elig.refund_cents / 100 : undefined;
  const ok = elig.ok !== false;

  return (
    <div style={{ ...S.orderCard, borderColor: ok ? "#d1fae5" : "#fecaca", background: ok ? "#f0fdf4" : "#fef2f2" }}>
      <div style={S.orderHead}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontWeight: 600 }}>🔁 {kindLabel}</span>
          {req?.id && (
            <span style={{ fontFamily: "monospace", fontSize: 12, background: "#fff", padding: "1px 6px", borderRadius: 4 }}>
              R-{req.id}
            </span>
          )}
          {req?.status && statusBadge(req.status)}
        </div>
        {refundYuan != null && (
          <div style={{ fontSize: 15, fontWeight: 600, color: "#059669" }}>
            {formatYuan(refundYuan)}
          </div>
        )}
      </div>
      {elig.item_titles && elig.item_titles.length > 0 && (
        <div style={{ fontSize: 12, color: "#555", marginTop: 4 }}>
          {elig.item_titles.join(" · ")}
        </div>
      )}
      <div style={{ display: "flex", gap: 10, marginTop: 6, fontSize: 12, color: "#666", flexWrap: "wrap" }}>
        {elig.days_left_in_window != null && (
          <span>⏱ 7 天无理由窗口剩 <b>{elig.days_left_in_window}</b> 天</span>
        )}
        {elig.current_status && <span>订单状态: {elig.current_status}</span>}
      </div>
      {ok && !req && (
        <div style={{ marginTop: 8, display: "flex", gap: 6 }}>
          <button style={S.primaryActionBtn} onClick={async () => {
            const r = await postAction("/agent/actions/confirm-return",
              { order_id: data.order_id, reason: "用户在 agent chat 确认退货" });
            if (r.ok) {
              const rid = r.data?.request?.request_id;
              setOverride({ request: { id: rid, status: "pending", kind: "return" } });
              toast(`已提交退货申请 R-${rid ?? "?"},预计退款 ¥${((r.data?.eligibility?.refund_cents ?? 0) / 100).toFixed(2)}`, "success");
            } else {
              toast(`提交失败:${r.error}`, "error");
            }
          }}>
            确认提交{kindLabel}申请
          </button>
          <button style={S.secondaryActionBtn}
                  onClick={() => toast("已为您转人工(demo,下一轮接 Admin 端)", "info")}>
            转人工
          </button>
        </div>
      )}
      {req && req.id && req.status !== "cancelled" && (
        <div style={{ marginTop: 8, display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
          <span style={{ fontSize: 12, color: "#065f46", background: "#d1fae5", padding: "2px 8px", borderRadius: 6 }}>
            ✓ 已提交 R-{req.id} · {req.status}
          </span>
          <button style={{ ...S.ghostBtn, padding: "4px 10px", color: "#c33", borderColor: "#fecaca" }}
                  onClick={async () => {
                    const r = await postAction("/agent/actions/cancel-return-request",
                      { request_id: req.id });
                    if (r.ok) {
                      setOverride({ request: { ...req, status: "cancelled" } });
                      toast(`R-${req.id} 已取消`, "success");
                    } else {
                      toast(`取消失败:${r.error}`, "error");
                    }
                  }}>
            取消此退货
          </button>
        </div>
      )}
      {req && req.status === "cancelled" && (
        <div style={{ marginTop: 8, display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 12, color: "#888" }}>
            退货申请 R-{req.id} 已取消
          </span>
          <button
            onClick={async () => {
              const r = await postAction("/agent/actions/reopen-return-request",
                { request_id: req.id });
              if (r.ok) {
                setOverride({ request: { ...req, status: "pending" } });
                toast(`R-${req.id} 已重新开启`, "success");
              } else {
                toast(`重新开启失败:${r.error}`, "error");
              }
            }}
            style={{
              padding: "3px 10px", borderRadius: 5, border: "1px solid #a7f3d0",
              background: "#fff", color: "#059669",
              fontSize: 11, cursor: "pointer",
            }}
          >
            🔓 重新开启
          </button>
        </div>
      )}
    </div>
  );
}

function ProductStrip({ items, anchor }: { items: any[]; anchor?: string }) {
  return (
    <div>
      {anchor && <div style={{ fontSize: 12, color: "#666", marginBottom: 6 }}>
        基于 <b>{anchor}</b> 的相似推荐 ({items.length})
      </div>}
      <div style={S.productStrip}>
        {items.map((it, i) => (
          <div key={i} style={S.productCard}>
            <div style={{ fontSize: 13, fontWeight: 500, lineHeight: 1.3 }}>{it.title}</div>
            <div style={{ marginTop: 6, fontSize: 15, color: "#0a84ff", fontWeight: 600 }}>
              {formatYuan(it.price_yuan)}
            </div>
            {typeof it.similarity === "number" && (
              <div style={{ fontSize: 11, color: "#888", marginTop: 2 }}>
                相似度 {(it.similarity * 100).toFixed(0)}%
              </div>
            )}
            {it.sku && <div style={{ fontSize: 11, color: "#aaa", marginTop: 4, fontFamily: "monospace" }}>
              {it.sku}
            </div>}
          </div>
        ))}
      </div>
    </div>
  );
}

function LogisticsTimeline({ data }: { data: any }) {
  const events: { ts: string; event: string }[] = data.timeline || [];
  return (
    <div>
      <div style={{ fontSize: 12, color: "#666", marginBottom: 6 }}>
        📦 {data.carrier || "物流"} · {data.tracking_no}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {events.map((e, i) => {
          const ts = e.ts ? new Date(e.ts).toLocaleString("zh-CN", { hour12: false }) : "";
          return (
            <div key={i} style={S.tlRow}>
              <span style={{ ...S.tlDot, background: i === events.length - 1 ? "#059669" : "#cbd5e1" }} />
              <span style={{ fontSize: 12, color: "#555", flex: 1 }}>{e.event}</span>
              <span style={{ fontSize: 11, color: "#888" }}>{ts}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function InvoiceCard({ data }: { data: any }) {
  const inv = data.invoice || {};
  const action = data.action; // "read" | "created"
  const statusColour =
    inv.status === "issued"    ? "#059669" :
    inv.status === "requested" ? "#d97706" :
    inv.status === "cancelled" ? "#64748b" : "#555";
  const statusLabel =
    inv.status === "issued"    ? "已开具" :
    inv.status === "requested" ? "处理中" :
    inv.status === "cancelled" ? "已取消" : (inv.status || "—");
  return (
    <div style={{ ...S.orderCard, borderColor: "#e5e7eb" }}>
      <div style={S.orderHead}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontWeight: 600 }}>🧾 发票</span>
          <span style={{ fontFamily: "monospace", fontSize: 12, color: "#555" }}>
            {inv.order_id}
          </span>
          <span style={{ ...S.statusBadge, background: statusColour }}>{statusLabel}</span>
          {action === "created" && (
            <span style={{ ...S.statusBadge, background: "#0a84ff" }}>刚提交</span>
          )}
        </div>
        <div style={{ fontSize: 15, fontWeight: 600 }}>{formatYuan(inv.amount_yuan)}</div>
      </div>
      <div style={{ fontSize: 12, color: "#555", display: "flex", gap: 10, flexWrap: "wrap" }}>
        <span>抬头:<b>{inv.title || "—"}</b></span>
        {inv.tax_id && <span>税号:<code style={S.code}>{inv.tax_id}</code></span>}
        <span>类型:{inv.invoice_type === "electronic" ? "电子发票" : "纸质发票"}</span>
      </div>
      {inv.status === "issued" && (
        <div style={{ marginTop: 4 }}>
          <a href={inv.download_url || `${API_BASE}/invoice/${inv.id}.pdf`}
             target="_blank" rel="noreferrer" style={S.primaryLink}>
            下载发票 ↗
          </a>
        </div>
      )}
      {inv.status === "requested" && (
        <div style={{ marginTop: 4, display: "flex", gap: 10, alignItems: "center" }}>
          <a href={`${API_BASE}/invoice/${inv.id}.pdf`}
             target="_blank" rel="noreferrer" style={S.primaryLink}>
            预览发票草稿 ↗
          </a>
          <span style={{ fontSize: 11, color: "#888" }}>
            (正式发票 1-2 工作日内开具)
          </span>
        </div>
      )}
      {data.reason === "order_not_eligible" && (
        <div style={{ fontSize: 12, color: "#c33" }}>
          订单当前状态暂不支持开票(需已发货或已签收)。
        </div>
      )}
    </div>
  );
}

function ComplaintCard({ data }: { data: any }) {
  // Two states this card can be in:
  //   1) Dry-run preview (data.preview present, data.complaint == null) —
  //      specialist just classified; user must click 提交工单 to create a row.
  //   2) Submitted (data.complaint has id) — either user clicked submit
  //      this turn (we hold it via override), or the backend pushed an
  //      already-existing complaint via a future SSE.
  // override carries either the freshly-submitted complaint or the cancel state.
  const [override, setOverride] = useState<{
    complaint?: any;
    status?: string;
    escalated?: boolean;
  }>({});

  const submitted = override.complaint || data.complaint;
  const isPreview = !submitted;
  const preview = data.preview || {};

  // Severity / topic come from whichever state we're in.
  const sev = (submitted?.severity || preview.severity) as "high" | "medium" | "low" | undefined;
  const topic = submitted?.topic || preview.topic;
  const escalated = override.status ? override.escalated : (submitted?.escalated ?? preview.would_escalate);

  const bg = sev === "high" ? "#fef2f2" : sev === "medium" ? "#fffbeb" : "#f3f4f6";
  const border = sev === "high" ? "#fecaca" : sev === "medium" ? "#fde68a" : "#e5e7eb";
  const sevLabel = sev === "high" ? "紧急" : sev === "medium" ? "普通" : "一般";
  const sevColour = sev === "high" ? "#dc2626" : sev === "medium" ? "#d97706" : "#64748b";
  const topicLabel = ({
    delivery: "物流延误", quality: "商品质量", service: "客服服务",
    refund: "退款问题", price: "价格问题", other: "其他",
  } as any)[topic] || topic;
  const status = override.status || submitted?.status;
  const sla = submitted?.sla_due_at
    ? new Date(submitted.sla_due_at).toLocaleString("zh-CN", { hour12: false })
    : null;

  async function onSubmit() {
    // Pull the active session id from localStorage so the submitted ticket
    // is linked to the session that filed it. (ComplaintCard is rendered
    // deep inside specialistRenderer; threading threadId via props would
    // mean updating 3 layers of components for one field.)
    const currentThreadId = localStorage.getItem("agent_chat_thread_id") || undefined;
    const r = await postAction("/agent/actions/submit-complaint", {
      severity: preview.severity,
      topic: preview.topic,
      user_id: preview.user_id,
      tenant: preview.tenant,
      thread_id: currentThreadId,
      order_id: preview.order_id,
      content: preview.content_for_submit,
    });
    if (r.ok) {
      const c = r.data?.complaint;
      setOverride({ complaint: c });
      toast(
        c?.escalated
          ? `已提交工单 C-${c.id} 并升级人工(${c.assigned_to})`
          : `已提交工单 C-${c.id}`,
        "success",
      );
      // Trigger the AgentChat-level inbox re-fetch so the "本会话工单"
      // panel at the top picks up the new ticket without a page reload.
      try { (window as any).__bumpInbox?.(); } catch {}
    } else {
      toast(`提交失败:${r.error}`, "error");
    }
  }

  async function onCancel() {
    if (!submitted?.id) return;
    const r = await postAction("/agent/actions/cancel-complaint", { complaint_id: submitted.id });
    if (r.ok) {
      setOverride({ ...override, status: "closed", escalated: false });
      toast(`工单 C-${submitted.id} 已${escalated ? "撤销升级并" : ""}关闭`, "success");
      // Trigger top "本会话工单" panel refresh too — without this the
      // ComplaintThreadCard up there keeps showing the input + escalation
      // chip until the user manually switches sessions.
      try { (window as any).__bumpInbox?.(); } catch {}
    } else {
      toast(`撤销失败:${r.error}`, "error");
    }
  }

  return (
    <div style={{ ...S.orderCard, background: bg, borderColor: border }}>
      <div style={S.orderHead}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <span style={{ fontWeight: 600 }}>
            {sev === "high" ? "🚨" : sev === "medium" ? "⚠️" : "🙋"} 投诉工单
            {isPreview && (
              <span style={{ marginLeft: 6, fontSize: 11, color: "#64748b", fontWeight: 400 }}>
                · 待用户确认
              </span>
            )}
          </span>
          {submitted?.id && (
            <span style={{ fontFamily: "monospace", fontSize: 12 }}>C-{submitted.id}</span>
          )}
          <span style={{ ...S.statusBadge, background: sevColour }}>{sevLabel}</span>
          <span style={{ ...S.statusBadge, background: "#475569" }}>{topicLabel}</span>
          {escalated && (
            <span style={{ ...S.statusBadge, background: "#dc2626" }}>
              {isPreview ? "将升级人工" : "已升级人工"}
            </span>
          )}
        </div>
      </div>
      {!isPreview && submitted?.escalated && submitted?.assigned_to && (
        <div style={{ fontSize: 13, color: "#7f1d1d" }}>
          已分配给资深客服 <code style={S.code}>{submitted.assigned_to}</code>,将在 1 小时内主动联系您。
        </div>
      )}
      {isPreview && preview.would_escalate && preview.suggested_assignee && (
        <div style={{ fontSize: 12, color: "#7f1d1d" }}>
          预计将分配给资深客服 <code style={S.code}>{preview.suggested_assignee}</code>(提交后立即生效)。
        </div>
      )}
      {sla && (
        <div style={{ fontSize: 12, color: "#666" }}>
          ⏱ SLA:{sla} 前处理
        </div>
      )}
      {Array.isArray(data.matched_high) && data.matched_high.length > 0 && (
        <div style={{ fontSize: 11, color: "#7f1d1d" }}>
          识别到升级信号:{data.matched_high.slice(0, 4).join("、")}
        </div>
      )}

      {/* Dry-run preview action row — submit / cancel-without-submitting */}
      {isPreview && (
        <div style={{ display: "flex", gap: 6, marginTop: 6 }}>
          <button style={S.primaryActionBtn} onClick={onSubmit}>
            提交工单{escalated ? "(并升级人工)" : ""}
          </button>
          <button style={S.secondaryActionBtn}
                  onClick={() => toast("已忽略此次投诉建议,未生成工单", "info")}>
            暂不提交
          </button>
        </div>
      )}

      {/* Submitted-and-still-open action row */}
      {!isPreview && status !== "closed" && (
        <div style={{ display: "flex", gap: 6, marginTop: 6 }}>
          {!escalated && (
            <button style={S.primaryActionBtn}
                    onClick={() => toast("已为您呼叫客服,接通后自动衔接工单(SSE 实时链路下一轮做)", "info")}>
              联系客服
            </button>
          )}
          <button style={S.secondaryActionBtn} onClick={onCancel}>
            {escalated ? "撤销升级并关闭" : "撤销工单"}
          </button>
        </div>
      )}
      {!isPreview && status === "closed" && (
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 4 }}>
          <span style={{ fontSize: 12, color: "#059669" }}>✓ 工单已关闭</span>
          <button
            onClick={async () => {
              const r = await postAction("/agent/actions/reopen-complaint",
                { complaint_id: submitted!.id });
              if (r.ok) {
                const reopened = r.data?.complaint;
                setOverride({ complaint: reopened, status: reopened?.status, escalated: reopened?.escalated });
                toast(`工单 C-${submitted!.id} 已重新开启`, "success");
                try { (window as any).__bumpInbox?.(); } catch {}
              } else {
                toast(`重新开启失败:${r.error}`, "error");
              }
            }}
            style={{
              padding: "4px 10px",
              borderRadius: 6,
              border: "1px solid #a7f3d0",
              background: "#fff",
              color: "#059669",
              fontSize: 11,
              cursor: "pointer",
            }}
          >
            🔓 重新开启
          </button>
        </div>
      )}
    </div>
  );
}

function AccountCard({ data }: { data: any }) {
  const intent = data.intent as string | undefined;

  if (intent === "list_addresses" && Array.isArray(data.addresses)) {
    return <AddressList addresses={data.addresses} />;
  }

  if (intent === "set_default_address" && Array.isArray(data.addresses)) {
    return (
      <div>
        <div style={{ fontSize: 13, color: "#059669", marginBottom: 6 }}>
          ✓ 默认地址已更新
        </div>
        <AddressList addresses={data.addresses} />
      </div>
    );
  }

  if (intent === "change_phone" && data.status === "pending_verification") {
    return (
      <div style={{ ...S.orderCard, borderColor: "#fde68a", background: "#fffbeb" }}>
        <div style={{ fontWeight: 600, display: "flex", alignItems: "center", gap: 8 }}>
          📱 手机号变更
          <span style={{ ...S.statusBadge, background: "#d97706" }}>待验证</span>
        </div>
        <div style={{ fontSize: 13 }}>
          <code style={S.code}>{data.current_phone_masked}</code>
          <span style={{ margin: "0 6px", color: "#888" }}>→</span>
          <code style={S.code}>{data.new_phone_masked}</code>
        </div>
        <div style={{ fontSize: 12, color: "#666" }}>{data.message}</div>
        <PhoneVerifyForm
          userId={data.user_id}
          newPhoneMasked={data.new_phone_masked}
          newPhoneRaw={data.new_phone_raw}
        />
        <div style={{ fontSize: 11, color: "#888" }}>
          (Demo 验证码:<code style={S.code}>123456</code>)
        </div>
      </div>
    );
  }

  if (intent === "profile" && data.profile) {
    const p = data.profile;
    const def = data.default_address;
    return (
      <div style={{ ...S.orderCard, borderColor: "#e5e7eb" }}>
        <div style={{ fontWeight: 600, marginBottom: 6 }}>👤 账户资料</div>
        <div style={{ fontSize: 13, display: "grid", gridTemplateColumns: "auto 1fr", gap: "4px 10px" }}>
          <span style={{ color: "#888" }}>ID:</span>
          <code style={S.code}>{p.id}</code>
          <span style={{ color: "#888" }}>昵称:</span>
          <span>{p.display_name}</span>
          <span style={{ color: "#888" }}>手机:</span>
          <code style={S.code}>{p.phone}</code>
          <span style={{ color: "#888" }}>地址数:</span>
          <span>{data.address_count ?? "?"}</span>
        </div>
        {def && (
          <div style={{ marginTop: 8, borderTop: "1px dashed #eee", paddingTop: 6 }}>
            <div style={{ fontSize: 11, color: "#888" }}>默认收货地址</div>
            <AddressRow addr={def} />
          </div>
        )}
      </div>
    );
  }

  if (intent === "add_address") {
    return (
      <div style={{ ...S.orderCard, borderColor: "#e5e7eb" }}>
        <div style={{ fontWeight: 600 }}>📮 新增收货地址</div>
        <div style={{ fontSize: 12, color: "#555" }}>
          请在下一轮提供:<br />
          <code style={S.code}>收件人 · 手机号 · 省/市/区 · 详细地址 · 标签(家/公司)</code>
        </div>
        {Array.isArray(data.addresses) && data.addresses.length > 0 && (
          <div style={{ marginTop: 6, borderTop: "1px dashed #eee", paddingTop: 6 }}>
            <div style={{ fontSize: 11, color: "#888" }}>现有地址</div>
            <AddressList addresses={data.addresses} />
          </div>
        )}
      </div>
    );
  }

  return null;
}

function PhoneVerifyForm(
  { userId, newPhoneMasked, newPhoneRaw }:
  { userId: string; newPhoneMasked: string; newPhoneRaw?: string },
) {
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState<string | null>(null);

  async function submit() {
    if (!newPhoneRaw) { toast("缺少新手机号,请在对话中重新发起"); return; }
    setBusy(true);
    const r = await postAction("/agent/actions/verify-phone-change", {
      user_id: userId, new_phone: newPhoneRaw, code,
    });
    setBusy(false);
    if (r.ok) {
      setDone(`✓ 已变更为 ${r.data.phone_masked}`);
    } else {
      toast(`验证失败:${r.error}`);
    }
  }

  if (done) return <div style={{ fontSize: 13, color: "#059669" }}>{done}</div>;

  return (
    <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
      <input
        type="text"
        placeholder="6 位验证码"
        maxLength={6}
        value={code}
        onChange={(e) => setCode(e.target.value.replace(/\D/g, ""))}
        style={{ ...S.input, padding: "6px 10px", fontSize: 13, width: 120 }}
      />
      <button
        style={S.primaryActionBtn}
        disabled={busy || code.length !== 6}
        onClick={submit}
      >
        {busy ? "..." : "确认变更"}
      </button>
    </div>
  );
}

function AddressList({ addresses }: { addresses: any[] }) {
  if (!addresses.length) return <div style={{ fontSize: 12, color: "#888" }}>暂无收货地址</div>;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {addresses.map((a) => <AddressRow key={a.id} addr={a} />)}
    </div>
  );
}

function AddressRow({ addr }: { addr: any }) {
  const region = [addr.province, addr.city, addr.district].filter(Boolean).join(" ");
  return (
    <div style={{
      border: "1px solid #e5e7eb", borderRadius: 8, padding: "8px 10px",
      background: addr.is_default ? "#f0fdf4" : "#fff",
      display: "flex", flexDirection: "column", gap: 2,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13 }}>
        <span style={{ ...S.statusBadge, background: "#475569" }}>{addr.label}</span>
        {addr.is_default && <span style={{ ...S.statusBadge, background: "#059669" }}>默认</span>}
        <span style={{ fontWeight: 500 }}>{addr.recipient}</span>
        <code style={{ ...S.code, marginLeft: "auto" }}>{addr.phone_masked || addr.phone}</code>
      </div>
      <div style={{ fontSize: 12, color: "#555", display: "flex", alignItems: "flex-end", gap: 6 }}>
        <div style={{ flex: 1 }}>
          {region && <span>{region}  </span>}
          <span>{addr.line1}</span>
        </div>
        {!addr.is_default && (
          <button
            style={{ ...S.ghostBtn, padding: "2px 8px", fontSize: 11 }}
            onClick={async () => {
              const r = await postAction("/agent/actions/set-default-address",
                { user_id: addr.user_id, address_id: addr.id });
              toast(r.ok ? `已将「${addr.label}」设为默认` : `失败:${r.error}`);
            }}
          >
            设为默认
          </button>
        )}
      </div>
    </div>
  );
}

function AnswerBubble({
  text, citations, entities, onFeedback,
}: {
  text: string; citations: Citation[]; entities?: any;
  onFeedback: (v: "up" | "down") => void;
}) {
  return (
    <div style={{ ...S.bubbleRow, justifyContent: "flex-start" }}>
      <div style={{ ...S.bubble, ...S.bubbleAgent }}>
        <div style={S.text}>{text}</div>
        {citations && citations.length > 0 && (
          <div style={S.citations}>
            <div style={S.citationsHeader}>citations ({citations.length})</div>
            {citations.map((c) => (
              <div key={c.n} style={S.citation}>
                <b>[{c.n}]</b> {c.title}
                <div style={S.citationPath}>{c.source_id}</div>
                {c.snippet && <div style={S.snippet}>{c.snippet}</div>}
              </div>
            ))}
          </div>
        )}
        {entities && Object.keys(entities).length > 0 && (
          <div style={S.entities}>
            entities: {Object.entries(entities).map(([k, v]) => `${k}=${String(v).slice(0, 30)}`).join(" · ")}
          </div>
        )}
        <div style={S.feedbackRow}>
          <button onClick={() => onFeedback("up")} style={S.fbBtn}>👍</button>
          <button onClick={() => onFeedback("down")} style={S.fbBtn}>👎</button>
        </div>
      </div>
    </div>
  );
}

function StatusChip({ children }: { children: React.ReactNode }) {
  return <div style={S.statusChip}>{children}</div>;
}

function ErrorChip({ children }: { children: React.ReactNode }) {
  return <div style={S.errorChip}>❌ {children}</div>;
}

function DoneChip({ latency }: { latency: number }) {
  return <div style={S.doneChip}>✓ done · {latency} ms</div>;
}

function ComplaintThreadCard({
  thread,
  currentUserLabel,
  onReply,
}: {
  thread: ComplaintThread;
  currentUserLabel: string;
  onReply: (content: string) => Promise<boolean>;
}) {
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const sevColor =
    thread.severity === "high" ? "#dc2626" :
    thread.severity === "medium" ? "#d97706" : "#64748b";
  const sevLabel =
    thread.severity === "high" ? "紧急" :
    thread.severity === "medium" ? "普通" : "一般";
  const sevIcon =
    thread.severity === "high" ? "🚨" :
    thread.severity === "medium" ? "⚠️" : "🙋";
  const topicLabel = ({
    delivery: "物流延误", quality: "商品质量", service: "客服服务",
    refund: "退款问题", price: "价格问题", other: "其他",
  } as any)[thread.topic || "other"] || thread.topic;

  async function handleSend() {
    if (!draft.trim() || sending) return;
    setSending(true);
    const ok = await onReply(draft);
    setSending(false);
    if (ok) setDraft("");
  }

  return (
    <div style={{
      border: "1px solid #e5e7eb",
      borderRadius: 10,
      background: "#fff",
      boxShadow: "0 1px 0 rgba(0,0,0,0.02)",
      padding: 12,
      display: "flex",
      flexDirection: "column",
      gap: 10,
    }}>
      {/* Thread header — what ticket is this */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
        <span style={{ fontWeight: 600 }}>{sevIcon} 工单对话</span>
        <span style={{ fontFamily: "monospace", fontSize: 12 }}>C-{thread.complaint_id}</span>
        <span style={{ ...S.statusBadge, background: sevColor }}>{sevLabel}</span>
        {thread.topic && (
          <span style={{ ...S.statusBadge, background: "#475569" }}>{topicLabel}</span>
        )}
        {thread.status === "closed" && (
          <span style={{
            fontSize: 11, padding: "2px 8px", borderRadius: 6,
            background: "#e5e7eb", color: "#6b7280", fontWeight: 500,
          }}>
            ✓ 已关闭
          </span>
        )}
        {thread.escalated && thread.assigned_to && thread.status !== "closed" && (
          <span style={{ fontSize: 11, color: "#7f1d1d" }}>
            归属:<code style={S.code}>{thread.assigned_to}</code>
          </span>
        )}
        <span style={{ marginLeft: "auto", fontSize: 11, color: "#888" }}>
          {thread.replies.length} 条消息
        </span>
      </div>

      {/* Bidirectional thread — admin replies on the left (green), user
          replies on the right (blue), chronological order. */}
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {thread.replies.map((r) => {
          const isUser = r.author_kind === "user";
          const time = new Date(r.at).toLocaleTimeString("zh-CN", { hour12: false });
          return (
            <div key={r.id}
                 style={{
                   display: "flex",
                   justifyContent: isUser ? "flex-end" : "flex-start",
                 }}>
              <div style={{
                maxWidth: "78%",
                padding: "8px 12px",
                borderRadius: 12,
                background: isUser ? "#dbeafe" : "#ecfdf5",
                border: `1px solid ${isUser ? "#bfdbfe" : "#a7f3d0"}`,
                fontSize: 13,
                color: "#111",
              }}>
                <div style={{
                  fontSize: 10,
                  color: isUser ? "#1e3a8a" : "#059669",
                  marginBottom: 3,
                  textAlign: isUser ? "right" : "left",
                }}>
                  {isUser ? "👤" : "🎧"} <b>{r.author_label}</b>
                  {r.author_label === currentUserLabel && isUser && " (我)"}
                  {" · "}{time}
                </div>
                <div style={{ whiteSpace: "pre-wrap" }}>{r.content}</div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Per-thread reply input — user can type a reply scoped to THIS
          ticket without it going through the agent planner / RAG. */}
      {/* Closed tickets show a notice + a "重新开启" button instead of the
          reply input. Reopen flips status back to open/escalated based on
          severity (high gets re-escalated with a fresh assignee + 1h SLA),
          so the user can keep talking without losing the thread history. */}
      {thread.status === "closed" ? (
        <div style={{
          paddingTop: 6,
          borderTop: "1px dashed #e5e7eb",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 10,
          flexWrap: "wrap",
        }}>
          <span style={{ fontSize: 12, color: "#6b7280" }}>
            🔒 工单已关闭。需要继续沟通吗?
          </span>
          <button
            onClick={async () => {
              const r = await postAction("/agent/actions/reopen-complaint",
                { complaint_id: thread.complaint_id });
              if (r.ok) {
                toast(`工单 C-${thread.complaint_id} 已重新开启`, "success");
                try { (window as any).__bumpInbox?.(); } catch {}
              } else {
                toast(`重新开启失败:${r.error}`, "error");
              }
            }}
            style={{
              padding: "5px 12px", borderRadius: 6, border: 0,
              background: "#059669", color: "#fff",
              fontSize: 12, fontWeight: 500, cursor: "pointer",
            }}
          >
            🔓 重新开启
          </button>
        </div>
      ) : (
        <div style={{ display: "flex", gap: 6, paddingTop: 6, borderTop: "1px dashed #e5e7eb" }}>
          <input
            type="text"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") handleSend(); }}
            placeholder={`回复客服(关于 C-${thread.complaint_id})…`}
            disabled={sending}
            style={{
              flex: 1,
              padding: "6px 10px",
              border: "1px solid #d1d5db",
              borderRadius: 6,
              fontSize: 13,
              fontFamily: "inherit",
            }}
          />
          <button
            onClick={handleSend}
            disabled={sending || !draft.trim()}
            style={{
              padding: "6px 14px",
              borderRadius: 6,
              border: 0,
              background: "#2563eb",
              color: "#fff",
              fontSize: 12,
              cursor: sending || !draft.trim() ? "not-allowed" : "pointer",
              opacity: sending || !draft.trim() ? 0.5 : 1,
            }}
          >
            {sending ? "…" : "回复客服"}
          </button>
        </div>
      )}
    </div>
  );
}

function TracePanel({ turn }: { turn: Turn | null }) {
  if (!turn) return <div style={{ color: "#aaa", padding: 16 }}>No trace yet. Ask something.</div>;
  const specialists = turn.stream.filter((m) => m.kind === "specialist") as Extract<AgentMessage, { kind: "specialist" }>[];
  const plan = turn.stream.find((m) => m.kind === "plan") as Extract<AgentMessage, { kind: "plan" }> | undefined;
  return (
    <div style={{ padding: 16 }}>
      <div style={S.tracePanelHeader}>Agent trace</div>
      {plan && (
        <div style={{ fontSize: 12, color: "#555", marginBottom: 10 }}>
          Plan: {plan.steps.length} step · {plan.steps.map((s) => s.agent).join(" → ")}
        </div>
      )}
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {specialists.map((m, i) => (
          <div key={i} style={S.traceRow}>
            <span style={{ ...S.spDot, background: m.trace.state === "done" ? "#2d9050" : "#e4a94a" }} />
            <span style={{ fontWeight: 600 }}>{m.trace.agent}</span>
            <span style={{ color: "#888", fontSize: 12 }}>step {m.trace.step_id}</span>
            <span style={{ color: "#888", fontSize: 12, marginLeft: "auto" }}>
              {m.trace.finished_at && m.trace.started_at
                ? `${m.trace.finished_at - m.trace.started_at} ms`
                : "…"}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------- helpers ----------

function parseFrame(frame: string): { event: string; data: any } | null {
  const lines = frame.split("\n");
  let ev = "message";
  const dataLines: string[] = [];
  for (const raw of lines) {
    if (raw.startsWith("event:")) ev = raw.slice(6).trim();
    else if (raw.startsWith("data:")) dataLines.push(raw.slice(5).trim());
  }
  if (!dataLines.length) return null;
  try { return { event: ev, data: JSON.parse(dataLines.join("\n")) }; }
  catch { return { event: ev, data: { raw: dataLines.join("\n") } }; }
}

// ---------- styles ----------

const S: Record<string, React.CSSProperties> = {
  root: {
    display: "flex", flexDirection: "column", height: "100vh",
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    background: "#f4f4f7", color: "#1c1c1c",
  },
  header: {
    padding: "10px 18px", background: "#fff", borderBottom: "1px solid #e5e5ea",
    display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap",
  },
  brand: { display: "flex", alignItems: "center", gap: 10, fontSize: 15 },
  badge: { background: "#0a84ff", color: "#fff", borderRadius: 10, padding: "2px 8px", fontSize: 11 },
  subtle: { color: "#888", fontSize: 11 },
  controls: { display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" },
  label: { fontSize: 12, color: "#555" },
  select: { padding: "4px 8px", fontSize: 13, border: "1px solid #d2d2d7", borderRadius: 6 },
  ghostBtn: { padding: "4px 10px", fontSize: 12, background: "#fff", border: "1px solid #d2d2d7", borderRadius: 6, cursor: "pointer" },
  body: { flex: 1, display: "flex", overflow: "hidden" },
  main: { flex: 1, overflowY: "auto", padding: "16px 20px", display: "flex", flexDirection: "column", gap: 12 },
  traceAside: { width: 280, background: "#fafafc", borderLeft: "1px solid #e5e5ea", overflowY: "auto" },
  empty: { textAlign: "left", color: "#666", marginTop: 24, padding: "0 8px" },
  bubbleRow: { display: "flex", width: "100%" },
  bubble: { maxWidth: "min(680px, 85%)", padding: "10px 14px", borderRadius: 14, fontSize: 14, lineHeight: 1.5, boxShadow: "0 1px 2px rgba(0,0,0,0.05)" },
  bubbleUser: { background: "#0a84ff", color: "#fff" },
  bubbleAgent: { background: "#fff", border: "1px solid #e5e5ea" },
  text: { whiteSpace: "pre-wrap" },
  citations: { marginTop: 10, borderTop: "1px solid #eee", paddingTop: 8 },
  citationsHeader: { fontSize: 11, color: "#888", marginBottom: 4 },
  citation: { fontSize: 12, marginBottom: 8 },
  citationPath: { fontSize: 11, color: "#888" },
  snippet: { fontSize: 12, color: "#555", marginTop: 2 },
  entities: { marginTop: 6, fontSize: 11, color: "#888" },
  feedbackRow: { marginTop: 6, display: "flex", gap: 4 },
  fbBtn: { background: "transparent", border: "1px solid #eee", borderRadius: 6, padding: "2px 8px", fontSize: 12, cursor: "pointer" },

  statusChip: { alignSelf: "flex-start", fontSize: 12, color: "#666", background: "#f0f0f2", padding: "4px 10px", borderRadius: 12 },
  errorChip: { alignSelf: "flex-start", fontSize: 12, color: "#c33", background: "#fee", padding: "4px 10px", borderRadius: 12 },
  doneChip: { alignSelf: "flex-start", fontSize: 11, color: "#aaa" },

  planCard: {
    alignSelf: "flex-start", background: "#eff6ff", border: "1px solid #bfdbfe",
    borderRadius: 12, padding: "10px 14px", fontSize: 13, maxWidth: "min(680px, 85%)",
  },
  planHeader: { fontWeight: 600, marginBottom: 6, color: "#1e40af" },
  planAgent: { fontFamily: "monospace", background: "#dbeafe", padding: "1px 6px", borderRadius: 4, fontSize: 12 },
  planDep: { color: "#666", fontSize: 11, marginLeft: 4 },

  spCard: {
    alignSelf: "flex-start", background: "#fff", border: "1px solid #e5e5ea",
    borderRadius: 10, fontSize: 13, maxWidth: "min(680px, 85%)", minWidth: 300,
  },
  spHeader: {
    display: "flex", alignItems: "center", gap: 8, padding: "8px 12px", cursor: "pointer",
  },
  spDot: { width: 8, height: 8, borderRadius: "50%", display: "inline-block" },
  spAgent: { fontFamily: "monospace", fontWeight: 600 },
  spStep: { color: "#888", fontSize: 11 },
  spElapsed: { color: "#aaa", fontSize: 11 },
  spSummary: { color: "#555", fontSize: 12, flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" },
  spCaret: { color: "#aaa" },
  spTrace: { margin: 0, padding: 12, background: "#f7f7f9", fontSize: 11, overflow: "auto", maxHeight: 280, borderTop: "1px solid #eee" },
  spFancy: { padding: "10px 12px", borderTop: "1px solid #eee" },

  statusBadge: { color: "#fff", fontSize: 11, padding: "2px 8px", borderRadius: 10, fontWeight: 500 },
  orderCard: {
    border: "1px solid #e5e5ea", borderRadius: 10, padding: "10px 12px",
    background: "#fafafc", display: "flex", flexDirection: "column", gap: 6,
  },
  orderHead: { display: "flex", justifyContent: "space-between", alignItems: "center" },
  orderItems: { display: "flex", flexDirection: "column", gap: 4, borderTop: "1px dashed #eee", paddingTop: 6 },
  orderItem: { display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 8 },
  orderFoot: { fontSize: 12, color: "#666", display: "flex", justifyContent: "space-between", gap: 8, flexWrap: "wrap", borderTop: "1px dashed #eee", paddingTop: 6 },

  primaryActionBtn: {
    background: "#0a84ff", color: "#fff", border: "none", padding: "6px 14px",
    borderRadius: 6, fontSize: 13, cursor: "pointer", fontWeight: 500,
  },
  secondaryActionBtn: {
    background: "#fff", color: "#0a84ff", border: "1px solid #0a84ff",
    padding: "6px 14px", borderRadius: 6, fontSize: 13, cursor: "pointer",
  },

  productStrip: { display: "flex", gap: 8, overflowX: "auto", paddingBottom: 4 },
  productCard: {
    minWidth: 160, maxWidth: 200, border: "1px solid #e5e5ea", borderRadius: 8,
    padding: 10, background: "#fff", display: "flex", flexDirection: "column",
  },

  tlRow: { display: "flex", alignItems: "center", gap: 8 },
  tlDot: { width: 8, height: 8, borderRadius: "50%", flexShrink: 0 },

  code: {
    fontFamily: "'SF Mono', Menlo, Consolas, monospace",
    fontSize: 12,
    background: "#f1f5f9",
    padding: "1px 6px",
    borderRadius: 4,
  },
  primaryLink: {
    color: "#0a84ff",
    textDecoration: "none",
    fontSize: 13,
    fontWeight: 500,
  },
  toastHost: {
    position: "fixed", right: 20, bottom: 20, zIndex: 1000,
    display: "flex", flexDirection: "column", gap: 6, maxWidth: 360,
  },
  toastItem: {
    padding: "10px 14px", borderRadius: 10, fontSize: 13,
    border: "1px solid", boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
  },

  tracePanelHeader: { fontWeight: 600, fontSize: 13, marginBottom: 10 },
  traceRow: { display: "flex", alignItems: "center", gap: 8, fontSize: 13, padding: "6px 8px", background: "#fff", border: "1px solid #eee", borderRadius: 6 },

  inputRow: {
    padding: "10px 16px", background: "#fff", borderTop: "1px solid #e5e5ea",
    display: "flex", gap: 8, alignItems: "center",
  },
  imgBtn: {
    padding: "8px 10px", cursor: "pointer", fontSize: 18, background: "#f2f2f4", borderRadius: 8,
  },
  imagePending: {
    display: "flex", alignItems: "center", gap: 6, padding: "2px 6px",
    background: "#f7f7f9", borderRadius: 6, border: "1px solid #eee",
  },
  input: { flex: 1, padding: "10px 14px", fontSize: 14, border: "1px solid #d2d2d7", borderRadius: 10, outline: "none" },
  sendBtn: { padding: "10px 18px", fontSize: 14, background: "#0a84ff", color: "#fff", border: "none", borderRadius: 10, cursor: "pointer", fontWeight: 500 },
};
