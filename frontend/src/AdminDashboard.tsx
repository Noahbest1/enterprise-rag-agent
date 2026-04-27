/**
 * Admin dashboard — `?view=admin`.
 *
 * Shows the complaint queue for admins (三个字:客服后台). The core flow is:
 *
 *   1. GET /admin/complaints?only_escalated=true  — poll every 5s.
 *   2. Pick a row  → GET /admin/complaints/{id}  (complaint + reply history).
 *   3. POST /admin/complaints/{id}/claim  — take ownership.
 *   4. POST /admin/complaints/{id}/reply  — send reply. The API publishes
 *      a `complaint_reply` event on the user's in-process SSE bus, and the
 *      user's open AgentChat injects a "客服 {admin} 回复:…" system bubble.
 *
 * Identity: the admin picks a label (jd-cs-senior-A / -B / tb-cs-senior-A / -B)
 * from a top-right dropdown; this label is sent as `assigned_to` on claim
 * and `author_label` on reply. No real auth — resume-project scope.
 */
import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = (import.meta as any).env?.VITE_API_BASE || "http://127.0.0.1:8008";
const POLL_MS = 5000;

// ---- types ----
type Complaint = {
  id: number;
  user_id: string | null;
  tenant: string;
  order_id: string | null;
  topic: string;
  severity: "low" | "medium" | "high";
  content_hash: string;
  status: "open" | "escalated" | "resolved" | "closed";
  escalated: boolean;
  assigned_to: string | null;
  sla_due_at: string | null;
  created_at: string | null;
};

type Reply = {
  id: number;
  complaint_id: number;
  author_kind: string;
  author_label: string;
  content: string;
  created_at: string | null;
};

// ---- toast ----
type ToastKind = "success" | "error" | "info";
type ToastItem = { id: number; kind: ToastKind; text: string };
let _toastId = 0;
const _toastSubs = new Set<(items: ToastItem[]) => void>();
let _toastItems: ToastItem[] = [];

function toast(text: string, kind: ToastKind = "info") {
  const item: ToastItem = { id: ++_toastId, kind, text };
  _toastItems = [..._toastItems, item];
  _toastSubs.forEach((s) => s(_toastItems));
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

// ---- helpers ----
const ADMIN_LABELS = [
  "jd-cs-senior-A",
  "jd-cs-senior-B",
  "tb-cs-senior-A",
  "tb-cs-senior-B",
];

function severityLabel(s: string) {
  return s === "high" ? "紧急" : s === "medium" ? "普通" : "一般";
}
function severityColor(s: string) {
  return s === "high" ? "#dc2626" : s === "medium" ? "#d97706" : "#6b7280";
}
function severityBg(s: string) {
  return s === "high" ? "#fef2f2" : s === "medium" ? "#fffbeb" : "#f9fafb";
}
function topicLabel(t: string) {
  return ({
    delivery: "物流",
    quality: "质量",
    service: "服务",
    refund: "退款",
    price: "价格",
    other: "其他",
  } as any)[t] || t;
}
function statusLabel(s: string) {
  return ({
    open: "待处理",
    escalated: "已升级",
    resolved: "已解决",
    closed: "已关闭",
  } as any)[s] || s;
}
function statusColor(s: string) {
  return s === "open" ? "#d97706" : s === "escalated" ? "#dc2626" : s === "resolved" ? "#059669" : "#6b7280";
}
function timeAgo(iso: string | null) {
  if (!iso) return "";
  const d = new Date(iso);
  const t = d.getTime();
  const now = Date.now();
  const dt = Math.max(0, now - t);
  // For very recent things, show a relative chip — but anything in the
  // last day shows the actual HH:MM:SS so you can distinguish messages
  // that arrived seconds apart (the "1 小时前" rounding hid this before).
  const hhmmss = d.toLocaleTimeString("zh-CN", { hour12: false });
  if (dt < 30_000) return `刚刚 · ${hhmmss}`;
  if (dt < 60_000) return `${Math.floor(dt / 1000)} 秒前 · ${hhmmss}`;
  if (dt < 3600_000) return `${Math.floor(dt / 60_000)} 分钟前 · ${hhmmss}`;
  // Same calendar day — show today + HH:MM:SS
  const sameDay =
    d.getFullYear() === new Date(now).getFullYear() &&
    d.getMonth() === new Date(now).getMonth() &&
    d.getDate() === new Date(now).getDate();
  if (sameDay) return `今天 ${hhmmss}`;
  // Older — show full M月D日 HH:MM
  return d.toLocaleString("zh-CN", {
    month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false,
  });
}

// ---- component ----
export default function AdminDashboard() {
  const [identity, setIdentity] = useState<string>(
    () => localStorage.getItem("admin_identity") || "jd-cs-senior-A",
  );
  const [onlyEscalated, setOnlyEscalated] = useState<boolean>(true);
  const [complaints, setComplaints] = useState<Complaint[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [replies, setReplies] = useState<Reply[]>([]);
  const [selectedComplaint, setSelectedComplaint] = useState<Complaint | null>(null);
  const [replyText, setReplyText] = useState<string>("");
  const [sending, setSending] = useState<boolean>(false);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    localStorage.setItem("admin_identity", identity);
  }, [identity]);

  // ---- polling the queue ----
  async function refreshList() {
    try {
      const qs = new URLSearchParams();
      if (onlyEscalated) qs.set("only_escalated", "true");
      qs.set("limit", "100");
      const r = await fetch(`${API_BASE}/admin/complaints?${qs.toString()}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      setComplaints(data.items || []);
    } catch (e: any) {
      // swallow; don't spam toasts during polling
      console.error("refreshList failed", e);
    }
  }

  async function refreshSelected(id: number) {
    try {
      const r = await fetch(`${API_BASE}/admin/complaints/${id}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      setSelectedComplaint(data.complaint);
      setReplies(data.replies || []);
    } catch (e: any) {
      toast(`加载工单 #${id} 失败:${String(e)}`, "error");
    }
  }

  useEffect(() => {
    refreshList();
    pollRef.current = window.setInterval(refreshList, POLL_MS);
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onlyEscalated]);

  useEffect(() => {
    if (selectedId) refreshSelected(selectedId);
    else { setSelectedComplaint(null); setReplies([]); }
  }, [selectedId]);

  // Auto-poll the selected complaint so user-side replies show up without
  // the admin pressing refresh. 4s is gentle enough on the API and faster
  // than a human will notice.
  useEffect(() => {
    if (!selectedId) return;
    const interval = window.setInterval(() => {
      refreshSelected(selectedId);
    }, 4000);
    return () => window.clearInterval(interval);
  }, [selectedId]);

  // ---- actions ----
  async function claimSelected() {
    if (!selectedId) return;
    try {
      const r = await fetch(`${API_BASE}/admin/complaints/${selectedId}/claim`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ assigned_to: identity }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      toast(`已认领 C-${selectedId}`, "success");
      await refreshSelected(selectedId);
      await refreshList();
    } catch (e: any) {
      toast(`认领失败:${String(e)}`, "error");
    }
  }

  async function sendReply() {
    if (!selectedId) return;
    const content = replyText.trim();
    if (!content) return;
    setSending(true);
    try {
      const r = await fetch(`${API_BASE}/admin/complaints/${selectedId}/reply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ author_label: identity, content }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      const n = data.subscribers_notified ?? 0;
      toast(
        n > 0
          ? `已回复 · 用户端已实时收到(${n} 个订阅者)`
          : "已回复 · 用户当前未在线,待下次打开时补充",
        "success",
      );
      setReplyText("");
      await refreshSelected(selectedId);
    } catch (e: any) {
      toast(`回复失败:${String(e)}`, "error");
    } finally {
      setSending(false);
    }
  }

  // ---- derived ----
  const counts = useMemo(() => {
    const c = { all: complaints.length, high: 0, open: 0, escalated: 0 };
    for (const x of complaints) {
      if (x.severity === "high") c.high += 1;
      if (x.status === "open") c.open += 1;
      if (x.status === "escalated") c.escalated += 1;
    }
    return c;
  }, [complaints]);

  return (
    <div style={S.page}>
      <header style={S.header}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <a href="/" style={{
            color: "#6b7280", textDecoration: "none", fontSize: 13,
            padding: "4px 10px", borderRadius: 6, border: "1px solid #e5e7eb",
            background: "#fff",
          }}>← 首页</a>
          <span style={{ fontSize: 18, fontWeight: 600 }}>🎧 客服后台</span>
          <span style={S.counterChip}>
            {counts.all} 条 · 升级 {counts.escalated} · 开放 {counts.open}
          </span>
          <span style={{ fontSize: 11, color: "#6b7280" }}>
            ↳ 用户在 <a href="/?view=agent" style={{ color: "#2563eb" }}>智能客服</a> 提交工单 → 此处认领回复 → SSE 实时推送回用户聊天页
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <label style={{ fontSize: 12, color: "#555", display: "flex", gap: 6, alignItems: "center" }}>
            <input
              type="checkbox"
              checked={onlyEscalated}
              onChange={(e) => setOnlyEscalated(e.target.checked)}
            />
            只看已升级
          </label>
          <select value={identity} onChange={(e) => setIdentity(e.target.value)} style={S.select}>
            {ADMIN_LABELS.map((l) => <option key={l} value={l}>{l}</option>)}
          </select>
          <button onClick={refreshList} style={S.ghostBtn}>刷新</button>
        </div>
      </header>

      <div style={S.body}>
        <aside style={S.listPane}>
          {complaints.length === 0 && (
            <div style={S.emptyList}>
              当前无工单。用户提交投诉后会在此出现。
            </div>
          )}
          {complaints.map((c) => (
            <div
              key={c.id}
              onClick={() => setSelectedId(c.id)}
              style={{
                ...S.listItem,
                background: selectedId === c.id ? "#eff6ff"
                  : c.status === "closed" ? "#f9fafb"
                  : severityBg(c.severity),
                borderLeft: `3px solid ${c.status === "closed" ? "#9ca3af" : severityColor(c.severity)}`,
                opacity: c.status === "closed" ? 0.65 : 1,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
                <span style={{ ...S.sevDot, background: severityColor(c.severity) }}>
                  {severityLabel(c.severity)}
                </span>
                <span style={{ fontFamily: "monospace", fontSize: 12 }}>C-{c.id}</span>
                <span style={{ fontSize: 11, color: statusColor(c.status) }}>
                  · {statusLabel(c.status)}
                </span>
                <span style={{ marginLeft: "auto", fontSize: 11, color: "#888" }}>
                  {timeAgo(c.created_at)}
                </span>
              </div>
              <div style={{ fontSize: 13 }}>
                {topicLabel(c.topic)} · {c.tenant}
                {c.order_id && <span style={{ fontFamily: "monospace", marginLeft: 6 }}>{c.order_id}</span>}
              </div>
              <div style={{ fontSize: 11, color: "#666", marginTop: 2 }}>
                {c.assigned_to ? `👤 ${c.assigned_to}` : "未认领"}
                {c.user_id && <span style={{ marginLeft: 6 }}>· 用户 {c.user_id}</span>}
              </div>
            </div>
          ))}
        </aside>

        <main style={S.detailPane}>
          {!selectedComplaint && (
            <div style={S.emptyDetail}>← 选择左侧一条工单开始处理</div>
          )}
          {selectedComplaint && (
            <>
              <div style={S.detailHead}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ ...S.sevDot, background: severityColor(selectedComplaint.severity) }}>
                    {severityLabel(selectedComplaint.severity)}
                  </span>
                  <span style={{ fontFamily: "monospace", fontWeight: 600 }}>
                    C-{selectedComplaint.id}
                  </span>
                  <span style={{ color: statusColor(selectedComplaint.status) }}>
                    · {statusLabel(selectedComplaint.status)}
                  </span>
                  <span style={{ color: "#888", fontSize: 12, marginLeft: "auto" }}>
                    {timeAgo(selectedComplaint.created_at)}
                  </span>
                </div>
                <div style={{ fontSize: 13, marginTop: 8, color: "#333" }}>
                  话题: {topicLabel(selectedComplaint.topic)} · 租户: {selectedComplaint.tenant}
                  {selectedComplaint.order_id && (
                    <span style={{ marginLeft: 8, fontFamily: "monospace" }}>
                      订单 {selectedComplaint.order_id}
                    </span>
                  )}
                </div>
                <div style={{ fontSize: 12, color: "#666", marginTop: 4 }}>
                  用户:<code>{selectedComplaint.user_id || "(匿名)"}</code>
                  {" · "}
                  内容摘要(hash): <code>{selectedComplaint.content_hash}</code>
                  {" · 原文不入库(合规)"}
                </div>
                <div style={{ fontSize: 12, color: "#666", marginTop: 4 }}>
                  {selectedComplaint.assigned_to
                    ? <>当前归属:<b>{selectedComplaint.assigned_to}</b></>
                    : <span style={{ color: "#d97706" }}>尚未认领</span>}
                  {selectedComplaint.sla_due_at && (
                    <span style={{ marginLeft: 10 }}>
                      SLA 截止:{new Date(selectedComplaint.sla_due_at).toLocaleString("zh-CN")}
                    </span>
                  )}
                </div>
                <div style={{ marginTop: 10, display: "flex", gap: 8 }}>
                  {selectedComplaint.assigned_to !== identity && (
                    <button onClick={claimSelected} style={S.primaryBtn}>
                      认领到「{identity}」
                    </button>
                  )}
                  {selectedComplaint.assigned_to === identity && (
                    <span style={S.claimedChip}>✓ 归你处理</span>
                  )}
                </div>
              </div>

              <section style={S.replyList}>
                <div style={{ fontSize: 12, color: "#666", marginBottom: 6 }}>
                  回复历史 ({replies.length})
                </div>
                {replies.length === 0 && (
                  <div style={{ color: "#888", fontSize: 12 }}>还没有回复,发第一条吧。</div>
                )}
                {replies.map((r) => {
                  const isUser = r.author_kind === "user";
                  return (
                    <div key={r.id} style={{
                      ...S.replyCard,
                      background: isUser ? "#eff6ff" : "#fff",
                      borderColor: isUser ? "#bfdbfe" : "#e5e7eb",
                      marginLeft: isUser ? 24 : 0,
                      marginRight: isUser ? 0 : 24,
                    }}>
                      <div style={{ fontSize: 11, color: "#666", display: "flex", justifyContent: "space-between" }}>
                        <span>
                          {isUser ? "👤" : "🎧"}{" "}
                          <b>{r.author_label}</b>
                          <span style={{
                            marginLeft: 6,
                            color: isUser ? "#1e3a8a" : "#2563eb",
                            background: isUser ? "#dbeafe" : "#dbeafe",
                            padding: "1px 6px",
                            borderRadius: 4,
                            fontSize: 10,
                          }}>
                            {isUser ? "用户" : "客服"}
                          </span>
                        </span>
                        <span>{timeAgo(r.created_at)}</span>
                      </div>
                      <div style={{ marginTop: 4, fontSize: 14, whiteSpace: "pre-wrap" }}>
                        {r.content}
                      </div>
                    </div>
                  );
                })}
              </section>

              {selectedComplaint.status === "closed" ? (
                <footer style={{
                  ...S.replyBox,
                  background: "#f3f4f6",
                  textAlign: "center",
                  padding: 14,
                }}>
                  <span style={{ fontSize: 13, color: "#6b7280" }}>
                    🔒 用户已主动关闭此工单,无法再向其推送消息。如有遗留问题请联系用户其他渠道。
                  </span>
                </footer>
              ) : (
                <footer style={S.replyBox}>
                  <textarea
                    value={replyText}
                    onChange={(e) => setReplyText(e.target.value)}
                    placeholder="输入给用户的回复内容(发出后会实时推送到用户的聊天窗口)"
                    rows={3}
                    style={S.textarea}
                  />
                  <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
                    <span style={{ fontSize: 11, color: "#888" }}>
                      以 <b>{identity}</b> 的身份回复
                    </span>
                    <button
                      onClick={sendReply}
                      disabled={sending || !replyText.trim()}
                      style={{ ...S.primaryBtn, opacity: sending || !replyText.trim() ? 0.6 : 1 }}
                    >
                      {sending ? "发送中…" : "发送回复 → 推送用户"}
                    </button>
                  </div>
                </footer>
              )}
            </>
          )}
        </main>
      </div>
      <ToastHost />
    </div>
  );
}

// ---- styles ----
const S: Record<string, React.CSSProperties> = {
  page: { display: "flex", flexDirection: "column", height: "100vh", fontFamily: "-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,PingFang SC,sans-serif", background: "#f9fafb" },
  header: { display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 18px", background: "#fff", borderBottom: "1px solid #e5e7eb" },
  counterChip: { fontSize: 12, color: "#555", background: "#f3f4f6", padding: "3px 10px", borderRadius: 12 },
  select: { padding: "5px 8px", borderRadius: 6, border: "1px solid #d1d5db", fontSize: 13 },
  ghostBtn: { padding: "5px 10px", borderRadius: 6, border: "1px solid #d1d5db", background: "#fff", fontSize: 12, cursor: "pointer" },
  body: { display: "grid", gridTemplateColumns: "340px 1fr", flex: 1, minHeight: 0 },
  listPane: { borderRight: "1px solid #e5e7eb", overflowY: "auto", background: "#fff" },
  emptyList: { padding: 20, color: "#888", fontSize: 13 },
  listItem: { padding: "10px 14px", borderBottom: "1px solid #f3f4f6", cursor: "pointer" },
  sevDot: { color: "#fff", fontSize: 11, padding: "1px 8px", borderRadius: 10 },
  detailPane: { display: "flex", flexDirection: "column", padding: 20, overflow: "hidden" },
  emptyDetail: { color: "#888", fontSize: 14 },
  detailHead: { background: "#fff", padding: 14, borderRadius: 8, border: "1px solid #e5e7eb" },
  primaryBtn: { padding: "7px 14px", background: "#2563eb", color: "#fff", border: 0, borderRadius: 6, fontSize: 13, cursor: "pointer" },
  claimedChip: { fontSize: 12, color: "#065f46", background: "#d1fae5", padding: "4px 10px", borderRadius: 6 },
  replyList: { marginTop: 14, flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: 8 },
  replyCard: { background: "#fff", padding: 10, borderRadius: 6, border: "1px solid #e5e7eb" },
  replyBox: { marginTop: 10, background: "#fff", padding: 12, borderRadius: 8, border: "1px solid #e5e7eb" },
  textarea: { width: "100%", padding: 8, fontSize: 14, borderRadius: 6, border: "1px solid #d1d5db", fontFamily: "inherit", resize: "vertical" },
  toastHost: { position: "fixed", bottom: 16, right: 16, display: "flex", flexDirection: "column", gap: 8, zIndex: 1000 },
  toastItem: { padding: "10px 14px", borderRadius: 8, border: "1px solid", fontSize: 13, maxWidth: 380 },
};
