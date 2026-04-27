/**
 * SessionSidebar — ChatGPT-style left panel listing the user's past chat
 * sessions. Click a row to switch the active thread_id; on switch, AgentChat
 * reloads its persisted turns from localStorage and re-fetches the inbox
 * scoped to that session.
 *
 * Sessions are auto-created server-side on the first message of a fresh
 * thread_id (see ``touch_session`` in the agent router); we just render
 * them here and emit `onSelect` / `onDelete` / `onRename` events back to
 * the parent.
 */
import { useEffect, useState } from "react";

const API_BASE = (import.meta as any).env?.VITE_API_BASE || "http://127.0.0.1:8008";

export type Session = {
  thread_id: string;
  user_id: string;
  tenant: string;
  title: string;
  first_msg_at: string | null;
  last_msg_at: string | null;
};

type Props = {
  userId: string;
  currentThreadId: string;
  onSelect: (threadId: string) => void;
  onNewSession: () => void;
  // Bumps when the parent wants the sidebar to re-fetch (e.g. after
  // agent_chat just touched a new session, or after a delete).
  refreshSignal: number;
};

function formatTime(iso: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  const now = new Date();
  const sameDay =
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate();
  if (sameDay) return d.toLocaleTimeString("zh-CN", { hour12: false });
  const yest = new Date(now); yest.setDate(now.getDate() - 1);
  if (
    d.getFullYear() === yest.getFullYear() &&
    d.getMonth() === yest.getMonth() &&
    d.getDate() === yest.getDate()
  ) return `昨天 ${d.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", hour12: false })}`;
  return d.toLocaleString("zh-CN", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false });
}

export default function SessionSidebar({
  userId,
  currentThreadId,
  onSelect,
  onNewSession,
  refreshSignal,
}: Props) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(false);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");

  async function refresh() {
    setLoading(true);
    try {
      const r = await fetch(`${API_BASE}/users/${encodeURIComponent(userId)}/sessions`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      setSessions(data.items || []);
    } catch {
      // sidebar is read-only / non-critical; don't crash the app
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId, refreshSignal]);

  async function handleDelete(threadId: string, title: string) {
    if (!confirm(`删除会话「${title}」?\n\n• 聊天记录会丢失\n• 该会话提交的工单不会删除,只是不再属于这个会话`)) return;
    try {
      const r = await fetch(`${API_BASE}/sessions/${encodeURIComponent(threadId)}`, {
        method: "DELETE",
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      // Also clear the session's persisted turns from localStorage.
      try {
        localStorage.removeItem(`agent_chat_turns_${threadId}`);
      } catch {}
      refresh();
      // If the deleted session was the active one, fall back to a new session.
      if (threadId === currentThreadId) onNewSession();
    } catch (e: any) {
      alert(`删除失败:${String(e)}`);
    }
  }

  async function handleRename(threadId: string) {
    const title = renameDraft.trim();
    if (!title) {
      setRenaming(null);
      return;
    }
    try {
      const r = await fetch(`${API_BASE}/sessions/${encodeURIComponent(threadId)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setRenaming(null);
      setRenameDraft("");
      refresh();
    } catch (e: any) {
      alert(`重命名失败:${String(e)}`);
    }
  }

  return (
    <aside style={S.aside}>
      <div style={S.header}>
        <div style={{ fontSize: 13, fontWeight: 600, color: "#111" }}>会话</div>
        <button onClick={onNewSession} style={S.newBtn} title="开始一个新的聊天会话">
          ＋ 新会话
        </button>
      </div>

      <div style={S.list}>
        {loading && sessions.length === 0 && (
          <div style={S.empty}>加载中…</div>
        )}
        {!loading && sessions.length === 0 && (
          <div style={S.empty}>
            还没有会话。点击「＋ 新会话」开始第一段对话。
          </div>
        )}
        {sessions.map((s) => {
          const active = s.thread_id === currentThreadId;
          const isRenaming = renaming === s.thread_id;
          return (
            <div
              key={s.thread_id}
              style={{
                ...S.row,
                background: active ? "#eff6ff" : "transparent",
                borderLeft: active ? "3px solid #2563eb" : "3px solid transparent",
              }}
              onClick={() => !isRenaming && onSelect(s.thread_id)}
            >
              <div style={{ flex: 1, minWidth: 0 }}>
                {isRenaming ? (
                  <input
                    autoFocus
                    value={renameDraft}
                    onChange={(e) => setRenameDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") handleRename(s.thread_id);
                      else if (e.key === "Escape") { setRenaming(null); setRenameDraft(""); }
                    }}
                    onBlur={() => handleRename(s.thread_id)}
                    style={S.renameInput}
                  />
                ) : (
                  <div style={S.title} title={s.title}>{s.title}</div>
                )}
                <div style={S.timeline}>
                  <span style={S.tenantChip}>{s.tenant.toUpperCase()}</span>
                  <span style={{ color: "#6b7280" }}>{formatTime(s.last_msg_at)}</span>
                </div>
              </div>
              {!isRenaming && (
                <div style={S.rowActions} onClick={(e) => e.stopPropagation()}>
                  <button
                    style={S.iconBtn}
                    title="重命名"
                    onClick={() => { setRenaming(s.thread_id); setRenameDraft(s.title); }}
                  >✎</button>
                  <button
                    style={S.iconBtn}
                    title="删除会话"
                    onClick={() => handleDelete(s.thread_id, s.title)}
                  >🗑</button>
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div style={S.footer}>
        <a href="/" style={{ color: "#6b7280", fontSize: 11, textDecoration: "none" }}>
          ← 返回首页
        </a>
      </div>
    </aside>
  );
}

const S: Record<string, React.CSSProperties> = {
  aside: {
    width: 240,
    minWidth: 240,
    borderRight: "1px solid #e5e7eb",
    background: "#f9fafb",
    display: "flex",
    flexDirection: "column",
    height: "100%",
  },
  header: {
    padding: "12px 14px",
    borderBottom: "1px solid #e5e7eb",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 8,
  },
  newBtn: {
    fontSize: 11,
    padding: "5px 10px",
    background: "#2563eb",
    color: "#fff",
    border: 0,
    borderRadius: 6,
    cursor: "pointer",
    fontWeight: 500,
  },
  list: {
    flex: 1,
    overflowY: "auto",
    paddingTop: 4,
  },
  empty: {
    padding: 16,
    color: "#888",
    fontSize: 12,
    lineHeight: 1.5,
  },
  row: {
    display: "flex",
    alignItems: "flex-start",
    gap: 6,
    padding: "8px 10px 8px 8px",
    borderBottom: "1px solid #f3f4f6",
    cursor: "pointer",
    transition: "background 0.1s ease",
  },
  title: {
    fontSize: 13,
    color: "#111",
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
    marginBottom: 3,
  },
  timeline: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    fontSize: 10,
  },
  tenantChip: {
    background: "#e5e7eb",
    color: "#374151",
    padding: "1px 5px",
    borderRadius: 4,
    fontFamily: "monospace",
  },
  rowActions: {
    display: "flex",
    gap: 2,
    opacity: 0.6,
  },
  iconBtn: {
    width: 20,
    height: 20,
    border: 0,
    background: "transparent",
    cursor: "pointer",
    fontSize: 12,
    color: "#6b7280",
    padding: 0,
  },
  renameInput: {
    width: "100%",
    fontSize: 13,
    padding: "2px 4px",
    border: "1px solid #2563eb",
    borderRadius: 4,
    fontFamily: "inherit",
    outline: "none",
  },
  footer: {
    padding: "10px 14px",
    borderTop: "1px solid #e5e7eb",
  },
};
