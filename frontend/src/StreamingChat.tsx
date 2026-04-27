/**
 * Minimal streaming chat UI that hits POST /answer/stream.
 *
 * Uses fetch + ReadableStream (EventSource can't POST). Parses SSE events
 * of the form:
 *     event: <name>\n
 *     data: {json}\n
 *     \n
 *
 * Event sequence we render:
 *     meta    -> show the LLM-rewritten query under the bubble
 *     hits    -> seed the citations panel with top-5 previews
 *     delta   -> stream into the assistant bubble
 *     abstain -> render the abstain reason verbatim (no delta)
 *     done    -> finalise citations, latency chip
 *     error   -> swap the bubble to an error chip
 *
 * Intentionally small: one file, no router, no store. Mount via `?view=stream`
 * in main.tsx.
 */
import { FormEvent, useEffect, useRef, useState } from "react";

type Citation = {
  n: number;
  source_id: string;
  title: string;
  source_path?: string;
  section_path?: string[];
  snippet?: string;
};

type KB = { kb_id: string; display_name?: string };

type Message = {
  role: "user" | "assistant";
  text: string;
  rewritten?: string;
  citations?: Citation[];
  hitsPreview?: { n: number; title: string; source_id: string; score: number }[];
  abstain?: { text: string; reason: string };
  latencyMs?: number;
  error?: string;
  streaming?: boolean;
};

const DEFAULT_API_BASE = (import.meta as any).env?.VITE_API_BASE || "http://127.0.0.1:8008";

export default function StreamingChat() {
  const [kbs, setKbs] = useState<KB[]>([]);
  const [kbId, setKbId] = useState<string>("");
  const [apiKey, setApiKey] = useState<string>("");
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [busy, setBusy] = useState(false);
  const endRef = useRef<HTMLDivElement | null>(null);

  // Load KB list on mount.
  useEffect(() => {
    fetch(`${DEFAULT_API_BASE}/kbs`)
      .then((r) => r.json())
      .then((d) => {
        const items: KB[] = (d.items || []).map((k: any) => ({
          kb_id: k.kb_id || k.id,
          display_name: k.display_name || k.kb_id || k.id,
        }));
        setKbs(items);
        if (items.length && !kbId) setKbId(items[0].kb_id);
      })
      .catch(() => setKbs([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!query.trim() || !kbId || busy) return;
    const userMsg: Message = { role: "user", text: query.trim() };
    const assistantMsg: Message = { role: "assistant", text: "", streaming: true };
    setMessages((m) => [...m, userMsg, assistantMsg]);
    const q = query.trim();
    setQuery("");
    setBusy(true);

    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (apiKey.trim()) headers["Authorization"] = `Bearer ${apiKey.trim()}`;

    // Multi-turn: send the last 6 turns as conversation history so
    // ``rewrite_query`` can resolve coreference ("那 PRO 呢?" → "PLUS PRO 多少钱").
    // 6 turns is the same window used by the server-side rewrite_query helper;
    // anything older rarely affects the current pronoun resolution.
    const conversation = messages
      .slice(-6)
      .map((m) => ({ role: m.role, content: m.text }))
      .filter((m) => m.content.trim().length > 0);

    try {
      const resp = await fetch(`${DEFAULT_API_BASE}/answer/stream`, {
        method: "POST",
        headers,
        body: JSON.stringify({ query: q, kb_id: kbId, conversation }),
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
        // SSE frames are separated by double newlines.
        let sep = buf.indexOf("\n\n");
        while (sep !== -1) {
          const frame = buf.slice(0, sep);
          buf = buf.slice(sep + 2);
          applyEvent(parseFrame(frame));
          sep = buf.indexOf("\n\n");
        }
      }
    } catch (err: any) {
      setMessages((all) => mutateLast(all, { error: String(err), streaming: false }));
    } finally {
      setBusy(false);
      setMessages((all) => mutateLast(all, { streaming: false }));
    }
  }

  function applyEvent(ev: { event: string; data: any } | null) {
    if (!ev) return;
    setMessages((all) => {
      const last = all[all.length - 1];
      if (!last || last.role !== "assistant") return all;
      const next = { ...last };
      switch (ev.event) {
        case "meta":
          next.rewritten = ev.data?.rewritten_query;
          break;
        case "hits":
          next.hitsPreview = ev.data?.citations_preview || [];
          break;
        case "delta":
          next.text = (next.text || "") + (ev.data?.text || "");
          break;
        case "abstain":
          next.abstain = { text: ev.data?.text || "", reason: ev.data?.reason || "" };
          next.text = ev.data?.text || next.text;
          break;
        case "done":
          if (Array.isArray(ev.data?.citations)) next.citations = ev.data.citations;
          if (typeof ev.data?.latency_ms === "number") next.latencyMs = ev.data.latency_ms;
          next.streaming = false;
          break;
        case "error":
          next.error = ev.data?.detail || "stream error";
          next.streaming = false;
          break;
      }
      return [...all.slice(0, -1), next];
    });
  }

  return (
    <div style={styles.root}>
      <header style={styles.header}>
        <div style={styles.brand}>💬 RAG Streaming Chat</div>
        <div style={styles.controls}>
          <label style={styles.label}>KB:</label>
          <select value={kbId} onChange={(e) => setKbId(e.target.value)} style={styles.select}>
            {kbs.length === 0 && <option value="">(loading...)</option>}
            {kbs.map((k) => (
              <option key={k.kb_id} value={k.kb_id}>
                {k.display_name || k.kb_id}
              </option>
            ))}
          </select>
          <label style={styles.label}>API key (optional):</label>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="Bearer token"
            style={{ ...styles.select, width: 180 }}
          />
        </div>
      </header>
      <div style={{
        background: "#f5f3ff", borderBottom: "1px solid #ddd6fe",
        padding: "8px 18px", fontSize: 12, color: "#5b21b6",
        display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 8,
      }}>
        <span>
          ℹ️ 这里是<b>纯 RAG 检索</b> — 适合"这门课讲什么 / 这份合同的截止日期是?"这种纯查文档的问题。
        </span>
        <span>
          想让 AI 帮你<b>办事</b>(查单 / 退货 / 投诉) →
          {" "}<a href="/?view=agent" style={{ color: "#7c3aed", textDecoration: "underline" }}>
            切到智能客服 Agent
          </a>
          {" "}|{" "}
          想<b>上传新文档</b>建 KB →
          {" "}<a href="/?view=upload" style={{ color: "#2563eb", textDecoration: "underline" }}>
            上传页
          </a>
        </span>
      </div>

      <main style={styles.main}>
        {messages.length === 0 && (
          <div style={styles.empty}>
            <p>
              Ask a question about <b>{kbId || "(pick a KB)"}</b>. The answer streams in
              token-by-token from <code>/answer/stream</code>.
            </p>
          </div>
        )}
        {messages.map((m, i) => (
          <Bubble key={i} msg={m} />
        ))}
        <div ref={endRef} />
      </main>

      <form onSubmit={onSubmit} style={styles.inputRow}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={busy ? "streaming..." : "Ask a question..."}
          disabled={busy || !kbId}
          style={styles.input}
        />
        <button type="submit" disabled={busy || !query.trim() || !kbId} style={styles.sendBtn}>
          {busy ? "Streaming…" : "Send"}
        </button>
      </form>
    </div>
  );
}

function Bubble({ msg }: { msg: Message }) {
  const isUser = msg.role === "user";
  return (
    <div style={{ ...styles.bubbleRow, justifyContent: isUser ? "flex-end" : "flex-start" }}>
      <div style={{ ...styles.bubble, ...(isUser ? styles.bubbleUser : styles.bubbleAssistant) }}>
        {!isUser && msg.rewritten && msg.rewritten !== msg.text && (
          <div style={styles.meta}>
            <span style={styles.metaLabel}>rewritten →</span> {msg.rewritten}
          </div>
        )}
        <div style={styles.text}>
          {msg.text || (msg.streaming ? <em style={{ color: "#888" }}>thinking...</em> : null)}
          {msg.streaming && <span style={styles.cursor}>▊</span>}
        </div>
        {msg.abstain && (
          <div style={styles.abstain}>abstain: {msg.abstain.reason}</div>
        )}
        {msg.error && <div style={styles.error}>error: {msg.error}</div>}
        {msg.citations && msg.citations.length > 0 && (
          <div style={styles.citations}>
            <div style={styles.citationsHeader}>citations</div>
            {msg.citations.map((c) => (
              <div key={c.n} style={styles.citation}>
                <b>[{c.n}]</b> {c.title}
                <div style={styles.citationPath}>{c.source_id}</div>
                {c.snippet && <div style={styles.snippet}>{c.snippet}</div>}
              </div>
            ))}
          </div>
        )}
        {msg.hitsPreview && msg.hitsPreview.length > 0 && !msg.citations && (
          <div style={styles.hitsPreview}>
            retrieved {msg.hitsPreview.length} hits (streaming answer...)
          </div>
        )}
        {typeof msg.latencyMs === "number" && (
          <div style={styles.latency}>{msg.latencyMs} ms</div>
        )}
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
  try {
    return { event: ev, data: JSON.parse(dataLines.join("\n")) };
  } catch {
    return { event: ev, data: { raw: dataLines.join("\n") } };
  }
}

function mutateLast(all: Message[], patch: Partial<Message>): Message[] {
  if (!all.length) return all;
  const last = { ...all[all.length - 1], ...patch };
  return [...all.slice(0, -1), last];
}

// ---------- inline styles ----------

const styles: Record<string, React.CSSProperties> = {
  root: {
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    fontFamily:
      "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif",
    color: "#1c1c1c",
    background: "#f5f5f7",
  },
  header: {
    padding: "12px 20px",
    background: "#fff",
    borderBottom: "1px solid #e5e5ea",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: 16,
    flexWrap: "wrap",
  },
  brand: { fontWeight: 600, fontSize: 16 },
  controls: { display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" },
  label: { fontSize: 12, color: "#555" },
  select: {
    padding: "4px 8px",
    fontSize: 13,
    border: "1px solid #d2d2d7",
    borderRadius: 6,
    background: "#fff",
  },
  main: {
    flex: 1,
    overflowY: "auto",
    padding: "16px 20px",
    display: "flex",
    flexDirection: "column",
    gap: 12,
  },
  empty: { textAlign: "center", color: "#888", marginTop: 48 },
  bubbleRow: { display: "flex", width: "100%" },
  bubble: {
    maxWidth: "min(640px, 85%)",
    padding: "10px 14px",
    borderRadius: 16,
    fontSize: 14,
    lineHeight: 1.5,
    boxShadow: "0 1px 2px rgba(0,0,0,0.05)",
  },
  bubbleUser: { background: "#0a84ff", color: "#fff" },
  bubbleAssistant: { background: "#fff", color: "#111", border: "1px solid #e5e5ea" },
  meta: { fontSize: 11, color: "#888", marginBottom: 6 },
  metaLabel: { color: "#aaa", marginRight: 4 },
  text: { whiteSpace: "pre-wrap" },
  cursor: { opacity: 0.5, marginLeft: 2 },
  abstain: { marginTop: 8, fontSize: 12, color: "#c93" },
  error: { marginTop: 8, fontSize: 12, color: "#c33" },
  citations: { marginTop: 10, borderTop: "1px solid #eee", paddingTop: 8 },
  citationsHeader: { fontSize: 11, color: "#888", marginBottom: 4 },
  citation: { fontSize: 12, marginBottom: 8 },
  citationPath: { fontSize: 11, color: "#888" },
  snippet: { fontSize: 12, color: "#555", marginTop: 2 },
  hitsPreview: { marginTop: 8, fontSize: 11, color: "#888" },
  latency: { marginTop: 6, fontSize: 11, color: "#aaa" },
  inputRow: {
    padding: "12px 20px",
    background: "#fff",
    borderTop: "1px solid #e5e5ea",
    display: "flex",
    gap: 8,
  },
  input: {
    flex: 1,
    padding: "10px 14px",
    fontSize: 14,
    border: "1px solid #d2d2d7",
    borderRadius: 10,
    outline: "none",
  },
  sendBtn: {
    padding: "10px 18px",
    fontSize: 14,
    background: "#0a84ff",
    color: "#fff",
    border: "none",
    borderRadius: 10,
    cursor: "pointer",
    fontWeight: 500,
  },
};
