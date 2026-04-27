/**
 * Landing page (the new "/").
 *
 * Replaces App.tsx, which was a 3379-line carry-over from a prior
 * "platform admin" iteration whose write-side endpoints the current
 * backend never implemented. The simpler answer: a clean four-card
 * homepage that fans out to the four working entry points:
 *
 *   📤 Upload & Build KB   →  /?view=upload
 *   💬 RAG Q&A             →  /?view=stream
 *   🤖 Customer Service    →  /?view=agent
 *   🎧 Admin Console       →  /?view=admin
 *
 * Plus a small "system status" strip so the user can immediately tell
 * the backend is reachable and which KBs are indexed.
 */
import { useEffect, useState } from "react";

const API_BASE = (import.meta as any).env?.VITE_API_BASE || "http://127.0.0.1:8008";

type KBSummary = { kb_id: string; description?: string; chunk_count?: number };

// The hero card is Agent — it's the orchestrator and the most demo-able.
// The 3 supporting cards are the layers Agent stands on (RAG / KB build /
// human handoff). The visual hierarchy makes "Agent uses these as tools"
// obvious without needing to narrate it.
const HERO = {
  href: "/?view=agent",
  icon: "🤖",
  title: "智能客服 Agent",
  subtitle: "LangGraph Plan-and-Execute · 9 specialists",
  blurb:
    "主入口。Agent 把 RAG / 数据库 / 多模态 / MCP tools 编排成多步工作流:" +
    "订单 / 物流 / 退货 / 投诉 / 发票 / 地址 / 推荐 / 商品 QA / 政策 QA。",
  accent: "#059669",
  bg: "#ecfdf5",
};

const SUPPORTING = [
  {
    href: "/?view=stream",
    icon: "💬",
    title: "RAG 知识问答",
    subtitle: "Agent 内部就在调用它 · 也可独立使用",
    blurb: "BM25 + BGE-M3 向量 + RRF + 重排 + 引用 + abstain。Agent 的 product_qa / policy_qa specialist 调的就是这个。",
    accent: "#7c3aed",
    bg: "#f5f3ff",
  },
  {
    href: "/?view=upload",
    icon: "📤",
    title: "上传知识库",
    subtitle: "给 RAG 喂数据",
    blurb: "拖拽 PDF / DOCX / MD / TXT,自动切块、嵌入、建索引。完成后 RAG 和 Agent 立刻能用。",
    accent: "#2563eb",
    bg: "#eff6ff",
  },
  {
    href: "/?view=admin",
    icon: "🎧",
    title: "客服后台",
    subtitle: "Agent 处理不了 → 转人工",
    blurb: "AI 升级后的工单出现在这里;客服回复经 SSE 实时推送回用户聊天页面。",
    accent: "#dc2626",
    bg: "#fef2f2",
  },
];

export default function Landing() {
  const [kbs, setKbs] = useState<KBSummary[]>([]);
  const [healthy, setHealthy] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [hRes, kRes] = await Promise.all([
          fetch(`${API_BASE}/health`),
          fetch(`${API_BASE}/kbs`),
        ]);
        if (cancelled) return;
        setHealthy(hRes.ok);
        if (kRes.ok) {
          const data = await kRes.json();
          setKbs(data.items || []);
        }
      } catch {
        if (!cancelled) setHealthy(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  return (
    <div style={S.page}>
      <header style={S.hero}>
        <div style={S.titleRow}>
          <h1 style={S.h1}>
            <span style={{ color: "#2563eb" }}>Enterprise</span>{" "}
            <span style={{ color: "#7c3aed" }}>Multimodal RAG</span>{" "}
            <span style={{ color: "#059669" }}>+ Agent</span>
          </h1>
          <span style={{
            ...S.statusPill,
            color: healthy === null ? "#888" : healthy ? "#059669" : "#dc2626",
            background: healthy === null ? "#f3f4f6" : healthy ? "#ecfdf5" : "#fef2f2",
            borderColor: healthy === null ? "#e5e7eb" : healthy ? "#a7f3d0" : "#fecaca",
          }}>
            {healthy === null ? "检测中…" : healthy ? "● 后端在线" : "● 后端离线"}
          </span>
        </div>
        <p style={S.subtitle}>
          多语言 RAG（BM25 + BGE-M3 + RRF + 重排）· LangGraph 多智能体 · 多模态 · 实时 SSE 推送 · 端到端审计
        </p>
        {kbs.length > 0 && (
          <div style={S.kbStrip}>
            <span style={{ fontSize: 12, color: "#6b7280" }}>已索引知识库 ({kbs.length}):</span>
            {kbs.map((k) => (
              <span key={k.kb_id} style={S.kbChip} title={k.description || ""}>
                {k.kb_id}{typeof k.chunk_count === "number" ? ` · ${k.chunk_count}` : ""}
              </span>
            ))}
          </div>
        )}
      </header>

      <main style={{ maxWidth: 1100, margin: "0 auto" }}>
        {/* Hero: Agent — the orchestrator */}
        <a href={HERO.href} style={{
          ...S.card,
          background: HERO.bg,
          borderColor: `${HERO.accent}33`,
          padding: 26,
          marginBottom: 18,
          display: "grid",
          gridTemplateColumns: "auto 1fr auto",
          gap: 18,
          alignItems: "center",
          textDecoration: "none",
          color: "inherit",
          boxShadow: `0 1px 0 ${HERO.accent}11`,
        }}>
          <div style={{
            ...S.cardIcon,
            background: `${HERO.accent}1a`,
            color: HERO.accent,
            width: 56, height: 56, fontSize: 28, borderRadius: 12,
          }}>
            {HERO.icon}
          </div>
          <div>
            <div style={{ ...S.cardTitle, color: HERO.accent, fontSize: 18 }}>
              {HERO.title}
              <span style={{
                marginLeft: 8, fontSize: 10, padding: "2px 8px", borderRadius: 8,
                background: HERO.accent, color: "#fff", letterSpacing: "0.5px",
                verticalAlign: "middle",
              }}>主入口</span>
            </div>
            <div style={S.cardSubtitle}>{HERO.subtitle}</div>
            <div style={{ ...S.cardBlurb, fontSize: 14 }}>{HERO.blurb}</div>
          </div>
          <div style={{ ...S.cardArrow, color: HERO.accent, fontSize: 28 }}>→</div>
        </a>

        {/* Layer label */}
        <div style={{
          display: "flex", alignItems: "center", gap: 10,
          margin: "8px 4px", color: "#6b7280", fontSize: 12,
        }}>
          <span style={{ flex: 1, height: 1, background: "#e5e7eb" }} />
          <span>Agent 调用 / 用户也可单独使用的能力</span>
          <span style={{ flex: 1, height: 1, background: "#e5e7eb" }} />
        </div>

        {/* Supporting cards: RAG / Upload / Admin */}
        <div style={S.cardsGrid}>
          {SUPPORTING.map((c) => (
            <a key={c.href} href={c.href} style={{
              ...S.card,
              background: c.bg,
              borderColor: `${c.accent}22`,
            }}>
              <div style={{ ...S.cardIcon, background: `${c.accent}11`, color: c.accent }}>
                {c.icon}
              </div>
              <div>
                <div style={{ ...S.cardTitle, color: c.accent }}>{c.title}</div>
                <div style={S.cardSubtitle}>{c.subtitle}</div>
                <div style={S.cardBlurb}>{c.blurb}</div>
              </div>
              <div style={{ ...S.cardArrow, color: c.accent }}>→</div>
            </a>
          ))}
        </div>
      </main>

      <footer style={S.footer}>
        <div>
          <strong>Stack:</strong> FastAPI · LangGraph · BGE-M3 · Qdrant / FAISS / PGVector · Qwen-VL · Prometheus · OpenTelemetry
        </div>
        <div style={{ marginTop: 4, color: "#9ca3af" }}>
          API: <code>{API_BASE}</code> · 前端: Vite + React
        </div>
      </footer>
    </div>
  );
}

const S: Record<string, React.CSSProperties> = {
  page: {
    minHeight: "100vh",
    background: "linear-gradient(180deg, #fafbff 0%, #ffffff 100%)",
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'PingFang SC', sans-serif",
    padding: "48px 32px",
    boxSizing: "border-box",
  },
  hero: { maxWidth: 1100, margin: "0 auto 32px", padding: "8px 4px" },
  titleRow: { display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12 },
  h1: { fontSize: 32, fontWeight: 700, margin: 0, letterSpacing: "-0.5px" },
  statusPill: { fontSize: 12, padding: "5px 12px", borderRadius: 14, border: "1px solid", fontWeight: 500 },
  subtitle: { fontSize: 15, color: "#4b5563", margin: "10px 0 14px", maxWidth: 900, lineHeight: 1.5 },
  kbStrip: { display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center" },
  kbChip: { fontSize: 11, fontFamily: "monospace", background: "#f3f4f6", color: "#374151", padding: "3px 9px", borderRadius: 8 },

  cardsGrid: {
    maxWidth: 1100,
    margin: "0 auto",
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
    gap: 20,
  },
  card: {
    display: "grid",
    gridTemplateColumns: "auto 1fr auto",
    gap: 16,
    padding: 22,
    borderRadius: 14,
    border: "1px solid",
    textDecoration: "none",
    color: "inherit",
    transition: "transform 0.15s ease, box-shadow 0.15s ease",
    cursor: "pointer",
    alignItems: "center",
  },
  cardIcon: {
    width: 44, height: 44, borderRadius: 10,
    display: "flex", alignItems: "center", justifyContent: "center",
    fontSize: 22, fontWeight: 600,
  },
  cardTitle: { fontSize: 16, fontWeight: 600, marginBottom: 2 },
  cardSubtitle: { fontSize: 11, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 6 },
  cardBlurb: { fontSize: 13, color: "#374151", lineHeight: 1.5 },
  cardArrow: { fontSize: 22, fontWeight: 300 },

  footer: {
    maxWidth: 1100, margin: "48px auto 0", padding: "18px 4px",
    fontSize: 12, color: "#6b7280", borderTop: "1px solid #e5e7eb", paddingTop: 18,
  },
};
