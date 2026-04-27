/**
 * Upload page (?view=upload).
 *
 * The missing link between "I have a PDF" and "I can ask the RAG questions
 * about it". Drag-and-drop or click to pick, type a KB name, click 上传并构建.
 * The page hits POST /kbs/{kb_id}/upload (multipart) — backend saves files,
 * runs ingest_directory + build_indexes, returns chunk count + skipped list.
 * On success we offer a one-click jump to /?view=stream&kb=<name> so the
 * user can immediately ask questions of the freshly indexed corpus.
 *
 * Design choices:
 *   - Single-purpose page. No tabs. No tree views. No per-document
 *     management. If the user wants to delete or re-ingest, they re-upload.
 *   - 8 MB per-file cap mirrors the backend's MAX_BYTES.
 *   - Skipped files (unsupported extension, too big, empty) come back in
 *     the response so the user can see exactly what didn't land.
 */
import { ChangeEvent, DragEvent, FormEvent, useMemo, useRef, useState } from "react";

const API_BASE = (import.meta as any).env?.VITE_API_BASE || "http://127.0.0.1:8008";
const MAX_BYTES_CLIENT_GUARD = 8 * 1024 * 1024;
const SUPPORTED_EXTS = ["pdf", "docx", "pptx", "xlsx", "md", "txt", "html", "htm", "jsonl", "png", "jpg", "jpeg", "webp"];

type UploadResult = {
  kb_id: string;
  chunk_count: number;
  indexed_chunks: number;
  vector_backend: string;
  saved: { filename: string; bytes: number }[];
  skipped: { filename: string; reason: string }[];
};

export default function UploadShell() {
  const [kbId, setKbId] = useState<string>("");
  const [description, setDescription] = useState<string>("");
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [result, setResult] = useState<UploadResult | null>(null);
  const [dragOver, setDragOver] = useState<boolean>(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const totalBytes = useMemo(() => files.reduce((a, f) => a + f.size, 0), [files]);
  const oversize = files.find((f) => f.size > MAX_BYTES_CLIENT_GUARD);
  const validKbId = /^[a-z0-9_]{3,40}$/.test(kbId);

  function appendFiles(incoming: FileList | File[]) {
    const arr = Array.from(incoming);
    setFiles((prev) => {
      const seen = new Set(prev.map((f) => f.name + ":" + f.size));
      return [...prev, ...arr.filter((f) => !seen.has(f.name + ":" + f.size))];
    });
  }

  function onPick(e: ChangeEvent<HTMLInputElement>) {
    if (e.target.files) appendFiles(e.target.files);
    // Reset so the same file can be re-picked
    if (inputRef.current) inputRef.current.value = "";
  }
  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files) appendFiles(e.dataTransfer.files);
  }
  function removeAt(i: number) {
    setFiles((prev) => prev.filter((_, j) => j !== i));
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError("");
    setResult(null);
    if (!validKbId) {
      setError("KB 名只允许小写字母 / 数字 / 下划线,3-40 字符");
      return;
    }
    if (files.length === 0) {
      setError("请先选择至少一个文件");
      return;
    }
    if (oversize) {
      setError(`文件 ${oversize.name} 超过 8 MB 上限`);
      return;
    }

    setBusy(true);
    try {
      const form = new FormData();
      for (const f of files) form.append("files", f);
      form.append("create_if_missing", "true");
      form.append("description", description || "");
      const r = await fetch(`${API_BASE}/kbs/${encodeURIComponent(kbId)}/upload`, {
        method: "POST",
        body: form,
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        const detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data);
        throw new Error(detail);
      }
      setResult(data);
      setFiles([]);
    } catch (err: any) {
      setError(String(err?.message || err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={S.page}>
      <header style={S.head}>
        <a href="/" style={S.backLink}>← 返回首页</a>
        <h1 style={S.title}>📤 上传知识库</h1>
        <p style={S.lede}>
          支持 PDF / DOCX / PPTX / XLSX / Markdown / TXT / HTML / JSONL,以及单图(PNG/JPEG/WebP)。每个文件 ≤ 8 MB。
          上传完成后可立即在 RAG 问答页选中它提问。
        </p>
      </header>

      <form onSubmit={onSubmit} style={S.form}>
        <div style={S.fieldRow}>
          <label style={S.label}>
            <span style={S.labelText}>KB 名称(小写字母/数字/下划线)</span>
            <input
              type="text"
              value={kbId}
              onChange={(e) => setKbId(e.target.value)}
              placeholder="例:my_pdf_demo"
              autoCapitalize="none"
              autoCorrect="off"
              style={{
                ...S.input,
                borderColor: kbId && !validKbId ? "#fca5a5" : "#d1d5db",
              }}
            />
            {kbId && !validKbId && (
              <span style={{ fontSize: 11, color: "#dc2626" }}>
                只允许小写字母 / 数字 / 下划线,长度 3-40
              </span>
            )}
          </label>
          <label style={S.label}>
            <span style={S.labelText}>描述(可选)</span>
            <input
              type="text"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="例:CS4303 课程材料"
              style={S.input}
            />
          </label>
        </div>

        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          style={{
            ...S.dropZone,
            borderColor: dragOver ? "#2563eb" : "#d1d5db",
            background: dragOver ? "#eff6ff" : "#fafbff",
          }}
        >
          <div style={{ fontSize: 36 }}>📄</div>
          <div style={{ fontSize: 14, fontWeight: 500, color: "#374151" }}>
            拖拽文件到这里,或点击选择
          </div>
          <div style={{ fontSize: 12, color: "#6b7280" }}>
            支持的扩展名:{SUPPORTED_EXTS.join(" · ")}
          </div>
          <input
            ref={inputRef}
            type="file"
            multiple
            onChange={onPick}
            style={{ display: "none" }}
            accept={SUPPORTED_EXTS.map((e) => `.${e}`).join(",")}
          />
        </div>

        {files.length > 0 && (
          <div style={S.fileList}>
            <div style={S.fileListHead}>
              已选 {files.length} 个 · 共 {fmtBytes(totalBytes)}
              {oversize && <span style={{ color: "#dc2626", marginLeft: 8 }}>⚠ 有文件超过 8 MB</span>}
            </div>
            {files.map((f, i) => (
              <div key={f.name + i} style={S.fileItem}>
                <span style={{ fontFamily: "monospace", fontSize: 12 }}>{f.name}</span>
                <span style={{ color: "#6b7280", fontSize: 12 }}>{fmtBytes(f.size)}</span>
                <button type="button" onClick={() => removeAt(i)} style={S.removeBtn}>×</button>
              </div>
            ))}
          </div>
        )}

        {error && <div style={S.errorBox}>❌ {error}</div>}

        <div style={S.actionRow}>
          <button
            type="submit"
            disabled={busy || !validKbId || files.length === 0 || !!oversize}
            style={{
              ...S.primaryBtn,
              opacity: busy || !validKbId || files.length === 0 || !!oversize ? 0.5 : 1,
              cursor: busy ? "wait" : "pointer",
            }}
          >
            {busy ? "上传 + 索引中…(可能需要几秒到几分钟)" : "上传并构建知识库"}
          </button>
        </div>
      </form>

      {result && (
        <section style={S.successBox}>
          <div style={{ fontSize: 16, fontWeight: 600, color: "#065f46" }}>
            ✓ 知识库 <code>{result.kb_id}</code> 已索引
          </div>
          <div style={{ marginTop: 8, fontSize: 13, color: "#374151" }}>
            共 <b>{result.chunk_count}</b> 个 chunk,其中 <b>{result.indexed_chunks}</b> 个 leaf 进入向量索引(后端 = {result.vector_backend})。
          </div>
          {result.saved.length > 0 && (
            <details style={{ marginTop: 8 }}>
              <summary style={{ fontSize: 12, color: "#6b7280", cursor: "pointer" }}>
                已保存 {result.saved.length} 个文件
              </summary>
              <ul style={{ marginTop: 4, fontSize: 12, color: "#374151" }}>
                {result.saved.map((s, i) => (
                  <li key={i}>{s.filename} ({fmtBytes(s.bytes)})</li>
                ))}
              </ul>
            </details>
          )}
          {result.skipped.length > 0 && (
            <details style={{ marginTop: 6 }} open>
              <summary style={{ fontSize: 12, color: "#d97706", cursor: "pointer" }}>
                ⚠ 跳过 {result.skipped.length} 个文件
              </summary>
              <ul style={{ marginTop: 4, fontSize: 12, color: "#92400e" }}>
                {result.skipped.map((s, i) => (
                  <li key={i}>{s.filename} — {s.reason}</li>
                ))}
              </ul>
            </details>
          )}
          <div style={{ marginTop: 14, display: "flex", gap: 10 }}>
            <a href={`/?view=stream&kb=${encodeURIComponent(result.kb_id)}`} style={S.successPrimary}>
              💬 立即提问
            </a>
            <a href={`/?view=agent&kb=${encodeURIComponent(result.kb_id)}`} style={S.successSecondary}>
              🤖 让 Agent 用它
            </a>
          </div>
        </section>
      )}
    </div>
  );
}

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(2)} MB`;
}

const S: Record<string, React.CSSProperties> = {
  page: {
    minHeight: "100vh",
    background: "linear-gradient(180deg, #fafbff 0%, #ffffff 100%)",
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'PingFang SC', sans-serif",
    padding: "32px 24px",
    boxSizing: "border-box",
  },
  head: { maxWidth: 760, margin: "0 auto 18px" },
  backLink: { fontSize: 12, color: "#6b7280", textDecoration: "none" },
  title: { fontSize: 26, fontWeight: 700, margin: "8px 0 6px" },
  lede: { fontSize: 14, color: "#4b5563", lineHeight: 1.5, margin: 0 },

  form: {
    maxWidth: 760, margin: "0 auto",
    background: "#fff", padding: 24, borderRadius: 14,
    border: "1px solid #e5e7eb",
    display: "flex", flexDirection: "column", gap: 16,
  },
  fieldRow: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 },
  label: { display: "flex", flexDirection: "column", gap: 4 },
  labelText: { fontSize: 12, color: "#374151", fontWeight: 500 },
  input: {
    padding: "8px 10px",
    borderRadius: 8, border: "1px solid #d1d5db",
    fontSize: 14, fontFamily: "inherit",
    outline: "none",
  },
  dropZone: {
    border: "2px dashed",
    borderRadius: 12,
    padding: "32px 16px",
    textAlign: "center",
    cursor: "pointer",
    display: "flex", flexDirection: "column", gap: 6, alignItems: "center",
    transition: "background 0.15s, border-color 0.15s",
  },
  fileList: { display: "flex", flexDirection: "column", gap: 4 },
  fileListHead: { fontSize: 12, color: "#374151", fontWeight: 500, marginBottom: 4 },
  fileItem: {
    display: "grid", gridTemplateColumns: "1fr auto auto", gap: 10, alignItems: "center",
    padding: "6px 10px", background: "#f9fafb", borderRadius: 6,
  },
  removeBtn: {
    width: 22, height: 22, borderRadius: "50%", border: "1px solid #d1d5db",
    background: "#fff", cursor: "pointer", color: "#6b7280", lineHeight: 1,
  },
  errorBox: {
    padding: 10, borderRadius: 8, background: "#fef2f2", color: "#991b1b",
    border: "1px solid #fecaca", fontSize: 13,
  },
  actionRow: { display: "flex", justifyContent: "flex-end" },
  primaryBtn: {
    padding: "10px 20px",
    background: "#2563eb", color: "#fff",
    border: 0, borderRadius: 8, fontSize: 14, fontWeight: 500,
  },

  successBox: {
    maxWidth: 760, margin: "20px auto 0",
    background: "#ecfdf5", border: "1px solid #a7f3d0",
    borderRadius: 14, padding: 20,
  },
  successPrimary: {
    padding: "8px 14px", background: "#059669", color: "#fff",
    borderRadius: 8, textDecoration: "none", fontSize: 13, fontWeight: 500,
  },
  successSecondary: {
    padding: "8px 14px", background: "#fff", color: "#059669",
    border: "1px solid #a7f3d0", borderRadius: 8, textDecoration: "none", fontSize: 13, fontWeight: 500,
  },
};
