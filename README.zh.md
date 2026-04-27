# 企业级多语言 RAG + 智能客服 Agent 平台

**🌐 Languages**: [English](./README.md) · 中文(本页)

[![Tests](https://img.shields.io/badge/tests-521_passed-brightgreen)](#)
[![Hit@5](https://img.shields.io/badge/Hit%405-1.000-brightgreen)](#)
[![MRR@10](https://img.shields.io/badge/MRR%4010-0.878-brightgreen)](#)
[![Eval](https://img.shields.io/badge/eval-84_rows_/_4_categories-blue)](#)
[![CI](https://img.shields.io/badge/CI-pytest_%2B_regression_gate-informational)](#)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#)

[![Python](https://img.shields.io/badge/Python-3.12-blue)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-009688)](#)
[![LangGraph](https://img.shields.io/badge/LangGraph-multi--agent-purple)](#)
[![BGE-M3](https://img.shields.io/badge/embedding-BGE--M3-orange)](#)
[![Qwen-VL](https://img.shields.io/badge/VLM-Qwen--VL-yellow)](#)
[![Postgres](https://img.shields.io/badge/Postgres-pgvector-336791)](#)
[![Qdrant](https://img.shields.io/badge/Qdrant-vector_db-DC382D)](#)
[![FAISS](https://img.shields.io/badge/FAISS-flat--IP-005571)](#)
[![Prometheus](https://img.shields.io/badge/Prometheus-metrics-E6522C)](#)
[![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-tracing-425CC7)](#)
[![React](https://img.shields.io/badge/React-Vite-61DAFB)](#)

一个**生产级倾向**的多语言(中英)RAG 平台,顶上叠加了一个**基于 LangGraph 的 9 节点客服 Agent**——支持完整会话管理、用户↔客服 SSE 实时双向推送、多模态文档摄入、端到端审计。

> 同时支持纯 RAG 知识问答 + 任务型 Agent 客服。**Agent 把 RAG 当工具用**,
> 不是两个独立系统 —— 这是平台型 AI 产品的标准形态。

---

## 0. 这个项目是什么,为什么做

**这是一个个人作品集项目**,目标是展示对**现代 LLM 应用的端到端工程能力**——不是玩具聊天机器人,也不是抄教程的实现。**目标场景:多租户电商客服**——用户从 "运费政策怎么算" 到 "我要把订单 #JD20260420456 投诉到 12315",平台都能回答、能办事、能跨会话持久化状态。

### 四大能力支柱(每个都有自己的架构思考)

| | 干什么 | 关键技术 |
|---|---|---|
| **🤖 Agent** | 多步业务任务规划 + 派发到 9 个 specialist 节点 | LangGraph StateGraph · Plan-and-Execute · 4 层多轮记忆 · AsyncSqliteSaver |
| **🔍 RAG** | 中英双语混合检索,质量可量化、可回归测试 | BM25(FTS5 unicode61)+ BGE-M3 + RRF + bge-reranker + **Anthropic 2024 Contextual Retrieval** |
| **🖼 多模态** | 图文互搜 + 版面感知的 PPT / 图表摄入 | **CLIP** ViT-B-32(512 维图文同空间)+ **Qwen-VL** 版面切分(标题 / 正文 / 表格 / 图形 / 代码) |
| **🔌 MCP** | 把项目能力以标准协议形式暴露给外部工具 | **Model Context Protocol**(Anthropic 2024 stdio)· 7 个 Tools + 2 个 Resources + 3 个 Prompts · 可被 Claude Desktop / Cursor 直接接入 |

**质量靠测,不靠吹**。84 行人工标注 eval 集,CI 自动阻断 Hit@5 / MRR 跌幅超 0.02 的 PR。**521 单元测试**覆盖 DB / API / SSE 总线 / Agent 流程 / RAG 检索 / 多轮一致性。

### 多轮对话做对了(3 层修复,都是真实测试中发现的 bug)

完整故事见 [第 10 节](#10-差异化亮点-不是玩具的理由)。简版:

- **改写器 recency bias** — "翻译那个" 锚定到**最近一轮 assistant**,而不是最早的 user 问题
- **意图分流路由器** — 元问题(如 "刚刚翻译的是什么")**跳过 RAG 检索**,直接从对话历史回答,修复了 naive RAG 常见的 "幻觉编造任意 chunks 摘要" 失败模式
- **答题器看历史(分层)** — LLM 看到最近 2 轮历史**仅作为指代理解上下文**,system prompt 硬规则要求事实必须来自 `[n]` 引用的 chunks,不会被旧答案污染

---

## 1. 架构总览

```
┌──────────────────────────────────────────────────────────────────────────┐
│                  Vite + React SPA(单 bundle,4 个路由)                    │
│   /            ?view=upload       ?view=stream    ?view=agent   ?view=admin │
│  Landing       拖拽建 KB         RAG 流式问答    9-specialist  客服后台    │
│   │                │                  │              │            │      │
│   └─────────┬──────┴──────┬───────────┴──────┬───────┴────┬───────┘      │
│             ▼             ▼                  ▼            ▼              │
└─────────────│─────────────│──────────────────│────────────│──────────────┘
              │             │                  │            │
              │   POST /kbs/{id}/upload        SSE: /agent/chat / /users/{id}/events
              │   POST /answer/stream                                       │
              ▼             ▼                  ▼            ▼              │
┌──────────────────────────────────────────────────────────────────────────┐
│                          FastAPI(async)端口 8008                         │
│  ┌─────────────────────────┐    ┌─────────────────────────────────────┐  │
│  │  RAG pipeline           │    │  Agent(LangGraph Plan-and-Execute) │  │
│  │  ─────────────          │    │  ─────────────────────────────────  │  │
│  │  1. NFKC 归一化         │    │   planner ─► advance ─┐             │  │
│  │  2. LLM 跨语言改写       │    │                       ▼             │  │
│  │  3. 混合检索             │    │   ┌──── 9 specialists ───┐          │  │
│  │     • BM25(FTS5 CJK)   │◀───┤   │ product_qa  ──┐      │          │  │
│  │     • Dense BGE-M3      │    │   │ policy_qa   ──┼─→ RAG│ ◀────┐   │  │
│  │     • RRF + 交叉重排     │    │   │ order        │       │     │   │  │
│  │  4. MMR 多样化           │    │   │ logistics   ─┘       │     │   │  │
│  │  5. Grounded LLM         │    │   │ aftersale   ─→ DB    │     │   │  │
│  │     [n] 引用             │    │   │ recommend   ─→ vec   │     │   │  │
│  │  6. 弃答(低分)         │    │   │ invoice     ─→ DB+PDF│     │   │  │
│  │  7. 幻觉校验(qwen-turbo)│    │   │ complaint   ─→ DB+SSE│     │   │  │
│  │                         │    │   │ account     ─→ DB+SMS│     │   │  │
│  │                         │    │   └──────────────────────┘     │   │  │
│  │                         │    │       ▲                        │   │  │
│  │                         │    │  4 层 Agent 记忆:              │   │  │
│  │                         │    │  • messages                     │   │  │
│  │                         │    │  • entities(last_order_id...) │   │  │
│  │                         │    │  • LangGraph AsyncSqliteSaver  │   │  │
│  │                         │    │    checkpoint(per thread_id)  │   │  │
│  │                         │    │  • UserPreference(永久跨会话)  │   │  │
│  └─────────────────────────┘    └─────────────────────────────────────┘  │
│                                                                          │
│  多模态 · 鉴权 · 限流 · token 预算 · 审计(仅哈希)                         │
│  Prometheus /metrics · OpenTelemetry traces · MCP server(7 tools)        │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 关键指标

| 指标 | 数值 | 怎么测 |
|---|---|---|
| 检索 Hit@5(84 行 eval) | **1.000** rewrite=on / **0.942** rewrite=off | BM25 + BGE-M3 + RRF + bge-reranker + MMR |
| 检索 MRR@10 | **0.878** rewrite=on | 同上 |
| Multi-hop bucket MRR | **0.971** | Contextual Retrieval 文档级前缀 |
| 答案忠实度(LLM-as-judge) | **0.90+** | Grounded prompt + 弃答策略 |
| 测试数 | **521 通过**, 4 跳过, 1 排除 | `pytest -m "not integration"` |
| 向量后端 | FAISS + Qdrant + **PGVector** | 统一 `VectorStore` ABC 接口 |
| Specialist 节点 | **9 个**(Plan-and-Execute) | product_qa / policy_qa / order / logistics / aftersale / recommend / invoice / complaint / account |
| 前端 bundle | 226 KB / 71 KB gzip | Vite + React, 5 路由 |

---

## 3. Quick Start — 自己跑起来(5 分钟)

> **🔑 给 fork 这个仓库的人**:本项目调用 **Qwen(阿里云 DashScope)** 作为 LLM。
> 你需要**自己的** `DASHSCOPE_API_KEY` —— 在 https://dashscope.aliyun.com/ 免费申请。
> **本仓库不包含任何 API key**(`.env.local` 已被 `.gitignore` 排除)。

### Option A — 本地 Python(推荐评估用)

```bash
# 1. clone + 安装
git clone <repo> && cd enterprise-multimodal-copilot
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements-api.txt

# 2. 配置你的 DashScope key(不会被提交,只在本地)
cp .env.example .env.local
#   然后编辑 .env.local 填:DASHSCOPE_API_KEY=sk-your-key-here

# 3. 数据库迁移 + 注入 demo 数据(2 个 mock 用户、6 个订单等)
./.venv/bin/alembic upgrade head
./.venv/bin/python scripts/seed_db.py

# 4. 启动后端(端口 8008)
./.venv/bin/python scripts/run_api.py &

# 5. 启动前端(端口 5714)
cd frontend && npm install && npm run dev
```

### Option B — Docker 一键启动

```bash
cp .env.example .env.local && 编辑 .env.local 填 DASHSCOPE_API_KEY
docker build -t rag-agent .
docker run --rm -p 8008:8008 --env-file .env.local rag-agent
# 前端单独跑:cd frontend && npm install && npm run dev
```

Dockerfile 启动期会自动跑 `alembic upgrade head`,容器起来 schema 就同步好了。

打开 `http://127.0.0.1:5714/` 即可看到 4 卡片 Landing 页。

### 5 分钟一键 demo 路径

1. 点 📤 **上传知识库** → 拖一个 PDF → KB 构建
2. 点 💬 **RAG 问答** → 提问 → 带 `[n]` 引用的 grounded 回答
3. 点 🤖 **智能客服** → "查我最近的 iPhone 订单" → 多步规划 + 卡片化展示
4. 同一个会话:"我要去 12315 投诉" → 预览卡片 → 点提交工单
5. 另开 🎧 **客服后台** tab → 认领 + 回复 → 用户聊天页 SSE 实时收到回复

---

## 4. 能力清单

### 核心 RAG
- ✅ **多语言**(中 + 英)摄入 + 问答
- ✅ **混合检索** — BM25(FTS5 CJK 分词) + 稠密 BGE-M3 + RRF
- ✅ **交叉重排**(BGE-reranker)
- ✅ **MMR 多样化**(Jaccard shingles)
- ✅ **跨语言改写** + 多 query 扩展(可选)
- ✅ **对话感知的指代消解**("那 PRO 呢" → 自动绑定 "PLUS PRO")
- ✅ **Parent-child 分块**(small-to-big 检索)
- ✅ **Contextual Retrieval**(Anthropic 2024,文档级前缀)
- ✅ **Grounded 答案** + 内联 `[n]` 引用 + 双层弃答
- ✅ **在线幻觉校验**(可选,qwen-turbo 逐句校验是否 grounded)
- ✅ **token 级流式**(`/answer/stream` SSE)
- ✅ **3 级 Fallback 链**(Qwen → Claude → 抽取式兜底)

### 向量后端(单 ABC 接口,3 个实现)
- FAISS flat-IP(零依赖,单文件)
- Qdrant(in-memory / 本地文件 / HTTP 远程)
- **PGVector**(Postgres 扩展,共用现有 DB)

### 文档摄入
- PDF(PyMuPDF)、DOCX、PPTX、XLSX、MD、TXT、HTML、JSONL
- 图片文件(PNG / JPG / WebP)→ Qwen-VL 描述 + OCR
- HTML 正文提取(readability-lxml,去模板)
- 摄入期 PII 检测脱敏(中国手机号 / 身份证 / 银行卡 Luhn / 邮箱 / IP)
- MinHash 去重(0.85 阈值)
- Metadata 增强(LLM 抽取实体 / 主题 / 日期,可选)
- 语义切块(Greg Kamradt 嵌入距离算法,可选)
- **连接器**:local_dir、http_page、Notion(REST API)
- **增量摄入**(content-hash diff,逐文档增 / 改 / 跳 / 删)

### 多模态
- Qwen-VL 描述 + OCR + 版面分析(标题 / 正文 / 表格 / 图形 / 代码区域)
- 多图输入(单次最多 5 张)
- 图像嵌入(CLIP ViT-B-32,512 维图文同空间)
- `/vision/search` — 用图搜知识库

### Agent(LangGraph)
- **Plan-and-Execute** + 9 个 specialist 节点
- **原生 function calling 循环**(OrderAgent v2)
- **MCP server**(7 tools / 2 resources / 3 prompts,stdio,Claude Desktop 兼容)
- **4 层记忆**:
  1. 当轮 `messages`
  2. 当轮 `entities`(last_order_id / last_item_title / ...)
  3. 跨会话 **AsyncSqliteSaver** checkpoint,按 `thread_id` 索引
  4. **永久 `user_preferences`**(durable,跨会话,Planner prompt 自动注入)
- **可逆状态机**:complaint 和 return_request 都支持 cancel ↔ reopen,严重度感知地恢复活跃状态

### 实时双向客服闭环
- **客服后台**(`?view=admin`)
- 工单队列 + 认领 + 带身份的回复
- **SSE 推送** 到用户的 Agent 聊天页(进程内总线,可换 Redis pub/sub)
- 用户侧也支持回复(用户和客服在同一个工单卡内)
- 选中工单自动 4s 轮询新消息

### 安全 / 治理
- **Prompt 注入防御**(中英 13 个 pattern)
- **输出 PII 脱敏**(`sk-...`、私钥、system-prompt 泄漏)
- **审计日志** sha256[:16] 仅哈希(原始 query / answer 永不入审计)
- **GDPR 删除**(级联订单 / 订单项 / 退货 / 反馈 / 审计)
- **保留清理 cron**(审计 180d / 反馈 365d / 已终态退货 365d)

### 运维 / 可观测性
- Prometheus `/metrics`(5 个指标族,低基数标签)
- OpenTelemetry SDK + OTLP 导出器(env 开关)
- 6 面板 Grafana 看板
- API key 鉴权(sha256 哈希,租户级滑动窗口限流)
- per-request + per-tenant token 预算
- Locust 压测脚手架

### 前端
- Vite + React,**单 SPA,5 路由**:
  - `/` — Landing(英雄卡 + 3 张副卡)
  - `?view=upload` — 拖拽上传 + KB 构建
  - `?view=stream` — RAG 流式问答
  - `?view=agent` — 多智能体客服(会话侧栏 + 工单面板 + 调用追踪)
  - `?view=admin` — 客服工单后台
- ChatGPT 风格的**会话侧栏**(重命名 / 删除 / 切换)
- 上线时自动重水化收件箱(离线时收到的客服回复)

### 评估
- **84 行人工标注 eval**(34 直接 / 18 对抗 / 17 多跳 / 15 OOD)
- 分类别 Hit@1/3/5 + MRR@10
- LLM-as-judge 答案质量(忠实 / 引用准确 / 实际作答 / 弃答正确)
- **回归 gate**(CI 在 metric 跌 >0.02 时阻断合并,rewrite=off 确定性)
- thumbs-down → 回归候选数据集 pipeline

---

## 5. 项目结构

```
src/
├── rag/                        # 无状态 RAG 内核
│   ├── ingest/                 # loader + chunking + 连接器 + 清洗
│   ├── index/                  # FAISS / Qdrant / PGVector + BM25
│   ├── query/                  # 归一化 + 改写 + 多 query + 意图分类
│   ├── retrieval/              # 混合检索 + RRF + 重排 + MMR + parent 扩展
│   ├── answer/                 # Grounded LLM + 弃答 + Fallback + meta_answer
│   │   └── hallucination_check.py  # qwen-turbo 校验器
│   ├── nlp/emotion.py          # 投诉严重度分类器
│   ├── vision/                 # Qwen-VL 描述 / OCR / 版面 / CLIP
│   ├── eval/                   # 检索 + 答案评估
│   ├── observability/          # Prometheus + OpenTelemetry
│   └── db/                     # SQLAlchemy 模型 + Alembic
├── rag_api/
│   ├── main.py                 # FastAPI 应用 + 端点
│   ├── agent_routes.py         # SSE /agent/chat
│   ├── admin_routes.py         # 客服后台
│   ├── user_events.py          # 进程内 SSE 总线
│   ├── invoice_pdf.py          # ReportLab CJK 发票 PDF
│   ├── auth.py                 # API key + 租户隔离
│   ├── audit.py                # 仅哈希的审计日志
│   └── rate_limit.py           # 滑动窗口限流
agent/                          # LangGraph 多智能体
├── planner.py                  # Plan-and-Execute
├── coordinator.py              # 路由 + 总结
├── graph.py                    # StateGraph + AsyncSqliteSaver
├── tool_calling.py             # OpenAI 兼容的 function calling 循环
├── specialists/                # 9 个 specialist
└── tools/                      # DB 后端的工具函数
mcp_server/                     # MCP stdio server
frontend/src/                   # Vite + React,5 路由
data/eval/                      # eval_rows.jsonl + baseline.json
scripts/                        # 构建 / 同步 / GDPR / 回归 gate
ops/grafana/                    # 看板 JSON
```

---

## 6. 关键工程决策(为什么这么做)

| 决策 | 原因 |
|---|---|
| RAG 无状态,Agent 通过 HTTP 调 RAG | 测试 Agent 不需要 KB;RAG 可以独立扩容 |
| Async 全链路(FastAPI + AsyncSqliteSaver) | 高并发不阻塞;LangGraph `astream` 模型天然契合 |
| 审计只存哈希 | 审计 DB 万一泄露,blast radius 小;原始内容可通过 trace_id 在应用日志里恢复 |
| Specialist 干跑、按钮提交 | 误分类的问候不会落库;DB 写入永远是用户主动确认 |
| 可逆状态机(complaint / return) | "关闭" 真关闭(UI 隐藏 + API 409);reopen 时按严重度恢复对应活跃状态 + 新 SLA |
| 4 层 Agent 记忆 | 面试问 "多轮怎么处理" 每一层都讲得出细节 |
| 严格的 per-session 工单作用域 | 客服回复不会跨 session 泄漏 |
| SSE 事件 payload 仅哈希 + thread_id | 前端能路由到正确 session;defense in depth |

---

## 7. 测试 + 评估

```bash
# 完整 pytest(其中 ~470 个不需要 LLM)
./.venv/bin/pytest -m "not integration"

# 回归 gate(确定性,rewrite=off)
./.venv/bin/python scripts/regression_gate.py --offline

# 完整检索 eval(rewrite=on,默认)
./.venv/bin/python scripts/run_eval.py --only retrieval

# 实时 SSE 烟囱测(需 API 在跑)
curl -N http://127.0.0.1:8008/users/jd-demo-user/events
```

CI(`.github/workflows/ci.yml`)在每次 PR 跑 pytest + 离线 gate。

---

## 8. 部署 & 文档

- **部署** — 详见 [docs/DEPLOY.md](./docs/DEPLOY.md),提供 Render / Fly.io / Vercel 一键部署指南
- **架构决策** — 详见 [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md),记录 10 个非显而易见的设计选择(Plan-and-Execute vs ReAct、4 层记忆拆分、三向量后端 ABC、specialist 干跑等),ADR 风格:**决策 + 为什么不选其他 + 大规模时会调整什么**

### ⚠️ 部署到公网前先锁紧防御

本项目默认 **匿名开放访问**——本地评估没事,**公网上很危险**(任何人打开你的 URL 都在烧你的 DashScope token)。推到任意公网部署平台前,在环境变量里改:

```bash
REQUIRE_API_KEY=true                # 受保护路由要求 Bearer 认证
RATE_LIMIT_ENABLED=true             # 滑动窗口限流(per-tenant)
RATE_LIMIT_AUTHENTICATED=50/minute  # 比默认 200/min 严
RATE_LIMIT_ANONYMOUS=5/minute       # 匿名近乎关闭
TENANT_TOKEN_BUDGET=200000          # 比默认 1_000_000/min 严
MAX_PROMPT_TOKENS_PER_REQUEST=4096  # 单请求 prompt 大小上限
```

然后用 `python scripts/create_api_key.py` 给每个使用者签发独立 key,不再用了立刻吊销。Key 以 sha256 存储,**原始 key 仅在创建时显示一次**。

面试 / 作品集 demo 场景的**最稳路径仍是本地 fork**(评审者用自己的 DashScope key,你的 key 不出本地)。如果他们没 key,引导他们用**抽取式 fallback 模式**(`ENABLE_FALLBACK_CHAIN=true` 且不设 DashScope key)——答案略不流畅但**零 LLM 调用**,评估免费。

---

## 9. 故障排查(本地跑遇到问题看这里)

| 现象 | 可能原因 | 解决 |
|---|---|---|
| 第一次 `pytest` 失败 | DB schema 没迁移 | `./.venv/bin/alembic upgrade head` |
| `LLMError: DASHSCOPE_API_KEY not set` | 忘填 `.env.local` | `cp .env.example .env.local && 编辑填入` |
| `/answer/stream` 返回 `KB '...' has no built indexes` | KB 还没建 | 通过 `?view=upload` 上传文档,或 `python scripts/build_kb.py <kb_id>` |
| 前端空白 | 后端端口对不上 | 确认后端在 `8008`(默认),Vite 通过 `VITE_API_BASE` 代理 |
| Docker build 每次都拉 torch | 仅首次 — 多阶段 build 已缓存 | 后续 build 复用 wheels 层 |
| `/answer` 多轮元问题感觉幻觉 | 应该被意图路由器自动修复 | 确认 `src/rag/config.py` 的 `enable_intent_routing` 为 `True`(默认) |
| 想评估但不想烧 token | 用抽取式 fallback | `ENABLE_FALLBACK_CHAIN=true` 且 unset `DASHSCOPE_API_KEY` —— 答案由 `[n]` 引用 chunks 的纯 Python 抽取式拼接生成 |

---

## 10. 差异化亮点(不是玩具的理由)

1. **真业务动作端点,不只是聊天**。`/invoice/{id}.pdf` 返回真实 47 KB ReportLab PDF;`/agent/actions/*` 写入审计的 DB 行(sha256 哈希保留)。前端按钮直接打这些端点。

2. **端到端 Agent + 后台闭环**。Agent 立投诉 → 客服后台认领 → 客服回复实时 SSE 推到用户聊天 → 用户在同一个工单卡内回复 → 双向都真正工作,API 还做了 cross-user spoof 拒绝。

3. **所有写入需要明确的用户 / 客服确认**(specialist 干跑)。误分类的问候不会凭空生成投诉工单。

4. **可逆状态机**。complaint 和 return_request 都支持 cancel ↔ reopen,API 层 409 保护错误状态切换 + 严重度感知的 SLA 恢复。

5. **可观测**。每个请求一个 trace_id,每个业务事件一行审计(仅哈希),Prometheus + OpenTelemetry 由 env flag 启停。

6. **可测**。**521 行 pytest** 覆盖 DB / API / SSE 总线 / Agent 流程 / RAG 检索 / 幻觉校验 / 会话 / 用户偏好 / 多轮一致性 / **3 路意图分流** / **改写器 recency bias** / **答题器看 2 轮历史**。

7. **多轮做对了(3 层修复)**。
   - **改写器**带历史看 + 显式 *recency bias* —— 用户说 "翻译那个" 时,内容来源是**最近一轮 assistant**,不是最早的 user 问题(这是真实的 W02-L01 长程指代 bug,通过 prompt-engineering + 4 个测试修复)。
   - **意图分流路由器**在检索之前给每条 query 分 `meta` / `chitchat` / `kb` 三类——元问题("刚刚翻译的是什么")**跳过 RAG**,直接从对话历史回答,修了 naive RAG "幻觉编造无关 chunks 摘要" 的失败模式。
   - **答题器**看最近 2 轮历史**仅作为意图理解上下文**(system prompt 硬规则:事实必须来自 `[n]` 引用的 chunks),给后续问题足够上下文,但不会重新引入旧答案污染。

---

## License

MIT.
