# Architecture Decisions

This document records the **non-obvious architectural choices** in this project — the ones where there were multiple reasonable alternatives, and the trade-offs aren't visible from reading the code alone.

Format inspired by [ADR (Architecture Decision Records)](https://adr.github.io/): each section states **the decision**, **why this over alternatives**, and **what we'd revisit at scale**.

---

## ADR-1: Plan-and-Execute (LangGraph) over ReAct (LangChain Agent)

**Decision**: Build the agent on `LangGraph` `StateGraph` with a Plan-and-Execute pattern — Planner emits a JSON plan, Coordinator dispatches each step to a specialist node — instead of a ReAct-style "LLM picks next action each turn".

**Alternatives considered**:
- LangChain `AgentExecutor` (ReAct) — LLM observes + thinks + acts in a loop.
- LlamaIndex `QueryEngine` — too retrieval-centric, no native multi-step business actions.
- Hand-rolled state machine — flexible, but reinventing checkpointing + streaming.

**Why Plan-and-Execute**:
- **Bounded behavior**. Customer service has 9 known specialists; LLM doesn't need `high-level decision freedom`. ReAct's "let the LLM pick" is for open-ended exploration, not bounded workflows.
- **Reproducibility**. A typed `StateGraph` is a finite-state machine I can write tests against. ReAct loops are non-deterministic in step count and branching, so failures are hard to reproduce.
- **Lower error compounding**. Multi-step tasks like "find my MacBook order → look up tracking" need to succeed end-to-end. ReAct compounds errors per step (~80%×80% = 64% on 2 steps). Plan-and-Execute makes the plan once at the start, then runs deterministic Python.
- **Streaming-friendly**. `graph.astream()` emits events per node, which maps cleanly to the SSE event protocol the front-end consumes.

**What we'd revisit**: For genuinely open-ended agentic tasks (research / coding / browser-use), ReAct is the right fit. The discipline here is "don't pick the trendier pattern — pick the one that matches the workload's structure."

---

## ADR-2: Four-layer Agent memory (not "throw history at LLM")

**Decision**: Memory is split by **information lifecycle**, not by storage mechanism:

| Layer | Lifetime | Storage | What lives here |
|---|---|---|---|
| L1 in-turn `messages` | This turn | LangGraph state | Working context for current LLM call |
| L2 in-turn `entities` | This turn | LangGraph state | Resolved variables (`last_order_id`, `last_item_title`) for downstream specialists |
| L3 cross-session checkpoint | Across days | `AsyncSqliteSaver` (`data/langgraph.sqlite`) by `thread_id` | Conversation continuity when user reconnects later |
| L4 durable user preferences | Forever | `user_preferences` table | "User prefers JD courier", "speaks English", "is allergic to lactose" — facts that should outlive the conversation |

**Alternatives considered**:
- One layer (a flat conversation buffer) — simple but inappropriate for L4 facts.
- Two layers (in-process + persistent) — collapses L1+L2 (which want different reducers in LangGraph) and L3+L4 (different invariants).

**Why four**:
- L2 needs **no reducer** so a fresh turn doesn't reset entities; L1 needs an **append reducer** for messages. Same struct, different semantics.
- L3 is keyed on `thread_id`; L4 is keyed on `user_id`. **Cardinality is different** — multiple threads per user.
- L4 facts are **planner-injected via `render_for_planner()`** so every plan benefits without explicit retrieval. L3 can't do that — it's per-thread.

**What we'd revisit at scale**: L4 needs vector indexing once preferences exceed ~100 per user (else the planner prompt blows up). For demo with ≤10 prefs/user, prompt-injection is fine.

---

## ADR-3: Three vector backends behind one ABC interface

**Decision**: Define a `VectorStore` ABC with `build / search / upsert / delete_by_source_id`, then implement three backends:

- **FAISS** (in-process, flat-IP) — zero-deps, the default
- **Qdrant** (in-memory / local file / HTTP) — for medium scale
- **PGVector** (Postgres extension) — for production reuse of the existing app DB

**Alternatives considered**:
- Hardcode FAISS — simpler, but blocks production reuse.
- Pick "the best" backend (`Qdrant`) — better single-shot performance but lock-in.

**Why all three**:
- **Real scaling story**. Demo runs FAISS (free, no infra). Production swaps to PGVector or Qdrant by changing **one env var**: `VECTOR_BACKEND=pgvector`. **Business code doesn't change**.
- **Each shines somewhere**. PGVector lets you do `WHERE kb_id = ? AND owner = ?` filtering with the same connection that's serving business queries — single transaction story. Qdrant has the best HNSW + payload-filter performance. FAISS is uncrashable.
- **Tests assert ABC compliance**. Switching backends doesn't silently downgrade behavior.

**What we'd revisit**: At >10M vectors, FAISS flat-IP becomes too slow for live search; it's a build-time tool only. The ABC already supports HNSW (Qdrant + PGVector); the FAISS implementation just doesn't.

---

## ADR-4: Specialists are **dry-run**; UI buttons are the **only commit path**

**Decision**: Every specialist that *could* mutate database state — `aftersale`, `complaint`, `account` — runs in dry-run mode by default. They **classify, validate, build a preview**, and return that preview to the front-end. The front-end shows a card with action buttons. **Only when the user clicks a button** does an explicit `/agent/actions/*` endpoint mutate the DB.

**Alternatives considered**:
- Specialists commit directly — simpler, fewer hops.
- Specialists write to a "pending" queue, admin approves — heavier.

**Why dry-run + button**:
- **Misclassification is unavoidable**. The complaint classifier is a regex; users say "你好你是什么客服" and it can match the keyword "客服". Without dry-run, that creates a ghost ticket every time.
- **Audit clarity**. Every DB write traces to a *specific user click event*, not "the LLM thought it should". That alignment makes audit logs interpretable.
- **Demoable**. The front-end shows "previewed action → confirm" UX, which mirrors how mature CS products work (Zendesk, Intercom). One screenshot conveys "production-grade" instantly.

**What we'd revisit**: For low-stakes actions (e.g. a thumb-up feedback), the dry-run hop is overkill. We use it uniformly for symmetry; a real product would tier actions by risk.

---

## ADR-5: 3-way intent routing (meta / chitchat / kb) before retrieval

**Decision**: Every query goes through a rule-based classifier first. If it's a **meta-question** ("what did you just translate") or **chitchat** ("hello"), skip retrieval entirely; answer from conversation history (meta) or with a short LLM reply (chitchat).

**Alternatives considered**:
- Always run RAG. Result: meta-questions hit the index, retrieve arbitrary chunks, hallucinate "summaries" of content the assistant never wrote (this was a real bug we fixed).
- LLM-based intent classifier. Adds 200-300ms per query for a decision rules can make in <1ms.
- Bias the answerer's prompt to "use history if relevant". Brittle; LLMs don't reliably skip retrieval on instruction.

**Why rule-based, pre-retrieval**:
- **Latency**. Pre-retrieval LLM call doubles base latency for every query. A 7-line regex covers 95% of meta cases.
- **Determinism**. Rules are testable. An LLM intent classifier that flips on edge cases is a regression nobody can reproduce.
- **Boundary**. Meta queries require **`has_conversation=True` AND `len(query) ≤ 40`**. Long technical questions mentioning "上面" in passing don't misroute — we'd rather pay one retrieval than misclassify.

**What we'd revisit**: Once the keyword list grows beyond ~30 patterns, switch to a small fine-tuned classifier (sentence-transformers MiniLM + a logistic head). For now, regex wins on ROI.

---

## ADR-6: Answerer sees last 2 turns — but with a strict "facts must come from chunks" system rule

**Decision**: When the user sends a follow-up like "那 PRO 呢" (where PRO is implicit from earlier turns), the LLM that generates the answer sees the last 2 turns of conversation. **But** the system prompt explicitly forbids using history as a fact source: every claim must trace to a `[n]`-cited chunk.

**Alternatives considered**:
- (a) Don't pass history to the answerer (status quo before this session) — fails on follow-up questions where the rewriter doesn't fully resolve coreference.
- (b) Pass full history to the answerer with no rules — ChatGPT-style. Makes the model repeat earlier wrong answers, ignore fresh chunks ("the price is still 198" without re-checking), and leak old facts into new turns.
- (c) **Selected**: Pass last 2 turns as **intent-only context**, with a hard-rule that facts must come from chunks.

**Why c**:
- The rewriter resolves *most* coreference (it's the primary mechanism). The answerer-history covers the **edge cases** the rewriter misses.
- The system rule prevents the failure mode of (b) — the LLM follows the constraint reliably in our tests because the rule is **explicit and short**.
- We bound history at 2 pairs, 300 chars/turn — predictable token cost regardless of session length.

**What we'd revisit**: For very factual domains (medical, legal), even (c) is risky — better to disable history and double down on rewriter quality. For e-commerce CS, (c) is the right balance.

---

## ADR-7: MCP Server in addition to OpenAI native function calling

**Decision**: We have **two distinct tool-calling abstractions**, and they serve different purposes:

| | OpenAI function calling (`tool_calls`) | MCP Server |
|---|---|---|
| **Layer** | LLM API protocol | Client-server protocol |
| **Who decides when to call** | The LLM (model-level decision) | The MCP client (Claude Desktop / Cursor) |
| **Used for** | OrderAgent v2 internal multi-step reasoning | Exposing project capabilities to **external** LLM clients |
| **Direction of call** | LLM → tool | External app → our server |

**Alternatives considered**:
- Only function calling (skip MCP) — internal-only, no story for "other apps reuse this".
- Only MCP (skip function calling) — internal multi-step reasoning loses LLM-driven adaptability.

**Why both**:
- They solve different problems. Conflating them confuses readers.
- **MCP is 2024-2025 industry direction** (Anthropic Claude Skills, Cursor integrations). Demonstrating that I understand it as a *peer* of OpenAI function calling, not a replacement, is the strongest possible signal of "I read the trade press".
- The Tool/Resource/Prompt split (only MCP has) is genuinely useful for **what** is exposed: Tools have side effects, Resources are read-only data the model can `@mention`, Prompts are reusable templates a user picks. Function calling collapses everything to "tool".

**What we'd revisit**: If the project goes single-tenant with a fixed LLM provider, MCP is overhead. For a multi-LLM, multi-client deployment story, MCP earns its keep.

---

## ADR-8: RAG is stateless; Agent calls RAG over HTTP

**Decision**: The RAG pipeline (`src/rag/pipeline.py`) is a pure function: `(query, kb_id, options) → Answer`. No global state, no DB writes from RAG. The Agent's `product_qa` and `policy_qa` specialists call RAG via HTTP **even though they live in the same process**.

**Alternatives considered**:
- Direct in-process Python call (faster, no HTTP) — couples Agent and RAG deployment lifecycles.
- Shared Python session, shared DB — couples session state.

**Why HTTP**:
- **Separable scaling**. RAG is CPU/memory-heavy (BGE-M3 + reranker on GPU); Agent is I/O-bound. They want to scale on different axes. HTTP is the contract that lets them.
- **Testable in isolation**. `tests/test_rag_pipeline.py` doesn't need an Agent. `tests/test_agent_specialists.py` doesn't need a built KB (mock the HTTP call).
- **No abstraction debt**. There's only one "is this product line up?" interaction: HTTP. Agent doesn't grow custom hooks into RAG internals.

**Cost**: ~5-10ms HTTP overhead per RAG call. For a 1.5s answer pipeline, this is negligible.

**What we'd revisit**: At scale, switch to gRPC for tighter binary protocol (but only if HTTP overhead becomes measurable, which it isn't here).

---

## ADR-9: Audit log stores `sha256[:16]` only, never raw query/answer

**Decision**: The `audit_logs` table stores `query_hash`, `answer_hash`, `user_id_hash` (all `sha256(...)[:16]`), never the raw text. Same for SSE event payloads.

**Alternatives considered**:
- Store raw — simpler debugging.
- Store encrypted — overhead, key management risk.
- Store nothing — lose audit ability.

**Why hash-only**:
- **Blast radius**. If `audit_logs` leaks (it's a separate table with broad SELECT for ops), the leak is *which queries happened*, not *what was asked*. PII compliance becomes much easier — there's nothing personal in the audit table.
- **Recoverability when needed**. The application log (separate, retention-managed) carries the raw text keyed by `trace_id`. To debug a real incident, an ops engineer joins audit + applog by `trace_id`. Routine analytics (rate, frequency, error rates) work on the audit alone.
- **GDPR delete is cheap**. A "forget this user" request needs to wipe `audit_logs WHERE user_id_hash = ?` — that's an indexed lookup.

**What we'd revisit**: The 16-char prefix has ~2^64 collision space — fine for our scale (~10^6 events/year). At 10^9 events, bump to 24 chars or full sha256.

---

## ADR-10: Reversible state machines (cancel ↔ reopen) for tickets

**Decision**: Both `complaint` and `return_request` support a full state cycle with API-layer enforcement. A "cancelled" complaint can be reopened, restoring the original severity-appropriate active state (severity high → escalated; severity medium/low → open) and resetting the SLA clock. Wrong-state transitions return `409 Conflict`.

**Alternatives considered**:
- One-way (close = terminal). Simpler model.
- Soft-delete (`is_active` flag) without state machine.

**Why reversible state machine**:
- **Real customer service requires it**. A user mistakenly cancelled a return request → can reopen. A misclassified complaint → admin reopens with proper severity. Without reversibility, the user has to file a new ticket and lose the conversation thread.
- **API enforcement matters**. UI can disable buttons, but a determined user with `curl` should still get `409`, not 500 or silent corruption.
- **Severity restoration**. When reopening, we re-derive the severity from the original classification — not just a flat "active" flag. That's because admin "only-escalated" filter depends on it, and a high-severity ticket reopened as merely "open" would silently disappear from the queue.

**What we'd revisit**: Once the state diagram has more than 5 states, a proper state machine library (`transitions`, `automaton`) earns its keep. At 4 states each, hand-coded `if state == ...` is fine.

---

## What's deliberately **not** here

A few decisions look load-bearing but are actually accidental — listing them so readers don't over-interpret:

- **Vite + React (not Next.js)** — chose Vite because the back-end is FastAPI; SSR isn't useful when the data layer is a separate service. Not a strong opinion, would happily port to Next.js if SEO became relevant.
- **Qwen as primary LLM (not OpenAI)** — pragmatic: free quota for development, decent CN+EN bilingual quality, OpenAI-compatible API so swap cost is one config. Not endorsing Qwen over GPT-4.
- **Pinning sentence-transformers 5.4.0** — no breakage observed across patch versions; the pin is for reproducibility of test snapshot data, not because of a bug.

---

## How to read this with the code

When you see something in the code that looks "weird" (extra abstraction, multiple paths for similar things, an unusual pattern), check whether it's listed here. If it's here, the trade-off was deliberate. If it's not here, it's probably accidental complexity — file an issue.
