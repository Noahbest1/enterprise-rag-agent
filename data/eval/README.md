# Eval Pack Roles

This directory now contains answer-eval packs with different roles. Do not treat every pack the same.

## Current Roles

### `airbyte_answer_eval_rows.jsonl`

- Role: `dev/regression`
- Purpose:
  - canonical curated regression gate
  - stable answer-layer comparisons
- Allowed uses:
  - day-to-day debugging
  - replay after workflow changes
  - regression checks before promotions

### `airbyte_answer_eval_rows_userlike.jsonl`

- Role: `dev/regression`
- Purpose:
  - short / colloquial / user-like single-turn queries
  - query-understanding hardening target
- Allowed uses:
  - day-to-day debugging
- query-understanding iteration
- answer/citation regression checks

### `airbyte_answer_eval_rows_llm_needed.jsonl`

- Role: `experimental llm-value set`
- Purpose:
  - heuristic-hard single-turn queries
  - measuring whether LLM query understanding adds value beyond the current heuristic layer
- Allowed uses:
  - baseline-vs-candidate comparisons
  - rewrite / ambiguity / focus-extraction experiments
- Rules:
  - keep this pack stable while comparing heuristic and LLM candidates
  - do not treat it as the only production-readiness gate

### `airbyte_answer_eval_rows_userlike_heldout.template.jsonl`

- Role: `future held-out test template`
- Purpose:
  - template for a user-like held-out set that should not be used during tuning
- Rules:
  - do not overwrite this with tuned rows
  - create a real held-out file from this template only when collection rules are agreed

### `airbyte_answer_eval_rows_userlike_heldout.jsonl`

- Role: `provisional held-out user-like test`
- Purpose:
  - first real held-out-style user-like single-turn set
  - sanity check whether gains on `airbyte_answer_eval_rows_userlike.jsonl` generalize at all
- Rules:
  - do not tune directly on this file
  - this is still only a provisional held-out because it was authored by the current tuner
  - future versions should ideally be expanded by someone else or collected from later logs

### `airbyte_answer_eval_rows_llm_needed.template.jsonl`

- Role: `future llm-value template`
- Purpose:
  - template for queries where heuristic understanding is likely weak but LLM query understanding may help
- Rules:
  - use this pack to justify the marginal value of LLM query understanding
  - do not use it as the only promotion gate for final answer quality

### `airbyte_answer_eval_rows_llm_needed_heldout.jsonl`

- Role: `provisional held-out llm-value test`
- Purpose:
  - first real held-out-style pack for checking whether `llm_candidate` gains survive outside the current `llm_needed` dev pack
- Rules:
  - do not tune directly on this file
  - treat it as a stricter experimental readout than `airbyte_answer_eval_rows_llm_needed.jsonl`
  - this is still provisional until an external collector or later time-split version replaces it

## Collection Principles

### Dev / Regression Packs

- can be inspected freely
- can be used during tuning
- should remain stable enough to detect regressions

### Held-out Packs

- must not be used during heuristic or prompt tuning
- should ideally be collected by someone other than the current tuner
- should be expanded over time with unseen phrasing

### LLM-Needed Packs

- should emphasize:
  - shorthand phrasing
  - missing domain terms
  - paraphrase beyond lexical overlap
  - ambiguity that requires rewrite or clarification
- should not be dominated by queries that heuristics already solve reliably

## Next Intended Additions

1. a real held-out user-like pack
2. expand the real `llm_needed` pack
3. later, a session-level multi-turn eval pack

## Retrieval Packs

### `airbyte_retrieval_queries.jsonl`

- Role: `retrieval benchmark / english`
- Purpose:
  - canonical retrieval benchmark over the current Airbyte corpus
  - compare lexical, embedding, hybrid, and rerank lanes on the established English pack

### `airbyte_retrieval_queries_zh.jsonl`

- Role: `retrieval benchmark / chinese smoke`
- Purpose:
  - first lightweight Chinese retrieval readout over the current mostly English Airbyte corpus
  - validate multilingual embedding behavior before claiming bilingual retrieval support
- Rules:
  - keep this pack small and clean
  - prefer translation-style or natural Chinese paraphrases of existing retrieval intents
  - do not treat this as a production-grade multilingual benchmark yet

## Session-Level Packs

### `airbyte_answer_eval_sessions_userlike_provisional.jsonl`

- Role: `provisional session-level multi-turn eval`
- Purpose:
  - first minimal multi-turn pack for follow-up rewriting, pronoun resolution, and session-level answer stability
- Rules:
  - do not tune directly on this file once first readout is taken
  - use it to validate whether conversation history is actually helping
  - future versions should expand coverage beyond current short follow-up patterns

### `airbyte_answer_eval_sessions_userlike_heldout.template.jsonl`

- Role: `future session-level held-out template`
- Purpose:
  - template for a harder multi-turn held-out pack
  - intended to replace the provisional session pack as the next differentiating signal
- Rules:
  - do not tune directly on this template
  - use it as a structure guide for collecting unseen multi-turn sessions
  - prefer later logs, external collectors, or time-split phrasing when turning this into a real held-out file

### `airbyte_answer_eval_sessions_userlike_heldout.jsonl`

- Role: `provisional session-level held-out`
- Purpose:
  - first real multi-turn held-out-style pack
  - checking whether the provisional session-level gains survive on unseen alternate phrasing
- Rules:
  - do not tune directly on this file
  - treat it as stricter than `airbyte_answer_eval_sessions_userlike_provisional.jsonl`
  - this is still provisional until later-log or externally collected sessions replace it

### `airbyte_answer_eval_sessions_userlike_heldout_v2.jsonl`

- Role: `provisional session-level held-out v2`
- Purpose:
  - second harder multi-turn held-out-style pack
  - adds stronger pronouns, ellipsis, clarification turns, and route ambiguity
- Rules:
  - do not tune directly on this file
  - use it only after freezing the first held-out pack
  - treat it as a broader generalization readout, not a new dev set

## Analysis Utility

### `scripts/analyze_session_answer_eval.py`

- summarize one session-level eval result
- compare two session-level eval outputs turn-by-turn
- use this before writing new heuristics so held-out regressions stay visible
