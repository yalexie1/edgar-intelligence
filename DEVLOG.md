# Dev Log

A chronological record of major changes and bugfixes to this codebase — newest entry
first. This is separate from `CLAUDE.md` (which describes the *current* state of the
project) and `V3_CHANGES.md` (a detailed one-time v3 walkthrough) — this file is an
ongoing, append-only history for answering "when/why/how did X change" questions.

**Entry format:**

```
## YYYY-MM-DD — Short title
**Commit:** `hash` — commit subject
**What:** what changed or was fixed, in plain terms.
**How:** brief rundown of the approach/mechanism.
**Notes:** anything else worth knowing — caveats, what wasn't done, follow-ups.
```

---

## 2026-07-10 — Fix section-diff follow-up hijacking bug in `_route()`

**Commit:** `e206d3e` — Fix section-diff follow-up hijacking bug in `_route()`

**What:** A conversational follow-up like "how does that compare to the prior year?"
was silently rerouted into the section-diff feature (`ask.py`'s `_route()`), discarding
whatever topic the conversation was actually about. Found via live Playwright testing:
asked "What was Microsoft's total revenue in fiscal year 2025?" then followed up with
that generic phrase — the app answered with a full MSFT risk-factors diff instead of
continuing the revenue discussion.

**How:** `_DIFF_PHRASES` (the phrase list that triggers section-diff routing) includes
generic comparative language ("compare to the prior year", "vs last year") that an
ordinary follow-up naturally contains regardless of topic. When the ticker is only known
via history fallback (not named in the current question) *and* no section is named
either, `_route()` now skips `section_diff` and falls through to temporal/plain instead
— that combination is the signature of a generic follow-up, not a deliberate diff
request. A ticker or section named explicitly in the current question still routes to
`section_diff` even with unrelated history present. Fixed test-first: wrote 3 new
`TestRoute` cases (the regression + two guard-rails so the fix doesn't overcorrect),
watched the regression case fail against the old code, then implemented the fix.

**Notes:** `evals/eval.py` never sends conversation `history` (confirmed via grep), so
this fix is logically guaranteed not to change any of the 110-case golden eval results —
no full eval re-run was needed. Unit suite: 98/98 passing. Verified live against the
actual bug scenario and the "legitimate diff-via-follow-up" case (explicit section named)
both before and after. `CLAUDE.md`'s Known Limitations entry for section diff was updated
with the root cause and fix.

---

## 2026-07-10 — Add section-to-section temporal diffs (v3 §4)

**Commit:** `bd74847` — Add section-to-section temporal diffs (v3 §4) — v3 now complete

**What:** New query type: "What changed in NVDA's risk factors since last year?" Retrieves
the same filing section across the two most recent reporting periods and generates a
structured added/removed/unchanged comparison with citations. This was the last of the
five planned v3 features.

**How:** Added `is_section_diff_question()` (matches a narrow `_DIFF_PHRASES` list) and
`detect_section()` (maps phrases like "md&a"/"risk factor" to canonical section labels,
defaulting to `risk_factors` when none is named) in `ask.py`. New retrieval path
`_prepare_section_diff()` filters by ticker + section, pulls a 50-candidate pool, buckets
by period, and keeps the top 2 chunks from each of the 2 most recent distinct periods
(falls back to the broader temporal path if fewer than 2 periods are found). Wired into
`_route()` before the temporal check so diff-shaped questions don't fall through to the
broader trend path. 6 new eval cases added (110 total, up from 104); 28 new unit tests,
written test-first (TDD).

**Notes:** A real bug was caught in pre-commit code review (not live testing) and fixed
same-day, before this commit: the diff prompt's `section` label was picked purely from
the question's wording, ignoring an already-pinned UI section filter — retrieval used the
filter correctly, but the prompt told the model/user the wrong section. Fixed by making an
explicit filter always win. Full suite at commit time: 95/95 passing (later grew to 98
with the follow-up-hijacking fix above). Eval result: 107/110 (97%) — no regression from
the pre-v3-§4 baseline; the new misses are retrieval/LLM answer variance on borderline
factual questions, confirmed unrelated to the new routing via direct `_route()` calls.

---

## 2026-07-10 — Ship v3 streaming/filter fixes and Cohere cross-encoder reranking (v3 §1/§2/§3/§5)

**Commit:** `c389e61` — Ship v3 streaming/filter fixes and Cohere cross-encoder reranking

**What:** Four features shipped together: (1) real-time token streaming via SSE so
answers render as they're generated instead of after a full 5–20s round trip; (2) a
two-stage retrieval pipeline — Pinecone ANN recall (top-50) → Cohere `rerank-v3.5`
cross-encoder — replacing the single-stage cosine+lexical reranker for the final top-k
cut; (3) inline source passage preview so citation cards show the actual cited text,
expandable in place; (4) a section filter (MD&A, Risk Factors, etc.) exposed end-to-end
from the UI dropdown through to the retrieval filter.

**How:** `ask.py`'s routing logic was extracted into a pure `_route()` function plus one
`_prepare_*` helper per retrieval strategy, so the blocking (`ask()`) and streaming
(`ask_stream()`) paths share identical logic and can't drift apart. `api.py` got a new
`POST /query/stream` endpoint (same validation/schema as `/query`, factored into shared
helpers) returning Server-Sent Events. Cohere reranking (`_rerank_with_cohere()`) is
wired into the plain and cross-company retrieval paths only — not diverse mode or
temporal, where relevance-based reordering would undermine what those paths are for — and
falls back silently to the local cosine+lexical order if the API key is unset or the call
fails for any reason. `index.html` was rewritten to consume the SSE stream via
`fetch().body.getReader()`, parse markdown/citations once at the end (not per token) to
avoid flicker, and added the section-filter `<select>` and per-citation preview toggle.

**Notes:** Measured (not assumed) time-to-first-token on a warm backend: ~7.1s of a
~10.3s total — retrieval (embedding + Pinecone), not generation, dominates latency;
streaming only stops the UI from being blank for the *entire* round trip, it doesn't
reduce actual latency. Two bugs were caught and fixed same-day: a UI cold-start-timer
change based on an unmeasured latency assumption (reverted after live testing showed it
fired on every request, not just real cold starts), and a routing bug where any explicit
filter (not just a ticker filter) was disabling structured cross-company/temporal
retrieval — fixed by checking specifically for a ticker filter instead of any filter.
Eval result after all four features: 103/104 (99%), no regression from the pre-v3
baseline. No claim was made that reranking measurably improved precision — only that it
shipped without regressing the aggregate eval (the eval's pass/fail checks can't detect a
reranking change that swaps *which* correct passage gets cited without flipping the
overall pass/fail outcome).
