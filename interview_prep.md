# Interview Prep: Explaining EDGAR Intelligence

KEEP THIS FILE UNTRACKED.

Two versions of the same answer to "tell me about this project" — one for a technical
interviewer, one for a non-technical one. Each includes 📌 notes marking exactly what
I'd pull up on screen (a file, a line range, or a live feature in the app) if asked to
back up a claim.

---

## For the technical interviewer

"EDGAR Intelligence is a RAG system over SEC filings — 10-Ks, 10-Qs, and 8-Ks — for 13
large public companies like Apple, Microsoft, and Nvidia. I built the whole pipeline:
ingestion, chunking, embedding, retrieval, and the answering layer, plus a FastAPI
backend and a vanilla-JS frontend, deployed on Render and Vercel.

The interesting engineering is in retrieval and routing. It's not one retrieval path —
`ask.py` has a `_route()` function 📌(`ask.py:812`) that classifies each question into
one of four strategies: a plain semantic search, structured per-ticker retrieval for
cross-company comparisons, structured per-period retrieval for trend questions, or a
dedicated section-diff path for 'what changed in NVIDIA's risk factors since last year'
style questions. That last one groups retrieved chunks by filing period and builds an
added/removed/unchanged comparison. Both my blocking and streaming answer functions call
the same `_route()`, so the two response modes can't drift out of sync 📌(`ask.py:930`,
`ask_stream()`).

For retrieval quality, it's two-stage: Pinecone ANN pulls a 50-candidate pool using
cosine similarity plus a lexical exact-term boost 📌(`embed_and_search.py:341`,
`search()` — the `rerank_score = similarity + KEYWORD_BOOST * lexical_score` line at
372), then I re-rank that pool with Cohere's `rerank-v3.5` cross-encoder, which reads the
question and each passage together instead of just comparing embeddings
📌(`ask.py:504`, `_rerank_with_cohere()`). The cross-encoder step is wrapped to fail open
— if the API key is missing or the call errors, it falls back to the local rerank order
rather than breaking the request.

Every answer is grounded: I enforce an answer contract in the prompt — every claim needs
a supporting quote, a precise source, and a confidence level, and the model is instructed
to emit a literal `INSUFFICIENT_EVIDENCE` token and abstain rather than guess
📌(`ask.py:216`, `build_prompt()`, rules 1–6 around line 250). I have a 110-case eval
harness that checks retrieval hit-rate and answer correctness automatically — currently
107/110, 97% 📌(`evals/eval.py`, `evals/dataset.json`) — plus 98 unit tests on the pure
logic (routing, section detection, prompt building) with no network calls
📌(`tests/test_pure.py`).

One bug I'd talk through if asked about debugging: I found via live testing that generic
follow-ups like 'how does that compare to the prior year?' were being misrouted into the
section-diff path because the phrase-matching was too broad, silently hijacking an
unrelated revenue question into a risk-factors diff. I fixed it by adding a guard — a
diff route now only fires if the ticker or section was named explicitly in the *current*
question, not just inherited from conversation history 📌(`ask.py:852–878`, the comment
block explaining the fix, and the regression test at
`test_diff_phrase_with_history_ticker_but_explicit_section_still_diffs`)."

---

## For the non-technical interviewer

"Public companies have to file long financial reports with the SEC — annual reports,
quarterly updates, that kind of thing. They're public, but they're also hundreds of
pages of dense legal and financial language, so almost nobody actually reads them
directly. I built a tool that lets you just ask a plain-English question — like 'what
did Apple say about iPhone revenue this year?' or 'what changed in Nvidia's risk factors
since last year?' — and get a short, direct answer instead of hunting through the filing
yourself.

Under the hood it's the same idea as ChatGPT, but pointed specifically at real SEC
filings from 13 major companies — Apple, Microsoft, Amazon, Tesla, and others — so it
can't just make things up the way a general chatbot sometimes does. Every answer has to
be backed by an actual quote from the actual document, with a link to where that quote
came from 📌(I'd pull up the live app and click the 'passage' toggle under a citation —
`index.html`, the `.source-toggle` button — so they can see the exact text the answer
was built from, not just a claim). If the documents don't actually contain the answer,
it says so instead of guessing — that was a deliberate design choice, because for
financial information being wrong confidently is worse than saying 'I don't know.'

I also built in a feature to compare the same section of a report across two years — so
instead of you manually diffing two 100-page filings to see what risks a company newly
disclosed, it does that comparison for you and tells you what was added, removed, or
stayed the same 📌(I'd demo this live: ask 'what changed in NVIDIA's risk factors between
its two most recent 10-Ks?' in the chat UI and walk through the answer).

To make sure it's actually accurate and not just plausible-sounding, I built an automated
test suite of 110 real questions with known correct answers, and I track the pass rate
on a dashboard — right now it's answering 97% of them correctly 📌(I'd open
`dashboard.html`, the eval dashboard, live). That's the same instinct as testing any
piece of software before you ship it — I didn't want to just eyeball a few examples and
call it done."

---

## Reference cheat-sheet (things to actually have open / pull up)

- `ask.py:812` — `_route()`, the four-way routing logic (technical, core architecture)
- `ask.py:504` — `_rerank_with_cohere()`, the cross-encoder rerank + fail-open fallback
- `embed_and_search.py:341` — `search()`, the two-stage retrieval + local rerank formula
- `ask.py:216` — `build_prompt()`, the answer contract (quote + source + confidence + abstain)
- `ask.py:852–878` — the follow-up hijacking bug fix + comment explaining the incident
- `tests/test_pure.py` — 98 unit tests, no network calls
- `evals/eval.py` + `evals/dataset.json` — 110-case golden set, 107/110 (97%)
- Live app: chat UI, citation "passage" preview toggle (`index.html`), the section-diff
  question demo, and `dashboard.html` for the eval pass rate

## Notes on this file

- This file is **not tracked by git** — it's local scratch for interview prep only, not
  part of the committed codebase.
- No code was changed while writing this.

---

## Deep dives: 10 core subjects

Standalone technical explanations for each part of the system, in case an interviewer
asks about one directly instead of through the general project pitch above.

### Ingestion

"`ingest.py` is the offline pipeline that builds the raw corpus, and it's frozen —
changing it means a full re-ingest plus a paid re-embed, so I don't touch it casually
📌(`CLAUDE.md`, 'Freeze rules'). It resolves each ticker to a CIK using the SEC's
official ticker map with a manual fallback dict 📌(`ingest.py:63`, `load_ticker_map()`;
`:77`, `resolve_cik()`), then pulls each company's recent 10-K/10-Q/8-K filings from the
SEC submissions API 📌(`:85`, `fetch_submissions()`) and filters them by form type and a
date window 📌(`:94`, `pick_filings()`). One easy-to-miss requirement: every request has
to send a real `User-Agent` with a name and email, or the SEC returns a 403 — that's set
once on the session 📌(`ingest.py:27`, `USER_AGENT`; `:58`). After downloading each
filing's HTML, I strip anything with `display:none` before extracting text, because SEC
filings embed a lot of hidden XBRL structured-data markup that would otherwise pollute
the plain-text extraction 📌(`ingest.py:146`, the `[style*='display:none']` selector
inside `html_to_text()`). Then the text gets split along the filing's own Part/Item
structure — Item 1A, Item 7, etc. — so a chunk knows which numbered section it came from
before it's ever embedded 📌(`:217`, `split_into_item_sections()`)."

### Chunking

"Chunking happens right after section splitting, still in `ingest.py`. Each item's text
is split at paragraph boundaries rather than by raw character count, so a chunk doesn't
cut a sentence or table row in half 📌(`ingest.py:284`, `split_paragraphs()`), then
reassembled into ~4,000-character chunks with 600 characters of overlap between
consecutive chunks 📌(`:41–42`, `CHUNK_SIZE`/`CHUNK_OVERLAP`; `:294`, `make_overlap()`;
`:306`, `chunk_section()`). The overlap matters because a fact that happens to sit right
at a chunk boundary — a sentence split across two paragraphs — is still fully present in
at least one chunk instead of being cut in two and never retrievable whole. 4,000
characters was a size trade-off: big enough that each chunk carries real context for the
embedding model, small enough that retrieval stays precise instead of pulling in a whole
page when only one paragraph is relevant. There's also a `MIN_PARAGRAPH_CHARS=40` floor
that filters out near-empty paragraph fragments — I found and fixed a bug where that
floor was too aggressive and silently dropped short but meaningful section headings; I
cover that one in the behavioral section on debugging."

### Embedding

"Embedding is the other frozen half of the pipeline, in `embed_and_search.py`. I use
OpenAI's `text-embedding-3-small`, 1,536 dimensions 📌(`embed_and_search.py:39–40`,
`PINECONE_DIMENSION`/`EMBED_MODEL`). `embed_texts()` batches 40 chunks per API call
rather than embedding one at a time, with exponential backoff on failures
📌(`:41`, `BATCH_SIZE`; `:160`, `embed_texts()`) — at 8,141 chunks that's roughly 200
calls instead of 8,141, which matters for both rate limits and cost. On the write side,
vectors get upserted to Pinecone in batches of 100 rather than one giant call, since
Pinecone's own guidance is to keep upsert batches smaller when each vector carries a lot
of metadata — and mine does, because I store the full chunk text in Pinecone's metadata
field rather than in a separate document store 📌(`:42`, `UPSERT_BATCH_SIZE`)."

### Retrieval

"Retrieval is two-stage, in `embed_and_search.py`'s `search()` 📌(`:341`). The question
is embedded and expanded slightly first — `expanded_query()` adds AI-domain synonyms — 
then Pinecone's ANN search returns a broad pool (50 candidates for a normal query) using
cosine similarity, which I combine with a small lexical exact-term boost so a chunk that
literally contains a term like 'seasonality' beats a chunk that's merely semantically
close but generic boilerplate 📌(`:372`, `rerank_score = similarity + KEYWORD_BOOST *
lexical_score`). For the plain query path, that 50-candidate pool then goes through
Cohere's `rerank-v3.5` cross-encoder, which scores the question against each full
passage jointly rather than comparing pre-computed embeddings, and is strictly better at
telling 'answers the question' apart from 'shares vocabulary with the question'
📌(`ask.py:504`, `_rerank_with_cohere()`). I also have a `diversify_results()` mode that
enforces one strong hit per ticker or per filing instead of pure relevance ranking, used
for cross-company 'which company' style questions where I want ticker spread, not just
the five overall-best chunks 📌(`embed_and_search.py:381`)."

### Answering layer

"The answering layer is `ask.py`. `_route()` decides which of four retrieval strategies
a question needs, and each strategy has its own prompt builder — `build_prompt()` for a
plain grounded answer, `build_cross_company_prompt()` for per-company comparisons,
`build_temporal_prompt()` for trend questions grouped chronologically by period, and
`build_diff_prompt()` for two-period section comparisons 📌(`ask.py:216, 266, 325,
377`). Every one of them enforces the same answer contract: cite the passage number,
quote it directly, name the exact source, and give a confidence level, with a literal
`INSUFFICIENT_EVIDENCE` token the model must emit if the evidence is too thin to answer
— which lets the code detect an abstain programmatically rather than trying to parse
free-form refusal language 📌(`ask.py:216`, rules 1–6). Conversation history — the last
three turns — gets folded into both retrieval (to resolve an unnamed company in a
follow-up) and the prompt itself, but the model is explicitly told to ground its answer
in the newly retrieved passages, not in what it said in a prior turn."

### FastAPI

"`api.py` is a fairly small FastAPI service. `Query` is a Pydantic model that validates
the incoming JSON — question length capped at 500 characters, filters typed — and
`_validate_query()` / `_build_where()` are shared between the blocking `/query` and
streaming `/query/stream` endpoints so the two can't validate differently
📌(`api.py:136, 182, 198`). There's an in-memory LRU cache — a 256-entry `OrderedDict` —
keyed on the question, filter, and diversity flag, bypassed whenever conversation
history is present, since a history-bearing request is context-dependent and shouldn't
be cached against a history-free version of the same question 📌(`api.py:38`, `_CACHE`;
`:42`, `_cache_key()`). Rate limiting is per-IP via `slowapi` — 10 requests a minute, 200
a day — plus a global daily cap, with localhost exempted so my own eval harness isn't
throttled 📌(`api.py:212–213`, the `@limiter.limit` decorators). `/health` responds
instantly because the Pinecone client lazy-connects on the first real `/query` call
instead of at process startup 📌(`api.py:124`, `connect()`) — that was deliberate, so
Render's health check doesn't itself trigger a slow cold-start path."

### Frontend

"`index.html` is a single-page vanilla-JS app — no framework, no build step, so
`python -m http.server` is enough to run it locally. The chat flow does a `fetch()`
against `/query/stream` and reads the response with `res.body.getReader()`, parsing
`data:` lines out of the buffered SSE stream and appending tokens into the answer div as
they arrive; markdown and citation-link parsing run once at the end rather than per
token, to avoid re-parsing and flickering on every chunk. Each citation card has a
'passage' toggle button that expands an inline 500-character preview of the actual
source text — implemented as one delegated click listener on the whole chat container
rather than per-card listeners, since cards are injected via `innerHTML` after the fact
📌(`index.html:695`, `renderSourceItems()`; `:706–707`, the toggle/preview markup;
`:724`, the delegated listener). On a cold Render backend, the UI shows a 'waking up'
message and retries automatically up to three times with a 5-second backoff, calibrated
against the ~7-second measured warm time-to-first-token so it doesn't fire on ordinary
requests."

### Backend

"I think of the backend as three layers with a firm boundary between them: `api.py` is
purely the HTTP layer — validation, rate limiting, caching, response shaping; `ask.py` is
the RAG logic — routing, prompting, the answer contract; and `embed_and_search.py` is the
retrieval and vector-store layer. `ask.py` only calls a documented interface out of
`embed_and_search.py` — `search()`, `build_where()`, `diversify_results()`, and a handful
of helpers — rather than reaching into Pinecone details directly 📌(`CLAUDE.md`, 'The
retrieval interface' section lists the exact contract). That boundary is why I can freely
iterate on routing and prompting in `ask.py` while `ingest.py` and `embed_and_search.py`
stay frozen — the interface between them hasn't had to change even though the logic on
top of it has gone through three iterations."

### Deployment

"It's split hosting: the FastAPI backend runs on Render's free tier, the static frontend
(`index.html`, `dashboard.html`, `themes.html`) is on Vercel. `start.sh` just execs
`uvicorn` against Render's `$PORT` 📌(`start.sh`), and `render.yaml` declares the required
env vars — API keys are `sync: false` so they're set manually in the Render dashboard and
never committed, and `COHERE_API_KEY` is explicitly marked optional in a comment, since
the app has to keep working if it's unset 📌(`render.yaml`). I split the two services
rather than serving the frontend from FastAPI because the frontend has zero server-side
logic — no reason to pay a cold-start cost on static HTML. The real cost of the free tier
is that Render spins the backend down after about 15 minutes idle, so the first request
after that takes 15–30 seconds; I don't try to eliminate that, I surface it honestly in
the UI instead."

### System design

"The design principle that shows up everywhere in this project is graceful degradation
over hard dependencies: Cohere reranking, conversation history, even the section filter
are all additive — if any of them is missing or fails, the system falls back to a
simpler behavior instead of erroring out. The API contract itself is additive-only too —
v3 added fields like `section` and each source's `text` preview, but never renamed or
removed an existing field, specifically so the frontend never breaks against an older or
newer backend version 📌(`CLAUDE.md`, 'Freeze rules': 'Do not change the `/query`
request/response JSON contract'). The other recurring pattern is a single source of
truth for a decision that has two call sites — `_route()` is called by both the blocking
`ask()` and the streaming `ask_stream()`, so I can't accidentally make the two response
modes route a question differently. And the eval harness plus unit tests exist
specifically so that pattern — 'two things that must never drift apart' — is enforced by
a test, not just a comment, whenever I add a new one."

---

## 5 expected follow-ups

### 1. "Walk me through what actually happens when a user submits a question."

"The frontend sends the question plus any filters and the last few turns of history to
`POST /query/stream` 📌(`api.py:252`, `query_stream()`). On the backend, `_route()`
looks at the question — does it name multiple companies, is it a trend question, is it a
'what changed' question — and picks one of four retrieval strategies. Whichever one it
picks, the corresponding `_prepare_*` function embeds the question with OpenAI, queries
Pinecone for the top candidates (50 for the plain path), optionally reranks them with
Cohere, and builds a prompt with the numbered passages and the answer contract baked in.
Then the prompt goes to Claude, and I stream the response token-by-token back to the
browser as Server-Sent Events, with the source citations sent as one event once retrieval
is done — that part doesn't have to wait for generation to finish 📌(`ask.py:930`,
`ask_stream()`, the docstring explains why metadata goes first). The frontend appends
tokens into the answer div as they arrive, then parses markdown and citations once
streaming ends."

### 2. "Why Pinecone instead of something like Chroma or FAISS?"

"I actually started with Chroma, a local vector database. It worked fine until I
expanded the corpus from about 5,400 to 8,141 chunks — at that point Chroma's local HNSW
index was around 336MB on disk, and Render's free tier only gives you 512MB of RAM, so
the backend was OOM-crashing on boot while trying to build the index. I migrated to
Pinecone serverless, which keeps the index in the cloud — Render only needs enough
memory to embed the incoming query, not to hold the whole index in memory. It's a good
example of a decision driven by an actual production constraint, not a preference —
I only found the problem by watching the deployed service crash, not by reasoning about
it in advance."

### 3. "What's the hardest bug you ran into on this project?"

"Probably the follow-up routing bug I mentioned — but a close second was a labeling bug
early on: `canonical_section()`, which maps raw filing headings like 'Item 1A. Risk
Factors' to a clean label, was silently losing short headings because a paragraph-length
filter (`MIN_PARAGRAPH_CHARS`) was dropping them before they ever reached the section
classifier. So a chunk that should've been labeled `risk_factors` was falling into an
`other` bucket instead, and the section filter looked broken even though the retrieval
underneath it was fine. It only showed up once I actually inspected section label
distribution across the corpus, not from any single failing test — a reminder that
aggregate metrics can look OK while a specific slice is quietly wrong."

### 4. "How do you know this actually works — how do you evaluate it?"

"Two layers. A 110-question golden set with expected answer substrings and expected
retrieval hits, split into groups — cross-company, temporal, factual, and abstain-only
cases 📌(`evals/dataset.json`, `evals/eval.py`) — currently 107/110. And 98 pure unit
tests on the deterministic logic — routing, section detection, prompt assembly — with no
network calls, so they run in CI-speed and catch regressions in the *decision* logic
separately from the *retrieval quality* 📌(`tests/test_pure.py`). I'm also honest in my
own docs about what isn't measured — the eval set is pass/fail on substring matches, so
a reranking change that shifts *which* correct passage gets cited without changing the
pass/fail outcome wouldn't show up. I'd rather say that plainly than imply a reranking
change I only spot-checked manually was rigorously measured."

### 5. "What's the biggest weakness right now, and what would you fix next?"

"Cross-company superlative questions — 'which company has the highest gross margin?' —
use a broad diverse-retrieval pass across all 8,141 vectors rather than guaranteed
per-company retrieval, so a company's single best chunk can theoretically miss the top-50
pool and get left out of the comparison even though the filing has the answer. Named
comparisons ('compare AAPL and MSFT margins') don't have this problem because they use
structured per-ticker retrieval instead. I'd fix it by detecting the superlative pattern
the same way I detect cross-company questions and running structured per-ticker retrieval
across *all* 13 companies instead of one broad pool. I also have a known retrieval gap on
one specific eval case — Broadcom's gross margin — where retrieval keeps surfacing an
unrelated restructuring 8-K instead of the margin table; the model correctly abstains
rather than guessing, but it's still a retrieval miss I haven't root-caused yet."

---

## 5 likely behavioral questions

### 1. "Tell me about a time you caught a subtle bug through code review, not testing."

"When I built the section-diff feature — comparing the same part of a filing across two
years — I did a self code-review pass on the diff before shipping it, and caught a real
inconsistency: the UI has a Section filter dropdown, and if a user picked 'MD&A' there
but phrased their question as 'what changed in the risk factors,' the retrieval code
correctly respected the UI filter and fetched MD&A chunks — but the *prompt* I sent to
the model was still labeling the comparison 'risk_factors,' based on the question's
wording, not the filter. So the model would be looking at MD&A text while telling the
user it was analyzing risk factors. I found it by deliberately constructing that
conflicting case and calling the routing function directly to see what it returned
📌(`ask.py`, the `_extract_section_filter()` fix and the regression test
`test_section_filter_where_overrides_question_wording`). It taught me that 'the retrieval
is correct' and 'the explanation of the retrieval is correct' are two different things I
have to check separately — a passing eval wouldn't have caught this, because the
underlying chunks were actually the right ones."

### 2. "Tell me about an engineering trade-off you made under a real constraint."

"Cohere's reranking API — which meaningfully improves retrieval precision by scoring the
question against each passage together — is free but capped at 1,000 calls a month on
the tier I'm using. For a demo project I can't guarantee traffic stays under that, and I
didn't want the whole app to break the moment I hit the quota. So I wrote the rerank call
to fail open: if the API key is missing, the call errors, or the quota's exhausted, it
silently falls back to the existing local cosine-plus-lexical ranking instead of raising
📌(`ask.py:504`, `_rerank_with_cohere()` — the `except Exception: return results[:k]`
line). The trade-off is that a user might get a slightly less precise answer with no
visible indication which ranking method produced it. I chose that over a hard failure
because a demo project going down over a paid feature's quota is a worse experience than
a quietly-degraded one — but I documented the trade-off explicitly in my project notes
rather than just letting it be invisible."

### 3. "Tell me about a time you got feedback — even from yourself — and changed direction."

"I'd planned to shorten a 'the backend is waking up' warning in the UI from an 8-second
delay to 2 seconds, reasoning that streaming responses make cold starts feel less
jarring so users could tolerate a shorter wait before seeing that message. But when I
actually tested it live, the warning fired on almost every request, cold-start or not —
because I'd already measured, earlier in the same session, that a normal *warm* request
takes about 7 seconds before the first token arrives, mostly from embedding and
retrieval, not generation. I had the number sitting in my own notes and didn't check it
against the change before making it. I reverted the timer back to 8 seconds and wrote
down the lesson explicitly: a UI-timing change justified by a latency claim should be
checked against the actual measured number, not just intuition, before it ships
📌(I keep this documented directly in `CLAUDE.md` under the streaming section as a
recorded correction, not just a fix)."

### 4. "Tell me about a time you had to prioritize with limited time."

"This whole project is solo, done alongside coursework, so I couldn't build everything
at once. For the third phase of the project I identified five possible improvements —
streaming responses, better reranking, citation previews, section-diff comparisons, and
a section filter in the UI — and ranked them by how visible the gap was to an actual
user rather than by what was most interesting to build. Streaming went first because a
5-to-20-second blank screen with no feedback was the single most noticeable gap versus
any real product. The section-filter UI went last because the backend support for it
already existed — it was the smallest amount of new work for the value, so it made sense
to sequence it after the things that needed real design decisions. Ranking by user-facing
impact instead of technical interest kept me from spending a limited number of hours on
the least-visible improvement first."

### 5. "Tell me about a time you took initiative without being asked."

"Nobody was requiring me to build a formal evaluation system for this — I could have
just tried a handful of questions manually and called it good, which is what I did at
first. But I didn't trust that impression, so I built a 110-question golden-set harness
with expected answers and expected retrieval hits, split by question type, plus a
separate unit test suite for the pure decision logic 📌(`evals/eval.py`,
`tests/test_pure.py`). Partway through I also noticed my own faithfulness metric was
misleading — it was scoring retrieval-only questions against an answer-correctness
denominator that didn't apply to them, making the number look worse than the system
actually was — so I fixed the metric to treat those cases honestly instead of leaving a
flattering-but-wrong number in place. Nobody assigned that fix; I only found it because
I'd already decided the evals needed to be something I could actually trust, not just
something that produced a number."

---

## Personal reference: how the files actually call each other

Not an interview answer — this is for me, so I can answer "walk me through the
architecture" from an accurate mental model instead of a guess if pressed on specifics.

**Two connection types in this codebase: real Python imports (same process, direct
function calls) and HTTP calls (separate processes, JSON over the network).**

1. **`ingest.py` → `embed_and_search.py`: file handoff, not an import.** `ingest.py`
   imports nothing from the rest of the pipeline — it just writes `data/corpus.jsonl`.
   `embed_and_search.py` never imports `ingest.py` either; it reads that same file via
   `load_corpus(CORPUS_PATH)` 📌(`embed_and_search.py:141`). They're two standalone CLI
   scripts chained by a file on disk, run in sequence by hand, not by code calling code —
   part of why they can be frozen independently with no import coupling to break.

2. **`embed_and_search.py` → `ask.py`: real Python import.** `ask.py:29` does
   `from embed_and_search import (...)`, pulling in `search()`, `build_where()`,
   `diversify_results()`, `get_pinecone_index()`, and a few helpers — this is the
   documented "retrieval interface" in `CLAUDE.md`. `ask.py` only touches these named
   functions, never Pinecone internals directly.

3. **`ask.py` → `api.py`: real Python import.** `api.py:25–26` does
   `from ask import ask, ask_stream, track_themes, ANSWER_MODEL` and
   `from embed_and_search import TOP_K, get_pinecone_index`. `api.py` calls
   `ask()`/`ask_stream()` as plain Python functions in the same process — no network hop
   between them. Layering is strictly one-directional:
   `embed_and_search.py → ask.py → api.py`; `ask.py` and `api.py` never import each
   other in reverse.

4. **`api.py` → Pinecone / OpenAI / Anthropic / Cohere: outbound HTTP via SDKs.**
   `connect()` 📌(`api.py:124`) lazily calls `get_pinecone_index()` on the first
   `/query`, not at process startup — why `/health` returns instantly. Inside
   `ask()`/`ask_stream()`, embedding calls go to OpenAI, generation calls go to
   Anthropic, rerank calls go to Cohere — all outbound calls from inside the same
   FastAPI process, not separate internal services.

5. **`index.html` / `dashboard.html` / `themes.html` → `api.py`: HTTP only, zero shared
   code.** The frontend is vanilla JS, a completely separate process (static hosting on
   Vercel) that only ever talks to the backend over `fetch()`:
   - `index.html:846` → `fetch(STREAM_URL, ...)` → `POST /query/stream`
   - `dashboard.html:223` → `fetch(`${API}/evals/results`)`
   - `dashboard.html:400` → `fetch(`${API}/evals/ragas`)`
   - `themes.html:187` → `fetch(`${API}/themes?ticker=...`)`
   This is the entire frontend/backend boundary — JSON over HTTP, matching the response
   contract in `CLAUDE.md`.

6. **`evals/eval.py` → `api.py`: HTTP, deliberately, not an import.** `evals/eval.py`
   imports only stdlib (`urllib.request`, `json`, etc.) — no `from api import` or
   `from ask import`. It fires real HTTP requests at a running `/query` endpoint (hence
   "API must be running" in the commands section of `CLAUDE.md`) instead of calling
   `ask()` in-process, so it exercises the same path a real user hits — FastAPI
   validation, rate limiting, caching — not just the logic underneath.

7. **`tests/test_pure.py` → real imports, but pure functions only.** It imports across
   all three pipeline files —
   `from embed_and_search import build_where, canonical_section, diversify_results, TOP_K`,
   `from ask import (...)`, `from ingest import chunk_section, CHUNK_SIZE, MIN_CHUNK_CHARS`
   — but only calls the pure, no-network functions (chunking math, routing logic, filter
   building), never `search()`, `ask()`, or anything hitting Pinecone/OpenAI/Anthropic.
   That's what lets the suite run with no API keys and no network.

Shape of it:

```
ingest.py --(writes corpus.jsonl)--> embed_and_search.py
                                            │ (import)
                                            ▼
                                          ask.py
                                            │ (import)
                                            ▼
                                          api.py  <──HTTP── index.html / dashboard.html / themes.html
                                            ▲
                                       HTTP │
                                     evals/eval.py

tests/test_pure.py --(imports, pure functions only)--> ingest.py, embed_and_search.py, ask.py
```
