# Article RAG

Retrieval over the four source articles of the Abalone dataset (`artifacts/guides/guides_abalone/`), stored in
ChromaDB and served as MCP tools by `code/article_mcp_server.py`, **one server per article**. Each article expert agent
(`declarations/agents/article_*_agent.json`) talks only to the server of its own article.

It follows the technique of `chimera-agent-baseline` (ChromaDB + `embeddinggemma`, search exposed as an MCP tool),
with three differences: the embeddings are served by Ollama (on the GPU) instead of an in-process sentence-transformers
model, the collection uses cosine distance, and the search is hybrid (vectors + BM25).

| `document_id` | Article | Chunks |
|---|---|---|
| `nash_et_al_1994_tr48` | Nash et al. 1994, Technical Report 48 (source of the dataset) | 213 |
| `prince_et_al_1988_ageing` | Prince et al. 1988, ageing method | 42 |
| `prince_et_al_1988_marbiol` | Prince et al. 1988, Marine Biology | 55 |
| `mcshane_smith_1992` | McShane and Smith 1992, growth checks | 26 |

`rag/key_facts.md` is not indexed: it mixes the four articles.

## The store

`artifacts/rag/abalone/chroma/` (not versioned, about 4.5 MB): Chroma's own database file, the HNSW index folder
and `manifest.json`, the same layout as chimera-agent-baseline's `resources/guidelines_db/`.

- One collection, `source_articles` (HNSW, cosine). Each record is a chunk: `id` = `chunk_id`, `document` = the chunk
  text, `embedding` = `embeddinggemma` (768 dims) of `title: <section path> | text: <overlap + text>`, and metadata
  `document_id`, `section_id`, `section_path`, `kind`, `seq`, `char_start`, `char_end`, `page_start`, `page_end`
  (`-1` when unknown) and `overlap`.
- `manifest.json` keeps the embedding model, the text templates, and each article's metadata and outline.
- `build_index.py` writes the store in a temporary folder, removes write permission and moves it into place. Chroma
  writes to its store files even to read, so each reader opens a private temporary copy (removed at exit): the canonical
  store can never be modified by a server.

## Isolation

- The article is fixed when the server starts (`--document-id`, set in the flow's `mcp_servers`); it is injected into
  the tools' `config` and is **not** a tool argument. Sending `document_id` is an `input_error`.
- Every Chroma `query` and `get` carries `where={"document_id": ...}`, so the filter runs inside the store's query.
  BM25 runs in memory over the chunks of that one article, loaded with the same filter.
- Least privilege: the server reads only the store, writes nothing to it, and its only network call is the loopback
  embedding request to Ollama (any non-loopback `base_url` is refused). Logs go to stderr as JSON lines; the runtime
  stores them in the run folder.

## Chunks

Sections come from the `##`-`####` headings (`section_id` = the heading number, e.g. `3.2`, or a slug of the heading
path, e.g. `results/growth`). Paragraphs are grouped up to about 1,000 characters (1,500 at most) without crossing
sections or kinds of block. Each chunk has `chunk_id` (`<document_id>:<nnn>`), section, kind (`text`, `table`,
`figure_description`, `editor_note`, `references`), printed pages from the `[p. N]` markers, and `char_start`/`char_end`
such that the chunk text is exactly `source[char_start:char_end]`. Overlap (the previous sentence of the same section,
or a table's header) is stored apart from the text.

## Search

`score = 0.8 * max(cosine, 0) + 0.2 * bm25 / (bm25 + 10)` over the 20 nearest chunks (Chroma HNSW) and the 20 best by
BM25 (Okapi, light stemming); BM25 candidates outside the nearest list get their exact cosine. Weights and candidates
live in the tools' `config`. If Ollama is down the search falls back to BM25 and says so (`retrieval_mode:
"bm25_only"`). `below_threshold` is true when the best score is under `min_score`, read from
`artifacts/rag/abalone/calibration.json` (written by the retrieval eval) or the config default. Reranking is not
implemented.

| Tool | Arguments | Use |
|---|---|---|
| `search_article` | `query` (3-400 chars, English), `top_k` (1-8, default 5) | Every claim about the article |
| `get_section` | `section_id`, `part` | Whole context of a section a result points to |

Resources: `article://<document_id>/outline` and `article://<document_id>/metadata`. Errors come back as `ok: false`
(MCP `isError: true`) with the fix in the message: `input_error`, `index_missing`, `index_mismatch`, `runtime_error`.

## Commands

Run from the repository root.

```bash
# Build the store (Ollama running; about 20 s on the GPU)
.venv/bin/python code/tools/article_rag/build_index.py --all
.venv/bin/python code/tools/article_rag/build_index.py --document-id mcshane_smith_1992

# Check or serve one article
.venv/bin/python code/article_mcp_server.py --document-id mcshane_smith_1992 --check
.venv/bin/python code/article_mcp_server.py --document-id mcshane_smith_1992

# Retrieval eval (no LLM) and min_score calibration
.venv/bin/python evals/article_agents/run_retrieval_eval.py --calibrate

# Tests (hashing embedder, no Ollama needed)
.venv/bin/python -m unittest tests.test_article_rag -v
```

## Retrieval eval

18 questions per article (10 in scope with the snippet that must be retrieved, 5 out of scope, 3 in Spanish),
`embeddinggemma`, top 5, measured on 2026-10-23:

| Article | hit@1 | recall@5 | MRR | recall@5 (Spanish) | isolation | calibrated `min_score` | out of scope rejected | in scope accepted |
|---|---|---|---|---|---|---|---|---|
| `mcshane_smith_1992` | 0.69 | 1.00 | 0.80 | 1.00 | 1.00 | 0.434 | 1.00 | 0.61 |
| `nash_et_al_1994_tr48` | 0.77 | 1.00 | 0.84 | 1.00 | 1.00 | 0.443 | 0.80 | 1.00 |
| `prince_et_al_1988_ageing` | 0.61 | 0.92 | 0.75 | 1.00 | 1.00 | 0.376 | 1.00 | 0.92 |
| `prince_et_al_1988_marbiol` | 0.46 | 0.85 | 0.62 | 0.67 | 1.00 | 0.446 | 1.00 | 0.85 |

In-scope and out-of-scope top scores overlap, so `min_score` is a filter, not a verdict: the expert still reads the
passages and abstains when they do not answer the question.
