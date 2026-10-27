"""code/tools/article_rag/

Retrieval over the source articles of a dataset (artifacts/guides/), one article per MCP server:
- chunking.py: sections and chunks with character offsets, pages and overlap.
- index.py: the shared ChromaDB store (chunks, metadata, vectors in an HNSW index, plus a manifest) and the
  per-article reader that filters by document_id inside every Chroma query and ranks BM25 in memory.
- embeddings.py: the Ollama embedding client (loopback only) and a deterministic hashing embedder for tests.
- search.py: the executors behind declarations/tools/article_search_article.json and article_get_section.json.
- build_index.py: command line to build or rebuild the index.
"""
