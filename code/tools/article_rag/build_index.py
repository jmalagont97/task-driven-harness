"""build_index.py

Build or rebuild the shared ChromaDB store of the articles (artifacts/rag/abalone/chroma/ by default).

Defaults come from declarations/tools/article_search_article.json (index path, metadata path, embedding model and text
templates), so the index always matches what the article servers expect.

Run from the repository root (Ollama must be running for the default embedding provider):
    .venv/bin/python code/tools/article_rag/build_index.py --all
    .venv/bin/python code/tools/article_rag/build_index.py --document-id mcshane_smith_1992
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import tool_registry  # noqa: E402
from tools.article_rag.embeddings import make_embedder  # noqa: E402
from tools.article_rag.index import build_index  # noqa: E402

DECLARATION = tool_registry.TOOLS_DIR / "article_search_article.json"


def main(argv: list[str] | None = None) -> int:
    config = tool_registry.read_declaration(DECLARATION)["config"]
    parser = argparse.ArgumentParser(description="Build the article index used by code/article_mcp_server.py.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--all", action="store_true", help="Rebuild every article of the collection.")
    target.add_argument("--document-id", action="append", help="Rebuild only this article inside the existing index (repeatable).")
    parser.add_argument("--metadata", default=config["metadata_path"], help="Collection metadata.json.")
    parser.add_argument("--output", default=config["index_path"], help="Store folder to write.")
    parser.add_argument("--embedding-provider", choices=["ollama", "hash"], default=config["embedding"]["provider"])
    args = parser.parse_args(argv)

    settings = dict(config["embedding"], provider=args.embedding_provider)
    embedder = make_embedder(settings)
    counts = build_index(
        tool_registry.REPO_ROOT / args.metadata,
        tool_registry.REPO_ROOT / args.output,
        embedder,
        settings["document_template"],
        settings["query_template"],
        document_ids=None if args.all else args.document_id,
    )
    print(f"{sum(counts.values())} chunks from {len(counts)} article(s) written to {args.output} with {embedder.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
