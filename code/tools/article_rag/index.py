"""index.py

The shared ChromaDB store of a collection's source articles, and the reader that serves one article.

Responsibilities:
- build_index: chunk every article listed in the collection's metadata.json, embed the chunks and write one
  persistent ChromaDB store (collection "source_articles", HNSW with cosine distance) plus a manifest.json with the
  embedding model, the text templates, each article's metadata and its outline. The store is built in a temporary
  folder, made read-only, and only then moved into place, so a failed build never leaves a half-written store.
- ArticleIndex: opens the store and serves exactly one document_id. Every Chroma query and get carries
  where={"document_id": ...}, so the filter is applied inside the store's query, never after retrieving. BM25 runs in
  memory over that article's chunks only. Chroma writes to its store files even to read, and the built store is
  read-only on disk, so each reader opens a private temporary copy; the canonical store can never be modified.

The store follows the technique of chimera-agent-baseline (ChromaDB + embeddinggemma), with embeddings served by
Ollama instead of an in-process sentence-transformers model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import atexit
import shutil
import stat
import tempfile
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .chunking import chunk_file
from .embeddings import Embedder

SCHEMA_VERSION = "2"
COLLECTION = "source_articles"
MANIFEST = "manifest.json"
ADD_BATCH = 500
METADATA_FIELDS = (
    "id", "title", "authors", "year", "type", "journal", "volume", "pages", "doi", "publisher", "series", "issn", "isbn",
    "citation", "species", "study_area", "study_period", "relation_to_dataset", "source_pdf", "processing", "editorial_notes",
)
STOPWORDS = frozenset(
    "a an and are as at be by did do does for from has have how in is it its of on or that the their this to was were "
    "what when where which who why with".split()
)
WORD = re.compile(r"[A-Za-z0-9]+(?:\.[0-9]+)?")
SUFFIXES = ("ations", "ation", "ings", "ing", "edly", "ies", "ed", "es", "ly", "s")
BM25_K1 = 1.2
BM25_B = 0.75


class IndexMissing(FileNotFoundError):
    """The store is missing or does not contain the requested document."""


class IndexMismatch(ValueError):
    """The store was built with another embedding model or schema than the reader expects."""


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def stem(word: str) -> str:
    """A light suffix stripper, so 'rings' matches 'ring' and 'matured' matches 'mature'."""
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def tokens(text: str) -> list[str]:
    return [stem(word) for word in WORD.findall(text.lower()) if word not in STOPWORDS]


def query_terms(text: str) -> list[str]:
    """Distinct content terms of a query, in order; empty when nothing is left to match."""
    return list(dict.fromkeys(tokens(text)))


def embedding_text(template: str, section_path: str, text: str, overlap: str) -> str:
    body = f"{overlap}\n{text}" if overlap else text
    return template.format(title=section_path or "none", text=body)


class BM25:
    """Okapi BM25 over a small, fixed set of documents."""

    def __init__(self, documents: dict[str, list[str]]):
        self.ids = list(documents)
        self.term_counts = {doc_id: Counter(words) for doc_id, words in documents.items()}
        self.lengths = {doc_id: len(words) for doc_id, words in documents.items()}
        self.average = (sum(self.lengths.values()) / len(self.lengths)) if self.lengths else 0.0
        frequency: Counter[str] = Counter()
        for counts in self.term_counts.values():
            frequency.update(counts.keys())
        total = len(self.ids)
        self.idf = {term: math.log(1 + (total - df + 0.5) / (df + 0.5)) for term, df in frequency.items()}

    def scores(self, terms: list[str]) -> dict[str, float]:
        result = {}
        for doc_id in self.ids:
            counts, length = self.term_counts[doc_id], self.lengths[doc_id]
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if tf:
                    norm = BM25_K1 * (1 - BM25_B + BM25_B * length / (self.average or 1))
                    score += self.idf[term] * tf * (BM25_K1 + 1) / (tf + norm)
            if score > 0:
                result[doc_id] = score
        return result


# ---------------------------------------------------------------------------
# Store helpers
# ---------------------------------------------------------------------------

def chroma_client(path: Path) -> Any:
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False, allow_reset=False))


def open_read_copy(path: Path) -> Any:
    """A Chroma client on a private, throwaway copy of a read-only store (removed when the process exits)."""
    if os.access(path, os.W_OK):
        return chroma_client(path)
    workdir = Path(tempfile.mkdtemp(prefix="article_store_"))
    copy = workdir / "store"
    shutil.copytree(path, copy)
    set_writable(copy, True)
    atexit.register(shutil.rmtree, workdir, True)
    return chroma_client(copy)


def set_writable(path: Path, writable: bool) -> None:
    """Add or remove write permission on a store folder and everything in it."""
    if not path.exists():
        return
    entries = [path, *path.rglob("*")]
    for entry in entries if writable else reversed(entries):
        mode = entry.stat().st_mode
        entry.chmod(mode | stat.S_IWUSR if writable else mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def collection_documents(metadata_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata, list(metadata["documents"])


def document_file(metadata_path: Path, document: dict[str, Any]) -> Path:
    """Article files sit one level above the rag/ folder that holds metadata.json."""
    return metadata_path.parent.parent / document["file"]


def chunk_metadata(chunk: Any) -> dict[str, Any]:
    """Chroma metadata values must be scalars; a missing page is stored as -1."""
    return {
        "document_id": chunk.document_id,
        "section_id": chunk.section_id,
        "section_path": chunk.section_path,
        "kind": chunk.kind,
        "seq": chunk.seq,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "page_start": -1 if chunk.page_start is None else chunk.page_start,
        "page_end": -1 if chunk.page_end is None else chunk.page_end,
        "overlap": chunk.overlap,
    }


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def build_index(
    metadata_path: Path,
    output_path: Path,
    embedder: Embedder,
    document_template: str,
    query_template: str,
    document_ids: Iterable[str] | None = None,
    log: Any = print,
) -> dict[str, int]:
    """Build the whole store, or rebuild only document_ids inside an existing one. Returns chunks per document."""
    _, documents = collection_documents(metadata_path)
    wanted = set(document_ids or [d["id"] for d in documents])
    unknown = wanted - {d["id"] for d in documents}
    if unknown:
        raise ValueError(f"{sorted(unknown)} not in {metadata_path}; known ids: {[d['id'] for d in documents]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    if temporary.exists():
        set_writable(temporary, True)
        shutil.rmtree(temporary)
    partial = document_ids is not None and (output_path / MANIFEST).is_file()
    if partial:
        shutil.copytree(output_path, temporary)
        set_writable(temporary, True)
        manifest = json.loads((temporary / MANIFEST).read_text(encoding="utf-8"))
        if manifest.get("embedding_model") != embedder.name or manifest.get("document_template") != document_template:
            set_writable(temporary, True)
            shutil.rmtree(temporary)
            raise IndexMismatch(
                f"The existing store uses {manifest.get('embedding_model')!r}; rebuild all documents with --all to switch to {embedder.name!r}."
            )
    else:
        temporary.mkdir(parents=True)
        manifest = {"documents": {}}

    try:
        client = chroma_client(temporary)
        if partial:
            collection = client.get_collection(COLLECTION)
            for document_id in wanted:
                collection.delete(where={"document_id": document_id})
        else:
            collection = client.create_collection(COLLECTION, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None)

        counts: dict[str, int] = {}
        dims = manifest.get("dims")
        for document in documents:
            if document["id"] not in wanted:
                continue
            path = document_file(metadata_path, document)
            started = time.perf_counter()
            article = chunk_file(document["id"], path)
            texts = [embedding_text(document_template, c.section_path, c.text, c.overlap) for c in article.chunks]
            vectors = embedder.embed(texts)
            dims = int(vectors.shape[1])
            for start in range(0, len(article.chunks), ADD_BATCH):
                batch = article.chunks[start : start + ADD_BATCH]
                collection.add(
                    ids=[c.chunk_id for c in batch],
                    embeddings=np.asarray(vectors[start : start + ADD_BATCH], dtype=np.float32).tolist(),
                    documents=[c.text for c in batch],
                    metadatas=[chunk_metadata(c) for c in batch],
                )
            manifest["documents"][document["id"]] = {
                "file": document["file"],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "n_chunks": len(article.chunks),
                "metadata": {key: document[key] for key in METADATA_FIELDS if key in document},
                "sections": [
                    {"section_id": s.section_id, "title": s.title, "level": s.level, "path": s.path,
                     "page_start": s.page_start, "page_end": s.page_end, "n_chunks": s.n_chunks}
                    for s in article.sections
                ],
            }
            counts[document["id"]] = len(article.chunks)
            log(f"{document['id']}: {len(article.chunks)} chunks, {len(article.sections)} sections ({time.perf_counter() - started:.1f} s)")

        manifest.update(
            schema_version=SCHEMA_VERSION,
            collection=COLLECTION,
            distance="cosine",
            embedding_model=embedder.name,
            dims=dims,
            document_template=document_template,
            query_template=query_template,
            metadata_path=str(metadata_path),
            built_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        (temporary / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        # Chroma's cached client for the temporary path is left alone: stopping cached clients would also stop the ones
        # other readers in this process still use. Moving the folder afterwards is safe.
        del collection, client
    except BaseException:
        set_writable(temporary, True)
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    set_writable(temporary, False)  # the servers only ever read the store
    if output_path.exists():
        set_writable(output_path, True)
        shutil.rmtree(output_path)
    os.replace(temporary, output_path)
    return counts


# ---------------------------------------------------------------------------
# Reading one article
# ---------------------------------------------------------------------------

class ArticleIndex:
    """One document of the shared store. Every query to Chroma is filtered by this document_id."""

    def __init__(self, path: Path, document_id: str):
        manifest_path = path / MANIFEST
        if not manifest_path.is_file():
            raise IndexMissing(f"Store not found at {path}. Build it with: .venv/bin/python code/tools/article_rag/build_index.py --all")
        self.path = path
        self.document_id = document_id
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise IndexMismatch(f"Store schema {self.manifest.get('schema_version')!r} is not {SCHEMA_VERSION!r}; rebuild it with build_index.py --all.")
        entry = self.manifest["documents"].get(document_id)
        if entry is None:
            raise IndexMissing(
                f"The store has no document {document_id!r}. Build it with: "
                f".venv/bin/python code/tools/article_rag/build_index.py --document-id {document_id}"
            )
        self.metadata_entry = entry["metadata"]
        self.sections = entry["sections"]
        self.collection = open_read_copy(path).get_collection(self.manifest["collection"])
        stored = self.collection.get(where={"document_id": document_id}, include=["documents", "metadatas"])
        self.rows: dict[str, dict[str, Any]] = {}
        for chunk_id, text, meta in zip(stored["ids"], stored["documents"], stored["metadatas"]):
            row = {"chunk_id": chunk_id, "text": text, **meta}
            for key in ("page_start", "page_end"):
                row[key] = None if row[key] == -1 else row[key]
            self.rows[chunk_id] = row
        self.order = sorted(self.rows, key=lambda chunk_id: self.rows[chunk_id]["seq"])
        self.lexical = BM25({cid: tokens(f"{r['section_path']} {r['overlap']} {r['text']}") for cid, r in self.rows.items()})

    def check_embedding(self, embedder_name: str, query_template: str) -> None:
        if self.manifest.get("embedding_model") != embedder_name or self.manifest.get("query_template") != query_template:
            raise IndexMismatch(
                f"The store was built with {self.manifest.get('embedding_model')!r}, but the tool is configured for "
                f"{embedder_name!r}. Rebuild the store with that model: build_index.py --all"
            )

    def bm25(self, query: str, limit: int) -> dict[str, float]:
        """chunk_id -> BM25 score (higher is better) for the best `limit` chunks of this document."""
        terms = query_terms(query)
        if not terms:
            return {}
        scores = self.lexical.scores(terms)
        return dict(sorted(scores.items(), key=lambda item: -item[1])[:limit])

    def cosine(self, query_vector: np.ndarray, limit: int) -> dict[str, float]:
        """chunk_id -> cosine similarity for this document's nearest `limit` chunks (Chroma HNSW, filtered in the query)."""
        result = self.collection.query(
            query_embeddings=[np.asarray(query_vector, dtype=np.float32).tolist()],
            n_results=max(1, min(limit, len(self.rows))),
            where={"document_id": self.document_id},
            include=["distances"],
        )
        return {chunk_id: 1.0 - float(distance) for chunk_id, distance in zip(result["ids"][0], result["distances"][0])}

    def cosine_of(self, query_vector: np.ndarray, chunk_ids: Iterable[str]) -> dict[str, float]:
        """Exact cosine for given chunks of this document (for BM25 candidates outside the nearest-neighbour list)."""
        ids = [cid for cid in chunk_ids if cid in self.rows]
        if not ids:
            return {}
        stored = self.collection.get(ids=ids, where={"document_id": self.document_id}, include=["embeddings"])
        query = np.asarray(query_vector, dtype=np.float32)
        return {cid: float(np.asarray(vector, dtype=np.float32) @ query) for cid, vector in zip(stored["ids"], stored["embeddings"])}

    def chunks(self, chunk_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        return {cid: self.rows[cid] for cid in chunk_ids if cid in self.rows}

    def section_chunks(self, section_id: str) -> list[dict[str, Any]]:
        return [self.rows[cid] for cid in self.order if self.rows[cid]["section_id"] == section_id]

    def outline(self) -> list[dict[str, Any]]:
        return list(self.sections)
