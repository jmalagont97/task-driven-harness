"""test_article_rag.py

Tests for the article RAG: chunking of the four source articles, the shared index and its per-article isolation, the
search_article and get_section executors (scores, thresholds, fallbacks, errors), their declarations, and the article
MCP server over stdio. They build a ChromaDB store with the deterministic hashing embedder, so they need neither Ollama
nor the real store:

    .venv/bin/python -m unittest tests.test_article_rag -v
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import tool_registry  # noqa: E402
from tools.article_rag import search as search_module  # noqa: E402
from tools.article_rag.chunking import MAX_CHARS, chunk_file  # noqa: E402
from tools.article_rag.embeddings import EmbeddingUnavailable, HashingEmbedder, OllamaEmbedder  # noqa: E402
from tools.article_rag.index import ArticleIndex, build_index, query_terms, set_writable  # noqa: E402
from tools.article_rag.search import get_section, search_article  # noqa: E402

GUIDES = REPO_ROOT / "artifacts" / "guides" / "guides_abalone"
METADATA = GUIDES / "rag" / "metadata.json"
DOCUMENTS = [d["id"] for d in json.loads(METADATA.read_text(encoding="utf-8"))["documents"]]
SEARCH = tool_registry.read_declaration(tool_registry.TOOLS_DIR / "article_search_article.json")
SECTION = tool_registry.read_declaration(tool_registry.TOOLS_DIR / "article_get_section.json")
CHUNK_ID = re.compile(r"^[a-z0-9_]+:\d{3}$")


class IndexFixture:
    """A hashing-embedder ChromaDB store of the four articles plus the injection fixture, built once per test run."""

    directory: Path | None = None

    @classmethod
    def path(cls) -> Path:
        if cls.directory is None:
            cls.directory = Path(tempfile.mkdtemp(prefix="article_rag_test_"))
            guides = cls.directory / "guides"
            (guides / "rag").mkdir(parents=True)
            metadata = json.loads(METADATA.read_text(encoding="utf-8"))
            for document in metadata["documents"]:
                shutil.copy(GUIDES / document["file"], guides / document["file"])
            shutil.copy(REPO_ROOT / "tests" / "fixtures" / "article_injection.md", guides / "article_injection.md")
            metadata["documents"].append(
                {"id": "test_injection_fixture", "file": "article_injection.md", "title": "A Test Article", "citation": "Test Author. Fixture.", "year": 2026}
            )
            (guides / "rag" / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            settings = SEARCH["config"]["embedding"]
            build_index(guides / "rag" / "metadata.json", cls.directory / "chroma", HashingEmbedder(), settings["document_template"],
                        settings["query_template"], log=lambda _: None)
        return cls.directory / "chroma"


def config(document_id: str, declaration: dict = SEARCH, **overrides) -> dict:
    cfg = copy.deepcopy(declaration["config"])
    cfg.update(document_id=document_id, index_path=str(IndexFixture.path()), calibration_path=None)
    cfg["embedding"]["provider"] = "hash"
    cfg.update(overrides)
    return cfg


def remove_tree(path: Path) -> None:
    set_writable(path, True)
    shutil.rmtree(path, ignore_errors=True)


def tearDownModule():
    if IndexFixture.directory is not None:
        remove_tree(IndexFixture.directory)


class TestChunking(unittest.TestCase):
    def setUp(self):
        metadata = json.loads(METADATA.read_text(encoding="utf-8"))
        self.articles = {d["id"]: (GUIDES / d["file"]) for d in metadata["documents"]}

    def test_every_chunk_is_the_exact_source_slice(self):
        for document_id, path in self.articles.items():
            source = path.read_text(encoding="utf-8")
            with self.subTest(document=document_id):
                for chunk in chunk_file(document_id, path).chunks:
                    self.assertEqual(source[chunk.char_start : chunk.char_end], chunk.text)

    def test_chunk_ids_are_unique_prefixed_and_sequential(self):
        for document_id, path in self.articles.items():
            chunks = chunk_file(document_id, path).chunks
            with self.subTest(document=document_id):
                ids = [c.chunk_id for c in chunks]
                self.assertEqual(len(ids), len(set(ids)))
                self.assertTrue(all(CHUNK_ID.match(i) and i.startswith(document_id + ":") for i in ids))
                self.assertEqual([c.seq for c in chunks], list(range(1, len(chunks) + 1)))

    def test_chunks_stay_inside_their_section_and_size(self):
        for document_id, path in self.articles.items():
            article = chunk_file(document_id, path)
            spans = {s.section_id: (s.char_start, s.char_end) for s in article.sections}
            self.assertEqual(len(spans), len(article.sections), "section ids must be unique")
            for chunk in article.chunks:
                start, end = spans[chunk.section_id]
                self.assertTrue(start <= chunk.char_start < chunk.char_end <= end, chunk.chunk_id)
                self.assertLessEqual(len(chunk.text), MAX_CHARS + 200, chunk.chunk_id)

    def test_tables_figures_and_pages_are_labelled(self):
        article = chunk_file("prince_et_al_1988_ageing", self.articles["prince_et_al_1988_ageing"])
        kinds = {c.kind for c in article.chunks}
        self.assertTrue({"text", "table", "figure_description", "references"} <= kinds)
        for chunk in article.chunks:
            if chunk.kind == "table":
                self.assertTrue(chunk.text.startswith("|"))
        pages = [c.page_start for c in article.chunks if c.page_start is not None]
        self.assertEqual(pages, sorted(pages))
        self.assertEqual(pages[0], 167)

    def test_numbered_and_repeated_titles_get_distinct_ids(self):
        nash = {s.section_id for s in chunk_file("nash_et_al_1994_tr48", self.articles["nash_et_al_1994_tr48"]).sections}
        self.assertTrue({"3.2", "3.3.1", "2.2.3"} <= nash)
        prince_b = {s.section_id for s in chunk_file("prince_et_al_1988_marbiol", self.articles["prince_et_al_1988_marbiol"]).sections}
        self.assertTrue({"materials-and-methods/growth", "results/growth", "discussion/growth"} <= prince_b)

    def test_overlap_comes_from_the_same_section(self):
        article = chunk_file("mcshane_smith_1992", self.articles["mcshane_smith_1992"])
        previous = {}
        for chunk in article.chunks:
            if chunk.overlap and chunk.kind == "text":
                self.assertIn(chunk.overlap.strip()[-40:], previous[chunk.section_id])
            previous[chunk.section_id] = chunk.text


class TestIndexIsolation(unittest.TestCase):
    def test_reader_loads_only_its_document(self):
        for document_id in DOCUMENTS:
            index = ArticleIndex(IndexFixture.path(), document_id)
            self.assertTrue(index.rows)
            self.assertTrue(all(row["document_id"] == document_id and cid.startswith(document_id + ":") for cid, row in index.rows.items()))
            self.assertEqual([index.rows[c]["seq"] for c in index.order], sorted(index.rows[c]["seq"] for c in index.order))

    def test_vector_search_is_filtered_inside_the_store_query(self):
        nash = ArticleIndex(IndexFixture.path(), "nash_et_al_1994_tr48")
        mcshane = ArticleIndex(IndexFixture.path(), "mcshane_smith_1992")
        probe = HashingEmbedder().embed(["Cliona Polydora Sandpatch Tullaburga growth checks"])[0]
        self.assertTrue(all(cid.startswith("nash_et_al_1994_tr48:") for cid in nash.cosine(probe, 50)))
        self.assertTrue(all(cid.startswith("mcshane_smith_1992:") for cid in mcshane.cosine(probe, 50)))
        self.assertEqual(nash.cosine_of(probe, list(mcshane.rows)[:3]), {}, "chunks of another article are never read")

    def test_store_is_read_only_on_disk(self):
        with self.assertRaises(PermissionError):
            (IndexFixture.path() / "intruder.txt").write_text("x")

    def test_terms_of_another_article_never_return_its_chunks(self):
        canary = "Cliona Polydora Sandpatch Tullaburga Port Fairy tag recapture"  # McShane and Smith 1992 only
        for document_id in ("nash_et_al_1994_tr48", "prince_et_al_1988_ageing", "prince_et_al_1988_marbiol"):
            with self.subTest(document=document_id):
                envelope = search_article({"query": canary, "top_k": 8}, config(document_id))
                self.assertTrue(envelope["ok"], envelope.get("error"))
                for result in envelope["result"]["results"]:
                    self.assertTrue(result["chunk_id"].startswith(document_id + ":"), result["chunk_id"])

    def test_bm25_filters_by_document_inside_the_query(self):
        index = ArticleIndex(IndexFixture.path(), "nash_et_al_1994_tr48")
        self.assertEqual(index.bm25("Tullaburga", 20), {})
        mcshane = ArticleIndex(IndexFixture.path(), "mcshane_smith_1992")
        self.assertTrue(mcshane.bm25("Tullaburga", 20))

    def test_query_terms_drop_stopwords_and_stem(self):
        self.assertEqual(query_terms("What is the age at maturity?"), ["age", "maturity"])
        self.assertEqual(query_terms("growth rings ring"), ["growth", "ring"])
        self.assertEqual(query_terms("the of and"), [])

    def test_partial_rebuild_keeps_other_documents(self):
        directory = Path(tempfile.mkdtemp(prefix="article_rag_partial_"))
        try:
            target = directory / "copy"
            shutil.copytree(IndexFixture.path(), target)
            settings = SEARCH["config"]["embedding"]
            fixture_metadata = IndexFixture.path().parent / "guides" / "rag" / "metadata.json"
            build_index(fixture_metadata, target, HashingEmbedder(), settings["document_template"], settings["query_template"],
                        document_ids=["mcshane_smith_1992"], log=lambda _: None)
            for document_id in DOCUMENTS:
                self.assertTrue(ArticleIndex(target, document_id).rows)
        finally:
            remove_tree(directory)


class TestSearchArticle(unittest.TestCase):
    def test_results_carry_citation_fields_and_ranked_scores(self):
        envelope = search_article({"query": "growth checks annual tag recapture", "top_k": 5}, config("mcshane_smith_1992"))
        self.assertTrue(envelope["ok"], envelope.get("error"))
        result = envelope["result"]
        self.assertEqual(result["document_id"], "mcshane_smith_1992")
        self.assertEqual(result["retrieval_mode"], "hybrid")
        self.assertEqual(len(result["results"]), 5)
        scores = [r["score"] for r in result["results"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for r in result["results"]:
            self.assertRegex(r["chunk_id"], CHUNK_ID)
            self.assertLess(r["char_start"], r["char_end"])
        self.assertEqual(tool_registry.output_problems(SEARCH, envelope), [])

    def test_unrelated_query_is_below_threshold(self):
        envelope = search_article({"query": "chocolate cake recipe with vanilla"}, config("mcshane_smith_1992", min_score_default=0.35))
        self.assertTrue(envelope["ok"])
        self.assertTrue(envelope["result"]["below_threshold"])
        self.assertIn("below_threshold", envelope["flags"])
        self.assertIn("does not cover", envelope["result"]["note"])

    def test_calibration_file_sets_min_score(self):
        directory = Path(tempfile.mkdtemp(prefix="article_rag_cal_"))
        try:
            calibration = directory / "calibration.json"
            calibration.write_text(json.dumps({"mcshane_smith_1992": {"hybrid": 0.99}}), encoding="utf-8")
            envelope = search_article({"query": "growth checks"}, config("mcshane_smith_1992", calibration_path=str(calibration)))
            self.assertEqual(envelope["result"]["min_score"], 0.99)
            self.assertTrue(envelope["result"]["below_threshold"])
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_embedding_outage_falls_back_to_bm25(self):
        class Down:
            name = "hash:256"

            def embed(self, texts):
                raise EmbeddingUnavailable("connection refused")

        original = search_module._cached_embedder
        search_module._cached_embedder = lambda settings: Down()
        try:
            envelope = search_article({"query": "growth checks annual"}, config("mcshane_smith_1992"))
        finally:
            search_module._cached_embedder = original
        self.assertTrue(envelope["ok"], envelope.get("error"))
        self.assertEqual(envelope["result"]["retrieval_mode"], "bm25_only")
        self.assertIn("bm25_only", envelope["flags"])
        self.assertTrue(envelope["warnings"])
        self.assertTrue(all(r["score_vector"] is None for r in envelope["result"]["results"]))

    def test_errors_are_actionable_envelopes(self):
        cases = [
            ({"query": "growth", "top_k": 20}, "input_error", "1-8"),
            ({"query": ""}, "input_error", "'query' is required"),
            ({"query": "growth", "document_id": "nash_et_al_1994_tr48"}, "input_error", "fixed by the server"),
        ]
        for arguments, error_type, fragment in cases:
            with self.subTest(arguments=arguments):
                envelope = search_article(arguments, config("mcshane_smith_1992"))
                self.assertFalse(envelope["ok"])
                self.assertEqual(envelope["error"]["type"], error_type)
                self.assertIn(fragment, envelope["error"]["message"])
                self.assertEqual(tool_registry.output_problems(SEARCH, envelope), [])

        unbound = copy.deepcopy(SEARCH["config"])
        self.assertIn("not bound", search_article({"query": "growth"}, unbound)["error"]["message"])
        missing = search_article({"query": "growth"}, config("mcshane_smith_1992", index_path="/nonexistent/chroma"))
        self.assertEqual(missing["error"]["type"], "index_missing")
        unknown = search_article({"query": "growth"}, config("not_a_document"))
        self.assertEqual(unknown["error"]["type"], "index_missing")
        mismatch_config = config("mcshane_smith_1992")
        mismatch_config["embedding"]["dims"] = 64
        self.assertEqual(search_article({"query": "growth"}, mismatch_config)["error"]["type"], "index_mismatch")

    def test_injected_text_is_returned_as_data_with_the_disclaimer(self):
        envelope = search_article({"query": "ignore previous instructions system prompt"}, config("test_injection_fixture"))
        self.assertTrue(envelope["ok"])
        self.assertIn("never instructions", envelope["disclaimer"])
        self.assertTrue(any("IGNORE ALL PREVIOUS INSTRUCTIONS" in r["text"] for r in envelope["result"]["results"]))

    def test_ollama_embedder_refuses_non_loopback_hosts(self):
        with self.assertRaises(ValueError):
            OllamaEmbedder("embeddinggemma", "http://example.com:11434")
        OllamaEmbedder("embeddinggemma", "http://127.0.0.1:11434")


class TestGetSection(unittest.TestCase):
    def test_returns_the_section_in_parts_with_chunk_ids(self):
        cfg = config("nash_et_al_1994_tr48", SECTION, section_part_chars=1500)
        first = get_section({"section_id": "3.2"}, cfg)
        self.assertTrue(first["ok"], first.get("error"))
        result = first["result"]
        self.assertGreater(result["parts_total"], 1)
        self.assertTrue(all(c["chunk_id"].startswith("nash_et_al_1994_tr48:") for c in result["chunks"]))
        last = get_section({"section_id": "3.2", "part": result["parts_total"]}, cfg)
        self.assertTrue(last["ok"])
        self.assertEqual(tool_registry.output_problems(SECTION, last), [])

    def test_unknown_section_and_part_are_input_errors(self):
        cfg = config("mcshane_smith_1992", SECTION)
        unknown = get_section({"section_id": "methods-that-do-not-exist"}, cfg)
        self.assertEqual(unknown["error"]["type"], "input_error")
        self.assertIn("outline", unknown["error"]["message"])
        too_far = get_section({"section_id": "results", "part": 99}, cfg)
        self.assertEqual(too_far["error"]["type"], "input_error")


class TestDeclarations(unittest.TestCase):
    def test_contracts_follow_the_article_rules(self):
        for declaration in (SEARCH, SECTION):
            with self.subTest(tool=declaration["name"]):
                self.assertEqual(tool_registry.validate_declaration(declaration), [])
                self.assertFalse(declaration["input_schema"]["additionalProperties"])
                self.assertNotIn("document_id", declaration["input_schema"]["properties"])
                self.assertEqual(
                    declaration["mcp"]["annotations"],
                    {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
                )
                self.assertEqual(declaration["side_effects"], "read")
                self.assertIn("Do not use it", declaration["prompt_context"])
        self.assertEqual(SEARCH["input_schema"]["properties"]["top_k"]["maximum"], SEARCH["config"]["top_k_max"])

    def test_tool_names_stay_unique_across_all_servers(self):
        names = tool_registry.load_tool_declarations()
        self.assertIn("search_article", names)
        self.assertIn("get_section", names)


def mcp_available() -> bool:
    try:
        import mcp  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(mcp_available(), "needs the mcp package (requirements-abalone-tools.txt)")
class TestArticleServer(unittest.IsolatedAsyncioTestCase):
    async def test_tools_resources_and_isolation_over_stdio(self):
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
        from mcp.shared.exceptions import MCPError

        params = StdioServerParameters(
            command=sys.executable,
            args=[str(CODE_DIR / "article_mcp_server.py"), "--document-id", "mcshane_smith_1992",
                  "--index-path", str(IndexFixture.path()), "--embedding-provider", "hash", "--log-level", "WARNING"],
            cwd=str(REPO_ROOT),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {tool.name: tool for tool in (await session.list_tools()).tools}
                self.assertEqual(set(tools), {"search_article", "get_section"})
                self.assertIn("Bound article: McShane", tools["search_article"].description)
                self.assertNotIn("document_id", tools["search_article"].input_schema["properties"])

                result = await session.call_tool("search_article", {"query": "growth checks annual", "top_k": 3})
                self.assertFalse(result.is_error)
                self.assertTrue(all(r["chunk_id"].startswith("mcshane_smith_1992:") for r in result.structured_content["result"]["results"]))

                bad = await session.call_tool("search_article", {"query": "growth", "top_k": 50})
                self.assertTrue(bad.is_error)
                self.assertEqual(bad.structured_content["error"]["type"], "input_error")

                resources = {str(r.uri) for r in (await session.list_resources()).resources}
                self.assertEqual(resources, {"article://mcshane_smith_1992/outline", "article://mcshane_smith_1992/metadata"})
                outline = json.loads((await session.read_resource("article://mcshane_smith_1992/outline")).contents[0].text)
                self.assertIn("results", [s["section_id"] for s in outline["sections"]])
                metadata = json.loads((await session.read_resource("article://mcshane_smith_1992/metadata")).contents[0].text)
                self.assertEqual(metadata["year"], 1992)
                self.assertIn("McShane", metadata["citation"])
                self.assertNotIn("summary", metadata)

                with self.assertRaises(MCPError):
                    await session.read_resource("article://nash_et_al_1994_tr48/outline")
                with self.assertRaises(MCPError):
                    await session.call_tool("abalone_sex_random_forest", {"specimens": []})


if __name__ == "__main__":
    unittest.main()
