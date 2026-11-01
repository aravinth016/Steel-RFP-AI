import unittest
from unittest.mock import patch

from src.rag import (
    bm25_scores,
    build_qa_prompt,
    chunk_documents,
    embed_texts,
    fuse_rankings,
    retrieve_fused,
    scope_documents,
    validate_answer_citations,
)


def document(document_id, filename, scope, supplier_id, text):
    return {
        "id": document_id,
        "filename": filename,
        "scope": scope,
        "supplier_id": supplier_id,
        "pages": [{"page": 3, "text": text}],
    }


class HybridRagTests(unittest.TestCase):
    def test_scopes_shared_documents_to_one_supplier(self):
        documents = [
            document("requirements", "spec.pdf", "shared", None, "Pump requirements."),
            document("supplier-a", "a.pdf", "supplier", "A", "Pump capacity is 20."),
            document("supplier-b", "b.pdf", "supplier", "B", "Pump capacity is 10."),
        ]
        scoped = scope_documents(documents, "A")
        self.assertEqual(
            [item["id"] for item in scoped],
            ["requirements", "supplier-a"],
        )

    def test_chunks_retain_source_page_and_supplier_provenance(self):
        text = " ".join(f"word{i}" for i in range(10))
        chunks = chunk_documents(
            [document("supplier-a", "a.pdf", "supplier", "A", text)],
            chunk_words=6,
            overlap_words=2,
        )
        self.assertEqual([chunk["page"] for chunk in chunks], [3, 3])
        self.assertEqual(chunks[0]["text"].split()[-2:], chunks[1]["text"].split()[:2])
        self.assertEqual(chunks[1]["supplier_id"], "A")

    def test_bm25_retrieves_relevant_chunk(self):
        chunks = [
            {"text": "The pump capacity is 20 litres per second."},
            {"text": "Delivery will occur next month."},
        ]
        scores = bm25_scores("pump capacity litres", chunks)
        self.assertGreater(scores[0], scores[1])

    def test_rrf_combines_keyword_and_semantic_rankings(self):
        chunks = [
            {"chunk_index": 0, "text": "exact keyword result"},
            {"chunk_index": 1, "text": "semantic result"},
            {"chunk_index": 2, "text": "weaker semantic result"},
        ]
        fused = fuse_rankings(
            chunks,
            lexical_scores=[5.0, 4.0, 0.0],
            semantic_scores=[0.35, 0.8, 0.7],
            limit=3,
        )
        self.assertEqual(fused[0]["chunk_index"], 1)
        self.assertEqual(fused[0]["bm25_rank"], 2)
        self.assertEqual(fused[0]["semantic_rank"], 1)
        self.assertAlmostEqual(
            fused[0]["rrf_score"],
            (1 / (60 + 2)) + (1 / (60 + 1)),
        )

    def test_fusion_returns_no_result_below_semantic_threshold(self):
        chunks = [{"chunk_index": 0, "text": "unrelated text"}]
        self.assertEqual(
            fuse_rankings(chunks, lexical_scores=[0.0], semantic_scores=[0.1]),
            [],
        )

    def test_retrieval_uses_cached_embeddings_for_document_chunks(self):
        chunks = [
            {
                "chunk_index": 0,
                "source_id": "source",
                "filename": "spec.pdf",
                "page": 3,
                "text": "Pump capacity is 20 litres per second.",
            }
        ]
        with patch("src.rag.embed_texts", return_value=[[1.0, 0.0]]) as embed:
            results = retrieve_fused(
                "pump capacity",
                chunks,
                api_key="key",
                document_embeddings=[[1.0, 0.0]],
            )
        embed.assert_called_once()
        self.assertEqual(embed.call_args.kwargs["input_type"], "search_query")
        self.assertEqual(results[0]["source_id"], "source")

    @patch("openai.OpenAI")
    def test_openrouter_embeddings_use_openrouter_key_and_endpoint(self, client_factory):
        response = type(
            "EmbeddingResponse",
            (),
            {
                "data": [
                    type("Item", (), {"index": 1, "embedding": [0.0, 1.0]})(),
                    type("Item", (), {"index": 0, "embedding": [1.0, 0.0]})(),
                ]
            },
        )()
        client_factory.return_value.embeddings.create.return_value = response
        vectors = embed_texts(
            ["first", "second"],
            api_key="openrouter-key",
            model="openai/text-embedding-3-small",
            input_type="search_document",
            provider="OpenRouter",
        )
        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])
        client_factory.assert_called_once_with(
            api_key="openrouter-key",
            base_url="https://openrouter.ai/api/v1",
            timeout=90,
            default_headers={"X-Title": "Tender Forge Steel AI"},
        )

    @patch("openai.OpenAI")
    def test_chatgpt_embeddings_use_evaluation_key_and_openai_endpoint(self, client_factory):
        response = type(
            "EmbeddingResponse",
            (),
            {
                "data": [
                    type("Item", (), {"index": 0, "embedding": [0.25, 0.75]})(),
                ]
            },
        )()
        client_factory.return_value.embeddings.create.return_value = response
        vectors = embed_texts(
            ["proposal text"],
            api_key="evaluation-key",
            model="text-embedding-3-small",
            input_type="search_document",
            provider="ChatGPT",
        )
        self.assertEqual(vectors, [[0.25, 0.75]])
        client_factory.assert_called_once_with(
            api_key="evaluation-key",
            base_url="https://api.openai.com/v1",
            timeout=90,
            default_headers=None,
        )

    @patch("requests.post")
    def test_gemini_embeddings_use_evaluation_key_and_retrieval_task(self, post):
        post.return_value.json.return_value = {
            "embeddings": [{"values": [0.5, 0.5]}]
        }
        vectors = embed_texts(
            ["proposal text"],
            api_key="evaluation-key",
            model="gemini-embedding-001",
            input_type="search_document",
            provider="Gemini",
        )
        self.assertEqual(vectors, [[0.5, 0.5]])
        self.assertEqual(post.call_args.kwargs["headers"], {"x-goog-api-key": "evaluation-key"})
        self.assertEqual(
            post.call_args.kwargs["json"]["requests"][0]["taskType"],
            "RETRIEVAL_DOCUMENT",
        )

    @patch("src.rag.embed_texts")
    def test_claude_uses_keyword_retrieval_without_embedding_key(self, embed):
        chunks = [
            {"chunk_index": 0, "text": "Pump capacity is 20 litres per second."},
            {"chunk_index": 1, "text": "Delivery arrives next month."},
        ]
        results = retrieve_fused(
            "pump capacity",
            chunks,
            api_key="claude-key",
            embedding_provider="Claude",
        )
        embed.assert_not_called()
        self.assertEqual(results[0]["chunk_index"], 0)

    def test_prompt_includes_only_retrieved_citation_labels(self):
        prompt = build_qa_prompt(
            "What is the pump capacity?",
            [
                {
                    "filename": "spec.pdf",
                    "page": 3,
                    "text": "Pump capacity must be 20 litres per second.",
                }
            ],
        )
        self.assertIn("[S1]", prompt)
        self.assertIn("spec.pdf", prompt)
        self.assertIn("page", prompt)

    def test_rejects_unknown_and_missing_citations(self):
        sources = [{"filename": "spec.pdf", "page": 3}]
        with self.assertRaisesRegex(ValueError, "not retrieved"):
            validate_answer_citations("Capacity is 20 [S2].", sources)
        with self.assertRaisesRegex(ValueError, "verifiable evidence"):
            validate_answer_citations("Capacity is 20.", sources)
        validate_answer_citations("Capacity is 20 [S1].", sources)
        validate_answer_citations(
            "This information does not appear in the uploaded documents.",
            sources,
        )


if __name__ == "__main__":
    unittest.main()
