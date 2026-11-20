"""Pruebas rápidas del motor con una colección Chroma aislada y reproducible."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_engine import KnowledgeBase


class TinyEmbeddings(Embeddings):
    """Embeddings deterministas para no descargar ni ejecutar un modelo real."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        return [float("python" in lowered), float("diseño" in lowered), 1.0]


class KnowledgeBaseTests(unittest.TestCase):
    def setUp(self) -> None:
        # Chroma mantiene archivos abiertos hasta finalizar el proceso en Windows.
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.kb = KnowledgeBase(
            persist_directory=Path(self.temp_dir.name) / "chroma_db",
            collection_name=f"test_{uuid4().hex}",
            embeddings=TinyEmbeddings(),
        )
        self.kb._vectorstore().add_documents(
            [
                Document(
                    page_content="Python facilita automatizar tareas.",
                    metadata={"source": "Python.pdf", "page": 3},
                ),
                Document(
                    page_content="El diseño modular separa responsabilidades.",
                    metadata={"source": "Diseño.pdf", "page": 8},
                ),
                Document(
                    page_content="Python permite escribir herramientas pequeñas.",
                    metadata={"source": "Python.pdf", "page": 4},
                ),
            ]
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_list_books_counts_indexed_pdf_chunks(self) -> None:
        self.assertEqual(
            self.kb.list_books(),
            [
                {"book": "Diseño.pdf", "chunks": 1},
                {"book": "Python.pdf", "chunks": 2},
            ],
        )

    def test_search_global_and_filtered(self) -> None:
        global_results = self.kb.search("Python automatizar", k=3)
        filtered_results = self.kb.search(
            "Python automatizar", book_filter="Python.pdf", k=3
        )
        self.assertTrue(global_results)
        self.assertEqual(len(filtered_results), 2)
        self.assertTrue(all(r["book"] == "Python.pdf" for r in filtered_results))
        self.assertTrue(all(r["text"] and isinstance(r["score"], float) for r in global_results))

    def test_search_results_include_book_and_page_for_citations(self) -> None:
        results = self.kb.search("Python", book_filter="Python.pdf", k=1)
        self.assertTrue(results)
        self.assertEqual(results[0]["book"], "Python.pdf")
        self.assertIn(results[0]["page"], {3, 4})


if __name__ == "__main__":
    unittest.main()
