"""Pruebas de las optimizaciones del pipeline Agentic RAG:
1. Clasificador de intención liviano (conversacional vs consulta_tecnica).
2. Limpieza de cabeceras/pies de página PDF en tiempo real y en ingesta.
3. Reflexión permitiendo 0 chunks retenidos y enrutamiento directo a sin_contexto.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from agent import build_tutor_graph, classify_intent
from config import clean_pdf_text
from reflection import reflect_and_filter
from retrieval import RetrievalBundle


def _base_state(mensaje: str, modo: str = "consulta", libro: str | None = None) -> dict:
    return {
        "mensaje": mensaje,
        "modo": modo,
        "libro": libro,
        "historial": [],
        "enable_reflection": True,
        "enable_requery": True,
        "intento": 0,
        "override_query": None,
        "query_efectiva": "",
        "estrategia": "estandar",
        "is_generic_quick_read": False,
        "is_roadmap": False,
        "documentos_candidatos": [],
        "documentos_filtrados": [],
        "reflexion": {},
        "rebusqueda_activada": False,
        "citas": [],
        "respuesta": "",
        "retrieval_ms": 0,
        "reflection_ms": 0,
        "generation_ms": 0,
    }


class PipelineOptimizationsTests(unittest.TestCase):
    def test_classify_intent_conversational_vs_technical(self) -> None:
        self.assertEqual(
            classify_intent(
                "Claro me gustaría eso para entender bien los conceptos y luego consultarle a alguna IA como Codex o Antigravity cómo desarrollarlo bien",
                "consulta",
            ),
            "conversacional",
        )
        self.assertEqual(classify_intent("Hola! Muchas gracias", "consulta"), "conversacional")
        self.assertEqual(classify_intent("Dale perfecto, sigamos", "consulta"), "conversacional")
        self.assertEqual(classify_intent("Que es Harness?", "consulta"), "consulta_tecnica")
        self.assertEqual(classify_intent("Explicame ACID según mis libros", "consulta"), "consulta_tecnica")
        self.assertEqual(
            classify_intent("Quiero desarrollar un ERP con Mercado Libre", "consulta"),
            "consulta_tecnica",
        )
        self.assertEqual(classify_intent("Hola", "lectura_rapida"), "consulta_tecnica")

    def test_clean_pdf_text_and_bundle_autoclean(self) -> None:
        dirty = "\n".join(
            [
                "Chapter 7: Transactions",
                "Weak Isolation Levels | 233",
                "Consistency (in the ACID sense) is a property of the application.",
                "233 | Weak Isolation Levels",
                "Distributed Systems ........... 214",
                "42",
                "Download from https://example.com",
            ]
        )
        cleaned = clean_pdf_text(dirty)
        self.assertEqual(
            cleaned,
            "Consistency (in the ACID sense) is a property of the application.",
        )
        bundle = RetrievalBundle(
            documentos=[Document(page_content=dirty, metadata={"source": "ddia.pdf", "page": 233})]
        )
        self.assertEqual(bundle.documentos[0].page_content, cleaned)

    def test_reflection_allows_zero_retained_chunks(self) -> None:
        docs = [Document(page_content="Irrelevant Java RMI text", metadata={"source": "ddia.pdf", "page": 10})]
        mock_json = json.dumps(
            {
                "indices_relevantes": [],
                "conflictos_resueltos": "",
                "notas_sintesis": "El libro no trata ese tema.",
                "requiere_rebusqueda": False,
                "nueva_query_sugerida": "",
            }
        )
        with patch("reflection.invoke_with_fallback", return_value=mock_json):
            filtered, res = reflect_and_filter("Que es Harness?", "consulta", docs, "libro_unico", intento=1)
        self.assertEqual(len(filtered), 0)
        self.assertEqual(res.chunks_retenidos, 0)
        self.assertEqual(res.chunks_descartados, 1)

    def test_graph_conversational_bypasses_rag(self) -> None:
        graph = build_tutor_graph()
        with patch("agent.generate_conversational_reply", return_value="¡Genial! Decime qué concepto querés ver."):
            st = graph.invoke(
                _base_state(
                    "Claro me gustaría eso para entender bien los conceptos y luego consultarle a alguna IA como Codex o Antigravity cómo desarrollarlo bien"
                )
            )
        self.assertEqual(st["estrategia"], "conversacional")
        self.assertEqual(st["retrieval_ms"], 0)
        self.assertEqual(st["reflection_ms"], 0)
        self.assertEqual(st["citas"], [])

    def test_graph_zero_chunks_routes_to_sin_contexto_without_generation(self) -> None:
        graph = build_tutor_graph()
        docs = [Document(page_content="Irrelevant Java RMI text", metadata={"source": "ddia.pdf", "page": 10})]
        mock_json = json.dumps(
            {
                "indices_relevantes": [],
                "conflictos_resueltos": "",
                "notas_sintesis": "El libro no trata ese tema.",
                "requiere_rebusqueda": False,
                "nueva_query_sugerida": "",
            }
        )
        with (
            patch(
                "agent.retrieve_candidates",
                return_value=RetrievalBundle(
                    documentos=docs, query_efectiva="Que es Harness?", estrategia="libro_unico"
                ),
            ),
            patch("reflection.invoke_with_fallback", return_value=mock_json),
        ):
            st = graph.invoke(_base_state("Que es Harness?", libro="ddia.pdf"))
        self.assertEqual(st["generation_ms"], 0)
        self.assertEqual(st["citas"], [])
        self.assertIn("El libro no trata ese tema.", st["respuesta"])


if __name__ == "__main__":
    unittest.main()
