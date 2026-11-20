"""Motor RAG reutilizable sobre una colección ChromaDB persistente."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_DIR = BASE_DIR / "chroma_db"
DEFAULT_COLLECTION = "tutor_rag_libros"

PROMPTS: dict[str, str] = {
    "explain": "Explica el tema paso a paso, con claridad pedagógica y ejemplos basados solo en las fuentes.",
    "code": "Responde aplicando las fuentes a código: explica el enfoque y muestra código idiomático cuando sea útil.",
    "compare": "Compara las perspectivas presentes en las fuentes, identifica acuerdos y diferencias y cita cada libro.",
    "quick_read": "Prepara una lectura breve: resumen, ideas prácticas, ejemplo si corresponde y una pregunta de autoevaluación.",
}


def _default_embeddings() -> Embeddings:
    """Carga el modelo requerido solo cuando se construye el motor por defecto."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise ImportError(
            "KnowledgeBase requiere langchain-huggingface y sentence-transformers "
            "para usar BAAI/bge-small-en-v1.5."
        ) from exc
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
    )


class KnowledgeBase:
    """Acceso a búsqueda, exploración y generación sobre libros indexados."""

    def __init__(
        self,
        persist_directory: str | Path = DEFAULT_DB_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embeddings: Embeddings | None = None,
        llm: Any | None = None,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embeddings = embeddings if embeddings is not None else _default_embeddings()
        self._client = chromadb.PersistentClient(path=str(self.persist_directory))
        self._llm = llm

    def _collection(self):
        names = {c.name for c in self._client.list_collections()}
        if self.collection_name not in names:
            return None
        return self._client.get_collection(name=self.collection_name)

    def _vectorstore(self) -> Chroma:
        return Chroma(
            client=self._client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )

    def list_books(self) -> list[dict[str, Any]]:
        """Devuelve libros ordenados con su cantidad de fragmentos indexados."""
        collection = self._collection()
        if collection is None or collection.count() == 0:
            return []
        data = collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for metadata in data.get("metadatas") or []:
            if metadata and metadata.get("source"):
                source = str(metadata["source"])
                counts[source] = counts.get(source, 0) + 1
        return [
            {"book": name, "chunks": counts[name]}
            for name in sorted(counts, key=str.casefold)
        ]

    def search(
        self, query: str, book_filter: str | None = None, k: int = 5
    ) -> list[dict[str, Any]]:
        """Busca fragmentos y devuelve distancia/similitud, libro y página."""
        if not query.strip() or k <= 0 or self._collection() is None:
            return []
        filter_value = {"source": book_filter} if book_filter is not None else None
        pairs = self._vectorstore().similarity_search_with_score(
            query=query, k=k, filter=filter_value
        )
        results: list[dict[str, Any]] = []
        for doc, distance in pairs:
            results.append({
                "text": doc.page_content,
                "book": str(doc.metadata.get("source", "")),
                "page": doc.metadata.get("page"),
                "score": float(1.0 - distance),
            })
        return results

    def get_table_of_contents(self, book_name: str) -> str:
        """Recupera los primeros fragmentos ordenados por página del libro."""
        collection = self._collection()
        if collection is None:
            return ""
        data = collection.get(
            where={"source": book_name}, include=["documents", "metadatas"]
        )
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        rows = sorted(
            zip(docs, metas),
            key=lambda pair: (pair[1] or {}).get("page", 0),
        )[:5]
        return "\n\n".join(str(doc).strip() for doc, _ in rows if doc)

    def ask(
        self,
        query: str,
        book_filter: str | None = None,
        mode: Literal["explain", "code", "compare", "quick_read"] = "explain",
    ) -> dict[str, Any]:
        """Responde con Gemini usando contexto recuperado y citas estructuradas."""
        if mode not in PROMPTS:
            raise ValueError(f"Modo no válido: {mode}")
        results = self.search(query, book_filter=book_filter)
        sources = [
            {"book": item["book"], "page": item["page"], "text": item["text"], "score": item["score"]}
            for item in results
        ]
        context = "\n\n".join(
            f"[{item['book']} — Pág. {item['page']}]\n{item['text']}"
            for item in sources
        )
        if not context:
            return {"answer": "No encontré información suficiente en los libros seleccionados.", "sources": []}
        system = (
            "Eres un tutor técnico. Responde en español, basándote únicamente en el contexto. "
            "Cita las afirmaciones con [Libro — Pág. X]. No inventes referencias. "
            "No uses la palabra píldora ni ninguna variación de ella.\n\n"
            f"Instrucción del modo: {PROMPTS[mode]}\n\nContexto:\n{context}"
        )
        llm = self._llm or self._create_llm()
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=query)])
        answer = self._content_as_text(response)
        answer = re.sub(r"p[ií]ldoras?", "unidad", answer, flags=re.IGNORECASE)
        return {"answer": answer, "sources": sources}

    @staticmethod
    def _content_as_text(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "\n".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            ).strip()
        return str(content).strip()

    @staticmethod
    def _create_llm() -> ChatGoogleGenerativeAI:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Configura GEMINI_API_KEY o GOOGLE_API_KEY para usar ask().")
        return ChatGoogleGenerativeAI(
            model=os.getenv("LLM_MODEL", "gemini-2.0-flash"),
            google_api_key=api_key,
            temperature=0.2,
        )
