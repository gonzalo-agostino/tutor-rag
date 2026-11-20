"""Configuración central y fábrica de clientes compartidos para Tutor RAG."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from embeddings import DEFAULT_LOCAL_REPO, get_local_embeddings

# Cargar variables de entorno desde .env en la raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent


def _sync_env() -> None:
    """Carga .env y desactiva LangSmith en memoria si no hay LANGSMITH_API_KEY."""
    load_dotenv(BASE_DIR / ".env", override=True)
    if not os.getenv("LANGSMITH_API_KEY", "").strip():
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"


_sync_env()

LIBROS_DIR = BASE_DIR / "libros"
CHROMA_DIR = BASE_DIR / "chroma_db"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

COLLECTION_NAME = "tutor_rag_libros"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
LOCAL_EMBEDDING_REPO = os.getenv("LOCAL_EMBEDDING_REPO", DEFAULT_LOCAL_REPO)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
LLM_FALLBACK_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
)

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

_metadata_cache: dict[str, Any] | None = None

# Nombres legibles para mostrar en citas, selector de libros y visor de PDF
BOOK_DISPLAY_TITLES: dict[str, str] = {
    "1 Artificial Intelligence A Modern Approach 3rd (2009).pdf": "Artificial Intelligence: A Modern Approach (3rd Ed.)",
    "1 Inteligencia Artificial - Un enfoque moderno.pdf": "Inteligencia Artificial: Un Enfoque Moderno",
    "3 Data Mining - Practical Machine Learning Tools and Techniques (3rd Ed).pdf": "Data Mining: Practical ML Tools (3rd Ed.)",
    "7 Introduction to Machine Learning with Python_ A Guide for Data Scientists-O’Reilly Media (2016).pdf": "Introduction to Machine Learning with Python",
    "[O'Reilly] AI Engineering_Building Applications with Foundation Models_Chip Huyen.pdf": "AI Engineering (Chip Huyen)",
    "[O'Reilly] AI-Powered Business Intelligence_Improve Forecasts and Decision Making with Machine Learning_Tobias Zwingmann.pdf": "AI-Powered Business Intelligence",
    "[O'Reilly] Architecting Data and Machine Learning Platforms.pdf": "Architecting Data & ML Platforms",
    "[O'Reilly] Beyond Vibe Coding From Coder to AI-Era Developer.pdf": "Beyond Vibe Coding",
    "[O'Reilly] Building Applications with AI Agents Designing and Implementing Multiagent Systems.pdf": "Building Applications with AI Agents",
    "[O'Reilly] Designing Machine Learning Systems.pdf": "Designing Machine Learning Systems",
    "[O'Reilly] Generative AI Design Patterns.pdf": "Generative AI Design Patterns",
    "[O'Reilly] Generative AI on Kubernetes Operationalizing Large Language Models.pdf": "Generative AI on Kubernetes",
    "[O'Reilly] Machine Learning Design Patterns_Solutions to Common Challenges in Data Preparation Model Building and MLOps.pdf": "Machine Learning Design Patterns",
    "Agentic_Design_Patterns.pdf": "Agentic Design Patterns",
    "Clean Code.pdf": "Clean Code",
    "Design Patterns - Elements of Reusable Object Oriented Software.pdf": "Design Patterns (GoF)",
    "Designing Data-Intensive Applications The Big Ideas Behind Reliable, Scalable, and Maintainable Systems by Martin Kleppmann (z-lib.org).pdf": "Designing Data-Intensive Applications",
    "domain-driven-design-distilled.pdf": "Domain-Driven Design Distilled",
    "Ingenieria-de-software-enfoque-practico-7ed-Pressman.PDF": "Ingeniería de Software (Pressman)",
    "libro-objetos-patrones.pdf": "Objetos y Patrones",
    "the-pragmatic-programmer.pdf": "The Pragmatic Programmer",
}


def format_book_title(filename: str) -> str:
    """Devuelve un título limpio y legible para un archivo PDF sin alterar su nombre real."""
    name = (filename or "").strip()
    if not name:
        return "Libro desconocido"
    if name in BOOK_DISPLAY_TITLES:
        return BOOK_DISPLAY_TITLES[name]

    # Limpieza automática para cualquier PDF nuevo agregado en libros/
    title = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
    title = re.sub(r"^\[[^\]]+\]\s*", "", title)  # quita prefijos tipo [O'Reilly]
    title = re.sub(r"^\d+\s+", "", title)  # quita números iniciales tipo "1 ", "7 "
    title = re.sub(r"\(z-lib\.org\)", "", title, flags=re.IGNORECASE)
    title = title.replace("_", " ").replace("-", " ")
    title = re.sub(r"\s+", " ", title).strip()
    return title or name


def ensure_directories() -> None:
    """Garantiza que existan las carpetas operativas del proyecto."""
    LIBROS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)


def invalidate_metadata_cache() -> None:
    """Invalida la caché en memoria de libros indexados tras una ingesta."""
    global _metadata_cache
    _metadata_cache = None


def get_gemini_api_key() -> str:
    """Obtiene la API key de Gemini desde el entorno o lanza un error claro."""
    # Releer .env por si el usuario editó el archivo sin reiniciar
    _sync_env()
    api_key = (
        os.getenv("GEMINI_API_KEY", "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
    )
    if not api_key:
        raise ValueError(
            "Falta configurar GEMINI_API_KEY en el archivo .env "
            f"({BASE_DIR / '.env'})."
        )
    return api_key


def is_api_key_configured() -> bool:
    """Indica si GEMINI_API_KEY está presente y no vacía."""
    try:
        get_gemini_api_key()
        return True
    except ValueError:
        return False


def get_embedding_provider() -> str:
    """Devuelve el proveedor de embeddings configurado ('local' o 'gemini')."""
    _sync_env()
    provider = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
    return provider if provider in {"local", "gemini"} else "local"


def get_embeddings() -> Embeddings:
    """Instancia el modelo de embeddings (ONNX local por defecto, o Gemini si se configura)."""
    provider = get_embedding_provider()
    if provider == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL),
            google_api_key=get_gemini_api_key(),
        )
    repo_id = os.getenv("LOCAL_EMBEDDING_REPO", LOCAL_EMBEDDING_REPO).strip() or DEFAULT_LOCAL_REPO
    return get_local_embeddings(repo_id=repo_id)


def get_llm_candidates(override_primary: str | None = None) -> list[str]:
    """Devuelve el modelo principal configurado seguido de fallbacks rápidos ante 503/504."""
    _sync_env()
    primary = (
        (override_primary or "").strip()
        or os.getenv("LLM_MODEL", LLM_MODEL).strip()
        or "gemini-3.5-flash-lite"
    )
    ordered: list[str] = [primary]
    for m in LLM_FALLBACK_MODELS:
        if m not in ordered:
            ordered.append(m)
    return ordered


def get_llm(
    temperature: float = 0.3,
    model_name: str | None = None,
    timeout: float = 30.0,
    max_retries: int = 1,
) -> ChatGoogleGenerativeAI:
    """Instancia el modelo generativo Gemini Flash con timeout acotado para evitar cuelgues."""
    api_key = get_gemini_api_key()
    chosen = (model_name or os.getenv("LLM_MODEL", LLM_MODEL)).strip() or "gemini-3.5-flash-lite"
    kwargs: dict[str, Any] = {
        "model": chosen,
        "google_api_key": api_key,
        "timeout": timeout,
        "max_retries": max_retries,
    }
    # Modelos con sampling fijo (ej. 3.5-flash-lite / 3.6-flash) emiten aviso si se pasa temperature
    if not any(tag in chosen for tag in ("3.5-flash-lite", "3.6-flash")):
        kwargs["temperature"] = temperature
    return ChatGoogleGenerativeAI(**kwargs)


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.ClientAPI:
    """Devuelve un cliente persistente cacheado de ChromaDB apuntando a chroma_db/."""
    ensure_directories()
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_total_chunks() -> int:
    """Cuenta rápida O(1) de fragmentos en ChromaDB sin cargar metadatos."""
    client = get_chroma_client()
    existing_names = {col.name for col in client.list_collections()}
    if COLLECTION_NAME not in existing_names:
        return 0
    return client.get_collection(name=COLLECTION_NAME).count()


def get_vectorstore(
    embeddings: Embeddings | None = None,
) -> Chroma:
    """Devuelve la instancia de LangChain Chroma conectada a la colección persistente."""
    client = get_chroma_client()
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings or get_embeddings(),
    )


def list_pdf_files(libros_dir: Path = LIBROS_DIR) -> list[Path]:
    """Lista todos los archivos PDF disponibles en libros/ ordenados por nombre."""
    if not libros_dir.exists():
        return []
    return sorted(
        (p for p in libros_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=lambda p: p.name.lower(),
    )


def get_indexed_metadata(use_cache: bool = True) -> dict[str, Any]:
    """Inspecciona la colección de ChromaDB sin requerir llamadas a la API de Gemini.

    Retorna:
        dict con:
        - books: lista ordenada de nombres de archivo indexados (`source`).
        - hashes_by_source: dict[source -> file_hash | None].
        - total_chunks: cantidad total de fragmentos almacenados.
    """
    global _metadata_cache
    if use_cache and _metadata_cache is not None:
        return _metadata_cache

    ensure_directories()
    client = get_chroma_client()
    existing_names = {col.name for col in client.list_collections()}
    if COLLECTION_NAME not in existing_names:
        result = {"books": [], "hashes_by_source": {}, "total_chunks": 0}
        _metadata_cache = result
        return result

    collection = client.get_collection(name=COLLECTION_NAME)
    total_chunks = collection.count()
    if total_chunks == 0:
        result = {"books": [], "hashes_by_source": {}, "total_chunks": 0}
        _metadata_cache = result
        return result

    data = collection.get(include=["metadatas"])
    metadatas = data.get("metadatas") or []
    hashes_by_source: dict[str, str | None] = {}
    for meta in metadatas:
        if not meta:
            continue
        source = meta.get("source")
        if source:
            hashes_by_source[str(source)] = meta.get("file_hash")

    result = {
        "books": sorted(hashes_by_source.keys(), key=str.lower),
        "hashes_by_source": hashes_by_source,
        "total_chunks": total_chunks,
    }
    _metadata_cache = result
    return result
