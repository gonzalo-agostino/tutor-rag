"""Backend FastAPI para el espacio de estudio y consulta técnica Tutor RAG."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

import pymupdf as fitz
from fastapi import BackgroundTasks, Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import AgenticTrace, Cita, ModoTutor, run_tutor
from config import (
    LIBROS_DIR,
    LLM_FALLBACK_MODELS,
    LLM_MODEL,
    STATIC_DIR,
    TEMPLATES_DIR,
    ensure_directories,
    format_book_title,
    get_embedding_provider,
    get_indexed_metadata,
    is_api_key_configured,
    list_pdf_files,
)
from db import (
    append_message,
    clear_metrics,
    create_session,
    delete_session,
    get_metrics_summary,
    get_runtime_settings,
    get_session,
    init_db,
    list_sessions,
    save_metric_trace,
    save_runtime_settings,
    update_metric_scores,
    update_session,
)
from evaluation import evaluate_rag_turn
from ingest import IngestResult, ingest_books

ensure_directories()
init_db()

app = FastAPI(
    title="Tutor RAG — Espacio de Estudio Técnico",
    description="Biblioteca técnica inteligente con LangGraph, ChromaDB, ONNX local y Gemini.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_ingest_lock = asyncio.Lock()


class ChatTurn(BaseModel):
    """Turno previo de conversación para mantener contexto en repreguntas y evaluaciones."""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=16000)


class ChatRequest(BaseModel):
    """Payload recibido por POST /api/chat."""

    mensaje: str = Field(default="", max_length=4000)
    modo: ModoTutor = "consulta"
    libro: str | None = Field(
        default=None,
        description="Nombre exacto del PDF para filtrar búsqueda en ChromaDB.",
    )
    historial: list[ChatTurn] = Field(default_factory=list)
    session_id: str | None = Field(
        default=None,
        description="ID de sesión SQLite donde persistir automáticamente el turno.",
    )


class SessionMessageIn(BaseModel):
    """Mensaje inicial al crear o migrar una sesión."""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=20000)
    modo: ModoTutor = "consulta"
    scopeBook: str | None = None
    citas: list[Cita] = Field(default_factory=list)
    isError: bool = False


class SessionCreateRequest(BaseModel):
    """Payload para POST /api/sessions."""

    title: str = Field(default="Nueva sesión", max_length=200)
    book: str = Field(default="", max_length=300)
    messages: list[SessionMessageIn] = Field(default_factory=list)


class SessionUpdateRequest(BaseModel):
    """Payload para PATCH /api/sessions/{session_id}."""

    title: str | None = Field(default=None, max_length=200)
    book: str | None = Field(default=None, max_length=300)


class RuntimeConfigUpdateRequest(BaseModel):
    """Payload para PATCH /api/config."""

    enable_reflection: bool | None = None
    enable_requery: bool | None = None
    enable_evaluation: bool | None = None
    llm_model: str | None = Field(default=None, max_length=100)


class ChatResponse(BaseModel):
    """Respuesta devuelta por POST /api/chat."""

    respuesta: str
    citas: list[Cita] = Field(default_factory=list)
    modo: ModoTutor
    agentic_trace: AgenticTrace = Field(default_factory=AgenticTrace)
    metric_id: int | None = None


class IngestRequest(BaseModel):
    """Payload opcional para POST /api/ingest."""

    reset: bool = False


class BooksStatusResponse(BaseModel):
    """Estado de la biblioteca local e índice vectorial en ChromaDB."""

    books: list[str]
    indexed_books: list[str]
    book_titles: dict[str, str]
    available_pdfs: list[str]
    total_books: int
    total_chunks: int
    api_key_configured: bool
    llm_model: str


@app.get("/", include_in_schema=False)
async def serve_frontend() -> FileResponse:
    """Sirve la plantilla principal templates/index.html."""
    index_path = TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=500,
            detail="No se encontró templates/index.html.",
        )
    return FileResponse(
        index_path,
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Verificación rápida de estado del servidor."""
    return {"status": "ok"}


@app.get("/books", response_model=BooksStatusResponse)
@app.get("/api/books", response_model=BooksStatusResponse, include_in_schema=False)
async def get_books() -> BooksStatusResponse:
    """Devuelve la lista real de PDFs indexados en ChromaDB y estadísticas de la biblioteca."""
    meta = await asyncio.to_thread(get_indexed_metadata)
    pdfs = [p.name for p in list_pdf_files()]
    indexed = meta["books"]
    return BooksStatusResponse(
        books=indexed,
        indexed_books=indexed,
        book_titles={b: format_book_title(b) for b in indexed},
        available_pdfs=pdfs,
        total_books=len(indexed),
        total_chunks=meta["total_chunks"],
        api_key_configured=is_api_key_configured(),
        llm_model=LLM_MODEL,
    )


def _resolve_safe_pdf_path(book: str) -> Path:
    """Valida que el nombre del libro no permita path traversal y exista dentro de libros/."""
    clean_name = (book or "").strip()
    if (
        not clean_name
        or "/" in clean_name
        or "\\" in clean_name
        or clean_name in {".", ".."}
        or Path(clean_name).name != clean_name
        or Path(clean_name).suffix.lower() != ".pdf"
    ):
        raise HTTPException(
            status_code=400,
            detail="Nombre de archivo PDF inválido.",
        )

    libros_root = LIBROS_DIR.resolve()
    candidate = (libros_root / clean_name).resolve()
    if candidate.parent != libros_root or not candidate.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"No se encontró el libro '{clean_name}' en la carpeta libros/.",
        )
    return candidate


def _render_pdf_page_png(pdf_path: Path, page: int) -> tuple[bytes, int]:
    """Renderiza una página específica (1-based) de un PDF a PNG nítido (Matrix 2x2)."""
    with fitz.open(str(pdf_path)) as doc:
        total_pages = doc.page_count
        if page < 1 or page > total_pages:
            raise HTTPException(
                status_code=400,
                detail=f"Página fuera de rango (pedido: {page}, total del libro: {total_pages}).",
            )
        pdf_page = doc.load_page(page - 1)
        pixmap = pdf_page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        return pixmap.tobytes("png"), total_pages


@app.get("/page-image")
async def get_page_image(
    book: str = Query(..., min_length=1, description="Nombre exacto del archivo PDF en libros/."),
    page: int = Query(..., ge=1, description="Número de página (1-based)."),
) -> Response:
    """Renderiza la página exacta del PDF original a PNG para ver diagramas y figuras."""
    pdf_path = _resolve_safe_pdf_path(book)
    img_bytes, total_pages = await asyncio.to_thread(_render_pdf_page_png, pdf_path, page)
    return Response(
        content=img_bytes,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=3600",
            "X-PDF-Total-Pages": str(total_pages),
        },
    )


@app.get("/api/sessions")
async def api_list_sessions() -> dict[str, list[dict]]:
    """Lista todas las conversaciones guardadas en SQLite (sincronizadas multi-dispositivo)."""
    sessions = await asyncio.to_thread(list_sessions)
    return {"sessions": sessions}


@app.post("/api/sessions")
async def api_create_session(payload: SessionCreateRequest | None = Body(default=None)) -> dict:
    """Crea una nueva sesión de chat en SQLite."""
    body = payload or SessionCreateRequest()
    msgs = [m.model_dump() for m in body.messages]
    return await asyncio.to_thread(
        create_session,
        title=body.title,
        book=body.book,
        messages=msgs,
    )


@app.get("/api/sessions/{session_id}")
async def api_get_session(session_id: str) -> dict:
    """Devuelve una sesión completa con todos sus mensajes y citas."""
    session = await asyncio.to_thread(get_session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Sesión no encontrada.")
    return session


@app.patch("/api/sessions/{session_id}")
async def api_update_session(session_id: str, payload: SessionUpdateRequest) -> dict:
    """Actualiza el título o libro filtrado de una sesión."""
    updated = await asyncio.to_thread(
        update_session,
        session_id,
        title=payload.title,
        book=payload.book,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Sesión no encontrada.")
    return updated


@app.delete("/api/sessions/{session_id}")
async def api_delete_session(session_id: str) -> dict[str, bool]:
    """Elimina una sesión y sus mensajes de SQLite."""
    deleted = await asyncio.to_thread(delete_session, session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Sesión no encontrada.")
    return {"deleted": True}


def _run_background_evaluation(
    metric_id: int,
    query: str,
    response: str,
    contexts: list[str],
) -> None:
    """Evalúa faithfulness y context_precision en segundo plano con DeepEval y persiste en SQLite."""
    scores = evaluate_rag_turn(query=query, response=response, contexts=contexts)
    update_metric_scores(
        metric_id,
        faithfulness=scores.faithfulness,
        context_precision=scores.context_precision,
        faithfulness_reason=scores.faithfulness_reason,
        context_precision_reason=scores.context_precision_reason,
        eval_engine=scores.engine,
    )


@app.get("/api/metrics")
async def api_get_metrics(limit: int = Query(default=20, ge=1, le=100)) -> dict:
    """Devuelve métricas agregadas de observabilidad (faithfulness, context_precision y trazas)."""
    return await asyncio.to_thread(get_metrics_summary, limit)


@app.delete("/api/metrics")
async def api_clear_metrics() -> dict[str, int]:
    """Limpia el historial de métricas de observabilidad en SQLite."""
    deleted = await asyncio.to_thread(clear_metrics)
    return {"deleted": deleted}


@app.get("/api/config")
async def api_get_config() -> dict:
    """Devuelve la configuración activa del pipeline Agentic RAG y opciones disponibles."""
    settings = await asyncio.to_thread(get_runtime_settings)
    active_model = str(settings.get("llm_model") or "").strip() or LLM_MODEL
    return {
        "enable_reflection": bool(settings.get("enable_reflection", True)),
        "enable_requery": bool(settings.get("enable_requery", True)),
        "enable_evaluation": bool(settings.get("enable_evaluation", True)),
        "llm_model": active_model,
        "available_models": list(LLM_FALLBACK_MODELS),
        "embedding_provider": get_embedding_provider(),
    }


@app.patch("/api/config")
async def api_update_config(payload: RuntimeConfigUpdateRequest) -> dict:
    """Actualiza en caliente los parámetros del pipeline Agentic RAG (persistidos en SQLite)."""
    updates = payload.model_dump(exclude_none=True)
    if "llm_model" in updates and updates["llm_model"] not in LLM_FALLBACK_MODELS:
        raise HTTPException(status_code=400, detail="Modelo Gemini no soportado en la lista configurada.")
    settings = await asyncio.to_thread(save_runtime_settings, updates)
    active_model = str(settings.get("llm_model") or "").strip() or LLM_MODEL
    return {
        "enable_reflection": bool(settings.get("enable_reflection", True)),
        "enable_requery": bool(settings.get("enable_requery", True)),
        "enable_evaluation": bool(settings.get("enable_evaluation", True)),
        "llm_model": active_model,
        "available_models": list(LLM_FALLBACK_MODELS),
        "embedding_provider": get_embedding_provider(),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    payload: ChatRequest,
    background_tasks: BackgroundTasks,
) -> ChatResponse:
    """Ejecuta el grafo Agentic RAG, persiste en SQLite y agenda evaluación DeepEval en background."""
    mensaje = payload.mensaje.strip()
    if payload.modo == "consulta" and not mensaje:
        raise HTTPException(
            status_code=400,
            detail="Escribí una pregunta antes de enviar en modo consulta.",
        )
    if payload.modo == "lectura_rapida" and not mensaje:
        mensaje = "Lectura rápida"

    runtime_cfg = await asyncio.to_thread(get_runtime_settings)
    session_id = (payload.session_id or "").strip() or None
    if session_id:
        await asyncio.to_thread(
            append_message,
            session_id=session_id,
            role="user",
            content=mensaje,
            modo=payload.modo,
            scope_book=payload.libro,
        )

    try:
        historial_dicts = [t.model_dump() for t in payload.historial]
        result = await asyncio.to_thread(
            run_tutor,
            mensaje=mensaje,
            modo=payload.modo,
            libro=payload.libro,
            historial=historial_dicts,
            enable_reflection=bool(runtime_cfg.get("enable_reflection", True)),
            enable_requery=bool(runtime_cfg.get("enable_requery", True)),
        )
        if session_id:
            await asyncio.to_thread(
                append_message,
                session_id=session_id,
                role="assistant",
                content=result.respuesta,
                modo=result.modo,
                scope_book=payload.libro,
                citas=[c.model_dump() for c in result.citas],
            )

        trace = result.agentic_trace
        metric_id = await asyncio.to_thread(
            save_metric_trace,
            session_id=session_id,
            query=mensaje,
            modo=result.modo,
            estrategia=trace.estrategia,
            chunks_iniciales=trace.chunks_iniciales,
            chunks_retenidos=trace.chunks_retenidos,
            chunks_descartados=trace.chunks_descartados,
            rebusqueda_activada=trace.rebusqueda_activada,
            conflictos_resueltos=trace.conflictos_resueltos,
            latencia_ms=trace.latencia_ms,
            retrieval_ms=trace.retrieval_ms,
            reflection_ms=trace.reflection_ms,
            generation_ms=trace.generation_ms,
        )

        contexts = [c.extracto for c in result.citas if c.extracto]
        if contexts and bool(runtime_cfg.get("enable_evaluation", True)):
            background_tasks.add_task(
                _run_background_evaluation,
                metric_id,
                mensaje,
                result.respuesta,
                contexts,
            )
        elif not bool(runtime_cfg.get("enable_evaluation", True)):
            await asyncio.to_thread(
                update_metric_scores,
                metric_id,
                faithfulness=None,
                context_precision=None,
                faithfulness_reason="Evaluación desactivada por configuración.",
                context_precision_reason="Evaluación desactivada por configuración.",
                eval_engine="disabled",
            )

        return ChatResponse(
            respuesta=result.respuesta,
            citas=result.citas,
            modo=result.modo,
            agentic_trace=trace,
            metric_id=metric_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error generando la respuesta del tutor: {exc}",
        ) from exc


@app.post("/api/ingest", response_model=IngestResult)
async def ingest_endpoint(
    payload: IngestRequest | None = Body(default=None),
) -> IngestResult:
    """Dispara la indexación incremental (o completa con reset=true) de libros/*.pdf."""
    if _ingest_lock.locked():
        raise HTTPException(
            status_code=409,
            detail="Ya hay una re-indexación en curso. Esperá unos segundos a que termine.",
        )

    reset = payload.reset if payload is not None else False
    async with _ingest_lock:
        try:
            return await asyncio.to_thread(ingest_books, reset=reset, verbose=False)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Error durante la re-indexación: {exc}",
            ) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
