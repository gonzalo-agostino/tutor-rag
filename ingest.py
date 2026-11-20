"""Script CLI y módulo de ingesta incremental y reanudable de PDFs hacia ChromaDB."""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import re
import sys
import time
from collections import deque
from pathlib import Path

import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from pypdf import PdfReader

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    LIBROS_DIR,
    clean_pdf_text,
    ensure_directories,
    get_chroma_client,
    get_embedding_provider,
    get_embeddings,
    get_indexed_metadata,
    get_vectorstore,
    invalidate_metadata_cache,
    list_pdf_files,
)

logging.getLogger("pypdf").setLevel(logging.ERROR)
# Aumentar presupuesto de parsing de streams para libros técnicos con gráficos vectoriales densos
pypdf.overwrite_configuration(
    pypdf.get_configuration().with_overwrites(
        stream_parsing_work_maximum_count=20_000_000,
    )
)

# Lote moderado para convivir cómodamente con Gemini Free Tier (100 req/min y 30.000 tokens/min)
DEFAULT_BATCH_SIZE = 32
FREE_TIER_MAX_RPM = 85
FREE_TIER_MAX_TPM = 24_000
RATE_WINDOW_SECONDS = 62.0


class DailyQuotaExceededError(RuntimeError):
    """Se lanza cuando la API de Gemini informa agotamiento de la cuota diaria."""


class IngestResult(BaseModel):
    """Resumen estructurado de una ejecución de ingesta."""

    reset: bool = False
    processed_files: list[str] = Field(default_factory=list)
    skipped_files: list[str] = Field(default_factory=list)
    failed_files: dict[str, str] = Field(default_factory=dict)
    total_chunks_added: int = 0
    total_chunks_in_db: int = 0
    indexed_books: list[str] = Field(default_factory=list)
    message: str = ""


class SlidingWindowRateLimiter:
    """Regula el envío de lotes para evitar errores 429 en Gemini Free Tier."""

    def __init__(
        self,
        max_requests_per_min: int = FREE_TIER_MAX_RPM,
        max_tokens_per_min: int = FREE_TIER_MAX_TPM,
        window_seconds: float = RATE_WINDOW_SECONDS,
        enabled: bool = True,
    ) -> None:
        self.max_rpm = max_requests_per_min
        self.max_tpm = max_tokens_per_min
        self.window_seconds = window_seconds
        self.enabled = enabled
        self._events: deque[tuple[float, int, int]] = deque()

    def _prune(self, now: float) -> None:
        while self._events and (now - self._events[0][0]) >= self.window_seconds:
            self._events.popleft()

    def wait_for_capacity(
        self,
        num_requests: int,
        est_tokens: int,
        verbose: bool = True,
    ) -> None:
        if not self.enabled:
            return

        while True:
            now = time.monotonic()
            self._prune(now)
            used_req = sum(ev[1] for ev in self._events)
            used_tok = sum(ev[2] for ev in self._events)

            fits_req = (used_req + num_requests) <= self.max_rpm
            fits_tok = (used_tok + est_tokens) <= self.max_tpm
            if not self._events or (fits_req and fits_tok):
                self._events.append((now, num_requests, est_tokens))
                return

            oldest_ts = self._events[0][0]
            sleep_for = max(1.0, (oldest_ts + self.window_seconds) - now + 0.5)
            if verbose:
                print(
                    f"  -> Pausando {sleep_for:.0f}s para respetar límite por minuto de Gemini Free Tier...",
                    flush=True,
                )
            time.sleep(sleep_for)


def _compute_file_hash(file_path: Path) -> str:
    """Calcula el hash SHA-256 de un archivo para detectar cambios."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _estimate_tokens(batch: list[Document]) -> int:
    """Estimación conservadora de tokens (caracteres / 3) para español y código."""
    return sum(max(1, len(doc.page_content) // 3) for doc in batch)


def _extract_retry_delay_seconds(error_text: str) -> float | None:
    """Extrae el tiempo de espera sugerido por Google en respuestas 429."""
    match = re.search(r"Please retry in\s+([0-9]+(?:\.[0-9]+)?)s", error_text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?([0-9]+(?:\.[0-9]+)?)s", error_text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _is_daily_quota_error(error_text: str) -> bool:
    """Detecta si el 429 corresponde al límite diario (PerDay) en lugar de por minuto."""
    upper = error_text.upper()
    return "PERDAY" in upper or "PER_DAY" in upper or "DAILY" in upper


def _reset_collection() -> None:
    """Elimina la colección de ChromaDB de forma segura (sin conflictos de lock en Windows)."""
    client = get_chroma_client()
    existing_names = {col.name for col in client.list_collections()}
    if COLLECTION_NAME in existing_names:
        client.delete_collection(name=COLLECTION_NAME)


def _delete_source_chunks(source_name: str) -> None:
    """Elimina los fragmentos anteriores de un libro actualizado o incompleto heredado."""
    client = get_chroma_client()
    existing_names = {col.name for col in client.list_collections()}
    if COLLECTION_NAME in existing_names:
        collection = client.get_collection(name=COLLECTION_NAME)
        collection.delete(where={"source": source_name})


def _get_existing_ids_for_source(source_name: str) -> list[str]:
    """Devuelve los IDs ya almacenados en ChromaDB para un libro dado."""
    client = get_chroma_client()
    existing_names = {col.name for col in client.list_collections()}
    if COLLECTION_NAME not in existing_names:
        return []
    collection = client.get_collection(name=COLLECTION_NAME)
    data = collection.get(where={"source": source_name}, include=[])
    return list(data.get("ids") or [])


def _add_batch_with_retry(
    vectorstore,
    batch: list[Document],
    batch_ids: list[str],
    rate_limiter: SlidingWindowRateLimiter,
    verbose: bool = True,
    max_retries: int = 8,
) -> None:
    """Agrega un lote a ChromaDB respetando rate limits y retryDelay exacto de Gemini."""
    est_tokens = _estimate_tokens(batch)
    for attempt in range(1, max_retries + 1):
        rate_limiter.wait_for_capacity(len(batch), est_tokens, verbose=verbose)
        try:
            vectorstore.add_documents(documents=batch, ids=batch_ids)
            return
        except Exception as exc:
            error_text = str(exc)
            if _is_daily_quota_error(error_text):
                raise DailyQuotaExceededError(
                    "Se alcanzó la cuota diaria de embeddings de Gemini Free Tier. "
                    "Todo el progreso anterior quedó guardado en chroma_db/."
                ) from exc

            upper = error_text.upper()
            is_transient = any(
                code in upper
                for code in ("429", "RESOURCE_EXHAUSTED", "RATE", "503", "UNAVAILABLE", "TIMEOUT")
            )
            if not is_transient or attempt == max_retries:
                raise

            suggested = _extract_retry_delay_seconds(error_text)
            wait_seconds = (math.ceil(suggested) + 3) if suggested is not None else min(65, 15 * attempt)
            if verbose:
                print(
                    f"  -> Límite temporal de API; esperando {wait_seconds}s "
                    f"(reintento {attempt}/{max_retries})...",
                    flush=True,
                )
            time.sleep(wait_seconds)


def _load_and_split_pdf(
    pdf_path: Path,
    file_hash: str,
    splitter: RecursiveCharacterTextSplitter,
    verbose: bool = True,
) -> list[Document]:
    """Carga un PDF con pypdf, conserva metadatos (source y page 1-based) y divide en chunks."""
    reader = PdfReader(str(pdf_path))

    normalized_pages: list[Document] = []
    skipped_pages = 0
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = clean_pdf_text(page.extract_text() or "")
        except Exception:
            skipped_pages += 1
            continue

        if not text:
            continue

        normalized_pages.append(
            Document(
                page_content=text,
                metadata={
                    "source": pdf_path.name,
                    "page": page_num,
                    "file_hash": file_hash,
                },
            )
        )

    if skipped_pages and verbose:
        print(
            f"  -> Aviso: se omitieron {skipped_pages} página(s) con error de lectura en {pdf_path.name}.",
            flush=True,
        )

    if not normalized_pages:
        return []

    return splitter.split_documents(normalized_pages)


def _chunk_id(source_name: str, file_hash: str, index: int) -> str:
    """Genera un ID determinista por chunk para permitir reanudar libros grandes."""
    return f"{source_name}::{file_hash[:12]}::{index}"


def ingest_books(
    reset: bool = False,
    libros_dir: Path = LIBROS_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    only_filters: list[str] | None = None,
    rate_limit: bool = True,
    verbose: bool = True,
) -> IngestResult:
    """Escanea libros/*.pdf e indexa en ChromaDB los archivos nuevos, modificados o incompletos."""
    ensure_directories()
    result = IngestResult(reset=reset)

    if reset:
        if verbose:
            print("Reiniciando colección vectorial en chroma_db/...", flush=True)
        _reset_collection()

    pdf_files = list_pdf_files(libros_dir)
    if only_filters:
        lowered = [f.lower() for f in only_filters if f.strip()]
        pdf_files = [
            p for p in pdf_files if any(token in p.name.lower() for token in lowered)
        ]

    if not pdf_files:
        meta = get_indexed_metadata()
        result.total_chunks_in_db = meta["total_chunks"]
        result.indexed_books = meta["books"]
        if only_filters:
            result.message = (
                f"No se encontraron PDFs en '{libros_dir}' que coincidan con: {', '.join(only_filters)}."
            )
        else:
            result.message = (
                f"No se encontraron archivos PDF en '{libros_dir}'. "
                "Agregá tus libros .pdf en esa carpeta y volvé a ejecutar la ingesta."
            )
        if verbose:
            print(result.message)
        return result

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    provider = get_embedding_provider()
    use_rate_limit = rate_limit and (provider == "gemini")
    rate_limiter = SlidingWindowRateLimiter(enabled=use_rate_limit)

    if verbose:
        mode_desc = (
            "local ONNX (Xenova/multilingual-e5-small, sin límites de API)"
            if provider == "local"
            else "Gemini API (con control de cuota Free Tier)"
        )
        print(f"Motor de embeddings activo: {mode_desc}\n", flush=True)

    embeddings = None
    vectorstore = None

    for pdf_path in pdf_files:
        file_hash = _compute_file_hash(pdf_path)
        prefix = f"{pdf_path.name}::{file_hash[:12]}::"
        existing_ids = [] if reset else _get_existing_ids_for_source(pdf_path.name)

        # Si hay IDs viejos de un hash distinto o formato anterior (UUID), limpiar antes de re-indexar
        has_legacy_or_stale_ids = bool(existing_ids) and not all(
            cid.startswith(prefix) for cid in existing_ids
        )
        existing_id_set = set() if has_legacy_or_stale_ids else set(existing_ids)

        if verbose:
            print(f"[LEYENDO] {pdf_path.name}...", flush=True)

        try:
            chunks = _load_and_split_pdf(pdf_path, file_hash, splitter, verbose=verbose)
            if not chunks:
                result.failed_files[pdf_path.name] = (
                    "No se pudo extraer texto (el PDF podría ser escaneado/imagen)."
                )
                if verbose:
                    print(f"  -> Aviso: {result.failed_files[pdf_path.name]}", flush=True)
                continue

            all_ids = [_chunk_id(pdf_path.name, file_hash, idx) for idx in range(len(chunks))]

            if not has_legacy_or_stale_ids and len(existing_id_set) == len(all_ids) and existing_id_set == set(all_ids):
                result.skipped_files.append(pdf_path.name)
                if verbose:
                    print(f"[OMITIDO] {pdf_path.name} (ya indexado completo: {len(chunks)} chunks)", flush=True)
                continue

            if has_legacy_or_stale_ids:
                if verbose:
                    print(f"  -> Limpiando fragmentos previos incompletos/obsoletos de {pdf_path.name}...", flush=True)
                _delete_source_chunks(pdf_path.name)

            pending_pairs = [
                (doc, cid)
                for doc, cid in zip(chunks, all_ids)
                if cid not in existing_id_set
            ]

            if verbose:
                if existing_id_set:
                    print(
                        f"[REANUDANDO] {pdf_path.name}: ya tenía {len(existing_id_set)}/{len(chunks)} chunks; "
                        f"faltan {len(pending_pairs)}.",
                        flush=True,
                    )
                else:
                    print(f"[INDEXANDO] {pdf_path.name} ({len(pending_pairs)} chunks)...", flush=True)

            # Instanciar embeddings solo cuando realmente hay chunks pendientes
            if vectorstore is None:
                embeddings = get_embeddings()
                vectorstore = get_vectorstore(embeddings=embeddings)

            total_batches = (len(pending_pairs) + batch_size - 1) // batch_size
            for i in range(0, len(pending_pairs), batch_size):
                slice_pairs = pending_pairs[i : i + batch_size]
                batch_docs = [pair[0] for pair in slice_pairs]
                batch_ids = [pair[1] for pair in slice_pairs]
                _add_batch_with_retry(
                    vectorstore,
                    batch_docs,
                    batch_ids,
                    rate_limiter=rate_limiter,
                    verbose=verbose,
                )
                result.total_chunks_added += len(batch_docs)
                if verbose:
                    batch_num = (i // batch_size) + 1
                    print(
                        f"  -> Lote {batch_num}/{total_batches} guardado ({len(batch_docs)} chunks)",
                        flush=True,
                    )

            result.processed_files.append(pdf_path.name)
            if verbose:
                print(f"  -> OK: {pdf_path.name} indexado completo ({len(chunks)} chunks).", flush=True)

        except DailyQuotaExceededError as exc:
            result.failed_files[pdf_path.name] = str(exc)
            if verbose:
                print(f"\n[DETENIDO POR CUOTA DIARIA] {exc}", flush=True)
                print(
                    "Volvé a ejecutar `uv run python ingest.py` cuando se renueve la cuota "
                    "(o usá `--only` para priorizar libros específicos): retomará exactamente donde quedó.",
                    flush=True,
                )
            break
        except Exception as exc:
            result.failed_files[pdf_path.name] = str(exc)
            if verbose:
                print(f"  -> ERROR procesando {pdf_path.name}: {exc}", flush=True)

    invalidate_metadata_cache()
    final_meta = get_indexed_metadata(use_cache=False)
    result.total_chunks_in_db = final_meta["total_chunks"]
    result.indexed_books = final_meta["books"]
    result.message = (
        f"Ingesta completada: {len(result.processed_files)} procesados, "
        f"{len(result.skipped_files)} omitidos, "
        f"{len(result.failed_files)} con error/pausa. "
        f"Chunks agregados en esta corrida: {result.total_chunks_added} "
        f"(total en base: {result.total_chunks_in_db})."
    )
    if verbose:
        print(f"\n{result.message}", flush=True)

    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define los argumentos de línea de comandos para ingest.py."""
    parser = argparse.ArgumentParser(
        description="Indexa archivos PDF de libros/ en ChromaDB para el Tutor RAG (incremental y reanudable).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Borra la colección existente en chroma_db/ y re-indexa todos los PDFs desde cero.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="Filtra qué PDFs indexar por subcadena del nombre (se puede repetir, ej: --only Pragmatic --only Clean).",
    )
    parser.add_argument(
        "--libros-dir",
        type=Path,
        default=LIBROS_DIR,
        help=f"Carpeta con archivos PDF (por defecto: {LIBROS_DIR}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Cantidad de chunks por lote al calcular embeddings (por defecto: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--no-rate-limit",
        action="store_true",
        help="Desactiva las pausas automáticas para Free Tier (recomendado solo si usás plan pago en Gemini).",
    )
    return parser.parse_args(argv)


def main() -> int:
    """Punto de entrada CLI."""
    args = parse_args()
    try:
        result = ingest_books(
            reset=args.reset,
            libros_dir=args.libros_dir,
            batch_size=max(1, args.batch_size),
            only_filters=args.only,
            rate_limit=not args.no_rate_limit,
            verbose=True,
        )
        return 1 if result.failed_files and not result.processed_files and result.total_chunks_added == 0 else 0
    except ValueError as exc:
        print(f"Error de configuración: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "\nIngesta pausada por el usuario. Todo lote ya guardado quedó persistido en chroma_db/ "
            "y se reanudará automáticamente en la próxima corrida.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
