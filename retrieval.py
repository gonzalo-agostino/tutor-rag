"""Capa 1 — Recuperación pasiva (ChromaDB, búsqueda bilingüe, multi-libro y contextualización)."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from config import (
    COLLECTION_NAME,
    clean_pdf_text,
    format_book_title,
    get_chroma_client,
    get_embeddings,
    get_indexed_metadata,
    get_total_chunks,
    get_vectorstore,
)
from generation import ModoTutor, invoke_with_fallback

GENERIC_QUICK_READ_TRIGGERS = {
    "",
    "lectura rapida",
    "lectura rápida",
    "dame una lectura rapida",
    "dame una lectura rápida",
    "lectura rapida de hoy",
    "lectura rápida de hoy",
    "explorar concepto",
    "sorprendeme",
    "sorpréndeme",
    "leccion de hoy",
    "lección de hoy",
}


@dataclass
class RetrievalBundle:
    """Resultado de la capa de recuperación pasiva."""

    documentos: list[Document] = field(default_factory=list)
    query_efectiva: str = ""
    inferred_books: list[str] = field(default_factory=list)
    estrategia: str = "estandar"
    is_generic_quick_read: bool = False
    is_roadmap: bool = False

    def __post_init__(self) -> None:
        cleaned_docs: list[Document] = []
        for doc in self.documentos:
            cleaned_text = clean_pdf_text(doc.page_content or "")
            if cleaned_text:
                cleaned_docs.append(
                    Document(page_content=cleaned_text, metadata=dict(doc.metadata or {}))
                )
        self.documentos = cleaned_docs


def is_generic_quick_read_request(mensaje: str) -> bool:
    """Detecta si el usuario pidió una Lectura rápida al azar sin especificar un tema concreto."""
    normalized = " ".join(mensaje.lower().strip().rstrip(".!?").split())
    return normalized in GENERIC_QUICK_READ_TRIGGERS


def is_compare_authors_request(mensaje: str) -> bool:
    """Detecta consultas comparativas para diversificar fragmentos entre varios libros."""
    lower = mensaje.lower()
    return any(k in lower for k in ("compar", "distintos libros", "diferentes libros", "autores"))


def is_library_roadmap_request(mensaje: str) -> bool:
    """Detecta pedidos de roadmap, orden de lectura, plan de estudio o catálogo completo."""
    lower = mensaje.lower()
    triggers = (
        "roadmap",
        "orden de lectura",
        "orden de libros",
        "orden para leer",
        "por cuál empezar",
        "por cual empezar",
        "qué libro empezar",
        "que libro empezar",
        "ruta de estudio",
        "plan de estudio",
        "secuencia de lectura",
        "qué libros tengo",
        "que libros tengo",
        "lista de libros",
    )
    return any(t in lower for t in triggers)


def is_book_summary_request(mensaje: str) -> bool:
    """Detecta pedidos de resumen general, índice o estructura completa de un libro."""
    lower = mensaje.lower()
    triggers = (
        "resumen del libro",
        "resumen de este libro",
        "resumen completo",
        "resumime el libro",
        "resumí el libro",
        "resumi el libro",
        "de qué trata",
        "de que trata",
        "índice del libro",
        "indice del libro",
        "capítulos del libro",
        "capitulos del libro",
        "estructura del libro",
        "visión general del libro",
        "vision general del libro",
        "temario del libro",
        "resumen de ",
    )
    return any(t in lower for t in triggers)


def _match_books_mentioned_in_text(text: str) -> list[str]:
    """Detecta qué archivos PDF de la biblioteca aparecen mencionados (por título o nombre)."""
    if not text:
        return []
    lower = text.lower()
    matched: list[str] = []
    for filename in get_indexed_metadata().get("books") or []:
        title = format_book_title(filename).lower()
        base_title = re.sub(r"\s*\([^)]*\)", "", title).strip()
        if title in lower or (len(base_title) >= 8 and base_title in lower) or filename.lower() in lower:
            matched.append(filename)
    return matched


def _extract_referenced_section(mensaje: str, last_assistant: str) -> str:
    """Si el usuario refiere a una fase/sección/punto del mensaje previo, extrae ese bloque exacto."""
    if not last_assistant:
        return ""
    match = re.search(r"\b(fase|etapa|paso|módulo|modulo|parte|nivel)\s*(\d+)\b", mensaje, re.IGNORECASE)
    if not match:
        return ""
    label, num = match.group(1), match.group(2)
    pattern = re.compile(rf"(?im)(?:^|\n)[^\n]*\b{re.escape(label)}\s*{re.escape(num)}\b[^\n]*")
    found = pattern.search(last_assistant)
    if not found:
        return ""
    start = found.start()
    next_heading = re.search(
        rf"(?im)\n(?:#+\s+|\*\*)?(?:{re.escape(label)}\s*(?!{re.escape(num)}\b)\d+|##\s+)",
        last_assistant[found.end() :],
    )
    end = found.end() + next_heading.start() if next_heading else min(len(last_assistant), start + 1800)
    return last_assistant[start:end].strip()


def _contextualize_followup(
    mensaje: str,
    historial: list[dict[str, str]],
) -> tuple[str, list[str]]:
    """Reescribe repreguntas cortas usando el historial y detecta libros referenciados."""
    if not historial:
        return mensaje, _match_books_mentioned_in_text(mensaje)

    last_assistant = ""
    for turn in reversed(historial):
        if (turn.get("role") or "").lower() == "assistant":
            last_assistant = (turn.get("content") or "").strip()
            break

    referenced_section = _extract_referenced_section(mensaje, last_assistant)
    if referenced_section:
        books_in_section = _match_books_mentioned_in_text(referenced_section)
        return f"{mensaje}\n{referenced_section[:800]}", books_in_section

    needs_rewrite = len(mensaje) < 110 or any(
        w in mensaje.lower()
        for w in (
            "esto",
            "eso",
            "ese",
            "esa",
            "ellos",
            "anterior",
            "arriba",
            "profundiz",
            "ampli",
            "ejemplo",
            "más sobre",
            "mas sobre",
        )
    )
    if not needs_rewrite:
        return mensaje, _match_books_mentioned_in_text(mensaje)

    recent_history = "\n".join(
        f"{(t.get('role') or 'user').upper()}: {(t.get('content') or '')[:1500]}"
        for t in historial[-4:]
    )
    rewrite_messages: list[BaseMessage] = [
        SystemMessage(
            content=(
                "Reescribí la última pregunta del usuario como una consulta técnica autónoma para buscar en una base vectorial "
                "de libros de ingeniería de software e IA (en inglés y español). "
                "Incluí los conceptos técnicos clave en español e inglés y los nombres exactos de los libros mencionados si aplica. "
                "Devolvé SOLO la consulta reescrita en 1 o 2 líneas, sin explicaciones."
            )
        ),
        HumanMessage(
            content=f"HISTORIAL RECIENTE:\n{recent_history}\n\nÚLTIMA PREGUNTA DEL USUARIO:\n{mensaje}"
        ),
    ]
    try:
        rewritten = invoke_with_fallback(rewrite_messages, temperature=0.0)
        if rewritten:
            return rewritten, _match_books_mentioned_in_text(rewritten)
    except Exception:
        pass

    return mensaje, _match_books_mentioned_in_text(mensaje)


def _sample_random_anchor_document(libro: str | None) -> Document | None:
    """Selecciona un fragmento aleatorio de ChromaDB en O(1) para variar la Lectura rápida."""
    client = get_chroma_client()
    existing_names = {col.name for col in client.list_collections()}
    if COLLECTION_NAME not in existing_names:
        return None

    collection = client.get_collection(name=COLLECTION_NAME)
    if libro:
        raw_ids = collection.get(where={"source": libro}, include=[])
        ids: list[str] = raw_ids.get("ids") or []
        if not ids:
            return None
        chosen_id = random.choice(ids)
        fetched = collection.get(ids=[chosen_id], include=["documents", "metadatas"])
    else:
        total = collection.count()
        if total == 0:
            return None
        offset = random.randrange(total)
        fetched = collection.get(limit=1, offset=offset, include=["documents", "metadatas"])

    docs = fetched.get("documents") or []
    metas = fetched.get("metadatas") or []
    if not docs or not docs[0]:
        return None

    return Document(page_content=str(docs[0]), metadata=dict(metas[0] or {}))


def _retrieve_book_summary_chunks(book_filename: str, max_chunks: int = 16) -> list[Document]:
    """Recupera índice/prefacio + muestras distribuidas en todo el libro para resumirlo entero."""
    client = get_chroma_client()
    collection = client.get_collection(name=COLLECTION_NAME)
    data = collection.get(
        where={"source": book_filename},
        include=["documents", "metadatas"],
    )
    docs_raw = data.get("documents") or []
    metas_raw = data.get("metadatas") or []
    if not docs_raw:
        return []

    paired: list[tuple[int, int, Document]] = []
    for idx, (text, meta) in enumerate(zip(docs_raw, metas_raw)):
        m = dict(meta or {})
        try:
            page = int(m.get("page", 0))
        except (TypeError, ValueError):
            page = 0
        paired.append((page, idx, Document(page_content=str(text), metadata=m)))

    paired.sort(key=lambda item: (item[0], item[1]))
    all_docs = [item[2] for item in paired]

    if len(all_docs) <= max_chunks:
        return all_docs

    selected: list[Document] = []
    seen_indices: set[int] = set()

    toc_keywords = ("table of contents", "contents", "capítulo", "chapter 1", "preface", "prefacio", "contenido")
    for idx, doc in enumerate(all_docs[:45]):
        low = doc.page_content.lower()
        if any(k in low for k in toc_keywords):
            selected.append(doc)
            seen_indices.add(idx)
            if len(selected) >= 6:
                break

    remaining = max_chunks - len(selected)
    if remaining > 0:
        step = max(1, len(all_docs) // remaining)
        for pos in range(0, len(all_docs), step):
            if pos not in seen_indices:
                selected.append(all_docs[pos])
                seen_indices.add(pos)
                if len(selected) >= max_chunks:
                    break

    selected.sort(key=lambda d: int(d.metadata.get("page", 0)))
    return selected


def _retrieve_multi_book_comparison(
    query_text: str,
    limit: int = 7,
    target_books: list[str] | None = None,
) -> list[Document]:
    """Busca el mejor fragmento por libro (1 embedding + consultas filtradas) para cubrir varios libros."""
    books = target_books or (get_indexed_metadata().get("books") or [])
    if not books:
        return []

    client = get_chroma_client()
    collection = client.get_collection(name=COLLECTION_NAME)
    query_vec = get_embeddings().embed_query(query_text)

    scored: list[tuple[float, Document]] = []
    for book in books:
        res = collection.query(
            query_embeddings=[query_vec],
            n_results=1,
            where={"source": book},
            include=["documents", "metadatas", "distances"],
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        if docs and docs[0]:
            dist = float(dists[0]) if dists else 1.0
            scored.append(
                (dist, Document(page_content=str(docs[0]), metadata=dict(metas[0] or {})))
            )

    scored.sort(key=lambda item: item[0])
    return [doc for _, doc in scored[:limit]]


def _translate_query_to_english(query_text: str) -> str | None:
    """Genera palabras clave técnicas en inglés (~0.5s) para equilibrar búsquedas globales."""
    try:
        translated = invoke_with_fallback(
            [
                SystemMessage(
                    content=(
                        "Traducí esta consulta técnica al inglés como una query corta de búsqueda "
                        "para libros de ingeniería de software e IA. Devolvé SOLO la query en inglés, sin comillas ni explicaciones."
                    )
                ),
                HumanMessage(content=query_text[:500]),
            ],
            temperature=0.0,
        )
        return translated.strip() if translated else None
    except Exception:
        return None


def _retrieve_bilingual_balanced(
    vectorstore,
    query_text: str,
    k: int = 8,
    max_per_book: int = 2,
) -> list[Document]:
    """Recupera candidatos en español + inglés (máx. 2 por libro) para que la capa de reflexión filtre."""
    es_docs = vectorstore.similarity_search(query=query_text, k=8)
    en_query = _translate_query_to_english(query_text)
    en_docs = vectorstore.similarity_search(query=en_query, k=8) if en_query else []

    interleaved: list[Document] = []
    for i in range(max(len(es_docs), len(en_docs))):
        if i < len(en_docs):
            interleaved.append(en_docs[i])
        if i < len(es_docs):
            interleaved.append(es_docs[i])

    selected: list[Document] = []
    seen_texts: set[str] = set()
    counts: dict[str, int] = {}
    for doc in interleaved:
        txt = doc.page_content or ""
        book = str(doc.metadata.get("source", ""))
        if txt in seen_texts or counts.get(book, 0) >= max_per_book:
            continue
        selected.append(doc)
        seen_texts.add(txt)
        counts[book] = counts.get(book, 0) + 1
        if len(selected) >= k:
            break
    return selected


def retrieve_candidates(
    mensaje: str,
    modo: ModoTutor,
    libro: str | None = None,
    historial: list[dict[str, str]] | None = None,
    override_query: str | None = None,
) -> RetrievalBundle:
    """Ejecuta la capa de recuperación pasiva devolviendo candidatos para la reflexión del agente."""
    if get_total_chunks() == 0:
        return RetrievalBundle(documentos=[], query_efectiva=mensaje, estrategia="sin_datos")

    clean_msg = (mensaje or "").strip()
    libro_seleccionado = (libro or "").strip() or None
    search_filter = {"source": libro_seleccionado} if libro_seleccionado else None
    vectorstore = get_vectorstore()

    is_generic_qr = modo == "lectura_rapida" and is_generic_quick_read_request(clean_msg)
    is_roadmap = modo == "consulta" and is_library_roadmap_request(clean_msg)

    if is_generic_qr and not override_query:
        anchor = _sample_random_anchor_document(libro_seleccionado)
        if anchor is None:
            return RetrievalBundle(documentos=[], query_efectiva=clean_msg, estrategia="lectura_rapida_azar", is_generic_quick_read=True)
        anchor_source = str(anchor.metadata.get("source", ""))
        anchor_filter = {"source": anchor_source} if anchor_source else search_filter
        similar = vectorstore.similarity_search(
            query=anchor.page_content[:500],
            k=4,
            filter=anchor_filter,
        )
        seen_texts = {anchor.page_content}
        docs = [anchor]
        for d in similar:
            if d.page_content not in seen_texts:
                seen_texts.add(d.page_content)
                docs.append(d)
        return RetrievalBundle(
            documentos=docs,
            query_efectiva=clean_msg,
            estrategia="lectura_rapida_azar",
            is_generic_quick_read=True,
        )

    if override_query:
        query_text = override_query.strip()
        inferred_books = _match_books_mentioned_in_text(query_text)
    else:
        query_text, inferred_books = _contextualize_followup(clean_msg, historial or [])

    if not query_text:
        query_text = "conceptos fundamentales arquitectura mejores practicas ejemplo"

    summary_target = libro_seleccionado or (
        inferred_books[0] if len(inferred_books) == 1 else None
    )
    if modo == "consulta" and summary_target and is_book_summary_request(clean_msg):
        docs = _retrieve_book_summary_chunks(summary_target, max_chunks=16)
        return RetrievalBundle(
            documentos=docs,
            query_efectiva=query_text,
            inferred_books=[summary_target],
            estrategia="resumen_libro",
        )

    if is_roadmap:
        docs = vectorstore.similarity_search(query=query_text, k=4)
        return RetrievalBundle(
            documentos=docs,
            query_efectiva=query_text,
            estrategia="roadmap",
            is_roadmap=True,
        )

    if modo == "consulta" and libro_seleccionado is None and inferred_books and not override_query:
        if len(inferred_books) == 1:
            docs = vectorstore.similarity_search(
                query=query_text,
                k=7,
                filter={"source": inferred_books[0]},
            )
            estrategia = "libro_inferido"
        else:
            docs = _retrieve_multi_book_comparison(
                query_text,
                limit=min(8, len(inferred_books)),
                target_books=inferred_books,
            )
            estrategia = "multi_libro_inferido"
        return RetrievalBundle(
            documentos=docs,
            query_efectiva=query_text,
            inferred_books=inferred_books,
            estrategia=estrategia,
        )

    if modo == "consulta" and libro_seleccionado is None and is_compare_authors_request(clean_msg):
        docs = _retrieve_multi_book_comparison(query_text, limit=8)
        return RetrievalBundle(
            documentos=docs,
            query_efectiva=query_text,
            estrategia="comparacion_autores",
        )

    if modo == "consulta" and libro_seleccionado is None:
        docs = _retrieve_bilingual_balanced(vectorstore, query_text, k=8, max_per_book=2)
        return RetrievalBundle(
            documentos=docs,
            query_efectiva=query_text,
            estrategia="bilingue_equilibrada",
        )

    k = 7 if modo == "consulta" else 5
    docs = vectorstore.similarity_search(
        query=query_text,
        k=k,
        filter=search_filter,
    )
    return RetrievalBundle(
        documentos=docs,
        query_efectiva=query_text,
        estrategia="libro_unico" if libro_seleccionado else "estandar",
    )
