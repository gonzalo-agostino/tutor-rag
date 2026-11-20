"""Capa 3 — Motor de generación final, prompts editoriales y gestión de citas."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from config import format_book_title, get_indexed_metadata, get_llm, get_llm_candidates

ModoTutor = Literal["lectura_rapida", "consulta"]

LECTURA_RAPIDA_SYSTEM_PROMPT = """Eres un Tutor Técnico RAG senior, preciso y pedagógico.
Tu objetivo es redactar una **Lectura rápida de 5 minutos** basada EXCLUSIVAMENTE en los fragmentos recuperados de la biblioteca técnica del usuario.

Reglas obligatorias:
1. Respondé siempre en español claro, editorial y cómodo de leer en iPad o escritorio.
2. Usá estrictamente la información contenida en los fragmentos proporcionados. No inventes conceptos ni afirmaciones fuera del contexto.
3. Citá SIEMPRE las fuentes dentro del texto usando el formato exacto `[Título del Libro — Pág. X]` tal como aparece en el encabezado de cada fragmento (por ejemplo: `[Clean Code — Pág. 42]`), sin extensiones `.pdf` ni nombres de archivo crudos.
4. Estructurá la respuesta EXACTAMENTE con estas tres secciones en Markdown:

## Resumen directo
- Explicación concisa y clara del concepto clave elegido, cuándo aplicarlo y qué problema resuelve (con citas `[Título del Libro — Pág. X]`).
- Incluí 2 o 3 viñetas con ideas clave accionables.

## Ejemplo de código práctico
- Un bloque de código limpio, idiomático y listo para producción (en Python, Java, TypeScript o SQL según el contexto del libro) con etiqueta de lenguaje en el fence (por ejemplo ```python).
- Explicá brevemente cómo el ejemplo aplica el principio citado `[Título del Libro — Pág. X]`.
- Cuando el concepto involucre arquitecturas, flujos de datos, estados o patrones complejos, incluí también un bloque ```mermaid claro y sintácticamente válido (por ejemplo `flowchart TD`, `sequenceDiagram`, `classDiagram` o `stateDiagram-v2`) encerrando entre comillas dobles las etiquetas con paréntesis o caracteres especiales, ej: `A["Servicio API (REST)"]`.

## Pregunta de autoevaluación
- Formulá **1 sola pregunta** concreta y desafiante para verificar la comprensión del concepto.
- No reveles la respuesta todavía; invitá al lector a responder en el chat para recibir corrección fundada en el libro.
"""

CONSULTA_SYSTEM_PROMPT = """Eres un Tutor Técnico RAG riguroso y didáctico, diseñado para ingeniería de software y ciencia de datos.
Respondés preguntas técnicas, comparaciones entre autores, aplicaciones a código y autoevaluaciones basándote **ESTRICTAMENTE** en los fragmentos verificados por la capa de reflexión.

Reglas obligatorias:
1. Respondé en español con Markdown editorial limpio: títulos breves, párrafos claros, viñetas y bloques de código etiquetados (`python`, `java`, `typescript`, `sql`, etc.) cuando correspondan.
2. Basá TODA afirmación técnica exclusivamente en los fragmentos del contexto proporcionado.
3. Citá siempre las fuentes usando el formato exacto `[Título del Libro — Pág. X]` tal como aparece en el encabezado de cada fragmento (sin extensiones `.pdf` ni guiones de archivo).
4. Si los fragmentos recuperados NO contienen información suficiente para responder la consulta, indicalo con claridad ("No encontré información suficiente sobre esto en los libros seleccionados...") sin inventar contenido externo.
5. Si el usuario pide **comparar autores o libros**, contrastá explícitamente qué plantea cada obra representada en los fragmentos, señalando coincidencias, matices y citas `[Título del Libro — Pág. X]` de cada libro.
6. Si el usuario pide **aplicar a código**, mostrá una implementación limpia, modular y lista para producción fundamentada en los principios de los fragmentos citados.
7. Si el usuario está respondiendo una pregunta previa de autoevaluación, evaluá su respuesta frente a los fragmentos del libro indicando aciertos y mejoras con citas `[Título del Libro — Pág. X]`.
8. Cuando expliques arquitecturas, flujos de datos, secuencias o patrones de diseño complejos, incluí un bloque ```mermaid claro y sintácticamente válido (`flowchart TD`, `flowchart LR`, `sequenceDiagram`, `classDiagram` o `stateDiagram-v2`) para ilustrarlo visualmente. Encerrá siempre entre comillas dobles los textos de nodos con paréntesis, dos puntos o símbolos (ej: `API["Gateway (HTTP)"]`) y evitá etiquetas HTML dentro de Mermaid.
9. Si la capa de reflexión señaló **conflictos entre documentos o versiones** (por ejemplo ediciones más antiguas frente a obras más recientes), resolvelos de forma explícita y transparente en tu explicación.
10. **Excepción para consultas sobre la biblioteca (Roadmap / Orden de lectura / Plan de estudio / Qué libros tengo):** Cuando el usuario pregunte qué libros tiene, por cuál empezar o pida un **roadmap / orden de lectura**, usá el **CATÁLOGO COMPLETO DE LA BIBLIOTECA INDEXADA** provisto en el contexto para armar una ruta progresiva por fases (Fundamentos → Arquitectura/Datos → Machine Learning → AI Engineering/Agentes), usando EXCLUSIVAMENTE los títulos de esa biblioteca, explicando brevemente por qué va cada libro en esa etapa e incluyendo un diagrama `flowchart TD` en ```mermaid.
"""


class Cita(BaseModel):
    """Referencia estructurada a un fragmento recuperado del PDF."""

    libro: str
    titulo: str = ""
    pagina: int
    extracto: str = ""


def extract_text_from_content(content: Any) -> str:
    """Normaliza la salida de AIMessage.content (str o lista de bloques)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


def invoke_with_fallback(
    messages: list[BaseMessage],
    temperature: float = 0.3,
    timeout: float = 30.0,
) -> str:
    """Invoca Gemini Flash probando fallbacks rápidos si el modelo principal da 503/504/404."""
    override_primary: str | None = None
    try:
        from db import get_runtime_settings

        override_primary = str(get_runtime_settings().get("llm_model") or "").strip() or None
    except Exception:
        override_primary = None

    last_exc: Exception | None = None
    for model_name in get_llm_candidates(override_primary=override_primary):
        try:
            llm = get_llm(
                temperature=temperature,
                model_name=model_name,
                timeout=timeout,
                max_retries=1,
            )
            ai_msg = llm.invoke(messages)
            return extract_text_from_content(ai_msg.content)
        except Exception as exc:
            last_exc = exc
            upper = str(exc).upper()
            retryable = any(
                code in upper
                for code in ("503", "504", "404", "429", "UNAVAILABLE", "DEADLINE", "NOT_FOUND", "OVERLOADED")
            )
            if not retryable:
                raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No hay modelos Gemini disponibles para generar la respuesta.")


def format_context(documentos: list[Document]) -> str:
    """Formatea los fragmentos filtrados usando títulos limpios y número de página."""
    blocks: list[str] = []
    for idx, doc in enumerate(documentos, start=1):
        source = str(doc.metadata.get("source", "Libro desconocido"))
        title = format_book_title(source)
        page = doc.metadata.get("page", "?")
        content = (doc.page_content or "").strip()
        blocks.append(f"### Fragmento {idx} — [{title} — Pág. {page}]\n{content}")
    return "\n\n---\n\n".join(blocks)


def build_citations(documentos: list[Document]) -> list[dict[str, Any]]:
    """Devuelve todos los fragmentos retenidos tras la reflexión (sin fusionar páginas distintas/múltiples chunks)."""
    citas: list[dict[str, Any]] = []
    seen_exact: set[tuple[str, int, str]] = set()

    for doc in documentos:
        libro = str(doc.metadata.get("source", "Libro desconocido"))
        titulo = format_book_title(libro)
        try:
            pagina = int(doc.metadata.get("page", 1))
        except (TypeError, ValueError):
            pagina = 1

        raw_fragment = (doc.page_content or "").strip()
        dedup_key = (libro, pagina, raw_fragment)
        if dedup_key in seen_exact:
            continue
        seen_exact.add(dedup_key)

        citas.append(
            {
                "libro": libro,
                "titulo": titulo,
                "pagina": pagina,
                "extracto": raw_fragment,
            }
        )

    return citas


def format_library_catalog() -> str:
    """Devuelve el catálogo completo formateado de todos los libros indexados en ChromaDB."""
    books = get_indexed_metadata().get("books") or []
    if not books:
        return "No hay libros indexados todavía."
    return "\n".join(f"- {format_book_title(b)}" for b in books)


def build_history_messages(historial: list[dict[str, str]]) -> list[BaseMessage]:
    """Convierte los últimos turnos del historial en mensajes LangChain."""
    messages: list[BaseMessage] = []
    for turn in historial[-6:]:
        role = (turn.get("role") or "").lower()
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def build_no_context_message(libro: str | None, notas: str = "") -> str:
    """Mensaje de respaldo cuando no hay fragmentos relevantes o la base está vacía."""
    clean_book = (libro or "").strip()
    clean_notes = (notas or "").strip()
    notes_block = f"\n\n> **Diagnóstico del agente:** {clean_notes}" if clean_notes else ""
    if clean_book:
        return (
            f"No encontré fragmentos relevantes para esta consulta dentro de **{format_book_title(clean_book)}**."
            f"{notes_block}\n\n"
            "Probá cambiar el selector de alcance a **Todos los libros** o reformular la pregunta con términos técnicos más específicos."
        )
    return (
        "No encontré fragmentos suficientemente relevantes en tu biblioteca indexada para responder esta consulta sin inventar información."
        f"{notes_block}\n\n"
        "Probá reformular la pregunta con conceptos técnicos más concretos o verificar si el tema está cubierto por los PDFs de `libros/`."
    )


CONVERSATIONAL_SYSTEM_PROMPT = """Eres un Tutor Técnico RAG cercano, claro y conciso.
El usuario acaba de enviar un mensaje puramente conversacional (saludo, agradecimiento, confirmación o comentario breve) que NO requiere buscar en los PDFs.

Reglas:
1. Respondé en español natural y breve (2 a 4 líneas máximo).
2. Tené en cuenta el historial de la conversación si el usuario está confirmando o comentando algo anterior.
3. No inventes citas ni referencias a páginas de libros.
4. Cerrá invitándolo con una sugerencia concreta a hacer la siguiente pregunta técnica o explorar un tema de su biblioteca.
"""


def generate_conversational_reply(
    mensaje: str,
    historial: list[dict[str, str]],
    libro: str | None = None,
) -> str:
    """Responde turnos puramente conversacionales sin disparar búsqueda vectorial ni reflexión."""
    messages: list[BaseMessage] = [SystemMessage(content=CONVERSATIONAL_SYSTEM_PROMPT)]
    messages.extend(build_history_messages(historial))
    scope_hint = f" (alcance actual: {format_book_title(libro)})" if (libro or "").strip() else ""
    messages.append(HumanMessage(content=f"{mensaje}{scope_hint}"))
    return invoke_with_fallback(messages, temperature=0.4, timeout=20.0)


def generate_quick_read(
    pedido: str,
    documentos: list[Document],
    is_generic: bool,
    notas_reflexion: str = "",
) -> str:
    """Genera una Lectura rápida de 5 minutos usando los fragmentos filtrados."""
    contexto = format_context(documentos)
    notas_block = f"\n\nNOTAS DE LA CAPA DE REFLEXIÓN:\n{notas_reflexion}" if notas_reflexion else ""

    if is_generic:
        user_prompt = (
            "Generá una Lectura rápida eligiendo el concepto técnico más valioso de estos fragmentos."
            f"{notas_block}\n\nFRAGMENTOS DE LOS LIBROS:\n{contexto}"
        )
    else:
        user_prompt = (
            f'Generá una Lectura rápida enfocada en el pedido del lector: "{pedido}".'
            f"{notas_block}\n\nFRAGMENTOS DE LOS LIBROS:\n{contexto}"
        )

    messages: list[BaseMessage] = [
        SystemMessage(content=LECTURA_RAPIDA_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    return invoke_with_fallback(messages, temperature=0.4)


def generate_consultation(
    pedido: str,
    documentos: list[Document],
    historial: list[dict[str, str]],
    is_roadmap: bool = False,
    conflictos_resueltos: str = "",
    notas_reflexion: str = "",
) -> str:
    """Genera la respuesta técnica final integrando fragmentos verificados y notas de reflexión."""
    contexto = format_context(documentos)
    messages: list[BaseMessage] = [SystemMessage(content=CONSULTA_SYSTEM_PROMPT)]
    messages.extend(build_history_messages(historial))

    reflection_parts: list[str] = []
    if conflictos_resueltos:
        reflection_parts.append(f"- Conflictos / versiones resueltos: {conflictos_resueltos}")
    if notas_reflexion:
        reflection_parts.append(f"- Guía de síntesis: {notas_reflexion}")
    reflection_block = (
        "\n\nANÁLISIS PREVIO DEL AGENTE DE REFLEXIÓN:\n" + "\n".join(reflection_parts)
        if reflection_parts
        else ""
    )

    if is_roadmap:
        catalogo = format_library_catalog()
        user_content = (
            f"CONSULTA DEL LECTOR:\n{pedido}\n\n"
            f"CATÁLOGO COMPLETO DE LA BIBLIOTECA INDEXADA:\n{catalogo}"
            f"{reflection_block}\n\n"
            f"FRAGMENTOS DE REFERENCIA (opcionales para contexto):\n{contexto}"
        )
    else:
        user_content = (
            f"CONSULTA DEL LECTOR:\n{pedido}"
            f"{reflection_block}\n\n"
            f"FRAGMENTOS VERIFICADOS DE LOS LIBROS:\n{contexto}"
        )

    messages.append(HumanMessage(content=user_content))
    return invoke_with_fallback(messages, temperature=0.2)
