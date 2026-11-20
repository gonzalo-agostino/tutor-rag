"""Capa 2 — Razonamiento, reflexión y filtrado inteligente (Agentic RAG).

Basado en Agentic Design Patterns (pág. 218, 220): antes de generar la respuesta final,
un agente crítico evalúa metadatos (libro, página, año/edición), descarta fragmentos
irrelevantes, resuelve conflictos entre fuentes y, si la evidencia es insuficiente,
propone una consulta refinada para re-intentar la recuperación.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config import format_book_title
from generation import ModoTutor, invoke_with_fallback

REFLECTION_SYSTEM_PROMPT = """Eres el Agente de Reflexión y Crítica de un sistema Agentic RAG técnico.
Tu tarea ocurre ANTES de generar la respuesta al usuario. Recibís la pregunta del usuario y una lista numerada de fragmentos recuperados con sus metadatos (título, página, año/edición estimada).

Debés analizar rigurosamente los fragmentos y devolver EXCLUSIVAMENTE un objeto JSON válido con esta estructura exacta:
{
  "indices_relevantes": [1, 3, 4],
  "conflictos_resueltos": "Texto breve describiendo si hay contradicciones o diferencias entre versiones/ediciones/autores y cómo reconciliarlas (o cadena vacía si no hay conflictos).",
  "notas_sintesis": "Indicación breve para el generador sobre qué enfoque priorizar o qué descartar.",
  "requiere_rebusqueda": false,
  "nueva_query_sugerida": ""
}

Reglas estrictas:
1. Descartá fragmentos irrelevantes, índices vacíos, agradecimientos, biografías del autor o textos que no respondan a la consulta.
2. Si existen dos documentos sobre el mismo tema con diferente vigencia (por ejemplo una obra de 2009 frente a un libro de 2024/2025 sobre LLMs/Agentes), priorizá la fuente más reciente o explicá el matiz en `conflictos_resueltos`.
3. `indices_relevantes` debe contener los números (1-based) de los fragmentos que SÍ aportan evidencia sólida, ordenados del más útil al menos útil (máximo 6).
4. Poné `requiere_rebusqueda: true` ÚNICAMENTE si casi ningún fragmento sirve (0 o 1 fragmento muy débil) y creés que otra query técnica (en inglés + español) recuperará mejores resultados; en ese caso escribí `nueva_query_sugerida`.
5. No incluyas texto fuera del JSON ni bloques Markdown ```json.
"""

# Estrategias que no deben ser recortadas por el filtro de relevancia puntual
BYPASS_STRATEGIES = {"resumen_libro", "lectura_rapida_azar", "roadmap", "sin_datos"}


class ReflectionResult(BaseModel):
    """Resultado estructurado de la capa de reflexión previa a la generación."""

    ejecutada: bool = False
    indices_relevantes: list[int] = Field(default_factory=list)
    chunks_iniciales: int = 0
    chunks_retenidos: int = 0
    chunks_descartados: int = 0
    conflictos_resueltos: str = ""
    notas_sintesis: str = ""
    requiere_rebusqueda: bool = False
    nueva_query_sugerida: str = ""


def _infer_year_or_edition(filename: str) -> str:
    """Extrae pistas de año o edición desde el nombre del archivo para ayudar a resolver conflictos."""
    hints: list[str] = []
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", filename)
    if year_match:
        hints.append(f"año {year_match.group(1)}")
    ed_match = re.search(r"\b(\d+(?:st|nd|rd|th|ed))\b", filename, re.IGNORECASE)
    if ed_match:
        hints.append(f"edición {ed_match.group(1)}")
    return ", ".join(hints) if hints else "edición estándar"


def _parse_json_object(raw_text: str) -> dict[str, Any] | None:
    """Extrae el primer objeto JSON válido incluso si el modelo agrega fences Markdown."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def reflect_and_filter(
    mensaje: str,
    modo: ModoTutor,
    documentos: list[Document],
    estrategia: str,
    intento: int = 0,
) -> tuple[list[Document], ReflectionResult]:
    """Evalúa metadatos y relevancia de candidatos; filtra ruido y detecta conflictos."""
    total = len(documentos)
    if total == 0:
        return [], ReflectionResult(ejecutada=False, chunks_iniciales=0, chunks_retenidos=0)

    if estrategia in BYPASS_STRATEGIES:
        return documentos, ReflectionResult(
            ejecutada=False,
            indices_relevantes=list(range(1, total + 1)),
            chunks_iniciales=total,
            chunks_retenidos=total,
            chunks_descartados=0,
        )

    candidates_lines: list[str] = []
    for idx, doc in enumerate(documentos, start=1):
        source = str(doc.metadata.get("source", "desconocido"))
        title = format_book_title(source)
        page = doc.metadata.get("page", "?")
        version_hint = _infer_year_or_edition(source)
        excerpt = " ".join((doc.page_content or "").split())[:520]
        candidates_lines.append(
            f"[{idx}] Libro: {title} | Pág.: {page} | Metadatos: {version_hint}\n"
            f"Extracto: {excerpt}"
        )

    user_prompt = (
        f"MODO: {modo}\n"
        f"INTENTO DE RECUPERACIÓN: {intento + 1}\n"
        f"CONSULTA DEL USUARIO: {mensaje}\n\n"
        f"FRAGMENTOS CANDIDATOS:\n" + "\n\n".join(candidates_lines)
    )

    try:
        raw = invoke_with_fallback(
            [
                SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ],
            temperature=0.0,
            timeout=20.0,
        )
        parsed = _parse_json_object(raw)
        if not parsed:
            raise ValueError("Respuesta de reflexión no parseable como JSON.")
    except Exception:
        # Fallback seguro: conservar los candidatos originales si el crítico falla por red
        return documentos, ReflectionResult(
            ejecutada=False,
            indices_relevantes=list(range(1, total + 1)),
            chunks_iniciales=total,
            chunks_retenidos=total,
            chunks_descartados=0,
        )

    raw_indices = parsed.get("indices_relevantes") or []
    valid_indices: list[int] = []
    for item in raw_indices:
        try:
            idx_int = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= idx_int <= total and idx_int not in valid_indices:
            valid_indices.append(idx_int)

    requiere_rebusqueda = bool(parsed.get("requiere_rebusqueda", False)) and intento < 1
    nueva_query = str(parsed.get("nueva_query_sugerida") or "").strip()
    if requiere_rebusqueda and not nueva_query:
        requiere_rebusqueda = False

    # Si no se va a re-buscar y el filtro vació todo, conservar los 3 mejores para que el generador responda con transparencia
    if not valid_indices and not requiere_rebusqueda:
        valid_indices = list(range(1, min(3, total) + 1))

    filtered_docs = [documentos[i - 1] for i in valid_indices[:6]]
    result = ReflectionResult(
        ejecutada=True,
        indices_relevantes=valid_indices[:6],
        chunks_iniciales=total,
        chunks_retenidos=len(filtered_docs),
        chunks_descartados=max(0, total - len(filtered_docs)),
        conflictos_resueltos=str(parsed.get("conflictos_resueltos") or "").strip(),
        notas_sintesis=str(parsed.get("notas_sintesis") or "").strip(),
        requiere_rebusqueda=requiere_rebusqueda,
        nueva_query_sugerida=nueva_query,
    )
    return filtered_docs, result
