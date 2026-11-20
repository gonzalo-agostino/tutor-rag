"""Orquestador LangGraph de Agentic RAG.

Conecta las capas modulares:
1. `retrieval.py`: Recuperación pasiva desde ChromaDB.
2. `reflection.py`: Razonamiento, filtrado de metadatos, resolución de conflictos y re-búsqueda adaptativa.
3. `generation.py`: Motor de generación final y citas verificables.
4. `evaluation.py`: Observabilidad en tiempo de ejecución (DeepEval: faithfulness y context_precision).
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Literal, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from generation import (
    Cita,
    ModoTutor,
    build_citations,
    build_no_context_message,
    generate_consultation,
    generate_quick_read,
)
from reflection import ReflectionResult, reflect_and_filter
from retrieval import retrieve_candidates


class AgenticTrace(BaseModel):
    """Telemetría de ejecución de las capas de recuperación, reflexión y generación."""

    estrategia: str = "estandar"
    reflexion_ejecutada: bool = False
    chunks_iniciales: int = 0
    chunks_retenidos: int = 0
    chunks_descartados: int = 0
    rebusqueda_activada: bool = False
    conflictos_resueltos: str = ""
    latencia_ms: int = 0
    retrieval_ms: int = 0
    reflection_ms: int = 0
    generation_ms: int = 0


class TutorResponse(BaseModel):
    """Salida estructurada del agente Agentic RAG."""

    respuesta: str
    citas: list[Cita] = Field(default_factory=list)
    modo: ModoTutor
    agentic_trace: AgenticTrace = Field(default_factory=AgenticTrace)


class AgenticRAGState(TypedDict):
    """Estado compartido a lo largo de los nodos del grafo LangGraph."""

    mensaje: str
    modo: ModoTutor
    libro: str | None
    historial: list[dict[str, str]]
    enable_reflection: bool
    enable_requery: bool
    # Capa 1: Recuperación pasiva
    intento: int
    override_query: str | None
    query_efectiva: str
    estrategia: str
    is_generic_quick_read: bool
    is_roadmap: bool
    documentos_candidatos: list[Document]
    # Capa 2: Reflexión y filtrado del agente
    documentos_filtrados: list[Document]
    reflexion: dict[str, Any]
    rebusqueda_activada: bool
    # Capa 3: Generación final
    citas: list[dict[str, Any]]
    respuesta: str
    # Tiempos por capa (ms)
    retrieval_ms: int
    reflection_ms: int
    generation_ms: int


def nodo_recuperacion_pasiva(state: AgenticRAGState) -> dict[str, Any]:
    """Nodo 1: Recuperación pasiva de fragmentos candidatos desde ChromaDB."""
    t0 = time.perf_counter()
    bundle = retrieve_candidates(
        mensaje=state["mensaje"],
        modo=state["modo"],
        libro=state.get("libro"),
        historial=state.get("historial") or [],
        override_query=state.get("override_query"),
    )
    elapsed = int((time.perf_counter() - t0) * 1000)
    return {
        "documentos_candidatos": bundle.documentos,
        "query_efectiva": bundle.query_efectiva,
        "estrategia": bundle.estrategia,
        "is_generic_quick_read": bundle.is_generic_quick_read,
        "is_roadmap": bundle.is_roadmap,
        "retrieval_ms": int(state.get("retrieval_ms", 0)) + elapsed,
    }


def nodo_reflexion_agente(state: AgenticRAGState) -> dict[str, Any]:
    """Nodo 2: Capa de razonamiento/reflexión que filtra ruido y resuelve conflictos."""
    t0 = time.perf_counter()
    candidatos = state.get("documentos_candidatos") or []
    if not state.get("enable_reflection", True):
        bypass = ReflectionResult(
            ejecutada=False,
            indices_relevantes=list(range(1, len(candidatos) + 1)),
            chunks_iniciales=len(candidatos),
            chunks_retenidos=len(candidatos),
            chunks_descartados=0,
        )
        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "override_query": None,
            "reflexion": bypass.model_dump(),
            "documentos_filtrados": candidatos,
            "citas": build_citations(candidatos),
            "reflection_ms": int(state.get("reflection_ms", 0)) + elapsed,
        }

    intento = int(state.get("intento", 0))
    filtered_docs, reflection = reflect_and_filter(
        mensaje=state["mensaje"],
        modo=state["modo"],
        documentos=candidatos,
        estrategia=state.get("estrategia", "estandar"),
        intento=intento if state.get("enable_requery", True) else 1,
    )
    elapsed = int((time.perf_counter() - t0) * 1000)
    total_refl = int(state.get("reflection_ms", 0)) + elapsed

    if state.get("enable_requery", True) and reflection.requiere_rebusqueda and reflection.nueva_query_sugerida:
        return {
            "intento": intento + 1,
            "override_query": reflection.nueva_query_sugerida,
            "rebusqueda_activada": True,
            "reflexion": reflection.model_dump(),
            "documentos_filtrados": [],
            "citas": [],
            "reflection_ms": total_refl,
        }

    return {
        "override_query": None,
        "reflexion": reflection.model_dump(),
        "documentos_filtrados": filtered_docs,
        "citas": build_citations(filtered_docs),
        "reflection_ms": total_refl,
    }


def enrutar_tras_reflexion(
    state: AgenticRAGState,
) -> Literal["recuperacion_pasiva", "sin_contexto", "generar_lectura_rapida", "generar_consulta"]:
    """Decide si el agente re-busca con una query refinada o avanza al motor de generación."""
    reflexion = state.get("reflexion") or {}
    if reflexion.get("requiere_rebusqueda") and state.get("override_query"):
        return "recuperacion_pasiva"
    if not state.get("documentos_filtrados"):
        return "sin_contexto"
    if state["modo"] == "lectura_rapida":
        return "generar_lectura_rapida"
    return "generar_consulta"


def nodo_sin_contexto(state: AgenticRAGState) -> dict[str, Any]:
    """Nodo de respaldo cuando no hay fragmentos relevantes tras la reflexión."""
    return {
        "respuesta": build_no_context_message(state.get("libro")),
        "citas": [],
    }


def nodo_generar_lectura_rapida(state: AgenticRAGState) -> dict[str, Any]:
    """Nodo 3A: Motor de generación en modo Lectura rápida."""
    t0 = time.perf_counter()
    reflexion = state.get("reflexion") or {}
    respuesta = generate_quick_read(
        pedido=state["mensaje"],
        documentos=state["documentos_filtrados"],
        is_generic=bool(state.get("is_generic_quick_read")),
        notas_reflexion=str(reflexion.get("notas_sintesis") or ""),
    )
    return {
        "respuesta": respuesta,
        "generation_ms": int((time.perf_counter() - t0) * 1000),
    }


def nodo_generar_consulta(state: AgenticRAGState) -> dict[str, Any]:
    """Nodo 3B: Motor de generación en modo Consulta técnica."""
    t0 = time.perf_counter()
    reflexion = state.get("reflexion") or {}
    respuesta = generate_consultation(
        pedido=state["mensaje"],
        documentos=state["documentos_filtrados"],
        historial=state.get("historial") or [],
        is_roadmap=bool(state.get("is_roadmap")),
        conflictos_resueltos=str(reflexion.get("conflictos_resueltos") or ""),
        notas_reflexion=str(reflexion.get("notas_sintesis") or ""),
    )
    return {
        "respuesta": respuesta,
        "generation_ms": int((time.perf_counter() - t0) * 1000),
    }


def build_tutor_graph():
    """Construye y compila el grafo Agentic RAG de LangGraph."""
    workflow = StateGraph(AgenticRAGState)

    workflow.add_node("recuperacion_pasiva", nodo_recuperacion_pasiva)
    workflow.add_node("reflexion_agente", nodo_reflexion_agente)
    workflow.add_node("sin_contexto", nodo_sin_contexto)
    workflow.add_node("generar_lectura_rapida", nodo_generar_lectura_rapida)
    workflow.add_node("generar_consulta", nodo_generar_consulta)

    workflow.add_edge(START, "recuperacion_pasiva")
    workflow.add_edge("recuperacion_pasiva", "reflexion_agente")
    workflow.add_conditional_edges(
        "reflexion_agente",
        enrutar_tras_reflexion,
        {
            "recuperacion_pasiva": "recuperacion_pasiva",
            "sin_contexto": "sin_contexto",
            "generar_lectura_rapida": "generar_lectura_rapida",
            "generar_consulta": "generar_consulta",
        },
    )
    workflow.add_edge("sin_contexto", END)
    workflow.add_edge("generar_lectura_rapida", END)
    workflow.add_edge("generar_consulta", END)

    return workflow.compile()


@lru_cache(maxsize=1)
def get_tutor_graph():
    """Devuelve una única instancia compilada del grafo."""
    return build_tutor_graph()


def run_tutor(
    mensaje: str,
    modo: ModoTutor = "consulta",
    libro: str | None = None,
    historial: list[dict[str, str]] | None = None,
    enable_reflection: bool = True,
    enable_requery: bool = True,
) -> TutorResponse:
    """Ejecuta el grafo Agentic RAG completo y devuelve respuesta, citas y telemetría."""
    t0 = time.perf_counter()
    graph = get_tutor_graph()
    initial_state: AgenticRAGState = {
        "mensaje": mensaje,
        "modo": modo,
        "libro": libro,
        "historial": historial or [],
        "enable_reflection": enable_reflection,
        "enable_requery": enable_requery,
        "intento": 0,
        "override_query": None,
        "query_efectiva": mensaje,
        "estrategia": "estandar",
        "is_generic_quick_read": False,
        "is_roadmap": False,
        "documentos_candidatos": [],
        "documentos_filtrados": [],
        "reflexion": ReflectionResult().model_dump(),
        "rebusqueda_activada": False,
        "citas": [],
        "respuesta": "",
        "retrieval_ms": 0,
        "reflection_ms": 0,
        "generation_ms": 0,
    }
    final_state = graph.invoke(initial_state)
    latencia_ms = int((time.perf_counter() - t0) * 1000)

    reflexion = final_state.get("reflexion") or {}
    trace = AgenticTrace(
        estrategia=str(final_state.get("estrategia") or "estandar"),
        reflexion_ejecutada=bool(reflexion.get("ejecutada", False)),
        chunks_iniciales=int(reflexion.get("chunks_iniciales", 0)),
        chunks_retenidos=int(reflexion.get("chunks_retenidos", 0)),
        chunks_descartados=int(reflexion.get("chunks_descartados", 0)),
        rebusqueda_activada=bool(final_state.get("rebusqueda_activada", False)),
        conflictos_resueltos=str(reflexion.get("conflictos_resueltos") or ""),
        latencia_ms=latencia_ms,
        retrieval_ms=int(final_state.get("retrieval_ms", 0)),
        reflection_ms=int(final_state.get("reflection_ms", 0)),
        generation_ms=int(final_state.get("generation_ms", 0)),
    )

    citas = [Cita(**c) for c in final_state.get("citas", [])]
    return TutorResponse(
        respuesta=final_state.get("respuesta", ""),
        citas=citas,
        modo=modo,
        agentic_trace=trace,
    )
