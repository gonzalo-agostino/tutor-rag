"""Capa 4 — Observabilidad y evaluación en tiempo de ejecución (DeepEval + Juez independiente).

Monitorea automáticamente fidelidad (`FaithfulnessMetric`) y relevancia del contexto
(`ContextualRelevancyMetric`) en segundo plano usando un modelo juez más capaz
(`gemini-3.6-flash` por defecto) para evitar sesgo de auto-evaluación.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from config import get_llm
from generation import extract_text_from_content

try:
    from deepeval.metrics import ContextualRelevancyMetric, FaithfulnessMetric
    from deepeval.models import DeepEvalBaseLLM
    from deepeval.test_case import LLMTestCase

    DEEPEVAL_AVAILABLE = True
except Exception:
    DEEPEVAL_AVAILABLE = False
    DeepEvalBaseLLM = object  # type: ignore[misc,assignment]

EVAL_JUDGE_MODELS = ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite")


class EvaluationScores(BaseModel):
    """Métricas de calidad RAG calculadas para una ejecución."""

    faithfulness: float | None = None
    context_precision: float | None = None
    faithfulness_reason: str = ""
    context_precision_reason: str = ""
    engine: str = "deepeval"


def _invoke_judge(messages: list, timeout: float = 35.0) -> tuple[str, str]:
    """Invoca el modelo juez independiente con fallback automático ante 503/504."""
    last_exc: Exception | None = None
    for model_name in EVAL_JUDGE_MODELS:
        try:
            llm = get_llm(
                temperature=0.0,
                model_name=model_name,
                timeout=timeout,
                max_retries=1,
            )
            ai_msg = llm.invoke(messages)
            return extract_text_from_content(ai_msg.content), model_name
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
    raise RuntimeError("No hay modelos Gemini disponibles para evaluación.")


class GeminiDeepEvalLLM(DeepEvalBaseLLM):
    """Adaptador de Gemini (juez independiente) para ejecutar métricas oficiales de DeepEval."""

    def __init__(self, model_name: str = EVAL_JUDGE_MODELS[0]) -> None:
        self._model_name = model_name

    def load_model(self) -> str:
        return self._model_name

    def generate(self, prompt: str, schema: Any = None) -> Any:
        raw, used_model = _invoke_judge([HumanMessage(content=prompt)], timeout=35.0)
        self._model_name = used_model
        if schema is not None:
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            try:
                return schema.model_validate_json(cleaned)
            except Exception:
                match = re.search(r"\{[\s\S]*\}", cleaned)
                if match:
                    return schema.model_validate_json(match.group(0))
        return raw

    async def a_generate(self, prompt: str, schema: Any = None) -> Any:
        return self.generate(prompt, schema=schema)

    def get_model_name(self) -> str:
        return self._model_name


def _fallback_llm_judge(query: str, response: str, contexts: list[str]) -> EvaluationScores:
    """Evaluador estructurado de respaldo si DeepEval falla al parsear una respuesta."""
    ctx_preview = "\n---\n".join(c[:450] for c in contexts[:6])
    prompt = (
        "Evaluá esta respuesta RAG de forma estricta en dos métricas de 0.0 a 1.0 y devolvé SOLO JSON:\n"
        '{"faithfulness": 0.0, "context_relevancy": 0.0, "faithfulness_reason": "...", "context_relevancy_reason": "..."}\n'
        "- faithfulness: proporción de afirmaciones de la respuesta respaldadas estrictamente por el contexto.\n"
        "- context_relevancy: proporción de oraciones/hechos del contexto recuperado que son directamente útiles para responder la pregunta.\n\n"
        f"PREGUNTA:\n{query[:500]}\n\n"
        f"CONTEXTO RECUPERADO:\n{ctx_preview}\n\n"
        f"RESPUESTA:\n{response[:1600]}"
    )
    try:
        raw, used_model = _invoke_judge(
            [
                SystemMessage(content="Respondé únicamente con JSON válido."),
                HumanMessage(content=prompt),
            ],
            timeout=25.0,
        )
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        match = re.search(r"\{[\s\S]*\}", cleaned)
        data = json.loads(match.group(0) if match else cleaned)
        f_score = max(0.0, min(1.0, float(data.get("faithfulness", 0.0))))
        r_score = max(0.0, min(1.0, float(data.get("context_relevancy", 0.0))))
        return EvaluationScores(
            faithfulness=round(f_score, 3),
            context_precision=round(r_score, 3),
            faithfulness_reason=str(data.get("faithfulness_reason", ""))[:300],
            context_precision_reason=str(data.get("context_relevancy_reason", ""))[:300],
            engine=f"judge_fallback:{used_model}",
        )
    except Exception:
        return EvaluationScores(engine="unavailable")


def evaluate_rag_turn(
    query: str,
    response: str,
    contexts: list[str],
) -> EvaluationScores:
    """Calcula faithfulness y contextual_relevancy usando DeepEval con juez independiente."""
    clean_contexts = [c.strip()[:1200] for c in contexts if c and c.strip()][:6]
    if not clean_contexts or not response.strip():
        return EvaluationScores(engine="skipped_no_context")

    if not DEEPEVAL_AVAILABLE:
        return _fallback_llm_judge(query, response, clean_contexts)

    try:
        judge = GeminiDeepEvalLLM()
        test_case = LLMTestCase(
            input=query[:800],
            actual_output=response[:2200],
            retrieval_context=clean_contexts,
        )

        faith_metric = FaithfulnessMetric(
            threshold=0.7,
            model=judge,
            include_reason=True,
            async_mode=False,
        )
        relevancy_metric = ContextualRelevancyMetric(
            threshold=0.7,
            model=judge,
            include_reason=True,
            async_mode=False,
        )

        faith_metric.measure(test_case)
        relevancy_metric.measure(test_case)

        f_val = round(float(faith_metric.score), 3) if faith_metric.score is not None else None
        r_val = round(float(relevancy_metric.score), 3) if relevancy_metric.score is not None else None

        return EvaluationScores(
            faithfulness=f_val,
            context_precision=r_val,
            faithfulness_reason=str(faith_metric.reason or "")[:350],
            context_precision_reason=str(relevancy_metric.reason or "")[:350],
            engine=f"deepeval:{judge.get_model_name()}",
        )
    except Exception:
        return _fallback_llm_judge(query, response, clean_contexts)
