"""Embeddings locales livianos con ONNX Runtime (sin PyTorch) para Tutor RAG."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from langchain_core.embeddings import Embeddings
from tokenizers import Tokenizer

DEFAULT_LOCAL_REPO = "Xenova/multilingual-e5-small"
DEFAULT_ONNX_FILE = "onnx/model_int8.onnx"
DEFAULT_TOKENIZER_FILE = "tokenizer.json"
DEFAULT_MAX_LENGTH = 512
DEFAULT_INFERENCE_BATCH = 16


class LocalOnnxE5Embeddings(Embeddings):
    """Embeddings multilingües (español + inglés) locales usando ONNX Runtime int8.

    Compatible con la interfaz `langchain_core.embeddings.Embeddings` y optimizado
    para ejecutarse en CPU con bajo consumo de RAM (~300 MB).
    """

    def __init__(
        self,
        repo_id: str = DEFAULT_LOCAL_REPO,
        onnx_filename: str = DEFAULT_ONNX_FILE,
        tokenizer_filename: str = DEFAULT_TOKENIZER_FILE,
        max_length: int = DEFAULT_MAX_LENGTH,
        inference_batch_size: int = DEFAULT_INFERENCE_BATCH,
        num_threads: int | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.onnx_filename = onnx_filename
        self.tokenizer_filename = tokenizer_filename
        self.max_length = max_length
        self.inference_batch_size = max(1, inference_batch_size)

        model_path = self._resolve_hub_file(repo_id, onnx_filename)
        tokenizer_path = self._resolve_hub_file(repo_id, tokenizer_filename)

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        pad_id = self.tokenizer.token_to_id("<pad>")
        if pad_id is None:
            pad_id = self.tokenizer.token_to_id("[PAD]")
        if pad_id is None:
            pad_id = 0
        pad_token = self.tokenizer.id_to_token(pad_id) or "<pad>"

        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        cpu_threads = num_threads or max(1, min(4, os.cpu_count() or 4))
        sess_options.intra_op_num_threads = cpu_threads
        sess_options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {inp.name for inp in self.session.get_inputs()}

    @staticmethod
    def _resolve_hub_file(repo_id: str, filename: str) -> str:
        """Usa el archivo ya cacheado localmente sin consultar la red; solo descarga si falta."""
        try:
            return hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
        except Exception:
            return hf_hub_download(repo_id=repo_id, filename=filename)

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        encodings = self.tokenizer.encode_batch(texts)
        input_ids = np.array([enc.ids for enc in encodings], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask for enc in encodings], dtype=np.int64)

        ort_inputs: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in self._input_names:
            token_type_ids = np.array([enc.type_ids for enc in encodings], dtype=np.int64)
            ort_inputs["token_type_ids"] = token_type_ids

        outputs = self.session.run(None, ort_inputs)
        last_hidden = outputs[0].astype(np.float32)  # [batch, seq_len, hidden]

        # Mean pooling ponderado por attention_mask (estándar oficial de E5)
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_embeddings = np.sum(last_hidden * mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
        pooled = sum_embeddings / sum_mask

        # Normalización L2 para búsqueda por similitud coseno
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        normalized = pooled / np.clip(norms, a_min=1e-12, a_max=None)
        return normalized.tolist()

    def _embed_prefixed(self, texts: Iterable[str], prefix: str) -> list[list[float]]:
        prepared = [f"{prefix}{t.strip()}" for t in texts]
        if not prepared:
            return []
        results: list[list[float]] = []
        step = self.inference_batch_size
        for i in range(0, len(prepared), step):
            results.extend(self._encode_batch(prepared[i : i + step]))
        return results

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Calcula embeddings para fragmentos de libros usando el prefijo 'passage: '."""
        return self._embed_prefixed(texts, prefix="passage: ")

    def embed_query(self, text: str) -> list[float]:
        """Calcula el embedding de una consulta del usuario usando el prefijo 'query: '."""
        return self._embed_prefixed([text], prefix="query: ")[0]


@lru_cache(maxsize=1)
def get_local_embeddings(
    repo_id: str = DEFAULT_LOCAL_REPO,
) -> LocalOnnxE5Embeddings:
    """Devuelve una instancia cacheada del modelo ONNX local para reutilizarla en memoria."""
    return LocalOnnxE5Embeddings(repo_id=repo_id)
