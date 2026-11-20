"""Persistencia permanente de chats en SQLite (chats.db) para sincronización multi-dispositivo."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from config import BASE_DIR

CHATS_DB_PATH: Path = BASE_DIR / "chats.db"


@contextmanager
def _get_conn() -> Iterator[sqlite3.Connection]:
    """Abre una conexión SQLite con claves foráneas y modo WAL para concurrencia segura."""
    conn = sqlite3.connect(str(CHATS_DB_PATH), timeout=15.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Crea las tablas de sesiones y mensajes si todavía no existen."""
    with _get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                book TEXT NOT NULL DEFAULT '',
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                modo TEXT NOT NULL DEFAULT 'consulta',
                scope_book TEXT,
                citas_json TEXT NOT NULL DEFAULT '[]',
                is_error INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session_id
                ON messages(session_id, id);

            CREATE INDEX IF NOT EXISTS idx_sessions_updated_at
                ON sessions(updated_at DESC);

            CREATE TABLE IF NOT EXISTS rag_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                query TEXT NOT NULL,
                modo TEXT NOT NULL,
                estrategia TEXT NOT NULL,
                chunks_iniciales INTEGER NOT NULL DEFAULT 0,
                chunks_retenidos INTEGER NOT NULL DEFAULT 0,
                chunks_descartados INTEGER NOT NULL DEFAULT 0,
                rebusqueda_activada INTEGER NOT NULL DEFAULT 0,
                conflictos_resueltos TEXT NOT NULL DEFAULT '',
                latencia_ms INTEGER NOT NULL DEFAULT 0,
                retrieval_ms INTEGER NOT NULL DEFAULT 0,
                reflection_ms INTEGER NOT NULL DEFAULT 0,
                generation_ms INTEGER NOT NULL DEFAULT 0,
                faithfulness REAL,
                context_precision REAL,
                faithfulness_reason TEXT NOT NULL DEFAULT '',
                context_precision_reason TEXT NOT NULL DEFAULT '',
                eval_engine TEXT NOT NULL DEFAULT 'pending',
                created_at INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_rag_metrics_created_at
                ON rag_metrics(created_at DESC);

            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        # Migración segura para bases existentes sin columnas de desglose de latencia
        existing_cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(rag_metrics)").fetchall()
        }
        for col in ("retrieval_ms", "reflection_ms", "generation_ms"):
            if col not in existing_cols:
                conn.execute(
                    f"ALTER TABLE rag_metrics ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0"
                )


def _now_ms() -> int:
    return int(time.time() * 1000)


def _row_to_message(row: sqlite3.Row) -> dict[str, Any]:
    try:
        citas = json.loads(row["citas_json"] or "[]")
    except json.JSONDecodeError:
        citas = []
    return {
        "id": row["id"],
        "role": row["role"],
        "content": row["content"],
        "modo": row["modo"],
        "scopeBook": row["scope_book"],
        "citas": citas,
        "isError": bool(row["is_error"]),
        "createdAt": row["created_at"],
    }


def list_sessions() -> list[dict[str, Any]]:
    """Devuelve todas las sesiones guardadas ordenadas por última actividad."""
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                s.id,
                s.title,
                s.book,
                s.created_at,
                s.updated_at,
                COUNT(m.id) AS message_count
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.id
            GROUP BY s.id
            ORDER BY s.updated_at DESC
            """
        ).fetchall()
    return [
        {
            "id": r["id"],
            "title": r["title"],
            "book": r["book"],
            "createdAt": r["created_at"],
            "updatedAt": r["updated_at"],
            "messageCount": r["message_count"],
        }
        for r in rows
    ]


def get_session(session_id: str) -> dict[str, Any] | None:
    """Devuelve una sesión junto con todos sus mensajes y citas."""
    with _get_conn() as conn:
        srow = conn.execute(
            "SELECT id, title, book, created_at, updated_at FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if srow is None:
            return None
        mrows = conn.execute(
            """
            SELECT id, role, content, modo, scope_book, citas_json, is_error, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()
    return {
        "id": srow["id"],
        "title": srow["title"],
        "book": srow["book"],
        "createdAt": srow["created_at"],
        "updatedAt": srow["updated_at"],
        "messages": [_row_to_message(r) for r in mrows],
    }


def create_session(
    title: str = "Nueva sesión",
    book: str = "",
    messages: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Crea una nueva sesión en SQLite (opcionalmente con mensajes iniciales)."""
    sid = (session_id or "").strip() or f"chat_{uuid.uuid4().hex[:12]}"
    clean_title = (title or "Nueva sesión").strip() or "Nueva sesión"
    clean_book = (book or "").strip()
    now = _now_ms()

    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO sessions (id, title, book, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (sid, clean_title, clean_book, now, now),
        )
        for msg in messages or []:
            role = str(msg.get("role", "user"))
            content = str(msg.get("content", ""))
            if not content:
                continue
            modo = str(msg.get("modo", "consulta"))
            scope_book = msg.get("scopeBook") or msg.get("scope_book")
            citas_json = json.dumps(msg.get("citas") or [], ensure_ascii=False)
            is_error = 1 if msg.get("isError") or msg.get("is_error") else 0
            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, modo, scope_book, citas_json, is_error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (sid, role, content, modo, scope_book, citas_json, is_error, _now_ms()),
            )

    return get_session(sid) or {
        "id": sid,
        "title": clean_title,
        "book": clean_book,
        "createdAt": now,
        "updatedAt": now,
        "messages": [],
    }


def update_session(
    session_id: str,
    title: str | None = None,
    book: str | None = None,
) -> dict[str, Any] | None:
    """Actualiza el título y/o libro filtrado de una sesión existente."""
    with _get_conn() as conn:
        existing = conn.execute(
            "SELECT id, title, book FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if existing is None:
            return None

        new_title = title.strip() if isinstance(title, str) and title.strip() else existing["title"]
        new_book = book.strip() if isinstance(book, str) else existing["book"]
        now = _now_ms()
        conn.execute(
            "UPDATE sessions SET title = ?, book = ?, updated_at = ? WHERE id = ?",
            (new_title, new_book, now, session_id),
        )
    return get_session(session_id)


def append_message(
    session_id: str,
    role: str,
    content: str,
    modo: str = "consulta",
    scope_book: str | None = None,
    citas: list[dict[str, Any]] | None = None,
    is_error: bool = False,
) -> None:
    """Agrega un mensaje a una sesión y actualiza automáticamente su título si estaba vacía."""
    now = _now_ms()
    citas_json = json.dumps(citas or [], ensure_ascii=False)

    with _get_conn() as conn:
        srow = conn.execute(
            "SELECT id, title FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if srow is None:
            auto_title = (content[:41] + "…") if len(content) > 44 else (content or "Nueva sesión")
            conn.execute(
                "INSERT INTO sessions (id, title, book, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, auto_title, scope_book or "", now, now),
            )
        else:
            count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
            if count == 0 and role == "user" and srow["title"] in {"Nueva sesión", ""}:
                auto_title = (content[:41] + "…") if len(content) > 44 else content
                conn.execute(
                    "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                    (auto_title, now, session_id),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?",
                    (now, session_id),
                )

        conn.execute(
            """
            INSERT INTO messages (session_id, role, content, modo, scope_book, citas_json, is_error, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, role, content, modo, scope_book, citas_json, 1 if is_error else 0, now),
        )


def delete_session(session_id: str) -> bool:
    """Elimina una sesión y todos sus mensajes asociados."""
    with _get_conn() as conn:
        cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cur.rowcount > 0


def save_metric_trace(
    *,
    session_id: str | None,
    query: str,
    modo: str,
    estrategia: str,
    chunks_iniciales: int,
    chunks_retenidos: int,
    chunks_descartados: int,
    rebusqueda_activada: bool,
    conflictos_resueltos: str,
    latencia_ms: int,
    retrieval_ms: int = 0,
    reflection_ms: int = 0,
    generation_ms: int = 0,
) -> int:
    """Registra la telemetría de una ejecución Agentic RAG y devuelve su ID."""
    now = _now_ms()
    with _get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO rag_metrics (
                session_id, query, modo, estrategia,
                chunks_iniciales, chunks_retenidos, chunks_descartados,
                rebusqueda_activada, conflictos_resueltos, latencia_ms,
                retrieval_ms, reflection_ms, generation_ms,
                eval_engine, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                session_id,
                query[:500],
                modo,
                estrategia,
                chunks_iniciales,
                chunks_retenidos,
                chunks_descartados,
                1 if rebusqueda_activada else 0,
                conflictos_resueltos[:500],
                latencia_ms,
                retrieval_ms,
                reflection_ms,
                generation_ms,
                now,
            ),
        )
        return int(cur.lastrowid)


def update_metric_scores(
    metric_id: int,
    *,
    faithfulness: float | None,
    context_precision: float | None,
    faithfulness_reason: str,
    context_precision_reason: str,
    eval_engine: str,
) -> None:
    """Actualiza las métricas de DeepEval (faithfulness y context_precision) de una traza."""
    with _get_conn() as conn:
        conn.execute(
            """
            UPDATE rag_metrics
            SET faithfulness = ?,
                context_precision = ?,
                faithfulness_reason = ?,
                context_precision_reason = ?,
                eval_engine = ?
            WHERE id = ?
            """,
            (
                faithfulness,
                context_precision,
                faithfulness_reason[:400],
                context_precision_reason[:400],
                eval_engine,
                metric_id,
            ),
        )


def get_metrics_summary(limit: int = 20) -> dict[str, Any]:
    """Devuelve promedios globales de observabilidad y las últimas trazas evaluadas."""
    with _get_conn() as conn:
        agg = conn.execute(
            """
            SELECT
                COUNT(*) AS total_runs,
                AVG(faithfulness) AS avg_faithfulness,
                AVG(context_precision) AS avg_context_precision,
                AVG(latencia_ms) AS avg_latency_ms,
                AVG(retrieval_ms) AS avg_retrieval_ms,
                AVG(reflection_ms) AS avg_reflection_ms,
                AVG(generation_ms) AS avg_generation_ms,
                SUM(chunks_descartados) AS total_discarded,
                SUM(rebusqueda_activada) AS total_requeries
            FROM rag_metrics
            """
        ).fetchone()

        rows = conn.execute(
            """
            SELECT
                id, session_id, query, modo, estrategia,
                chunks_iniciales, chunks_retenidos, chunks_descartados,
                rebusqueda_activada, conflictos_resueltos, latencia_ms,
                retrieval_ms, reflection_ms, generation_ms,
                faithfulness, context_precision,
                faithfulness_reason, context_precision_reason,
                eval_engine, created_at
            FROM rag_metrics
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return {
        "total_runs": int(agg["total_runs"] or 0),
        "avg_faithfulness": round(float(agg["avg_faithfulness"]), 3) if agg["avg_faithfulness"] is not None else None,
        "avg_context_precision": round(float(agg["avg_context_precision"]), 3) if agg["avg_context_precision"] is not None else None,
        "avg_latency_ms": int(agg["avg_latency_ms"] or 0),
        "avg_retrieval_ms": int(agg["avg_retrieval_ms"] or 0),
        "avg_reflection_ms": int(agg["avg_reflection_ms"] or 0),
        "avg_generation_ms": int(agg["avg_generation_ms"] or 0),
        "total_discarded_chunks": int(agg["total_discarded"] or 0),
        "total_requeries": int(agg["total_requeries"] or 0),
        "recent_traces": [
            {
                "id": r["id"],
                "session_id": r["session_id"],
                "query": r["query"],
                "modo": r["modo"],
                "estrategia": r["estrategia"],
                "chunks_iniciales": r["chunks_iniciales"],
                "chunks_retenidos": r["chunks_retenidos"],
                "chunks_descartados": r["chunks_descartados"],
                "rebusqueda_activada": bool(r["rebusqueda_activada"]),
                "conflictos_resueltos": r["conflictos_resueltos"],
                "latencia_ms": r["latencia_ms"],
                "retrieval_ms": r["retrieval_ms"],
                "reflection_ms": r["reflection_ms"],
                "generation_ms": r["generation_ms"],
                "faithfulness": r["faithfulness"],
                "context_precision": r["context_precision"],
                "faithfulness_reason": r["faithfulness_reason"],
                "context_precision_reason": r["context_precision_reason"],
                "eval_engine": r["eval_engine"],
                "created_at": r["created_at"],
            }
            for r in rows
        ],
    }


def clear_metrics() -> int:
    """Elimina todas las trazas históricas de rag_metrics y devuelve cuántas se borraron."""
    with _get_conn() as conn:
        cur = conn.execute("DELETE FROM rag_metrics")
        return int(cur.rowcount)


DEFAULT_RUNTIME_SETTINGS: dict[str, Any] = {
    "enable_reflection": True,
    "enable_requery": True,
    "enable_evaluation": True,
    "llm_model": "",
}


def get_runtime_settings() -> dict[str, Any]:
    """Lee la configuración de runtime persistida en SQLite (combinada con defaults)."""
    settings = dict(DEFAULT_RUNTIME_SETTINGS)
    with _get_conn() as conn:
        rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
    for row in rows:
        key = row["key"]
        if key not in settings:
            continue
        try:
            settings[key] = json.loads(row["value"])
        except json.JSONDecodeError:
            settings[key] = row["value"]
    return settings


def save_runtime_settings(updates: dict[str, Any]) -> dict[str, Any]:
    """Guarda uno o más ajustes de runtime en SQLite y devuelve la configuración completa."""
    with _get_conn() as conn:
        for key, value in updates.items():
            if key not in DEFAULT_RUNTIME_SETTINGS or value is None:
                continue
            conn.execute(
                """
                INSERT INTO app_settings (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, json.dumps(value, ensure_ascii=False)),
            )
    return get_runtime_settings()
