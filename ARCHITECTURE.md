# Arquitectura y Funcionamiento de Tutor RAG

Esta guía explica cómo están conectadas todas las piezas de **Tutor RAG** y qué ocurre exactamente cada vez que hacés una pregunta, agregás un PDF nuevo en Google Drive o abrís el Dashboard de Observabilidad.

---

## 1. Mapa general del sistema

El proyecto combina procesamiento **100% local** para tus libros, embeddings y chats (sin costo de API ni límites diarios de embeddings) con **Gemini** solo para razonamiento, redacción y evaluación de calidad.

```mermaid
flowchart TD
    UI["Frontend Dark UI (PC / iPad)\ntemplates/index.html · static/app.js"]

    subgraph Backend["Backend FastAPI (app.py)"]
        API["REST API\n/api/chat · /books · /page-image\n/api/sessions · /api/metrics · /api/config"]
        SYNC["Sync & Watcher 30s\n/api/books/sync · lifespan"]
        BG["BackgroundTasks\nEvaluación DeepEval"]
    end

    subgraph Agentic["Pipeline Agentic RAG (LangGraph)"]
        L0["Router de Intención (0 ms)\nagent.py"]
        L1["Capa 1: Recuperación + Limpieza\nretrieval.py · config.py"]
        L2["Capa 2: Reflexión y Re-búsqueda\nreflection.py"]
        L3["Capa 3: Generación y Citas\ngeneration.py"]
        L4["Capa 4: Observabilidad DeepEval\nevaluation.py"]
    end

    subgraph Local["Almacenamiento y Cómputo Local"]
        DRIVE["Google Drive Desktop\nBOOKS_DIR (libros/*.pdf)"]
        ONNX["Embeddings Locales ONNX\nembeddings.py (e5-small int8)"]
        CHROMA["ChromaDB Vectorial\nchroma_db/"]
        SQLITE["SQLite Persistente\nchats.db (sesiones, métricas, config)"]
    end

    GEMINI["Gemini API\nGenerador: 3.5-flash-lite\nJuez DeepEval: 3.6-flash"]

    UI <--> API
    API --> L0
    L0 --> L1 --> L2 --> L3
    L1 <--> CHROMA
    L1 --> ONNX
    L2 <--> GEMINI
    L3 <--> GEMINI
    API --> SQLITE
    API --> BG --> L4 <--> GEMINI
    L4 --> SQLITE
    DRIVE --> SYNC --> ONNX --> CHROMA
    DRIVE --> API
```

---

## 2. Cómo piensa el agente: Grafo LangGraph paso a paso

Cada mensaje que enviás al chat entra a `run_tutor()` en [agent.py](agent.py). En vez de hacer un RAG rígido de un solo paso, el grafo decide dinámicamente qué camino tomar:

```mermaid
flowchart TD
    START(["Mensaje del usuario"]) --> INTENT{"Clasificador de Intención\n(0 ms, heurístico)"}

    INTENT -- "Saludo / confirmación\n(ej. 'Dale, gracias')" --> CONV["nodo_conversacional\n(0 ms retrieval, 0 ms reflection)"]
    CONV --> END_OK(["Respuesta rápida (~2s)\nEval: skipped_conversational"])

    INTENT -- "Pregunta técnica o\nLectura rápida" --> RET["Capa 1: recuperacion_pasiva\nChromaDB + clean_pdf_text()"]
    RET --> REFL{"Capa 2: reflexion_agente\nAnaliza relevancia, edición y conflictos"}

    REFL -- "Evidencia débil y\nintento == 0" --> REQUERY["Re-búsqueda adaptativa\nTraduce negocio a términos técnicos EN"]
    REQUERY --> RET

    REFL -- "0 fragmentos útiles\n(sin rellenar con ruido)" --> NOCONTEXT["nodo_sin_contexto\n(0 tokens de generación)\nEval: skipped_no_context"]
    NOCONTEXT --> END_OK2(["Respuesta honesta con diagnóstico"])

    REFL -- "Fragmentos verificados\nmodo == lectura_rapida" --> GEN_QR["Capa 3A: generar_lectura_rapida\nResumen + Código + Autoevaluación"]
    REFL -- "Fragmentos verificados\nmodo == consulta" --> GEN_Q["Capa 3B: generar_consulta\nRespuesta + Mermaid + Citas exactas"]

    GEN_QR --> END_RAG(["Respuesta + Citas verificables + Telemetría"])
    GEN_Q --> END_RAG
```

> [!TIP]
> **Por qué esto es mejor que un RAG clásico:**
> 1. **Ahorra ~8–10s** en mensajes conversacionales porque ni siquiera toca ChromaDB.
> 2. **No alucina cuando un libro no tiene el tema:** si la reflexión descarta todo (`0` chunks), corta directo a `sin_contexto` sin gastar tokens de generación.
> 3. **Limpia ruido editorial** (`Weak Isolation Levels | 233`, índices con puntos, números de página sueltos) automáticamente en tiempo real sin tener que re-indexar `chroma_db/`.

---

## 3. Secuencia de una consulta + Evaluación en segundo plano

Para que el chat se sienta rápido, la respuesta te llega apenas termina la generación, mientras **DeepEval** audita fidelidad y relevancia en segundo plano con un modelo juez independiente:

```mermaid
sequenceDiagram
    actor U as Usuario (PC / iPad)
    participant FE as Frontend (app.js)
    participant API as FastAPI (app.py)
    participant G as LangGraph (agent.py)
    participant DB as SQLite (chats.db)
    participant EV as DeepEval (evaluation.py)

    U->>FE: Envía pregunta (permite cambiar de chat mientras procesa)
    FE->>API: POST /api/chat (mensaje, libro, historial, session_id)
    API->>DB: Guarda mensaje del usuario
    API->>G: run_tutor(...)
    G-->>API: Respuesta + Citas 1:1 + Tiempos por capa (ms)
    API->>DB: Guarda respuesta + citas + traza inicial (pending)
    API-->>FE: 200 OK (muestra respuesta, Mermaid y botón Ver página original)
    Note over API,EV: En paralelo (BackgroundTasks), sin bloquear al usuario
    API->>EV: evaluate_rag_turn(query, response, contexts)
    EV-->>DB: Actualiza faithfulness y context_precision + motivos
    FE->>API: Polling /api/metrics (Dashboard en vivo + Exportar JSON)
```

---

## 4. Auto-indexación incremental desde Google Drive Desktop

Cuando copiás un PDF nuevo en tu carpeta sincronizada (`BOOKS_DIR`), tenés dos caminos automáticos y seguros contra concurrencia:

```mermaid
flowchart LR
    GDRIVE["Google Drive Desktop\nCopias un PDF nuevo"] --> DIR["BOOKS_DIR (libros/)"]

    subgraph Auto["Camino 1: Watcher automático (cada 30s)"]
        W1["os.scandir(BOOKS_DIR)"] --> W2{"¿Tamaño y mtime\nestables en 2 ciclos?"}
        W2 -- "No (Drive descargando)" --> WAIT["Esperar 30s"]
        W2 -- "Sí y no indexado" --> LOCK
    end

    subgraph Manual["Camino 2: Botón 'Sincronizar libros'"]
        BTN["POST /api/books/sync"] --> DIFF["get_new_books()\nCompara contra ChromaDB"]
        DIFF -- "Sin cambios" --> UP["Devuelve 'up_to_date'"]
        DIFF -- "Hay nuevos" --> LOCK
    end

    LOCK{"Lock anti-concurrencia\n(_ingest_lock + thread lock)"} --> SYNC["sync_new_books()\nreset=False · solo PDFs nuevos"]
    SYNC --> CLEAN["pypdf + clean_pdf_text()\nChunks de 1200 chars"]
    CLEAN --> ONNX2["ONNX e5-small local\n(~14 chunks/seg)"]
    ONNX2 --> DB2["ChromaDB actualizado\nGET /books recarga el sidebar"]
```

---

## 5. Mapa rápido de archivos del proyecto

| Archivo | Rol en el sistema |
|---|---|
| [config.py](config.py) | Rutas (`BOOKS_DIR`, `CHROMA_DIR`), carga de `.env`, títulos legibles de libros, lista de modelos Gemini y `clean_pdf_text()`. |
| [embeddings.py](embeddings.py) | Embeddings locales ONNX (`Xenova/multilingual-e5-small` int8, 384 dims) sin PyTorch y sin cuota de API. |
| [ingest.py](ingest.py) | Ingesta incremental/reanudable por IDs deterministas, `get_new_books()` y `sync_new_books()` con lock de concurrencia. |
| [retrieval.py](retrieval.py) | Capa 1: búsqueda bilingüe equilibrada, filtro por libro, comparación entre autores, resumen de libro entero, muestreo O(1) para Lectura rápida y contextualización de repreguntas. |
| [reflection.py](reflection.py) | Capa 2: crítico previo a la generación; descarta ruido, resuelve conflictos entre ediciones, permite devolver `0` chunks y traduce consultas de negocio en re-búsquedas técnicas. |
| [generation.py](generation.py) | Capa 3: prompts editoriales (Consulta, Lectura rápida, Conversacional), guía de Mermaid, citas 1:1 y fallback automático de modelos Gemini. |
| [evaluation.py](evaluation.py) | Capa 4: evaluación online con DeepEval (`FaithfulnessMetric` + `ContextualRelevancyMetric`) usando juez independiente y fallback JSON estructurado. |
| [agent.py](agent.py) | Orquestador LangGraph que une router de intención + las 3 capas y mide latencia por etapa (`retrieval_ms`, `reflection_ms`, `generation_ms`). |
| [db.py](db.py) | Persistencia permanente en SQLite (`chats.db`, modo WAL) para sesiones multi-dispositivo, mensajes, citas, métricas RAG y configuración en caliente. |
| [app.py](app.py) | Servidor FastAPI: endpoints de chat, sesiones, visor de páginas PDF (`/page-image` con PyMuPDF), observabilidad, configuración, `/api/books/sync` y watcher en `lifespan`. |
| [templates/index.html](templates/index.html) / [static/app.js](static/app.js) / [static/styles.css](static/styles.css) | Interfaz dark responsive (PC + iPad): chats concurrentes, hilo paralelo (*Deep Dive*), menú contextual/barra flotante, Mermaid.js, visor PDF, botón de sincronización y Dashboard de Observabilidad con exportación JSON. |
| [tests/](tests/) | 12 pruebas unitarias automatizadas (motor vectorial, router de intención, limpieza PDF, reflexión `0` chunks y sync incremental). |
