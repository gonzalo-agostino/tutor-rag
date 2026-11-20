/* Lógica del cliente para el espacio de estudio técnico Tutor RAG
 * Incluye: multi-chats (localStorage), hilo paralelo (Deep Dive), menú contextual (clic derecho),
 * barra flotante de selección (iPad/móvil), Mermaid.js y visor de página original del PDF.
 */

(() => {
  const LEGACY_STORAGE_KEY = "tutor_rag_sessions_v1";
  const LAST_SESSION_KEY = "tutor_rag_last_session_id";

  const state = {
    books: [],
    bookTitles: {},
    selectedBook: "",
    filterQuery: "",
    ingesting: false,
    sessions: [],
    activeSessionId: null,
    activeSession: null,
    pendingSessions: new Map(),
    unreadSessions: new Set(),
    sidebarOpenMobile: false,
    sidebarCollapsedDesktop: false,
    mermaidReady: false,
    mermaidCounter: 0,
    currentSelectionText: "",
    pdfViewer: {
      open: false,
      book: "",
      page: 1,
      totalPages: null,
      zoomed: false,
      objectUrl: null,
    },
    sideThread: {
      open: false,
      title: "",
      book: null,
      loading: false,
      messages: [],
    },
  };

  const els = {
    appShell: document.getElementById("appShell"),
    sidebar: document.getElementById("sidebar"),
    sidebarBackdrop: document.getElementById("sidebarBackdrop"),
    toggleSidebarBtn: document.getElementById("toggleSidebarBtn"),
    closeSidebarBtn: document.getElementById("closeSidebarBtn"),
    newSessionBtn: document.getElementById("newSessionBtn"),
    sidebarChatList: document.getElementById("sidebarChatList"),
    sidebarChatCount: document.getElementById("sidebarChatCount"),
    bookFilterInput: document.getElementById("bookFilterInput"),
    sidebarBookList: document.getElementById("sidebarBookList"),
    sidebarBookCount: document.getElementById("sidebarBookCount"),
    sidebarStats: document.getElementById("sidebarStats"),
    bookSelect: document.getElementById("bookSelect"),
    scopePillText: document.getElementById("scopePillText"),
    statusDot: document.getElementById("statusDot"),
    modelLabel: document.getElementById("modelLabel"),
    chunkBadge: document.getElementById("chunkBadge"),
    openDashboardBtn: document.getElementById("openDashboardBtn"),
    headerDashboardBtn: document.getElementById("headerDashboardBtn"),
    metricsQuickScores: document.getElementById("metricsQuickScores"),
    dashboardModal: document.getElementById("dashboardModal"),
    closeDashboardBtn: document.getElementById("closeDashboardBtn"),
    dashTabMetrics: document.getElementById("dashTabMetrics"),
    dashTabConfig: document.getElementById("dashTabConfig"),
    dashAutoRefresh: document.getElementById("dashAutoRefresh"),
    dashRefreshBtn: document.getElementById("dashRefreshBtn"),
    dashExportBtn: document.getElementById("dashExportBtn"),
    dashClearBtn: document.getElementById("dashClearBtn"),
    dashLastUpdated: document.getElementById("dashLastUpdated"),
    livePulseDot: document.getElementById("livePulseDot"),
    kpiFaithfulness: document.getElementById("kpiFaithfulness"),
    kpiFaithBar: document.getElementById("kpiFaithBar"),
    kpiPrecision: document.getElementById("kpiPrecision"),
    kpiPrecBar: document.getElementById("kpiPrecBar"),
    kpiLatency: document.getElementById("kpiLatency"),
    kpiRuns: document.getElementById("kpiRuns"),
    kpiDiscarded: document.getElementById("kpiDiscarded"),
    kpiRequeries: document.getElementById("kpiRequeries"),
    latTotalLabel: document.getElementById("latTotalLabel"),
    latBarRetrieval: document.getElementById("latBarRetrieval"),
    latBarReflection: document.getElementById("latBarReflection"),
    latBarGeneration: document.getElementById("latBarGeneration"),
    latRetrievalVal: document.getElementById("latRetrievalVal"),
    latReflectionVal: document.getElementById("latReflectionVal"),
    latGenerationVal: document.getElementById("latGenerationVal"),
    dashChartWrap: document.getElementById("dashChartWrap"),
    dashTraceCount: document.getElementById("dashTraceCount"),
    dashTraceList: document.getElementById("dashTraceList"),
    cfgReflection: document.getElementById("cfgReflection"),
    cfgRequery: document.getElementById("cfgRequery"),
    cfgEvaluation: document.getElementById("cfgEvaluation"),
    cfgModelSelect: document.getElementById("cfgModelSelect"),
    cfgSystemInfo: document.getElementById("cfgSystemInfo"),
    ingestBtn: document.getElementById("ingestBtn"),
    ingestIcon: document.getElementById("ingestIcon"),
    ingestBtnText: document.getElementById("ingestBtnText"),
    configBanner: document.getElementById("configBanner"),
    configBannerText: document.getElementById("configBannerText"),
    chatScroll: document.getElementById("chatScroll"),
    chatContainer: document.getElementById("chatContainer"),
    emptyState: document.getElementById("emptyState"),
    chatInput: document.getElementById("chatInput"),
    sendBtn: document.getElementById("sendBtn"),
    toast: document.getElementById("toast"),
    // Visor PDF
    pdfModal: document.getElementById("pdfModal"),
    pdfModalTitle: document.getElementById("pdfModalTitle"),
    pdfModalSubtitle: document.getElementById("pdfModalSubtitle"),
    pdfPrevBtn: document.getElementById("pdfPrevBtn"),
    pdfNextBtn: document.getElementById("pdfNextBtn"),
    pdfZoomBtn: document.getElementById("pdfZoomBtn"),
    pdfCloseBtn: document.getElementById("pdfCloseBtn"),
    pdfLoader: document.getElementById("pdfLoader"),
    pdfError: document.getElementById("pdfError"),
    pdfPageImg: document.getElementById("pdfPageImg"),
    // Hilo paralelo (Deep Dive)
    sideThread: document.getElementById("sideThread"),
    sideThreadBackdrop: document.getElementById("sideThreadBackdrop"),
    sideThreadTitle: document.getElementById("sideThreadTitle"),
    sideThreadScroll: document.getElementById("sideThreadScroll"),
    sideThreadMessages: document.getElementById("sideThreadMessages"),
    sideThreadInput: document.getElementById("sideThreadInput"),
    sideThreadSendBtn: document.getElementById("sideThreadSendBtn"),
    closeSideThreadBtn: document.getElementById("closeSideThreadBtn"),
    promoteSideThreadBtn: document.getElementById("promoteSideThreadBtn"),
    // Menú contextual y barra de selección
    contextMenu: document.getElementById("contextMenu"),
    contextMenuHeader: document.getElementById("contextMenuHeader"),
    contextMenuItems: document.getElementById("contextMenuItems"),
    selectionToolbar: document.getElementById("selectionToolbar"),
  };

  let toastTimer = null;
  let selectionTimer = null;

  /* ---------- Utilidades ---------- */

  function uid() {
    return `chat_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;
  }

  function showToast(message) {
    els.toast.textContent = message;
    els.toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => els.toast.classList.remove("show"), 3400);
  }

  function escapeHtml(text) {
    return String(text || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function prettyBookName(filename) {
    const key = String(filename || "").trim();
    if (key && state.bookTitles[key]) {
      return state.bookTitles[key];
    }
    return key
      .replace(/\.pdf$/i, "")
      .replace(/^\[[^\]]+\]\s*/, "")
      .replace(/^\d+\s+/, "")
      .replace(/\(z-lib\.org\)/gi, "")
      .replace(/[_-]+/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function isDesktop() {
    return window.matchMedia("(min-width: 1024px)").matches;
  }

  function scrollToBottom(el = els.chatScroll) {
    requestAnimationFrame(() => {
      el.scrollTop = el.scrollHeight;
    });
  }

  function autoResizeTextarea(textarea = els.chatInput, maxH = 160) {
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, maxH)}px`;
  }

  async function copyToClipboard(text) {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return;
    }
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.left = "-9999px";
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    document.execCommand("copy");
    document.body.removeChild(ta);
  }

  /* ---------- Manejo del teclado táctil en iPad / móvil ---------- */

  function syncViewportHeight() {
    if (window.visualViewport) {
      els.appShell.style.height = `${window.visualViewport.height}px`;
      els.appShell.style.maxHeight = `${window.visualViewport.height}px`;
      scrollToBottom();
    }
  }

  if (window.visualViewport) {
    window.visualViewport.addEventListener("resize", syncViewportHeight);
  }

  /* ---------- Gestión de múltiples chats en SQLite (/api/sessions) ---------- */

  function getActiveSession() {
    return state.activeSession;
  }

  function rememberLastSessionId(id) {
    try {
      if (id) localStorage.setItem(LAST_SESSION_KEY, id);
    } catch (_) {}
  }

  function getLastSessionId() {
    try {
      return localStorage.getItem(LAST_SESSION_KEY);
    } catch (_) {
      return null;
    }
  }

  async function migrateLegacyLocalStorageIfAny() {
    let raw = null;
    try {
      raw = localStorage.getItem(LEGACY_STORAGE_KEY);
    } catch (_) {
      return;
    }
    if (!raw) return;

    try {
      const parsed = JSON.parse(raw);
      const legacySessions = Array.isArray(parsed.sessions) ? parsed.sessions : [];
      const nonEmpty = legacySessions.filter((s) => Array.isArray(s.messages) && s.messages.length > 0);
      for (const s of nonEmpty.reverse()) {
        await fetch("/api/sessions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            title: s.title || "Chat migrado",
            book: s.book || "",
            messages: s.messages.map((m) => ({
              role: m.role === "assistant" ? "assistant" : "user",
              content: String(m.content || ""),
              modo: m.modo === "lectura_rapida" ? "lectura_rapida" : "consulta",
              scopeBook: m.scopeBook || null,
              citas: Array.isArray(m.citas) ? m.citas : [],
              isError: Boolean(m.isError),
            })),
          }),
        });
      }
      localStorage.removeItem(LEGACY_STORAGE_KEY);
      if (nonEmpty.length > 0) {
        showToast(`Se migraron ${nonEmpty.length} chat(s) locales a la base permanente SQLite.`);
      }
    } catch (_) {}
  }

  async function refreshSessionList() {
    const resp = await fetch("/api/sessions");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    state.sessions = Array.isArray(data.sessions) ? data.sessions : [];
    renderSidebarChatList();
    return state.sessions;
  }

  async function loadSessionsFromServer() {
    try {
      await migrateLegacyLocalStorageIfAny();
      let list = await refreshSessionList();
      if (list.length === 0) {
        await createNewSession({ title: "Nueva sesión", book: "" });
        return;
      }
      const preferredId = getLastSessionId();
      const targetId = list.some((s) => s.id === preferredId) ? preferredId : list[0].id;
      await switchSession(targetId, false);
    } catch (_) {
      showToast("No se pudieron cargar las sesiones desde SQLite.");
    }
  }

  function buildHistoryFromMessages(messages) {
    return (messages || [])
      .filter((m) => !m.isError && (m.role === "user" || m.role === "assistant"))
      .slice(-6)
      .map((m) => ({ role: m.role, content: m.content }));
  }

  function renderSidebarChatList() {
    els.sidebarChatList.innerHTML = "";
    els.sidebarChatCount.textContent = String(state.sessions.length);

    state.sessions.forEach((session) => {
      const isPending = state.pendingSessions.has(session.id);
      const isUnread = state.unreadSessions.has(session.id);

      const row = document.createElement("div");
      row.className = `chat-session-item ${session.id === state.activeSessionId ? "active" : ""}`;

      let statusIndicator = "";
      if (isPending) {
        statusIndicator = `<span class="h-2 w-2 rounded-full bg-amberSoft animate-pulse shrink-0" title="Generando respuesta..."></span>`;
      } else if (isUnread) {
        statusIndicator = `<span class="h-2 w-2 rounded-full bg-emerald-400 shrink-0" title="Respuesta lista"></span>`;
      }

      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "flex-1 flex items-center gap-2 min-w-0 text-left";
      btn.title = session.title;
      btn.innerHTML = `
        <svg class="shrink-0 opacity-70" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
        </svg>
        <span class="truncate flex-1">${escapeHtml(session.title || "Nueva sesión")}</span>
        ${statusIndicator}
      `;
      btn.addEventListener("click", () => switchSession(session.id, true));

      const delBtn = document.createElement("button");
      delBtn.type = "button";
      delBtn.className = "chat-delete-btn shrink-0";
      delBtn.title = "Eliminar chat";
      delBtn.setAttribute("aria-label", "Eliminar chat");
      delBtn.textContent = "×";
      delBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        deleteSession(session.id);
      });

      row.appendChild(btn);
      row.appendChild(delBtn);
      els.sidebarChatList.appendChild(row);
    });
  }

  function renderActiveSessionMessages() {
    Array.from(els.chatContainer.children).forEach((child) => {
      if (child !== els.emptyState) {
        child.remove();
      }
    });

    const session = getActiveSession();
    if (!session || !Array.isArray(session.messages) || session.messages.length === 0) {
      if (els.emptyState) els.emptyState.classList.remove("hidden");
      return;
    }

    if (els.emptyState) els.emptyState.classList.add("hidden");
    session.messages.forEach((msg) => {
      if (msg.role === "user") {
        appendUserMessage(msg.content, msg.modo || "consulta", msg.scopeBook || null, els.chatContainer);
      } else {
        appendAssistantMessage(
          msg.content,
          msg.citas || [],
          msg.modo || "consulta",
          Boolean(msg.isError),
          els.chatContainer
        );
      }
    });
    const pending = state.pendingSessions.get(session.id);
    if (pending) {
      appendLoadingMessage(pending.modo || "consulta", els.chatContainer);
    }
    syncActiveComposerBusy();
    scrollToBottom(els.chatScroll);
  }

  async function createNewSession({ title = "Nueva sesión", book = state.selectedBook, messages = [] } = {}) {
    const resp = await fetch("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title,
        book: book || "",
        messages: (messages || []).map((m) => ({
          role: m.role === "assistant" ? "assistant" : "user",
          content: String(m.content || ""),
          modo: m.modo === "lectura_rapida" ? "lectura_rapida" : "consulta",
          scopeBook: m.scopeBook || null,
          citas: Array.isArray(m.citas) ? m.citas : [],
          isError: Boolean(m.isError),
        })),
      }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const created = await resp.json();

    state.activeSessionId = created.id;
    state.activeSession = created;
    rememberLastSessionId(created.id);
    await refreshSessionList();
    setSelectedBook(created.book || "", false, false);
    renderActiveSessionMessages();
    return created;
  }

  async function switchSession(sessionId, closeDrawer = false) {
    try {
      const resp = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const fullSession = await resp.json();

      state.activeSessionId = fullSession.id;
      state.activeSession = fullSession;
      state.unreadSessions.delete(fullSession.id);
      rememberLastSessionId(fullSession.id);
      setSelectedBook(fullSession.book || "", closeDrawer, false);
      renderSidebarChatList();
      renderActiveSessionMessages();
      syncActiveComposerBusy();
    } catch (_) {
      showToast("No se pudo abrir la conversación seleccionada.");
    }
  }

  async function deleteSession(sessionId) {
    try {
      const resp = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {
        method: "DELETE",
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const list = await refreshSessionList();
      if (list.length === 0) {
        await createNewSession({ title: "Nueva sesión", book: "" });
      } else if (state.activeSessionId === sessionId) {
        await switchSession(list[0].id, false);
      }
    } catch (_) {
      showToast("No se pudo eliminar el chat.");
    }
  }

  /* ---------- Sidebar colapsable ---------- */

  function openMobileSidebar() {
    state.sidebarOpenMobile = true;
    els.sidebar.classList.remove("-translate-x-full");
    els.sidebarBackdrop.classList.remove("opacity-0", "pointer-events-none");
  }

  function closeMobileSidebar() {
    state.sidebarOpenMobile = false;
    els.sidebar.classList.add("-translate-x-full");
    els.sidebarBackdrop.classList.add("opacity-0", "pointer-events-none");
  }

  function toggleSidebar() {
    if (isDesktop()) {
      state.sidebarCollapsedDesktop = !state.sidebarCollapsedDesktop;
      els.sidebar.classList.toggle("desktop-collapsed", state.sidebarCollapsedDesktop);
    } else if (state.sidebarOpenMobile) {
      closeMobileSidebar();
    } else {
      openMobileSidebar();
    }
  }

  /* ---------- Selector de alcance (Filter by Book) ---------- */

  function setSelectedBook(bookName, closeDrawer = false, persist = true) {
    state.selectedBook = bookName || "";
    els.bookSelect.value = state.selectedBook;

    const active = getActiveSession();
    if (active) {
      active.book = state.selectedBook;
      if (persist && active.id) {
        fetch(`/api/sessions/${encodeURIComponent(active.id)}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ book: state.selectedBook }),
        }).catch(() => {});
      }
    }

    if (state.selectedBook) {
      const label = prettyBookName(state.selectedBook);
      els.scopePillText.textContent = label.length > 34 ? `${label.slice(0, 31)}…` : label;
    } else {
      els.scopePillText.textContent = `Todos los libros (${state.books.length})`;
    }

    renderSidebarBookList();
    if (closeDrawer && !isDesktop()) {
      closeMobileSidebar();
    }
  }

  function renderBookSelect() {
    const count = state.books.length;
    els.bookSelect.innerHTML = "";

    const allOpt = document.createElement("option");
    allOpt.value = "";
    allOpt.textContent = `Todos los libros (${count})`;
    els.bookSelect.appendChild(allOpt);

    state.books.forEach((book) => {
      const opt = document.createElement("option");
      opt.value = book;
      opt.textContent = prettyBookName(book);
      els.bookSelect.appendChild(opt);
    });

    if (state.selectedBook && state.books.includes(state.selectedBook)) {
      els.bookSelect.value = state.selectedBook;
    } else {
      state.selectedBook = "";
      els.bookSelect.value = "";
    }
  }

  function renderSidebarBookList() {
    const filter = state.filterQuery.trim().toLowerCase();
    els.sidebarBookList.innerHTML = "";

    const allBtn = document.createElement("button");
    allBtn.type = "button";
    allBtn.className = `book-item ${state.selectedBook === "" ? "active" : ""}`;
    allBtn.innerHTML = `
      <svg class="shrink-0 text-amberSoft" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
        <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
      </svg>
      <span class="truncate flex-1">Todos los libros (${state.books.length})</span>
    `;
    allBtn.addEventListener("click", () => setSelectedBook("", true));
    els.sidebarBookList.appendChild(allBtn);

    const visibleBooks = state.books.filter(
      (b) => !filter || prettyBookName(b).toLowerCase().includes(filter)
    );

    if (visibleBooks.length === 0 && state.books.length > 0) {
      const empty = document.createElement("p");
      empty.className = "px-3 py-4 text-xs text-muted text-center";
      empty.textContent = "Sin coincidencias en la biblioteca.";
      els.sidebarBookList.appendChild(empty);
      return;
    }

    visibleBooks.forEach((book) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.title = prettyBookName(book);
      btn.className = `book-item ${state.selectedBook === book ? "active" : ""}`;
      btn.innerHTML = `
        <svg class="shrink-0 opacity-70" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
        </svg>
        <span class="truncate flex-1">${escapeHtml(prettyBookName(book))}</span>
      `;
      btn.addEventListener("click", () => setSelectedBook(book, true));
      els.sidebarBookList.appendChild(btn);
    });
  }

  async function loadBooks() {
    try {
      const resp = await fetch("/books");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();

      state.books = Array.isArray(data.books) ? data.books : data.indexed_books || [];
      state.bookTitles = data.book_titles || {};
      renderBookSelect();
      const active = getActiveSession();
      setSelectedBook(active && active.book ? active.book : state.selectedBook, false);

      els.sidebarBookCount.textContent = String(state.books.length);
      els.chunkBadge.textContent = `${Number(data.total_chunks || 0).toLocaleString("es-AR")} frags`;
      els.modelLabel.textContent = data.llm_model ? `ONNX · ${data.llm_model}` : "ONNX · Gemini";

      if (state.books.length === 0) {
        els.statusDot.className = "h-1.5 w-1.5 rounded-full bg-amber-400";
        els.sidebarStats.textContent =
          data.available_pdfs && data.available_pdfs.length > 0
            ? `${data.available_pdfs.length} PDF(s) pendientes de indexar`
            : "Sin libros indexados";
      } else {
        els.statusDot.className = "h-1.5 w-1.5 rounded-full bg-emerald-400";
        els.sidebarStats.textContent = `${state.books.length} libros indexados`;
      }

      if (!data.api_key_configured) {
        els.configBannerText.textContent =
          "Falta configurar GEMINI_API_KEY en el archivo .env del proyecto para generar respuestas.";
        els.configBanner.classList.remove("hidden");
      } else {
        els.configBanner.classList.add("hidden");
      }
    } catch (_) {
      els.statusDot.className = "h-1.5 w-1.5 rounded-full bg-rose-400";
      els.sidebarStats.textContent = "Sin conexión con el servidor";
    }
  }

  /* ---------- Visor de página original del PDF (GET /page-image) ---------- */

  function updatePdfModalNav() {
    const { page, totalPages, zoomed } = state.pdfViewer;
    els.pdfModalSubtitle.textContent = totalPages
      ? `Página ${page} de ${totalPages}`
      : `Página ${page}`;
    els.pdfPrevBtn.disabled = page <= 1;
    els.pdfNextBtn.disabled = totalPages ? page >= totalPages : false;
    els.pdfZoomBtn.textContent = zoomed ? "Ajustar" : "Zoom 100%";
    els.pdfPageImg.classList.toggle("pdf-img-fit", !zoomed);
    els.pdfPageImg.classList.toggle("pdf-img-zoom", zoomed);
  }

  async function loadPdfPageImage(book, page) {
    state.pdfViewer.book = book;
    state.pdfViewer.page = page;
    els.pdfModalTitle.textContent = prettyBookName(book);
    updatePdfModalNav();

    els.pdfLoader.classList.remove("hidden");
    els.pdfError.classList.add("hidden");
    els.pdfPageImg.classList.add("hidden");

    const params = new URLSearchParams({ book, page: String(page) });
    try {
      const resp = await fetch(`/page-image?${params.toString()}`);
      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.detail || `Error HTTP ${resp.status}`);
      }
      const totalHeader = resp.headers.get("X-PDF-Total-Pages");
      state.pdfViewer.totalPages = totalHeader ? Number(totalHeader) : null;

      const blob = await resp.blob();
      if (state.pdfViewer.objectUrl) {
        URL.revokeObjectURL(state.pdfViewer.objectUrl);
      }
      const url = URL.createObjectURL(blob);
      state.pdfViewer.objectUrl = url;

      els.pdfPageImg.src = url;
      els.pdfPageImg.alt = `${prettyBookName(book)} — Página ${page}`;
      els.pdfLoader.classList.add("hidden");
      els.pdfPageImg.classList.remove("hidden");
      updatePdfModalNav();
    } catch (err) {
      els.pdfLoader.classList.add("hidden");
      els.pdfError.textContent = err.message || "No se pudo cargar la página original del PDF.";
      els.pdfError.classList.remove("hidden");
    }
  }

  function openPdfModal(book, page) {
    state.pdfViewer.open = true;
    state.pdfViewer.zoomed = false;
    state.pdfViewer.totalPages = null;
    els.pdfModal.classList.remove("hidden");
    els.pdfModal.classList.add("flex");
    loadPdfPageImage(book, Math.max(1, Number(page) || 1));
  }

  function closePdfModal() {
    state.pdfViewer.open = false;
    els.pdfModal.classList.add("hidden");
    els.pdfModal.classList.remove("flex");
    if (state.pdfViewer.objectUrl) {
      URL.revokeObjectURL(state.pdfViewer.objectUrl);
      state.pdfViewer.objectUrl = null;
    }
    els.pdfPageImg.removeAttribute("src");
  }

  /* ---------- Renderizado Markdown + Mermaid.js + Highlight.js + Copiar código ---------- */

  function ensureMermaidInitialized() {
    if (!window.mermaid || state.mermaidReady) return Boolean(window.mermaid);
    window.mermaid.initialize({
      startOnLoad: false,
      theme: "dark",
      securityLevel: "strict",
      fontFamily: "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
      themeVariables: {
        darkMode: true,
        background: "#0c0e13",
        primaryColor: "#1c2029",
        primaryTextColor: "#e6eaf2",
        primaryBorderColor: "#818cf8",
        lineColor: "#8b95a7",
        secondaryColor: "#161920",
        tertiaryColor: "#13161d",
        edgeLabelBackground: "#161920",
      },
    });
    state.mermaidReady = true;
    return true;
  }

  function renderMermaidBlock(pre, rawCode) {
    const card = document.createElement("div");
    card.className = "mermaid-card";

    const header = document.createElement("div");
    header.className = "mermaid-card-header";
    header.innerHTML = `
      <span class="inline-flex items-center gap-1.5 text-[11px] font-mono font-medium text-amberSoft/90">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="3" y="3" width="7" height="7" rx="1" />
          <rect x="14" y="3" width="7" height="7" rx="1" />
          <rect x="14" y="14" width="7" height="7" rx="1" />
          <path d="M10 6.5h4M17.5 10v4" />
        </svg>
        <span>Diagrama · Mermaid</span>
      </span>
      <div class="flex items-center gap-1.5">
        <button type="button" class="toggle-mermaid-src copy-code-btn">Ver código</button>
        <button type="button" class="copy-mermaid-btn copy-code-btn">Copiar</button>
      </div>
    `;

    const stage = document.createElement("div");
    stage.className = "mermaid-stage custom-scroll";
    stage.innerHTML = `<span class="text-xs text-muted">Renderizando diagrama...</span>`;

    const sourcePre = document.createElement("pre");
    sourcePre.className = "mermaid-source hidden";
    sourcePre.textContent = rawCode;

    const toggleBtn = header.querySelector(".toggle-mermaid-src");
    toggleBtn.addEventListener("click", () => {
      const hidden = sourcePre.classList.toggle("hidden");
      toggleBtn.textContent = hidden ? "Ver código" : "Ocultar código";
    });

    const copyBtn = header.querySelector(".copy-mermaid-btn");
    copyBtn.addEventListener("click", async () => {
      try {
        await copyToClipboard(rawCode);
        copyBtn.classList.add("copied");
        copyBtn.textContent = "¡Copiado!";
        setTimeout(() => {
          copyBtn.classList.remove("copied");
          copyBtn.textContent = "Copiar";
        }, 1800);
      } catch (_) {
        showToast("No se pudo copiar el código Mermaid.");
      }
    });

    card.appendChild(header);
    card.appendChild(stage);
    card.appendChild(sourcePre);
    pre.parentNode.replaceChild(card, pre);

    if (!ensureMermaidInitialized()) {
      stage.innerHTML = `<span class="text-xs text-muted">Mermaid no disponible sin conexión; mostrando código fuente.</span>`;
      sourcePre.classList.remove("hidden");
      toggleBtn.textContent = "Ocultar código";
      return;
    }

    const diagramId = `mermaid-svg-${++state.mermaidCounter}`;
    window.mermaid
      .render(diagramId, rawCode)
      .then(({ svg, bindFunctions }) => {
        stage.innerHTML = svg;
        if (typeof bindFunctions === "function") {
          bindFunctions(stage);
        }
      })
      .catch(() => {
        stage.innerHTML = `<span class="text-xs text-amber-300/90">No se pudo renderizar el SVG de este diagrama; podés revisar el código fuente debajo.</span>`;
        sourcePre.classList.remove("hidden");
        toggleBtn.textContent = "Ocultar código";
      });
  }

  function fallbackMarkdown(md) {
    const codeBlocks = [];
    let html = String(md || "").replace(/```([\w+-]*)\n([\s\S]*?)```/g, (_, lang, code) => {
      const idx = codeBlocks.length;
      codeBlocks.push(
        `<pre><code class="language-${escapeHtml(lang || "text")}">${escapeHtml(code)}</code></pre>`
      );
      return `@@CODEBLOCK_${idx}@@`;
    });

    html = escapeHtml(html)
      .replace(/^### (.*)$/gm, "<h3>$1</h3>")
      .replace(/^## (.*)$/gm, "<h2>$1</h2>")
      .replace(/^# (.*)$/gm, "<h1>$1</h1>")
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/^\s*[-*] (.*)$/gm, "<li>$1</li>")
      .replace(/(<li>.*<\/li>)/gs, "<ul>$1</ul>")
      .replace(/\n{2,}/g, "</p><p>");

    html = `<p>${html}</p>`;
    codeBlocks.forEach((block, idx) => {
      html = html.replace(`@@CODEBLOCK_${idx}@@`, block);
    });
    return html;
  }

  function renderMarkdown(md) {
    let rawHtml = "";
    if (window.marked) {
      window.marked.setOptions({ gfm: true, breaks: true });
      rawHtml = window.marked.parse(md || "");
    } else {
      rawHtml = fallbackMarkdown(md || "");
    }
    return window.DOMPurify ? window.DOMPurify.sanitize(rawHtml) : rawHtml;
  }

  function enhanceCodeBlocks(rootEl) {
    const pres = rootEl.querySelectorAll("pre");
    pres.forEach((pre) => {
      if (
        pre.parentElement &&
        (pre.parentElement.classList.contains("code-card") ||
          pre.parentElement.classList.contains("mermaid-card"))
      ) {
        return;
      }

      const codeEl = pre.querySelector("code");
      const rawCode = (codeEl ? codeEl.textContent : pre.textContent) || "";
      let lang = "código";

      if (codeEl) {
        const match = codeEl.className.match(/language-([\w+-]+)/);
        if (match) lang = match[1].toLowerCase();
      }

      if (lang === "mermaid") {
        renderMermaidBlock(pre, rawCode.trim());
        return;
      }

      if (codeEl && window.hljs) {
        try {
          window.hljs.highlightElement(codeEl);
        } catch (_) {}
      }

      const card = document.createElement("div");
      card.className = "code-card";

      const header = document.createElement("div");
      header.className = "code-card-header";

      const langLabel = document.createElement("span");
      langLabel.className = "code-lang";
      langLabel.textContent = lang;

      const copyBtn = document.createElement("button");
      copyBtn.type = "button";
      copyBtn.className = "copy-code-btn";
      copyBtn.innerHTML = `
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
        </svg>
        <span>Copiar código</span>
      `;
      copyBtn.addEventListener("click", async () => {
        try {
          await copyToClipboard(rawCode);
          copyBtn.classList.add("copied");
          copyBtn.querySelector("span").textContent = "¡Copiado!";
          setTimeout(() => {
            copyBtn.classList.remove("copied");
            copyBtn.querySelector("span").textContent = "Copiar código";
          }, 1800);
        } catch (_) {
          showToast("No se pudo copiar el bloque de código.");
        }
      });

      header.appendChild(langLabel);
      header.appendChild(copyBtn);
      pre.parentNode.insertBefore(card, pre);
      card.appendChild(header);
      card.appendChild(pre);
    });
  }

  /* ---------- Fuentes colapsables ([Título — Pág. X]) ---------- */

  function buildCitationsSection(citas) {
    const section = document.createElement("div");
    section.className = "mt-5 pt-4 border-t border-hairline";

    const label = document.createElement("div");
    label.className = "flex items-center gap-2 text-[11px] font-medium uppercase tracking-wider text-muted mb-2.5";
    label.innerHTML = `
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
        <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
      </svg>
      <span>Fuentes verificables (${citas.length}) · Clic o clic derecho para opciones</span>
    `;

    const pillsWrap = document.createElement("div");
    pillsWrap.className = "flex flex-wrap gap-1.5";

    const drawerSlot = document.createElement("div");
    let activeKey = null;

    citas.forEach((cita, index) => {
      const displayTitle = (cita.titulo || prettyBookName(cita.libro)).trim();
      const shortTitle = displayTitle.length > 44 ? `${displayTitle.slice(0, 41)}…` : displayTitle;
      const key = `${cita.libro}::${cita.pagina}::${index}`;

      const pill = document.createElement("button");
      pill.type = "button";
      pill.className = "citation-pill";
      pill.dataset.book = cita.libro;
      pill.dataset.title = displayTitle;
      pill.dataset.page = String(cita.pagina);
      pill.dataset.excerpt = (cita.extracto || "").slice(0, 400);
      pill.title = `${displayTitle} — Pág. ${cita.pagina}`;
      pill.innerHTML = `
        <span>[${escapeHtml(shortTitle)} — Pág. ${escapeHtml(cita.pagina)}]</span>
        <svg class="opacity-65 shrink-0" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="6 9 12 15 18 9" />
        </svg>
      `;

      pill.addEventListener("click", () => {
        pillsWrap.querySelectorAll(".citation-pill").forEach((el) => el.classList.remove("active"));

        if (activeKey === key) {
          activeKey = null;
          drawerSlot.innerHTML = "";
          return;
        }

        activeKey = key;
        pill.classList.add("active");
        drawerSlot.innerHTML = "";

        const drawer = document.createElement("div");
        drawer.className = "citation-drawer";
        drawer.innerHTML = `
          <div class="flex flex-wrap items-center justify-between gap-2 px-3.5 py-2.5 bg-surface border-b border-hairline">
            <div class="min-w-0 flex-1">
              <p class="text-xs font-medium text-ink truncate">${escapeHtml(displayTitle)}</p>
              <p class="text-[11px] font-mono text-amberSoft/90">Fragmento original · Página ${escapeHtml(cita.pagina)}</p>
            </div>
            <div class="flex items-center gap-1.5 shrink-0">
              <button
                type="button"
                class="view-pdf-page-btn inline-flex items-center gap-1.5 rounded-lg border border-amber-500/35 bg-amber-500/10 hover:bg-amber-500/20 px-2.5 py-1 text-xs font-medium text-amber-200 transition-colors"
              >
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                  <circle cx="12" cy="12" r="3" />
                </svg>
                <span>Ver página original</span>
              </button>
              <button type="button" class="close-drawer-btn text-xs text-muted hover:text-ink px-2 py-1 rounded-md hover:bg-elevated transition-colors">
                Cerrar
              </button>
            </div>
          </div>
          <div class="citation-excerpt custom-scroll px-4 py-3">${escapeHtml(cita.extracto || "Sin extracto disponible.")}</div>
        `;

        drawer.querySelector(".view-pdf-page-btn").addEventListener("click", () => {
          openPdfModal(cita.libro, cita.pagina);
        });

        drawer.querySelector(".close-drawer-btn").addEventListener("click", () => {
          activeKey = null;
          pill.classList.remove("active");
          drawerSlot.innerHTML = "";
        });

        drawerSlot.appendChild(drawer);
      });

      pillsWrap.appendChild(pill);
    });

    section.appendChild(label);
    section.appendChild(pillsWrap);
    section.appendChild(drawerSlot);
    return section;
  }

  /* ---------- Renderizado de mensajes (reutilizable en chat principal e hilo paralelo) ---------- */

  function appendUserMessage(text, modo, scopeBook, container = els.chatContainer) {
    if (container === els.chatContainer && els.emptyState) {
      els.emptyState.classList.add("hidden");
    }
    const article = document.createElement("article");
    article.className = "msg-enter flex flex-col items-end gap-1.5";

    const modeLabel = modo === "lectura_rapida" ? "Lectura rápida" : "Consulta";
    const scopeLabel = scopeBook ? prettyBookName(scopeBook) : `Todos (${state.books.length})`;
    const shortScope = scopeLabel.length > 32 ? `${scopeLabel.slice(0, 29)}…` : scopeLabel;

    article.innerHTML = `
      <div class="flex items-center gap-2 px-1 text-[11px] text-muted">
        <span class="font-mono text-[10px] opacity-80">${escapeHtml(shortScope)}</span>
        <span>·</span>
        <span class="${modo === "lectura_rapida" ? "text-amberSoft" : "text-indigoSoft"} font-medium">${modeLabel}</span>
      </div>
      <div class="max-w-[92%] sm:max-w-[85%] rounded-2xl rounded-br-sm bg-elevated border border-hairline px-4 py-2.5 text-[14.5px] leading-relaxed text-ink">
        ${escapeHtml(text).replace(/\n/g, "<br>")}
      </div>
    `;
    container.appendChild(article);
  }

  function appendLoadingMessage(modo, container = els.chatContainer) {
    if (container === els.chatContainer && els.emptyState) {
      els.emptyState.classList.add("hidden");
    }
    const article = document.createElement("article");
    article.className = "msg-enter flex flex-col gap-2 loading-msg";
    const modeLabel = modo === "lectura_rapida" ? "Lectura rápida" : "Consulta técnica";
    article.innerHTML = `
      <div class="flex items-center gap-2 text-xs text-muted">
        <span class="inline-flex h-5 w-5 items-center justify-center rounded-md bg-surface border border-hairline text-amberSoft text-[10px] font-semibold">R</span>
        <span class="font-medium text-ink/90">${modeLabel}</span>
      </div>
      <div class="rounded-2xl bg-surface border border-hairline px-5 py-4 flex items-center gap-3 text-sm text-muted">
        <span>Buscando fragmentos relevantes y redactando</span>
        <span class="typing-dots inline-flex items-center gap-1"><span></span><span></span><span></span></span>
      </div>
    `;
    container.appendChild(article);
    return article;
  }

  function appendAssistantMessage(
    markdown,
    citas,
    modo,
    isError = false,
    container = els.chatContainer,
    agenticTrace = null
  ) {
    const article = document.createElement("article");
    article.className = "msg-enter flex flex-col gap-2";

    const modeBadge =
      modo === "lectura_rapida"
        ? `<span class="rounded-full bg-amber-500/10 border border-amber-500/30 px-2 py-0.5 text-[10.5px] font-medium text-amber-300">Lectura rápida</span>`
        : `<span class="rounded-full bg-indigo-500/10 border border-indigo-500/30 px-2 py-0.5 text-[10.5px] font-medium text-indigo-300">Consulta</span>`;

    let traceBadge = "";
    if (!isError && agenticTrace && agenticTrace.reflexion_ejecutada) {
      const kept = agenticTrace.chunks_retenidos || 0;
      const init = agenticTrace.chunks_iniciales || kept;
      const parts = [`Reflexión ${kept}/${init}`];
      if (agenticTrace.rebusqueda_activada) parts.push("Re-búsqueda");
      if (agenticTrace.conflictos_resueltos) parts.push("Conflictos resueltos");
      const titleTip = agenticTrace.conflictos_resueltos
        ? `title="${escapeHtml(agenticTrace.conflictos_resueltos)}"`
        : "";
      traceBadge = `<span ${titleTip} class="rounded-full bg-emerald-500/10 border border-emerald-500/25 px-2 py-0.5 text-[10px] font-mono text-emerald-300/90">${escapeHtml(parts.join(" · "))}</span>`;
    }

    const header = document.createElement("div");
    header.className = "flex flex-wrap items-center gap-2 px-1";
    header.innerHTML = `
      <span class="inline-flex h-5 w-5 items-center justify-center rounded-md bg-surface border border-hairline text-amberSoft text-[10px] font-semibold">R</span>
      <span class="text-xs font-medium text-ink/90">Tutor RAG</span>
      ${modeBadge}
      ${traceBadge}
    `;

    const isSide = container === els.sideThreadMessages;
    const card = document.createElement("div");
    card.className = isError
      ? "rounded-2xl border border-rose-500/35 bg-rose-500/10 px-4 py-3 text-sm text-rose-200"
      : isSide
      ? "rounded-2xl border border-hairline bg-canvas px-4 py-4 shadow-sm shadow-black/20"
      : "rounded-2xl border border-hairline bg-surface px-5 sm:px-7 py-5 sm:py-6 shadow-sm shadow-black/20";

    const body = document.createElement("div");
    body.className = isError ? "" : "prose-editorial";
    body.innerHTML = isError ? escapeHtml(markdown) : renderMarkdown(markdown);
    if (!isError) {
      enhanceCodeBlocks(body);
    }
    card.appendChild(body);

    if (!isError && Array.isArray(citas) && citas.length > 0) {
      card.appendChild(buildCitationsSection(citas));
    }

    article.appendChild(header);
    article.appendChild(card);
    container.appendChild(article);
  }

  /* ---------- Envío de consultas en el chat principal (concurrente por sesión) ---------- */

  function isActiveSessionBusy() {
    return Boolean(state.activeSessionId && state.pendingSessions.has(state.activeSessionId));
  }

  function syncActiveComposerBusy() {
    const busy = isActiveSessionBusy();
    els.sendBtn.disabled = busy;
    document.querySelectorAll(".study-chip, .starter-card").forEach((b) => {
      b.disabled = busy;
    });
  }

  async function sendChat(mensaje, modo = "consulta", overrideBook = undefined) {
    const trimmed = String(mensaje || "").trim();
    if (modo === "consulta" && !trimmed) return;

    const session = getActiveSession() || (await createNewSession());
    const targetSessionId = session.id;
    if (state.pendingSessions.has(targetSessionId)) {
      showToast("Este chat ya está generando una respuesta. Podés cambiar a otro chat mientras tanto.");
      return;
    }

    const displayMsg = trimmed || "Lectura rápida de un concepto clave de mi biblioteca";
    const scopeBook = overrideBook !== undefined ? overrideBook : state.selectedBook || null;

    if (!Array.isArray(session.messages)) {
      session.messages = [];
    }

    const historyPayload = buildHistoryFromMessages(session.messages);
    session.messages.push({ role: "user", content: displayMsg, modo, scopeBook });

    state.pendingSessions.set(targetSessionId, { modo, startedAt: Date.now() });
    renderSidebarChatList();
    syncActiveComposerBusy();

    if (state.activeSessionId === targetSessionId) {
      appendUserMessage(displayMsg, modo, scopeBook, els.chatContainer);
      appendLoadingMessage(modo, els.chatContainer);
      scrollToBottom(els.chatScroll);
    }

    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mensaje: displayMsg,
          modo,
          libro: scopeBook,
          historial: historyPayload,
          session_id: targetSessionId,
        }),
      });

      const data = await resp.json().catch(() => ({}));
      state.pendingSessions.delete(targetSessionId);

      if (!resp.ok) {
        throw new Error(data.detail || `Error HTTP ${resp.status}`);
      }

      const finalModo = data.modo || modo;
      const citas = data.citas || [];

      if (state.activeSessionId === targetSessionId) {
        els.chatContainer.querySelectorAll(".loading-msg").forEach((el) => el.remove());
        appendAssistantMessage(
          data.respuesta,
          citas,
          finalModo,
          false,
          els.chatContainer,
          data.agentic_trace || null
        );
        if (state.activeSession && state.activeSession.id === targetSessionId) {
          state.activeSession.messages.push({
            role: "assistant",
            content: data.respuesta,
            citas,
            modo: finalModo,
          });
        }
        scrollToBottom(els.chatScroll);
      } else {
        state.unreadSessions.add(targetSessionId);
        const match = state.sessions.find((s) => s.id === targetSessionId);
        const label = match ? match.title : "otro chat";
        showToast(`Respuesta lista en "${label}".`);
      }

      await refreshSessionList().catch(() => {});
      setTimeout(() => loadMetricsSummary(false), 4500);
    } catch (err) {
      state.pendingSessions.delete(targetSessionId);
      if (state.activeSessionId === targetSessionId) {
        els.chatContainer.querySelectorAll(".loading-msg").forEach((el) => el.remove());
        appendAssistantMessage(
          err.message || "Error al conectar con el servidor.",
          [],
          modo,
          true,
          els.chatContainer
        );
        scrollToBottom(els.chatScroll);
      } else {
        showToast(err.message || "Error en una consulta en segundo plano.");
      }
      renderSidebarChatList();
    } finally {
      syncActiveComposerBusy();
    }
  }

  /* ---------- Hilo paralelo (Panel lateral derecho Deep Dive) ---------- */

  function openSideThreadDrawer() {
    state.sideThread.open = true;
    els.sideThread.classList.remove("translate-x-full");
    els.sideThreadBackdrop.classList.remove("opacity-0", "pointer-events-none");
  }

  function closeSideThreadDrawer() {
    state.sideThread.open = false;
    els.sideThread.classList.add("translate-x-full");
    els.sideThreadBackdrop.classList.add("opacity-0", "pointer-events-none");
  }

  async function sendSideThreadMessage(mensaje, modo = "consulta") {
    if (state.sideThread.loading) return;
    const trimmed = String(mensaje || "").trim();
    if (!trimmed) return;

    const scopeBook = state.sideThread.book || null;
    const historyPayload = buildHistoryFromMessages(state.sideThread.messages);

    state.sideThread.messages.push({ role: "user", content: trimmed, modo, scopeBook });
    appendUserMessage(trimmed, modo, scopeBook, els.sideThreadMessages);
    const loader = appendLoadingMessage(modo, els.sideThreadMessages);
    scrollToBottom(els.sideThreadScroll);

    state.sideThread.loading = true;
    els.sideThreadSendBtn.disabled = true;

    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mensaje: trimmed,
          modo,
          libro: scopeBook,
          historial: historyPayload,
        }),
      });
      const data = await resp.json().catch(() => ({}));
      loader.remove();
      if (!resp.ok) {
        throw new Error(data.detail || `Error HTTP ${resp.status}`);
      }

      const finalModo = data.modo || modo;
      const citas = data.citas || [];
      appendAssistantMessage(data.respuesta, citas, finalModo, false, els.sideThreadMessages);
      state.sideThread.messages.push({
        role: "assistant",
        content: data.respuesta,
        citas,
        modo: finalModo,
      });
      scrollToBottom(els.sideThreadScroll);
    } catch (err) {
      loader.remove();
      appendAssistantMessage(
        err.message || "Error al conectar con el servidor.",
        [],
        modo,
        true,
        els.sideThreadMessages
      );
      scrollToBottom(els.sideThreadScroll);
    } finally {
      state.sideThread.loading = false;
      els.sideThreadSendBtn.disabled = false;
    }
  }

  function startParallelInquiry({ title, book = null, prompt, modo = "consulta" }) {
    state.sideThread.title = title || "Consulta rápida";
    state.sideThread.book = book;
    state.sideThread.messages = [];
    els.sideThreadTitle.textContent = state.sideThread.title;
    els.sideThreadMessages.innerHTML = "";
    openSideThreadDrawer();
    sendSideThreadMessage(prompt, modo);
  }

  async function promoteSideThreadToNewChat() {
    if (state.sideThread.messages.length === 0) {
      showToast("Todavía no hay mensajes en el hilo paralelo.");
      return;
    }
    try {
      await createNewSession({
        title: state.sideThread.title || "Consulta paralela",
        book: state.sideThread.book || "",
        messages: state.sideThread.messages,
      });
      closeSideThreadDrawer();
      showToast("Hilo paralelo guardado permanentemente como chat en SQLite.");
    } catch (_) {
      showToast("No se pudo guardar el hilo paralelo.");
    }
  }

  /* ---------- Menú contextual (clic derecho) y Barra de selección (iPad/táctil) ---------- */

  function hideContextMenu() {
    els.contextMenu.classList.add("hidden");
  }

  function hideSelectionToolbar() {
    els.selectionToolbar.classList.add("hidden");
    els.selectionToolbar.classList.remove("flex");
  }

  function runSelectionAction(action, selectedText) {
    const clean = String(selectedText || "").trim();
    if (!clean) return;
    hideContextMenu();
    hideSelectionToolbar();
    try {
      window.getSelection().removeAllRanges();
    } catch (_) {}

    const shortLabel = clean.length > 42 ? `${clean.slice(0, 39)}…` : clean;
    const activeBook = state.selectedBook || null;

    if (action === "parallel") {
      startParallelInquiry({
        title: shortLabel,
        book: activeBook,
        prompt: `Explicame en profundidad "${clean}" según mis libros, con contexto técnico, ejemplos prácticos y citas exactas.`,
      });
      return;
    }

    if (action === "new_chat") {
      createNewSession({ title: shortLabel, book: activeBook || "" })
        .then(() =>
          sendChat(
            `Explicame en profundidad "${clean}" según mis libros, con ejemplos prácticos y cuándo aplicarlo o evitarlo.`,
            "consulta",
            activeBook
          )
        )
        .catch(() => showToast("No se pudo crear el chat nuevo."));
      return;
    }

    if (action === "code") {
      startParallelInquiry({
        title: `Código · ${shortLabel}`,
        book: activeBook,
        prompt: `Mostrame cómo implementar "${clean}" en código limpio y listo para producción según mis libros.`,
      });
      return;
    }

    if (action === "compare") {
      startParallelInquiry({
        title: `Comparar · ${shortLabel}`,
        book: null,
        prompt: `Compará qué dicen distintos libros de mi biblioteca sobre "${clean}".`,
      });
    }
  }

  function showContextMenu(x, y, headerText, items) {
    hideSelectionToolbar();
    els.contextMenuHeader.textContent = headerText;
    els.contextMenuItems.innerHTML = "";

    items.forEach((item) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.role = "menuitem";
      btn.className = `ctx-item ${item.accent ? "ctx-item-accent" : ""}`;
      btn.innerHTML = `${item.icon || ""}<span>${escapeHtml(item.label)}</span>`;
      btn.addEventListener("click", () => {
        hideContextMenu();
        item.onClick();
      });
      els.contextMenuItems.appendChild(btn);
    });

    els.contextMenu.classList.remove("hidden");
    const rect = els.contextMenu.getBoundingClientRect();
    const clampedX = Math.min(x, window.innerWidth - rect.width - 10);
    const clampedY = Math.min(y, window.innerHeight - rect.height - 10);
    els.contextMenu.style.left = `${Math.max(8, clampedX)}px`;
    els.contextMenu.style.top = `${Math.max(8, clampedY)}px`;
  }

  function handleContextMenu(e) {
    // 1. Clic derecho sobre una etiqueta de cita [Libro — Pág. X]
    const pill = e.target.closest(".citation-pill");
    if (pill) {
      e.preventDefault();
      const book = pill.dataset.book;
      const title = pill.dataset.title || prettyBookName(book);
      const page = Number(pill.dataset.page || 1);
      const excerpt = pill.dataset.excerpt || "";

      showContextMenu(e.clientX, e.clientY, `${title} · Pág. ${page}`, [
        {
          label: "Consultar referencia en hilo paralelo",
          accent: true,
          onClick: () =>
            startParallelInquiry({
              title: `${title} (Pág. ${page})`,
              book,
              prompt: `Explicame en profundidad los conceptos clave que desarrolla "${title}" alrededor de la página ${page}: "${excerpt.slice(0, 180)}..."`,
            }),
        },
        {
          label: "Abrir nuevo chat en este libro",
          onClick: () => {
            createNewSession({ title: `${title} · Pág. ${page}`, book })
              .then(() =>
                sendChat(
                  `Profundizá y dame un ejemplo práctico sobre el tema tratado en la página ${page} de "${title}".`,
                  "consulta",
                  book
                )
              )
              .catch(() => showToast("No se pudo crear el chat nuevo."));
          },
        },
        {
          label: "Ver página original del PDF",
          onClick: () => openPdfModal(book, page),
        },
      ]);
      return;
    }

    // 2. Clic derecho con texto seleccionado dentro del chat o del hilo paralelo
    const insideReading = e.target.closest("#chatContainer, #sideThreadMessages");
    const sel = window.getSelection();
    const selectedText = sel ? sel.toString().trim() : "";

    if (insideReading && selectedText.length >= 2 && selectedText.length <= 320) {
      e.preventDefault();
      const preview = selectedText.length > 34 ? `"${selectedText.slice(0, 31)}…"` : `"${selectedText}"`;
      showContextMenu(e.clientX, e.clientY, `Referencia: ${preview}`, [
        {
          label: "Consultar en hilo paralelo",
          accent: true,
          onClick: () => runSelectionAction("parallel", selectedText),
        },
        {
          label: "Abrir en chat nuevo",
          onClick: () => runSelectionAction("new_chat", selectedText),
        },
        {
          label: "Ver ejemplo en código (paralelo)",
          onClick: () => runSelectionAction("code", selectedText),
        },
        {
          label: "Comparar autores (paralelo)",
          onClick: () => runSelectionAction("compare", selectedText),
        },
      ]);
    }
  }

  function updateFloatingSelectionToolbar() {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || sel.rangeCount === 0) {
      hideSelectionToolbar();
      return;
    }

    const text = sel.toString().trim();
    if (text.length < 2 || text.length > 320) {
      hideSelectionToolbar();
      return;
    }

    const anchorNode = sel.anchorNode;
    const containerEl =
      anchorNode && anchorNode.nodeType === 1 ? anchorNode : anchorNode && anchorNode.parentElement;
    if (!containerEl || !containerEl.closest(".prose-editorial, .citation-excerpt")) {
      hideSelectionToolbar();
      return;
    }

    const range = sel.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    if (!rect || (rect.width === 0 && rect.height === 0)) {
      hideSelectionToolbar();
      return;
    }

    state.currentSelectionText = text;
    els.selectionToolbar.classList.remove("hidden");
    els.selectionToolbar.classList.add("flex");

    const tbRect = els.selectionToolbar.getBoundingClientRect();
    const left = Math.min(
      Math.max(8, rect.left + rect.width / 2 - tbRect.width / 2),
      window.innerWidth - tbRect.width - 8
    );
    const top = rect.top > tbRect.height + 16 ? rect.top - tbRect.height - 10 : rect.bottom + 10;

    els.selectionToolbar.style.left = `${left}px`;
    els.selectionToolbar.style.top = `${Math.max(8, top)}px`;
  }

  /* ---------- Modos rápidos de estudio ---------- */

  function fillTemplateAndSelectPlaceholder(template, placeholder = "[tema]") {
    els.chatInput.value = template;
    autoResizeTextarea();
    els.chatInput.focus();
    const start = template.indexOf(placeholder);
    if (start !== -1) {
      els.chatInput.setSelectionRange(start, start + placeholder.length);
    }
  }

  function handleStudyModeAction(action) {
    if (isActiveSessionBusy()) return;
    const currentInput = els.chatInput.value.trim();
    const activeSession = getActiveSession();
    const hasMessages = activeSession && activeSession.messages.length > 0;

    if (action === "lectura_rapida") {
      els.chatInput.value = "";
      autoResizeTextarea();
      const prompt = currentInput ? `Lectura rápida sobre: ${currentInput}` : "Lectura rápida";
      sendChat(prompt, "lectura_rapida");
      return;
    }

    if (action === "explicar") {
      if (currentInput && !currentInput.includes("[tema]")) {
        els.chatInput.value = "";
        autoResizeTextarea();
        sendChat(
          `Explicame ${currentInput} de forma clara con ejemplos prácticos y cuándo evitarlo según el libro.`,
          "consulta"
        );
      } else {
        fillTemplateAndSelectPlaceholder(
          "Explicame [tema] de forma clara con ejemplos prácticos y cuándo evitarlo según el libro."
        );
      }
      return;
    }

    if (action === "codigo") {
      const prompt = "Mostrame cómo implementar esto en código limpio y listo para producción.";
      if (currentInput) {
        els.chatInput.value = "";
        autoResizeTextarea();
        sendChat(`${currentInput}\n\n${prompt}`, "consulta");
      } else if (hasMessages) {
        sendChat(prompt, "consulta");
      } else {
        els.chatInput.value = prompt;
        autoResizeTextarea();
        els.chatInput.focus();
      }
      return;
    }

    if (action === "roadmap") {
      if (state.selectedBook) {
        setSelectedBook("", false);
      }
      els.chatInput.value = "";
      autoResizeTextarea();
      const prompt = currentInput
        ? `Armame un roadmap de lectura ordenado por fases con los libros de mi biblioteca enfocado en: ${currentInput}. Incluí un diagrama Mermaid.`
        : "Armame un roadmap y orden de lectura progresivo por fases usando todos los libros de mi biblioteca, explicando por cuál empezar e incluyendo un diagrama Mermaid.";
      sendChat(prompt, "consulta");
      return;
    }

    if (action === "comparar") {
      if (state.selectedBook) {
        setSelectedBook("", false);
        showToast("Alcance cambiado a Todos los libros para comparar autores.");
      }
      if (currentInput && !currentInput.includes("[tema]")) {
        els.chatInput.value = "";
        autoResizeTextarea();
        sendChat(
          `Compará qué dicen distintos libros de mi biblioteca sobre ${currentInput}.`,
          "consulta"
        );
      } else {
        fillTemplateAndSelectPlaceholder(
          "Compará qué dicen distintos libros de mi biblioteca sobre [tema]."
        );
      }
    }
  }

  async function triggerIngest() {
    if (state.ingesting) return;
    state.ingesting = true;
    els.ingestBtn.disabled = true;
    els.ingestIcon.classList.add("spin");
    els.ingestBtnText.textContent = "Indexando...";
    showToast("Escaneando libros/ e indexando archivos nuevos...");

    try {
      const resp = await fetch("/api/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reset: false }),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        throw new Error(data.detail || `Error HTTP ${resp.status}`);
      }
      showToast(data.message || "Biblioteca actualizada.");
      await loadBooks();
    } catch (err) {
      showToast(err.message || "No se pudo completar la re-indexación.");
    } finally {
      state.ingesting = false;
      els.ingestBtn.disabled = false;
      els.ingestIcon.classList.remove("spin");
      els.ingestBtnText.textContent = "Re-indexar biblioteca";
    }
  }

  /* ---------- Event Listeners ---------- */

  els.toggleSidebarBtn.addEventListener("click", toggleSidebar);
  els.closeSidebarBtn.addEventListener("click", closeMobileSidebar);
  els.sidebarBackdrop.addEventListener("click", closeMobileSidebar);
  els.ingestBtn.addEventListener("click", triggerIngest);

  els.newSessionBtn.addEventListener("click", async () => {
    try {
      await createNewSession({ title: "Nueva sesión", book: state.selectedBook });
      if (!isDesktop()) closeMobileSidebar();
      els.chatInput.focus();
    } catch (_) {
      showToast("No se pudo crear una nueva sesión.");
    }
  });

  els.bookSelect.addEventListener("change", () => {
    setSelectedBook(els.bookSelect.value, false);
  });

  els.bookFilterInput.addEventListener("input", () => {
    state.filterQuery = els.bookFilterInput.value;
    renderSidebarBookList();
  });

  document.querySelectorAll("[data-chip]").forEach((btn) => {
    btn.addEventListener("click", () => handleStudyModeAction(btn.dataset.chip));
  });

  document.querySelectorAll("[data-starter]").forEach((btn) => {
    btn.addEventListener("click", () => handleStudyModeAction(btn.dataset.starter));
  });

  els.chatInput.addEventListener("input", () => autoResizeTextarea(els.chatInput, 160));
  els.chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      els.sendBtn.click();
    }
  });
  els.sendBtn.addEventListener("click", () => {
    const text = els.chatInput.value.trim();
    if (!text) return;
    els.chatInput.value = "";
    autoResizeTextarea(els.chatInput, 160);
    sendChat(text, "consulta");
  });

  // Hilo paralelo listeners
  els.closeSideThreadBtn.addEventListener("click", closeSideThreadDrawer);
  els.sideThreadBackdrop.addEventListener("click", closeSideThreadDrawer);
  els.promoteSideThreadBtn.addEventListener("click", promoteSideThreadToNewChat);
  els.sideThreadInput.addEventListener("input", () => autoResizeTextarea(els.sideThreadInput, 112));
  els.sideThreadInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      els.sideThreadSendBtn.click();
    }
  });
  els.sideThreadSendBtn.addEventListener("click", () => {
    const text = els.sideThreadInput.value.trim();
    if (!text) return;
    els.sideThreadInput.value = "";
    autoResizeTextarea(els.sideThreadInput, 112);
    sendSideThreadMessage(text, "consulta");
  });

  // Menú contextual + Barra flotante de selección
  document.addEventListener("contextmenu", handleContextMenu);
  document.addEventListener("click", (e) => {
    if (!e.target.closest("#contextMenu")) {
      hideContextMenu();
    }
    if (!e.target.closest("#selectionToolbar")) {
      setTimeout(() => {
        const sel = window.getSelection();
        if (!sel || sel.isCollapsed) hideSelectionToolbar();
      }, 20);
    }
  });

  document.addEventListener("selectionchange", () => {
    clearTimeout(selectionTimer);
    selectionTimer = setTimeout(updateFloatingSelectionToolbar, 180);
  });

  els.selectionToolbar.querySelectorAll("[data-sel-action]").forEach((btn) => {
    btn.addEventListener("mousedown", (e) => e.preventDefault()); // Mantiene la selección activa al tocar
    btn.addEventListener("click", () => {
      runSelectionAction(btn.dataset.selAction, state.currentSelectionText);
    });
  });

  els.chatScroll.addEventListener("scroll", () => {
    hideContextMenu();
    hideSelectionToolbar();
  });

  // Visor PDF listeners
  els.pdfCloseBtn.addEventListener("click", closePdfModal);
  els.pdfModal.addEventListener("click", (e) => {
    if (e.target === els.pdfModal) closePdfModal();
  });
  els.pdfPrevBtn.addEventListener("click", () => {
    if (state.pdfViewer.page > 1) {
      loadPdfPageImage(state.pdfViewer.book, state.pdfViewer.page - 1);
    }
  });
  els.pdfNextBtn.addEventListener("click", () => {
    const next = state.pdfViewer.page + 1;
    if (!state.pdfViewer.totalPages || next <= state.pdfViewer.totalPages) {
      loadPdfPageImage(state.pdfViewer.book, next);
    }
  });
  els.pdfZoomBtn.addEventListener("click", () => {
    state.pdfViewer.zoomed = !state.pdfViewer.zoomed;
    updatePdfModalNav();
  });

  // Atajos de teclado globales
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      hideContextMenu();
      hideSelectionToolbar();
      if (state.dashboardOpen) {
        closeDashboardModal();
      } else if (state.pdfViewer.open) {
        closePdfModal();
      } else if (state.sideThread.open) {
        closeSideThreadDrawer();
      } else if (state.sidebarOpenMobile) {
        closeMobileSidebar();
      }
    } else if (state.pdfViewer.open && e.key === "ArrowLeft" && state.pdfViewer.page > 1) {
      loadPdfPageImage(state.pdfViewer.book, state.pdfViewer.page - 1);
    } else if (
      state.pdfViewer.open &&
      e.key === "ArrowRight" &&
      (!state.pdfViewer.totalPages || state.pdfViewer.page < state.pdfViewer.totalPages)
    ) {
      loadPdfPageImage(state.pdfViewer.book, state.pdfViewer.page + 1);
    }
  });

  /* ---------- Dashboard de Observabilidad en vivo & Configuración ---------- */

  let dashboardPollTimer = null;

  function renderMetricsSvgChart(traces) {
    const ordered = [...(traces || [])]
      .filter((t) => t.faithfulness !== null || t.context_precision !== null)
      .reverse();

    if (ordered.length === 0) {
      els.dashChartWrap.innerHTML = `<p class="text-xs text-muted py-8 text-center">Todavía no hay trazas evaluadas. Hacé una consulta para ver el gráfico moverse en tiempo real.</p>`;
      return;
    }

    const width = 460;
    const height = 195;
    const padL = 38;
    const padR = 16;
    const padT = 14;
    const padB = 26;
    const innerW = width - padL - padR;
    const innerH = height - padT - padB;

    const xAt = (idx) =>
      ordered.length === 1 ? padL + innerW / 2 : padL + (idx / (ordered.length - 1)) * innerW;
    const yAt = (val) => padT + innerH * (1 - Math.max(0, Math.min(1, Number(val))));

    const gridLines = [0, 0.5, 0.75, 1]
      .map((g) => {
        const y = yAt(g);
        return `
          <line x1="${padL}" y1="${y}" x2="${width - padR}" y2="${y}" stroke="#242936" stroke-dasharray="3 3" />
          <text x="${padL - 6}" y="${y + 3}" text-anchor="end" fill="#8b95a7" font-size="10" font-family="JetBrains Mono, monospace">${Math.round(g * 100)}%</text>
        `;
      })
      .join("");

    const buildSeries = (key, color) => {
      const pts = [];
      ordered.forEach((t, idx) => {
        if (t[key] !== null && t[key] !== undefined) {
          pts.push({ x: xAt(idx), y: yAt(t[key]), val: t[key], id: t.id });
        }
      });
      if (pts.length === 0) return "";
      const poly =
        pts.length > 1
          ? `<polyline fill="none" stroke="${color}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round" points="${pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ")}" />`
          : "";
      const dots = pts
        .map(
          (p) =>
            `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="3.8" fill="${color}" stroke="#0f1115" stroke-width="1.5"><title>#${p.id}: ${Math.round(p.val * 100)}%</title></circle>`
        )
        .join("");
      return poly + dots;
    };

    const xLabels = ordered
      .map(
        (t, idx) =>
          `<text x="${xAt(idx).toFixed(1)}" y="${height - 6}" text-anchor="middle" fill="#647087" font-size="10" font-family="JetBrains Mono, monospace">#${t.id}</text>`
      )
      .join("");

    els.dashChartWrap.innerHTML = `
      <svg viewBox="0 0 ${width} ${height}" class="w-full h-52" role="img" aria-label="Evolución de fidelidad y precisión">
        ${gridLines}
        ${buildSeries("faithfulness", "#F59E0B")}
        ${buildSeries("context_precision", "#818CF8")}
        ${xLabels}
      </svg>
    `;
  }

  function scoreBadgeClass(val, pending) {
    if (val === null || val === undefined) {
      return pending
        ? "bg-canvas border-hairline text-muted"
        : "bg-canvas border-hairline text-muted/70";
    }
    if (val >= 0.75) {
      return "bg-emerald-500/12 border-emerald-500/35 text-emerald-300";
    }
    if (val >= 0.5) {
      return "bg-amber-500/12 border-amber-500/35 text-amber-300";
    }
    return "bg-rose-500/15 border-rose-500/40 text-rose-300";
  }

  function renderTraceRows(traces) {
    els.dashTraceList.innerHTML = "";
    els.dashTraceCount.textContent = String((traces || []).length);

    if (!traces || traces.length === 0) {
      els.dashTraceList.innerHTML = `
        <tr>
          <td colspan="6" class="px-4 py-8 text-xs text-muted text-center">Sin consultas registradas todavía.</td>
        </tr>
      `;
      return;
    }

    traces.forEach((t) => {
      const isPending = t.eval_engine === "pending";
      const fPct = t.faithfulness !== null && t.faithfulness !== undefined ? `${Math.round(t.faithfulness * 100)}%` : isPending ? "…" : "—";
      const pPct = t.context_precision !== null && t.context_precision !== undefined ? `${Math.round(t.context_precision * 100)}%` : isPending ? "…" : "—";
      const sec = (Number(t.latencia_ms || 0) / 1000).toFixed(2);

      const flags = [];
      if (t.rebusqueda_activada) {
        flags.push(`<span class="px-1.5 py-0.5 rounded bg-amber-500/15 border border-amber-500/30 text-amber-300">Re-búsqueda</span>`);
      }
      if (t.conflictos_resueltos) {
        flags.push(`<span class="px-1.5 py-0.5 rounded bg-emerald-500/15 border border-emerald-500/30 text-emerald-300">Conflictos</span>`);
      }
      if (flags.length === 0) {
        flags.push(`<span class="px-1.5 py-0.5 rounded bg-canvas border border-hairline text-muted">${escapeHtml(t.estrategia)}</span>`);
      }

      const tr = document.createElement("tr");
      tr.className = "cursor-pointer transition-colors hover:bg-elevated/60";
      tr.innerHTML = `
        <td class="py-2.5 px-3 max-w-[240px]">
          <div class="flex items-center gap-1.5">
            <span class="font-mono text-[10px] text-muted shrink-0">#${t.id}</span>
            <span class="font-medium text-ink truncate" title="${escapeHtml(t.query)}">${escapeHtml(t.query)}</span>
          </div>
        </td>
        <td class="py-2.5 px-2.5 text-center font-mono text-[11px] text-ink/90 whitespace-nowrap">
          ${t.chunks_retenidos}/${t.chunks_iniciales}
        </td>
        <td class="py-2.5 px-2.5 font-mono text-[10.5px] whitespace-nowrap">
          <div class="flex flex-wrap gap-1">${flags.join("")}</div>
        </td>
        <td class="py-2.5 px-2.5 text-right font-mono text-[11px] text-muted whitespace-nowrap">
          ${sec}s
        </td>
        <td class="py-2.5 px-2.5 text-right font-mono text-[11px] whitespace-nowrap">
          <span class="inline-block px-2 py-0.5 rounded-md border ${scoreBadgeClass(t.faithfulness, isPending)}">${fPct}</span>
        </td>
        <td class="py-2.5 px-3 text-right font-mono text-[11px] whitespace-nowrap">
          <span class="inline-block px-2 py-0.5 rounded-md border ${scoreBadgeClass(t.context_precision, isPending)}">${pPct}</span>
        </td>
      `;

      const detailTr = document.createElement("tr");
      detailTr.className = "hidden bg-canvas/50";
      const retMs = Number(t.retrieval_ms || 0);
      const refMs = Number(t.reflection_ms || 0);
      const genMs = Number(t.generation_ms || 0);
      detailTr.innerHTML = `
        <td colspan="6" class="px-4 py-3 text-xs text-muted space-y-1.5">
          <div class="flex flex-wrap gap-3 font-mono text-[11px] text-ink/80 pb-1 border-b border-hairline/60">
            <span>Motor eval: <strong>${escapeHtml(t.eval_engine)}</strong></span>
            <span>Recuperación: <strong>${(retMs / 1000).toFixed(2)}s</strong></span>
            <span>Reflexión: <strong>${(refMs / 1000).toFixed(2)}s</strong></span>
            <span>Generación: <strong>${(genMs / 1000).toFixed(2)}s</strong></span>
          </div>
          <p><strong class="text-amber-300">Motivo Fidelidad:</strong> ${escapeHtml(t.faithfulness_reason || "Pendiente o sin detalle.")}</p>
          <p><strong class="text-indigo-300">Motivo Relevancia:</strong> ${escapeHtml(t.context_precision_reason || "Pendiente o sin detalle.")}</p>
          ${t.conflictos_resueltos ? `<p><strong class="text-emerald-300">Conflictos resueltos:</strong> ${escapeHtml(t.conflictos_resueltos)}</p>` : ""}
        </td>
      `;

      tr.addEventListener("click", () => detailTr.classList.toggle("hidden"));
      els.dashTraceList.appendChild(tr);
      els.dashTraceList.appendChild(detailTr);
    });
  }

  function renderLatencyBreakdown(data) {
    const ret = Number(data.avg_retrieval_ms || 0);
    const ref = Number(data.avg_reflection_ms || 0);
    const gen = Number(data.avg_generation_ms || 0);
    const sum = ret + ref + gen;

    els.latRetrievalVal.textContent = ret > 0 ? `${(ret / 1000).toFixed(2)}s` : "—";
    els.latReflectionVal.textContent = ref > 0 ? `${(ref / 1000).toFixed(2)}s` : "—";
    els.latGenerationVal.textContent = gen > 0 ? `${(gen / 1000).toFixed(2)}s` : "—";

    if (sum > 0) {
      els.latTotalLabel.textContent = `Total etapas: ${(sum / 1000).toFixed(2)}s`;
      els.latBarRetrieval.style.width = `${((ret / sum) * 100).toFixed(1)}%`;
      els.latBarReflection.style.width = `${((ref / sum) * 100).toFixed(1)}%`;
      els.latBarGeneration.style.width = `${((gen / sum) * 100).toFixed(1)}%`;
    } else {
      els.latTotalLabel.textContent = "Sin desglose previo";
      els.latBarRetrieval.style.width = "0%";
      els.latBarReflection.style.width = "0%";
      els.latBarGeneration.style.width = "0%";
    }
  }

  async function loadMetricsSummary(fullRender = false) {
    try {
      const resp = await fetch("/api/metrics?limit=20");
      if (!resp.ok) return;
      const data = await resp.json();

      if (data.avg_faithfulness !== null || data.avg_context_precision !== null) {
        const f = data.avg_faithfulness !== null ? `${Math.round(data.avg_faithfulness * 100)}%` : "—";
        const p = data.avg_context_precision !== null ? `${Math.round(data.avg_context_precision * 100)}%` : "—";
        els.metricsQuickScores.textContent = `F ${f} · P ${p}`;
      } else {
        els.metricsQuickScores.textContent = "";
      }

      if (!fullRender && !state.dashboardOpen) return;

      const fPct = data.avg_faithfulness !== null ? Math.round(data.avg_faithfulness * 100) : null;
      const pPct = data.avg_context_precision !== null ? Math.round(data.avg_context_precision * 100) : null;

      els.kpiFaithfulness.textContent = fPct !== null ? `${fPct}%` : "—";
      els.kpiFaithBar.style.width = `${fPct || 0}%`;
      els.kpiPrecision.textContent = pPct !== null ? `${pPct}%` : "—";
      els.kpiPrecBar.style.width = `${pPct || 0}%`;
      els.kpiLatency.textContent = data.avg_latency_ms ? `${(data.avg_latency_ms / 1000).toFixed(2)}s` : "—";
      els.kpiRuns.textContent = `${data.total_runs} consulta(s) registradas`;
      els.kpiDiscarded.textContent = `${data.total_discarded_chunks} desc.`;
      els.kpiRequeries.textContent = `${data.total_requeries} re-búsqueda(s) adaptativas`;
      els.dashLastUpdated.textContent = `· ${new Date().toLocaleTimeString("es-AR")}`;

      renderMetricsSvgChart(data.recent_traces || []);
      renderLatencyBreakdown(data);
      renderTraceRows(data.recent_traces || []);
    } catch (_) {}
  }

  async function loadRuntimeConfig() {
    try {
      const resp = await fetch("/api/config");
      if (!resp.ok) return;
      const cfg = await resp.json();
      els.cfgReflection.checked = Boolean(cfg.enable_reflection);
      els.cfgRequery.checked = Boolean(cfg.enable_requery);
      els.cfgEvaluation.checked = Boolean(cfg.enable_evaluation);

      els.cfgModelSelect.innerHTML = "";
      (cfg.available_models || []).forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = m;
        if (m === cfg.llm_model) opt.selected = true;
        els.cfgModelSelect.appendChild(opt);
      });

      els.modelLabel.textContent = `ONNX · ${cfg.llm_model}`;
      els.cfgSystemInfo.textContent =
        `Embeddings: ${cfg.embedding_provider} (Xenova/multilingual-e5-small) · Modelo activo: ${cfg.llm_model} · Libros: ${state.books.length}`;
    } catch (_) {}
  }

  async function patchRuntimeConfig(updates) {
    try {
      const resp = await fetch("/api/config", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updates),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const cfg = await resp.json();
      els.modelLabel.textContent = `ONNX · ${cfg.llm_model}`;
      els.cfgSystemInfo.textContent =
        `Embeddings: ${cfg.embedding_provider} (Xenova/multilingual-e5-small) · Modelo activo: ${cfg.llm_model} · Libros: ${state.books.length}`;
      showToast("Configuración guardada y aplicada en caliente.");
    } catch (_) {
      showToast("No se pudo actualizar la configuración.");
    }
  }

  function setDashboardTab(tab) {
    document.querySelectorAll("[data-dash-tab]").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.dashTab === tab);
    });
    els.dashTabMetrics.classList.toggle("hidden", tab !== "metrics");
    els.dashTabConfig.classList.toggle("hidden", tab !== "config");
  }

  function syncDashboardPolling() {
    clearInterval(dashboardPollTimer);
    if (state.dashboardOpen && els.dashAutoRefresh.checked) {
      els.livePulseDot.classList.add("animate-pulse");
      dashboardPollTimer = setInterval(() => loadMetricsSummary(true), 3000);
    } else {
      els.livePulseDot.classList.remove("animate-pulse");
    }
  }

  function openDashboardModal(tab = "metrics") {
    state.dashboardOpen = true;
    els.dashboardModal.classList.remove("hidden");
    els.dashboardModal.classList.add("flex");
    setDashboardTab(tab);
    loadMetricsSummary(true);
    loadRuntimeConfig();
    syncDashboardPolling();
  }

  function closeDashboardModal() {
    state.dashboardOpen = false;
    els.dashboardModal.classList.add("hidden");
    els.dashboardModal.classList.remove("flex");
    clearInterval(dashboardPollTimer);
  }

  els.openDashboardBtn.addEventListener("click", () => openDashboardModal("metrics"));
  els.headerDashboardBtn.addEventListener("click", () => openDashboardModal("metrics"));
  els.closeDashboardBtn.addEventListener("click", closeDashboardModal);
  els.dashboardModal.addEventListener("click", (e) => {
    if (e.target === els.dashboardModal) closeDashboardModal();
  });

  document.querySelectorAll("[data-dash-tab]").forEach((btn) => {
    btn.addEventListener("click", () => setDashboardTab(btn.dataset.dashTab));
  });

  els.dashAutoRefresh.addEventListener("change", syncDashboardPolling);
  els.dashRefreshBtn.addEventListener("click", () => loadMetricsSummary(true));
  els.dashExportBtn.addEventListener("click", async () => {
    try {
      const [metricsResp, configResp] = await Promise.all([
        fetch("/api/metrics?limit=100"),
        fetch("/api/config"),
      ]);
      if (!metricsResp.ok) throw new Error(`HTTP ${metricsResp.status}`);
      const metrics = await metricsResp.json();
      const config = configResp.ok ? await configResp.json() : null;

      const payload = {
        exported_at: new Date().toISOString(),
        config,
        summary: {
          total_runs: metrics.total_runs,
          avg_faithfulness: metrics.avg_faithfulness,
          avg_context_relevancy: metrics.avg_context_precision,
          avg_latency_ms: metrics.avg_latency_ms,
          avg_retrieval_ms: metrics.avg_retrieval_ms,
          avg_reflection_ms: metrics.avg_reflection_ms,
          avg_generation_ms: metrics.avg_generation_ms,
          total_discarded_chunks: metrics.total_discarded_chunks,
          total_requeries: metrics.total_requeries,
        },
        traces: metrics.recent_traces || [],
      };

      const jsonText = JSON.stringify(payload, null, 2);
      const blob = new Blob([jsonText], { type: "application/json;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
      a.href = url;
      a.download = `tutor-rag-metrics-${stamp}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      showToast("Reporte JSON exportado correctamente.");
    } catch (_) {
      showToast("No se pudo exportar el JSON de métricas.");
    }
  });
  els.dashClearBtn.addEventListener("click", async () => {
    try {
      await fetch("/api/metrics", { method: "DELETE" });
      showToast("Historial de métricas limpiado.");
      await loadMetricsSummary(true);
    } catch (_) {
      showToast("No se pudo limpiar el historial.");
    }
  });

  els.cfgReflection.addEventListener("change", () =>
    patchRuntimeConfig({ enable_reflection: els.cfgReflection.checked })
  );
  els.cfgRequery.addEventListener("change", () =>
    patchRuntimeConfig({ enable_requery: els.cfgRequery.checked })
  );
  els.cfgEvaluation.addEventListener("change", () =>
    patchRuntimeConfig({ enable_evaluation: els.cfgEvaluation.checked })
  );
  els.cfgModelSelect.addEventListener("change", () =>
    patchRuntimeConfig({ llm_model: els.cfgModelSelect.value })
  );

  /* ---------- Inicialización ---------- */

  loadBooks().then(() => {
    loadSessionsFromServer();
    loadMetricsSummary(false);
    loadRuntimeConfig();
  });
})();
