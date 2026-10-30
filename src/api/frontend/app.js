(() => {
  "use strict";

  const api = window.CouncilApi;
  const state = {
    issues: [],
    currentIssueKey: null,
    analysis: null,
    observations: [],
    selectedObservationIndex: null,
    observationMarkers: [],
    timelineIndex: 0,
    timelineTimer: null,
    map: null,
    markerLayer: null,
    routeLayer: null,
    chatHistory: [],
    conversationId: null,
    loadingTasks: new Set(),
    loadingSequence: 0,
  };

  const COUNCIL_COORDS = {
    "서울특별시의회": [37.5665, 126.9780], "부산광역시의회": [35.1796, 129.0756],
    "대구광역시의회": [35.8714, 128.6014], "인천광역시의회": [37.4563, 126.7052],
    "광주광역시의회": [35.1601, 126.8515], "대전광역시의회": [36.3504, 127.3845],
    "울산광역시의회": [35.5384, 129.3114], "세종특별자치시의회": [36.4800, 127.2890],
    "경기도의회": [37.2636, 127.0286], "강원특별자치도의회": [37.8854, 127.7298],
    "충청북도의회": [36.6357, 127.4913], "충청남도의회": [36.6588, 126.6728],
    "전북특별자치도의회": [35.8202, 127.1088], "전라남도의회": [34.8161, 126.4630],
    "경상북도의회": [36.5760, 128.5056], "경상남도의회": [35.2383, 128.6924],
    "제주특별자치도의회": [33.4890, 126.4983],
  };

  const $ = (id) => document.getElementById(id);
  const text = (id, value) => { const node = $(id); if (node) node.textContent = value ?? "—"; };
  const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[ch]);

  function showNotice(message, kind = "info") {
    const box = $("globalNotice");
    if (!box) return;
    box.textContent = message;
    box.className = "mx-4 lg:mx-6 mt-3 max-w-[1800px] w-[calc(100%-2rem)] lg:w-[calc(100%-3rem)] self-center rounded-xl border px-4 py-3 text-sm font-semibold";
    const styles = {
      info: "bg-sky-50 border-sky-200 text-sky-900",
      error: "bg-rose-50 border-rose-200 text-rose-900",
      success: "bg-emerald-50 border-emerald-200 text-emerald-900",
    };
    box.classList.add(...styles[kind].split(" "));
  }

  function clearNotice() {
    const box = $("globalNotice");
    if (box) box.classList.add("hidden");
  }

  function setApiStatus(label, ok) {
    const badge = $("apiStatus");
    if (!badge) return;
    badge.textContent = label;
    badge.className = `shrink-0 text-xs font-bold px-3 py-2 rounded-lg ${ok ? "bg-emerald-100 text-emerald-900" : "bg-rose-100 text-rose-900"}`;
  }

  const SPINNER_MARKUP = '<svg class="h-4 w-4 animate-spin" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="4" class="opacity-25"></circle><path fill="currentColor" d="M4 12a8 8 0 0 1 8-8V0C5.373 0 0 5.373 0 12h4z"></path></svg>';

  function updateLoadingOverlay() {
    const overlay = $("screenLoadingOverlay");
    if (!overlay) return;
    let activeScreenTask = null;
    for (const task of state.loadingTasks) {
      if (task.showScreen) activeScreenTask = task;
    }
    if (!activeScreenTask) {
      overlay.classList.add("hidden");
      overlay.setAttribute("aria-hidden", "true");
      return;
    }
    overlay.classList.remove("hidden");
    overlay.setAttribute("aria-hidden", "false");
    text("screenLoadingMessage", activeScreenTask.label);
  }

  function beginLoading(button, label, showScreen = true, showButtonSpinner = true) {
    const task = {
      id: String(++state.loadingSequence),
      button,
      label,
      showScreen,
      showButtonSpinner,
      finished: false,
    };
    state.loadingTasks.add(task);
    if (button) {
      button.dataset.loadingToken = task.id;
      button.dataset.loadingOriginalHtml = button.innerHTML;
      button.dataset.loadingOriginalDisabled = String(Boolean(button.disabled));
      button.dataset.loadingOriginalPosition = button.style.position || "";
      button.dataset.loadingOriginalZIndex = button.style.zIndex || "";
      if (showButtonSpinner) button.innerHTML = `${SPINNER_MARKUP}<span>${escapeHtml(label)}</span>`;
      button.disabled = true;
      button.setAttribute("aria-busy", "true");
      if (showScreen) {
        button.style.position = "relative";
        button.style.zIndex = "2001";
      }
    }
    updateLoadingOverlay();

    return () => {
      if (task.finished) return;
      task.finished = true;
      state.loadingTasks.delete(task);
      if (button && button.dataset.loadingToken === task.id) {
        if (task.showButtonSpinner) button.innerHTML = button.dataset.loadingOriginalHtml || "";
        button.disabled = button.dataset.loadingOriginalDisabled === "true";
        button.style.position = button.dataset.loadingOriginalPosition || "";
        button.style.zIndex = button.dataset.loadingOriginalZIndex || "";
        button.removeAttribute("aria-busy");
        delete button.dataset.loadingToken;
        delete button.dataset.loadingOriginalHtml;
        delete button.dataset.loadingOriginalDisabled;
        delete button.dataset.loadingOriginalPosition;
        delete button.dataset.loadingOriginalZIndex;
      }
      updateLoadingOverlay();
    };
  }

  function switchTab(tabId) {
    document.querySelectorAll(".tab-content").forEach((section) => section.classList.add("hidden"));
    document.querySelectorAll(".tab-btn").forEach((button) => button.classList.remove("active"));
    const panel = $(`tabContent-${tabId}`);
    const button = $(`tab-${tabId}`);
    if (panel) panel.classList.remove("hidden");
    if (button) button.classList.add("active");
    if (tabId === "radar" && state.map) setTimeout(() => state.map.invalidateSize(), 100);
  }

  function setFontScale(scale) {
    const sizes = { small: "16px", normal: "18px", large: "20px" };
    const selectedScale = Object.prototype.hasOwnProperty.call(sizes, scale) ? scale : "normal";
    document.documentElement.style.fontSize = sizes[selectedScale];
    document.querySelectorAll("[data-font-scale]").forEach((button) => {
      const selected = button.dataset.fontScale === selectedScale;
      button.classList.toggle("is-active", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
    try { localStorage.setItem("council_radar_font_scale", selectedScale); } catch (_) { /* private browsing */ }
  }

  function initMap() {
    const node = $("koreaLeafletMap");
    if (!node || !window.L || typeof window.maplibreGLLeaflet !== "function") {
      text("mapNotice", "지도 라이브러리를 불러오지 못했습니다. 관측 목록은 계속 사용할 수 있습니다.");
      return;
    }
    try {
      state.map = window.L.map(node, {
        scrollWheelZoom: true,
        maxBounds: [[180, -Infinity], [-180, Infinity]],
        maxBoundsViscosity: 1,
        minZoom: 1,
      }).setView([36.2, 127.8], 7);
      window.maplibreGLLeaflet({
        style: "https://tiles.openfreemap.org/styles/positron",
        attributionControl: false,
      }).addTo(state.map);
      state.map.attributionControl?.addAttribution(
        '<a href="https://openfreemap.org" target="_blank" rel="noopener">OpenFreeMap</a> © <a href="https://openmaptiles.org/" target="_blank" rel="noopener">OpenMapTiles</a> · Data from <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a>'
      );
      state.routeLayer = window.L.layerGroup().addTo(state.map);
      state.markerLayer = window.L.layerGroup().addTo(state.map);
      const notice = $("mapNotice");
      if (notice) notice.textContent = "이슈를 선택하면 실제 DB 관측 지점이 표시됩니다.";
    } catch (_) {
      text("mapNotice", "지도를 초기화하지 못했습니다. 관측 목록은 사용할 수 있습니다.");
      state.map = null;
    }
  }

  function normalizeObservations(rows) {
    return (rows || []).map((row, index) => ({
      order: index + 1,
      council: String(row.council || "의회 미상"),
      date: String(row.first_observation_date || ""),
      documentCount: Number(row.document_count || 0),
    }));
  }

  function dateDifference(first, current) {
    const a = Date.parse(`${first}T00:00:00Z`);
    const b = Date.parse(`${current}T00:00:00Z`);
    return Number.isFinite(a) && Number.isFinite(b) ? Math.max(0, Math.round((b - a) / 86400000)) : 0;
  }

  function renderCatalog() {
    const body = $("catalogTableBody");
    if (!body) return;
    body.replaceChildren();
    text("catalogCount", `${state.issues.length}개 이슈`);
    if (!state.issues.length) {
      body.innerHTML = '<tr><td colspan="4" class="p-5 text-center text-slate-500">등록된 고정 이슈가 없습니다.</td></tr>';
      return;
    }
    state.issues.forEach((issue, index) => {
      const row = document.createElement("tr");
      row.className = "hover:bg-sky-50/70 transition-colors";
      row.innerHTML = `
        <td class="p-3 font-black text-slate-950">${escapeHtml(issue.issue_name)}</td>
        <td class="p-3"><span class="px-2 py-1 rounded bg-slate-100 border border-slate-200 text-xs">${escapeHtml(issue.category || "기타")}</span></td>
        <td class="p-3 text-slate-700">${escapeHtml(issue.query)}</td>
        <td class="p-3 text-center"><button type="button" data-issue-index="${index}" class="px-3 py-1.5 rounded-lg bg-sky-700 hover:bg-sky-600 text-white text-xs font-black">레이더 보기</button></td>`;
      row.querySelector("button").addEventListener("click", (event) => selectAndSwitchToRadar(index, event.currentTarget));
      body.appendChild(row);
    });
  }

  async function loadIssueData(index, triggerButton = null) {
    const issue = state.issues[index];
    if (!issue || triggerButton?.disabled) return;
    state.currentIssueKey = issue.issue_key;
    text("radarState", "고정 이슈 · PostgreSQL 검색 중");
    text("radarTitle", issue.issue_name);
    text("radarQuery", `대표 검색어: ${issue.query}`);
    switchTab("radar");
    clearNotice();
    const finishLoading = beginLoading(triggerButton, "분석 중…", true, false);
    try {
      const analysis = await api.getIssue(issue.issue_key);
      renderAnalysis(analysis, issue);
    } catch (error) {
      text("radarState", "분석 요청 실패");
      showNotice(`이슈 분석에 실패했습니다: ${error.message}`, "error");
    } finally {
      finishLoading();
    }
  }

  function selectAndSwitchToRadar(index, triggerButton = null) {
    return loadIssueData(Number(index), triggerButton);
  }

  function renderAnalysis(data, definition = null) {
    state.analysis = data || {};
    state.currentIssueKey = data?.issue?.issue_key || definition?.issue_key || null;
    const issue = data?.issue || {};
    const query = data?.query || definition?.query || "";
    const summary = data?.summary || {};
    const forecast = data?.forecast || {};
    const monthly = data?.monthly_counts || [];
    state.observations = normalizeObservations(data?.propagation || []);
    state.timelineIndex = Math.max(0, state.observations.length - 1);
    state.selectedObservationIndex = null;

    text("radarState", data?.match_type === "dynamic" ? "자유 검색 · 검토 전 임시 분석" : (issue.status === "fixed" ? "고정 이슈 · PostgreSQL 분석" : "PostgreSQL 분석"));
    text("radarTitle", issue.issue_name || query || "이슈 분석");
    text("radarQuery", query ? `검색어: ${query}` : "");
    text("kpiCouncilCount", `${Number(summary.council_count || 0)}곳`);
    const docTotal = summary.evidence_unit_count;
    text("kpiEvidenceCount", docTotal == null ? "—" : `${Number(docTotal).toLocaleString()}건`);
    text("kpiFirstObservation", summary.first_observation_date || "기록 없음");
    const predicted = forecast.predicted_document_count;
    text("kpiForecastValue", predicted == null ? "기록 없음" : `${Number(predicted).toLocaleString()}건`);
    text("forecastHeadline", forecast.target_month ? `${forecast.target_month} 매칭 문서 기준선` : "기준선 계산 불가");
    text("forecastDescription", forecast.recent_3_month_mean == null
      ? "최근 3개월 집계가 부족해 기준선을 계산할 수 없습니다."
      : `최근 3개월 매칭 문서 수 평균은 ${forecast.recent_3_month_mean}건입니다. 다음 달의 단순 기준선이며 확률 예측은 아닙니다.`);
    text("topForecastBriefCard", forecast.recent_3_month_mean == null
      ? "최근 3개월 집계가 부족해 기준선을 계산할 수 없습니다."
      : `다음 달 기준선: ${predicted ?? "—"}건 · 최근 3개월 평균: ${forecast.recent_3_month_mean}건`);
    text("forecastMethod", `방법: ${forecast.method || "—"} · 단위: ${forecast.count_unit || "—"} · 자료: ${forecast.basis || "—"}`);
    text("radarEvidenceCount", `${(data?.evidence || []).length}건`);
    text("monthlyCountCaption", `${monthly.length}개월 · 월별 distinct 문서 수`);

    renderMonthly(monthly);
    renderObservations();
    renderEvidence(data?.evidence || []);
    renderTimelineStep(state.timelineIndex);
  }

  function renderMonthly(rows) {
    const container = $("monthlyChartBars");
    if (!container) return;
    container.replaceChildren();
    if (!rows.length) {
      const empty = document.createElement("p");
      empty.className = "p-4 text-sm text-slate-500";
      empty.textContent = "월별 매칭 이력이 없습니다.";
      container.appendChild(empty);
      return;
    }
    const values = rows.map((row) => ({
      month: String(row.month || "").slice(0, 7),
      count: Math.max(0, Number(row.document_count || 0)),
    })).sort((a, b) => a.month.localeCompare(b.month));
    const width = Math.max(640, values.length * 84 + 100);
    const height = 248;
    const padding = { left: 64, right: 28, top: 34, bottom: 52 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const maxCount = Math.max(0, ...values.map((row) => row.count));
    const rawStep = maxCount > 0 ? maxCount / 3.5 : 1;
    const magnitude = 10 ** Math.floor(Math.log10(rawStep));
    const normalizedStep = rawStep / magnitude;
    const stepFactor = normalizedStep <= 1 ? 1 : (normalizedStep <= 2 ? 2 : (normalizedStep <= 5 ? 5 : 10));
    const tickStep = stepFactor * magnitude;
    const scaleMax = tickStep * 4;
    const pointX = (index) => values.length === 1
      ? padding.left + plotWidth / 2
      : padding.left + (index / (values.length - 1)) * plotWidth;
    const pointY = (count) => padding.top + plotHeight - (count / scaleMax) * plotHeight;
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "monthly-line-chart");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("width", String(width));
    svg.setAttribute("height", String(height));
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "월별 매칭 문서 수 선형 그래프, 단위 건");
    const add = (tag, attributes, label = null) => {
      const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
      if (label !== null) node.textContent = label;
      svg.appendChild(node);
      return node;
    };

    for (let tick = 0; tick <= 4; tick += 1) {
      const value = tickStep * tick;
      const y = padding.top + plotHeight - (value / scaleMax) * plotHeight;
      add("line", { x1: padding.left, y1: y, x2: width - padding.right, y2: y, stroke: "#cbd5e1", "stroke-width": 1 });
      add("text", { x: padding.left - 10, y: y + 4, fill: "#475569", "font-size": 12, "text-anchor": "end" }, value.toLocaleString("ko-KR"));
    }

    const points = values.map((row, index) => `${pointX(index)},${pointY(row.count)}`).join(" ");
    add("polyline", { points, fill: "none", stroke: "#0284c7", "stroke-width": 3, "stroke-linecap": "round", "stroke-linejoin": "round" });
    values.forEach((row, index) => {
      const x = pointX(index);
      const y = pointY(row.count);
      add("circle", { cx: x, cy: y, r: 5, fill: "#ffffff", stroke: "#0369a1", "stroke-width": 3, "data-month": row.month, "data-count": row.count });
      add("text", { x, y: Math.max(18, y - 12), fill: "#0f172a", "font-size": 13, "font-weight": 700, "text-anchor": "middle", "data-count-label": row.count }, `${row.count.toLocaleString("ko-KR")}건`);
      add("text", { x, y: height - 14, fill: "#334155", "font-size": 11, "font-weight": 600, "text-anchor": "middle", "data-month-label": row.month }, row.month);
    });
    container.appendChild(svg);
  }

  function evidenceForCouncil(council) {
    return (state.analysis?.evidence || []).find((item) => item.metadata?.council === council);
  }

  function renderObservations() {
    const container = $("councilFlowList");
    if (!container) return;
    container.replaceChildren();
    if (!state.observations.length) {
      container.textContent = "매칭된 의회별 날짜 자료가 없습니다.";
      return;
    }
    state.observations.forEach((row, index) => {
      const card = document.createElement("article");
      card.className = "observation-card rounded-xl border border-slate-200 bg-white p-3 hover:border-sky-400 cursor-pointer";
      card.dataset.observationIndex = String(index);
      card.dataset.council = row.council;
      card.setAttribute("aria-selected", "false");
      const evidence = evidenceForCouncil(row.council);
      card.innerHTML = `
        <div class="flex justify-between gap-3 items-start"><div><div class="font-black text-slate-950">${index + 1}. ${escapeHtml(row.council)}</div><div class="text-xs text-slate-600 mt-1">가장 이른 매칭 날짜: ${escapeHtml(row.date || "날짜 없음")}</div></div><span class="shrink-0 text-xs font-bold rounded bg-sky-50 text-sky-900 px-2 py-1">${row.documentCount.toLocaleString()}개 문서</span></div>
        <div class="mt-2 text-xs text-slate-700">${evidence ? escapeHtml((evidence.text || "").slice(0, 240)) : "상위 검색 근거 목록에 이 의회의 발언이 포함되지 않았습니다."}</div>`;
      card.addEventListener("click", () => renderTimelineStep(index));
      if (evidence?.chunk_id) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "mt-2 text-xs font-bold text-sky-800 underline";
        button.dataset.evidenceChunkId = evidence.chunk_id;
        button.setAttribute("aria-expanded", "false");
        button.textContent = "근거 청크 조회";
        button.addEventListener("click", (event) => {
          event.stopPropagation();
          loadEvidenceInto(button, evidence.chunk_id);
        });
        card.appendChild(button);
      }
      container.appendChild(card);
    });
  }

  function renderEvidence(rows, target = "radarEvidenceList") {
    const container = $(target);
    if (!container) return;
    container.replaceChildren();
    if (!rows.length) {
      container.textContent = "검색 근거가 없습니다.";
      return;
    }
    rows.forEach((item, index) => {
      const meta = item.metadata || {};
      const card = document.createElement("article");
      card.className = "rounded-xl border border-slate-200 bg-white p-4 space-y-2";
      const score = Number(item.retrieval_score);
      const scoreLabel = Number.isFinite(score) ? score.toFixed(4) : "—";
      const sources = (item.retrieval_sources || []).join(" + ") || "하이브리드";
      card.innerHTML = `<div class="flex flex-wrap items-center justify-between gap-2 text-xs"><strong class="text-slate-950">${index + 1}. ${escapeHtml(meta.council || "의회 미상")}</strong><span>${escapeHtml(meta.meeting_date || "날짜 미상")}</span><span class="rounded bg-purple-50 px-2 py-1 text-purple-900">통합 점수 ${scoreLabel}</span><span class="text-slate-500">${escapeHtml(sources)}</span></div><div class="text-sm font-bold text-slate-800">${escapeHtml(meta.meeting_name || meta.agenda_title || "회의록")}</div><p class="whitespace-pre-wrap text-sm leading-relaxed text-slate-900">${escapeHtml(item.text || "")}</p>`;
      const actions = document.createElement("div");
      actions.className = "flex gap-3";
      if (item.chunk_id) {
        const evidenceButton = document.createElement("button");
        evidenceButton.type = "button";
        evidenceButton.className = "text-xs font-bold text-sky-800 underline";
        evidenceButton.textContent = "근거 상세";
        evidenceButton.addEventListener("click", () => loadEvidenceInto(evidenceButton, item.chunk_id));
        actions.appendChild(evidenceButton);
      }
      if (item.document_id) {
        const documentButton = document.createElement("button");
        documentButton.type = "button";
        documentButton.className = "text-xs font-bold text-purple-800 underline";
        documentButton.textContent = "같은 회의록의 청크 보기";
        documentButton.addEventListener("click", () => loadDocumentInto(documentButton, item.document_id));
        actions.appendChild(documentButton);
      }
      card.appendChild(actions);
      container.appendChild(card);
    });
  }

  async function loadEvidenceInto(button, chunkId) {
    if (!button || button.dataset.evidenceLoading === "true") return null;
    if (button.dataset.evidenceOpened === "true") return null;
    const panel = document.createElement("pre");
    panel.className = "mt-2 whitespace-pre-wrap rounded-lg bg-slate-50 border border-slate-200 p-3 text-xs text-slate-800";
    panel.setAttribute("aria-live", "polite");
    panel.textContent = "근거를 불러오는 중…";
    button.insertAdjacentElement("afterend", panel);
    button.disabled = true;
    button.dataset.evidenceLoading = "true";
    button.dataset.evidenceOpened = "true";
    button.setAttribute("aria-expanded", "true");
    try {
      const result = await api.getEvidence(chunkId);
      const meta = result.metadata || {};
      panel.textContent = `${meta.council || "의회 미상"} · ${meta.meeting_date || "날짜 미상"}\n${meta.meeting_name || ""}\n\n${result.text || ""}`;
    } catch (error) {
      panel.textContent = `근거 조회 실패: ${error.message}`;
      delete button.dataset.evidenceOpened;
      button.setAttribute("aria-expanded", "false");
    } finally {
      button.disabled = false;
      delete button.dataset.evidenceLoading;
    }
    return panel;
  }

  async function loadDocumentInto(button, documentId) {
    const panel = document.createElement("div");
    panel.className = "mt-2 space-y-2 rounded-lg bg-slate-50 border border-slate-200 p-3";
    panel.textContent = "회의록 청크를 불러오는 중…";
    button.insertAdjacentElement("afterend", panel);
    button.disabled = true;
    try {
      const result = await api.getDocument(documentId, 100);
      panel.replaceChildren();
      (result.items || []).forEach((item) => {
        const block = document.createElement("p");
        const meta = item.metadata || {};
        block.className = "whitespace-pre-wrap text-xs text-slate-800 border-b border-slate-200 pb-2";
        block.textContent = `${meta.meeting_date || ""} · ${meta.committee || ""}\n${item.text || ""}`;
        panel.appendChild(block);
      });
      if (!result.items?.length) panel.textContent = "이 회의록의 청크가 없습니다.";
    } catch (error) {
      panel.textContent = `회의록 조회 실패: ${error.message}`;
    } finally {
      button.disabled = false;
    }
  }

  function renderTimelineStep(index) {
    if (!state.observations.length) return;
    state.timelineIndex = Math.max(0, Math.min(Number(index) || 0, state.observations.length - 1));
    const slider = $("timelineSlider");
    if (slider) {
      slider.max = String(state.observations.length - 1);
      slider.value = String(state.timelineIndex);
    }
    const row = state.observations[state.timelineIndex];
    text("mapCurrentStepDate", `${row.date || "날짜 없음"} · ${row.council}`);
    text("playerStateText", `${state.timelineIndex + 1}/${state.observations.length} 의회 관측`);
    text("totalStepCount", String(state.observations.length));
    const visible = state.observations.slice(0, state.timelineIndex + 1);
    if (state.markerLayer) state.markerLayer.clearLayers();
    if (state.routeLayer) state.routeLayer.clearLayers();
    state.observationMarkers = [];
    const points = [];
    visible.forEach((item, index) => {
      const coordinates = COUNCIL_COORDS[item.council];
      if (!coordinates || !state.markerLayer) return;
      points.push(coordinates);
      const selected = index === state.selectedObservationIndex;
      const marker = window.L.circleMarker(coordinates, {
        radius: selected ? 11 : (item.order === 1 ? 11 : 8),
        color: "#ffffff",
        weight: selected ? 4 : 2,
        fillColor: selected ? "#0284c7" : (item.order === 1 ? "#f59e0b" : "#e95b4c"),
        fillOpacity: 0.95,
      }).addTo(state.markerLayer);
      state.observationMarkers[index] = marker;
      marker.bindTooltip(`${item.order}. ${escapeHtml(item.council)}<br>${escapeHtml(item.date)} · ${item.documentCount}개 문서`);
      marker.on("click", () => focusObservation(index));
    });
    if (state.routeLayer && points.length > 1) {
      window.L.polyline(points, { color: "#ee5e4c", weight: 3, opacity: 0.75, dashArray: "7 6" }).addTo(state.routeLayer);
    }
    const notice = $("mapNotice");
    if (notice) {
      notice.textContent = state.observations.length ? `관측 의회 ${state.observations.length}곳 · 날짜순 표시` : "관측 자료가 없습니다.";
      notice.classList.toggle("hidden", Boolean(state.observations.length));
    }
    document.querySelectorAll("[data-observation-index]").forEach((card) => {
      card.classList.toggle("ring-2", Number(card.dataset.observationIndex) === state.timelineIndex);
      card.classList.toggle("ring-sky-500", Number(card.dataset.observationIndex) === state.timelineIndex);
    });
  }

  async function focusObservation(index) {
    const selectedIndex = Number(index);
    const row = state.observations[selectedIndex];
    if (!row) return;
    state.selectedObservationIndex = selectedIndex;
    state.observationMarkers.forEach((marker, markerIndex) => {
      if (!marker) return;
      const selected = markerIndex === selectedIndex;
      const observation = state.observations[markerIndex];
      marker.setStyle?.({
        radius: selected ? 11 : (observation.order === 1 ? 11 : 8),
        weight: selected ? 4 : 2,
        fillColor: selected ? "#0284c7" : (observation.order === 1 ? "#f59e0b" : "#e95b4c"),
      });
    });
    document.querySelectorAll("[data-observation-index]").forEach((card) => {
      const selected = Number(card.dataset.observationIndex) === selectedIndex;
      card.classList.toggle("is-selected", selected);
      card.setAttribute("aria-selected", String(selected));
      if (selected) card.scrollIntoView?.({ behavior: "smooth", block: "center" });
    });
    const card = [...document.querySelectorAll("[data-observation-index]")]
      .find((item) => Number(item.dataset.observationIndex) === selectedIndex);
    const evidenceButton = card?.querySelector("[data-evidence-chunk-id]");
    if (evidenceButton && evidenceButton.dataset.evidenceChunkId) {
      const panel = await loadEvidenceInto(evidenceButton, evidenceButton.dataset.evidenceChunkId);
      panel?.scrollIntoView?.({ behavior: "smooth", block: "nearest" });
      if (panel) card.dataset.evidenceExpanded = "true";
    }
  }

  function stopTimeline() {
    if (state.timelineTimer) clearInterval(state.timelineTimer);
    state.timelineTimer = null;
    const icon = $("playToggleIcon");
    if (icon) icon.className = "fa-solid fa-play";
  }

  function toggleTimelinePlay() {
    if (state.timelineTimer) { stopTimeline(); return; }
    if (state.observations.length < 2) return;
    const icon = $("playToggleIcon");
    if (icon) icon.className = "fa-solid fa-pause";
    state.timelineTimer = setInterval(() => {
      if (state.timelineIndex >= state.observations.length - 1) { stopTimeline(); return; }
      renderTimelineStep(state.timelineIndex + 1);
    }, 1300);
  }

  function stepTimeline(delta) { stopTimeline(); renderTimelineStep(state.timelineIndex + delta); }
  function resetTimelineToStart() { stopTimeline(); renderTimelineStep(0); }
  function onTimelineSliderChange(value) { stopTimeline(); renderTimelineStep(value); }
  function resetMapView() {
    if (state.map) state.map.flyTo([36.2, 127.8], 7, { duration: 0.5 });
    if (state.observations.length) renderTimelineStep(state.observations.length - 1);
  }

  async function handleGlobalSearch() {
    const input = $("globalSearchInput");
    const query = input?.value.trim();
    if (!query) { showNotice("검색어를 입력하세요.", "info"); return; }
    const button = $("globalSearchButton");
    if (button?.disabled) return;
    clearNotice();
    switchTab("radar");
    setApiStatus("레이더 검색 중", true);
    const finishLoading = beginLoading(button, "분석 중…", true, false);
    try {
      const result = await api.searchRadar(query);
      const definition = state.issues.find((item) => item.issue_key === result?.issue?.issue_key) || null;
      renderAnalysis(result, definition);
      setApiStatus("API 연결됨", true);
    } catch (error) {
      setApiStatus("API 오류", false);
      showNotice(`레이더 검색에 실패했습니다: ${error.message}`, "error");
    } finally {
      finishLoading();
    }
  }

  async function executeRawVectorSearch(event) {
    if (event) event.preventDefault();
    const query = $("rawSearchQuery")?.value.trim() || "";
    if (!query) { text("documentSearchStatus", "검색어를 입력하세요."); return; }
    const dateFrom = $("rawSearchDateFrom")?.value || "";
    const dateTo = $("rawSearchDateTo")?.value || "";
    if (dateFrom && dateTo && dateFrom > dateTo) {
      text("documentSearchStatus", "시작일은 종료일보다 늦을 수 없습니다.");
      return;
    }
    const filters = {
      q: query,
      council: $("rawSearchCouncilFilter")?.value.trim() || "",
      year: $("rawSearchYearFilter")?.value || "",
      committee: $("rawSearchCommitteeFilter")?.value.trim() || "",
      date_from: dateFrom,
      date_to: dateTo,
      top_k: Number($("rawSearchTopK")?.value || 20),
    };
    const button = $("documentSearchButton");
    if (button?.disabled) return;
    const finishLoading = beginLoading(button, "검색 중…");
    text("documentSearchStatus", "PostgreSQL 검색 중…");
    const results = $("rawSearchResultsContainer");
    if (results) results.textContent = "검색 중…";
    try {
      const response = await api.searchDocuments(filters);
      text("rawSearchResultCount", String(response.result_count ?? response.results?.length ?? 0));
      text("rawSearchDbMs", response.search_ms == null ? "—" : Number(response.search_ms).toFixed(1));
      text("documentSearchStatus", "검색 완료");
      renderEvidence(response.results || [], "rawSearchResultsContainer");
    } catch (error) {
      text("rawSearchResultCount", "0");
      text("rawSearchDbMs", "—");
      text("documentSearchStatus", "검색 실패");
      if (results) results.textContent = `검색에 실패했습니다: ${error.message}`;
    } finally {
      finishLoading();
    }
  }

  function toggleFloatingChat() {
    const modal = $("floatingChatModal");
    if (!modal) return;
    modal.classList.toggle("hidden");
    modal.classList.toggle("flex");
    if (!modal.classList.contains("hidden")) setTimeout(() => $("agentInputQuery")?.focus(), 80);
  }

  function appendChatMessage(role, message, citations = []) {
    const container = $("agentChatContainer");
    if (!container) return;
    const row = document.createElement("div");
    row.className = role === "user" ? "flex justify-end" : "flex justify-start";
    const bubble = document.createElement("div");
    bubble.className = role === "user"
      ? "max-w-[88%] rounded-2xl bg-sky-700 text-white p-3 whitespace-pre-wrap"
      : "max-w-[92%] rounded-2xl bg-white border border-slate-200 text-slate-900 p-3 whitespace-pre-wrap";
    bubble.textContent = message;
    row.appendChild(bubble);
    container.appendChild(row);
    if (citations.length) {
      const sources = document.createElement("div");
      sources.className = "ml-2 space-y-1 text-xs text-slate-700";
      citations.forEach((citation) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "block text-left underline text-sky-800";
        button.textContent = `[근거 ${citation.citation}] ${citation.council || "의회 미상"} · ${citation.meeting_date || "날짜 미상"}`;
        button.addEventListener("click", () => {
          if (citation.chunk_id) loadEvidenceInto(button, citation.chunk_id);
        });
        sources.appendChild(button);
      });
      container.appendChild(sources);
    }
    container.scrollTop = container.scrollHeight;
  }

  async function sendChatMessage(query) {
    const message = String(query || "").trim();
    if (!message) return;
    const button = $("chatSendButton");
    if (button?.disabled) return;
    appendChatMessage("user", message);
    const finishLoading = beginLoading(button, "답변 생성 중…", false, false);
    try {
      const history = state.chatHistory.slice(-8).map((item) => ({
        role: item.role,
        content: String(item.content || "").slice(-3000),
      }));
      const response = await api.chat({
        message,
        conversation_id: state.conversationId,
        history,
        top_k: 6,
      });
      state.conversationId = response.conversation_id || state.conversationId;
      state.chatHistory.push({ role: "user", content: message.slice(-3000) });
      state.chatHistory.push({ role: "assistant", content: String(response.answer || "").slice(-3000) });
      state.chatHistory = state.chatHistory.slice(-8);
      appendChatMessage("assistant", response.answer || "답변이 비어 있습니다.", response.citations || []);
    } catch (error) {
      appendChatMessage("assistant", `답변 생성에 실패했습니다: ${error.message}`);
    } finally {
      finishLoading();
    }
  }

  function handleAgentSubmit(event) {
    if (event) event.preventDefault();
    const input = $("agentInputQuery");
    const query = input?.value.trim();
    if (!query) return;
    input.value = "";
    return sendChatMessage(query);
  }

  function askAgentPredefined(query) {
    const modal = $("floatingChatModal");
    if (modal?.classList.contains("hidden")) toggleFloatingChat();
    sendChatMessage(query);
  }

  async function refreshChatStatus() {
    try {
      const status = await api.chatStatus();
      const label = status.model_available ? `Ollama 사용 가능 · ${status.model}` : (status.reachable ? "Ollama 실행 중 · 모델 확인 필요" : "Ollama 연결 안 됨");
      text("ollamaStatus", label);
      text("chatStatus", label);
    } catch (_) {
      text("ollamaStatus", "Ollama 상태 확인 실패");
      text("chatStatus", "Ollama 상태 확인 실패");
    }
  }

  async function initialize() {
    try {
      const saved = localStorage.getItem("council_radar_font_scale") || "normal";
      setFontScale(saved);
    } catch (_) { setFontScale("normal"); }
    initMap();
    const globalSearch = $("globalSearchInput");
    if (globalSearch) {
      globalSearch.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.isComposing) {
          event.preventDefault();
          return handleGlobalSearch();
        }
      });
    }
    const chatInput = $("agentInputQuery");
    if (chatInput) {
      chatInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.isComposing) {
          event.preventDefault();
          return handleAgentSubmit(event);
        }
      });
    }
    if (!api) {
      setApiStatus("API 클라이언트 없음", false);
      showNotice("API 클라이언트 스크립트를 불러오지 못했습니다.", "error");
      return;
    }
    try {
      const payload = await api.listIssues();
      state.issues = Array.isArray(payload.items) ? payload.items : [];
      renderCatalog();
      setApiStatus("API 연결됨", true);
    } catch (error) {
      setApiStatus("API 연결 실패", false);
      showNotice(`API에서 키워드 목록을 불러오지 못했습니다: ${error.message}`, "error");
      const body = $("catalogTableBody");
      if (body) body.textContent = "API 서버가 실행 중인지 확인하세요.";
    }
    refreshChatStatus();
  }

  Object.assign(window, {
    switchTab,
    setFontScale,
    handleGlobalSearch,
    loadIssueData,
    selectAndSwitchToRadar,
    renderTimelineStep,
    toggleTimelinePlay,
    stepTimeline,
    resetTimelineToStart,
    onTimelineSliderChange,
    resetMapView,
    executeRawVectorSearch,
    toggleFloatingChat,
    handleAgentSubmit,
    askAgentPredefined,
    escapeHtml,
  });

  window.addEventListener("DOMContentLoaded", initialize, { once: true });
})();
