(function attachCouncilApi(global) {
  const API_BASE = "/api/v1";

  async function requestJSON(path, options = {}) {
    const headers = { Accept: "application/json", ...(options.headers || {}) };
    const response = await fetch(`${API_BASE}${path}`, { ...options, headers });
    let payload = null;
    try {
      payload = await response.json();
    } catch (_) {
      payload = null;
    }
    if (!response.ok) {
      const detail = payload && payload.detail;
      const message = typeof detail === "string" ? detail : `API 요청 실패 (${response.status})`;
      throw new Error(message);
    }
    return payload;
  }

  function queryString(values) {
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(values || {})) {
      if (value !== undefined && value !== null && value !== "") params.set(key, String(value));
    }
    const encoded = params.toString();
    return encoded ? `?${encoded}` : "";
  }

  global.CouncilApi = Object.freeze({
    API_BASE,
    listIssues: () => requestJSON("/issues"),
    getIssue: (issueKey) => requestJSON(`/issues/${encodeURIComponent(issueKey)}`),
    searchRadar: (query) => requestJSON("/radar/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    }),
    searchDocuments: (filters) => requestJSON(`/documents/search${queryString(filters)}`),
    getEvidence: (chunkId) => requestJSON(`/evidence/${encodeURIComponent(chunkId)}`),
    getDocument: (documentId, limit = 100) => requestJSON(
      `/documents/${encodeURIComponent(documentId)}${queryString({ limit })}`
    ),
    chatStatus: () => requestJSON("/chat/status"),
    chat: (body) => requestJSON("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  });
})(window);
