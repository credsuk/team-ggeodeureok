from pathlib import Path
import re

path = Path("frontend/app.js")
js = path.read_text(encoding="utf-8-sig")

raw_search = r'''
async function executeRawVectorSearch() {
    const query = document.getElementById("rawSearchQuery")?.value?.trim() || "";
    const council = document.getElementById("rawSearchCouncilFilter")?.value || "";
    const yearValue = document.getElementById("rawSearchYearFilter")?.value || "all";
    const container = document.getElementById("rawSearchResultsContainer");
    const countEl = document.getElementById("rawSearchResultCount");
    const timeEl = document.getElementById("rawSearchDbMs");

    if (!container || !query) return;

    container.innerHTML = '<p class="p-4 text-slate-700">회의록 검색 중입니다...</p>';
    if (countEl) countEl.textContent = "0";
    if (timeEl) timeEl.textContent = "...";

    try {
        const payload = { query, limit: 20 };
        if (council) payload.council = council;
        if (yearValue !== "all") payload.year = Number(yearValue);

        const response = await fetch("/api/search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || `HTTP ${response.status}`);
        }

        const results = data.results || [];
        container.innerHTML = "";
        if (countEl) countEl.textContent = results.length;
        if (timeEl) timeEl.textContent = data.elapsed_ms ?? "-";

        if (!results.length) {
            container.innerHTML = '<p class="p-4 text-slate-700">검색 결과가 없습니다.</p>';
            return;
        }

        results.forEach((r, index) => {
            const card = document.createElement("div");
            card.className = "p-4 rounded-xl border border-slate-300 bg-white space-y-2.5 shadow-sm";
            const pct = Math.round((Number(r.similarity) || 0) * 100);
            const body = r.text || r.preview || "";
            const parent = r.parent || body;

            card.innerHTML = `
                <div class="flex items-center justify-between flex-wrap gap-2 border-b border-slate-200 pb-2">
                    <div class="flex items-center gap-2">
                        <span class="w-6 h-6 rounded bg-purple-100 text-purple-900 font-mono font-bold flex items-center justify-center text-xs">${index + 1}</span>
                        <span class="font-black text-slate-950">${escapeHtml(r.assembly || "")}</span>
                        <span class="text-slate-600 font-mono text-xs">${escapeHtml(r.date || "")}</span>
                    </div>
                    <span class="font-mono text-xs font-bold px-2 py-1 rounded bg-purple-100 text-purple-900">${pct}% 유사도</span>
                </div>
                <div class="text-sm text-slate-800 font-bold">${escapeHtml(r.meetingName || r.agenda || "회의록")}</div>
                <div class="text-sm text-slate-900 leading-relaxed bg-slate-50 p-3 rounded-lg border border-slate-200 whitespace-pre-wrap">${escapeHtml(body)}</div>
                <div class="text-right">
                    <button type="button" class="text-xs text-purple-700 font-bold">본문 더 보기</button>
                    <div class="hidden mt-2 p-3 bg-slate-50 text-sm text-slate-800 border rounded-lg text-left whitespace-pre-wrap">${escapeHtml(parent)}</div>
                </div>`;

            const button = card.querySelector("button");
            const detail = button.nextElementSibling;
            button.addEventListener("click", () => {
                detail.classList.toggle("hidden");
                button.textContent = detail.classList.contains("hidden") ? "본문 더 보기" : "본문 접기";
            });
            container.appendChild(card);
        });
    } catch (error) {
        console.error("회의록 검색 실패:", error);
        container.innerHTML = `<p class="p-4 text-red-700">검색 실패: ${escapeHtml(error.message || "서버 오류")}</p>`;
        if (timeEl) timeEl.textContent = "-";
    }
}
'''

agent_response = r'''
async function respondAgentQuery(query) {
    try {
        const response = await fetch("/api/search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query, limit: 5 })
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || `HTTP ${response.status}`);
        }

        const answer = data.answer || "AI 답변이 비어 있습니다.";
        const results = data.results || [];
        const evidence = results.slice(0, 3).map((r, i) => `
            <li class="mb-2">
                <b>${i + 1}. ${escapeHtml(r.assembly || "")} ${escapeHtml(r.date || "")}</b>
                <div>${escapeHtml(r.meetingName || r.agenda || "")}</div>
                <div class="text-slate-700">${escapeHtml((r.text || r.preview || "").slice(0, 450))}</div>
            </li>`).join("");

        const responseHTML = `
            <div class="space-y-2">
                <div class="font-bold text-sky-800">회의록 기반 AI 답변</div>
                <div class="whitespace-pre-wrap">${escapeHtml(answer)}</div>
                <div class="border-t border-slate-200 pt-2">
                    <b>검색된 근거 (${results.length}건)</b>
                    <ol class="list-decimal pl-5 mt-2">${evidence || "<li>관련 회의록을 찾지 못했습니다.</li>"}</ol>
                </div>
                <div class="text-xs text-slate-500">응답 시간: ${escapeHtml(String(data.elapsed_ms ?? "-"))} ms</div>
            </div>`;

        appendAgentMessage("assistant", responseHTML);
    } catch (error) {
        console.error("AI 질의 실패:", error);
        appendAgentMessage(
            "assistant",
            `<p class="text-red-700">질의 처리 실패: ${escapeHtml(error.message || "서버 오류")}</p>`
        );
    }
}
'''

js, raw_count = re.subn(
    r'function executeRawVectorSearch\(\)\s*\{.*?(?=function toggleCardParentContext\()',
    lambda _: raw_search + "\n",
    js,
    count=1,
    flags=re.S
)

js, agent_count = re.subn(
    r'function respondAgentQuery\(query\)\s*\{.*?(?=function escapeHtml\()',
    lambda _: agent_response + "\n",
    js,
    count=1,
    flags=re.S
)

if raw_count != 1 or agent_count != 1:
    raise SystemExit(
        f"수정 중단: 검색 함수 매칭={raw_count}, 채팅 함수 매칭={agent_count}. 파일은 저장하지 않았습니다."
    )

path.write_text(js, encoding="utf-8")
print("완료: 원문 검색과 AI 채팅이 /api/search를 호출하도록 수정했습니다.")
