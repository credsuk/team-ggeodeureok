from pathlib import Path
from datetime import datetime
import re
import shutil

root = Path(".")
html_path = root / "frontend" / "index.html"
js_path = root / "frontend" / "app.js"

html = html_path.read_text(encoding="utf-8-sig")
js = js_path.read_text(encoding="utf-8-sig")

# 1. 수정 직전 백업
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
shutil.copy2(html_path, html_path.with_name(f"index.html.{stamp}.bak"))
shutil.copy2(js_path, js_path.with_name(f"app.js.{stamp}.bak"))

# 2. 동적 분석 컨테이너를 지도 섹션 바로 앞으로 이동
container = '<section id="dynamicAnalysisResultsContainer" class="col-span-12 space-y-4" aria-live="polite"></section>'
map_marker = '<section class="col-span-12 lg:col-span-7 flex flex-col glass-panel rounded-2xl border border-slate-300 overflow-hidden shadow-sm relative min-h-[580px]">'

if html.count(container) != 1:
    raise SystemExit(f"중단: 동적 분석 컨테이너 일치 개수={html.count(container)}")
if html.count(map_marker) != 1:
    raise SystemExit(f"중단: 지도 섹션 일치 개수={html.count(map_marker)}")

html = html.replace(container, "", 1)
html = html.replace(map_marker, container + "\n\n                " + map_marker, 1)

# 3. 동적 분석 활성 이슈 상태와 공통 getter 추가
anchor = "const COUNCIL_COORDS = {"
if js.count(anchor) != 1:
    raise SystemExit("중단: COUNCIL_COORDS 선언을 정확히 찾을 수 없음")

if "let dynamicAnalysisIssue = null;" not in js:
    # 상태 선언은 전역 좌표 객체보다 앞에 배치
    js = js.replace(anchor, "let dynamicAnalysisIssue = null;\n\n" + anchor, 1)

getter = """function getActiveIssue() {
    return dynamicAnalysisIssue || MASTER_ISSUES[currentIssueIndex];
}

"""
if "function getActiveIssue()" not in js:
    # 좌표 객체 선언의 종료를 찾아 getter를 그 뒤에 삽입
    start = js.index(anchor)
    open_brace = js.index("{", start)
    depth = 0
    end = None
    for i in range(open_brace, len(js)):
        if js[i] == "{":
            depth += 1
        elif js[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise SystemExit("중단: 좌표 객체 끝을 찾을 수 없음")
    js = js[:end] + "\n\n" + getter + js[end:]

# 4. 지도/타임라인 관련 함수만 활성 이슈 getter 사용
# 카탈로그/등록 이슈 렌더링 함수의 참조는 그대로 둔다.
function_ranges = [
    ("function updateLeafletArcs()", "function loadIssueData(index)"),
    ("function renderTimelineStep(stepIdx)", "function onCouncilMapClick(councilName)"),
    ("function onCouncilMapClick(councilName)", "function selectCouncilDetail(item)"),
    ("function toggleTimelinePlay()", "function renderCouncilFlowList(issue)"),
    ("function highlightFlowListItem(idx)", "function toggleParentContext()"),
]
for start_marker, end_marker in function_ranges:
    if start_marker not in js or end_marker not in js:
        raise SystemExit(f"중단: 함수 경계를 찾을 수 없음: {start_marker}")
    start = js.index(start_marker)
    end = js.index(end_marker, start)
    part = js[start:end].replace(
        "MASTER_ISSUES[currentIssueIndex]", "getActiveIssue()"
    )
    js = js[:start] + part + js[end:]

# 5. 동적 API 응답을 기존 레이더 UI 모델로 변환하는 함수 추가
helper = r'''
function applyDynamicAnalysisToRadar(data, query) {
    const rawPropagation = Array.isArray(data.propagation) ? data.propagation : [];
    const rawMapNodes = Array.isArray(data.map?.nodes) ? data.map.nodes : [];
    const summary = data.summary || {};
    const forecast = data.forecast || {};
    const evidence = Array.isArray(data.evidence) ? data.evidence : [];
    const rawMonthly = Array.isArray(data.monthly_series) ? data.monthly_series : [];

    const mapByCouncil = new Map();
    rawMapNodes.forEach(node => {
        const name = node.council ?? node.assembly ?? node.name;
        if (name) mapByCouncil.set(name, node);
    });

    const dynamicPropagation = rawPropagation.map((p, index) => {
        const council = p.council ?? p.assembly ?? p.name ?? "의회 정보 없음";
        const ev = Array.isArray(p.evidence) ? p.evidence[0] : null;
        const date = p.first_observation_date ?? p.date ?? ev?.meeting_date ?? ev?.date ?? "-";
        const coordNode = mapByCouncil.get(council);
        if (coordNode) {
            const lat = Number(coordNode.latitude ?? coordNode.lat);
            const lon = Number(coordNode.longitude ?? coordNode.lon ?? coordNode.lng);
            if (Number.isFinite(lat) && Number.isFinite(lon) && !COUNCIL_COORDS[council]) {
                COUNCIL_COORDS[council] = {
                    lat,
                    lon,
                    short: String(council).replace(/(광역시|특별자치도|특별자치시|자치도|도|의회)$/g, "").slice(0, 4)
                };
            }
        }
        return {
            order: Number(p.observation_order ?? p.order ?? index + 1),
            letter: String.fromCharCode(65 + (index % 26)),
            council,
            date: String(date),
            lagDays: Number(p.lag_days_from_first ?? p.lag_days ?? p.lagDays ?? 0),
            mentions: Number(p.mention_count ?? p.count ?? p.mentions ?? 0),
            action: String(p.stage ?? p.action ?? "관측"),
            view: "동적 분석",
            speaker: ev?.speaker ?? ev?.speaker_name ?? "회의록 발언",
            title: ev?.meeting_name ?? ev?.meetingName ?? ev?.agenda_title ?? query,
            excerpt: ev?.excerpt ?? ev?.text ?? ev?.content ?? "",
            similarity: Number(ev?.similarity ?? 0.92)
        };
    }).sort((a, b) => a.order - b.order);

    const firstRaw = rawPropagation[0] || {};
    const firstEv = Array.isArray(firstRaw.evidence) ? firstRaw.evidence[0] : evidence[0];
    const first = dynamicPropagation[0];
    const monthlySeries = rawMonthly.map(m => ({
        month: String(m.month ?? m.month_start ?? m.period ?? m.date ?? ""),
        count: Number(m.mention_count ?? m.count ?? m.mentions ?? 0)
    })).filter(m => m.month);

    if (!dynamicPropagation.length) {
        dynamicAnalysisIssue = null;
        return false;
    }

    dynamicAnalysisIssue = {
        id: "dynamic",
        name: data.issue?.issue_name ?? data.issue?.name ?? query,
        shortName: data.issue?.issue_name ?? data.issue?.name ?? query,
        category: "동적 분석",
        councilsCount: Number(summary.council_count ?? dynamicPropagation.length),
        evidenceCount: Number(summary.evidence_unit_count ?? evidence.length),
        dataRange: `${summary.first_observation_date ?? first?.date ?? "-"} ~ ${summary.last_observation_date ?? "-"}`,
        firstObservation: {
            date: first?.date ?? summary.first_observation_date ?? "-",
            council: first?.council ?? firstRaw.council ?? "-",
            speaker: first?.speaker ?? "회의록 발언",
            committee: firstEv?.committee ?? "",
            meetingName: first?.title ?? query,
            agenda: firstEv?.agenda_title ?? firstEv?.agenda ?? "",
            excerpt: first?.excerpt ?? "",
            parentContext: data.interpretation ?? first?.excerpt ?? ""
        },
        propagation: dynamicPropagation,
        monthlySeries,
        topPredict: null,
        mlForecast: { predictedMentions: Number(forecast.predicted_mention_count ?? 0) }
    };

    const issue = dynamicAnalysisIssue;
    const setText = (id, value) => {
        const el = document.getElementById(id);
        if (el) el.textContent = String(value ?? "-");
    };

    setText("kpiCouncilCount", `${issue.councilsCount} / 17곳`);
    setText("kpiCouncilRatio", `${((issue.councilsCount / 17) * 100).toFixed(1)}%`);
    setText("kpiEvidenceCount", `${issue.evidenceCount.toLocaleString("ko-KR")} 건`);
    setText("kpiDataRange", `※ ${issue.dataRange} 회의록 데이터`);
    setText("kpiTopPredictedCouncil", "예측 검증 필요");
    setText("kpiPredDays", forecast.target_month ? `대상 월 ${forecast.target_month}` : "예측 정보 확인");
    setText("kpiPredReason", forecast.warning ?? forecast.direction ?? "동적 분석 결과");
    setText("forecastCategoryBadge", "동적 분석");
    setText("forecastHeadline", `[${issue.shortName}] 검색 결과 기반 분석`);
    setText("forecastDescription", data.interpretation ?? `${issue.councilsCount}개 의회에서 ${issue.evidenceCount}건의 근거 자료를 확인했습니다.`);
    setText("forecastReasonText", `예측 상태: ${forecast.status ?? "검증 필요"} · ${forecast.warning ?? ""}`);

    const slider = document.getElementById("timelineSlider");
    const maxIdx = dynamicPropagation.length - 1;
    if (slider) {
        slider.min = 0;
        slider.max = maxIdx;
        slider.value = maxIdx;
    }
    setText("totalStepCount", dynamicPropagation.length);

    renderCouncilFlowList(issue);
    renderMonthlyChart(monthlySeries);
    currentTimelineStep = maxIdx;
    selectedCouncilName = first.council;
    selectCouncilDetail(first);
    highlightFlowListItem(0);
    renderTimelineStep(maxIdx);
    if (leafletMap && dynamicPropagation.length) {
        const coords = dynamicPropagation
            .map(p => COUNCIL_COORDS[p.council])
            .filter(Boolean);
        if (coords.length) {
            const bounds = L.latLngBounds(coords.map(c => [c.lat, c.lon]));
            leafletMap.fitBounds(bounds, { padding: [30, 30], maxZoom: 8 });
        }
    }
    return true;
}

'''
if "function applyDynamicAnalysisToRadar(data, query)" not in js:
    marker = "async function runDynamicIssueAnalysis(query)"
    if js.count(marker) != 1:
        raise SystemExit("중단: runDynamicIssueAnalysis 시작 위치가 불명확함")
    js = js.replace(marker, helper + marker, 1)

# 6. API 성공 시 동적 모델과 지도/차트를 갱신
success_anchor = '        const evidence = Array.isArray(data.evidence) ? data.evidence : [];'
if js.count(success_anchor) != 1:
    raise SystemExit("중단: API 응답 처리 위치를 찾을 수 없음")
call = success_anchor + '''

        const radarApplied = applyDynamicAnalysisToRadar(data, query);
        if (!radarApplied) {
            dynamicAnalysisIssue = null;
        }
'''
js = js.replace(success_anchor, call, 1)

# 기존 등록 이슈 클릭 시 동적 상태 초기화
load_anchor = 'function loadIssueData(index) {\n    hidePolicyTooltip();'
if js.count(load_anchor) != 1:
    raise SystemExit("중단: loadIssueData 시작 위치를 찾을 수 없음")
js = js.replace(load_anchor, 'function loadIssueData(index) {\n    dynamicAnalysisIssue = null;\n    hidePolicyTooltip();', 1)

# HTML/JS 변경을 디스크에 기록
html_path.write_text(html, encoding="utf-8")
js_path.write_text(js, encoding="utf-8")

# 간단한 구조 검사
html_check = html_path.read_text(encoding="utf-8")
js_check = js_path.read_text(encoding="utf-8")
assert html_check.count('id="dynamicAnalysisResultsContainer"') == 1
assert html_check.index('id="dynamicAnalysisResultsContainer"') < html_check.index('id="koreaLeafletMap"')
assert "function applyDynamicAnalysisToRadar(data, query)" in js_check
assert "const issue = getActiveIssue();" in js_check
print("수정 완료: index.html, app.js")
print("동적 결과 영역 위치: 지도 섹션 앞")
print("동적 지도/타임라인/월별 차트 연결 코드 추가")
print("수정 직전 백업: frontend/index.html.*.bak, frontend/app.js.*.bak")
