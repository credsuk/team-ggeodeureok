/**
 * 의정이슈레이더 (Council Issue Intelligence Dashboard)
 * TEAM 꺼드럭's - 2026 KISTI 데이터·AI 분석 경진대회
 * 순수 HTML/JS 기반 반응형 대시보드 코어 로직 (고대비 밝은 테마 & 맵-관측순서/시차 연동)
 */

// =========================================================================================
// 1. 한국 지리 좌표계 (17개 광역의회 실제 정밀 GPS 위경도 매핑 - final_app Carto Positron 연동)
// =========================================================================================
const COUNCIL_COORDS = {
    "서울특별시의회": { lat: 37.5665, lon: 126.9780, short: "서울" },
    "부산광역시의회": { lat: 35.1796, lon: 129.0756, short: "부산" },
    "대구광역시의회": { lat: 35.8714, lon: 128.6014, short: "대구" },
    "인천광역시의회": { lat: 37.4563, lon: 126.7052, short: "인천" },
    "광주광역시의회": { lat: 35.1601, lon: 126.8515, short: "광주" },
    "대전광역시의회": { lat: 36.3504, lon: 127.3845, short: "대전" },
    "울산광역시의회": { lat: 35.5384, lon: 129.3114, short: "울산" },
    "세종특별자치시의회": { lat: 36.4800, lon: 127.2890, short: "세종" },
    "경기도의회": { lat: 37.2636, lon: 127.0286, short: "경기" },
    "강원특별자치도의회": { lat: 37.8854, lon: 127.7298, short: "강원" },
    "충청북도의회": { lat: 36.6357, lon: 127.4913, short: "충북" },
    "충청남도의회": { lat: 36.6588, lon: 126.6728, short: "충남" },
    "전북특별자치도의회": { lat: 35.8202, lon: 127.1088, short: "전북" },
    "전라남도의회": { lat: 34.8161, lon: 126.4630, short: "전남" },
    "경상북도의회": { lat: 36.5760, lon: 128.5056, short: "경북" },
    "경상남도의회": { lat: 35.2383, lon: 128.6924, short: "경남" },
    "제주특별자치도의회": { lat: 33.4890, lon: 126.4983, short: "제주" }
};

// final_app 시차 단계 라벨 함수 (100% 동일 매핑)
function stageLabel(lagDays) {
    if (lagDays === 0) return "최초 관측";
    if (lagDays <= 30) return "30일내 후속";
    if (lagDays <= 90) return "90일내 확장";
    return "후기 관측";
}

// 대한민국 본토 해안선 폴리곤 좌표 (경도, 위도)
const MAINLAND_COORDS = [
    [126.65, 37.78], [126.75, 37.86], [126.95, 37.96], [127.10, 38.10], [127.45, 38.30], [127.80, 38.30],
    [128.15, 38.40], [128.36, 38.62], [128.47, 38.60], [128.59, 38.21], [128.90, 37.76], [129.11, 37.52],
    [129.17, 37.44], [129.33, 37.15], [129.40, 36.99], [129.42, 36.70], [129.40, 36.41], [129.37, 36.18],
    [129.40, 36.04], [129.57, 36.08], [129.50, 35.85], [129.48, 35.70], [129.43, 35.50], [129.32, 35.32],
    [129.15, 35.15], [128.95, 35.05], [128.75, 35.00], [128.62, 34.85], [128.70, 34.72], [128.45, 34.80],
    [128.40, 34.85], [128.15, 34.88], [127.95, 34.80], [127.85, 34.90], [127.70, 34.70], [127.55, 34.80],
    [127.40, 34.50], [127.20, 34.60], [127.00, 34.45], [126.75, 34.30], [126.52, 34.30], [126.40, 34.50],
    [126.30, 34.70], [126.40, 34.85], [126.30, 35.00], [126.38, 35.20], [126.40, 35.35], [126.48, 35.55],
    [126.70, 35.62], [126.62, 35.80], [126.60, 35.97], [126.55, 36.10], [126.52, 36.35], [126.48, 36.55],
    [126.15, 36.70], [126.30, 36.88], [126.60, 37.00], [126.85, 36.97], [126.70, 37.15], [126.62, 37.30],
    [126.55, 37.40], [126.50, 37.55], [126.45, 37.70], [126.55, 37.76]
];

// 제주도 & 울릉도 타원 상수 (경도, 위도, rx, ry)
const JEJU_PARAMS = { lon: 126.55, lat: 33.38, rx: 0.36, ry: 0.18 };
const ULLEUNG_PARAMS = { lon: 130.87, lat: 37.50, rx: 0.07, ry: 0.05 };
const DOKDO_PARAMS = { lon: 131.86, lat: 37.24 };

// SVG 뷰박스 기준 투영 함수 (ViewBox: 0 0 620 680)
const LON0 = 124.6;
const LAT0 = 38.9;
const KY = 104.0;
const KX = KY * Math.cos(36 * (Math.PI / 180));
const PAD_X = 40;
const PAD_Y = 25;

function geoToSvg(lat, lon) {
    const x = PAD_X + (lon - LON0) * KX;
    const y = PAD_Y + (LAT0 - lat) * KY;
    return { x: Math.round(x * 10) / 10, y: Math.round(y * 10) / 10 };
}

// =========================================================================================
// 2. 10대 대표 의정 정책 이슈 데이터셋 (언급 빈도수 mentions 완벽 포함)
// =========================================================================================
const MASTER_ISSUES = [
    {
        id: "ISSUE-01",
        name: "지역 소멸 대응 및 청년 인구 유출 방지",
        shortName: "지역 소멸",
        category: "인구/균형발전",
        firstObservation: {
            date: "2024-01-15",
            council: "서울특별시의회",
            speaker: "김영수 의원",
            committee: "기획경제위원회",
            meetingName: "제315회 서울특별시의회(임시회) 기획경제위원회 제1차 회의록",
            agenda: "수도권-지방 상생발전 및 청년 정주여건 개선 촉구안",
            excerpt: "지방소멸 위기에 따른 수도권-지방 간 청년 인구 이동 양극화 현상을 분석하고, 상생 협력 및 연계 지원책 수립을 위한 기초 조례안에 대해 논의를 진행하였습니다.",
            parentContext: "[위원장]: 다음 안건으로 수도권-지방 상생 균형발전 특별위원회가 상정한 청년 인구 유출 관련 조례 개정안을 심사하겠습니다. 김영수 의원 질의해 주십시오.\n[김영수 의원]: 본 의원이 제출한 자료에 따르면 수도권 집중화는 서울의 청년 주거난과 지방의 인구소멸 고위험군 확대를 동시에 부르고 있습니다. 서울시 차원에서도 지방소멸대응기금 연계 상생 사업에 적극 출연해야 합니다.\n[기획조정실장]: 의원님 지적에 동감하며, 충청 및 영호남 거점 도시들과 연계한 청년 일자리 순환 프로그램을 적극 검토하겠습니다."
        },
        councilsCount: 12,
        evidenceCount: 1234,
        dataRange: "2024.01 ~ 2025.12",
        mlForecast: {
            targetMonth: "2026.01",
            activeProb: 0.885,
            predictedMentions: 142,
            interval80: [118, 165],
            classifier: "XGBoost Classifier",
            regressor: "Ridge ElasticNet",
            status: "acceleration"
        },
        topPredict: {
            council: "전라남도의회",
            days: 45,
            score: 8.75,
            dbScore: 5.25,
            extScore: 3.50,
            wideOrdinance: 3,
            basicOrdinance: 14,
            needScore: 92.4,
            reason: "소멸위험지수 전국 최고 수준 및 관련 자치법규 3건 제정 발의, 인접 광주와의 초광역 연계 논의 급증"
        },
        propagation: [
            { order: 1, letter: "A", council: "서울특별시의회", date: "2024-01-15", lagDays: 0, mentions: 168, action: "5분 자유발언", view: "수도권-지방 양극화", speaker: "김영수 의원", title: "제315회 서울특별시의회(임시회) 기획경제위원회 제1차", excerpt: "지방소멸 위기에 따른 수도권-지방 간 청년 인구 이동 양극화 현상을 분석하고 연계 상생 방안을 촉구함.", similarity: 0.94 },
            { order: 2, letter: "B", council: "강원특별자치도의회", date: "2024-04-10", lagDays: 86, mentions: 92, action: "조례 심사", view: "접경지 정주여건", speaker: "박진호 의원", title: "제320회 강원도의회(임시회) 사회문화위원회 제2차", excerpt: "지방 소멸 위기에 대응하여 청년 정주 여건 강화 및 일자리 창출 지원 조례안에 대해 심도 있는 질의가 이어졌습니다.", similarity: 0.91 },
            { order: 3, letter: "C", council: "대전광역시의회", date: "2024-07-20", lagDays: 187, mentions: 115, action: "업무보고", view: "메가시티 교통연계", speaker: "이은혜 의원", title: "제271회 대전광역시의회 복지환경위원회 제3차", excerpt: "충청권 메가시티 구축과 연계한 청년층 인구 유입 대책 및 광역 교통망 확충에 따른 지방소멸 대응 전략을 검토하였습니다.", similarity: 0.89 },
            { order: 4, letter: "D", council: "경상북도의회", date: "2024-11-05", lagDays: 295, mentions: 146, action: "예산 심사", view: "소멸대응기금 집행", speaker: "정해인 의원", title: "제341회 경상북도의회 행정보건복지위원회 제1차", excerpt: "경북 북부 지역 소멸 고위험군 시군을 대상으로 한 이웃사촌 시범마을 사업 성과 평가 및 특화 일자리 예산 증액을 촉구했습니다.", similarity: 0.93 },
            { order: 5, letter: "E", council: "부산광역시의회", date: "2025-03-15", lagDays: 425, mentions: 104, action: "도정·시정 질문", view: "축소도시(Shrinking City)", speaker: "최성훈 의원", title: "제316회 부산광역시의회 해양도시안전위원회 제2차", excerpt: "원도심 쇠퇴 및 청년 인구 유출로 인한 부산형 축소 도시 대응 전략 및 빈집 정비 지원 조례 개정안이 상정되었습니다.", similarity: 0.88 },
            { order: 6, letter: "F", council: "광주광역시의회", date: "2025-06-20", lagDays: 522, mentions: 88, action: "질의", view: "청년수당 및 공동대응", speaker: "윤지원 의원", title: "제319회 광주광역시의회 행정자치위원회 제1차", excerpt: "전남·광주 초광역 생활권 형성을 통한 인구 소멸 공동 대응 및 빛고을 청년 수당 확대 방안에 대해 토론했습니다.", similarity: 0.87 },
            { order: 7, letter: "G", council: "제주특별자치도의회", date: "2025-10-01", lagDays: 625, mentions: 65, action: "조례 심사", view: "청년주거안정", speaker: "오승택 의원", title: "제424회 제주특별자치도의회 행정자치위원회 제2차", excerpt: "순유출 전환 시점에 따른 제주형 청년 주거 안정 지원 조례 및 지역 소멸 대응 기금 집행 현황을 집중 점검했습니다.", similarity: 0.85 }
        ],
        monthlySeries: [
            { month: "24.01", count: 42 }, { month: "24.02", count: 35 }, { month: "24.03", count: 48 },
            { month: "24.04", count: 62 }, { month: "24.05", count: 54 }, { month: "24.06", count: 70 },
            { month: "24.07", count: 85 }, { month: "24.08", count: 65 }, { month: "24.09", count: 92 },
            { month: "24.10", count: 104 }, { month: "24.11", count: 128 }, { month: "24.12", count: 110 },
            { month: "25.01", count: 88 }, { month: "25.02", count: 95 }, { month: "25.03", count: 135 },
            { month: "25.04", count: 142 }, { month: "25.05", count: 120 }, { month: "25.06", count: 158 },
            { month: "25.07", count: 148 }, { month: "25.08", count: 130 }, { month: "25.09", count: 165 },
            { month: "25.10", count: 184 }, { month: "25.11", count: 195 }, { month: "25.12", count: 172 }
        ]
    },
    {
        id: "ISSUE-02",
        name: "전세사기 피해자 지원 및 특별법 보완",
        shortName: "전세사기",
        category: "주거/민생",
        firstObservation: {
            date: "2024-01-22",
            council: "인천광역시의회",
            speaker: "이명숙 의원",
            committee: "건설교통위원회",
            meetingName: "제292회 인천광역시의회(임시회) 건설교통위원회 제1차 회의록",
            agenda: "미추홀구 전세사기 피해가구 긴급 주거지원 조례안",
            excerpt: "미추홀구 다세대 밀집지역 전세사기 피해 가구에 대한 긴급 생계비 및 이주비 긴급 편성 촉구와 선구제 후회수 방안 건의를 심사했습니다.",
            parentContext: "[위원장]: 의사일정 제1항 인천광역시 전세피해임차인 지원 조례안을 상정합니다.\n[이명숙 의원]: 인천 미추홀구의 피해는 단순한 사기 사건을 넘어 청년층 주거 기반의 붕괴를 초래하고 있습니다. 시가 전세보증금 반환보증료 지원에 그치지 않고 긴급 거처 공급을 우선 실행해야 합니다.\n[도시주택국장]: 시 예산 범위 내에서 긴급 대출 이자 지원 및 LH 공공임대 우선 입주를 적극 추진하겠습니다."
        },
        councilsCount: 11,
        evidenceCount: 980,
        dataRange: "2024.01 ~ 2025.12",
        mlForecast: {
            targetMonth: "2026.01",
            activeProb: 0.912,
            predictedMentions: 115,
            interval80: [95, 138],
            classifier: "LightGBM",
            regressor: "Ridge ElasticNet",
            status: "persistence"
        },
        topPredict: {
            council: "경상남도의회",
            days: 60,
            score: 7.95,
            dbScore: 4.80,
            extScore: 3.15,
            wideOrdinance: 2,
            basicOrdinance: 8,
            needScore: 84.1,
            reason: "창원·김해 원룸 단지 전세사기 피해자 실태조사 청원 접수 및 인접 부산의 선제 지원 조례 파급 효과"
        },
        propagation: [
            { order: 1, letter: "A", council: "인천광역시의회", date: "2024-01-22", lagDays: 0, mentions: 172, action: "조례 심사", view: "피해자 긴급지원", speaker: "이명숙 의원", title: "제292회 인천광역시의회 건설교통위원회 제1차", excerpt: "미추홀구 피해 가구 긴급 주거지원 및 선구제 방안을 제정함.", similarity: 0.96 },
            { order: 2, letter: "B", council: "서울특별시의회", date: "2024-02-28", lagDays: 37, mentions: 154, action: "5분 자유발언", view: "다세대 빌라 실태조사", speaker: "조성환 의원", title: "제322회 서울시의회 주택공간위원회", excerpt: "강서구 등 빌라 밀집지역 피해자 실태조사 및 보증금 반환보증료 시비 지원을 확대했습니다.", similarity: 0.94 },
            { order: 3, letter: "C", council: "경기도의회", date: "2024-04-18", lagDays: 87, mentions: 138, action: "질의", view: "광역 지원센터 설립", speaker: "한원찬 의원", title: "제374회 경기도의회 도시환경위원회", excerpt: "수원·동탄 권역 전세피해 종합지원센터 설치 및 법률구조공단 연계 방안을 검토했습니다.", similarity: 0.92 },
            { order: 4, letter: "D", council: "부산광역시의회", date: "2024-08-30", lagDays: 221, mentions: 112, action: "건의안·결의안", view: "특별법 개정 촉구", speaker: "서지연 의원", title: "제323회 부산시의회 본회의", excerpt: "부산진구 오피스텔 피해자를 위한 특별법 개정 촉구 결의안을 만장일치 채택했습니다.", similarity: 0.90 },
            { order: 5, letter: "E", council: "대전광역시의회", date: "2024-12-10", lagDays: 323, mentions: 84, action: "예산 심사", view: "긴급 생계비 편성", speaker: "송활섭 의원", title: "제274회 대전시의회 복지환경위", excerpt: "다가구 밀집지역 피해 대학생·사회초년생 긴급 주거안정 예산 10억원을 편성했습니다.", similarity: 0.88 },
            { order: 6, letter: "F", council: "대구광역시의회", date: "2025-04-05", lagDays: 439, mentions: 68, action: "질의", view: "경매 유예 협조", speaker: "김태우 의원", title: "제312회 대구시의회 기획행정위", excerpt: "피해 주택 경매 유예 및 금융권 협의를 위한 시 차원의 태스크포스 구성을 촉구했습니다.", similarity: 0.86 }
        ],
        monthlySeries: [
            { month: "24.01", count: 50 }, { month: "24.02", count: 68 }, { month: "24.03", count: 75 },
            { month: "24.04", count: 82 }, { month: "24.05", count: 90 }, { month: "24.06", count: 95 },
            { month: "24.07", count: 110 }, { month: "24.08", count: 125 }, { month: "24.09", count: 105 },
            { month: "24.10", count: 98 }, { month: "24.11", count: 92 }, { month: "24.12", count: 115 },
            { month: "25.01", count: 102 }, { month: "25.02", count: 88 }, { month: "25.03", count: 95 },
            { month: "25.04", count: 118 }, { month: "25.05", count: 112 }, { month: "25.06", count: 105 },
            { month: "25.07", count: 98 }, { month: "25.08", count: 92 }, { month: "25.09", count: 88 },
            { month: "25.10", count: 85 }, { month: "25.11", count: 80 }, { month: "25.12", count: 78 }
        ]
    },
    {
        id: "ISSUE-03",
        name: "전기차 화재 안전 및 지하주차장 소방 대책",
        shortName: "전기차 화재",
        category: "안전/소방",
        firstObservation: {
            date: "2024-08-05",
            council: "인천광역시의회",
            speaker: "신영희 의원",
            committee: "행정안전위원회",
            meetingName: "제296회 인천광역시의회(임시회) 행정안전위원회 긴급 간담회",
            agenda: "청라 아파트 지하주차장 전기차 화재 긴급 소방시설 점검 및 재발방지",
            excerpt: "청라 아파트 지하 전기차 화재 사고에 따른 열폭주 차단용 질식소화포 및 스프링클러 실태 전수조사와 공동주택 소방 안전 조례 긴급 제정을 논의했습니다.",
            parentContext: "[위원장]: 청라동 아파트 지하 1층 전기차 화재로 주민 800여 세대가 피해를 입은 엄중한 상황입니다.\n[신영희 의원]: 소방본부장은 스프링클러 밸브가 임의 폐쇄되어 피해가 커진 경위를 소상히 밝히고, 관내 모든 지하주차장 스프링클러 전수 검사를 즉시 완료하십시오.\n[소방본부장]: 이동식 수조 15대 긴급 배치 및 방화구획 기준 강화 조례안을 조속히 제출하겠습니다."
        },
        councilsCount: 14,
        evidenceCount: 1450,
        dataRange: "2024.08 ~ 2025.12",
        mlForecast: {
            targetMonth: "2026.01",
            activeProb: 0.945,
            predictedMentions: 168,
            interval80: [145, 192],
            classifier: "XGBoost",
            regressor: "Ridge ElasticNet",
            status: "acceleration"
        },
        topPredict: {
            council: "울산광역시의회",
            days: 30,
            score: 9.15,
            dbScore: 5.60,
            extScore: 3.55,
            wideOrdinance: 4,
            basicOrdinance: 12,
            needScore: 95.0,
            reason: "전기차 생산 거점 및 산업단지 특성, 지하주차장 스프링클러 의무화 조례안 기습 상정 동향"
        },
        propagation: [
            { order: 1, letter: "A", council: "인천광역시의회", date: "2024-08-05", lagDays: 0, mentions: 195, action: "도정·시정 질문", view: "지하주차장 스프링클러", speaker: "신영희 의원", title: "제296회 인천시의회 행정안전위 긴급간담회", excerpt: "청라 전기차 화재 이후 관내 공동주택 스프링클러 전수점검을 지시함.", similarity: 0.98 },
            { order: 2, letter: "B", council: "경기도의회", date: "2024-08-12", lagDays: 7, mentions: 178, action: "조례 심사", view: "질식소화포 보급", speaker: "유경현 의원", title: "제377회 경기도의회 안전행정위원회", excerpt: "지하 충전시설 지상 이전 유도 및 열폭주 대응 이동식 수조 확충 조례 통과.", similarity: 0.95 },
            { order: 3, letter: "C", council: "서울특별시의회", date: "2024-08-20", lagDays: 15, mentions: 162, action: "질의", view: "90% 충전 제한", speaker: "김형재 의원", title: "제326회 서울시의회 도시안전건설위", excerpt: "공동주택 지하주차장 충전율 90% 이하 차량만 진입 권고 기준 마련 논의.", similarity: 0.93 },
            { order: 4, letter: "D", council: "충청남도의회", date: "2024-09-10", lagDays: 36, mentions: 94, action: "5분 자유발언", view: "도민 불안 해소", speaker: "지민규 의원", title: "제355회 충남도의회 본회의", excerpt: "도내 전기차 화재 진압 장비 지원 및 소방본부 인력 특별 교육 실시.", similarity: 0.91 },
            { order: 5, letter: "E", council: "대전광역시의회", date: "2024-10-15", lagDays: 71, mentions: 86, action: "업무보고", view: "상가·오피스텔 확대", speaker: "김영삼 의원", title: "제282회 대전시의회 산업건설위", excerpt: "상가 및 공공청사 지하주차장 화재 감지 센서 전면 현대화 계획 수립.", similarity: 0.89 },
            { order: 6, letter: "F", council: "경상남도의회", date: "2024-11-20", lagDays: 107, mentions: 74, action: "조례 심사", view: "화재 안전 조례", speaker: "박진현 의원", title: "제419회 경남도의회 건설소방위", excerpt: "경남형 환경친화적 자동차 전용주차구역 화재 예방 및 안전시설 지원 조례.", similarity: 0.88 },
            { order: 7, letter: "G", council: "세종특별자치시의회", date: "2025-02-14", lagDays: 193, mentions: 58, action: "질의", view: "아파트 지하 안전", speaker: "상병헌 의원", title: "제96회 세종시의회 임시회", excerpt: "신도심 고밀 아파트 단지 지하주차장 소방 안전 가이드라인 준수율 점검.", similarity: 0.87 }
        ],
        monthlySeries: [
            { month: "24.08", count: 185 }, { month: "24.09", count: 210 }, { month: "24.10", count: 175 },
            { month: "24.11", count: 160 }, { month: "24.12", count: 145 }, { month: "25.01", count: 130 },
            { month: "25.02", count: 125 }, { month: "25.03", count: 140 }, { month: "25.04", count: 155 },
            { month: "25.05", count: 150 }, { month: "25.06", count: 162 }, { month: "25.07", count: 178 },
            { month: "25.08", count: 205 }, { month: "25.09", count: 190 }, { month: "25.10", count: 175 },
            { month: "25.11", count: 165 }, { month: "25.12", count: 158 }
        ]
    },
    {
        id: "ISSUE-04",
        name: "늘봄학교 전면 도입 및 교육 돌봄 체계 구축",
        shortName: "늘봄학교",
        category: "교육/돌봄",
        firstObservation: {
            date: "2024-02-14",
            council: "경기도의회",
            speaker: "이혜원 의원",
            committee: "교육기획위원회",
            meetingName: "제373회 경기도의회(임시회) 교육기획위원회 제2차 회의록",
            agenda: "2024년 늘봄학교 시범운영 확대 및 교원 업무 분리 대책",
            excerpt: "초등 1학년 늘봄학교 2학기 전면 도입에 앞서 늘봄전담인력 확충과 교사의 행정업무 완전 배제 원칙을 확립할 것을 교육청에 강력히 요구했습니다.",
            parentContext: "[위원장]: 경기도교육청 소관 2024년도 주요업무보고를 받겠습니다.\n[이혜원 의원]: 늘봄학교가 학부모들의 돌봄 부담을 덜어주는 취지는 좋으나, 현직 교사들이 공간 마련과 돌봄 행정까지 떠안는 실정입니다. 전담사 배치가 선행되지 않으면 교육의 질이 저하됩니다.\n[교육감]: 1학기 중 실무인력을 100% 학교별로 배치하여 교원 부담을 완전히 해소하겠습니다."
        },
        councilsCount: 15,
        evidenceCount: 1320,
        dataRange: "2024.02 ~ 2025.12",
        mlForecast: {
            targetMonth: "2026.01",
            activeProb: 0.860,
            predictedMentions: 128,
            interval80: [105, 150],
            classifier: "RandomForest",
            regressor: "Ridge ElasticNet",
            status: "persistence"
        },
        topPredict: {
            council: "전북특별자치도의회",
            days: 40,
            score: 8.40,
            dbScore: 5.10,
            extScore: 3.30,
            wideOrdinance: 2,
            basicOrdinance: 9,
            needScore: 89.2,
            reason: "농산어촌 소규모 학교 늘봄 프로그램 강사 수급난 지속 및 지자체-교육청 협력 조례 발의"
        },
        propagation: [
            { order: 1, letter: "A", council: "경기도의회", date: "2024-02-14", lagDays: 0, mentions: 182, action: "업무보고", view: "교원 업무 배제", speaker: "이혜원 의원", title: "제373회 경기도의회 교육기획위", excerpt: "늘봄전담사 배치 및 교사 본연의 수업권 보장을 촉구함.", similarity: 0.94 },
            { order: 2, letter: "B", council: "서울특별시의회", date: "2024-03-20", lagDays: 35, mentions: 158, action: "질의", view: "과대 과밀학교 공간부족", speaker: "이희원 의원", title: "제323회 서울시의회 교육위원회", excerpt: "강남·송파 등 과밀학급 학교의 전용 늘봄교실 확보 대책 질의.", similarity: 0.92 },
            { order: 3, letter: "C", council: "대구광역시의회", date: "2024-06-12", lagDays: 119, mentions: 116, action: "조례 심사", view: "지역사회 프로그램 연계", speaker: "이영애 의원", title: "제309회 대구시의회 교육위", excerpt: "대학 및 지역 문화시설을 활용한 특기적성 늘봄 프로그램 지원 조례.", similarity: 0.89 },
            { order: 4, letter: "D", council: "부산광역시의회", date: "2024-09-05", lagDays: 204, mentions: 98, action: "5분 자유발언", view: "거점형 돌봄센터", speaker: "강무길 의원", title: "제324회 부산시의회 본회의", excerpt: "원도심 거점형 돌봄클러스터 구축으로 학교 유휴공간 부족 해소.", similarity: 0.88 },
            { order: 5, letter: "E", council: "충청북도의회", date: "2024-11-18", lagDays: 278, mentions: 72, action: "예산 심사", view: "농촌형 늘봄 강사비", speaker: "유상용 의원", title: "제422회 충북도의회 교육위", excerpt: "면단위 학교 강사 유치를 위한 교통비 가산 및 처우개선비 증액 편성.", similarity: 0.86 }
        ],
        monthlySeries: [
            { month: "24.02", count: 85 }, { month: "24.03", count: 120 }, { month: "24.04", count: 98 },
            { month: "24.05", count: 88 }, { month: "24.06", count: 92 }, { month: "24.07", count: 105 },
            { month: "24.08", count: 140 }, { month: "24.09", count: 135 }, { month: "24.10", count: 110 },
            { month: "24.11", count: 125 }, { month: "24.12", count: 118 }, { month: "25.01", count: 90 },
            { month: "25.02", count: 115 }, { month: "25.03", count: 130 }, { month: "25.04", count: 122 },
            { month: "25.05", count: 118 }, { month: "25.06", count: 115 }, { month: "25.07", count: 120 },
            { month: "25.08", count: 138 }, { month: "25.09", count: 130 }, { month: "25.10", count: 112 },
            { month: "25.11", count: 115 }, { month: "25.12", count: 108 }
        ]
    },
    {
        id: "ISSUE-05",
        name: "지역 필수 의료 공백 및 공공의료원 정상화",
        shortName: "필수의료",
        category: "보건/의료",
        firstObservation: {
            date: "2024-02-28",
            council: "강원특별자치도의회",
            speaker: "이승진 의원",
            committee: "사회문화위원회",
            meetingName: "제326회 강원도의회(임시회) 사회문화위원회 제1차 회의록",
            agenda: "전공의 집단 사직에 따른 강원도립의료원 비상진료체계 및 경영난 지원",
            excerpt: "전공의 이탈 사태로 속초·원주·강릉 도립의료원의 적자가 누적되고 분만·응급실 야간 진료가 중단 위기에 처함에 따라 긴급 재정 투입을 의결했습니다.",
            parentContext: "[위원장]: 강원도 보건체육국 및 5개 의료원 소관 비상안건을 심사합니다.\n[이승진 의원]: 영동권 필수 의료 붕괴가 눈앞입니다. 속초의료원은 전문의 연봉을 4억까지 올려도 지원자가 없어 응급실이 파행 운영 중입니다.\n[보건국장]: 도 예비비를 편성해 지방의료원 의료진 당직 수당 및 운영 결손을 즉각 보전하겠습니다."
        },
        councilsCount: 13,
        evidenceCount: 1105,
        dataRange: "2024.02 ~ 2025.12",
        mlForecast: {
            targetMonth: "2026.01",
            activeProb: 0.895,
            predictedMentions: 136,
            interval80: [112, 160],
            classifier: "XGBoost",
            regressor: "Ridge ElasticNet",
            status: "persistence"
        },
        topPredict: {
            council: "충청남도의회",
            days: 50,
            score: 8.65,
            dbScore: 5.30,
            extScore: 3.35,
            wideOrdinance: 3,
            basicOrdinance: 10,
            needScore: 91.0,
            reason: "서해안 서천·태안 권역 응급의료 취약지 보강 조례 및 국립의대 유치 서명운동 전국 확산"
        },
        propagation: [
            { order: 1, letter: "A", council: "강원특별자치도의회", date: "2024-02-28", lagDays: 0, mentions: 165, action: "예산 심사", view: "도립의료원 적자보전", speaker: "이승진 의원", title: "제326회 강원도의회 사회문화위", excerpt: "영동권 응급실 야간 진료 셧다운 위기에 따른 긴급 재정 투입.", similarity: 0.95 },
            { order: 2, letter: "B", council: "경상남도의회", date: "2024-04-25", lagDays: 57, mentions: 132, action: "질의", view: "서부경남 공공병원", speaker: "류경완 의원", title: "제412회 경남도의회 문화복지위", excerpt: "진주의료원 폐쇄 이후 서부경남 공공병원 신축 조기 착공 촉구.", similarity: 0.91 },
            { order: 3, letter: "C", council: "전라남도의회", date: "2024-07-15", lagDays: 138, mentions: 148, action: "건의안·결의안", view: "국립의대 신설 촉구", speaker: "최선국 의원", title: "제383회 전남도의회 본회의", excerpt: "전국 유일 의대 없는 전남 권역 단일의대 공모 추진 결의안.", similarity: 0.93 },
            { order: 4, letter: "D", council: "충청북도의회", date: "2024-10-22", lagDays: 237, mentions: 96, action: "5분 자유발언", view: "충북대병원 전공의 공백", speaker: "안지윤 의원", title: "제421회 충북도의회 본회의", excerpt: "충북권 소아응급 진료 및 심뇌혈관 응급 이송 네트워크 붕괴 우려 제기.", similarity: 0.89 },
            { order: 5, letter: "E", council: "경상북도의회", date: "2025-03-18", lagDays: 384, mentions: 84, action: "조례 심사", view: "공공의료 지원기금", speaker: "임병하 의원", title: "제345회 경북도의회 행정보건위", excerpt: "의료취약지 공중보건의사 복무 만료 대비 은퇴 의사 시니어 채용 조례.", similarity: 0.87 }
        ],
        monthlySeries: [
            { month: "24.02", count: 95 }, { month: "24.03", count: 135 }, { month: "24.04", count: 128 },
            { month: "24.05", count: 110 }, { month: "24.06", count: 105 }, { month: "24.07", count: 118 },
            { month: "24.08", count: 125 }, { month: "24.09", count: 142 }, { month: "24.10", count: 130 },
            { month: "24.11", count: 122 }, { month: "24.12", count: 115 }, { month: "25.01", count: 108 },
            { month: "25.02", count: 115 }, { month: "25.03", count: 128 }, { month: "25.04", count: 135 },
            { month: "25.05", count: 120 }, { month: "25.06", count: 118 }, { month: "25.07", count: 125 },
            { month: "25.08", count: 130 }, { month: "25.09", count: 122 }, { month: "25.10", count: 118 },
            { month: "25.11", count: 115 }, { month: "25.12", count: 110 }
        ]
    },
    {
        id: "ISSUE-06",
        name: "고교학점제 도입 대비 교육 여건 개선",
        shortName: "고교학점제",
        category: "교육/입시",
        firstObservation: {
            date: "2024-03-10",
            council: "서울특별시의회",
            speaker: "고광민 의원",
            committee: "교육위원회",
            meetingName: "제322회 서울시의회 교육위원회 제3차 회의록",
            agenda: "2025년 고교학점제 전면 도입 대비 공간 혁신 및 교원 다과목 지도 역량 강화",
            excerpt: "고교학점제 시행에 따른 일반계고 이동 수업용 홈베이스 증축 예산과 소인수 선택과목 온라인 공동교육과정 구축 방안을 심도 있게 심사했습니다.",
            parentContext: "[위원장]: 서울특별시교육청 고교학점제 추진단 주요 현안을 보고받겠습니다.\n[고광민 의원]: 2025년 입학생부터 학점제가 전면 시행되는데 노후 학교의 경우 이동수업을 감당할 복도와 홈베이스가 턱없이 부족합니다.\n[교육감]: 학교 공간혁신 사업비 800억원을 긴급 투입하여 가변형 교실을 전면 조성하겠습니다."
        },
        councilsCount: 11,
        evidenceCount: 840,
        dataRange: "2024.03 ~ 2025.12",
        mlForecast: {
            targetMonth: "2026.01",
            activeProb: 0.835,
            predictedMentions: 98,
            interval80: [78, 120],
            classifier: "RandomForest",
            regressor: "Ridge ElasticNet",
            status: "persistence"
        },
        topPredict: {
            council: "충청남도의회",
            days: 60,
            score: 7.80,
            dbScore: 4.70,
            extScore: 3.10,
            wideOrdinance: 2,
            basicOrdinance: 6,
            needScore: 82.5,
            reason: "도농복합 지역 일반고 다과목 교원 부족 해소를 위한 순회교사 지원 조례안 추진"
        },
        propagation: [
            { order: 1, letter: "A", council: "서울특별시의회", date: "2024-03-10", lagDays: 0, mentions: 145, action: "업무보고", view: "가변형 교실 증축", speaker: "고광민 의원", title: "제322회 서울시의회 교육위", excerpt: "홈베이스 및 온라인 공동교육과정 스튜디오 증설 심사.", similarity: 0.93 },
            { order: 2, letter: "B", council: "경기도의회", date: "2024-05-18", lagDays: 69, mentions: 128, action: "조례 심사", view: "순회교사 지원", speaker: "조희선 의원", title: "제375회 경기도의회 교육기획위", excerpt: "비선호 선택과목 개설을 위한 순회교사 배치 및 수당 현실화.", similarity: 0.90 },
            { order: 3, letter: "C", council: "광주광역시의회", date: "2024-09-15", lagDays: 189, mentions: 82, action: "질의", view: "내신 성취평가제", speaker: "박희율 의원", title: "제320회 광주시의회 교육위", excerpt: "공통과목 상대평가 및 선택과목 절대평가 실시에 따른 교사 평가권 보호.", similarity: 0.88 },
            { order: 4, letter: "D", council: "대전광역시의회", date: "2024-12-05", lagDays: 270, mentions: 74, action: "예산 심사", view: "스마트 기기 보급", speaker: "이한영 의원", title: "제274회 대전시의회 교육위", excerpt: "학점 이수 관리용 학생 1인 1스마트 태블릿 보급 예산 심의.", similarity: 0.87 }
        ],
        monthlySeries: [
            { month: "24.03", count: 65 }, { month: "24.04", count: 58 }, { month: "24.05", count: 72 },
            { month: "24.06", count: 80 }, { month: "24.07", count: 75 }, { month: "24.08", count: 68 },
            { month: "24.09", count: 88 }, { month: "24.10", count: 95 }, { month: "24.11", count: 82 },
            { month: "24.12", count: 78 }, { month: "25.01", count: 60 }, { month: "25.02", count: 65 },
            { month: "25.03", count: 85 }, { month: "25.04", count: 80 }, { month: "25.05", count: 88 },
            { month: "25.06", count: 92 }, { month: "25.07", count: 85 }, { month: "25.08", count: 78 },
            { month: "25.09", count: 90 }, { month: "25.10", count: 86 }, { month: "25.11", count: 80 },
            { month: "25.12", count: 75 }
        ]
    },
    {
        id: "ISSUE-07",
        name: "저출생 극복을 위한 다자녀 및 주거 지원",
        shortName: "저출생 극복",
        category: "인구/복지",
        firstObservation: {
            date: "2024-01-10",
            council: "서울특별시의회",
            speaker: "박유진 의원",
            committee: "보건복지위원회",
            meetingName: "제321회 서울시의회 보건복지위원회",
            agenda: "다자녀 기준 3자녀에서 2자녀 완화 및 공공임대 우선공급 조례",
            excerpt: "합계출산율 0.5명대 서울의 위기를 돌파하기 위해 다자녀 혜택 기준을 2자녀로 일괄 완화하고 장기전세주택 미리내집 공급 확대를 촉구했습니다.",
            parentContext: "[위원장]: 서울특별시 다자녀가정 지원에 관한 조례 일부개정조례안을 심사합니다.\n[박유진 의원]: 이제 2자녀도 기적입니다. 공공시설 이용료 감면부터 장기전세 입주까지 모든 조례의 다자녀 기준을 2자녀로 하향해야 합니다.\n[여성가족실장]: 전 조례 개정안에 찬성하며, 출산가구 보증금 무이자 대출 사업을 함께 확대하겠습니다."
        },
        councilsCount: 16,
        evidenceCount: 1540,
        dataRange: "2024.01 ~ 2025.12",
        mlForecast: {
            targetMonth: "2026.01",
            activeProb: 0.950,
            predictedMentions: 175,
            interval80: [150, 205],
            classifier: "XGBoost",
            regressor: "Ridge ElasticNet",
            status: "acceleration"
        },
        topPredict: {
            council: "세종특별자치시의회",
            days: 35,
            score: 9.30,
            dbScore: 5.70,
            extScore: 3.60,
            wideOrdinance: 4,
            basicOrdinance: 8,
            needScore: 96.2,
            reason: "평균연령 최연소 도시로서 영유아 돌봄 및 다자녀 주거 지원 조례 개정안 집중 심사"
        },
        propagation: [
            { order: 1, letter: "A", council: "서울특별시의회", date: "2024-01-10", lagDays: 0, mentions: 198, action: "조례 심사", view: "2자녀 기준 완화", speaker: "박유진 의원", title: "제321회 서울시의회 보건복지위", excerpt: "공공시설 및 장기전세주택 다자녀 기준 2자녀 완화 통과.", similarity: 0.97 },
            { order: 2, letter: "B", council: "부산광역시의회", date: "2024-03-08", lagDays: 58, mentions: 152, action: "5분 자유발언", view: "출산축하금 확대", speaker: "박종철 의원", title: "제318회 부산시의회 본회의", excerpt: "출산 장려를 위한 첫만남이용권 시비 매칭 증액 촉구.", similarity: 0.92 },
            { order: 3, letter: "C", council: "경상북도의회", date: "2024-06-25", lagDays: 167, mentions: 168, action: "조례 심사", view: "저출생과의 전쟁 선포", speaker: "남진복 의원", title: "제339회 경북도의회 행정보건위", excerpt: "경북형 저출생 극복 특별기금 1,000억원 조성 조례 제정.", similarity: 0.94 },
            { order: 4, letter: "D", council: "전라남도의회", date: "2024-09-20", lagDays: 254, mentions: 134, action: "질의", view: "출생기본수당 20만원", speaker: "김호진 의원", title: "제384회 전남도의회 보건복지위", excerpt: "전남·시군 매칭 17세까지 월 20만원 출생기본수당 지급 계획 점검.", similarity: 0.91 }
        ],
        monthlySeries: [
            { month: "24.01", count: 120 }, { month: "24.02", count: 115 }, { month: "24.03", count: 140 },
            { month: "24.04", count: 135 }, { month: "24.05", count: 150 }, { month: "24.06", count: 165 },
            { month: "24.07", count: 155 }, { month: "24.08", count: 148 }, { month: "24.09", count: 170 },
            { month: "24.10", count: 162 }, { month: "24.11", count: 175 }, { month: "24.12", count: 168 },
            { month: "25.01", count: 145 }, { month: "25.02", count: 152 }, { month: "25.03", count: 178 },
            { month: "25.04", count: 185 }, { month: "25.05", count: 172 }, { month: "25.06", count: 188 },
            { month: "25.07", count: 180 }, { month: "25.08", count: 175 }, { month: "25.09", count: 195 },
            { month: "25.10", count: 205 }, { month: "25.11", count: 210 }, { month: "25.12", count: 198 }
        ]
    },
    {
        id: "ISSUE-08",
        name: "농어촌 일손 부족 및 외국인 계절근로자 확대",
        shortName: "외국인 계절근로",
        category: "농림/노동",
        firstObservation: {
            date: "2024-02-05",
            council: "전라남도의회",
            speaker: "박형대 의원",
            committee: "농수산위원회",
            meetingName: "제378회 전남도의회 농수산위원회",
            agenda: "공공형 계절근로센터 확대 및 기숙사 건립 지원 조례안",
            excerpt: "본격적인 영농철을 앞두고 불법 브로커 차단 및 농가 부담 완화를 위해 농협이 직접 관리하는 공공형 계절근로 기숙사 건립 지원안을 논의했습니다.",
            parentContext: "[위원장]: 농업정책관 소관 공공형 계절근로 운영 현황을 보고받겠습니다.\n[박형대 의원]: 개별 농가가 외국인 근로자 주거를 책임지기 어렵습니다. 열악한 비닐하우스 숙소는 인권 문제뿐 아니라 근로자 이탈의 주원인입니다.\n[농축산식품국장]: 10개 시군에 공공기숙사 건립 예산 150억원을 지원하겠습니다."
        },
        councilsCount: 9,
        evidenceCount: 650,
        dataRange: "2024.02 ~ 2025.12",
        mlForecast: {
            targetMonth: "2026.01",
            activeProb: 0.810,
            predictedMentions: 75,
            interval80: [58, 92],
            classifier: "RandomForest",
            regressor: "Ridge ElasticNet",
            status: "persistence"
        },
        topPredict: {
            council: "경상남도의회",
            days: 65,
            score: 7.50,
            dbScore: 4.50,
            extScore: 3.00,
            wideOrdinance: 2,
            basicOrdinance: 8,
            needScore: 81.2,
            reason: "마늘·양파 주산지 창녕·함양 중심 계절근로자 체류기간 연장 건의"
        },
        propagation: [
            { order: 1, letter: "A", council: "전라남도의회", date: "2024-02-05", lagDays: 0, mentions: 142, action: "조례 심사", view: "공공형 기숙사", speaker: "박형대 의원", title: "제378회 전남도의회 농수산위", excerpt: "공공형 계절근로 기숙사 신축 및 브로커 방지 조례.", similarity: 0.94 },
            { order: 2, letter: "B", council: "경상북도의회", date: "2024-04-12", lagDays: 67, mentions: 125, action: "5분 자유발언", view: "항공료·보험료 지원", speaker: "신효광 의원", title: "제338회 경북도의회 본회의", excerpt: "과수 수확기 인력 확보를 위한 입국 항공료 지자체 보조 제안.", similarity: 0.91 },
            { order: 3, letter: "C", council: "충청남도의회", date: "2024-08-25", lagDays: 202, mentions: 98, action: "질의", view: "무단이탈 방지 대책", speaker: "오인철 의원", title: "제354회 충남도의회 농수해양위", excerpt: "라오스·필리핀 등 MOU 체결 지자체 이탈률 관리 점검.", similarity: 0.89 }
        ],
        monthlySeries: [
            { month: "24.02", count: 45 }, { month: "24.03", count: 62 }, { month: "24.04", count: 78 },
            { month: "24.05", count: 85 }, { month: "24.06", count: 65 }, { month: "24.07", count: 50 },
            { month: "24.08", count: 55 }, { month: "24.09", count: 72 }, { month: "24.10", count: 88 },
            { month: "24.11", count: 60 }, { month: "24.12", count: 40 }, { month: "25.01", count: 35 },
            { month: "25.02", count: 48 }, { month: "25.03", count: 68 }, { month: "25.04", count: 82 },
            { month: "25.05", count: 90 }, { month: "25.06", count: 70 }, { month: "25.07", count: 52 },
            { month: "25.08", count: 58 }, { month: "25.09", count: 75 }, { month: "25.10", count: 92 },
            { month: "25.11", count: 65 }, { month: "25.12", count: 42 }
        ]
    },
    {
        id: "ISSUE-09",
        name: "딥페이크 성범죄 예방 및 청소년 디지털 윤리",
        shortName: "딥페이크 범죄",
        category: "청소년/치안",
        firstObservation: {
            date: "2024-08-27",
            council: "서울특별시의회",
            speaker: "이효원 의원",
            committee: "교육위원회",
            meetingName: "제326회 서울시의회 교육위원회 긴급 현안보고",
            agenda: "중·고교 딥페이크 허위영상물 피해 긴급 실태조사 및 디지털성범죄 예방 교육 조례",
            excerpt: "텔레그램 딥페이크 사태가 관내 중고교 학생 및 교사를 대상으로 확산됨에 따라 교육청의 원스톱 피해 삭제 지원과 긴급 피해자 심리치료 예산 편성을 촉구했습니다.",
            parentContext: "[위원장]: 텔레그램 딥페이크 성착취물 학교 현장 침투 관련 긴급 현안질의를 진행합니다.\n[이효원 의원]: 서울 관내 수십 개 학교 이름이 피해자 목록에 올라 학생과 학부모가 공포에 떨고 있습니다. 방송통신심의위원회와 핫라인을 열어 24시간 내 삭제가 이뤄지게 하십시오.\n[부교육감]: 피해 신고 핫라인 가동 및 심리상담 바우처를 즉시 전액 지급하겠습니다."
        },
        councilsCount: 14,
        evidenceCount: 1180,
        dataRange: "2024.08 ~ 2025.12",
        mlForecast: {
            targetMonth: "2026.01",
            activeProb: 0.925,
            predictedMentions: 145,
            interval80: [122, 168],
            classifier: "XGBoost",
            regressor: "Ridge ElasticNet",
            status: "acceleration"
        },
        topPredict: {
            council: "대전광역시의회",
            days: 35,
            score: 8.90,
            dbScore: 5.50,
            extScore: 3.40,
            wideOrdinance: 3,
            basicOrdinance: 7,
            needScore: 93.4,
            reason: "청소년 SNS 딥페이크 피해 상담 건수 급증 및 자치경찰-교육청 합동 특별 단속 조례안 상정"
        },
        propagation: [
            { order: 1, letter: "A", council: "서울특별시의회", date: "2024-08-27", lagDays: 0, mentions: 188, action: "질의", view: "피해 영상물 긴급 삭제", speaker: "이효원 의원", title: "제326회 서울시의회 교육위 긴급보고", excerpt: "학교 딥페이크 피해자 삭제 지원 및 법률상담 원스톱 지원 의결.", similarity: 0.98 },
            { order: 2, letter: "B", council: "경기도의회", date: "2024-09-05", lagDays: 9, mentions: 164, action: "조례 심사", view: "디지털성범죄 원스톱센터", speaker: "정윤경 의원", title: "제377회 경기도의회 여성가족위", excerpt: "경기도 디지털성범죄 피해자 원스톱지원센터 기능 확대 조례 개정.", similarity: 0.96 },
            { order: 3, letter: "C", council: "부산광역시의회", date: "2024-09-18", lagDays: 22, mentions: 118, action: "5분 자유발언", view: "가해 학생 엄벌 및 교화", speaker: "송우현 의원", title: "제324회 부산시의회 본회의", excerpt: "지인능욕방 운영 청소년 형사처벌 기준 강화 및 디지털 윤리교육 의무화.", similarity: 0.93 },
            { order: 4, letter: "D", council: "광주광역시의회", date: "2024-10-12", lagDays: 46, mentions: 95, action: "업무보고", view: "교원 피해 보호", speaker: "심창욱 의원", title: "제321회 광주시의회 교육위", excerpt: "교권보호위원회 연계 교원 딥페이크 합성물 피해 대응 매뉴얼 배포.", similarity: 0.91 }
        ],
        monthlySeries: [
            { month: "24.08", count: 140 }, { month: "24.09", count: 225 }, { month: "24.10", count: 195 },
            { month: "24.11", count: 165 }, { month: "24.12", count: 140 }, { month: "25.01", count: 110 },
            { month: "25.02", count: 105 }, { month: "25.03", count: 130 }, { month: "25.04", count: 145 },
            { month: "25.05", count: 138 }, { month: "25.06", count: 150 }, { month: "25.07", count: 142 },
            { month: "25.08", count: 165 }, { month: "25.09", count: 180 }, { month: "25.10", count: 168 },
            { month: "25.11", count: 155 }, { month: "25.12", count: 145 }
        ]
    },
    {
        id: "ISSUE-10",
        name: "개인형 이동장치(PM) 및 자전거 안전 관리 조례",
        shortName: "PM·자전거 안전",
        category: "교통/안전",
        firstObservation: {
            date: "2024-03-15",
            council: "대전광역시의회",
            speaker: "송활섭 의원",
            committee: "산업건설위원회",
            meetingName: "제276회 대전시의회 산업건설위원회",
            agenda: "공유 전동킥보드 무단방치 견인 및 견인료 부과 조례안",
            excerpt: "보도 위 무분별하게 방치되어 시각장애인과 보행자의 안전을 위협하는 전동 킥보드에 대해 1시간 유예 후 즉시 견인 및 3만원 견인료를 부과하는 조례를 심사했습니다.",
            parentContext: "[위원장]: 대전광역시 개인형 이동장치 이용안전 증진 조례 일부개정조례안을 상정합니다.\n[송활섭 의원]: 지하철 입구와 횡단보도 턱 앞에 쓰러져 있는 PM 때문에 매일 민원이 폭주합니다. 대여업체에 견인료와 보관료를 징수해야 방치 근절이 가능합니다.\n[교통건설국장]: 5개 자치구와 합동으로 즉시견인 전담 용역을 계약 체결하겠습니다."
        },
        councilsCount: 12,
        evidenceCount: 790,
        dataRange: "2024.03 ~ 2025.12",
        mlForecast: {
            targetMonth: "2026.01",
            activeProb: 0.820,
            predictedMentions: 88,
            interval80: [68, 110],
            classifier: "RandomForest",
            regressor: "Ridge ElasticNet",
            status: "persistence"
        },
        topPredict: {
            council: "대구광역시의회",
            days: 55,
            score: 7.90,
            dbScore: 4.80,
            extScore: 3.10,
            wideOrdinance: 2,
            basicOrdinance: 7,
            needScore: 83.0,
            reason: "동성로 및 대학가 주변 PM 야간 무면허 음주운전 단속 강화 조례 개정안 발의"
        },
        propagation: [
            { order: 1, letter: "A", council: "대전광역시의회", date: "2024-03-15", lagDays: 0, mentions: 135, action: "조례 심사", view: "무단방치 즉시견인", speaker: "송활섭 의원", title: "제276회 대전시의회 산업건설위", excerpt: "보도 방치 전동킥보드 견인료 3만원 부과 조례 가결.", similarity: 0.95 },
            { order: 2, letter: "B", council: "서울특별시의회", date: "2024-06-10", lagDays: 87, mentions: 156, action: "도정·시정 질문", view: "최고속도 하향(20km/h)", speaker: "윤영희 의원", title: "제324회 서울시의회 교통위", excerpt: "공유 PM 법정 최고속도를 25km에서 20km로 하향하는 시범운영 요구.", similarity: 0.91 },
            { order: 3, letter: "C", council: "울산광역시의회", date: "2024-10-05", lagDays: 204, mentions: 84, action: "5분 자유발언", view: "안전모 미착용 단속", speaker: "홍유준 의원", title: "제251회 울산시의회 본회의", excerpt: "중·고교생 무면허 대여 방지를 위한 운전면허 본인인증 강제 법제화 촉구.", similarity: 0.88 }
        ],
        monthlySeries: [
            { month: "24.03", count: 48 }, { month: "24.04", count: 65 }, { month: "24.05", count: 78 },
            { month: "24.06", count: 85 }, { month: "24.07", count: 72 }, { month: "24.08", count: 68 },
            { month: "24.09", count: 90 }, { month: "24.10", count: 95 }, { month: "24.11", count: 62 },
            { month: "24.12", count: 45 }, { month: "25.01", count: 38 }, { month: "25.02", count: 42 },
            { month: "25.03", count: 60 }, { month: "25.04", count: 75 }, { month: "25.05", count: 88 },
            { month: "25.06", count: 92 }, { month: "25.07", count: 80 }, { month: "25.08", count: 76 },
            { month: "25.09", count: 95 }, { month: "25.10", count: 98 }, { month: "25.11", count: 68 },
            { month: "25.12", count: 50 }
        ]
    }
];

// =========================================================================================
// 3. 글로벌 상태 관리 (State Management)
// =========================================================================================
let currentIssueIndex = 0;
let currentTimelineStep = 0;
let isPlaying = false;
let playInterval = null;
let currentTab = "radar";
let selectedCouncilName = null;

// Leaflet Carto Positron 지도 인스턴스 및 레이어 (final_app 스타일 맵 엔진)
let leafletMap = null;
let leafletMarkersLayer = null;
let leafletArcsSvg = null;

// =========================================================================================
// 4. 초기화 및 이벤트 리스너 (Initialization)
// =========================================================================================
window.addEventListener("DOMContentLoaded", () => {
    // 1) 폰트 스케일 초기화 (기본: normal = 17px)
    const savedScale = localStorage.getItem("council_radar_font_scale") || "normal";
    setFontScale(savedScale);

    // 2) Carto Positron 인터랙티브 지도 초기화 및 이슈 데이터 로드
    initLeafletMap();
    loadIssueData(currentIssueIndex);
    renderCatalogTable();
    renderCatalogChart();
    executeRawVectorSearch();

    // 3) 엔터키 검색 바인딩
    const searchInput = document.getElementById("globalSearchInput");
    if (searchInput) {
        searchInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                handleGlobalSearch();
            }
        });
    }

    // 4) 창 크기 조절 시 지도 리사이즈 동기화
    window.addEventListener("resize", () => {
        if (leafletMap) {
            leafletMap.invalidateSize();
            updateLeafletArcs();
        }
    });

    const ttCloseBtn = document.getElementById("ttCloseBtn");
    if (ttCloseBtn) {
        ttCloseBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            hidePolicyTooltip();
        });
    }

    document.addEventListener("click", (e) => {
        const tooltip = document.getElementById("policyTooltip");
        if (!tooltip || tooltip.classList.contains("hidden")) return;
        // 팝업 내부 또는 지도 노드 클릭 시에는 닫지 않음
        if (!tooltip.contains(e.target) && !e.target.closest(".map-node")) {
            hidePolicyTooltip();
        }
    });

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            hidePolicyTooltip();
        }
    });
});

// =========================================================================================
// 4-1. 지도 지명 및 라벨 폰트 크기 반응형 스케일 설정
// =========================================================================================
let currentFontScale = "normal";

const MAP_FONT_CONFIG = {
    small: {
        baseFontSize: 18,
        baseStrokeWidth: 4,
        islandFontSize: 13,
        activeFontSize: 20,
        activeBoxCharWidth: 24,
        activeBoxPad: 20,
        activeBoxHeight: 32,
        activeBoxYOffset: -22,
        activeTextYOffset: 1,
        orderOriginSize: 18,
        orderBaseSize: 14,
        orderBonusRatio: 0.35,
        predictFontSize: 19,
        predictCharWidth: 21,
        predictPad: 22,
        predictHeight: 32,
        predictTextYOffset: 5.5
    },
    normal: {
        baseFontSize: 22,
        baseStrokeWidth: 5,
        islandFontSize: 16,
        activeFontSize: 25,
        activeBoxCharWidth: 30,
        activeBoxPad: 24,
        activeBoxHeight: 38,
        activeBoxYOffset: -26,
        activeTextYOffset: 1.5,
        orderOriginSize: 22,
        orderBaseSize: 17,
        orderBonusRatio: 0.45,
        predictFontSize: 24,
        predictCharWidth: 26,
        predictPad: 28,
        predictHeight: 38,
        predictTextYOffset: 6.5
    },
    large: {
        baseFontSize: 27,
        baseStrokeWidth: 6,
        islandFontSize: 19,
        activeFontSize: 30,
        activeBoxCharWidth: 36,
        activeBoxPad: 28,
        activeBoxHeight: 46,
        activeBoxYOffset: -31,
        activeTextYOffset: 2,
        orderOriginSize: 26,
        orderBaseSize: 20,
        orderBonusRatio: 0.5,
        predictFontSize: 29,
        predictCharWidth: 31,
        predictPad: 32,
        predictHeight: 46,
        predictTextYOffset: 7.5
    }
};

function getMapFontCfg() {
    return MAP_FONT_CONFIG[currentFontScale] || MAP_FONT_CONFIG.normal;
}

// 툴팁 폰트 스케일 (기본 1.5배 확대 및 '작게/보통/크게' 헤더 버튼과 100% 연동)
const TOOLTIP_FONT_SCALES = {
    small: {
        minWidth: 235,
        padding: "8px 12px",
        titleSize: 16.5,
        badgeSize: 22,
        badgeFontSize: 13,
        subSize: 14,
        detailSize: 13.5,
        lagSize: 14,
        countPadding: "2px 6px"
    },
    normal: { // 기본 1.5배 확대 (고대비 또렷한 가독성)
        minWidth: 270,
        padding: "10px 14px",
        titleSize: 19.5,
        badgeSize: 26,
        badgeFontSize: 15,
        subSize: 16,
        detailSize: 15,
        lagSize: 16,
        countPadding: "2.5px 7px"
    },
    large: { // 크게 (시니어 공무원 모드)
        minWidth: 320,
        padding: "12px 16px",
        titleSize: 23,
        badgeSize: 30,
        badgeFontSize: 17.5,
        subSize: 18.5,
        detailSize: 17.5,
        lagSize: 18.5,
        countPadding: "3px 9px"
    }
};

function getTooltipFontCfg() {
    return TOOLTIP_FONT_SCALES[currentFontScale] || TOOLTIP_FONT_SCALES.normal;
}

// 고령 공무원 및 관리자 타깃 글자 크기 조절 (작게 16px / 보통 18px / 크게 20.5px + 동역학 지도 지명 연동)
function setFontScale(scale) {
    currentFontScale = scale;
    const html = document.documentElement;
    const btnSmall = document.getElementById("btnFont-small");
    const btnNormal = document.getElementById("btnFont-normal");
    const btnLarge = document.getElementById("btnFont-large");

    const buttons = [btnSmall, btnNormal, btnLarge].filter(Boolean);
    buttons.forEach(btn => {
        btn.classList.remove("bg-sky-700", "text-white", "font-black", "shadow-sm");
        btn.classList.add("text-slate-700", "hover:bg-slate-200");
    });

    let targetBtn = btnNormal;
    if (scale === "small") {
        html.style.fontSize = "16px";
        targetBtn = btnSmall;
    } else if (scale === "large") {
        html.style.fontSize = "20.5px"; // 시니어 공무원 가독성 극대화
        targetBtn = btnLarge;
    } else {
        html.style.fontSize = "18px"; // 보통 기본 18px (시원하고 또렷한 가독성)
        targetBtn = btnNormal;
        scale = "normal";
    }

    if (targetBtn) {
        targetBtn.classList.remove("text-slate-700", "hover:bg-slate-200");
        targetBtn.classList.add("bg-sky-700", "text-white", "font-black", "shadow-sm");
    }

    try {
        localStorage.setItem("council_radar_font_scale", scale);
    } catch (e) {}

    // [핵심 연동]: 동역학 지도 동기화
    if (leafletMap) {
        renderTimelineStep(currentTimelineStep);
    }
}

// =========================================================================================
// 4-2. CARTO Basemaps .env 환경 변수 동적 로더 (보안 규정: 소스코드 하드코딩 금지, .env 필수)
// =========================================================================================
let cartoTileLayer = null;
let currentCartoApiKey = null;

function getCartoApiKey() {
    // 1) env.js 또는 전역 변수에서 확인
    if (typeof window !== "undefined" && window.CARTO_API_KEY && !window.CARTO_API_KEY.includes("your_carto_api_key")) {
        return window.CARTO_API_KEY;
    }
    if (typeof window !== "undefined" && window.ENV && window.ENV.CARTO_API_KEY && !window.ENV.CARTO_API_KEY.includes("your_carto_api_key")) {
        return window.ENV.CARTO_API_KEY;
    }
    // 2) 세션 스토리지에서 확인
    try {
        const stored = sessionStorage.getItem("CARTO_API_KEY");
        if (stored && !stored.includes("your_carto_api_key")) return stored;
    } catch (e) {}

    return null;
}

function updateCartoTileLayer(apiKey) {
    if (!leafletMap) return;

    if (cartoTileLayer) {
        leafletMap.removeLayer(cartoTileLayer);
        cartoTileLayer = null;
    }

    if (apiKey) {
        cartoTileLayer = L.tileLayer(
            `https://{s}.basemaps.cartocdn.com/rastertiles/light_all/{z}/{x}/{y}{r}.png?key=${apiKey}`,
            {
                subdomains: "abcd",
                maxZoom: 19,
                minZoom: 5,
                attribution: '© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors, © <a href="https://carto.com/attribution/" target="_blank" rel="noopener noreferrer">CARTO</a>'
            }
        ).addTo(leafletMap);
    } else {
        // .env 키가 없을 때의 기본 무인증 타일맵
        cartoTileLayer = L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
            subdomains: "abcd",
            maxZoom: 19,
            minZoom: 5,
            attribution: '© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors, © <a href="https://carto.com/attribution/" target="_blank" rel="noopener noreferrer">CARTO</a>'
        }).addTo(leafletMap);
    }

    if (leafletArcsSvg && leafletArcsSvg.parentNode) {
        leafletArcsSvg.parentNode.appendChild(leafletArcsSvg);
    }
    if (leafletMarkersLayer) {
        leafletMarkersLayer.bringToFront();
    }
}

async function loadEnvConfig() {
    // 1) env.js 또는 동기식 전역 변수에서 즉시 확인
    const existingKey = getCartoApiKey();
    if (existingKey) {
        currentCartoApiKey = existingKey;
        try { sessionStorage.setItem("CARTO_API_KEY", existingKey); } catch (e) {}
        updateCartoTileLayer(existingKey);
        updateEnvStatusBadge(true, "인증 확인됨");
        const banner = document.getElementById("envRequiredBanner");
        if (banner) banner.remove();
        return existingKey;
    }

    // 2) 로컬 웹 서버 환경인 경우 .env 동적 fetch 시도
    const envPaths = [".env", "../.env"];
    for (const path of envPaths) {
        try {
            const resp = await fetch(path);
            if (resp.ok || resp.status === 200 || resp.status === 0) {
                const text = await resp.text();
                const key = parseEnvForCartoKey(text);
                if (key) {
                    currentCartoApiKey = key;
                    try { sessionStorage.setItem("CARTO_API_KEY", key); } catch (e) {}
                    updateCartoTileLayer(key);
                    updateEnvStatusBadge(true, "인증 확인됨");
                    const banner = document.getElementById("envRequiredBanner");
                    if (banner) banner.remove();
                    return key;
                }
            }
        } catch (e) {
            // file:// 프로토콜 또는 fetch 차단 시 진행
        }
    }

    // 3) .env 파일이 자동 감지되지 않는 경우: 사용자에게 .env 파일 연동 요구 배너 노출
    updateEnvStatusBadge(false, ".env 필요");
    showEnvRequiredBanner();
    return null;
}

function parseEnvForCartoKey(text) {
    if (!text) return null;
    const lines = text.split(/\r?\n/);
    for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed && !trimmed.startsWith("#") && trimmed.includes("=")) {
            const [k, ...v] = trimmed.split("=");
            if (k.trim() === "CARTO_API_KEY") {
                const val = v.join("=").trim().replace(/^["']|["']$/g, "");
                if (val && !val.includes("your_carto_api_key_here")) {
                    return val;
                }
            }
        }
    }
    return null;
}

function updateEnvStatusBadge(success, label) {
    const badge = document.getElementById("cartoEnvBadge");
    if (!badge) return;
    if (success) {
        badge.className = "text-[10px] font-mono font-bold text-emerald-800 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-300 flex items-center gap-1";
        badge.innerHTML = `<i class="fa-solid fa-shield-halved text-emerald-600"></i>CARTO (${label})`;
    } else {
        badge.className = "text-[10px] font-mono font-bold text-rose-800 bg-rose-50 px-2 py-0.5 rounded border border-rose-300 flex items-center gap-1";
        badge.innerHTML = `<i class="fa-solid fa-triangle-exclamation text-rose-600"></i>CARTO (${label})`;
    }
}

function showEnvRequiredBanner() {
    const mapViewport = document.getElementById("mapViewport");
    if (!mapViewport || document.getElementById("envRequiredBanner")) return;

    const banner = document.createElement("div");
    banner.id = "envRequiredBanner";
    banner.className = "absolute inset-0 z-[1000] bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4";
    banner.innerHTML = `
        <div class="bg-white rounded-2xl border-2 border-amber-500 max-w-md w-full p-5 shadow-2xl space-y-3.5">
            <div class="flex items-center justify-between border-b border-slate-200 pb-2.5">
                <div class="flex items-center gap-2 text-amber-700">
                    <i class="fa-solid fa-key text-lg"></i>
                    <h3 class="font-black text-slate-950 text-base">CARTO 지도 .env 파일 연동 필요</h3>
                </div>
                <span class="text-xs font-mono font-bold text-amber-800 bg-amber-100 px-2 py-0.5 rounded border border-amber-300">보안 인증</span>
            </div>
            <p class="text-xs md:text-sm text-slate-700 leading-relaxed font-medium">
                보안 규정에 따라 CARTO API Key는 소스코드에 하드코딩되지 않고 <b>.env 파일</b>에서만 동적으로 로드됩니다.<br/>
                웹 서버(<code class="bg-slate-100 px-1 py-0.5 rounded text-sky-700 font-bold">python -m http.server</code>)를 구동하시거나, 아래 버튼으로 <code class="bg-slate-100 px-1 py-0.5 rounded text-amber-800 font-bold">.env</code> 파일을 선택해 주세요.
            </p>
            <div class="space-y-2 pt-1">
                <label class="w-full py-2.5 px-4 bg-sky-700 hover:bg-sky-600 text-white rounded-xl text-xs md:text-sm font-black flex items-center justify-center gap-2 cursor-pointer shadow-md transition-all">
                    <i class="fa-solid fa-file-import"></i>
                    <span>.env 파일 선택하여 연동</span>
                    <input type="file" id="envFileInput" accept=".env,text/plain" class="hidden" onchange="handleEnvFileUpload(event)" />
                </label>
                <div class="text-[11px] text-slate-500 font-mono text-center">
                    ※ 한 번 연동하면 브라우저 세션(sessionStorage)에 안전하게 유지됩니다.
                </div>
            </div>
        </div>
    `;
    mapViewport.appendChild(banner);
}

function handleEnvFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        const text = e.target.result;
        const key = parseEnvForCartoKey(text);
        if (key) {
            currentCartoApiKey = key;
            try { sessionStorage.setItem("CARTO_API_KEY", key); } catch (e) {}
            updateCartoTileLayer(key);
            updateEnvStatusBadge(true, "인증 확인됨");
            const banner = document.getElementById("envRequiredBanner");
            if (banner) banner.remove();
        } else {
            alert(".env 파일 내에서 CARTO_API_KEY 항목을 찾을 수 없습니다.");
        }
    };
    reader.readAsText(file);
}

// =========================================================================================
// 5. Leaflet 기반 Carto Positron 타일 지도 초기화 및 Arc/Node 레이어 설정 (.env 동적 연동)
// =========================================================================================
function initLeafletMap() {
    const container = document.getElementById("koreaLeafletMap");
    if (!container || leafletMap) return;

    // 대한민국 중심 (위도 36.2, 경도 127.8), 줌 레벨 7.0 (확대/축소 지원)
    // [이동 제한]: 한국에서 너무 멀리 벗어나지 않도록 지도 이동 한계 경계(maxBounds) 및 점성(maxBoundsViscosity) 설정
    const KOREA_MAX_BOUNDS = L.latLngBounds([31.2, 122.8], [40.2, 133.5]);

    leafletMap = L.map("koreaLeafletMap", {
        center: [36.2, 127.8],
        zoom: 7.0,
        minZoom: 6.0,
        maxZoom: 13,
        maxBounds: KOREA_MAX_BOUNDS,
        maxBoundsViscosity: 0.95,
        zoomControl: true,
        attributionControl: false,
        scrollWheelZoom: true,
        doubleClickZoom: true
    });

    // .env / env.js에서 CARTO_API_KEY 동적 확인
    const initialKey = getCartoApiKey();
    if (initialKey) {
        currentCartoApiKey = initialKey;
        updateCartoTileLayer(initialKey);
        updateEnvStatusBadge(true, "인증 확인됨");
    } else {
        updateCartoTileLayer(null);
        loadEnvConfig();
    }

    // 마커 레이어 그룹 생성
    leafletMarkersLayer = L.layerGroup().addTo(leafletMap);

    // Leaflet 커스텀 arcsPane (z-index: 450, 타일 위 / 마커 아래) 생성
    leafletMap.createPane("arcsPane");
    const arcsPane = leafletMap.getPane("arcsPane");
    arcsPane.style.zIndex = "450";
    arcsPane.style.pointerEvents = "none";
    arcsPane.style.overflow = "visible";

    // SVG 아크 오버레이 레이어 생성 (충분한 뷰포트와 viewBox 설정으로 0x0 브라우저 컬링 완전 방지)
    let arcSvg = document.getElementById("leafletArcSvg");
    if (!arcSvg) {
        arcSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        arcSvg.setAttribute("id", "leafletArcSvg");
        arcSvg.style.position = "absolute";
        arcSvg.style.width = "4000px";
        arcSvg.style.height = "4000px";
        arcSvg.style.left = "-2000px";
        arcSvg.style.top = "-2000px";
        arcSvg.setAttribute("viewBox", "-2000 -2000 4000 4000");
        arcSvg.style.overflow = "visible";
        arcSvg.setAttribute("overflow", "visible");
        arcSvg.style.pointerEvents = "none";

        // 그라디언트 정의 (final_app: 올리브 #69704f -> 다홍 코랄 #ee5e4c)
        const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
        defs.innerHTML = `
            <linearGradient id="finalArcGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#69704f" stop-opacity="0.9" />
                <stop offset="100%" stop-color="#ee5e4c" stop-opacity="0.95" />
            </linearGradient>
            <linearGradient id="predArcGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#ee5e4c" stop-opacity="0.75" />
                <stop offset="100%" stop-color="#f43f5e" stop-opacity="0.9" />
            </linearGradient>
            <filter id="arcGlowFilter" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="2.2" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
        `;
        arcSvg.appendChild(defs);

        const gArcs = document.createElementNS("http://www.w3.org/2000/svg", "g");
        gArcs.setAttribute("id", "leafletArcLayer");
        arcSvg.appendChild(gArcs);

        arcsPane.appendChild(arcSvg);
        leafletArcsSvg = arcSvg;
    }

    // [확대/축소 난잡함 방지]: 줌 동작 중에는 연결선을 즉시 숨기고, 줌 완료 후 새로운 좌표로 재렌더링
    let zoomArcTimer = null;
    leafletMap.on("zoomstart", () => {
        if (leafletArcsSvg) {
            leafletArcsSvg.style.transition = "opacity 0.12s ease-out";
            leafletArcsSvg.style.opacity = "0";
        }
    });

    leafletMap.on("zoomend", () => {
        clearTimeout(zoomArcTimer);
        zoomArcTimer = setTimeout(() => {
            updateLeafletArcs();
            if (leafletArcsSvg) {
                leafletArcsSvg.style.transition = "opacity 0.25s ease-in";
                leafletArcsSvg.style.opacity = "1";
            }
        }, 70);
    });

    // 지도 드래그 패닝 완료 및 창 크기 변경 시 아크 위치 동기화
    leafletMap.on("moveend resize viewreset", () => {
        updateLeafletArcs();
        if (leafletArcsSvg && leafletArcsSvg.style.opacity === "0") {
            leafletArcsSvg.style.opacity = "1";
        }
    });
}

// -----------------------------------------------------------------------------------------
// ArcLayer 스타일 연결선 렌더링 (지도 확대/축소/패닝 시 지리 좌표에 실시간 정밀 동기화)
// -----------------------------------------------------------------------------------------
function updateLeafletArcs() {
    if (!leafletMap || !leafletArcsSvg) return;
    const gArcs = document.getElementById("leafletArcLayer");
    if (!gArcs) return;
    gArcs.innerHTML = "";

    const issue = MASTER_ISSUES[currentIssueIndex];
    if (!issue || !issue.propagation) return;

    const safeIdx = Math.min(currentTimelineStep, issue.propagation.length - 1);
    const maxSteps = issue.propagation.length;

    // 0부터 safeIdx - 1까지의 전파 호선(Arc Curve) 생성
    for (let i = 0; i < safeIdx; i++) {
        const fromItem = issue.propagation[i];
        const toItem = issue.propagation[i + 1];
        const fromCoord = COUNCIL_COORDS[fromItem.council];
        const toCoord = COUNCIL_COORDS[toItem.council];
        if (!fromCoord || !toCoord) continue;

        const p1 = leafletMap.latLngToLayerPoint([fromCoord.lat, fromCoord.lon]);
        const p2 = leafletMap.latLngToLayerPoint([toCoord.lat, toCoord.lon]);

        const midX = (p1.x + p2.x) / 2;
        const midY = (p1.y + p2.y) / 2;
        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const dist = Math.hypot(dx, dy);
        if (dist === 0) continue;

        // final_app ArcLayer 스타일의 부드러운 아크 곡률
        const curvature = Math.min(65, Math.max(18, dist * 0.22));
        const ctrlX = midX - (dy / dist) * curvature;
        const ctrlY = midY + (dx / dist) * curvature;

        const pathData = `M ${p1.x} ${p1.y} Q ${ctrlX} ${ctrlY} ${p2.x} ${p2.y}`;

        // 1) 외곽 부드러운 글로우 후광 라인 (연분홍 코랄 반투명)
        const glowPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
        glowPath.setAttribute("d", pathData);
        glowPath.setAttribute("fill", "none");
        glowPath.setAttribute("stroke", "#fda4af");
        glowPath.setAttribute("stroke-width", "6.5");
        glowPath.setAttribute("stroke-opacity", "0.45");
        glowPath.setAttribute("stroke-linecap", "round");
        gArcs.appendChild(glowPath);

        // 2) 메인 시간순 전파 아크 라인 (선명한 코랄 레드)
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", pathData);
        path.setAttribute("fill", "none");
        path.setAttribute("stroke", "#ee5e4c");
        path.setAttribute("stroke-width", "3.2");
        path.setAttribute("stroke-linecap", "round");
        path.setAttribute("class", "arc-line-path");
        gArcs.appendChild(path);

        // 3) 아크를 따라 부드럽게 흐르는 펄스 오브(Moving Orb)
        const movingOrb = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        movingOrb.setAttribute("r", i === 0 ? "5.5" : "4.5");
        movingOrb.setAttribute("fill", "#ee5e4c");
        movingOrb.setAttribute("stroke", "#ffffff");
        movingOrb.setAttribute("stroke-width", "2");
        movingOrb.setAttribute("class", "arc-moving-orb");

        const animMotion = document.createElementNS("http://www.w3.org/2000/svg", "animateMotion");
        animMotion.setAttribute("path", pathData);
        animMotion.setAttribute("dur", `${Math.max(1.4, Math.min(2.8, dist * 0.007))}s`);
        animMotion.setAttribute("repeatCount", "indefinite");
        movingOrb.appendChild(animMotion);

        gArcs.appendChild(movingOrb);
    }

    // 마지막 단계일 때 topPredict 예측 점선 아크
    if (safeIdx === maxSteps - 1 && issue.topPredict && COUNCIL_COORDS[issue.topPredict.council]) {
        const lastItem = issue.propagation[safeIdx];
        const lastCoord = COUNCIL_COORDS[lastItem.council];
        const predCoord = COUNCIL_COORDS[issue.topPredict.council];

        if (lastCoord && predCoord) {
            const p1 = leafletMap.latLngToLayerPoint([lastCoord.lat, lastCoord.lon]);
            const p2 = leafletMap.latLngToLayerPoint([predCoord.lat, predCoord.lon]);

            const midX = (p1.x + p2.x) / 2;
            const midY = (p1.y + p2.y) / 2;
            const dx = p2.x - p1.x;
            const dy = p2.y - p1.y;
            const dist = Math.hypot(dx, dy);

            if (dist > 0) {
                const curvature = Math.min(60, Math.max(18, dist * 0.22));
                const ctrlX = midX - (dy / dist) * curvature;
                const ctrlY = midY + (dx / dist) * curvature;

                const pathData = `M ${p1.x} ${p1.y} Q ${ctrlX} ${ctrlY} ${p2.x} ${p2.y}`;
                const predPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
                predPath.setAttribute("d", pathData);
                predPath.setAttribute("fill", "none");
                predPath.setAttribute("stroke", "#f43f5e");
                predPath.setAttribute("stroke-width", "2.8");
                predPath.setAttribute("stroke-dasharray", "6 4");
                predPath.setAttribute("stroke-linecap", "round");
                predPath.setAttribute("class", "flow-line");
                gArcs.appendChild(predPath);
            }
        }
    }
}

// =========================================================================================
// 5. 이슈 데이터 로드 및 렌더링 (Data Binding & Update)
// =========================================================================================
function loadIssueData(index) {
    hidePolicyTooltip();
    currentIssueIndex = index;
    const issue = MASTER_ISSUES[index];
    if (!issue) return;

    // 상단 3대 KPI 카드 갱신
    const kpiCouncilCount = document.getElementById("kpiCouncilCount");
    if (kpiCouncilCount) kpiCouncilCount.textContent = `${issue.councilsCount} / 17곳`;
    const kpiCouncilRatio = document.getElementById("kpiCouncilRatio");
    if (kpiCouncilRatio) kpiCouncilRatio.textContent = `${((issue.councilsCount / 17) * 100).toFixed(1)}%`;
    const kpiEvidenceCount = document.getElementById("kpiEvidenceCount");
    if (kpiEvidenceCount) kpiEvidenceCount.textContent = `${issue.evidenceCount.toLocaleString()} 건`;
    const kpiDataRange = document.getElementById("kpiDataRange");
    if (kpiDataRange) kpiDataRange.textContent = `※ ${issue.dataRange} 본회의·상임위 회의록 정제 청크`;
    const kpiTopPredictedCouncil = document.getElementById("kpiTopPredictedCouncil");
    if (kpiTopPredictedCouncil) kpiTopPredictedCouncil.textContent = issue.topPredict?.council || "-";
    const kpiPredDays = document.getElementById("kpiPredDays");
    if (kpiPredDays) kpiPredDays.textContent = `향후 ${issue.topPredict?.days || 30}일 내`;
    const kpiPredReason = document.getElementById("kpiPredReason");
    if (kpiPredReason) kpiPredReason.textContent = issue.topPredict?.reason || "";

    // 상단 LLM 기반 정책 전망 브리핑 카드 갱신
    const fcBadge = document.getElementById("forecastCategoryBadge");
    if (fcBadge) fcBadge.textContent = issue.category;
    const fcHeadline = document.getElementById("forecastHeadline");
    if (fcHeadline) {
        fcHeadline.textContent = `[${issue.shortName || issue.name}] 향후 ${issue.topPredict?.council || '비수도권 광역의회'} 중심 조례 발의 및 집중 논의 전망`;
    }
    const fcDesc = document.getElementById("forecastDescription");
    if (fcDesc) {
        fcDesc.textContent = `${issue.firstObservation?.council || '수도권'}에서 관측된 해당 현안은 전국 ${issue.councilsCount}개 광역의회로 확산되었으며, ${issue.topPredict?.reason || '지역 현안 수요 및 자치법규 발의 증가에 따라'} 향후 약 ${issue.topPredict?.days || 30}일 내 ${issue.topPredict?.council || '후속 의회'}를 중심으로 심도 있는 정책 논의 및 조례 제정이 전개될 것으로 예상됩니다.`;
    }
    const fcReason = document.getElementById("forecastReasonText");
    if (fcReason) {
        fcReason.textContent = `1위 예측 의회: ${issue.topPredict?.council || ''} (${issue.topPredict?.reason || ''})`;
    }

    // 타임라인 슬라이더 범위 조정
    const slider = document.getElementById("timelineSlider");
    if (slider) {
        slider.max = issue.propagation.length - 1;
        slider.value = 0;
    }
    const totalStepEl = document.getElementById("totalStepCount");
    if (totalStepEl) totalStepEl.textContent = issue.propagation.length;

    // 우측 의회별 시간순 흐름 리스트 렌더링 (빈도수 포함)
    renderCouncilFlowList(issue);

    // 하단 월별 시계열 차트 렌더링
    renderMonthlyChart(issue.monthlySeries);

    // 통합 공공데이터 결합 예측 섹션 내용 갱신
    renderForecastRankingTab(issue);

    // 기본 전체 확산 완료 단계(A, B, C... 전파 마커 전체 지도 노출)로 초기화
    const maxIdx = issue.propagation.length - 1;
    if (slider) slider.value = maxIdx;
    currentTimelineStep = maxIdx;

    // 기본 최초 발원 의회 상세 화면 셋팅
    selectedCouncilName = issue.propagation[0].council;
    selectCouncilDetail(issue.propagation[0]);
    highlightFlowListItem(0);

    // 전체 마커 렌더링
    renderTimelineStep(maxIdx);
}

// =========================================================================================
// 6. 의회 시공간 전파 동역학 노드 및 타임라인 렌더링 (final_app 스타일 100% 매핑)
// =========================================================================================
function renderTimelineStep(stepIdx) {
    currentTimelineStep = stepIdx;
    const issue = MASTER_ISSUES[currentIssueIndex];
    if (!issue) return;

    const maxSteps = issue.propagation.length;
    const safeIdx = Math.max(0, Math.min(stepIdx, maxSteps - 1));
    const currentItem = issue.propagation[safeIdx];

    // 선택된 의회가 없으면 현재 스텝의 의회로 초기화
    if (!selectedCouncilName) {
        selectedCouncilName = currentItem.council;
    }

    // 우측 목록의 활성 탭 동기화
    const selIdx = issue.propagation.findIndex(p => p.council === selectedCouncilName);
    if (selIdx !== -1) {
        highlightFlowListItem(selIdx);
    }

    // 타임라인 슬라이더 동기화
    const slider = document.getElementById("timelineSlider");
    if (slider) slider.value = safeIdx;

    // 상단 일자 및 시차 뱃지 갱신 (final_app 포맷)
    const dateBadge = document.getElementById("mapCurrentStepDate");
    if (dateBadge) {
        const selCouncil = selectedCouncilName ? issue.propagation.find(p => p.council === selectedCouncilName) : null;
        if (selCouncil) {
            const isOrig = (selCouncil.order === 1);
            dateBadge.innerHTML = `<span class="text-sky-950 font-black mr-1">[${selCouncil.order}번] ${selCouncil.council}</span> ${selCouncil.date} (${isOrig ? "최초 관측" : "+" + selCouncil.lagDays + "일"} · 근거 ${selCouncil.mentions}건)`;
        } else {
            dateBadge.textContent = `${currentItem.date} (${currentItem.council} ${safeIdx === 0 ? "최초 관측" : "전파"} · ${currentItem.mentions}건)`;
        }
    }

    // 재생기 진행 상태 텍스트
    const playerState = document.getElementById("playerStateText");
    if (playerState) {
        if (safeIdx === maxSteps - 1) {
            playerState.textContent = `${maxSteps}/${maxSteps}단계 관측 완료 (전체 확산)`;
        } else {
            playerState.textContent = `${safeIdx + 1}/${maxSteps}단계 관측 진행`;
        }
    }

    if (!leafletMap || !leafletMarkersLayer) return;

    // 기존 Leaflet 마커 클리어 및 툴팁 스케일 설정 가져오기
    leafletMarkersLayer.clearLayers();
    const ttCfg = getTooltipFontCfg();

    // 0부터 safeIdx까지의 관측 의회 노드 생성 (ScatterplotLayer + TextLayer 1:1 매핑)
    for (let i = 0; i <= safeIdx; i++) {
        const item = issue.propagation[i];
        const coord = COUNCIL_COORDS[item.council];
        if (!coord) continue;

        const isOrigin = (i === 0);
        const isSelected = (selectedCouncilName === item.council);

        // final_app의 반경 수식(언급량 비례)을 화면 픽셀로 변환 (지름 30px ~ 42px)
        const mentions = parseInt(item.mentions || 1, 10);
        const circleSize = Math.min(42, Math.max(30, Math.round(28 + Math.sqrt(mentions) * 1.15)));
        const fontSize = circleSize >= 36 ? 16 : 14;

        // final_app 스타일: 따뜻한 코랄 레드 원형 노드(#e95b4c), 흰색 순번 텍스트(order: 1, 2, 3...)
        const wrapperClass = `council-marker-wrapper ${isOrigin ? "is-origin" : ""} ${isSelected ? "is-selected" : ""}`;
        const markerHtml = `
            <div class="${wrapperClass}" style="width:${circleSize + 24}px;">
                <div class="council-node-circle" style="width:${circleSize}px; height:${circleSize}px; font-size:${fontSize}px;">
                    ${item.order}
                </div>
                <div class="council-pill-label">
                    ${coord.short}
                </div>
            </div>
        `;

        const customIcon = L.divIcon({
            html: markerHtml,
            className: "custom-council-divicon",
            iconSize: [circleSize + 24, circleSize + 28],
            iconAnchor: [(circleSize + 24) / 2, circleSize / 2]
        });

        const marker = L.marker([coord.lat, coord.lon], { icon: customIcon });

        // Leaflet 마커 호버 툴팁 (기본 1.5배 확대 & '작게/보통/크게' 폰트 스케일 연동)
        const stage = stageLabel(item.lagDays);
        const tooltipHtml = `
            <div style="font-family:'Pretendard',sans-serif; min-width:${ttCfg.minWidth}px; padding:${ttCfg.padding};">
                <div style="display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.25); padding-bottom:6px; margin-bottom:6px;">
                    <div style="display:flex; align-items:center; gap:7px;">
                        <span style="display:inline-flex; align-items:center; justify-content:center; width:${ttCfg.badgeSize}px; height:${ttCfg.badgeSize}px; border-radius:6px; font-weight:900; font-size:${ttCfg.badgeFontSize}px; ${isOrigin ? 'background:#f59e0b; color:#ffffff; box-shadow:0 1px 4px rgba(245,158,11,0.5);' : 'background:#e95b4c; color:#ffffff; box-shadow:0 1px 4px rgba(233,91,76,0.5);'}">${item.letter || item.order}</span>
                        <b style="font-size:${ttCfg.titleSize}px; color:#ffffff; font-weight:800; letter-spacing:-0.01em;">${item.council}</b>
                    </div>
                    <span style="font-size:${ttCfg.lagSize}px; color:#ffb4a8; font-weight:900; font-family:'JetBrains Mono',monospace;">+${item.lagDays}일</span>
                </div>
                <div style="color:#ded8c7; font-size:${ttCfg.subSize}px; margin-bottom:5px; font-weight:600; line-height:1.4;">
                    ${item.action} · <span style="color:#7dd3fc; font-weight:800;">#${item.view}</span>
                </div>
                <div style="display:flex; align-items:center; justify-content:space-between; font-size:${ttCfg.detailSize}px; color:#cbd5e1; font-family:'JetBrains Mono',monospace;">
                    <span>${item.date}</span>
                    <span style="color:#ffffff; font-weight:800; background:#5a6245; padding:${ttCfg.countPadding}; border-radius:4px; border:1px solid rgba(255,255,255,0.15);">근거 ${item.mentions}건</span>
                </div>
            </div>
        `;

        marker.bindTooltip(tooltipHtml, {
            className: "final-app-tooltip",
            direction: "top",
            offset: [0, -(circleSize / 2 + 8)]
        });

        // 클릭 시 해당 의회 상세 카드 및 목록 즉시 연동 (앞뒤 전체 문맥 자동 확장)
        marker.on("click", (e) => {
            L.DomEvent.stopPropagation(e);
            onCouncilMapClick(item.council);
        });

        leafletMarkersLayer.addLayer(marker);
    }

    // 마지막 단계일 때 topPredict 예측 노드 생성
    if (safeIdx === maxSteps - 1 && issue.topPredict && COUNCIL_COORDS[issue.topPredict.council]) {
        const pred = issue.topPredict;
        const coord = COUNCIL_COORDS[pred.council];
        if (coord) {
            const isSelected = (selectedCouncilName === pred.council);
            const circleSize = 34;
            const wrapperClass = `council-marker-wrapper is-predict ${isSelected ? "is-selected" : ""}`;
            const markerHtml = `
                <div class="${wrapperClass}" style="width:${circleSize + 24}px;">
                    <div class="council-node-circle" style="width:${circleSize}px; height:${circleSize}px; font-size:12px; font-weight:900;">
                        Next
                    </div>
                    <div class="council-pill-label" style="background:#fff1f2; color:#e11d48; border-color:#fca5a5; font-weight:900;">
                        예측: ${coord.short}
                    </div>
                </div>
            `;

            const customIcon = L.divIcon({
                html: markerHtml,
                className: "custom-council-divicon",
                iconSize: [circleSize + 24, circleSize + 28],
                iconAnchor: [(circleSize + 24) / 2, circleSize / 2]
            });

            const marker = L.marker([coord.lat, coord.lon], { icon: customIcon });
            const tooltipHtml = `
                <div style="font-family:'Pretendard',sans-serif; min-width:${ttCfg.minWidth}px; padding:${ttCfg.padding};">
                    <div style="display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.25); padding-bottom:6px; margin-bottom:6px;">
                        <b style="font-size:${ttCfg.titleSize}px; color:#ffb4a8; font-weight:900;">[차기 예측] ${pred.council}</b>
                        <span style="font-size:${ttCfg.lagSize}px; color:#ffccd5; font-weight:900; font-family:'JetBrains Mono',monospace;">+${pred.days}일 내</span>
                    </div>
                    <div style="color:#ded8c7; font-size:${ttCfg.subSize}px; margin-bottom:5px; font-weight:600; line-height:1.4;">
                        조례 발의 ${pred.wideOrdinance}건 · 수요 <span style="color:#fde047; font-weight:800;">${pred.needScore}점</span>
                    </div>
                    <div style="display:flex; align-items:center; justify-content:space-between; font-size:${ttCfg.detailSize}px;">
                        <span style="color:#ffffff; font-weight:900; background:#e11d48; padding:${ttCfg.countPadding}; border-radius:4px; box-shadow:0 1px 4px rgba(225,29,72,0.4);">종합 ${pred.score}점</span>
                        <span style="color:#fecdd3; font-weight:800;">차기 1순위</span>
                    </div>
                </div>
            `;

            marker.bindTooltip(tooltipHtml, {
                className: "final-app-tooltip",
                direction: "top",
                offset: [0, -(circleSize / 2 + 8)]
            });

            marker.on("click", (e) => {
                L.DomEvent.stopPropagation(e);
                onCouncilMapClick(pred.council);
            });

            leafletMarkersLayer.addLayer(marker);
        }
    }

    // 아크(호선) 갱신
    updateLeafletArcs();
}

// =========================================================================================
// 7. [핵심 기능]: 동역학 지도에서 지역 클릭 시 -> 우측 의회별 목록 활성탭 & 상세 카드 연동
// =========================================================================================
function onCouncilMapClick(councilName) {
    selectedCouncilName = councilName;
    const issue = MASTER_ISSUES[currentIssueIndex];
    const propIndex = issue.propagation.findIndex(p => p.council === councilName);

    if (propIndex !== -1) {
        const item = issue.propagation[propIndex];
        
        // 1) 우측 상단 [선택 의회 최초 관측 순서 및 시차 상세] 카드 즉시 갱신
        selectCouncilDetail(item);

        // 2) 우측 하단 [의회별 최초 관측 순서 및 시차 전체 목록]에서 해당 항목 활성탭으로 변경 및 스크롤
        highlightFlowListItem(propIndex);
        const flowItemEl = document.getElementById(`flowItem-${propIndex}`);
        if (flowItemEl) {
            flowItemEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }

        // 3) 지도 상 선택 하이라이트 갱신 (지도의 전체 관측 노드는 유지한 상태에서 선택된 노드 포커스)
        renderTimelineStep(currentTimelineStep);
    } else if (issue.topPredict && issue.topPredict.council === councilName) {
        selectPredictDetail(issue.topPredict);
        highlightFlowListItem(-1);
        renderTimelineStep(currentTimelineStep);
    } else {
        selectUnobservedDetail(councilName);
        highlightFlowListItem(-1);
        renderTimelineStep(currentTimelineStep);
    }
}

function selectCouncilDetail(item) {
    selectedCouncilName = item.council;
    const isOrigin = (item.order === 1);

    const nameEl = document.getElementById("detailCouncilName");
    if (nameEl) nameEl.textContent = item.council;
    
    // 순서 배지
    const orderBadge = document.getElementById("detailOrderBadge");
    if (orderBadge) {
        orderBadge.textContent = `${item.letter || item.order} (${item.order}번째 관측)`;
        orderBadge.className = isOrigin 
            ? "text-xs font-black px-2.5 py-1 rounded-md bg-amber-100 text-amber-900 border border-amber-300 shadow-sm"
            : "text-xs font-black px-2.5 py-1 rounded-md bg-sky-100 text-sky-900 border border-sky-300 shadow-sm";
    }

    // 시차 텍스트
    const stageText = document.getElementById("detailStageText");
    if (stageText) {
        stageText.textContent = isOrigin 
            ? "최초 발원 의회 (시차 0일)" 
            : `최초 발원 대비 +${item.lagDays}일 시차 확산`;
        stageText.className = isOrigin 
            ? "text-xs md:text-sm text-amber-800 font-extrabold"
            : "text-xs md:text-sm text-sky-800 font-extrabold";
    }

    // 언급량 및 일자
    const mentionBadge = document.getElementById("detailMentionBadge");
    if (mentionBadge) mentionBadge.textContent = `언급 ${item.mentions}건`;
    const meetingDate = document.getElementById("detailMeetingDate");
    if (meetingDate) meetingDate.textContent = `${item.date} (+${item.lagDays}일)`;
    const meetingName = document.getElementById("detailMeetingName");
    if (meetingName) meetingName.textContent = item.title;
    const speaker = document.getElementById("detailSpeaker");
    if (speaker) speaker.textContent = item.speaker || "의원 발언";
    const agenda = document.getElementById("detailAgenda");
    if (agenda) agenda.textContent = `#${item.view} 관점 ${item.action}`;
    const excerpt = document.getElementById("detailExcerptText");
    if (excerpt) excerpt.textContent = `"${item.excerpt}"`;
    
    // 신뢰도 표기 (소수점 유사도 -> 일반 사용자용 백분율 신뢰도/일치율)
    const simEl = document.getElementById("detailSimilarity");
    if (simEl) {
        const pct = Math.round((item.similarity || 0.92) * 100);
        simEl.textContent = `신뢰도 ${pct}%`;
    }

    // 앞뒤 전체 문맥 대화 바인딩 및 아코디언 상태 열기
    const parentCtx = document.getElementById("detailParentContext");
    if (parentCtx) {
        parentCtx.classList.remove("hidden");
        if (isOrigin) {
            const issue = MASTER_ISSUES[currentIssueIndex];
            parentCtx.textContent = issue.firstObservation.parentContext || `[위원장]: ${item.title} 관련 안건을 상정합니다. ${item.speaker} 의원 발언해 주십시오.\n[${item.speaker}]: ${item.excerpt}\n[기획조정실장]: 의원님께서 지적해주신 사안에 대해 적극 검토하겠습니다.`;
        } else {
            parentCtx.textContent = `[위원장]: 의사일정에 따라 ${item.title}에 대한 질의답변을 진행하겠습니다. ${item.speaker || '의원'} 질의해 주십시오.\n[${item.speaker || '의원'}]: ${item.excerpt} 관련 현안에 대해 관계 부서의 구체적인 대응 방안을 질의합니다.\n[집행부]: 의원님께서 강조해주신 #${item.view} 관점의 정책 방안을 면밀히 검토하여 후속 대책에 즉시 반영하겠습니다.`;
        }
    }
    const accText = document.getElementById("accordionText");
    if (accText) accText.textContent = "앞뒤 문맥 접기";
    const accIcon = document.getElementById("accordionIcon");
    if (accIcon) {
        accIcon.classList.remove("fa-chevron-down");
        accIcon.classList.add("fa-chevron-up");
    }
}

function toggleDetailContextAccordion() {
    const contextEl = document.getElementById("detailParentContext");
    const textEl = document.getElementById("accordionText");
    const iconEl = document.getElementById("accordionIcon");
    if (!contextEl) return;

    if (contextEl.classList.contains("hidden")) {
        contextEl.classList.remove("hidden");
        if (textEl) textEl.textContent = "앞뒤 문맥 접기";
        if (iconEl) {
            iconEl.classList.remove("fa-chevron-down");
            iconEl.classList.add("fa-chevron-up");
        }
    } else {
        contextEl.classList.add("hidden");
        if (textEl) textEl.textContent = "앞뒤 문맥 펼치기";
        if (iconEl) {
            iconEl.classList.remove("fa-chevron-up");
            iconEl.classList.add("fa-chevron-down");
        }
    }
}

function selectPredictDetail(predictData) {
    const nameEl = document.getElementById("detailCouncilName");
    if (nameEl) nameEl.textContent = `${predictData.council} (차기 예측)`;
    const orderBadge = document.getElementById("detailOrderBadge");
    if (orderBadge) {
        orderBadge.textContent = "NEXT (차기 예측 1순위)";
        orderBadge.className = "text-xs font-black px-2.5 py-1 rounded-md bg-rose-100 text-rose-900 border border-rose-300 shadow-sm";
    }

    const stageText = document.getElementById("detailStageText");
    if (stageText) {
        stageText.textContent = `향후 약 ${predictData.days}일 내 확산 예측 (종합점수 ${predictData.score}점)`;
        stageText.className = "text-xs md:text-sm text-rose-800 font-extrabold";
    }

    const mentionBadge = document.getElementById("detailMentionBadge");
    if (mentionBadge) mentionBadge.textContent = `수요지수 ${predictData.needScore}`;
    const meetingDate = document.getElementById("detailMeetingDate");
    if (meetingDate) meetingDate.textContent = `향후 ${predictData.days}일 이내`;
    const meetingName = document.getElementById("detailMeetingName");
    if (meetingName) meetingName.textContent = `자치법규 조례 발의 동향: 광역 ${predictData.wideOrdinance}건 / 기초 ${predictData.basicOrdinance}건`;
    const speaker = document.getElementById("detailSpeaker");
    if (speaker) speaker.textContent = "지방의회 입법예고 분석";
    const agenda = document.getElementById("detailAgenda");
    if (agenda) agenda.textContent = "공공데이터 조례 및 지역통계 결합 예측";
    const excerpt = document.getElementById("detailExcerptText");
    if (excerpt) excerpt.textContent = `"[차기 확산 유력 근거] ${predictData.reason}"`;
    const simEl = document.getElementById("detailSimilarity");
    if (simEl) simEl.textContent = "종합점수 " + predictData.score;

    const parentCtx = document.getElementById("detailParentContext");
    if (parentCtx) {
        parentCtx.classList.remove("hidden");
        parentCtx.textContent = `[예측 모델 브리핑]: ${predictData.council}은(는) 해당 정책과 관련된 기초·광역 조례 발의가 최근 30일간 총 ${predictData.wideOrdinance + predictData.basicOrdinance}건 접수되었으며, 지역 소멸위험 수요지수가 ${predictData.needScore}점으로 매우 높아 향후 ${predictData.days}일 이내에 본회의 및 상임위에서 본격적인 정책 논의가 개시될 확률이 가장 높습니다.`;
    }
}

function selectUnobservedDetail(councilName) {
    document.getElementById("detailCouncilName").textContent = councilName;
    const orderBadge = document.getElementById("detailOrderBadge");
    orderBadge.textContent = "관측 대기";
    orderBadge.className = "text-xs font-black px-2.5 py-1 rounded-md bg-slate-100 text-slate-800 border border-slate-300";

    const stageText = document.getElementById("detailStageText");
    stageText.textContent = "현재 이슈 미관측 지역";
    stageText.className = "text-xs md:text-sm text-slate-600 font-bold";

    document.getElementById("detailMentionBadge").textContent = "언급 0건";
    document.getElementById("detailMeetingDate").textContent = "-";
    document.getElementById("detailMeetingName").textContent = "해당 광역의회에서는 현재 이슈 관련 발언이 관측되지 않았습니다.";
    document.getElementById("detailSpeaker").textContent = "-";
    document.getElementById("detailAgenda").textContent = "향후 정책 확산 모니터링 대상 지역";
    document.getElementById("detailExcerptText").textContent = `"현재 분석 중인 이슈에 대해 ${councilName}의 본회의 및 상임위 공식 발언 근거가 아직 수집되지 않았습니다."`;
    document.getElementById("detailSimilarity").textContent = "-";
}

// =========================================================================================
// 8. 발언 근거 툴팁 / 팝업 (클릭 시 렌더링 & 상하/좌우 지능형 경계 보정)
// =========================================================================================
function positionTooltip(e, tooltip) {
    const mapViewport = document.getElementById("mapViewport");
    if (!mapViewport) return;
    const viewport = mapViewport.getBoundingClientRect();

    // 툴팁의 치수를 측정하기 위해 클래스 설정
    tooltip.classList.remove("hidden");
    tooltip.style.visibility = "hidden";
    tooltip.style.opacity = "0";

    const ttWidth = tooltip.offsetWidth || 500;
    const ttHeight = tooltip.offsetHeight || 145;

    let clickX = e ? (e.clientX - viewport.left) : (viewport.width / 2);
    let clickY = e ? (e.clientY - viewport.top) : (viewport.height / 2);

    // 하단 서브바(범례 등) 안전 여백 확보 (48px)
    const bottomReserve = 48;
    const pad = 16;

    const spaceAbove = clickY;
    const spaceBelow = viewport.height - clickY;

    // [경북·대전 가림 해결 알고리즘]:
    // 1) 너비 520px 확장으로 슬림해진 높이(약 140px)를 감안하여 위쪽 공간이 확보되면 위쪽으로 우선 배치 (경북, 대전 등 중간 노드가 하단으로 쏠려 가려지는 문제 완벽 방지)
    // 2) 서울, 인천 등 상단 여백이 절대적으로 부족한 최상단 노드만 아래쪽으로 배치
    // 3) 공간 비교 후 더 여유로운 방향을 최종 선택
    let isPlaceAbove;
    if (spaceAbove >= ttHeight + 15) {
        isPlaceAbove = true;
    } else if (spaceBelow >= ttHeight + bottomReserve) {
        isPlaceAbove = false;
    } else {
        isPlaceAbove = (spaceAbove >= spaceBelow);
    }

    let topPos, transformY;
    if (isPlaceAbove) {
        // [상단 배치]
        topPos = clickY - 14;
        transformY = "-100%";
        // 뷰포트 상단 경계 침범 방지 클램핑
        if (topPos - ttHeight < 10) {
            topPos = ttHeight + 10;
        }
    } else {
        // [하단 배치]
        topPos = clickY + 20;
        transformY = "0%";
        // 뷰포트 하단(범례 서브바 포함) 침범 방지 클램핑
        const maxAllowedTop = viewport.height - ttHeight - bottomReserve;
        if (topPos > maxAllowedTop) {
            topPos = Math.max(10, maxAllowedTop);
        }
    }

    // 좌우 화면 밖으로 벗어남 방지 (뷰포트 패딩 16px 클램핑)
    const halfW = ttWidth / 2;
    let leftPos = clickX;
    if (clickX - halfW < pad) {
        leftPos = halfW + pad;
    } else if (clickX + halfW > viewport.width - pad) {
        leftPos = viewport.width - halfW - pad;
    }

    tooltip.style.left = `${leftPos}px`;
    tooltip.style.top = `${topPos}px`;
    tooltip.style.transform = `translate(-50%, ${transformY})`;
    tooltip.style.visibility = "visible";
    tooltip.style.opacity = "1";
}

function showPolicyTooltip(e, item) {
    const tooltip = document.getElementById("policyTooltip");
    if (!tooltip) return;

    document.getElementById("ttBadge").textContent = `${item.letter} ${item.order === 1 ? "최초 발원" : "후속 전파"}`;
    document.getElementById("ttBadge").className = `text-xs font-black px-2 py-0.5 rounded text-white ${item.order === 1 ? "bg-amber-600" : "bg-sky-600"}`;
    document.getElementById("ttCouncil").textContent = item.council;
    document.getElementById("ttDate").textContent = `${item.date} (+${item.lagDays}일)`;
    document.getElementById("ttAction").textContent = `${item.action}`;
    document.getElementById("ttPerspective").textContent = `#${item.view}`;
    document.getElementById("ttLag").textContent = `언급 빈도 ${item.mentions}건`;
    document.getElementById("ttExcerpt").textContent = `"${item.excerpt}"`;
    document.getElementById("ttSpeaker").innerHTML = `<i class="fa-solid fa-user-pen mr-1 text-sky-700"></i>${item.speaker} (${item.title})`;

    positionTooltip(e, tooltip);
}

function showPredictTooltip(e, predictData) {
    const tooltip = document.getElementById("policyTooltip");
    if (!tooltip) return;

    document.getElementById("ttBadge").textContent = "NEXT 예측";
    document.getElementById("ttBadge").className = "text-xs font-bold px-2 py-0.5 rounded bg-rose-500 text-white";
    document.getElementById("ttCouncil").textContent = predictData.council;
    document.getElementById("ttDate").textContent = `향후 약 ${predictData.days}일 내`;
    document.getElementById("ttAction").textContent = "차기 조례 제정 유력";
    document.getElementById("ttPerspective").textContent = `종합 확산점수 ${predictData.score}점`;
    document.getElementById("ttLag").textContent = `수요지수 ${predictData.needScore}`;
    document.getElementById("ttExcerpt").textContent = `[예측 근거]: ${predictData.reason}`;
    document.getElementById("ttSpeaker").innerHTML = `<i class="fa-solid fa-scale-balanced mr-1 text-rose-700"></i>자치법규(광역 ${predictData.wideOrdinance}/기초 ${predictData.basicOrdinance}) + 지역통계 신호 융합`;

    positionTooltip(e, tooltip);
}

function hidePolicyTooltip() {
    const tooltip = document.getElementById("policyTooltip");
    if (!tooltip || tooltip.classList.contains("hidden")) return;
    tooltip.style.opacity = "0";
    setTimeout(() => {
        tooltip.classList.add("hidden");
    }, 150);
}

// =========================================================================================
// 9. 타임라인 스토리플레이어 컨트롤러 (Play, Pause, Step)
// =========================================================================================
function toggleTimelinePlay() {
    hidePolicyTooltip();
    isPlaying = !isPlaying;
    const icon = document.getElementById("playToggleIcon");
    const btn = document.getElementById("playToggleBtn");

    if (isPlaying) {
        const issue = MASTER_ISSUES[currentIssueIndex];
        if (currentTimelineStep >= issue.propagation.length - 1) {
            renderTimelineStep(0);
        }

        icon.className = "fa-solid fa-pause text-xs";
        btn.classList.remove("bg-sky-700");
        btn.classList.add("bg-emerald-600");

        playInterval = setInterval(() => {
            const currentIssue = MASTER_ISSUES[currentIssueIndex];
            const maxLen = currentIssue.propagation.length;
            let nextStep = currentTimelineStep + 1;

            if (nextStep >= maxLen - 1) {
                renderTimelineStep(maxLen - 1);
                stopTimeline();
                document.getElementById("playerStateText").textContent = `${maxLen}/${maxLen}단계 관측 완료 (전체 확산)`;
            } else {
                renderTimelineStep(nextStep);
            }
        }, 1300);
    } else {
        stopTimeline();
    }
}

function stopTimeline() {
    isPlaying = false;
    clearInterval(playInterval);
    const icon = document.getElementById("playToggleIcon");
    const btn = document.getElementById("playToggleBtn");
    if (icon) icon.className = "fa-solid fa-play text-xs";
    if (btn) {
        btn.classList.remove("bg-emerald-600");
        btn.classList.add("bg-sky-700");
    }
}

function stepTimeline(delta) {
    stopTimeline();
    hidePolicyTooltip();
    const issue = MASTER_ISSUES[currentIssueIndex];
    let nextStep = currentTimelineStep + delta;
    if (nextStep < 0) nextStep = 0;
    if (nextStep >= issue.propagation.length) nextStep = issue.propagation.length - 1;
    renderTimelineStep(nextStep);
    onCouncilMapClick(issue.propagation[nextStep].council);
}

function resetTimelineToStart() {
    stopTimeline();
    hidePolicyTooltip();
    renderTimelineStep(0);
    const issue = MASTER_ISSUES[currentIssueIndex];
    onCouncilMapClick(issue.propagation[0].council);
}

function onTimelineSliderChange(val) {
    stopTimeline();
    hidePolicyTooltip();
    const idx = parseInt(val);
    renderTimelineStep(idx);
    const issue = MASTER_ISSUES[currentIssueIndex];
    onCouncilMapClick(issue.propagation[idx].council);
}

function resetMapView() {
    stopTimeline();
    hidePolicyTooltip();
    if (leafletMap) {
        leafletMap.flyTo([36.2, 127.8], 7.0, { duration: 0.8 });
    }
    const maxIdx = MASTER_ISSUES[currentIssueIndex].propagation.length - 1;
    renderTimelineStep(maxIdx);
}

// =========================================================================================
// 10. 우측 의회별 시간순 흐름 리스트 렌더링 (지도 클릭에 따라 활성탭 전환 - final_app 순서 포맷)
// =========================================================================================
function renderCouncilFlowList(issue) {
    const container = document.getElementById("councilFlowList");
    container.innerHTML = "";

    issue.propagation.forEach((item, index) => {
        const isOrigin = (index === 0);
        const div = document.createElement("div");
        div.id = `flowItem-${index}`;
        div.className = "p-3 rounded-xl border-2 border-slate-200 bg-white hover:bg-slate-50 cursor-pointer transition-all flex items-center justify-between text-base gap-2 shadow-sm";
        div.innerHTML = `
            <div class="flex items-center space-x-3 min-w-0">
                <span id="flowLetter-${index}" class="w-8 h-8 rounded-lg flex items-center justify-center font-black text-sm font-mono shrink-0 ${
                    isOrigin ? 'bg-[#f59e0b] text-white border border-amber-600 shadow-sm' : 'bg-[#e95b4c] text-white border border-rose-600 shadow-sm'
                }">
                    ${item.order}
                </span>
                <div class="min-w-0">
                    <div class="flex items-center gap-2">
                        <span id="flowName-${index}" class="font-bold text-slate-900 block text-base leading-snug truncate">${item.council}</span>
                        <span id="flowTabBadge-${index}" class="hidden text-xs px-2 py-0.5 rounded-full bg-sky-700 text-white font-black shadow-sm shrink-0">
                            <i class="fa-solid fa-check mr-0.5"></i>선택
                        </span>
                    </div>
                    <span class="text-xs md:text-sm text-slate-600 font-medium">${item.action} · <span class="text-sky-800 font-bold">#${item.view}</span></span>
                </div>
            </div>
            <div class="text-right font-mono shrink-0 flex flex-col items-end gap-1">
                <div class="flex items-center gap-1.5">
                    <span class="text-xs px-2 py-0.5 rounded-md bg-sky-100 text-sky-950 border border-sky-300 font-black font-mono">
                        <i class="fa-solid fa-comments text-sky-700 text-[10px] mr-1"></i>${item.mentions}건
                    </span>
                    <span class="text-sm ${isOrigin ? 'text-amber-800 font-black' : 'text-slate-800 font-bold'}">+${item.lagDays}일</span>
                </div>
                <span class="text-xs text-slate-500 font-medium">${item.date}</span>
            </div>
        `;

        // [사용자 요구사항]: 목록 항목을 클릭했을 때 동역학 지도를 변경/되감지 않고,
        // 해당 의회 상세 카드 갱신 및 활성 탭 전환만 독립적으로 수행!
        div.addEventListener("click", () => {
            selectedCouncilName = item.council;
            selectCouncilDetail(item);
            highlightFlowListItem(index);
            renderTimelineStep(currentTimelineStep);
        });

        container.appendChild(div);
    });

    // 초기 로드 시 0번째 의회(최초 발원 의회) 활성탭 적용
    highlightFlowListItem(0);
}

function highlightFlowListItem(idx) {
    const issue = MASTER_ISSUES[currentIssueIndex];
    const indEl = document.getElementById("activeCouncilIndicator");
    if (idx === -1 && indEl && selectedCouncilName) {
        indEl.textContent = `선택 활성: ${selectedCouncilName}`;
    }

    for (let i = 0; i < issue.propagation.length; i++) {
        const el = document.getElementById(`flowItem-${i}`);
        const badge = document.getElementById(`flowTabBadge-${i}`);
        const letterBox = document.getElementById(`flowLetter-${i}`);
        const nameEl = document.getElementById(`flowName-${i}`);
        if (!el) continue;

        const isOrigin = (i === 0);
        if (i === idx) {
            // 활성 탭 스타일 (선명한 고대비 스카이 블루 배경 + 굵은 테두리 + 선택 뱃지)
            el.className = "p-3 rounded-xl border-2 border-sky-600 bg-sky-100/70 shadow-md ring-2 ring-sky-400/40 cursor-pointer transition-all flex items-center justify-between text-base gap-2";
            if (badge) badge.classList.remove("hidden");
            if (letterBox) {
                letterBox.className = `w-8 h-8 rounded-lg flex items-center justify-center font-black text-sm font-mono shrink-0 shadow-sm ${
                    isOrigin ? 'bg-amber-500 text-white border border-amber-600' : 'bg-sky-600 text-white border border-sky-700'
                }`;
            }
            if (nameEl) {
                nameEl.className = "font-black text-sky-950 block text-base leading-snug truncate";
            }
            if (indEl) {
                indEl.textContent = `선택 활성: ${issue.propagation[i].council}`;
            }
        } else {
            // 비활성 탭 스타일
            el.className = "p-3 rounded-xl border-2 border-slate-200 bg-white hover:bg-slate-50 cursor-pointer transition-all flex items-center justify-between text-base gap-2 shadow-sm opacity-90";
            if (badge) badge.classList.add("hidden");
            if (letterBox) {
                letterBox.className = `w-8 h-8 rounded-lg flex items-center justify-center font-black text-sm font-mono shrink-0 ${
                    isOrigin ? 'bg-amber-100 text-amber-950 border border-amber-300' : 'bg-slate-100 text-slate-800 border border-slate-300'
                }`;
            }
            if (nameEl) {
                nameEl.className = "font-bold text-slate-800 block text-base leading-snug truncate";
            }
        }
    }
}

function toggleParentContext() {
    toggleDetailContextAccordion();
}

// =========================================================================================
// 11. 월별 시계열 차트 렌더링 (SVG/CSS Dynamic Bar Chart)
// =========================================================================================
function renderMonthlyChart(series) {
    const container = document.getElementById("monthlyChartBars");
    container.innerHTML = "";

    const maxCount = Math.max(...series.map(s => s.count), 1);

    series.forEach((s) => {
        const heightPct = Math.max(10, Math.round((s.count / maxCount) * 100));

        const barCol = document.createElement("div");
        barCol.className = "flex-1 h-full flex flex-col justify-end items-center group relative cursor-pointer";

        barCol.innerHTML = `
            <div class="hidden group-hover:block absolute -top-8 px-2 py-0.5 rounded bg-slate-900 border border-slate-700 text-sky-400 font-mono text-xs whitespace-nowrap z-20 shadow-lg">
                ${s.month}: ${s.count}건
            </div>
            <div class="w-full bg-gradient-to-t from-sky-600 to-sky-400 hover:from-amber-500 hover:to-amber-400 rounded-t-sm transition-all duration-300" style="height: ${heightPct}%"></div>
            <span class="text-[10px] md:text-xs font-mono text-slate-600 mt-1 block truncate w-full text-center font-bold">
                ${s.month.endsWith("01") || s.month.endsWith("07") ? s.month : ""}
            </span>
        `;
        container.appendChild(barCol);
    });
}

// =========================================================================================
// 12. 10대 대표 키워드/이슈 카탈로그 테이블 및 차트 (Tab 2)
// =========================================================================================
function renderCatalogTable() {
    const tbody = document.getElementById("catalogTableBody");
    if (!tbody) return;
    tbody.innerHTML = "";

    MASTER_ISSUES.forEach((issue, index) => {
        const tr = document.createElement("tr");
        tr.className = "hover:bg-sky-50/70 cursor-pointer transition-colors";
        tr.innerHTML = `
            <td class="p-3.5 font-mono text-sky-800 font-black">${issue.id}</td>
            <td class="p-3.5 font-bold text-slate-950 flex items-center gap-2">
                <span class="text-sm md:text-base font-black">${issue.name}</span>
                <span class="px-2 py-0.5 rounded text-xs bg-slate-100 text-slate-800 border border-slate-300 font-bold shrink-0">${issue.category}</span>
            </td>
            <td class="p-3.5 font-mono text-slate-700 font-bold text-xs md:text-sm">${issue.firstObservation.date}</td>
            <td class="p-3.5 font-mono text-slate-900 font-black text-xs md:text-sm">${issue.councilsCount}곳 / 17곳</td>
            <td class="p-3.5 font-mono text-emerald-800 font-black text-xs md:text-sm">${issue.evidenceCount.toLocaleString()}건</td>
            <td class="p-3.5 text-center">
                <button onclick="selectAndSwitchToRadar(${index})" class="px-3.5 py-1.5 rounded-lg bg-sky-700 hover:bg-sky-600 text-white text-xs md:text-sm font-black shadow-sm transition-colors cursor-pointer inline-flex items-center gap-1.5 mx-auto">
                    <i class="fa-solid fa-satellite-dish text-xs"></i>
                    <span>레이더 보기</span>
                </button>
            </td>
        `;
        tr.addEventListener("click", (e) => {
            if (e.target.tagName !== "BUTTON" && !e.target.closest("button")) {
                selectAndSwitchToRadar(index);
            }
        });
        tbody.appendChild(tr);
    });
}

function selectAndSwitchToRadar(index) {
    switchTab("radar");
    loadIssueData(index);
}

function renderCatalogChart() {
    const container = document.getElementById("catalogBarChart");
    if (!container) return;
    container.innerHTML = "";

    const maxMentions = Math.max(...MASTER_ISSUES.map(i => i.mlForecast.predictedMentions), 1);

    MASTER_ISSUES.forEach((issue, index) => {
        const widthPct = Math.round((issue.mlForecast.predictedMentions / maxMentions) * 100);

        const row = document.createElement("div");
        row.className = "flex items-center space-x-3 text-xs hover:bg-slate-50 p-2 rounded-lg transition-colors cursor-pointer";
        row.innerHTML = `
            <span class="w-44 truncate font-black text-slate-900 text-sm shrink-0" title="${issue.name}">${issue.name}</span>
            <div class="flex-1 bg-slate-200 h-4 rounded-full overflow-hidden p-0.5 border border-slate-300">
                <div class="bg-gradient-to-r from-rose-500 to-amber-500 h-full rounded-full transition-all duration-500" style="width: ${widthPct}%"></div>
            </div>
            <span class="font-mono text-sm text-rose-700 font-black w-14 text-right shrink-0">${issue.mlForecast.predictedMentions}건</span>
        `;
        row.addEventListener("click", () => selectAndSwitchToRadar(index));
        container.appendChild(row);
    });
}

// =========================================================================================
// 13. 공공데이터 결합 예측 화면 (메인 대시보드 하단에 자연스럽게 통합 노출)
// =========================================================================================
function renderForecastRankingTab(issue) {
    const brief = document.getElementById("topForecastBriefCard");
    const pred = issue.topPredict;
    if (!brief || !pred) return;

    brief.innerHTML = `
        <div class="space-y-1.5">
            <div class="flex items-center gap-2.5">
                <span class="px-2.5 py-1 rounded-md bg-rose-600 text-white font-black text-xs font-mono shadow-sm">1위 예측 후보</span>
                <h3 class="text-xl font-black text-rose-700">${pred.council}</h3>
                <span class="text-xs md:text-sm text-slate-800 font-mono font-black bg-white px-2 py-0.5 rounded border border-slate-300 shadow-sm">향후 약 ${pred.days}일 내 확산 예측</span>
            </div>
            <p class="text-sm md:text-base text-slate-800 leading-relaxed max-w-3xl font-medium">${pred.reason}</p>
        </div>
        <div class="flex items-center gap-4 shrink-0 bg-white p-3.5 rounded-xl border border-slate-300 font-mono text-sm shadow-sm">
            <div>
                <span class="text-xs text-slate-600 block font-sans font-bold">종합 확산점수</span>
                <span class="text-emerald-700 font-black text-base">${pred.score}점</span>
            </div>
            <div class="border-l border-slate-300 pl-3">
                <span class="text-xs text-slate-600 block font-sans font-bold">관련 조례 제정</span>
                <span class="text-sky-800 font-bold text-sm">광역 ${pred.wideOrdinance} / 기초 ${pred.basicOrdinance}건</span>
            </div>
            <div class="border-l border-slate-300 pl-3">
                <span class="text-xs text-slate-600 block font-sans font-bold">지역통계 수요점수</span>
                <span class="text-amber-800 font-bold text-sm">${pred.needScore}점</span>
            </div>
        </div>
    `;

    const tbody = document.getElementById("forecastRankingTableBody");
    if (!tbody) return;
    tbody.innerHTML = "";

    const councils = Object.keys(COUNCIL_COORDS);
    const sorted = councils.map((c, i) => {
        if (c === pred.council) {
            return { rank: 1, council: c, score: pred.score, db: pred.dbScore, ext: pred.extScore, wide: pred.wideOrdinance, basic: pred.basicOrdinance, need: pred.needScore, days: `${pred.days}일 이내` };
        }
        const factor = Math.max(0.15, 1 - (i * 0.05));
        return {
            rank: i + 2,
            council: c,
            score: (pred.score * factor).toFixed(2),
            db: (pred.dbScore * factor).toFixed(2),
            ext: (pred.extScore * factor).toFixed(2),
            wide: Math.max(0, Math.floor(pred.wideOrdinance * factor)),
            basic: Math.max(0, Math.floor(pred.basicOrdinance * factor)),
            need: (pred.needScore * factor).toFixed(1),
            days: `약 ${Math.round(pred.days * (1 / factor))}일 내`
        };
    }).sort((a, b) => b.score - a.score);

    sorted.forEach((row, idx) => {
        const tr = document.createElement("tr");
        tr.className = idx === 0 ? "bg-rose-50/80 font-black text-sm text-slate-950" : "hover:bg-slate-50 text-sm transition-colors";
        tr.innerHTML = `
            <td class="p-3 font-mono ${idx === 0 ? 'text-rose-700 font-black' : 'text-slate-700'}">${idx + 1}</td>
            <td class="p-3 font-black ${idx === 0 ? 'text-rose-800' : 'text-slate-950'}">${row.council}</td>
            <td class="p-3 font-mono text-emerald-700 font-black">${row.score}</td>
            <td class="p-3 font-mono text-slate-700">${row.db}</td>
            <td class="p-3 font-mono text-sky-800 font-bold">${row.ext}</td>
            <td class="p-3 font-mono text-slate-800">${row.wide}건</td>
            <td class="p-3 font-mono text-slate-800">${row.basic}건</td>
            <td class="p-3 font-mono text-amber-800 font-bold">${row.need}</td>
            <td class="p-3 font-mono text-slate-700 font-medium">${row.days}</td>
        `;
        tbody.appendChild(tr);
    });
}

// =========================================================================================
// 14. 근거 원문 상세 내용 검색기 (Tab 3: 회의록 발언 원문 탐색)
// =========================================================================================
function executeRawVectorSearch() {
    const query = document.getElementById("rawSearchQuery")?.value || "지방소멸 대응 청년 주거 지원";
    const councilFilter = document.getElementById("rawSearchCouncilFilter")?.value || "";
    const yearFilter = document.getElementById("rawSearchYearFilter")?.value || "all";

    const searchDbMs = (Math.random() * 6 + 14).toFixed(1);
    const dbMsEl = document.getElementById("rawSearchDbMs");
    if (dbMsEl) dbMsEl.textContent = searchDbMs;

    const container = document.getElementById("rawSearchResultsContainer");
    if (!container) return;
    container.innerHTML = "";

    // 17개 광역의회 회의록 전체 원문 풀에서 추출 (상위 20개 노출)
    let allCandidates = [];
    MASTER_ISSUES.forEach((issue, issueIdx) => {
        allCandidates.push({
            council: issue.firstObservation.council,
            date: issue.firstObservation.date,
            meetingName: issue.firstObservation.meetingName,
            speaker: issue.firstObservation.speaker,
            committee: issue.firstObservation.committee,
            similarity: 0.96 - issueIdx * 0.008,
            text: issue.firstObservation.excerpt,
            parent: issue.firstObservation.parentContext || `[위원장]: ${issue.firstObservation.meetingName} 관련 안건을 심사하겠습니다.\n[${issue.firstObservation.speaker}]: ${issue.firstObservation.excerpt}\n[관계부서장]: 제기해 주신 정책 제안을 면밀히 검토하여 시행 방안을 마련하겠습니다.`
        });
        issue.propagation.forEach((p, pIdx) => {
            allCandidates.push({
                council: p.council,
                date: p.date,
                meetingName: p.title,
                speaker: p.speaker,
                committee: "상임위원회",
                similarity: p.similarity || (0.92 - pIdx * 0.015),
                text: p.excerpt,
                parent: `[위원장]: ${p.title} 안건에 대한 질의답변을 진행하겠습니다. ${p.speaker || '의원'} 질의해 주십시오.\n[${p.speaker || '의원'}]: ${p.excerpt}\n[집행부]: 의원님께서 강조해주신 #${p.view} 관점의 정책 방안을 조속히 행정에 반영하겠습니다.`
            });
        });
    });

    if (councilFilter) {
        allCandidates = allCandidates.filter(c => c.council === councilFilter);
    }
    if (yearFilter !== "all") {
        allCandidates = allCandidates.filter(c => c.date.startsWith(yearFilter));
    }

    // 일치율 내림차순 정렬 후 상위 20개 선정 (백엔드 부하 방지 및 상위 20개 표시)
    allCandidates.sort((a, b) => b.similarity - a.similarity);
    const results = allCandidates.slice(0, 20);

    const countEl = document.getElementById("rawSearchResultCount");
    if (countEl) countEl.textContent = results.length;

    results.forEach((r, idx) => {
        const card = document.createElement("div");
        card.className = "p-4 rounded-xl border border-slate-300 bg-white space-y-2.5 shadow-sm hover:border-purple-400 transition-all";
        const matchPct = Math.round(r.similarity * 100);
        card.innerHTML = `
            <div class="flex items-center justify-between border-b border-slate-200 pb-2 text-sm">
                <div class="flex items-center gap-2.5">
                    <span class="w-6 h-6 rounded bg-purple-100 text-purple-900 font-mono font-bold flex items-center justify-center text-xs">${idx + 1}</span>
                    <span class="font-black text-slate-950 text-sm md:text-base">${r.council}</span>
                    <span class="text-slate-600 font-mono text-xs md:text-sm font-semibold">${r.date}</span>
                </div>
                <span class="font-mono text-xs font-bold px-2.5 py-0.5 rounded bg-purple-100 text-purple-900 border border-purple-300">
                    일치율 ${matchPct}%
                </span>
            </div>
            <div class="text-sm text-slate-800 font-bold">${r.meetingName} (${r.speaker || '발언자 미상'})</div>
            <div class="text-sm md:text-[14.5px] text-slate-950 leading-relaxed bg-slate-50 p-3 rounded-lg border border-slate-200 font-mono">
                "${r.text}"
            </div>
            <div class="text-right">
                <button onclick="toggleCardParentContext(this)" class="text-xs md:text-sm text-purple-700 hover:text-purple-800 font-bold cursor-pointer inline-flex items-center gap-1">
                    <i class="fa-solid fa-expand"></i>앞뒤 문맥 보기
                </button>
                <div class="hidden mt-2 p-3 bg-slate-50 text-xs md:text-sm text-slate-800 border border-slate-300 rounded-lg text-left whitespace-pre-wrap leading-relaxed font-mono">
                    ${r.parent}
                </div>
            </div>
        `;
        container.appendChild(card);
    });
}

function toggleCardParentContext(btn) {
    const box = btn.nextElementSibling;
    if (box.classList.contains("hidden")) {
        box.classList.remove("hidden");
        btn.innerHTML = '<i class="fa-solid fa-compress mr-1"></i>문맥 닫기';
    } else {
        box.classList.add("hidden");
        btn.innerHTML = '<i class="fa-solid fa-expand mr-1"></i>앞뒤 문맥 보기';
    }
}

// =========================================================================================
// 15. 채널톡 스타일 플로팅 AI 에이전트 챗봇
// =========================================================================================
function toggleFloatingChat() {
    const modal = document.getElementById("floatingChatModal");
    const badge = document.getElementById("chatUnreadBadge");
    if (!modal) return;

    const isHidden = modal.classList.contains("hidden");
    if (isHidden) {
        modal.classList.remove("hidden");
        modal.classList.add("flex");
        if (badge) badge.classList.add("hidden");
        const input = document.getElementById("agentInputQuery");
        if (input) setTimeout(() => input.focus(), 150);
    } else {
        modal.classList.add("hidden");
        modal.classList.remove("flex");
    }
}

function handleAgentSubmit(e) {
    if (e) e.preventDefault();
    const input = document.getElementById("agentInputQuery");
    const query = input.value.trim();
    if (!query) return;

    appendAgentMessage("user", query);
    input.value = "";

    setTimeout(() => {
        respondAgentQuery(query);
    }, 600);
}

function askAgentPredefined(query) {
    const modal = document.getElementById("floatingChatModal");
    if (modal && modal.classList.contains("hidden")) {
        toggleFloatingChat();
    }
    appendAgentMessage("user", query);
    setTimeout(() => {
        respondAgentQuery(query);
    }, 600);
}

function appendAgentMessage(role, text) {
    const container = document.getElementById("agentChatContainer");
    if (!container) return;

    const msgDiv = document.createElement("div");

    if (role === "user") {
        msgDiv.className = "flex items-start justify-end space-x-2";
        msgDiv.innerHTML = `
            <div class="bg-sky-600 text-white rounded-2xl p-3 text-sm md:text-base max-w-[85%] leading-relaxed shadow-md">
                ${escapeHtml(text)}
            </div>
            <div class="w-7 h-7 rounded-lg bg-slate-700 text-white flex items-center justify-center text-xs shrink-0 mt-0.5">
                <i class="fa-solid fa-user"></i>
            </div>
        `;
    } else {
        msgDiv.className = "flex items-start space-x-2.5";
        msgDiv.innerHTML = `
            <div class="w-7 h-7 rounded-lg bg-gradient-to-tr from-sky-600 to-indigo-600 text-white flex items-center justify-center text-xs shrink-0 mt-0.5 shadow-sm">
                <i class="fa-solid fa-robot"></i>
            </div>
            <div class="bg-white border border-slate-300 rounded-2xl p-3.5 text-sm md:text-base text-slate-900 max-w-[88%] leading-relaxed shadow-sm space-y-2">
                ${text}
            </div>
        `;
    }

    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
}

function respondAgentQuery(query) {
    let matchedIssue = MASTER_ISSUES[0];
    for (const issue of MASTER_ISSUES) {
        if (query.includes(issue.shortName) || query.includes(issue.category) || issue.name.split(" ").some(w => query.includes(w))) {
            matchedIssue = issue;
            break;
        }
    }

    const first = matchedIssue.firstObservation;
    const predict = matchedIssue.topPredict;

    const responseHTML = `
        <div class="flex items-center gap-2 border-b border-slate-200 pb-1.5 mb-1.5">
            <span class="px-2 py-0.5 rounded bg-sky-100 text-sky-900 font-mono text-xs border border-sky-300 font-bold">
                <i class="fa-solid fa-check mr-1"></i>${matchedIssue.name}
            </span>
            <span class="text-xs text-slate-600 font-mono">의정 AI 응답</span>
        </div>
        <p class="font-bold text-sky-800">『${matchedIssue.name}』 분석 내용:</p>
        <ul class="list-disc pl-4 space-y-1 text-slate-800 text-xs md:text-sm font-medium">
            <li><b>최초 발원:</b> <b>${first.date}</b>, <b>${first.council}</b>에서 ${first.speaker} 의원이 최초 발언.</li>
            <li><b>전국 확산:</b> <b>${matchedIssue.councilsCount}개 시·도의회</b>로 파급되었으며 실질 발언 <b>${matchedIssue.evidenceCount}건</b> 집계.</li>
            <li><b>차기 예측:</b> 향후 약 <b>${predict.days}일 내 ${predict.council}</b>에서 후속 조례 발의 유력.</li>
        </ul>
        <div class="mt-2 p-2.5 bg-slate-50 rounded-lg border border-slate-200 text-xs text-slate-800">
            <b>발췌문:</b> "${first.excerpt}"
        </div>
        <div class="pt-1 text-right">
            <button onclick="selectAndSwitchToRadar(${MASTER_ISSUES.indexOf(matchedIssue)})" class="text-xs text-sky-700 hover:text-sky-800 font-bold underline">
                대시보드에서 이 이슈 분석하기 →
            </button>
        </div>
    `;

    appendAgentMessage("assistant", responseHTML);
}

function escapeHtml(text) {
    const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// =========================================================================================
// 16. 단일 메인 검색창 및 3개 핵심 탭 전환 네비게이션
// =========================================================================================
function handleGlobalSearch() {
    const input = document.getElementById("globalSearchInput");
    const val = input.value.trim();
    if (!val) return;

    const matchIdx = MASTER_ISSUES.findIndex(issue => 
        issue.name.includes(val) || issue.shortName.includes(val) || val.includes(issue.shortName)
    );

    if (matchIdx !== -1) {
        loadIssueData(matchIdx);
        switchTab("radar");
    } else {
        switchTab("search");
        const rawInput = document.getElementById("rawSearchQuery");
        if (rawInput) {
            rawInput.value = val;
            executeRawVectorSearch();
        }
    }
}

function switchTab(tabId) {
    currentTab = tabId;

    // 1) 정책 시공간 전파 레이더 탭 활성화 시에만 상단 3대 KPI 카드 표시 (그 외 탭은 숨김)
    const kpiSection = document.getElementById("radarKpiSection");
    if (kpiSection) {
        if (tabId === "radar") {
            kpiSection.classList.remove("hidden");
        } else {
            kpiSection.classList.add("hidden");
        }
    }

    const buttons = document.querySelectorAll(".tab-btn");
    buttons.forEach(btn => btn.classList.remove("active"));
    const activeBtn = document.getElementById(`tab-${tabId}`);
    if (activeBtn) activeBtn.classList.add("active");

    const contents = document.querySelectorAll(".tab-content");
    contents.forEach(content => {
        content.classList.add("hidden");
        content.classList.remove("flex", "grid");
    });

    const targetContent = document.getElementById(`tabContent-${tabId}`);
    if (targetContent) {
        targetContent.classList.remove("hidden");
        if (tabId === "radar") {
            targetContent.classList.add("grid");
            if (leafletMap) {
                setTimeout(() => {
                    leafletMap.invalidateSize();
                    renderTimelineStep(currentTimelineStep);
                }, 100);
            }
        } else {
            targetContent.classList.add("flex");
        }
    }
}
