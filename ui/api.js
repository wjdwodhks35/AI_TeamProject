// 백엔드 API 호출 계층. 화면 코드는 fetch 를 직접 쓰지 않고 이 파일만 사용한다.
// 요청·응답 형식은 docs/API_CONTRACT.md 와 /openapi.json 을 따른다.

/**
 * API 주소.
 *
 * - 배포(`./deploy.sh`): UI 와 API 가 같은 출처(8010)라서 빈 문자열 → 상대 경로로 호출한다.
 * - 개발: UI 를 `python -m http.server 5173` 로 따로 띄운 경우 백엔드(8000)를 가리킨다.
 * - `?api=http://호스트:포트` 쿼리로 직접 덮어쓸 수 있다.
 */
const DEV_API_BASE = "http://127.0.0.1:8000";
const STATIC_DEV_PORTS = new Set(["3000", "5173"]);

function resolveApiBase() {
  const override = new URLSearchParams(location.search).get("api");
  if (override) return override.replace(/\/+$/, "");
  if (location.protocol === "file:") return DEV_API_BASE;
  if (STATIC_DEV_PORTS.has(location.port)) return DEV_API_BASE;
  return ""; // 같은 출처
}

export const API_BASE = resolveApiBase();

/** 오류 메시지에 쓸 표시용 주소. */
export const API_ORIGIN_LABEL = API_BASE || location.origin;

/** 서울 25개 자치구. src/apartment_ml/schemas.py 의 SEOUL_DISTRICTS 와 같아야 한다. */
export const SEOUL_DISTRICTS = [
  "강남구", "강동구", "강북구", "강서구", "관악구",
  "광진구", "구로구", "금천구", "노원구", "도봉구",
  "동대문구", "동작구", "마포구", "서대문구", "서초구",
  "성동구", "성북구", "송파구", "양천구", "영등포구",
  "용산구", "은평구", "종로구", "중구", "중랑구",
];

/**
 * 백엔드가 돌려준 오류를 담는다.
 * 백엔드는 모든 오류를 { detail: { code, message, issues? } } 형태로 보낸다.
 */
export class ApiError extends Error {
  constructor({ status, code, message, issues = [], requestId = null }) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.issues = issues;
    this.requestId = requestId;
  }
}

async function request(path, { method = "GET", body } = {}) {
  let response;
  try {
    response = await fetch(API_BASE + path, {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    // fetch 자체가 실패하면 서버가 꺼져 있거나 CORS 가 막힌 경우다.
    throw new ApiError({
      status: 0,
      code: "NETWORK_ERROR",
      message: `${API_ORIGIN_LABEL} 에 연결할 수 없습니다. 서버가 실행 중인지, 이 페이지를 http:// 로 열었는지 확인하세요.`,
    });
  }

  const requestId = response.headers.get("X-Request-ID");

  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null; // 본문이 없거나 JSON 이 아닌 응답
  }

  if (!response.ok) {
    const detail = payload?.detail;
    const info = detail !== null && typeof detail === "object" ? detail : {};
    throw new ApiError({
      status: response.status,
      code: info.code ?? `HTTP_${response.status}`,
      message:
        info.message ??
        (typeof detail === "string" ? detail : "요청을 처리할 수 없습니다."),
      issues: Array.isArray(info.issues) ? info.issues : [],
      requestId,
    });
  }

  return payload;
}

/**
 * GET /locations/{자치구} — 그 자치구의 동·단지 목록.
 *
 * 단지 식별자(`complex_id`)는 국토부 단지 일련번호라서 사용자가 타이핑할 수 없다.
 * 이 목록으로 단지명을 보여주고 일련번호를 보낸다. 자치구를 고를 때 한 번만
 * 호출하면 동·단지 두 선택칸을 모두 채울 수 있다.
 */
export function getLocations(district) {
  return request(`/locations/${encodeURIComponent(district)}`);
}

/**
 * POST /forecast/price — 향후 N개월 예상 거래가격.
 *
 * 회귀가 시세 대비 상대가격을 학습해서, 미래 가격은 시장지수만 내다보면 된다.
 * 응답의 low_krw~high_krw 는 80% 구간이고 지평이 길수록 넓어진다.
 */
export function forecastPrice(features, months = 12) {
  return request("/forecast/price", { method: "POST", body: { ...features, months } });
}

/** GET /health — 서버 버전과 모델별 준비 여부 */
export function getHealth() {
  return request("/health");
}

/**
 * POST /predict/price — 거래 조건만 보낸다. 실제 거래가격을 넣으면 422 가 된다.
 *
 * `complex_id` 는 형식상 선택이지만 사실상 필수다. 없으면 모델이 자치구 수준으로만
 * 추정해서 평균오차가 1.41억 -> 4.18억으로 늘어난다 (docs/API_CONTRACT.md).
 */
export function predictPrice(features) {
  return request("/predict/price", { method: "POST", body: features });
}

/** POST /predict/anomaly — 거래 조건 + 실제 거래가격(필수) */
export function predictAnomaly(features, priceKrw) {
  return request("/predict/anomaly", {
    method: "POST",
    body: { ...features, price_krw: priceKrw },
  });
}

/** POST /scenario/interest-rate — features.base_rate 는 비교 금리로 덮어써진다. */
export function scenarioInterestRate(features, rateBefore, rateAfter) {
  return request("/scenario/interest-rate", {
    method: "POST",
    body: { features, base_rate_before: rateBefore, base_rate_after: rateAfter },
  });
}
