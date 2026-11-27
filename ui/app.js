// 화면 동작: 입력 수집 → API 호출 → 결과/오류 표시.

import {
  ApiError,
  SEOUL_DISTRICTS,
  getHealth,
  forecastPrice,
  getLocations,
  getRegulation,
  predictAnomaly,
  predictPrice,
  scenarioInterestRate,
} from "./api.js";
import { setupDistrictMap } from "./map.js";
import {
  mockPredictAnomaly,
  mockPredictPrice,
  mockForecast,
  mockScenarioInterestRate,
} from "./mock.js";

const $ = (id) => document.getElementById(id);

let useMock = false;

/** 현재 자치구의 동·단지 목록. 자치구를 바꿀 때마다 한 번 받아 둔다. */
let locations = null;

/**
 * 지금 고른 단지. 목록을 다시 그려도(동 변경·검색) 이 값으로 선택을 되살린다.
 * select 의 value 만 믿으면, 검색어가 하나도 안 맞는 순간 목록이 비면서 고른
 * 단지가 조용히 풀린다. 단지가 빠지면 평균오차가 1.41억 -> 4.18억이라 그냥
 * 넘길 수 없다.
 */
let selectedComplex = null;

/** 자치구를 고르고 나서의 단지 안내문. 검색 상태를 띄웠다가 되돌릴 때 쓴다. */
let complexHintBase = "";

/**
 * 배포된 회귀 모델의 test 구간 평균오차(MAE). 큰 숫자 하나만 띄우면 그 값이 정확한
 * 것처럼 보여서 같이 보여준다. **모델을 다시 학습하면 이 값도 고쳐야 한다** —
 * 출처는 docs/MODEL_PLAN.md 1장의 배포본 행이다.
 */
const MODEL_MAE_KRW = 112_000_000;

const BUILT_YEAR_HINT = "1950 ~ 2030, 비워도 됨. 단지를 고르면 자동으로 들어갑니다.";

// ---------------------------------------------------------------- 숫자 표시

/** 2350000000 → "23억 5,000만원" */
function formatKrw(value) {
  if (value === null || value === undefined) return "-";
  const n = Math.round(Number(value));
  if (!Number.isFinite(n)) return "-";

  const sign = n < 0 ? "-" : "";
  const abs = Math.abs(n);
  const eok = Math.floor(abs / 100_000_000);
  const manwon = Math.floor((abs % 100_000_000) / 10_000);

  const parts = [];
  if (eok) parts.push(`${eok.toLocaleString("ko-KR")}억`);
  if (manwon) parts.push(`${manwon.toLocaleString("ko-KR")}만`);
  if (parts.length === 0) return `${sign}${abs.toLocaleString("ko-KR")}원`;
  return `${sign}${parts.join(" ")}원`;
}

function formatWon(value) {
  if (value === null || value === undefined) return "-";
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return `${Math.round(n).toLocaleString("ko-KR")}원`;
}

function formatPct(value, digits = 2) {
  if (value === null || value === undefined) return "-";
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return `${n.toFixed(digits)}%`;
}

// ---------------------------------------------------------------- 입력 수집

/** 값이 비어 있으면 undefined 를 돌려준다 (선택 항목은 아예 보내지 않기 위해). */
function numberOrUndefined(input) {
  const raw = input.value.trim();
  if (raw === "") return undefined;
  const n = Number(raw);
  return Number.isFinite(n) ? n : NaN;
}

/**
 * 공통 거래 조건을 모은다.
 * 필수값 누락과 숫자 형식만 여기서 막고, 범위 검증은 서버(422)에 맡긴다.
 */
function collectFeatures() {
  const errors = {};
  const features = {};

  const district = $("district").value;
  if (!district) errors.district = "자치구를 선택하세요.";
  else features.district = district;

  // 단지는 형식상 선택 입력이지만 정확도에 가장 큰 영향을 준다. 서버는 단지가 없으면
  // 동도 쓰지 않는다 — 동만 있는 조합은 학습 때 본 적이 없어 오차가 더 커진다.
  const complexId = $("complex_id").value;
  const dong = $("dong").value;
  if (complexId) features.complex_id = complexId;
  if (dong) features.dong = dong;

  const contractDate = $("contract_date").value;
  if (!contractDate) errors.contract_date = "계약일을 입력하세요.";
  else features.contract_date = contractDate;

  const area = numberOrUndefined($("area_m2"));
  if (area === undefined) errors.area_m2 = "전용면적을 입력하세요.";
  else if (Number.isNaN(area)) errors.area_m2 = "숫자로 입력하세요.";
  else features.area_m2 = area;

  const floor = numberOrUndefined($("floor"));
  if (floor === undefined) errors.floor = "층을 입력하세요.";
  else if (Number.isNaN(floor)) errors.floor = "숫자로 입력하세요.";
  else features.floor = floor;

  const builtYear = numberOrUndefined($("built_year"));
  if (Number.isNaN(builtYear)) errors.built_year = "숫자로 입력하세요.";
  else if (builtYear !== undefined) features.built_year = builtYear;

  const baseRate = numberOrUndefined($("base_rate"));
  if (Number.isNaN(baseRate)) errors.base_rate = "숫자로 입력하세요.";
  else if (baseRate !== undefined) features.base_rate = baseRate;

  const directDeal = $("is_direct_deal").value;
  if (directDeal !== "") features.is_direct_deal = directDeal === "1";

  // 규제는 보내지 않는다. 자치구와 계약일만 있으면 정해지는 값이라 서버가
  // 고시 이력에서 직접 채운다 (GET /regulation/{자치구} 가 같은 값을 보여준다).

  return { features, errors };
}

// ---------------------------------------------------------------- 오류 표시

function clearFieldErrors() {
  document.querySelectorAll(".field-error").forEach((el) => {
    el.textContent = "";
  });
}

function showFieldErrors(errors) {
  for (const [field, message] of Object.entries(errors)) {
    const el = document.querySelector(`[data-error-for="${field}"]`);
    if (el) el.textContent = message;
  }
}

/** 422 응답의 issues 를 각 입력칸 아래로 옮긴다. */
function showIssues(issues) {
  const unmapped = [];
  for (const issue of issues) {
    const location = Array.isArray(issue.location) ? issue.location : [];
    const field = [...location].reverse().find((part) => typeof part === "string" && part !== "body");
    const el = field ? document.querySelector(`[data-error-for="${field}"]`) : null;
    if (el) el.textContent = issue.message ?? "입력값을 확인하세요.";
    else unmapped.push(`${location.join(".") || "요청"}: ${issue.message ?? ""}`);
  }
  return unmapped;
}

function renderError(container, error) {
  if (!(error instanceof ApiError)) {
    container.innerHTML = `<div class="box bad"><h3>알 수 없는 오류</h3><p>${escapeHtml(
      String(error?.message ?? error),
    )}</p></div>`;
    return;
  }

  let title = "요청 실패";
  let body = escapeHtml(error.message);

  switch (error.code) {
    case "NETWORK_ERROR":
      title = "서버에 연결할 수 없음";
      break;
    case "MODEL_NOT_READY":
      title = "모델 준비 중";
      body = `${escapeHtml(error.message)}<p class="hint">모델 파일이 배포되면 결과가 표시됩니다. 지금 화면만 확인하려면 위의 '개발용 가짜 응답'을 켜세요.</p>`;
      break;
    case "VALIDATION_ERROR": {
      title = "입력값 확인 필요";
      const unmapped = showIssues(error.issues);
      body = escapeHtml(error.message);
      if (unmapped.length) {
        body += `<ul>${unmapped.map((m) => `<li>${escapeHtml(m)}</li>`).join("")}</ul>`;
      }
      break;
    }
    case "SCENARIO_NOT_SUPPORTED":
      title = "금리 시나리오를 계산할 수 없음";
      body = `${escapeHtml(error.message)}<p class="hint">현재 가격 모델이 기준금리를 입력 변수로 사용하지 않습니다.</p>`;
      break;
    case "INTERNAL_ERROR":
      title = "서버 처리 실패";
      body = `${escapeHtml(error.message)}<p class="hint">잠시 후 다시 시도하세요.</p>`;
      break;
    default:
      break;
  }

  const requestId = error.requestId
    ? `<p class="hint">X-Request-ID: <code>${escapeHtml(error.requestId)}</code></p>`
    : "";
  const status = error.status ? `HTTP ${error.status} · ` : "";

  container.innerHTML = `<div class="box bad">
      <h3>${escapeHtml(title)}</h3>
      <p class="code-line">${status}<code>${escapeHtml(error.code)}</code></p>
      <p>${body}</p>
      ${requestId}
    </div>`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (c) => {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
  });
}

/**
 * 서버 경고문은 API 사용자를 향해 쓰여 있어서 엔드포인트 이름과 필드명이 들어간다.
 * 화면에서는 이 페이지에서 할 수 있는 조작으로 바꿔 보여준다.
 * 원문은 api/main.py 의 NO_COMPLEX_WARNING 과 docs/API_CONTRACT.md 참고.
 */
function humanizeWarning(text) {
  if (text.includes("complex_id") && text.includes("지정하지 않아")) {
    return "단지를 고르지 않아 자치구 평균으로 추정했습니다. 옆의 '단지'에서 고르면 평균오차가 4.18억에서 1.41억으로 줄어듭니다.";
  }
  if (text.includes("찾지 못했습니다")) {
    return `${text} 목록에 있는 단지를 고르면 정확해집니다.`;
  }
  return text;
}

function warningList(warnings) {
  if (!Array.isArray(warnings) || warnings.length === 0) return "";
  return `<ul class="warnings">${warnings
    .map((w) => `<li>${escapeHtml(humanizeWarning(String(w)))}</li>`)
    .join("")}</ul>`;
}

function rows(pairs) {
  return `<table class="kv">${pairs
    .map(
      ([k, v]) =>
        `<tr><th>${escapeHtml(k)}</th><td>${escapeHtml(v)}</td></tr>`,
    )
    .join("")}</table>`;
}

// ---------------------------------------------------------------- 결과 표시

function renderPrice(container, data) {
  container.innerHTML = `<div class="box">
      <h3>예상 거래가격</h3>
      <p class="headline">${escapeHtml(formatKrw(data.predicted_price_krw))}</p>
      <p class="hint">
        이 모델은 실제 거래가격을 평균
        ±${escapeHtml((MODEL_MAE_KRW / 100_000_000).toFixed(2))}억 차이로 맞힙니다.
        한 건의 정답이 아니라 추정 범위로 보세요.
      </p>
      ${rows([
        ["정확한 금액", formatWon(data.predicted_price_krw)],
        ["사용 모델", data.model_name ?? "-"],
      ])}
      ${warningList(data.warnings)}
    </div>`;
}

function renderAnomaly(container, data) {
  // 규칙 판정을 앞에 둔다. 분류 모델은 레이블이 쓰는 기준가를 입력으로 받지 않아
  // 규칙을 복원하지 못한다 (test F1 규칙 0.429 vs 모델 0.194).
  const modelVerdict = data.is_suspicious ? "의심" : "의심 아님";
  const reasons = (data.rule_reasons ?? [])
    .map((r) => `<li>${escapeHtml(r)}</li>`)
    .join("");

  container.innerHTML = `<div class="box">
      <h3>이상거래 의심 판정</h3>
      <p class="headline">${escapeHtml(data.rule_label ?? modelVerdict)}</p>
      ${reasons ? `<ul class="reasons">${reasons}</ul>` : ""}
      ${rows([
        ["기준가격", data.reference_price_krw === null ? "없음 (가격 모델 미준비)" : formatKrw(data.reference_price_krw)],
        ["기준가격 대비", data.price_gap_pct === null ? "-" : formatPct(data.price_gap_pct)],
        [
          "분류 모델 판정",
          `${escapeHtml(modelVerdict)} (점수 ${formatPct(data.suspicion_score * 100)}, 임계값 ${formatPct(data.threshold * 100)})`,
        ],
        ["사용 모델", data.model_name ?? "-"],
      ])}
      ${warningList(data.warnings)}
    </div>`;
}

/**
 * 금리 반응 곡선. 두 값만 비교하면 매끄러운 반응으로 오해하는데, 실제로는 분기 몇 개가
 * 만든 계단이다. 관측 범위 전체를 그려서 그 사실을 그대로 보여준다.
 */
function rateCurveChart(curve, before, after) {
  if (!Array.isArray(curve) || curve.length < 2) return "";
  const { w, h, left, right, top, bottom } = CHART;
  const innerW = w - left - right;
  const innerH = h - top - bottom;

  const rates = curve.map((c) => c.base_rate);
  const prices = curve.map((c) => c.predicted_price_krw);
  const rLo = Math.min(...rates);
  const rHi = Math.max(...rates);
  const pad = (Math.max(...prices) - Math.min(...prices)) * 0.12 || 1;
  const pLo = Math.min(...prices) - pad;
  const pHi = Math.max(...prices) + pad;

  const x = (r) => left + ((r - rLo) / (rHi - rLo || 1)) * innerW;
  const y = (v) => top + innerH - ((v - pLo) / (pHi - pLo)) * innerH;

  const grid = ticks(pLo, pHi)
    .map(
      (v) =>
        `<line x1="${left}" x2="${w - right}" y1="${y(v).toFixed(1)}" y2="${y(v).toFixed(1)}" stroke="${CHART.grid}"/>` +
        `<text x="${left - 8}" y="${(y(v) + 4).toFixed(1)}" text-anchor="end" class="axis">${escapeHtml(formatKrw(v))}</text>`,
    )
    .join("");

  // 계단이라는 것이 보이도록 직선 보간 그대로 둔다 (곡선으로 부드럽게 만들지 않는다).
  const line = curve
    .map((c, i) => `${i ? "L" : "M"}${x(c.base_rate).toFixed(1)},${y(c.predicted_price_krw).toFixed(1)}`)
    .join("");

  const marks = [before, after]
    .filter((r) => r >= rLo && r <= rHi)
    .map((r) => {
      const near = curve.reduce((a, b) =>
        Math.abs(b.base_rate - r) < Math.abs(a.base_rate - r) ? b : a,
      );
      return (
        `<circle cx="${x(near.base_rate).toFixed(1)}" cy="${y(near.predicted_price_krw).toFixed(1)}" r="4.5" fill="${CHART.line}"/>` +
        `<text x="${x(near.base_rate).toFixed(1)}" y="${(y(near.predicted_price_krw) - 10).toFixed(1)}" text-anchor="middle" class="point-label">${near.base_rate.toFixed(2)}%</text>`
      );
    })
    .join("");

  const axis = [rLo, (rLo + rHi) / 2, rHi]
    .map(
      (r, i) =>
        `<text x="${x(r).toFixed(1)}" y="${h - 8}" text-anchor="${i === 0 ? "start" : i === 2 ? "end" : "middle"}" class="axis">${r.toFixed(2)}%</text>`,
    )
    .join("");

  const hover = curve
    .map(
      (c) =>
        `<g class="hit"><rect x="${(x(c.base_rate) - innerW / curve.length / 2).toFixed(1)}" y="${top}" ` +
        `width="${(innerW / curve.length).toFixed(1)}" height="${innerH}" fill="transparent"/>` +
        `<line x1="${x(c.base_rate).toFixed(1)}" x2="${x(c.base_rate).toFixed(1)}" y1="${top}" y2="${top + innerH}" stroke="${CHART.line}" stroke-dasharray="2 2"/>` +
        `<title>기준금리 ${c.base_rate.toFixed(2)}% · ${escapeHtml(formatKrw(c.predicted_price_krw))}</title></g>`,
    )
    .join("");

  return (
    `<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" ` +
    `aria-label="기준금리별 예측가격">${grid}` +
    `<path d="${line}" fill="none" stroke="${CHART.line}" stroke-width="2"/>${marks}${axis}${hover}</svg>`
  );
}

function renderScenario(container, data) {
  const diff = Number(data.price_diff_krw);
  const direction = diff > 0 ? "상승" : diff < 0 ? "하락" : "변화 없음";

  container.innerHTML = `<div class="box">
      <h3>금리 변경 시 예측가격 비교</h3>
      <p class="headline">${escapeHtml(formatKrw(Math.abs(diff)))} ${escapeHtml(direction)}</p>
      ${
        data.extrapolated && Array.isArray(data.observed_range)
          ? `<p class="hint"><strong>학습 데이터의 기준금리는 ${data.observed_range[0]}~${data.observed_range[1]}% 뿐입니다.</strong>
             그 밖의 값은 모델이 구분하지 못해 경계와 같은 답이 나옵니다.</p>`
          : ""
      }
      ${rateCurveChart(data.curve ?? [], data.base_rate_before, data.base_rate_after)}
      ${rows([
        [`변경 전 (${formatPct(data.base_rate_before)})`, formatKrw(data.predicted_price_before_krw)],
        [`변경 후 (${formatPct(data.base_rate_after)})`, formatKrw(data.predicted_price_after_krw)],
        ["차액", formatWon(data.price_diff_krw)],
        ["변화율", data.price_change_pct === null ? "계산 불가 (변경 전 예측가격이 0)" : formatPct(data.price_change_pct)],
        ["사용 모델", data.model_name ?? "-"],
      ])}
      ${warningList(data.warnings)}
      <p class="disclaimer">${escapeHtml(data.disclaimer ?? "")}</p>
    </div>`;
}

// ---------------------------------------------------------------- 실행

// ---------------------------------------------------------------- 전망 그래프

const CHART = {
  w: 640,
  h: 240,
  left: 56,
  right: 16,
  top: 16,
  bottom: 28,
  line: "#1f1f1f",
  band: "#c9ccd1",
  grid: "#e6e6e6",
};

/** 가격 축 눈금을 읽기 좋은 간격으로 고른다. */
function ticks(min, max, count = 4) {
  const raw = (max - min) / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((v) => v >= raw) ?? raw;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max; v += step) out.push(v);
  return out;
}

/**
 * 전망 밴드 차트. 선 하나 + 80% 구간 음영.
 *
 * 구간 음영은 흰 바탕 대비 1.61:1 로 낮다. 색만으로는 못 읽으므로 양 끝 값을 직접
 * 적고 아래에 표를 같이 둔다 (색에 의존하지 않는 경로).
 */
function bandChart(points) {
  const { w, h, left, right, top, bottom } = CHART;
  const innerW = w - left - right;
  const innerH = h - top - bottom;

  const lows = points.map((p) => p.low_krw);
  const highs = points.map((p) => p.high_krw);
  const min = Math.min(...lows);
  const max = Math.max(...highs);
  const pad = (max - min) * 0.08 || 1;
  const lo = min - pad;
  const hi = max + pad;

  const x = (i) => left + (points.length === 1 ? innerW / 2 : (i * innerW) / (points.length - 1));
  const y = (v) => top + innerH - ((v - lo) / (hi - lo)) * innerH;

  const area =
    points.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.high_krw).toFixed(1)}`).join("") +
    // 상한을 왼쪽에서 오른쪽으로 그린 뒤, 하한을 오른쪽에서 왼쪽으로 되짚어 닫는다.
    // 마지막 하한점을 빠뜨리면 오른쪽 끝이 어긋난다.
    points
      .map((_, i) => {
        const j = points.length - 1 - i;
        return `L${x(j).toFixed(1)},${y(points[j].low_krw).toFixed(1)}`;
      })
      .join("") +
    "Z";
  const line = points
    .map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.predicted_price_krw).toFixed(1)}`)
    .join("");

  const gridLines = ticks(lo, hi)
    .map(
      (v) =>
        `<line x1="${left}" x2="${w - right}" y1="${y(v).toFixed(1)}" y2="${y(v).toFixed(1)}" stroke="${CHART.grid}"/>` +
        `<text x="${left - 8}" y="${(y(v) + 4).toFixed(1)}" text-anchor="end" class="axis">${escapeHtml(formatKrw(v))}</text>`,
    )
    .join("");

  const monthLabel = (i) =>
    `<text x="${x(i).toFixed(1)}" y="${h - 8}" text-anchor="${i === 0 ? "start" : "end"}" class="axis">${escapeHtml(points[i].month)}</text>`;

  const endLabels = [0, points.length - 1]
    .map((i) => {
      const p = points[i];
      return (
        `<circle cx="${x(i).toFixed(1)}" cy="${y(p.predicted_price_krw).toFixed(1)}" r="4" fill="${CHART.line}"/>` +
        `<text x="${x(i).toFixed(1)}" y="${(y(p.predicted_price_krw) - 10).toFixed(1)}" ` +
        `text-anchor="${i === 0 ? "start" : "end"}" class="point-label">${escapeHtml(formatKrw(p.predicted_price_krw))}</text>`
      );
    })
    .join("");

  const hover = points
    .map(
      (p, i) =>
        `<g class="hit"><rect x="${(x(i) - innerW / points.length / 2).toFixed(1)}" y="${top}" ` +
        `width="${(innerW / points.length).toFixed(1)}" height="${innerH}" fill="transparent"/>` +
        `<line x1="${x(i).toFixed(1)}" x2="${x(i).toFixed(1)}" y1="${top}" y2="${top + innerH}" stroke="${CHART.line}" stroke-dasharray="2 2"/>` +
        `<title>${escapeHtml(p.month)} · ${escapeHtml(formatKrw(p.predicted_price_krw))} ` +
        `(${escapeHtml(formatKrw(p.low_krw))} ~ ${escapeHtml(formatKrw(p.high_krw))})</title></g>`,
    )
    .join("");

  return (
    `<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" ` +
    `aria-label="향후 ${points.length}개월 예상 거래가격과 80% 구간">` +
    `${gridLines}<path d="${area}" fill="${CHART.band}" opacity="0.55"/>` +
    `<path d="${line}" fill="none" stroke="${CHART.line}" stroke-width="2"/>` +
    `${endLabels}${monthLabel(0)}${monthLabel(points.length - 1)}${hover}</svg>`
  );
}

function forecastTable(points) {
  const body = points
    .map(
      (p) =>
        `<tr><td>${escapeHtml(p.month)}</td><td>${escapeHtml(formatKrw(p.predicted_price_krw))}</td>` +
        `<td>${escapeHtml(formatKrw(p.low_krw))} ~ ${escapeHtml(formatKrw(p.high_krw))}</td></tr>`,
    )
    .join("");
  return (
    `<details class="table-view"><summary>숫자로 보기</summary>` +
    `<table><thead><tr><th>달</th><th>예상</th><th>80% 구간</th></tr></thead>` +
    `<tbody>${body}</tbody></table></details>`
  );
}

function renderForecast(container, data) {
  const points = data.points ?? [];
  if (points.length === 0) {
    container.innerHTML = `<div class="box bad"><h3>전망 없음</h3><p>예측 결과가 비어 있습니다.</p></div>`;
    return;
  }
  const last = points[points.length - 1];
  container.innerHTML = `<div class="box">
      <h3>향후 ${points.length}개월 예상 거래가격</h3>
      <p class="hint">
        ${escapeHtml(data.base_month)} 실거래 집계를 기준으로 합니다. 음영은 80% 구간이고
        멀리 볼수록 넓어집니다.
      </p>
      ${bandChart(points)}
      ${rows([
        [`${escapeHtml(last.month)} 예상`, formatKrw(last.predicted_price_krw)],
        ["같은 달 80% 구간", `${formatKrw(last.low_krw)} ~ ${formatKrw(last.high_krw)}`],
      ])}
      ${forecastTable(points)}
      ${warningList(data.warnings)}
    </div>`;
}

const RUNNERS = {
  price: {
    container: () => $("result-price"),
    async run(features) {
      return useMock ? mockPredictPrice(features) : predictPrice(features);
    },
    render: renderPrice,
  },
  anomaly: {
    container: () => $("result-anomaly"),
    extra() {
      const price = numberOrUndefined($("price_krw"));
      if (price === undefined) return { errors: { price_krw: "실제 거래가격을 입력하세요." } };
      if (Number.isNaN(price)) return { errors: { price_krw: "숫자로 입력하세요." } };
      return { value: Math.round(price) };
    },
    async run(features, priceKrw) {
      return useMock
        ? mockPredictAnomaly(features, priceKrw)
        : predictAnomaly(features, priceKrw);
    },
    render: renderAnomaly,
  },
  forecast: {
    container: () => $("result-forecast"),
    extra() {
      return { value: Number($("forecast_months").value) };
    },
    async run(features, months) {
      return useMock ? mockForecast(features, months) : forecastPrice(features, months);
    },
    render: renderForecast,
  },
  scenario: {
    container: () => $("result-scenario"),
    extra() {
      const errors = {};
      const before = numberOrUndefined($("base_rate_before"));
      const after = numberOrUndefined($("base_rate_after"));
      if (before === undefined) errors.base_rate_before = "변경 전 금리를 입력하세요.";
      else if (Number.isNaN(before)) errors.base_rate_before = "숫자로 입력하세요.";
      if (after === undefined) errors.base_rate_after = "변경 후 금리를 입력하세요.";
      else if (Number.isNaN(after)) errors.base_rate_after = "숫자로 입력하세요.";
      if (Object.keys(errors).length) return { errors };
      return { value: [before, after] };
    },
    async run(features, [before, after]) {
      return useMock
        ? mockScenarioInterestRate(features, before, after)
        : scenarioInterestRate(features, before, after);
    },
    render: renderScenario,
  },
};

async function handleRun(name, button) {
  const runner = RUNNERS[name];
  const container = runner.container();

  clearFieldErrors();
  container.innerHTML = "";

  const { features, errors } = collectFeatures();
  let extraValue;
  if (runner.extra) {
    const extra = runner.extra();
    if (extra.errors) Object.assign(errors, extra.errors);
    else extraValue = extra.value;
  }

  if (Object.keys(errors).length > 0) {
    showFieldErrors(errors);
    container.innerHTML = `<div class="box bad"><h3>입력값 확인 필요</h3><p>표시된 항목을 채운 뒤 다시 실행하세요.</p></div>`;
    return;
  }

  button.disabled = true;
  container.innerHTML = `<p class="hint">계산 중…</p>`;
  try {
    const data = await runner.run(features, extraValue);
    runner.render(container, data);
  } catch (error) {
    renderError(container, error);
  } finally {
    button.disabled = false;
  }
}

// ---------------------------------------------------------------- 서버 상태

async function refreshHealth() {
  const el = $("health-text");
  el.className = "health";
  el.textContent = "서버 상태 확인 중…";

  try {
    const health = await getHealth();
    const price = health.models?.price;
    const anomaly = health.models?.anomaly;
    el.textContent = `서버 v${health.version} · 가격모델 ${price ? "준비됨" : "미준비"} · 이상거래모델 ${anomaly ? "준비됨" : "미준비"}`;
    el.classList.add(price && anomaly ? "ok" : "warn");
  } catch (error) {
    el.textContent = error instanceof ApiError ? error.message : "서버 상태를 확인할 수 없습니다.";
    el.classList.add("bad");
  }
}

// ---------------------------------------------------------------- 초기화

function setupDistricts() {
  const select = $("district");
  select.innerHTML =
    `<option value="">선택하세요</option>` +
    SEOUL_DISTRICTS.map((d) => `<option value="${d}">${d}</option>`).join("");
}

// ---------------------------------------------------------------- 동·단지 선택

function setPlaceholder(select, text) {
  select.innerHTML = `<option value="">${escapeHtml(text)}</option>`;
  select.disabled = true;
}

/** 검색어 정규화: 공백을 지워 "래미안 원베일리" 로도 "래미안원베일리" 가 걸리게. */
function normalize(text) {
  return text.replace(/\s+/g, "").toLowerCase();
}

/**
 * 단지 목록을 채운다. 동이 골라져 있으면 그 동만, 검색어가 있으면 이름이 맞는 것만.
 * 거래 많은 순 — 위쪽 몇 개가 사실상 그 동의 대표 단지다.
 *
 * 25개 자치구 중에는 단지가 700개 넘는 곳도 있어서 펼침 목록만으로는 못 찾는다.
 * 검색은 동 선택을 건너뛰는 지름길이라, 검색어가 있으면 동 필터는 무시한다.
 */
function fillComplexes(dong) {
  const select = $("complex_id");
  const hint = $("complex-hint");
  if (!locations) return;

  const raw = $("complex_search").value.trim();
  const query = normalize(raw);

  const items = locations.complexes
    .filter((c) => (query ? normalize(c.apartment_name).includes(query) : !dong || c.dong === dong))
    .slice()
    .sort((a, b) => b.n_deals - a.n_deals);

  // 고른 단지가 지금 조건에 안 걸려도 목록 맨 위에 남긴다. 검색을 잘못 쳤다고
  // 선택이 풀리면 사용자는 풀린 줄도 모르고 자치구 평균 예측을 받게 된다.
  const shown = items.slice();
  if (selectedComplex && !shown.some((c) => c.complex_id === selectedComplex.complex_id)) {
    shown.unshift(selectedComplex);
  }

  if (shown.length === 0) {
    setPlaceholder(select, query ? "맞는 단지 없음" : "해당 동에 단지가 없습니다");
  } else {
    // 동이 정해져 있고 검색 중이 아니면 동 이름은 줄마다 되풀이하지 않는다.
    const showDong = Boolean(query) || !dong;
    const label = (c) =>
      `${showDong ? `${c.dong} ` : ""}${c.apartment_name} (${c.n_deals.toLocaleString("ko-KR")}건)`;

    select.innerHTML =
      `<option value="">선택 안 함</option>` +
      shown
        .map(
          (c) =>
            `<option value="${escapeHtml(c.complex_id)}" data-dong="${escapeHtml(c.dong)}"` +
            ` data-built-year="${c.built_year ?? ""}">` +
            `${escapeHtml(label(c))}</option>`,
        )
        .join("");
    select.disabled = false;
    if (selectedComplex) select.value = selectedComplex.complex_id;
  }

  if (!query) hint.textContent = complexHintBase;
  else if (items.length === 0) hint.textContent = `'${raw}' 에 맞는 단지가 없습니다.`;
  else hint.textContent = `'${raw}' 검색 결과 ${items.length.toLocaleString("ko-KR")}개.`;
}

/**
 * 단지를 고르면 그 단지에서 정해지는 값들을 같이 채운다.
 *
 * 지금은 준공연도 하나뿐이다. 면적·층은 같은 단지 안에서도 집마다 달라서 채울 수 없다.
 */
function applyComplex(option) {
  const id = option?.value || "";
  selectedComplex = id ? locations?.complexes.find((c) => c.complex_id === id) ?? null : null;

  const dong = option?.dataset.dong;
  if (dong) $("dong").value = dong;

  const builtYear = option?.dataset.builtYear;
  const input = $("built_year");
  const hint = $("built-year-hint");
  if (builtYear) {
    input.value = builtYear;
    hint.textContent = `${option.textContent.trim().split(" (")[0]} 준공연도입니다. 고치면 그 값이 쓰입니다.`;
  } else if (option?.value) {
    hint.textContent = "이 단지는 준공연도가 없어 비워 뒀습니다. 알면 직접 넣으세요.";
  } else {
    hint.textContent = BUILT_YEAR_HINT;
  }
}

async function loadLocations(district) {
  const dongSelect = $("dong");
  const complexSelect = $("complex_id");
  const search = $("complex_search");
  const hint = $("complex-hint");
  locations = null;

  // 자치구가 바뀌면 이전 자치구의 단지·검색어는 모두 무효다.
  search.value = "";
  search.disabled = true;
  selectedComplex = null;
  $("built-year-hint").textContent = BUILT_YEAR_HINT;

  if (!district) {
    setPlaceholder(dongSelect, "자치구를 먼저 선택");
    setPlaceholder(complexSelect, "자치구를 먼저 선택");
    return;
  }

  setPlaceholder(dongSelect, "불러오는 중…");
  setPlaceholder(complexSelect, "불러오는 중…");

  try {
    locations = await getLocations(district);
  } catch (error) {
    const reason =
      error instanceof ApiError && error.code === "MODEL_NOT_READY"
        ? "단지 목록이 서버에 아직 배포되지 않았습니다."
        : error instanceof ApiError
          ? error.message
          : "단지 목록을 불러올 수 없습니다.";
    setPlaceholder(dongSelect, "목록 없음");
    setPlaceholder(complexSelect, "목록 없음");
    hint.textContent = `${reason} 단지 없이도 예측은 되지만 자치구 평균 수준의 추정이 됩니다.`;
    return;
  }

  complexHintBase =
    `${district} 단지 ${locations.complexes.length.toLocaleString("ko-KR")}개. ` +
    `비워 두면 자치구 평균으로만 추정해서 오차가 3배 가까이 커집니다.`;
  hint.textContent = complexHintBase;

  dongSelect.innerHTML =
    `<option value="">전체</option>` +
    locations.dongs
      .map((d) => `<option value="${escapeHtml(d)}">${escapeHtml(d)}</option>`)
      .join("");
  dongSelect.disabled = false;
  search.disabled = false;
  fillComplexes("");
}

// ---------------------------------------------------------------- 규제 표시

/**
 * 규제는 입력칸이 아니라 표시만 한다. 자치구와 계약일이 정해지면 고시 이력에서
 * 한 가지 답이 나오는 값이라, 사용자가 고를 여지가 없다. 예측 요청에서도 서버가
 * 같은 이력으로 채우므로 여기 보이는 것과 모델이 받는 값이 같다.
 */
async function refreshRegulation() {
  const line = $("regulation-state");
  const district = $("district").value;
  const on = $("contract_date").value;

  if (!district || !on) {
    line.textContent = "자치구와 계약일을 고르면 여기에 표시됩니다.";
    return;
  }

  line.textContent = "확인 중…";
  let state;
  try {
    state = await getRegulation(district, on);
  } catch (error) {
    line.textContent =
      error instanceof ApiError && error.code === "MODEL_NOT_READY"
        ? "규제 이력을 서버에서 읽지 못했습니다. 예측은 규제 없음으로 계산됩니다."
        : "규제 상태를 확인하지 못했습니다.";
    return;
  }

  const on_ = [
    state.speculation_overheated && "투기과열지구",
    state.adjustment_target && "조정대상지역",
    state.land_permit && "토지거래허가구역",
  ].filter(Boolean);

  line.textContent =
    on_.length === 0
      ? `${state.on} 기준 ${state.district}: 해당 없음`
      : `${state.on} 기준 ${state.district}: ${on_.join(", ")}`;
}

function setupLocations() {
  // 지도를 클릭해도 같은 select 의 change 가 일어나므로 여기 하나만 달면 된다.
  $("district").addEventListener("change", (event) => loadLocations(event.target.value));
  $("dong").addEventListener("change", (event) => fillComplexes(event.target.value));

  // 단지를 고르면 그 단지의 동과 준공연도를 같이 맞춰 준다.
  $("complex_id").addEventListener("change", (event) => {
    applyComplex(event.target.selectedOptions[0]);
  });

  // 검색은 동 선택과 같은 목록을 다시 그린다.
  $("complex_search").addEventListener("input", () => fillComplexes($("dong").value));
}

function setupRegulation() {
  $("district").addEventListener("change", refreshRegulation);
  $("contract_date").addEventListener("change", refreshRegulation);
  refreshRegulation();
}

function setupTabs() {
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((other) => {
        const selected = other === tab;
        other.setAttribute("aria-selected", String(selected));
        $(`panel-${other.dataset.tab}`).hidden = !selected;
      });
    });
  });
}

function setupPriceHint() {
  const input = $("price_krw");
  const hint = $("price-krw-hint");
  const update = () => {
    const raw = input.value.trim();
    hint.textContent =
      raw === "" ? "원 단위로 입력합니다." : `= ${formatKrw(Number(raw))}`;
  };
  input.addEventListener("input", update);
  update();
}

function setupMockToggle() {
  const toggle = $("mock-toggle");
  const banner = $("mock-banner");
  toggle.addEventListener("change", () => {
    useMock = toggle.checked;
    banner.hidden = !useMock;
  });
}

function setupRunButtons() {
  document.querySelectorAll("[data-run]").forEach((button) => {
    button.addEventListener("click", () => handleRun(button.dataset.run, button));
  });
}

setupDistricts();
setupLocations();
setupRegulation();
setupDistrictMap($("district-map"), $("district"));
setupTabs();
setupPriceHint();
setupMockToggle();
setupRunButtons();
$("health-refresh").addEventListener("click", refreshHealth);
refreshHealth();
