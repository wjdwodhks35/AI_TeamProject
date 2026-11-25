// 화면 동작: 입력 수집 → API 호출 → 결과/오류 표시.

import {
  ApiError,
  SEOUL_DISTRICTS,
  getHealth,
  predictAnomaly,
  predictPrice,
  scenarioInterestRate,
} from "./api.js";
import {
  mockPredictAnomaly,
  mockPredictPrice,
  mockScenarioInterestRate,
} from "./mock.js";

const $ = (id) => document.getElementById(id);

let useMock = false;

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

function regulationOrUndefined() {
  const entries = [...document.querySelectorAll("[data-reg]")].map((el) => [
    el.dataset.reg,
    el.value === "" ? null : el.value === "true",
  ]);
  if (entries.every(([, v]) => v === null)) return undefined;
  return Object.fromEntries(entries);
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

  const regulation = regulationOrUndefined();
  if (regulation !== undefined) features.regulation = regulation;

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

function warningList(warnings) {
  if (!Array.isArray(warnings) || warnings.length === 0) return "";
  return `<ul class="warnings">${warnings
    .map((w) => `<li>${escapeHtml(w)}</li>`)
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
      ${rows([
        ["정확한 금액", formatWon(data.predicted_price_krw)],
        ["사용 모델", data.model_name ?? "-"],
      ])}
      ${warningList(data.warnings)}
    </div>`;
}

function renderAnomaly(container, data) {
  const verdict = data.is_suspicious ? "의심" : "의심 아님";
  container.innerHTML = `<div class="box">
      <h3>이상거래 의심 판정</h3>
      <p class="headline">${escapeHtml(verdict)}</p>
      ${rows([
        ["의심 점수", `${formatPct(data.suspicion_score * 100)} (임계값 ${formatPct(data.threshold * 100)})`],
        ["기준가격", data.reference_price_krw === null ? "없음 (가격 모델 미준비)" : formatKrw(data.reference_price_krw)],
        ["기준가격 대비", data.price_gap_pct === null ? "-" : formatPct(data.price_gap_pct)],
        ["사용 모델", data.model_name ?? "-"],
      ])}
      ${warningList(data.warnings)}
    </div>`;
}

function renderScenario(container, data) {
  const diff = Number(data.price_diff_krw);
  const direction = diff > 0 ? "상승" : diff < 0 ? "하락" : "변화 없음";

  container.innerHTML = `<div class="box">
      <h3>금리 변경 시 예측가격 비교</h3>
      <p class="headline">${escapeHtml(formatKrw(Math.abs(diff)))} ${escapeHtml(direction)}</p>
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
setupTabs();
setupPriceHint();
setupMockToggle();
setupRunButtons();
$("health-refresh").addEventListener("click", refreshHealth);
refreshHealth();
