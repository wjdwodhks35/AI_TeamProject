// 개발용 가짜 응답.
//
// 목적은 하나뿐이다: 학습된 모델이 없어서 예측 엔드포인트가 503 만 돌려주는 동안
// '성공 응답 화면'이 제대로 그려지는지 확인하는 것.
//
// 여기서 나온 숫자는 모델 예측이 아니다. 성능 수치로 해석하거나 보고서·발표에
// 옮기면 안 된다. 실제 모델이 배포되면 토글을 끄고 서버 응답으로 확인한다.

const MOCK_PRICE_MODEL = "mock_not_a_real_model";

// 자치구별 ㎡당 대략적인 호가 차이를 흉내낸 계수. 근거 없는 임의 값이다.
const DISTRICT_FACTOR = {
  강남구: 2.6, 서초구: 2.4, 송파구: 1.9, 용산구: 1.9, 성동구: 1.6,
  마포구: 1.5, 광진구: 1.5, 양천구: 1.4, 강동구: 1.3, 동작구: 1.3,
  영등포구: 1.3, 종로구: 1.2, 중구: 1.2, 서대문구: 1.1, 강서구: 1.1,
  동대문구: 1.0, 성북구: 1.0, 은평구: 0.95, 관악구: 0.95, 구로구: 0.9,
  노원구: 0.9, 중랑구: 0.85, 강북구: 0.8, 금천구: 0.8, 도봉구: 0.8,
};

const BASE_PRICE_PER_M2 = 11_000_000; // 원/㎡

function fakePrice(features) {
  const district = DISTRICT_FACTOR[features.district] ?? 1.0;
  const area = Number(features.area_m2) || 0;
  const floor = Number(features.floor) || 0;
  const floorFactor = 1 + Math.max(-5, Math.min(30, floor)) * 0.004;

  const builtYear = features.built_year;
  const age = builtYear ? Math.max(0, 2026 - Number(builtYear)) : 20;
  const ageFactor = Math.max(0.6, 1 - age * 0.006);

  const rate = features.base_rate;
  const rateFactor = rate === null || rate === undefined ? 1 : 1 - (Number(rate) - 3) * 0.03;

  const price = BASE_PRICE_PER_M2 * area * district * floorFactor * ageFactor * rateFactor;
  return Math.max(1, Math.round(price));
}

const MOCK_WARNING = "개발용 가짜 응답입니다. 모델 예측값이 아닙니다.";

export function mockPredictPrice(features) {
  return {
    predicted_price_krw: fakePrice(features),
    model_name: MOCK_PRICE_MODEL,
    warnings: [MOCK_WARNING],
  };
}

export function mockPredictAnomaly(features, priceKrw) {
  const reference = fakePrice(features);
  const gap = reference > 0 ? ((priceKrw - reference) / reference) * 100 : null;

  // 기준가격에서 멀어질수록 점수가 올라가게만 해 둔 계산식. 학습된 분류기가 아니다.
  const score = Math.min(1, Math.abs(gap ?? 0) / 40);
  const threshold = 0.5;

  return {
    is_suspicious: score >= threshold,
    suspicion_score: Number(score.toFixed(4)),
    threshold,
    reference_price_krw: reference,
    price_gap_pct: gap === null ? null : Number(gap.toFixed(2)),
    model_name: "mock_not_a_real_model",
    warnings: [
      "이 결과는 이상거래 의심 판정이며 위법 여부를 확정하지 않습니다.",
      MOCK_WARNING,
    ],
  };
}

export function mockForecast(features, months = 12) {
  const base = 900_000_000 + (Number(features.area_m2) || 84) * 11_000_000;
  const points = Array.from({ length: months }, (_, i) => {
    const month = new Date(Date.UTC(2026, 9 + i, 1));
    const price = base * (1 + 0.002 * (i + 1));
    const spread = 1.2 + 0.004 * (i + 1); // 멀수록 넓어지는 모양만 흉내낸다
    return {
      month: `${month.getUTCFullYear()}-${String(month.getUTCMonth() + 1).padStart(2, "0")}`,
      horizon: i + 1,
      predicted_price_krw: Math.round(price),
      low_krw: Math.round(price / spread),
      high_krw: Math.round(price * spread),
    };
  });
  return {
    base_month: "2026-09",
    model_name: "mock_model_TEST_ONLY",
    points,
    warnings: [MOCK_WARNING],
  };
}

export function mockScenarioInterestRate(features, rateBefore, rateAfter) {
  const before = fakePrice({ ...features, base_rate: rateBefore });
  const after = fakePrice({ ...features, base_rate: rateAfter });
  const diff = after - before;

  return {
    base_rate_before: rateBefore,
    base_rate_after: rateAfter,
    predicted_price_before_krw: before,
    predicted_price_after_krw: after,
    price_diff_krw: diff,
    price_change_pct: before === 0 ? null : Number(((diff / before) * 100).toFixed(2)),
    model_name: MOCK_PRICE_MODEL,
    warnings: [MOCK_WARNING],
    disclaimer:
      "모델 예측값의 조건별 비교 결과이며, 금리 변동이 실제 시장 가격에 미치는 인과효과를 의미하지 않습니다.",
  };
}
