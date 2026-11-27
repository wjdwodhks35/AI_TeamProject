"""이상거래 규칙 판정.

왜 규칙을 같이 두는가
---------------------
분류 모델의 레이블은 "같은 조건 기준가보다 20~30% 싼 거래" 인데, 분류 입력 44개에는
**기준가가 들어 있지 않다.** 기준가를 주면 모델이 레이블 공식을 그대로 되계산하므로
일부러 뺀 것이다. 그 결과 모델은 규칙을 복원하지 못하고 상관있는 흔적만 줍는다.

그런데 API 응답에는 회귀가 낸 기준가(`reference_price_krw`)가 **이미 들어 있다.**
그 재료로 만든 한 줄 규칙이 모델보다 낫다 (test 28,337건, 실제 이상 674건):

    판정                                    정밀도   재현율     F1
    분류 모델                                0.265   0.153   0.194
    직거래                                   0.363   0.378   0.371
    기준가 대비 -20% 이하                      0.391   0.371   0.381
    둘 중 하나 (= SUSPECT)                    0.331   0.607   0.429
    둘 다 (= STRONG)                         0.889   0.142   0.246

둘 다 해당하면 10건 중 9건이 실제 이상이다. 그래서 3단계로 나눠 돌려준다.

이 판정이 뜻하는 것은 **"같은 조건의 주변 거래에서 가격이 크게 벗어났다"** 이지
"불법이다" 가 아니다. 싸게 팔린 데에도 정상적인 이유가 있을 수 있다 (가족 간 거래,
급매, 특수 물건).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

# 기준가 대비 이만큼 아래면 규칙이 걸린다. test 에서 F1 이 가장 높았다
# (10% 0.216 / 15% 0.306 / **20% 0.381** / 25% 0.339 / 30% 0.228).
GAP_THRESHOLD = -0.20


class RuleLevel(StrEnum):
    STRONG = "strong"
    SUSPECT = "suspect"
    NORMAL = "normal"


LEVEL_LABEL = {
    RuleLevel.STRONG: "강한 의심",
    RuleLevel.SUSPECT: "의심",
    RuleLevel.NORMAL: "의심 아님",
}


@dataclass(frozen=True)
class RuleVerdict:
    level: RuleLevel
    reasons: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return LEVEL_LABEL[self.level]


def rule_verdict(price_gap_pct: float | None, is_direct_deal: bool | None = None) -> RuleVerdict:
    """회귀 기준가 대비 격차와 직거래 여부로 판정한다.

    ``price_gap_pct`` 는 (실제가격 / 기준가 - 1) × 100 이다. 기준가를 모르면
    (가격 모델 미준비) 판정하지 않는다.
    """
    if price_gap_pct is None:
        return RuleVerdict(RuleLevel.NORMAL, ["기준가격이 없어 규칙 판정을 하지 않았습니다."])

    cheap = price_gap_pct / 100.0 <= GAP_THRESHOLD
    direct = bool(is_direct_deal)

    reasons: list[str] = []
    if cheap:
        reasons.append(
            f"기준가격보다 {abs(price_gap_pct):.0f}% 쌉니다 "
            f"({abs(GAP_THRESHOLD) * 100:.0f}% 이상이면 걸립니다)."
        )
    if direct:
        reasons.append(
            "중개 없이 직접 거래했습니다. 이상거래로 확인된 건에 직거래가 몰려 있습니다."
        )

    if cheap and direct:
        reasons.append("두 조건이 겹칩니다. 이 경우 10건 중 9건이 실제 이상거래였습니다.")
        return RuleVerdict(RuleLevel.STRONG, reasons)
    if cheap or direct:
        return RuleVerdict(RuleLevel.SUSPECT, reasons)

    if is_direct_deal is None:
        reasons.append("직거래 여부를 보내면 판정이 정확해집니다.")
    return RuleVerdict(RuleLevel.NORMAL, reasons)
