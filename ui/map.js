// 서울 자치구 선택 지도.
//
// 지도는 입력 장치일 뿐이고, 실제 값은 기존 <select id="district"> 가 그대로 들고 있다.
// 그래서 입력 수집(collectFeatures)과 422 오류 표시는 손댈 필요가 없다.

import { MAP_HEIGHT, MAP_WIDTH, SEOUL_GEO } from "./seoul-geo.js";

const SVG_NS = "http://www.w3.org/2000/svg";

/**
 * 자치구별 색상환 각도(0~359).
 *
 * 공식 '자치구 대표색' 같은 건 없어서 직접 배정했다. 색상환을 25등분한 뒤,
 * 경계를 맞댄 구끼리 색이 비슷해지지 않도록 인접 관계를 계산해 배치했다.
 * 맞닿은 구 사이 색상 거리는 최소 44°다.
 *
 * 실제 색은 style.css 에서 이 값으로 만든다 (채도·명도는 거기서 조절).
 */
const DISTRICT_HUE = {
  강남구: 187, 강동구: 72, 강북구: 144, 강서구: 317, 관악구: 43,
  광진구: 14, 구로구: 245, 금천구: 331, 노원구: 216, 도봉구: 288,
  동대문구: 202, 동작구: 259, 마포구: 173, 서대문구: 58, 서초구: 115,
  성동구: 101, 성북구: 29, 송파구: 274, 양천구: 158, 영등포구: 86,
  용산구: 0, 은평구: 346, 종로구: 302, 중구: 230, 중랑구: 130,
};

/** 지도 위 라벨. '강남구' → '강남'. '중구' 처럼 두 글자면 그대로 둔다. */
function shortLabel(name) {
  return name.length <= 2 ? name : name.slice(0, -1);
}

function createDistrict(gu) {
  const group = document.createElementNS(SVG_NS, "g");
  group.setAttribute("class", "gu");
  group.dataset.district = gu.name;
  // 색상값만 넘기고 실제 칠은 CSS 가 한다. 규칙 하나로 25개 구를 다 칠할 수 있다.
  group.style.setProperty("--gu-h", DISTRICT_HUE[gu.name] ?? 0);

  // <title> 이 첫 자식이어야 브라우저 기본 툴팁으로 뜬다.
  const title = document.createElementNS(SVG_NS, "title");
  title.textContent = gu.name;

  const path = document.createElementNS(SVG_NS, "path");
  path.setAttribute("d", gu.d);

  const label = document.createElementNS(SVG_NS, "text");
  label.setAttribute("x", gu.lx);
  label.setAttribute("y", gu.ly);
  label.textContent = shortLabel(gu.name);

  group.append(title, path, label);
  return group;
}

/**
 * 지도를 그리고 select 와 양방향으로 묶는다.
 *
 * @param {HTMLElement} container 지도를 넣을 자리
 * @param {HTMLSelectElement} select 자치구 값을 들고 있는 요소
 */
export function setupDistrictMap(container, select) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${MAP_WIDTH} ${MAP_HEIGHT}`);
  svg.setAttribute("class", "seoul-map");
  // 키보드 사용자는 select 로 고르면 되므로 지도는 보조 수단으로 둔다.
  svg.setAttribute("role", "presentation");

  const groups = new Map();
  for (const gu of SEOUL_GEO) {
    const group = createDistrict(gu);
    groups.set(gu.name, group);
    svg.append(group);
  }

  function highlight() {
    for (const [name, group] of groups) {
      group.classList.toggle("selected", name === select.value);
    }
  }

  svg.addEventListener("click", (event) => {
    const group = event.target.closest(".gu");
    if (!group) return;
    select.value = group.dataset.district;
    // 값을 코드로 바꾸면 change 가 자동으로 나지 않으므로 직접 알린다.
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });

  select.addEventListener("change", highlight);

  container.replaceChildren(svg);
  highlight();
}
