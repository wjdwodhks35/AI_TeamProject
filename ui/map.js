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
 * 실제 색은 style.css 에서 이 값으로 만든다. 채도를 아주 낮게 줘서 거의 무채색으로
 * 보이지만, 맞닿은 구끼리는 여전히 미세하게 달라 경계가 읽힌다.
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

const FULL_VIEW = [0, 0, MAP_WIDTH, MAP_HEIGHT];
const ZOOM_PADDING = 0.18; // 고른 구 둘레에 남기는 여백 (구 크기 대비)
const ZOOM_MS = 260;

/** 고른 구가 화면을 채우도록 viewBox 를 계산한다. 가로세로 비율은 그대로 둔다. */
function frameFor(box) {
  const pad = Math.max(box.width, box.height) * ZOOM_PADDING;
  const w = box.width + pad * 2;
  const h = box.height + pad * 2;
  const ratio = MAP_WIDTH / MAP_HEIGHT;
  const [fw, fh] = w / h > ratio ? [w, w / ratio] : [h * ratio, h];
  return [
    box.x + box.width / 2 - fw / 2,
    box.y + box.height / 2 - fh / 2,
    fw,
    fh,
  ];
}

/**
 * 지도를 그리고 select 와 양방향으로 묶는다.
 *
 * 자치구를 고르면 그 구로 확대한다. 동 단위 경계 데이터가 없어서 지도에서 동까지
 * 고르지는 못하고, 확대는 "지금 어디를 보고 있는지" 를 보여주는 역할만 한다.
 *
 * @param {HTMLElement} container 지도를 넣을 자리
 * @param {HTMLSelectElement} select 자치구 값을 들고 있는 요소
 */
export function setupDistrictMap(container, select) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", FULL_VIEW.join(" "));
  svg.setAttribute("class", "seoul-map");
  // 키보드 사용자는 select 로 고르면 되므로 지도는 보조 수단으로 둔다.
  svg.setAttribute("role", "presentation");

  const groups = new Map();
  for (const gu of SEOUL_GEO) {
    const group = createDistrict(gu);
    groups.set(gu.name, group);
    svg.append(group);
  }

  const reset = document.createElement("button");
  reset.type = "button";
  reset.className = "map-reset";
  reset.textContent = "서울 전체";
  reset.hidden = true;

  let current = FULL_VIEW.slice();
  let frame = 0;

  function applyView(view) {
    svg.setAttribute("viewBox", view.join(" "));
    // 글자는 사용자 좌표계라 확대하면 같이 커진다. 배율만큼 되돌려 크기를 유지한다.
    svg.style.setProperty("--map-scale", (view[2] / MAP_WIDTH).toFixed(4));
  }

  function moveTo(target) {
    cancelAnimationFrame(frame);
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    const from = current.slice();
    const start = performance.now();
    current = target;
    if (reduced) {
      applyView(target);
      return;
    }
    const step = (now) => {
      const t = Math.min((now - start) / ZOOM_MS, 1);
      const ease = 1 - (1 - t) ** 3;
      applyView(from.map((v, i) => v + (target[i] - v) * ease));
      if (t < 1) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
  }

  /** 고른 구를 칠한다. 확대 여부와는 별개다 — 전체 지도에서도 어디를 골랐는지 보여야 한다. */
  function paint() {
    for (const [name, group] of groups) {
      group.classList.toggle("selected", name === select.value);
    }
  }

  /** 한 구로 확대하거나(name), 서울 전체로 되돌린다(null). 선택은 건드리지 않는다. */
  function zoomTo(name) {
    const chosen = name ? groups.get(name) : null;
    svg.classList.toggle("zoomed", Boolean(chosen));
    reset.hidden = !chosen;
    moveTo(chosen ? frameFor(chosen.querySelector("path").getBBox()) : FULL_VIEW.slice());
  }

  svg.addEventListener("click", (event) => {
    const group = event.target.closest(".gu");
    if (!group) return;
    const name = group.dataset.district;

    // 이미 고른 구를 다시 누르면 확대만 한다. change 를 또 쏘면 동·단지 목록을
    // 다시 받아 오면서 골라 둔 단지가 풀린다.
    if (select.value === name) {
      zoomTo(name);
      return;
    }

    select.value = name;
    // 값을 코드로 바꾸면 change 가 자동으로 나지 않으므로 직접 알린다.
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });

  // 전체 보기로 돌아갈 뿐, 고른 자치구는 그대로 둔다. 여기서 선택까지 풀면
  // 동·단지·준공연도가 같이 날아간다.
  reset.addEventListener("click", () => zoomTo(null));

  select.addEventListener("change", () => {
    paint();
    zoomTo(select.value || null);
  });

  container.replaceChildren(svg, reset);
  paint();
  zoomTo(select.value || null);
}
