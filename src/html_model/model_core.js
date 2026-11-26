// 콘크리트 목표 강도 달성 확률 — 브라우저 계산 코어 (모델 v2: ANN 단독 예측, 공식은 비교용)
// Colab 노트북 concrete_ann_prob.ipynb 가 만든 model_export.json 과 같은 계산을 한다
(function (root) {
  "use strict";
  const MATERIALS = ["시멘트", "고로슬래그", "플라이애시", "물", "고성능감수제", "굵은골재", "잔골재"];

  function erf(x) { // Abramowitz & Stegun 7.1.26 (최대 오차 1.5e-7)
    const s = x < 0 ? -1 : 1; x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  const normCdf = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
  const ACT = { relu: (v) => Math.max(0, v), tanh: Math.tanh, logistic: (v) => 1 / (1 + Math.exp(-v)), identity: (v) => v };
  const ageFactor = (t) => t / (4.0 + 0.85 * t);

  // ANN: 입력 = 재료 7종 + 재령 (log_age면 재령에 자연로그) → MinMaxScaler → MLP
  function annStrength(model, mix, age) {
    const x = MATERIALS.map((k) => mix[k]).concat([model.log_age ? Math.log(age) : age]);
    const net = model.reg;
    let h = x.map((v, i) => v * net.scale[i] + net.min[i]);
    const L = net.coefs.length;
    for (let l = 0; l < L; l++) {
      const W = net.coefs[l], b = net.intercepts[l], out = new Array(b.length);
      for (let j = 0; j < b.length; j++) {
        let s = b[j];
        for (let i = 0; i < h.length; i++) s += h[i] * W[i][j];
        out[j] = l < L - 1 ? ACT[net.activation](s) : ACT[net.out_activation](s);
      }
      h = out;
    }
    return h[0];
  }

  // 비교용 공식: Abrams 법칙(물-결합재비) × ACI 209 재령 계수 — 예측에는 쓰지 않음
  function formula(model, mix, age) {
    const binder = mix["시멘트"] + mix["고로슬래그"] + mix["플라이애시"];
    const wb = mix["물"] / binder;
    return { wb, wc: mix["물"] / mix["시멘트"], strength: model.formula.A / Math.pow(model.formula.B, wb) * ageFactor(age) / ageFactor(28) };
  }

  function sigmaOf(model, mu, age) {
    if (model.chosen.startsWith("σ가변")) {
      const c = model.sigma_var.coef;
      return Math.max(model.sigma_var.min, (c[0] + c[1] * mu + c[2] * Math.log(age)) * Math.sqrt(Math.PI / 2));
    }
    return model.sigma;
  }

  function interp(x, xs, ys) { // numpy.interp와 같음
    if (x <= xs[0]) return ys[0];
    if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
    let lo = 0, hi = xs.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] <= x) lo = m; else hi = m; }
    return xs[hi] === xs[lo] ? ys[hi] : ys[lo] + (ys[hi] - ys[lo]) * (x - xs[lo]) / (xs[hi] - xs[lo]);
  }

  function strength(model, mix, age) {
    const mu = annStrength(model, mix, age);
    const f = formula(model, mix, age);
    return { mu, sigma: sigmaOf(model, mu, age), formula: f.strength, wb: f.wb, wc: f.wc };
  }

  function probability(model, mix, age, target) {
    const mu = annStrength(model, mix, age);
    let p = 1 - normCdf((target - mu) / sigmaOf(model, mu, age));
    if (model.chosen.endsWith("+보정") && model.iso) p = interp(p, model.iso.x, model.iso.y);
    return p;
  }

  function requiredStrength(fck, s) { // 시방서 배합강도 식으로 알려진 형태 (계수는 원문 확인 필요)
    if (s === null || s === undefined || Number.isNaN(s)) return fck < 21 ? fck + 7.0 : (fck <= 35 ? fck + 8.5 : 1.1 * fck + 5.0);
    return fck <= 35 ? Math.max(fck + 1.34 * s, (fck - 3.5) + 2.33 * s) : Math.max(fck + 1.34 * s, 0.9 * fck + 2.33 * s);
  }

  function verdict(p) {
    if (p >= 0.8) return { level: "high", text: "시험해 볼 가치가 높음" };
    if (p >= 0.5) return { level: "mid", text: "가능성 있음 · 배합 조정 검토" };
    if (p >= 0.2) return { level: "low", text: "가능성 낮음 · 배합 수정 권장" };
    return { level: "none", text: "달성 어려움" };
  }

  function rangeWarnings(model, mix, age) {
    const out = [];
    for (const [k, v] of Object.entries({ ...mix, "재령": age })) {
      const r = model.ranges[k];
      if (r && (v < r[0] || v > r[1])) out.push(`${k} ${v}: 학습 데이터 범위 ${r[0]}~${r[1]} 밖`);
    }
    return out;
  }

  const api = { MATERIALS, annStrength, formula, strength, probability, requiredStrength, verdict, rangeWarnings, normCdf };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ConcreteModel = api;
})(typeof window !== "undefined" ? window : globalThis);
