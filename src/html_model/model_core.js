// 콘크리트 목표 강도 달성 확률 — 브라우저 계산 코어
// Colab 노트북의 model_export.json과 같은 계산을 한다 (MinMaxScaler → MLP → 확률)
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
  const logit = (p) => { p = Math.min(Math.max(p, 1e-6), 1 - 1e-6); return Math.log(p / (1 - p)); };
  const expit = (z) => 1 / (1 + Math.exp(-z));
  const ACT = {
    relu: (v) => Math.max(0, v), tanh: Math.tanh, logistic: expit, identity: (v) => v,
  };

  const ageFactor = (t) => t / (4.0 + 0.85 * t);

  function features(mix, age, model) {
    const binder = mix["시멘트"] + mix["고로슬래그"] + mix["플라이애시"];
    const wb = mix["물"] / binder;
    const f = {
      ...mix, "재령": age, "결합재": binder, "W/C": mix["물"] / mix["시멘트"], "W/B": wb,
      "재령계수": ageFactor(age),
    };
    f["공식강도"] = model.formula.A / Math.pow(model.formula.B, wb) * ageFactor(age) / ageFactor(28);
    return f;
  }

  function mlp(net, feat) {
    let h = net.features.map((k, i) => feat[k] * net.scale[i] + net.min[i]);
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

  function interp(x, xs, ys) { // numpy.interp와 같음 (양끝은 끝값으로 고정)
    if (x <= xs[0]) return ys[0];
    if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
    let lo = 0, hi = xs.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] <= x) lo = m; else hi = m; }
    return xs[hi] === xs[lo] ? ys[hi] : ys[lo] + (ys[hi] - ys[lo]) * (x - xs[lo]) / (xs[hi] - xs[lo]);
  }

  function sigmaVar(model, mu, af) {
    const c = model.sigma_var.coef;
    return Math.max(model.sigma_var.min, (c[0] + c[1] * mu + c[2] * af) * Math.sqrt(Math.PI / 2));
  }

  function strength(model, mix, age) {
    const feat = features(mix, age, model);
    const mu = mlp(model.reg, feat);
    const sigma = model.chosen.startsWith("A2") ? sigmaVar(model, mu, feat["재령계수"]) : model.sigma;
    return { feat, mu, sigma };
  }

  function probability(model, mix, age, target) {
    const { feat, mu, sigma } = strength(model, mix, age);
    const m = model.chosen;
    if (m.startsWith("A")) return 1 - normCdf((target - mu) / sigma);
    if (!model.clf) throw new Error("이 모델 파일에는 분류 모델 가중치가 없습니다.");
    const ts = model.t_train.filter((t) => t < target).concat([target]);
    let p = 1;
    for (const t of ts) { // 낮은 목표부터 누적 최솟값 → 목표가 높을수록 확률이 커지지 않음
      let q = mlp(model.clf, { ...feat, "목표강도": t });
      if (m.startsWith("B1")) q = expit(model.platt[0] * logit(q) + model.platt[1]);
      else if (m.startsWith("B2")) q = interp(q, model.iso.x, model.iso.y);
      p = Math.min(p, q);
    }
    return p;
  }

  function requiredStrength(fck, s) { // 시방서 배합강도 식으로 알려진 형태 (계수는 원문 확인 필요)
    if (s === null || s === undefined || Number.isNaN(s)) {
      return fck < 21 ? fck + 7.0 : (fck <= 35 ? fck + 8.5 : 1.1 * fck + 5.0);
    }
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

  function validate(model) {
    const need = ["chosen", "formula", "sigma", "sigma_var", "reg", "ranges"];
    const miss = need.filter((k) => !(k in model));
    if (miss.length) throw new Error(`모델 파일에 ${miss.join(", ")} 항목이 없습니다.`);
    if (model.chosen.startsWith("B") && !model.clf) throw new Error("선택 방법이 분류(B)인데 분류 모델 가중치가 없습니다.");
    return true;
  }

  const api = { MATERIALS, features, mlp, strength, probability, requiredStrength, verdict, rangeWarnings, validate, normCdf, ageFactor };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ConcreteModel = api;
})(typeof window !== "undefined" ? window : globalThis);
