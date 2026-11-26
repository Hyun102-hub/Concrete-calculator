"""concrete_ann_prob.ipynb 생성: 1,030건으로 학습한 ANN 단독 예측 + 공식은 비교용 + 성능 검증"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# 인공신경망(ANN) 콘크리트 목표 강도 달성 확률 예측 시스템

**입력**: 재료 7종 단위량(kg/m³) + 재령(일), 그리고 목표 강도(MPa)
**출력**: 그 재령에서 압축강도가 목표 강도 이상일 **확률**, ANN 예측 강도 μ ± σ

- **예측은 ANN 단독**: UCI *Concrete Compressive Strength*(I-Cheng Yeh, CC BY 4.0, DOI 10.24432/C5PK67) **1,030건**만으로 학습. 입력은 원자료의 8개 열 그대로
- **공식은 비교용**: Abrams 법칙 + ACI 209 재령식으로 계산한 강도를 옆에 보여 주고, ANN 예측과 얼마나 다른지 표시 (모델 입력에는 쓰지 않음)
- **성능 검증 7가지**: 테스트 평가와 기준 모델 비교, 반복 교차검증, 신뢰구간, 오차 분석, 확률 보정 확인, 물리적 타당성, (선택) 중첩 교차검증

**실행**: 런타임 → 모두 실행. `FAST_MODE = True`면 무료 Colab에서 몇 분 안에 끝납니다.
""")

md("## 0. 설정")
code(r"""
DATA_SOURCE = "auto"      # "auto": UCI에서 자동 다운로드 / "upload": 파일 직접 올리기
LOCAL_PATH = ""           # (선택) 파일 경로. 예: "/content/drive/MyDrive/Concrete_Data.xls"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip"
RANDOM_STATE = 42
TEST_SIZE = 0.30
FAST_MODE = True          # True: 축소 탐색 / False: 전체 탐색 + 중첩 교차검증 (오래 걸림)
CV_REPEATS = 3            # 반복 그룹 교차검증 반복 횟수 (5겹 × 반복)
N_BOOT = 1000             # 부트스트랩 반복 횟수
EVAL_TARGETS = [21, 24, 27, 30, 35, 40]
LOG_AGE = True            # True: 재령에 로그를 씌워 넣음 (초기에 빨리, 나중에 천천히 오르는 강도 특성을 학습하기 쉬움) / False: 그대로
""")

md("## 라이브러리")
code(r"""
import io, json, zipfile, warnings, subprocess, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV, cross_val_predict, train_test_split
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, brier_score_loss, log_loss, roc_auc_score
from sklearn.exceptions import ConvergenceWarning
import joblib
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def pip_install(pkg):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

try:
    import xlrd  # noqa: F401
except ImportError:
    pip_install("xlrd")
try:
    import koreanize_matplotlib  # noqa: F401
except ImportError:
    pip_install("koreanize-matplotlib")
    try:
        import koreanize_matplotlib  # noqa: F401
    except ImportError:
        print("한글 글꼴 설치 실패: 그래프의 한글이 깨질 수 있습니다.")
plt.rcParams["axes.unicode_minus"] = False

try:
    from google.colab import files as colab_files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
print("Colab 환경:", IN_COLAB)
""")

md("## 1. 데이터 불러오기")
code(r"""
NAMES = ["시멘트", "고로슬래그", "플라이애시", "물", "고성능감수제", "굵은골재", "잔골재", "재령", "압축강도"]
MATERIALS = NAMES[:7]
FEATURES = NAMES[:8]          # ANN 입력: 원자료 8개 열 그대로


def read_table(name, data):
    if str(name).lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(io.BytesIO(data) if isinstance(data, bytes) else data)
    return pd.read_csv(io.BytesIO(data) if isinstance(data, bytes) else data)


def load_raw():
    if LOCAL_PATH:
        print("파일 읽기:", LOCAL_PATH)
        return read_table(LOCAL_PATH, LOCAL_PATH)
    if DATA_SOURCE == "auto":
        try:
            import urllib.request
            with urllib.request.urlopen(UCI_ZIP_URL, timeout=60) as r:
                z = zipfile.ZipFile(io.BytesIO(r.read()))
            name = next(n for n in z.namelist() if n.lower().endswith(".xls"))
            print("UCI에서 내려받음:", name)
            return read_table(name, z.read(name))
        except Exception as e:
            print("자동 다운로드 실패 → 직접 올려 주세요:", e)
    print("Concrete_Data.xls (또는 같은 9개 열의 CSV)를 올려 주세요.")
    if IN_COLAB:
        up = colab_files.upload()
        name = next(iter(up))
        return read_table(name, up[name])
    path = input("파일 경로: ").strip()
    return read_table(path, path)


raw = load_raw()
assert raw.shape[1] == 9, f"열이 9개여야 합니다 (현재 {raw.shape[1]}개)"
raw.columns = NAMES
print(raw.shape)
""")

md(r"""
## 2. 데이터 점검 (1,030건 전부 사용)

결측값·불가능한 값은 확인해서 있으면 빼고, 완전히 같은 행과 배합·재령이 같은 행은 **건수만 기록하고 그대로 씁니다**.
재료 7종이 같으면 같은 배합 → **배합번호**. 같은 배합은 학습·테스트·교차검증 묶음에서 항상 한쪽에만 들어갑니다.
""")
code(r"""
df = raw.copy()
n_na = int(df.isna().any(axis=1).sum())
bad = (df[NAMES] < 0).any(axis=1) | (df["시멘트"] <= 0) | (df["물"] <= 0) | (df["재령"] <= 0) | (df["압축강도"] <= 0)
check = pd.DataFrame([
    ("원자료", len(raw), "-"), ("결측값이 있는 행", n_na, "있으면 제외"), ("물리적으로 불가능한 값", int(bad.sum()), "있으면 제외"),
    ("완전히 같은 행 (두 번째 이후)", int(df.duplicated().sum()), "그대로 사용"),
    ("배합·재령이 같은 행 (완전 중복 포함)", int(df.duplicated(subset=FEATURES, keep=False).sum()), "그대로 사용"),
], columns=["항목", "건수", "처리"])
df = df[~(df.isna().any(axis=1) | bad)].reset_index(drop=True)
df["배합번호"] = df.groupby(MATERIALS).ngroup() + 1
check.loc[len(check)] = ("최종 사용 행", len(df), "-")
check.loc[len(check)] = ("서로 다른 배합", df["배합번호"].nunique(), "-")
display(check)
""")

md("## 3. 배합 기준 그룹 분할 (학습 70% / 테스트 30%)")
code(r"""
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
tr_i, te_i = next(gss.split(df, groups=df["배합번호"]))
train, test = df.iloc[tr_i].reset_index(drop=True), df.iloc[te_i].reset_index(drop=True)
print(f"학습 {len(train)}건 / 테스트 {len(test)}건 ({len(test) / len(df):.1%}), "
      f"겹치는 배합 {len(set(train['배합번호']) & set(test['배합번호']))}개, 테스트 중 28일 {int((test['재령'] == 28).sum())}건")
""")

md(r"""
## 4. ANN 학습 (입력: 재료 7종 + 재령)

(`LOG_AGE = True`면 재령에 자연로그) → MinMaxScaler(학습용으로만 기준) → MLPRegressor, GridSearchCV 5겹 **GroupKFold**(배합 기준)로 은닉층·활성화 함수·규제·학습률 선택.
입력은 원자료의 8개 열뿐이며, 공식 값은 넣지 않습니다.
""")
code(r"""
if FAST_MODE:
    grid = {"mlp__hidden_layer_sizes": [(64,), (64, 32), (128, 64, 32)], "mlp__activation": ["relu", "tanh"],
            "mlp__alpha": [1e-4, 1e-3], "mlp__learning_rate_init": [0.01]}
else:
    grid = {"mlp__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64), (128, 64, 32)],
            "mlp__activation": ["relu", "tanh"], "mlp__alpha": [1e-4, 1e-3, 1e-2], "mlp__learning_rate_init": [0.001, 0.01]}


def ann():
    pre = ColumnTransformer([("재료", "passthrough", MATERIALS),
                             ("재령", FunctionTransformer(np.log) if LOG_AGE else "passthrough", ["재령"])])
    return Pipeline([("pre", pre), ("scale", MinMaxScaler()),
                     ("mlp", MLPRegressor(max_iter=5000, early_stopping=True, validation_fraction=0.15,
                                          n_iter_no_change=50, random_state=RANDOM_STATE))])


search = GridSearchCV(ann(), grid, cv=GroupKFold(5), scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=1)
search.fit(train[FEATURES], train["압축강도"], groups=train["배합번호"])
REG = search.best_estimator_
BEST = search.best_params_
n_w = sum(w.size for w in REG.named_steps["mlp"].coefs_) + sum(b.size for b in REG.named_steps["mlp"].intercepts_)
print("최적 설정:", BEST, f"| 교차검증 RMSE {-search.best_score_:.2f} MPa | 가중치 {n_w}개")
""")

md(r"""
## 5. 비교용 공식 (Abrams 법칙 + ACI 209)

모델 입력에는 쓰지 않고, **ANN 예측과 비교만** 합니다.
- W/B = 물 ÷ (시멘트 + 고로슬래그 + 플라이애시)
- 공식 강도 = A ÷ B<sup>W/B</sup> × g(t) ÷ g(28), g(t) = t ÷ (4 + 0.85t)
- A, B는 학습용 데이터로만 맞춤
""")
code(r"""
def age_factor(t):
    return t / (4.0 + 0.85 * t)


def wb_of(d):
    return d["물"] / (d["시멘트"] + d["고로슬래그"] + d["플라이애시"])


def formula_strength(wb, t, A, B):
    return A / B ** wb * age_factor(t) / age_factor(28.0)


(A_FIT, B_FIT), _ = curve_fit(lambda X, A, B: formula_strength(X[0], X[1], A, B),
                              (wb_of(train).values, train["재령"].values), train["압축강도"].values, p0=[100, 10], maxfev=20000)
print(f"공식 계수 (학습용으로 맞춤): A = {A_FIT:.2f}, B = {B_FIT:.3f}")
""")

md(r"""
---
# 성능 검증

| # | 방법 | 확인하는 것 |
|---|---|---|
| ① | 테스트 평가 + 기준 모델 비교 | 처음 보는 배합에서의 정확도, 평균값·선형회귀·공식보다 나은지 |
| ② | 반복 그룹 교차검증 | 데이터를 어떻게 나눠도 성능이 비슷한지 (안정성) |
| ③ | 부트스트랩 신뢰구간 | 테스트 성능 숫자의 불확실성 (95% 범위) |
| ④ | 오차 분석 | 어떤 강도·재령·배합에서 잘 맞고 틀리는지, 과적합 여부 |
| ⑤ | 확률 보정 확인 | "80%"라고 한 배합이 실제로 80% 달성하는지 |
| ⑥ | 물리적 타당성 | 물을 늘리면 강도가 내려가는 등 공학 상식과 맞는지 |
| ⑦ | 중첩 교차검증 (`FAST_MODE = False`) | 설정 고르기까지 포함한 성능을 편향 없이 추정 |
""")

md("## ① 테스트 평가와 기준 모델 비교")
code(r"""
def rscore(y, p):
    return {"R²": r2_score(y, p), "MAE(MPa)": mean_absolute_error(y, p), "RMSE(MPa)": np.sqrt(mean_squared_error(y, p))}


y_tr, y_te = train["압축강도"].values, test["압축강도"].values
p_tr, p_te = REG.predict(train[FEATURES]), REG.predict(test[FEATURES])
lin = Pipeline([("scale", MinMaxScaler()), ("lr", LinearRegression())]).fit(train[FEATURES], y_tr)
f_te = formula_strength(wb_of(test), test["재령"], A_FIT, B_FIT).values
comp = pd.DataFrame({
    "ANN (학습용)": rscore(y_tr, p_tr),
    "ANN (테스트)": rscore(y_te, p_te),
    "기준: 평균값 (테스트)": rscore(y_te, np.full(len(y_te), y_tr.mean())),
    "기준: 선형회귀 (테스트)": rscore(y_te, lin.predict(test[FEATURES])),
    "참고: 공식 (테스트)": rscore(y_te, f_te),
}).T.round(3)
display(comp)
err = p_te - y_te
print(f"오차 ±3 MPa 이내 {np.mean(np.abs(err) <= 3):.1%}, ±5 이내 {np.mean(np.abs(err) <= 5):.1%}, ±10 이내 {np.mean(np.abs(err) <= 10):.1%}")
print(f"상대오차 중앙값 {100 * np.median(np.abs(err) / y_te):.1f}%, 평균 편향 {err.mean():+.2f} MPa")
""")

md(r"""
## ② 반복 그룹 교차검증 (5겹 × 반복)

전체 1,030건을 배합 기준으로 5묶음으로 나눠, 한 묶음씩 시험용으로 돌려 가며 5번 학습·평가합니다. 묶음을 섞는 방법을 바꿔 `CV_REPEATS`번 반복합니다.
설정(은닉층 등)은 4번에서 학습용으로만 고른 값을 씁니다.
""")
code(r"""
rows = []
groups_all = df["배합번호"].values
for rep in range(CV_REPEATS):
    rng = np.random.default_rng(RANDOM_STATE + rep)
    ug = np.unique(groups_all); perm = rng.permutation(ug)
    fold_of = {g: i % 5 for i, g in enumerate(perm)}
    fold = np.array([fold_of[g] for g in groups_all])
    for k in range(5):
        trm, tem = fold != k, fold == k
        m = clone(REG).fit(df.loc[trm, FEATURES], df.loc[trm, "압축강도"])
        p = m.predict(df.loc[tem, FEATURES])
        rows.append({"반복": rep + 1, "묶음": k + 1, "시험 건수": int(tem.sum()), **rscore(df.loc[tem, "압축강도"], p)})
cvres = pd.DataFrame(rows)
cv_sum = cvres[["R²", "MAE(MPa)", "RMSE(MPa)"]].agg(["mean", "std", "min", "max"]).round(3)
print(f"반복 그룹 교차검증: 5겹 × {CV_REPEATS}회 = {len(cvres)}번 학습·평가")
display(cv_sum)
""")

md(r"""
## ③ 부트스트랩 95% 신뢰구간 (테스트 성능)

테스트 배합을 **배합 단위로** 복원 추출해 `N_BOOT`번(기본 1,000번) 다시 계산하고, 그 분포의 2.5%~97.5% 범위를 신뢰구간으로 봅니다.
""")
code(r"""
rng = np.random.default_rng(RANDOM_STATE)
tg = test["배합번호"].values; ug = np.unique(tg); idx_of = {g: np.where(tg == g)[0] for g in ug}
boot = []
for _ in range(N_BOOT):
    pick = np.concatenate([idx_of[g] for g in rng.choice(ug, len(ug), replace=True)])
    boot.append(rscore(y_te[pick], p_te[pick]))
boot = pd.DataFrame(boot)
ci = pd.DataFrame({"테스트 값": comp.loc["ANN (테스트)"], "95% 하한": boot.quantile(0.025), "95% 상한": boot.quantile(0.975)}).round(3)
display(ci)
""")

md("## ④ 오차 분석")
code(r"""
t2 = test.assign(예측=p_te, 오차=err, 절대오차=np.abs(err))
t2["강도 구간"] = pd.cut(t2["압축강도"], [0, 20, 40, 60, 90], labels=["20 이하", "20~40", "40~60", "60 이상"])
t2["재령 구간"] = pd.cut(t2["재령"], [0, 7, 14, 28, 56, 100, 365], labels=["1~7일", "14일", "28일", "56일", "90~100일", "120~365일"])
t2["배합 종류"] = np.where((t2["고로슬래그"] == 0) & (t2["플라이애시"] == 0), "시멘트만", "혼화재 사용")
def agg(col):
    return pd.DataFrame([{col: k, "건수": len(g), "MAE": g["절대오차"].mean(), "RMSE": np.sqrt((g["오차"] ** 2).mean()),
                          "평균 편향": g["오차"].mean()} for k, g in t2.groupby(col, observed=True)]).set_index(col).round(2)
by_str, by_age, by_kind = agg("강도 구간"), agg("재령 구간"), agg("배합 종류")
print("강도 구간별"); display(by_str)
print("재령 구간별"); display(by_age)
print("배합 종류별"); display(by_kind)
gap = comp.loc["ANN (학습용)", "R²"] - comp.loc["ANN (테스트)", "R²"]
print(f"과적합 확인: 학습 R² {comp.loc['ANN (학습용)', 'R²']:.3f} − 테스트 R² {comp.loc['ANN (테스트)', 'R²']:.3f} = {gap:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
ax[0].scatter(y_te, p_te, s=12, alpha=0.6, color="#2a78d6"); ax[0].plot([0, 90], [0, 90], color="#0b0b0b", lw=0.8)
ax[0].set_xlabel("실제 강도 (MPa)"); ax[0].set_ylabel("ANN 예측 (MPa)"); ax[0].set_title("테스트: 실제 vs 예측")
ax[1].hist(err, bins=30, color="#2a78d6", edgecolor="white"); ax[1].axvline(0, color="#0b0b0b", lw=0.8)
ax[1].set_xlabel("예측 - 실제 (MPa)"); ax[1].set_title("테스트 오차 분포")
plt.tight_layout(); plt.show()
""")

md(r"""
## 확률 계산 방법 고르기

달성 확률 = 1 − Φ((목표 − μ) ÷ σ). μ는 ANN 예측, σ는 **학습용 데이터의 그룹 교차검증 예측 오차**로 구합니다.

| 후보 | 내용 |
|---|---|
| σ고정 | σ 하나 |
| σ가변 | σ = √(π/2) × (a + b·μ + c·ln 재령), 최소 2 MPa |
| σ고정+보정, σ가변+보정 | 위 확률을 isotonic 회귀로 한 번 더 보정 (중간 확률이 부풀려지는 문제 대응) |

**선택은 학습용 데이터 안에서만** 합니다(교차검증 예측으로 계산한 로그손실 최소). 테스트 데이터는 고르는 데 쓰지 않고 마지막 평가에만 씁니다.
""")
code(r"""
oof = cross_val_predict(REG, train[FEATURES], y_tr, groups=train["배합번호"], cv=GroupKFold(5))
res = y_tr - oof
SIGMA = float(np.std(res, ddof=1))
sl = LinearRegression().fit(np.c_[oof, np.log(train["재령"])], np.abs(res))
SIG_COEF = [float(sl.intercept_), float(sl.coef_[0]), float(sl.coef_[1])]; SIG_MIN = 2.0
T_GRID = np.arange(5.0, 80.01, 1.0)


def sigma_of(method, mu, age):
    if method.startswith("σ가변"):
        return np.maximum(SIG_MIN, (SIG_COEF[0] + SIG_COEF[1] * mu + SIG_COEF[2] * np.log(age)) * np.sqrt(np.pi / 2))
    return np.full_like(np.asarray(mu, float), SIGMA)


def raw_prob(method, mu, age, targets):
    s = sigma_of(method, mu, age)
    return 1 - norm.cdf((np.asarray(targets)[None, :] - np.asarray(mu)[:, None]) / np.asarray(s)[:, None])


def pscore(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return {"Brier": brier_score_loss(y, p), "로그손실": log_loss(y, p, labels=[0, 1]),
            "AUC": roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan}


Y_oof = (y_tr[:, None] >= T_GRID[None, :]).astype(int)
gtr = train["배합번호"].values
sel = {}
for base in ["σ고정", "σ가변"]:
    P = raw_prob(base, oof, train["재령"].values, T_GRID)
    sel[base] = pscore(Y_oof.ravel(), P.ravel())
    Pc = np.zeros_like(P)                          # 보정도 그룹 교차검증으로 평가 (보정에 쓴 데이터로 평가하지 않음)
    for a, b in GroupKFold(5).split(P, groups=gtr):
        iso_k = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(P[a].ravel(), Y_oof[a].ravel())
        Pc[b] = iso_k.predict(P[b].ravel()).reshape(P[b].shape)
    sel[base + "+보정"] = pscore(Y_oof.ravel(), np.minimum.accumulate(Pc, axis=1).ravel())
sel = pd.DataFrame(sel).T.round(4)
display(sel)
CHOSEN = sel["로그손실"].idxmin()
ISO = None
if CHOSEN.endswith("+보정"):
    ISO = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
        raw_prob(CHOSEN, oof, train["재령"].values, T_GRID).ravel(), Y_oof.ravel())
print(f"선택: {CHOSEN} (학습용 교차검증 로그손실 최소) | σ고정 = {SIGMA:.2f} MPa | "
      f"σ가변 = √(π/2) × ({SIG_COEF[0]:.2f} + {SIG_COEF[1]:.3f}·μ + {SIG_COEF[2]:.2f}·ln 재령)")


def prob(mu, age, targets, method=None):
    method = method or CHOSEN
    P = raw_prob(method, np.atleast_1d(mu), np.atleast_1d(age), np.atleast_1d(targets))
    if method.endswith("+보정"):
        P = ISO.predict(P.ravel()).reshape(P.shape)
    return P
""")

md(r"""
## ⑤ 확률 보정 확인 (테스트)

테스트 행 × 목표 5~80 MPa 전체와, 재령 28일 × 목표 21~40 MPa를 따로 봅니다. **확률 구간별 표**에서 "평균 예측"과 "실제 달성"이 가까울수록 정직한 확률입니다.
""")
code(r"""
Y_te = (y_te[:, None] >= T_GRID[None, :]).astype(int)
P_te = prob(p_te, test["재령"].values, T_GRID)
P_base = np.tile([(y_tr >= t).mean() for t in T_GRID], (len(y_te), 1))


def ece(y, p, bins=10):
    e = np.linspace(0, 1, bins + 1); i = np.clip(np.digitize(p, e[1:-1]), 0, bins - 1)
    return sum(abs(p[i == k].mean() - y[i == k].mean()) * (i == k).mean() for k in range(bins) if (i == k).any())


m28 = (test["재령"] == 28).values
cols = [int(np.where(T_GRID == t)[0][0]) for t in EVAL_TARGETS]
pt = pd.DataFrame({
    "ANN 확률 (전체 재령)": {**pscore(Y_te.ravel(), P_te.ravel()), "ECE": ece(Y_te.ravel(), P_te.ravel())},
    "ANN 확률 (28일, 목표 21~40)": {**pscore(Y_te[m28][:, cols].ravel(), P_te[m28][:, cols].ravel()),
                                   "ECE": ece(Y_te[m28][:, cols].ravel(), P_te[m28][:, cols].ravel())},
    "기준: 배합 무시 (전체 재령)": {**pscore(Y_te.ravel(), P_base.ravel()), "ECE": ece(Y_te.ravel(), P_base.ravel())},
}).T.round(4)
display(pt)

pp, yy = P_te[m28][:, cols].ravel(), Y_te[m28][:, cols].ravel()
bins = [(0, .2), (.2, .5), (.5, .8), (.8, .95), (.95, 1.0001)]
rel = pd.DataFrame([{"확률 구간": f"{lo:.0%}~{min(hi, 1):.0%}", "건수": int(((pp >= lo) & (pp < hi)).sum()),
                     "평균 예측": pp[(pp >= lo) & (pp < hi)].mean(), "실제 달성": yy[(pp >= lo) & (pp < hi)].mean()}
                    for lo, hi in bins]).round(3)
print("확률 구간별 (재령 28일 테스트 × 목표 21·24·27·30·35·40)"); display(rel)
sg_te = sigma_of(CHOSEN, p_te, test["재령"].values)
z = np.abs(err) / sg_te
print(f"실제 오차가 ±1σ 안: {np.mean(z <= 1):.1%} (정규분포 68.3%), ±2σ 안: {np.mean(z <= 2):.1%} (정규분포 95.4%)")
by_t = pd.DataFrame([{"목표(MPa)": t, "실제 달성 비율": Y_te[m28, j].mean(), **pscore(Y_te[m28, j], P_te[m28, j])}
                     for t, j in zip(EVAL_TARGETS, cols)]).set_index("목표(MPa)").round(3)
print("목표 강도별 (재령 28일 테스트)"); display(by_t)
""")

md(r"""
## ⑥ 물리적 타당성

테스트 배합마다 재료 하나만 바꿔 ANN 예측이 공학 상식대로 움직이는지 봅니다 (28일 기준).
""")
code(r"""
base28 = test.drop_duplicates(subset=MATERIALS)[MATERIALS].assign(재령=28.0)
def frac_change(col, delta, expect):
    a = REG.predict(base28[FEATURES]); b2 = base28.copy(); b2[col] = b2[col] + delta
    d = REG.predict(b2[FEATURES]) - a
    ok = (d < 0) if expect == "감소" else (d > 0)
    return {"바꾼 것": f"{col} {delta:+g}", "기대": expect, "기대대로 움직인 배합": f"{ok.mean():.1%}", "평균 변화(MPa)": round(float(d.mean()), 2)}
phys = pd.DataFrame([frac_change("물", +10, "감소"), frac_change("시멘트", +20, "증가"), frac_change("고성능감수제", +2, "증가")])
a7 = REG.predict(base28.assign(재령=7.0)[FEATURES]); a28 = REG.predict(base28[FEATURES]); a91 = REG.predict(base28.assign(재령=91.0)[FEATURES])
phys.loc[len(phys)] = {"바꾼 것": "재령 7 → 28일", "기대": "증가", "기대대로 움직인 배합": f"{np.mean(a28 > a7):.1%}", "평균 변화(MPa)": round(float((a28 - a7).mean()), 2)}
phys.loc[len(phys)] = {"바꾼 것": "재령 28 → 91일", "기대": "증가", "기대대로 움직인 배합": f"{np.mean(a91 > a28):.1%}", "평균 변화(MPa)": round(float((a91 - a28).mean()), 2)}
print(f"테스트 배합 {len(base28)}개 기준"); display(phys)
""")

md(r"""
## ⑦ 중첩 교차검증 (`FAST_MODE = False`일 때만)

바깥 5겹(배합 기준)마다 안쪽 교차검증으로 설정을 **새로 고른 뒤** 바깥 묶음으로 평가합니다. 설정 고르기까지 포함한 성능이라 가장 엄격한 추정입니다.
""")
code(r"""
if FAST_MODE:
    nested = None
    print("FAST_MODE = True 라서 건너뜀 (False로 바꾸면 실행)")
else:
    rows = []
    for k, (a, b) in enumerate(GroupKFold(5).split(df, groups=groups_all)):
        s = GridSearchCV(ann(), grid, cv=GroupKFold(5), scoring="neg_root_mean_squared_error", n_jobs=-1)
        s.fit(df.loc[a, FEATURES], df.loc[a, "압축강도"], groups=groups_all[a])
        rows.append({"바깥 묶음": k + 1, "고른 설정": str(s.best_params_), **rscore(df.loc[b, "압축강도"], s.predict(df.loc[b, FEATURES]))})
    nested = pd.DataFrame(rows)
    display(nested)
    print(nested[["R²", "MAE(MPa)", "RMSE(MPa)"]].agg(["mean", "std"]).round(3))
""")

md(r"""
## ANN 예측 vs 공식 계산 (보조)

테스트 배합에서 ANN 예측과 공식 계산이 얼마나 다르고, 어느 쪽이 실제에 더 가까운지 봅니다.
""")
code(r"""
dif = p_te - f_te
closer = np.abs(err) < np.abs(f_te - y_te)
fdiff = pd.DataFrame({
    "건수": [len(dif)], "평균 차이 (ANN − 공식)": [dif.mean()], "평균 절대 차이": [np.abs(dif).mean()],
    "차이 10 MPa 이상 비율": [np.mean(np.abs(dif) >= 10)], "ANN이 실제에 더 가까운 비율": [closer.mean()],
}).round(3)
display(fdiff)
kind = t2["배합 종류"].values
print({k: f"ANN이 더 가까운 비율 {closer[kind == k].mean():.1%} (평균 절대 차이 {np.abs(dif[kind == k]).mean():.1f} MPa)" for k in ["시멘트만", "혼화재 사용"]})
""")

md(r"""
## 예측 프로그램

`predict(재료 7종, 목표강도, 재령=28)` → 달성 확률, ANN 예측 μ ± σ, 공식 계산 강도와 차이, W/B, 판정
""")
code(r"""
RANGES = df[FEATURES[:-1] + ["재령"]].agg(["min", "max"])
WB_RANGE = (float(wb_of(df).min()), float(wb_of(df).max()))


def verdict(p):
    return ("시험해 볼 가치가 높음" if p >= 0.8 else "가능성 있음 — 배합 조정 검토" if p >= 0.5
            else "가능성 낮음 — 배합 수정 권장" if p >= 0.2 else "달성 어려움")


def required_strength(fck, s=None):
    if s is None:
        return fck + 7.0 if fck < 21 else (fck + 8.5 if fck <= 35 else 1.1 * fck + 5.0)
    return max(fck + 1.34 * s, (fck - 3.5) + 2.33 * s) if fck <= 35 else max(fck + 1.34 * s, 0.9 * fck + 2.33 * s)


def predict(시멘트, 고로슬래그, 플라이애시, 물, 고성능감수제, 굵은골재, 잔골재, 목표강도, 재령=28, fck=None, s=None):
    mix = dict(시멘트=시멘트, 고로슬래그=고로슬래그, 플라이애시=플라이애시, 물=물, 고성능감수제=고성능감수제, 굵은골재=굵은골재, 잔골재=잔골재)
    d = pd.DataFrame([{**mix, "재령": float(재령)}])
    mu = float(REG.predict(d[FEATURES])[0]); sg = float(sigma_of(CHOSEN, np.array([mu]), np.array([float(재령)]))[0])
    p = float(prob(mu, 재령, 목표강도)[0, 0])
    wb = float(wb_of(d).iloc[0]); fs = float(formula_strength(wb, float(재령), A_FIT, B_FIT))
    warn = [f"{k} {v}: 학습 범위 {RANGES.loc['min', k]:g}~{RANGES.loc['max', k]:g} 밖" for k, v in {**mix, "재령": 재령}.items()
            if not RANGES.loc["min", k] <= v <= RANGES.loc["max", k]]
    if not WB_RANGE[0] <= wb <= WB_RANGE[1]:
        warn.append(f"W/B {wb:.3f}: 학습 범위 밖")
    out = {"목표 강도 달성 확률": f"{100 * p:.1f}%", "판정": verdict(p), "ANN 예측 강도": f"{mu:.1f} ± {sg:.1f} MPa",
           "공식 계산 강도 (비교용)": f"{fs:.1f} MPa", "차이 (ANN − 공식)": f"{mu - fs:+.1f} MPa", "W/B": round(wb, 3)}
    if fck is not None:
        fcr = required_strength(fck, s)
        out["fck 기준 확률"] = f"{100 * float(prob(mu, 재령, fck)[0, 0]):.1f}%"
        out["fcr 기준 확률"] = f"{100 * float(prob(mu, 재령, fcr)[0, 0]):.1f}% (fcr = {fcr:.1f} MPa)"
    out["경고"] = warn if warn else "없음"
    return out


EXAMPLES = [
    dict(시멘트=350, 고로슬래그=0, 플라이애시=0, 물=180, 고성능감수제=0, 굵은골재=1000, 잔골재=780, 목표강도=24),
    dict(시멘트=350, 고로슬래그=0, 플라이애시=0, 물=180, 고성능감수제=0, 굵은골재=1000, 잔골재=780, 목표강도=35),
    dict(시멘트=250, 고로슬래그=100, 플라이애시=50, 물=170, 고성능감수제=8, 굵은골재=950, 잔골재=780, 목표강도=40, fck=30, s=3.5),
]
for e in EXAMPLES:
    print(e, "\n →", predict(**e), "\n")
""")
code(r"""
RUN_INTERACTIVE = False   # True 로 바꾸고 실행
if RUN_INTERACTIVE:
    vals = [float(input(f"{k} (kg/m³): ").replace(",", "")) for k in MATERIALS]
    t = float(input("목표 강도 (MPa): ")); age = float(input("재령 (일, 기본 28): ") or 28)
    print(predict(*vals, 목표강도=t, 재령=age))
""")

md("## 저장과 HTML 모델용 내보내기")
code(r"""
sc, m = REG.named_steps["scale"], REG.named_steps["mlp"]
pct = lambda v: round(float(v), 4)
export = {
    "version": 2, "chosen": CHOSEN, "features": FEATURES, "log_age": bool(LOG_AGE),
    "reg": {"scale": sc.scale_.tolist(), "min": sc.min_.tolist(), "activation": m.activation, "out_activation": m.out_activation_,
            "coefs": [c.tolist() for c in m.coefs_], "intercepts": [b.tolist() for b in m.intercepts_], "best": {k: str(v) for k, v in BEST.items()},
            "n_weights": int(n_w)},
    "sigma": SIGMA, "sigma_var": {"coef": SIG_COEF, "min": SIG_MIN},
    "iso": {"x": ISO.X_thresholds_.tolist(), "y": ISO.y_thresholds_.tolist()} if ISO is not None else None,
    "formula": {"A": float(A_FIT), "B": float(B_FIT)},
    "ranges": {**{k: [float(RANGES.loc["min", k]), float(RANGES.loc["max", k])] for k in RANGES.columns}, "W/B": list(WB_RANGE)},
    "data": {"rows": int(len(df)), "mixes": int(df["배합번호"].nunique()), "train": int(len(train)), "test": int(len(test))},
    "validation": {
        "test": {k: pct(v) for k, v in comp.loc["ANN (테스트)"].items()}, "train": {k: pct(v) for k, v in comp.loc["ANN (학습용)"].items()},
        "baselines": {i: {k: pct(v) for k, v in comp.loc[i].items()} for i in comp.index if not i.startswith("ANN")},
        "cv": {"runs": int(len(cvres)), **{k: {"mean": pct(cv_sum.loc["mean", k]), "std": pct(cv_sum.loc["std", k])} for k in cv_sum.columns}},
        "ci": {k: [pct(ci.loc[k, "95% 하한"]), pct(ci.loc[k, "95% 상한"])] for k in ci.index},
        "prob": {k: pct(v) for k, v in pt.loc["ANN 확률 (28일, 목표 21~40)"].items()},
        "reliability": rel.to_dict("records"),
        "within5": pct(np.mean(np.abs(err) <= 5)), "within10": pct(np.mean(np.abs(err) <= 10)),
        "phys": phys.to_dict("records"),
        "formula_diff": {k: pct(v) for k, v in fdiff.iloc[0].items()},
    },
    "checks": [{"mix": {k: v for k, v in e.items() if k in MATERIALS}, "target": e["목표강도"], "age": 28,
                "p": float(prob(float(REG.predict(pd.DataFrame([{**{k: e[k] for k in MATERIALS}, "재령": 28.0}])[FEATURES])[0]), 28, e["목표강도"])[0, 0])}
               for e in EXAMPLES],
}
with open("model_export.json", "w", encoding="utf-8") as f:
    json.dump(export, f, ensure_ascii=False)
joblib.dump({"reg": REG, "iso": ISO, **{k: v for k, v in export.items() if k not in ("reg", "iso")}}, "concrete_ann_model.joblib")
print("저장: model_export.json, concrete_ann_model.joblib | 확률 방법:", CHOSEN)
if IN_COLAB:
    colab_files.download("model_export.json")
""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}, "colab": {"provenance": []}}
nbf.write(nb, ROOT / "concrete_ann_prob.ipynb")
print("written", len(C), "cells")
