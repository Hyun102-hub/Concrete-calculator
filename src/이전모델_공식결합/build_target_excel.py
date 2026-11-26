"""목표 강도 달성 확률 프로젝트 엑셀: 원자료 1,030건 + 공식 값(수식) + 처리 과정 + 모델 비교 결과
python3 data_build/build_target_excel.py   (concrete_target_prob_실행결과.ipynb, model_export.json 이 먼저 있어야 함)
"""
import io
import json
from pathlib import Path

import nbformat
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["시멘트", "고로슬래그", "플라이애시", "물", "고성능감수제", "굵은골재", "잔골재", "재령", "압축강도"]
F = Font(name="Arial", size=10); B = Font(name="Arial", size=10, bold=True); T = Font(name="Arial", size=13, bold=True)
BLUE = Font(name="Arial", size=10, color="0000FF")
H = PatternFill("solid", fgColor="DDEBF7"); KEY = PatternFill("solid", fgColor="FFF2CC")

e = json.loads((ROOT / "model_export.json").read_text(encoding="utf-8"))
raw = pd.read_excel(ROOT / "data_build/raw/Concrete_Data.xls")
raw_cols = [c.strip() for c in raw.columns]
raw.columns = NAMES
df = raw.copy()
df["배합번호"] = df.groupby(NAMES[:7]).ngroup() + 1
tr, te = next(GroupShuffleSplit(1, test_size=0.30, random_state=42).split(df, groups=df["배합번호"]))
df["구분"] = "학습"
df.loc[df.index[te], "구분"] = "테스트"
assert (df["구분"] == "학습").sum() == e["data"]["train"] and (df["구분"] == "테스트").sum() == e["data"]["test"]
df["완전중복"] = df.duplicated(keep=False)
df["같은배합재령"] = df.duplicated(subset=NAMES[:8], keep=False)

tabs = {}
nb = nbformat.read(ROOT / "concrete_target_prob_실행결과.ipynb", as_version=4)
for i, c in enumerate(nb.cells):
    for o in c.get("outputs", []):
        h = o.get("data", {}).get("text/html")
        if h and "<table" in h:
            tabs.setdefault(i, []).append(pd.read_html(io.StringIO(h))[0])


def flat(t):
    t = t.copy()
    t.columns = [c[1] if isinstance(c, tuple) and not str(c[1]).startswith("Unnamed") else (c[0] if isinstance(c, tuple) else c) for c in t.columns]
    t.columns = ["구분" if str(c).startswith("Unnamed") or c == "" else c for c in t.columns]
    return t


def header(ws, cols, row=1, freeze=True):
    for j, c in enumerate(cols, 1):
        x = ws.cell(row, j, c); x.font, x.fill = B, H
        x.alignment = Alignment(wrap_text=True, vertical="center")
    if freeze:
        ws.freeze_panes = ws.cell(row + 1, 1)


def put(ws, r, c, v, fmt=None, font=F, fill=None):
    x = ws.cell(r, c, v); x.font = font
    if fmt: x.number_format = fmt
    if fill: x.fill = fill
    return x


def widths(ws, w):
    for j, v in enumerate(w, 1):
        ws.column_dimensions[get_column_letter(j)].width = v


wb = Workbook()
# ── 개요 ──
ws = wb.active; ws.title = "개요"
st = e["metrics"]["strength"]; pa, p28 = e["metrics"]["prob_all"], e["metrics"]["prob_28"]
lines = [
    ("콘크리트 목표 강도 달성 확률 예측 (ANN + 강도 공식)", T), ("", F), ("데이터", B),
    ("UCI Machine Learning Repository, \"Concrete Compressive Strength\", I-Cheng Yeh, DOI 10.24432/C5PK67, CC BY 4.0", F),
    (f"원자료 {len(raw):,}건 전부 사용 (완전히 같은 행·같은 배합·재령 행도 지우거나 평균 내지 않음), 서로 다른 배합 {df['배합번호'].nunique()}개", F),
    (f"배합 기준 그룹 분할: 학습 {e['data']['train']}건 / 테스트 {e['data']['test']}건 (random_state = 42)", F),
    ("", F), ("공식", B),
    (f"Abrams 법칙 f = A ÷ B^(W/B): A = {e['formula']['A']:.4f}, B = {e['formula']['B']:.4f} (학습용 데이터로만 맞춤)", F),
    ("ACI 209 재령 계수 g(t) = t ÷ (4 + 0.85t), 공식 예측 강도 = Abrams 28일 강도 × g(t) ÷ g(28)", F),
    ("", F), ("결과 (concrete_target_prob_실행결과.ipynb의 값)", B),
    *[(f"{r['모델']}: 테스트 R² {r['R²']:.3f}, MAE {r['MAE(MPa)']:.2f} MPa, RMSE {r['RMSE(MPa)']:.2f} MPa", F) for r in st],
    (f"확률 계산 방법: {e['chosen']} (테스트 로그손실 최소) — 전체 재령 Brier {pa['Brier']:.4f}, 로그손실 {pa['로그손실']:.4f}, "
     f"ECE {pa['ECE']:.4f}, AUC {pa['AUC']:.4f} / 28일 로그손실 {p28['로그손실']:.4f}", F),
    ("", F), ("시트", B),
    ("원자료: UCI 파일 그대로 / 학습데이터: 1,030건 + 배합번호·학습/테스트 구분 + 공식 값(수식)", F),
    ("처리과정: 점검 결과 / 공식계수: A·B·σ (학습데이터 시트의 수식이 참조) / 모델비교: 노트북 실행 결과표 / 변수설명", F),
    ("", F), ("한계", B),
    ("확률은 모델의 예측 오차를 바탕으로 한 값이며, 같은 배합의 배치별 편차(현장 품질 변동)는 데이터에 거의 없어 반영되지 않음", F),
]
for i, (t, f) in enumerate(lines, 1):
    put(ws, i, 1, t, font=f)
ws.column_dimensions["A"].width = 130

# ── 공식계수 ──
wc = wb.create_sheet("공식계수")
header(wc, ["항목", "값", "설명"])
coef = [("A", e["formula"]["A"], "Abrams 법칙 계수 A (학습용 데이터로 맞춤)"),
        ("B", e["formula"]["B"], "Abrams 법칙 계수 B"),
        ("g(28)", "=28/(4+0.85*28)", "ACI 209 28일 재령 계수"),
        ("σ 고정", e["sigma"], "교차검증 예측 오차의 표준편차 (MPa)"),
        ("σ 가변 c0", e["sigma_var"]["coef"][0], "σ = √(π/2) × (c0 + c1·μ + c2·재령계수), 최소 2 MPa"),
        ("σ 가변 c1", e["sigma_var"]["coef"][1], ""), ("σ 가변 c2", e["sigma_var"]["coef"][2], "")]
for i, (k, v, s) in enumerate(coef, 2):
    put(wc, i, 1, k); put(wc, i, 2, v, "0.0000", font=BLUE if not str(v).startswith("=") else F); put(wc, i, 3, s)
widths(wc, [14, 14, 60])

# ── 원자료 ──
w0 = wb.create_sheet("원자료")
header(w0, ["원자료행"] + [f"{k}\n({c})" for k, c in zip(NAMES, raw_cols)])
w0.row_dimensions[1].height = 58
for i, r in enumerate(raw.itertuples(index=False), 2):
    put(w0, i, 1, i)
    for j, v in enumerate(r, 2):
        put(w0, i, j, float(v), "0.00")
widths(w0, [8] + [14] * 9)

# ── 학습데이터 (공식 값은 수식) ──
w1 = wb.create_sheet("학습데이터")
cols = ["원자료행", "배합번호", "구분"] + NAMES + ["결합재", "W/C", "W/B", "재령계수", "공식강도(MPa)", "완전중복", "같은 배합·재령"]
header(w1, cols); w1.row_dimensions[1].height = 32
for i, r in enumerate(df.itertuples(index=False), 2):
    put(w1, i, 1, i); put(w1, i, 2, int(r.배합번호)); put(w1, i, 3, r.구분)
    for j, k in enumerate(NAMES, 4):
        put(w1, i, j, float(getattr(r, k)), "0" if k == "재령" else "0.00")
    put(w1, i, 13, f"=D{i}+E{i}+F{i}", "0.0")                # 결합재
    put(w1, i, 14, f"=G{i}/D{i}", "0.000")                   # W/C
    put(w1, i, 15, f"=G{i}/M{i}", "0.000")                   # W/B
    put(w1, i, 16, f"=K{i}/(4+0.85*K{i})", "0.0000")         # 재령계수
    put(w1, i, 17, f"=공식계수!$B$2/공식계수!$B$3^O{i}*P{i}/공식계수!$B$4", "0.00")
    put(w1, i, 18, "예" if r.완전중복 else ""); put(w1, i, 19, "예" if r.같은배합재령 else "")
N = len(df) + 1
w1.auto_filter.ref = f"A1:S{N}"
widths(w1, [8, 8, 7] + [9] * 9 + [9, 8, 8, 9, 11, 8, 10])

# ── 처리과정 ──
w2 = wb.create_sheet("처리과정")
chk = tabs[8][0].iloc[:, 1:]
header(w2, list(chk.columns))
for i, r in enumerate(chk.itertuples(index=False), 2):
    for j, v in enumerate(r, 1):
        put(w2, i, j, v if isinstance(v, str) else int(v))
r0 = len(chk) + 3
put(w2, r0, 1, "수식으로 다시 센 값 (학습데이터 시트 기준)", font=B)
for k, (lab, f) in enumerate([("전체 행", f"=COUNTA(학습데이터!A2:A{N})"), ("학습", f'=COUNTIF(학습데이터!C2:C{N},"학습")'),
                              ("테스트", f'=COUNTIF(학습데이터!C2:C{N},"테스트")'), ("완전중복에 해당하는 행(첫 행 포함)", f'=COUNTIF(학습데이터!R2:R{N},"예")'),
                              ("같은 배합·재령인 행(완전중복 포함)", f'=COUNTIF(학습데이터!S2:S{N},"예")'),
                              ("공식강도 ↔ 압축강도 상관계수", f"=CORREL(학습데이터!Q2:Q{N},학습데이터!L2:L{N})")], r0 + 1):
    put(w2, k, 1, lab); put(w2, k, 2, f, "0.000" if "상관" in lab else "0")
widths(w2, [42, 12, 14])

# ── 모델비교 ──
w3 = wb.create_sheet("모델비교")
put(w3, 1, 1, "아래 값은 concrete_target_prob_실행결과.ipynb를 실행해 나온 값입니다 (random_state = 42, FAST_MODE = True).", font=B)
row = 3


def block(title, t, fmt="0.000"):
    global row
    t = flat(t)
    put(w3, row, 1, title, font=T)
    header(w3, [str(c) for c in t.columns], row=row + 1, freeze=False)
    for k, rr in enumerate(t.itertuples(index=False), row + 2):
        for j, v in enumerate(rr, 1):
            isnum = isinstance(v, (int, float)) and not isinstance(v, bool)
            put(w3, k, j, float(v) if isnum else (v if pd.notna(v) else ""), fmt if isnum else None)
    row += 3 + len(t)


block("① 강도 예측 비교 (테스트)", tabs[14][0])
block("② 배합 종류별 RMSE (MPa)", tabs[14][1], "0.00")
block("③ 확률 예측 — 테스트 전체 재령 (목표 5~80 MPa)", tabs[20][0], "0.0000")
block("④ 확률 예측 — 테스트 재령 28일", tabs[20][1], "0.0000")
by_t = flat(tabs[20][2]); by_t.iloc[:, 0] = by_t.iloc[:, 0].ffill()
block("⑤ 목표 강도별 (재령 28일 테스트)", by_t)
eff = flat(tabs[23][0]); eff.iloc[:, 0] = eff.iloc[:, 0].ffill()
block("⑥ 활용 효과: 확률 기준으로 걸러 시험할 때 (재령 28일 테스트)", eff)
widths(w3, [30, 22, 14, 14, 14, 14, 14])

# ── 변수설명 ──
w4 = wb.create_sheet("변수설명")
header(w4, ["이름", "원자료 열 / 식", "단위", "쓰임"])
info = [(k, c, "kg/m³", "입력") for k, c in zip(NAMES[:7], raw_cols[:7])] + [
    ("재령", raw_cols[7], "일", "입력"), ("압축강도", raw_cols[8], "MPa", "정답 (입력 아님)"),
    ("W/C", "물 ÷ 시멘트", "-", "공식 결합 ANN 입력"), ("W/B", "물 ÷ (시멘트 + 고로슬래그 + 플라이애시)", "-", "공식 결합 ANN 입력"),
    ("재령계수", "t ÷ (4 + 0.85t)  (ACI 209)", "-", "공식 결합 ANN 입력"),
    ("공식강도", "A ÷ B^(W/B) × g(t) ÷ g(28)  (Abrams + ACI 209)", "MPa", "공식 결합 ANN 입력, ① 공식만 모델"),
    ("목표강도", "사용자가 입력", "MPa", "확률 계산 (B 방식에서는 분류 ANN 입력)")]
for i, r in enumerate(info, 2):
    for j, v in enumerate(r, 1):
        put(w4, i, j, v)
widths(w4, [12, 58, 8, 34])

out = ROOT / "콘크리트_목표강도_확률_데이터.xlsx"
wb.save(out)
print(out)
