"""반복 무작위 7:3 검증 결과 PDF (validation_runs.csv, validation_summary.json 에서 모든 숫자를 읽음)
python3 build_validation_pdf.py <출력 PDF>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import koreanize_matplotlib
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

H = Path(__file__).resolve().parent
OUT = Path(sys.argv[1])
FIG = H / "_figs"; FIG.mkdir(exist_ok=True)
rep = pd.read_csv(H / "validation_runs.csv")
S = json.loads((H / "validation_summary.json").read_text(encoding="utf-8"))
N = S["n"]; CL = S["clean"]; SUM = S["summary"]; CONV = pd.DataFrame(S["convergence"])
RND, GRP = "무작위", "배합 기준"
g = lambda sch, stat, m: SUM[sch][stat][m]
FD = Path(koreanize_matplotlib.__file__).parent / "fonts"
pdfmetrics.registerFont(TTFont("NG", str(FD / "NanumGothic.ttf"))); pdfmetrics.registerFont(TTFont("NGB", str(FD / "NanumGothicBold.ttf")))
plt.rcParams["axes.unicode_minus"] = False
BLUE, ORANGE, INK, INK2, LINE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#d9d8d4"


def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(LINE)
    ax.tick_params(colors=INK2, labelsize=8.5); ax.grid(axis="y", color=LINE, lw=0.6); ax.set_axisbelow(True)


# ── 그림 1: R²·MAE 분포 ──
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8), dpi=200)
for i, (m, lab) in enumerate([("R²", "검증 R²"), ("MAE", "검증 MAE (MPa)")]):
    for sch, c in [(RND, BLUE), (GRP, ORANGE)]:
        ax[i].hist(rep[rep["방식"] == sch][m], bins=35, color=c, alpha=0.65, label=f"{sch} 7:3", edgecolor="white", linewidth=0.4)
    ax[i].set_xlabel(lab, fontsize=9, color=INK2); ax[i].set_ylabel("횟수", fontsize=9, color=INK2); style(ax[i])
ax[0].legend(fontsize=8, frameon=False)
plt.tight_layout(); fig.savefig(FIG / "dist.png"); plt.close(fig)

# ── 그림 2: 누적 평균 (반복 횟수별 안정성) ──
fig, ax = plt.subplots(figsize=(7.4, 2.2), dpi=200)
for sch, c in [(RND, BLUE), (GRP, ORANGE)]:
    r2 = rep[rep["방식"] == sch].sort_values("회차")["R²"].values; n = np.arange(1, len(r2) + 1)
    ax.plot(n, np.cumsum(r2) / n, color=c, lw=1.8, label=f"{sch} 7:3")
for x in (100,):
    ax.axvline(x, color=INK2, lw=0.8, ls=":"); ax.text(x * 1.05, ax.get_ylim()[0] + 0.004, "100회", fontsize=8, color=INK2)
ax.set_xscale("log"); ax.set_xlabel("반복 횟수 (로그 눈금)", fontsize=9, color=INK2); ax.set_ylabel("누적 평균 R²", fontsize=9, color=INK2)
ax.legend(fontsize=8, frameon=False); style(ax)
plt.tight_layout(); fig.savefig(FIG / "conv.png"); plt.close(fig)

# ── 그림 3: 대표 회차 (R²가 중앙값인 회차) 예측 vs 실제 ──
raw = pd.read_excel(Path("/home/user/concrete_strength/data_build/raw/Concrete_Data.xls"))
NAMES = ["시멘트", "고로슬래그", "플라이애시", "물", "고성능감수제", "굵은골재", "잔골재", "재령", "압축강도"]; raw.columns = NAMES
d1 = raw.drop_duplicates().reset_index(drop=True); dc = d1[~d1.duplicated(subset=NAMES[:8], keep=False)].reset_index(drop=True)
rr = rep[rep["방식"] == RND].reset_index(drop=True)
med = rr.iloc[(rr["R²"] - rr["R²"].median()).abs().idxmin()]; seed = int(med["회차"]) - 1
a, b = train_test_split(np.arange(len(dc)), test_size=0.3, random_state=seed)
pre = ColumnTransformer([("재료", "passthrough", NAMES[:7]), ("재령", FunctionTransformer(np.log), ["재령"])])
m = Pipeline([("pre", pre), ("scale", MinMaxScaler()), ("mlp", MLPRegressor(hidden_layer_sizes=(64,), activation="tanh", alpha=1e-4,
              learning_rate_init=0.01, max_iter=5000, early_stopping=True, validation_fraction=0.15, n_iter_no_change=50, random_state=seed))])
m.fit(dc.iloc[a][NAMES[:8]], dc.iloc[a]["압축강도"]); y = dc.iloc[b]["압축강도"].values; p = m.predict(dc.iloc[b][NAMES[:8]])
r2_check = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
assert abs(r2_check - med["R²"]) < 1e-9, (r2_check, med["R²"])    # 저장된 결과와 같은 회차인지 확인
fig, ax = plt.subplots(figsize=(3.6, 3.4), dpi=200)
ax.fill_between([0, 90], [-5, 85], [5, 95], color=BLUE, alpha=0.10, lw=0)
ax.scatter(y, p, s=9, color=BLUE, alpha=0.75, linewidths=0); ax.plot([0, 90], [0, 90], color=INK, lw=0.8)
ax.set_xlim(0, 90); ax.set_ylim(0, 90); ax.set_xlabel("실제 강도 (MPa)", fontsize=9, color=INK2); ax.set_ylabel("ANN 예측 (MPa)", fontsize=9, color=INK2)
ax.text(3, 84, f"{int(med['회차'])}회차 · R² {med['R²']:.3f}\n띠 = ±5 MPa", fontsize=8, color=INK, va="top"); style(ax); ax.grid(axis="x", color=LINE, lw=0.6)
plt.tight_layout(); fig.savefig(FIG / "scatter.png"); plt.close(fig)

# ── 문서 ──
ST = {
    "title": ParagraphStyle("t", fontName="NGB", fontSize=20, leading=27, textColor=INK),
    "sub": ParagraphStyle("s", fontName="NG", fontSize=10.5, leading=16, textColor=INK2),
    "h1": ParagraphStyle("h1", fontName="NGB", fontSize=14, leading=20, textColor=INK, spaceBefore=8, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="NGB", fontSize=11, leading=16, textColor=BLUE, spaceBefore=8, spaceAfter=4),
    "p": ParagraphStyle("p", fontName="NG", fontSize=9.6, leading=15.5, textColor=INK, spaceAfter=4),
    "li": ParagraphStyle("li", fontName="NG", fontSize=9.6, leading=15, textColor=INK, leftIndent=11, firstLineIndent=-8, spaceAfter=2),
    "small": ParagraphStyle("sm", fontName="NG", fontSize=8.2, leading=12, textColor=INK2, spaceAfter=3),
    "cell": ParagraphStyle("c", fontName="NG", fontSize=8.6, leading=12, textColor=INK),
    "cellb": ParagraphStyle("cb", fontName="NGB", fontSize=8.6, leading=12, textColor=INK),
    "big": ParagraphStyle("big", fontName="NGB", fontSize=19, leading=23, textColor=BLUE, alignment=TA_CENTER),
    "bigo": ParagraphStyle("bigo", fontName="NGB", fontSize=19, leading=23, textColor=ORANGE, alignment=TA_CENTER),
    "bigl": ParagraphStyle("bigl", fontName="NG", fontSize=8.3, leading=11, textColor=INK2, alignment=TA_CENTER),
}
W = A4[0] - 36 * mm
P = lambda t, s="p": Paragraph(t, ST[s]); LI = lambda t: Paragraph("• " + t, ST["li"])


def T(rows, widths, right=(), hl=None):
    data = [[Paragraph(str(v), ST["cellb"] if i == 0 else ST["cell"]) for v in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=1)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(LINE)), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7f0f4")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]
    st += [("ALIGN", (c, 1), (c, -1), "RIGHT") for c in right]
    if hl: st += [("BACKGROUND", (0, r), (-1, r), colors.HexColor("#eef6ee")) for r in hl]
    t.setStyle(TableStyle(st)); return t


def IM(name, frac, h_over_w):
    return Image(str(FIG / name), width=W * frac, height=W * frac * h_over_w)


f3 = lambda v: f"{v:.3f}"; f2 = lambda v: f"{v:.2f}"; pc = lambda v: f"{100 * v:.1f}%"
story = [P("콘크리트 강도 ANN 성능 검증 결과", "title"), P(f"반복 무작위 7:3 검증 · {N:,}회 · 중복·강도 충돌 데이터를 뺀 {CL['검증용 데이터']:,}건", "sub"), Spacer(1, 12)]
tiles = Table([[P(f3(g(RND, "평균", "R²")), "big"), P(f"{g(RND, '2.5%', 'R²'):.2f}~{g(RND, '97.5%', 'R²'):.2f}", "big"),
                P(f"{f2(g(RND, '평균', 'MAE'))} MPa", "big"), P(f3(g(GRP, "평균", "R²")), "bigo")],
               [P(f"무작위 7:3 · 검증 R² 평균", "bigl"), P("무작위 7:3 · R² 95% 범위", "bigl"), P("무작위 7:3 · 평균 오차(MAE)", "bigl"),
                P("배합 기준 7:3 · 검증 R² 평균", "bigl")]], colWidths=[W / 4] * 4)
tiles.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(LINE)), ("LINEAFTER", (0, 0), (2, -1), 0.5, colors.HexColor(LINE)),
                           ("TOPPADDING", (0, 0), (-1, 0), 9), ("BOTTOMPADDING", (0, 1), (-1, 1), 9)]))
story += [tiles, Spacer(1, 10), P("요약", "h2"),
          LI(f"데이터를 무작위로 7:3으로 나눠 7로 학습하고 3으로 검증하는 과정을 <b>{N:,}번</b> 반복했습니다."),
          LI(f"검증 데이터에서 ANN 예측과 실제 강도의 일치도는 <b>R² 평균 {f3(g(RND, '평균', 'R²'))}</b>(95% 범위 {f3(g(RND, '2.5%', 'R²'))}~{f3(g(RND, '97.5%', 'R²'))})였고, "
             f"예측의 <b>{pc(g(RND, '평균', '±5 MPa 이내'))}</b>가 실제 강도 ±5 MPa 안, <b>{pc(g(RND, '평균', '±10 MPa 이내'))}</b>가 ±10 MPa 안에 들었습니다."),
          LI(f"같은 배합이 학습·검증에 섞이지 않게 나누면(처음 보는 배합만 검증) R² 평균 <b>{f3(g(GRP, '평균', 'R²'))}</b>로 조금 낮아집니다. 새 배합을 예측하는 실제 사용에는 이 값이 더 가깝습니다."),
          LI(f"반복 횟수에 따른 변화를 보면 <b>100회면 평균과 95% 범위 모두 충분히 안정</b>되며, {N:,}회는 이를 확인한 결과입니다 (5장).")]

story += [P("1. 검증에 쓴 데이터", "h1"),
          T([["단계", "건수", "처리"], ["UCI Concrete Compressive Strength 원자료", f"{CL['원자료']:,}", "-"],
             ["배합·재령·강도가 완전히 같은 행", f"{CL['완전 중복 제외']}", "하나만 남기고 제외"],
             [f"배합·재령이 같은데 강도가 다른 행 ({CL['강도 충돌 조건 수']}개 조건)", f"{CL['강도 충돌 행 제외']}", "어느 값이 맞는지 알 수 없어 모두 제외"],
             ["검증용 데이터", f"{CL['검증용 데이터']:,}", f"서로 다른 배합 {CL['배합 수']}개"]], [0.55, 0.12, 0.33], right=(1,), hl=[4]),
          P("'배합이 같다'는 재료 7종과 재령이 모두 같은 경우로 보았습니다. 재령과 무관하게 재료만 같은 행까지 빼면 같은 배합을 여러 재령에서 잰 정상 데이터까지 지워져 246건만 남기 때문입니다.", "small")]

story += [P("2. 검증 방법", "h1"),
          T([["순서", "내용"], ["①", f"검증용 데이터 {CL['검증용 데이터']:,}건을 무작위로 7:3으로 나눔 (학습 약 690건, 검증 약 297건)"],
             ["②", "7로 ANN 학습: 입력 = 재료 7종 + 재령(로그 변환), MinMaxScaler → 신경망(은닉층 뉴런 64개, tanh, 규제 0.0001, 학습률 0.01, 조기 종료)"],
             ["③", "3의 배합·재령을 모델에 넣어 나온 예측 강도를 실제 강도와 비교 (R², MAE, RMSE, ±5·±10 MPa 이내 비율 등)"],
             ["④", f"①~③을 나누는 방식을 바꿔 가며 {N:,}번 반복하고, 결과의 평균·표준편차·95% 범위를 계산"]], [0.08, 0.92]),
          P("비교: 배합 기준 7:3 분할", "h2"),
          P("무작위로 나누면 같은 배합의 다른 재령(예: 7일과 28일)이 학습과 검증에 나뉘어 들어갈 수 있어, 모델이 이미 본 배합을 맞히는 경우가 생깁니다. "
            "그래서 <b>같은 배합은 학습이나 검증 한쪽에만</b> 들어가게 나누는 방식도 같은 횟수만큼 반복했습니다. 새 배합을 예측하는 실제 사용 상황에 더 가깝습니다."),
          P("ANN 구조와 설정은 계산기 모델과 같으며 매 회차 가중치만 새로 학습했습니다. 설정은 전체 데이터의 학습용 70%로 미리 고른 값이라, 결과가 아주 조금 유리할 수 있습니다.", "small")]
story.append(PageBreak())

rows = [["지표", "뜻", "무작위 7:3 평균 (95% 범위)", "배합 기준 7:3 평균 (95% 범위)"]]
for mkey, lab, fmt in [("R²", "1에 가까울수록 예측이 실제와 일치", f3), ("MAE", "평균적으로 빗나간 강도 (MPa)", f2), ("RMSE", "큰 오차에 벌점을 더 준 오차 (MPa)", f2),
                       ("±5 MPa 이내", "예측이 실제 ±5 MPa 안에 든 비율", pc), ("±10 MPa 이내", "예측이 실제 ±10 MPa 안에 든 비율", pc),
                       ("상대오차 중앙값", "실제 강도 대비 오차 (중앙값)", pc), ("평균 편향", "+면 높게, −면 낮게 예측 (MPa)", f2), ("학습 R²", "학습 데이터에서의 R² (과적합 확인용)", f3)]:
    rows.append([mkey, lab] + [f"<b>{fmt(g(s, '평균', mkey))}</b> ({fmt(g(s, '2.5%', mkey))} ~ {fmt(g(s, '97.5%', mkey))})" for s in (RND, GRP)])
story += [P(f"3. 결과 ({N:,}회 반복)", "h1"), T(rows, [0.15, 0.31, 0.27, 0.27], hl=[1]),
          P(f"95% 범위 = {N:,}번 결과를 크기순으로 놓았을 때 가운데 95%가 들어가는 구간 (2.5%~97.5%). "
            f"무작위 7:3에서 R²가 가장 낮았던 회차 {f3(g(RND, '최소', 'R²'))}, 가장 높았던 회차 {f3(g(RND, '최대', 'R²'))}.", "small"),
          P("R²·MAE 분포", "h2"), IM("dist.png", 1.0, 2.8 / 7.4),
          P("두 분포 모두 한쪽으로 크게 치우치지 않고 모여 있어, 데이터를 어떻게 나눠도 성능이 비슷하게 나옵니다. 배합 기준 분할(주황)이 무작위 분할(파랑)보다 조금 왼쪽(낮은 R²)에 있습니다.", "small")]
story.append(PageBreak())
story += [P("4. 예측이 실제와 얼마나 비슷한가 (대표 회차)", "h1"),
          Table([[IM("scatter.png", 0.47, 3.4 / 3.6),
                  [P(f"무작위 7:3 {N:,}회 중 <b>R²가 중앙값에 가장 가까운 {int(med['회차'])}회차</b>의 검증 데이터 {len(y)}건입니다."),
                   LI("점 하나가 검증 배합 하나: 가로 = 실제 강도, 세로 = ANN 예측"),
                   LI("대각선에 가까울수록 정확하고, 파란 띠는 실제 ±5 MPa 범위"),
                   LI(f"이 회차: R² {med['R²']:.3f}, MAE {med['MAE']:.2f} MPa, ±5 MPa 이내 {pc(med['±5 MPa 이내'])}"),
                   LI("고강도(60 MPa 이상)에서 대각선 아래로 내려가는 점이 많아, 고강도는 낮게 예측하는 경향이 있습니다.")]]],
                colWidths=[0.5 * W, 0.5 * W], style=[("VALIGN", (0, 0), (-1, -1), "MIDDLE")])]

cv = CONV
crow = [["반복 횟수"] + [f"{s} R² 평균 (± 표준오차)" for s in (RND, GRP)] + [f"{s} 95% 범위" for s in (RND, GRP)]]
for n in sorted(cv["반복 횟수"].unique()):
    r = {s: cv[(cv["방식"] == s) & (cv["반복 횟수"] == n)].iloc[0] for s in (RND, GRP)}
    crow.append([f"{n:,}"] + [f"{r[s]['R² 평균']:.3f} (± {r[s]['평균의 표준오차']:.4f})" for s in (RND, GRP)] +
                [f"{r[s]['95% 범위 하한']:.3f} ~ {r[s]['95% 범위 상한']:.3f}" for s in (RND, GRP)])
hl = [i for i, r in enumerate(crow) if r[0] in ("100", f"{N:,}")]
r100 = {s: cv[(cv["방식"] == s) & (cv["반복 횟수"] == 100)].iloc[0] for s in (RND, GRP)}
rN = {s: cv[(cv["방식"] == s) & (cv["반복 횟수"] == N)].iloc[0] for s in (RND, GRP)}
dm = max(abs(r100[s]["R² 평균"] - rN[s]["R² 평균"]) for s in (RND, GRP))
dlo = max(abs(r100[s]["95% 범위 하한"] - rN[s]["95% 범위 하한"]) for s in (RND, GRP))
story += [Spacer(1, 8), P("5. 반복 횟수는 충분한가", "h1"), T(crow, [0.12, 0.22, 0.22, 0.22, 0.22], hl=hl),
          P("표준오차 = 표준편차 ÷ √반복 횟수. 평균 R²가 이 값의 약 2배 범위 안에서 흔들릴 수 있다는 뜻입니다.", "small"),
          IM("conv.png", 1.0, 2.2 / 7.4),
          P("판단", "h2"),
          LI(f"<b>평균 R²</b>: 100회와 {N:,}회의 차이가 최대 {dm:.3f}로 매우 작고, 100회의 표준오차도 약 {max(r100[s]['평균의 표준오차'] for s in (RND, GRP)):.3f}입니다. "
             "평균 성능을 보고하는 목적이라면 <b>100회로 충분</b>합니다."),
          LI(f"<b>95% 범위(양 끝)</b>: 100회와 {N:,}회의 범위 하한 차이도 최대 {dlo:.3f}로 작습니다. 다만 50회 이하에서는 범위가 크게 흔들립니다"
             f"(배합 기준 하한이 30회 {cv[(cv['방식'] == GRP) & (cv['반복 횟수'] == 30)].iloc[0]['95% 범위 하한']:.3f}, 50회 {cv[(cv['방식'] == GRP) & (cv['반복 횟수'] == 50)].iloc[0]['95% 범위 하한']:.3f}). "
             "범위는 양 끝 몇 개 회차가 정하기 때문에 평균보다 더 많은 반복이 필요합니다."),
          LI(f"<b>결론</b>: 이 데이터에서는 <b>100회면 평균과 범위 모두 충분</b>했고, {N:,}회는 그것을 확인하는 용도입니다. 보고서 숫자는 {N:,}회 결과를 썼습니다. "
             "반복을 더 늘려도 숫자는 거의 바뀌지 않으며, 반복 횟수가 모델 성능 자체를 높여 주지는 않습니다.")]
story += [Spacer(1, 6), P("6. 해석과 주의할 점", "h1"),
          LI("<b>무작위 7:3 vs 배합 기준 7:3</b>: 무작위 분할은 같은 배합의 다른 재령이 학습에 들어가 있을 수 있어 성능이 조금 높게 나옵니다. "
             f"발표에서는 두 값을 함께 제시하는 것을 권합니다 (무작위 {f3(g(RND, '평균', 'R²'))}, 처음 보는 배합 {f3(g(GRP, '평균', 'R²'))})."),
          LI(f"<b>과적합</b>: 학습 R² 평균 {f3(g(RND, '평균', '학습 R²'))}와 검증 R² 평균 {f3(g(RND, '평균', 'R²'))}의 차이가 작아, 학습 데이터만 외운 상태는 아닙니다."),
          LI(f"<b>치우침</b>: 무작위 7:3의 평균 편향 {f2(g(RND, '평균', '평균 편향'))} MPa로, 전체적으로 한쪽으로 치우쳐 예측하지 않습니다. 다만 고강도(60 MPa 이상)는 낮게 예측하는 경향이 있습니다."),
          LI("<b>반복 검증의 한계</b>: 반복 검증은 같은 데이터(UCI 1,030건) 안에서 나누는 방식입니다. 다른 기관·다른 재료의 시험 데이터로 확인하는 <b>외부 검증</b>이 가장 강력한 검증이며, 국내 데이터를 구하면 추가로 확인할 수 있습니다."),
          LI("<b>설정 고정</b>: 매 회차 신경망 구조·설정은 고정하고 가중치만 새로 학습했습니다. 설정은 전체 데이터의 학습용 70%로 고른 값이라 결과가 아주 조금 유리할 수 있습니다."),
          LI("<b>계산기와의 관계</b>: 이 검증은 계산기와 별도로 수행했습니다. 계산기 모델은 1,030건 전부로 학습했고, 이 보고서는 같은 구조의 모델이 새 데이터에서 얼마나 맞는지를 확인한 것입니다."),
          Spacer(1, 8), P("재현 방법", "h2"),
          P("Colab에서 <b>concrete_validation_7_3.ipynb</b>를 열고 런타임 → 모두 실행. N_REPEAT로 반복 횟수를 바꿀 수 있습니다. "
            "모든 회차 결과는 validation_runs.csv에 저장됩니다. 무작위 분할 결과는 회차 번호(random_state)로 똑같이 재현됩니다."),
          Spacer(1, 6), P("데이터: I-Cheng Yeh, Concrete Compressive Strength, UCI Machine Learning Repository, DOI 10.24432/C5PK67 (CC BY 4.0).", "small")]


def footer(c, d):
    c.saveState(); c.setFont("NG", 7.5); c.setFillColor(colors.HexColor(INK2))
    c.drawString(18 * mm, 10 * mm, "콘크리트 강도 ANN — 반복 무작위 7:3 성능 검증 결과"); c.drawRightString(A4[0] - 18 * mm, 10 * mm, str(d.page)); c.restoreState()


SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                  title="콘크리트 강도 ANN — 반복 무작위 7:3 성능 검증 결과", author="").build(story, onFirstPage=footer, onLaterPages=footer)
print("saved", OUT, "| 대표 회차", int(med["회차"]), round(float(med["R²"]), 4))
