"""1,000회 반복 7:3 검증 결과로 정리한 모델 성능 보고서 PDF
입력: validation_runs.csv(회차별 지표), validation_predictions.parquet(회차별·배합별 예측), validation_data_987.csv
python3 build_performance_pdf.py <출력 PDF>
"""
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
from sklearn.metrics import r2_score

H = Path(__file__).resolve().parent
OUT = Path(sys.argv[1])
FIG = H / "_figs"; FIG.mkdir(exist_ok=True)
RND, GRP = "무작위", "배합 기준"
runs = pd.read_csv(H / "validation_runs.csv")
dc = pd.read_csv(H / "validation_data_987.csv")
pr = pd.read_parquet(H / "validation_predictions.parquet")
pr["예측"] = pr["예측"].astype(float)
pr = pr.join(dc.reset_index(drop=True), on="행")
pr["오차"] = pr["예측"] - pr["압축강도"]; pr["절대오차"] = pr["오차"].abs()
NREP = int(runs.groupby("방식")["회차"].nunique().min())
assert NREP == 1000 and len(runs) == 2000

FD = Path(koreanize_matplotlib.__file__).parent / "fonts"
pdfmetrics.registerFont(TTFont("NG", str(FD / "NanumGothic.ttf"))); pdfmetrics.registerFont(TTFont("NGB", str(FD / "NanumGothicBold.ttf")))
plt.rcParams["axes.unicode_minus"] = False
BLUE, ORANGE, INK, INK2, LINE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#d9d8d4"


def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(LINE)
    ax.tick_params(colors=INK2, labelsize=8.5); ax.grid(axis="y", color=LINE, lw=0.6); ax.set_axisbelow(True)


R = runs[runs["방식"] == RND]; G = runs[runs["방식"] == GRP]
PR, PG = pr[pr["방식"] == RND], pr[pr["방식"] == GRP]
q = lambda s, p: float(s.quantile(p))

# ── 그림: 회차별 R² 분포 ──
fig, ax = plt.subplots(figsize=(7.4, 2.5), dpi=200)
ax.hist(R["R²"], bins=45, color=BLUE, edgecolor="white", linewidth=0.4)
ax.axvline(R["R²"].mean(), color=INK, lw=1.2, ls="--"); ax.text(R["R²"].mean(), ax.get_ylim()[1] * 0.92, f"  평균 {R['R²'].mean():.3f}", fontsize=8, color=INK)
ax.set_xlabel("검증 R² (회차별)", fontsize=9, color=INK2); ax.set_ylabel("회차 수", fontsize=9, color=INK2); style(ax)
plt.tight_layout(); fig.savefig(FIG / "p_r2.png"); plt.close(fig)

# ── 그림: 오차 분포 (모든 예측) ──
fig, ax = plt.subplots(figsize=(7.4, 2.4), dpi=200)
ax.hist(PR["오차"].clip(-25, 25), bins=np.arange(-25, 25.5, 1), color=BLUE, edgecolor="white", linewidth=0.4)
for v in (-5, 5): ax.axvline(v, color=INK2, lw=0.8, ls=":")
ax.axvline(0, color=INK, lw=0.8)
ax.set_xlabel("예측 − 실제 (MPa, ±25에서 자름)".replace("−", "-"), fontsize=9, color=INK2); ax.set_ylabel("예측 건수", fontsize=9, color=INK2); style(ax)
plt.tight_layout(); fig.savefig(FIG / "p_err.png"); plt.close(fig)

# ── 그림: 배합별 평균 예측 vs 실제 ──
per = PR.groupby("행").agg(실제=("압축강도", "first"), 평균예측=("예측", "mean"), 검증횟수=("예측", "size"), 예측표준편차=("예측", "std"))
fig, ax = plt.subplots(figsize=(3.6, 3.4), dpi=200)
ax.fill_between([0, 90], [-5, 85], [5, 95], color=BLUE, alpha=0.10, lw=0)
ax.scatter(per["실제"], per["평균예측"], s=7, color=BLUE, alpha=0.7, linewidths=0); ax.plot([0, 90], [0, 90], color=INK, lw=0.8)
ax.set_xlim(0, 90); ax.set_ylim(0, 90); ax.set_xlabel("실제 강도 (MPa)", fontsize=9, color=INK2); ax.set_ylabel("평균 예측 (MPa)", fontsize=9, color=INK2)
style(ax); ax.grid(axis="x", color=LINE, lw=0.6)
plt.tight_layout(); fig.savefig(FIG / "p_per.png"); plt.close(fig)

# ── 표 계산 ──
f3 = lambda v: f"{v:.3f}"; f2 = lambda v: f"{v:.2f}"; pc = lambda v: f"{100 * v:.1f}%"


def dist_rows(df):
    rows = []
    for m, lab, fmt in [("R²", "R²", f3), ("MAE", "MAE (MPa)", f2), ("RMSE", "RMSE (MPa)", f2), ("±5 MPa 이내", "±5 MPa 이내", pc),
                        ("±10 MPa 이내", "±10 MPa 이내", pc), ("상대오차 중앙값", "상대오차 중앙값", pc), ("평균 편향", "평균 편향 (MPa)", f2)]:
        s = df[m]
        rows.append([lab, f"<b>{fmt(s.mean())}</b>", fmt(s.median()), fmt(s.std()), f"{fmt(q(s, .025))} ~ {fmt(q(s, .975))}", f"{fmt(s.min())} ~ {fmt(s.max())}"])
    return rows


def seg(df, col, order=None):
    out = []
    for k, g in df.groupby(col, observed=True):
        out.append([str(k), f"{len(g):,}", f2(g["절대오차"].mean()), f2(np.sqrt((g["오차"] ** 2).mean())), f"{g['오차'].mean():+.2f}",
                    pc((g["절대오차"] <= 5).mean()), pc((g["절대오차"].div(g["압축강도"])).median())])
    return out


for df in (PR, PG):
    df.loc[:, "강도 구간"] = pd.cut(df["압축강도"], [0, 20, 40, 60, 90], labels=["20 MPa 이하", "20~40 MPa", "40~60 MPa", "60 MPa 이상"])
    df.loc[:, "재령 구간"] = pd.cut(df["재령"], [0, 7, 14, 28, 56, 100, 365], labels=["1~7일", "14일", "28일", "56일", "90~100일", "120~365일"])
    df.loc[:, "배합 종류"] = np.where((df["고로슬래그"] == 0) & (df["플라이애시"] == 0),
                                  np.where(df["고성능감수제"] == 0, "시멘트만", "시멘트 + 감수제"),
                                  np.where(df["고성능감수제"] == 0, "혼화재 사용", "혼화재 + 감수제"))
    wb = df["물"] / (df["시멘트"] + df["고로슬래그"] + df["플라이애시"])
    df.loc[:, "W/B 구간"] = pd.cut(wb, [0, 0.35, 0.45, 0.55, 1.0], labels=["0.35 이하", "0.35~0.45", "0.45~0.55", "0.55 초과"])
SEGH = ["구간", "예측 건수", "MAE", "RMSE", "평균 편향", "±5 MPa 이내", "상대오차 중앙값"]
SEGW = [0.2, 0.13, 0.11, 0.11, 0.13, 0.15, 0.17]

r2_avg = r2_score(per["실제"], per["평균예측"])
per["평균오차"] = per["평균예측"] - per["실제"]
worst = per.reindex(per["평균오차"].abs().sort_values(ascending=False).index).head(6).join(dc)
ae = PR["절대오차"]
share = lambda df, lo, hi: float(((df["R²"] >= lo) & (df["R²"] < hi)).mean())

# ── 문서 ──
ST = {
    "title": ParagraphStyle("t", fontName="NGB", fontSize=20, leading=27, textColor=INK),
    "sub": ParagraphStyle("s", fontName="NG", fontSize=10.5, leading=16, textColor=INK2),
    "h1": ParagraphStyle("h1", fontName="NGB", fontSize=14, leading=20, textColor=INK, spaceBefore=8, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="NGB", fontSize=11, leading=16, textColor=BLUE, spaceBefore=8, spaceAfter=4),
    "p": ParagraphStyle("p", fontName="NG", fontSize=9.6, leading=15.5, textColor=INK, spaceAfter=4),
    "li": ParagraphStyle("li", fontName="NG", fontSize=9.6, leading=15, textColor=INK, leftIndent=11, firstLineIndent=-8, spaceAfter=2),
    "small": ParagraphStyle("sm", fontName="NG", fontSize=8.2, leading=12, textColor=INK2, spaceAfter=3),
    "cell": ParagraphStyle("c", fontName="NG", fontSize=8.4, leading=11.5, textColor=INK),
    "cellb": ParagraphStyle("cb", fontName="NGB", fontSize=8.4, leading=11.5, textColor=INK),
    "big": ParagraphStyle("big", fontName="NGB", fontSize=19, leading=23, textColor=BLUE, alignment=TA_CENTER),
    "bigo": ParagraphStyle("bigo", fontName="NGB", fontSize=19, leading=23, textColor=ORANGE, alignment=TA_CENTER),
    "bigl": ParagraphStyle("bigl", fontName="NG", fontSize=8.3, leading=11, textColor=INK2, alignment=TA_CENTER),
    "box": ParagraphStyle("bx", fontName="NG", fontSize=9.6, leading=15.5, textColor=INK),
}
W = A4[0] - 36 * mm
P = lambda t, s="p": Paragraph(t, ST[s]); LI = lambda t: Paragraph("• " + t, ST["li"])


def T(rows, widths, right=(), hl=None):
    data = [[Paragraph(str(v), ST["cellb"] if i == 0 else ST["cell"]) for v in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=1)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(LINE)), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7f0f4")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2)]
    st += [("ALIGN", (c, 1), (c, -1), "RIGHT") for c in right]
    if hl: st += [("BACKGROUND", (0, r), (-1, r), colors.HexColor("#eef6ee")) for r in hl]
    t.setStyle(TableStyle(st)); return t


def BOX(html):
    t = Table([[Paragraph(html, ST["box"])]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e7f0f4")), ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor(BLUE)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return t


IM = lambda n, frac, hw: Image(str(FIG / n), width=W * frac, height=W * frac * hw)
story = [P("콘크리트 강도 ANN 모델 성능", "title"),
         P(f"반복 무작위 7:3 검증 {NREP:,}회 결과로 정리 · 검증 예측 {len(PR):,}건", "sub"), Spacer(1, 12)]
tiles = Table([[P(f3(R["R²"].mean()), "big"), P(f"{f2(R['MAE'].mean())} MPa", "big"), P(pc(R["±5 MPa 이내"].mean()), "big"), P(f"{q(R['R²'], .025):.2f}~{q(R['R²'], .975):.2f}", "big")],
               [P("검증 R² 평균", "bigl"), P("평균 오차 (MAE)", "bigl"), P("예측이 실제 ±5 MPa 안", "bigl"), P("검증 R² 95% 범위", "bigl")]], colWidths=[W / 4] * 4)
tiles.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(LINE)), ("LINEAFTER", (0, 0), (2, -1), 0.5, colors.HexColor(LINE)),
                           ("TOPPADDING", (0, 0), (-1, 0), 9), ("BOTTOMPADDING", (0, 1), (-1, 1), 9)]))
story += [tiles, Spacer(1, 10),
          BOX(f"<b>성능 요약</b><br/>중복·강도 충돌 데이터를 뺀 {len(dc):,}건을 무작위로 7:3으로 나눠 학습·검증하는 과정을 {NREP:,}번 반복한 결과, "
              f"ANN의 검증 R²는 평균 <b>{f3(R['R²'].mean())}</b>(95% 범위 {f3(q(R['R²'], .025))}~{f3(q(R['R²'], .975))}), 평균 오차는 <b>{f2(R['MAE'].mean())} MPa</b>였다. "
              f"검증 예측의 <b>{pc((ae <= 5).mean())}</b>가 실제 강도 ±5 MPa 안, <b>{pc((ae <= 10).mean())}</b>가 ±10 MPa 안에 들었다. "
              f"회차별 결과 {NREP:,}개는 부록에 모두 나열했다."),
          P("검증 방법", "h2"),
          T([["순서", "내용"],
             ["① 데이터 정리", f"UCI Concrete Compressive Strength 1,030건에서 완전히 같은 행 25건(하나만 남김)과 배합·재령이 같은데 강도가 다른 18행을 제외 → <b>{len(dc):,}건</b>"],
             ["② 7:3 분할", f"{len(dc):,}건을 무작위로 학습용 7({int(R['학습 건수'].iloc[0])}건)과 검증용 3({int(R['검증 건수'].iloc[0])}건)으로 나눔"],
             ["③ 학습", "학습용 7로 ANN 학습: 입력 = 재료 7종 + 재령(로그 변환) → MinMaxScaler → 신경망(은닉층 뉴런 64개, tanh, 규제 0.0001, 학습률 0.01, 조기 종료). 계산기 모델과 같은 구조"],
             ["④ 검증", "검증용 3의 배합(재료 7종 + 재령)을 그대로 모델에 입력 → 나온 예측 강도를 검증용 3의 실제 강도와 비교 (R², MAE, RMSE, ±5·±10 MPa 이내 비율, 평균 편향)"],
             ["⑤ 반복", f"②~④를 나누는 방식을 바꿔 가며 <b>{NREP:,}번</b> 반복 (회차 번호 = 분할의 random_state)"]], [0.17, 0.83])]

story += [Spacer(1, 6), P(f"1. 전체 성능 ({NREP:,}회 분포)", "h1"),
          T([["지표", "평균", "중앙값", "표준편차", "95% 범위", "최소 ~ 최대"]] + dist_rows(R), [0.2, 0.12, 0.12, 0.12, 0.22, 0.22], hl=[1]),
          P("95% 범위 = 1,000회 결과를 크기순으로 놓았을 때 가운데 95% (2.5%~97.5%). 표준편차는 회차마다 성능이 얼마나 달라지는지.", "small"),
          IM("p_r2.png", 1.0, 2.5 / 7.4),
          P("성능 수준별 회차 비율 (무작위 7:3)", "h2"),
          T([["검증 R²", "0.80 미만", "0.80 ~ 0.85", "0.85 ~ 0.90", "0.90 이상"],
             ["회차 수", f"{int(round(share(R, -9, .80) * NREP))}회", f"{int(round(share(R, .80, .85) * NREP))}회", f"{int(round(share(R, .85, .90) * NREP))}회", f"{int(round(share(R, .90, 9) * NREP))}회"],
             ["비율", pc(share(R, -9, .80)), pc(share(R, .80, .85)), pc(share(R, .85, .90)), pc(share(R, .90, 9))]],
            [0.2, 0.2, 0.2, 0.2, 0.2], right=(1, 2, 3, 4)),
          P(f"과적합 확인: 학습 R² 평균 {f3(R['학습 R²'].mean())} vs 검증 R² 평균 {f3(R['R²'].mean())} (차이 {R['학습 R²'].mean() - R['R²'].mean():.3f}).", "small")]

story += [P(f"2. 오차 분포 (검증 예측 {len(PR):,}건)", "h1"), IM("p_err.png", 1.0, 2.4 / 7.4),
          T([["항목", "값"], ["절대오차 중앙값 (예측의 절반은 이 안)", f"{ae.median():.2f} MPa"], ["절대오차 90% 지점", f"{q(ae, .9):.2f} MPa"],
             ["절대오차 95% 지점", f"{q(ae, .95):.2f} MPa"], ["±3 / ±5 / ±10 MPa 이내", f"{pc((ae <= 3).mean())} / {pc((ae <= 5).mean())} / {pc((ae <= 10).mean())}"],
             ["실제보다 높게 예측한 비율", pc((PR['오차'] > 0).mean())], ["평균 편향", f"{PR['오차'].mean():+.2f} MPa"]], [0.6, 0.4], right=(1,)),
          P("3. 어디서 잘 맞고 어디서 틀리나 (모든 검증 예측)", "h1"),
          KeepTogether([P("강도 구간별", "h2"), T([SEGH] + seg(PR, "강도 구간"), SEGW, right=(1, 2, 3, 4, 5, 6))])]
story += [KeepTogether([P("재령 구간별", "h2"), T([SEGH] + seg(PR, "재령 구간"), SEGW, right=(1, 2, 3, 4, 5, 6))]),
          KeepTogether([P("배합 종류별", "h2"), T([SEGH] + seg(PR, "배합 종류"), SEGW, right=(1, 2, 3, 4, 5, 6))]),
          KeepTogether([P("물-결합재비(W/B) 구간별", "h2"), T([SEGH] + seg(PR, "W/B 구간"), SEGW, right=(1, 2, 3, 4, 5, 6))]),
          P("평균 편향 +는 높게, −는 낮게 예측한다는 뜻. 상대오차 = |예측 − 실제| ÷ 실제 강도.", "small")]

wrows = [["배합 (시멘트·슬래그·플라이애시·물·감수제)", "재령", "실제", "평균 예측", "차이", "검증 횟수"]]
for i, r in worst.iterrows():
    wrows.append([f"{r['시멘트']:g} · {r['고로슬래그']:g} · {r['플라이애시']:g} · {r['물']:g} · {r['고성능감수제']:g}", f"{r['재령']:g}일",
                  f"{r['실제']:.1f}", f"{r['평균예측']:.1f}", f"{r['평균오차']:+.1f}", f"{int(r['검증횟수'])}"])
story += [P("4. 배합별로 본 예측 안정성", "h1"),
          Table([[IM("p_per.png", 0.47, 3.4 / 3.6),
                  [P(f"{len(dc):,}건 각각은 1,000회 중 평균 <b>{per['검증횟수'].mean():.0f}번</b> 검증 데이터로 뽑혔습니다. "
                     "그때마다 나온 예측을 평균한 값을 실제 강도와 비교했습니다."),
                   LI(f"배합별 평균 예측과 실제의 R²: <b>{r2_avg:.3f}</b>"),
                   LI(f"같은 배합이라도 회차마다 예측이 달라지는 폭(표준편차) 평균 <b>{per['예측표준편차'].mean():.2f} MPa</b>"),
                   LI(f"평균 예측이 실제 ±5 MPa 안인 배합 <b>{pc((per['평균오차'].abs() <= 5).mean())}</b>"),
                   LI("파란 띠 = 실제 ±5 MPa")]]], colWidths=[0.5 * W, 0.5 * W], style=[("VALIGN", (0, 0), (-1, -1), "MIDDLE")]),
          P("평균 예측이 가장 크게 빗나간 배합", "h2"), T(wrows, [0.4, 0.1, 0.11, 0.13, 0.11, 0.15], right=(2, 3, 4, 5)),
          P("이 배합들은 비슷한 배합이 데이터에 적거나, 같은 조건의 다른 시험값과 강도 차이가 큰 경우입니다.", "small")]

sr = {k: v for k, v in zip(["20 MPa 이하", "20~40 MPa", "40~60 MPa", "60 MPa 이상"], seg(PR, "강도 구간"))}
story += [Spacer(1, 6), KeepTogether([P("5. 성능 정리", "h1"),
          T([["관점", "결과", "판단"],
             ["정확도", f"검증 R² 평균 {f3(R['R²'].mean())}, MAE {f2(R['MAE'].mean())} MPa", "실험 데이터의 강도 차이 대부분을 설명"],
             ["안정성", f"1,000회 R² 표준편차 {f3(R['R²'].std())}, 95% 범위 {f3(q(R['R²'], .025))}~{f3(q(R['R²'], .975))}", "데이터를 어떻게 나눠도 성능이 비슷"],
             ["오차 크기", f"예측의 {pc((ae <= 5).mean())}가 ±5 MPa, {pc((ae <= 10).mean())}가 ±10 MPa 안", "배합 후보 선별에는 충분, 시험 대체는 불가"],
             ["치우침", f"평균 편향 {PR['오차'].mean():+.2f} MPa", "전체적으로 한쪽으로 치우치지 않음"],
             ["과적합", f"학습 {f3(R['학습 R²'].mean())} vs 검증 {f3(R['R²'].mean())}", "학습 데이터를 외운 상태 아님"],
             ["약점", f"60 MPa 이상: MAE {sr['60 MPa 이상'][2]} MPa, 평균 편향 {sr['60 MPa 이상'][4]} MPa", "고강도는 낮게 예측 (데이터 부족)"]], [0.13, 0.5, 0.37])]),
          Spacer(1, 8),
          BOX(f"<b>발표용 한 문장</b><br/>중복·충돌 데이터를 제외한 {len(dc):,}건으로 무작위 7:3 검증을 {NREP:,}회 반복한 결과, ANN은 검증 R² 평균 {f3(R['R²'].mean())} "
              f"(95% 범위 {f3(q(R['R²'], .025))}~{f3(q(R['R²'], .975))}), 평균 오차 {f2(R['MAE'].mean())} MPa로 실제 강도를 예측했으며, "
              f"검증 예측의 {pc((ae <= 10).mean())}가 실제 강도 ±10 MPa 안에 들었다."),
          Spacer(1, 8), P("주의할 점", "h2"),
          LI(f"무작위로 나누므로 같은 배합의 다른 재령(예: 7일과 28일)이 학습과 검증에 나뉘어 들어갈 수 있습니다. 같은 배합은 한쪽에만 들어가게 나눠 같은 횟수를 반복하면 "
             f"검증 R² 평균이 {f3(G['R²'].mean())}로 조금 낮아집니다. 학습에 없던 새 배합을 예측할 때는 이 정도를 기대하는 것이 안전합니다."),
          LI("모든 검증은 같은 데이터(UCI) 안에서 나눈 것입니다. 다른 기관·재료의 시험 데이터로 확인하는 외부 검증이 가장 강력한 검증입니다."),
          LI("매 회차 신경망 구조와 설정은 고정하고 가중치만 새로 학습했습니다. 설정은 전체 데이터의 학습용 70%로 고른 값이라 결과가 아주 조금 유리할 수 있습니다."),
          LI(f"반복 횟수: 100회와 1,000회의 평균 R² 차이는 약 {abs(R.sort_values('회차')['R²'].head(100).mean() - R['R²'].mean()):.3f}로, {NREP:,}회 결과는 충분히 안정된 값입니다."),
          Spacer(1, 6), P("데이터: I-Cheng Yeh, Concrete Compressive Strength, UCI Machine Learning Repository, DOI 10.24432/C5PK67 (CC BY 4.0). "
                          "재현: src/validation/concrete_validation_7_3.ipynb", "small")]

story.append(PageBreak())
RR = R.sort_values("회차")
story += [P(f"부록. {NREP:,}회 회차별 결과", "h1"),
          P(f"각 회차: {len(dc):,}건을 무작위 7:3으로 나눠 학습 {int(RR['학습 건수'].iloc[0])}건으로 학습하고, 검증 {int(RR['검증 건수'].iloc[0])}건의 배합을 입력해 나온 예측을 실제 강도와 비교한 결과. "
            f"<b>굵게</b> 표시한 회차는 R²가 가장 낮은 회차({int(RR.loc[RR['R²'].idxmin(), '회차'])}회차)와 가장 높은 회차({int(RR.loc[RR['R²'].idxmax(), '회차'])}회차).", "small")]
AH = ["회차", "학습", "검증", "R²", "MAE", "RMSE", "±5 MPa", "±10 MPa", "편향"]
ext = {int(RR.loc[RR["R²"].idxmin(), "회차"]), int(RR.loc[RR["R²"].idxmax(), "회차"])}
cs = ParagraphStyle("ac", fontName="NG", fontSize=6.6, leading=8.2, textColor=INK, alignment=2)
cb = ParagraphStyle("acb", parent=cs, fontName="NGB")
ch = ParagraphStyle("ach", parent=cs, fontName="NGB", alignment=1)
def arow(r):
    st = cb if int(r["회차"]) in ext else cs
    v = [f"{int(r['회차'])}", f"{int(r['학습 건수'])}", f"{int(r['검증 건수'])}", f"{r['R²']:.3f}", f"{r['MAE']:.2f}", f"{r['RMSE']:.2f}",
         f"{100 * r['±5 MPa 이내']:.1f}%", f"{100 * r['±10 MPa 이내']:.1f}%", f"{r['평균 편향']:+.2f}"]
    return [Paragraph(x, st) for x in v]
rows_all = [arow(r) for _, r in RR.iterrows()]
PER_COL = 50                                     # 한 쪽에 두 단 × 50행 = 100회차
for start in range(0, len(rows_all), PER_COL * 2):
    left, right = rows_all[start:start + PER_COL], rows_all[start + PER_COL:start + 2 * PER_COL]
    hdr = [Paragraph(h, ch) for h in AH]
    data = [hdr + [Paragraph("", cs)] + hdr]
    for i2 in range(PER_COL):
        l = left[i2] if i2 < len(left) else [Paragraph("", cs)] * 9
        rr_ = right[i2] if i2 < len(right) else [Paragraph("", cs)] * 9
        data.append(l + [Paragraph("", cs)] + rr_)
    cw = [0.055, 0.045, 0.045, 0.055, 0.05, 0.05, 0.06, 0.065, 0.055]
    t = Table(data, colWidths=[w * W for w in cw] + [0.02 * W] + [w * W for w in cw], repeatRows=1)
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (8, 0), 0.6, colors.HexColor(INK2)), ("LINEBELOW", (10, 0), (18, 0), 0.6, colors.HexColor(INK2)),
                           ("BACKGROUND", (0, 0), (8, 0), colors.HexColor("#e7f0f4")), ("BACKGROUND", (10, 0), (18, 0), colors.HexColor("#e7f0f4")),
                           ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2), ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
                          + [("BACKGROUND", (c0, r0), (c0 + 8, r0), colors.HexColor("#f5f6f5")) for r0 in range(2, PER_COL + 1, 2) for c0 in (0, 10)]))
    story += [t] + ([PageBreak()] if start + PER_COL * 2 < len(rows_all) else [])
story += [Spacer(1, 4), P(f"전체 {NREP:,}회 평균: R² {RR['R²'].mean():.3f}, MAE {RR['MAE'].mean():.2f} MPa, RMSE {RR['RMSE'].mean():.2f} MPa, "
                          f"±5 MPa {100 * RR['±5 MPa 이내'].mean():.1f}%, ±10 MPa {100 * RR['±10 MPa 이내'].mean():.1f}%, 편향 {RR['평균 편향'].mean():+.2f} MPa", "small")]


def footer(c, d):
    c.saveState(); c.setFont("NG", 7.5); c.setFillColor(colors.HexColor(INK2))
    c.drawString(18 * mm, 10 * mm, "콘크리트 강도 ANN 성능 검증 — 무작위 7:3 분할 1,000회 반복"); c.drawRightString(A4[0] - 18 * mm, 10 * mm, str(d.page)); c.restoreState()


SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                  title="콘크리트 강도 ANN 모델 성능 — 반복 무작위 7:3 검증 1,000회", author="").build(story, onFirstPage=footer, onLaterPages=footer)
print("saved", OUT)
print("R2", R["R²"].mean(), "MAE", R["MAE"].mean(), "pooled ±5", (ae <= 5).mean(), "per-mix R2", r2_avg)
