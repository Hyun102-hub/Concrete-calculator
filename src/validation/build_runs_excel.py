"""1,000회 회차별 검증 결과 엑셀 (요약은 수식)"""
import sys
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
OUT = sys.argv[1]
r = pd.read_csv("validation_runs.csv"); r = r[r["방식"] == "무작위"].sort_values("회차")
F = Font(name="Arial", size=10); B = Font(name="Arial", size=10, bold=True); H = PatternFill("solid", fgColor="DDEBF7")
wb = Workbook(); ws = wb.active; ws.title = "1000회 결과"
cols = ["회차", "학습 건수", "검증 건수", "R²", "MAE(MPa)", "RMSE(MPa)", "±5 MPa 이내", "±10 MPa 이내", "상대오차 중앙값", "평균 편향(MPa)", "학습 R²"]
src = ["회차", "학습 건수", "검증 건수", "R²", "MAE", "RMSE", "±5 MPa 이내", "±10 MPa 이내", "상대오차 중앙값", "평균 편향", "학습 R²"]
fmt = ["0", "0", "0", "0.000", "0.00", "0.00", "0.0%", "0.0%", "0.0%", "+0.00;-0.00", "0.000"]
for j, c in enumerate(cols, 1):
    x = ws.cell(1, j, c); x.font, x.fill = B, H; x.alignment = Alignment(wrap_text=True, horizontal="center")
for i, (_, row) in enumerate(r.iterrows(), 2):
    for j, (c, f) in enumerate(zip(src, fmt), 1):
        v = row[c]; x = ws.cell(i, j, int(v) if f == "0" else float(v)); x.font = F; x.number_format = f
N = len(r) + 1
ws.freeze_panes = "B2"; ws.auto_filter.ref = f"A1:K{N}"
for j, w in enumerate([7, 9, 9, 8, 10, 10, 11, 12, 13, 13, 9], 1):
    ws.column_dimensions[chr(64 + j)].width = w
s = wb.create_sheet("요약", 0)
s["A1"] = "무작위 7:3 분할 반복 검증 요약 (아래 값은 '1000회 결과' 시트의 수식)"; s["A1"].font = Font(name="Arial", size=12, bold=True)
s["A2"] = "데이터: UCI Concrete Compressive Strength 1,030건 중 완전 중복 25건·배합·재령이 같고 강도가 다른 18행을 뺀 987건 / 학습 690건, 검증 297건"; s["A2"].font = F
hdr = ["지표", "평균", "중앙값", "표준편차", "2.5%", "97.5%", "최소", "최대"]
for j, c in enumerate(hdr, 1):
    x = s.cell(4, j, c); x.font, x.fill = B, H
for i, (c, col, f) in enumerate(zip(cols[3:], "DEFGHIJK", fmt[3:]), 5):
    rng = f"'1000회 결과'!{col}2:{col}{N}"
    s.cell(i, 1, c).font = F
    for j, fx in enumerate([f"=AVERAGE({rng})", f"=MEDIAN({rng})", f"=STDEV({rng})", f"=PERCENTILE({rng},0.025)",
                            f"=PERCENTILE({rng},0.975)", f"=MIN({rng})", f"=MAX({rng})"], 2):
        x = s.cell(i, j, fx); x.font = F; x.number_format = f
s.cell(14, 1, "회차 수").font = F; s.cell(14, 2, f"=COUNT('1000회 결과'!A2:A{N})").font = F
s.cell(15, 1, "R² 0.90 이상 회차").font = F; s.cell(15, 2, f"=COUNTIF('1000회 결과'!D2:D{N},\">=0.9\")").font = F
s.cell(16, 1, "R² 0.85 미만 회차").font = F; s.cell(16, 2, f"=COUNTIF('1000회 결과'!D2:D{N},\"<0.85\")").font = F
s.column_dimensions["A"].width = 18
for c in "BCDEFGH": s.column_dimensions[c].width = 11
wb.save(OUT); print("saved", OUT, len(r))
