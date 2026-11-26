# Concrete-calculator

콘크리트 배합(재료 7종 단위량)과 목표 강도, 재령을 넣으면 그 재령에서 압축강도가 **목표 강도 이상일 확률**을 계산하는 웹 계산기입니다.

- 사이트: https://hyun102-hub.github.io/Concrete-calculator/ (GitHub Pages)
- 예측: UCI *Concrete Compressive Strength* 실험 데이터 **1,030건**으로 학습한 인공신경망(ANN). 입력은 원자료의 8개 열(재료 7종 + 재령, 재령은 로그 변환)뿐
- 확률: ANN 예측 강도 μ와 교차검증 예측 오차 σ로 1 − Φ((목표 − μ) ÷ σ)
- 공식(Abrams 법칙 + ACI 209 재령식)은 **비교용**으로만 표시하며 예측에는 쓰지 않음
- 데이터: I-Cheng Yeh, UCI Machine Learning Repository, DOI 10.24432/C5PK67 (CC BY 4.0)
- 확률은 모델의 예측 오차를 바탕으로 한 값이며, 실제 강도 시험을 대신하지 않습니다.

## 성능 검증 결과

| 검증 | 결과 |
|---|---|
| **반복 무작위 7:3 검증** (완전 중복 25건·강도 충돌 18행 제외 987건, 1,000회, 모델과 별도) | R² 0.892 ± 0.021 (95% 범위 0.849~0.921), MAE 3.85 MPa |
| 같은 방식, 배합 기준 7:3 (처음 보는 배합만 검증, 1,000회) | R² 0.849 ± 0.032 (95% 범위 0.775~0.901), MAE 4.50 MPa |
| 테스트 (301건) | R² 0.886, MAE 4.14 MPa, RMSE 5.33 MPa |
| 테스트 R² 95% 신뢰구간 (배합 단위 부트스트랩 1,000회) | 0.848 ~ 0.916 |
| 반복 그룹 교차검증 (5겹 × 3회) | R² 0.863 ± 0.031, RMSE 6.05 ± 0.53 MPa |
| 과적합 확인 | 학습 R² 0.910 vs 테스트 0.886 |
| 비교: 공식만 / 선형회귀 | R² 0.751 / 0.579 |

## 폴더

| 경로 | 내용 |
|---|---|
| `index.html` | 사이트로 보이는 계산기 (파일 하나로 동작) |
| `src/concrete_ann_prob.ipynb` | Google Colab 학습·검증 노트북 |
| `src/concrete_ann_prob_실행결과.ipynb` | 노트북 실행 결과 |
| `src/model_export.json` | 학습된 모델 (가중치·σ·비교용 공식 계수·검증 결과) |
| `src/html_model/` | 계산기 화면(`template.html`), 계산 코드(`model_core.js`), 생성 스크립트(`build_html.py`) |
| `src/data_build/raw/` | UCI 원자료 |
| `src/validation/` | 모델과 별도로 돌리는 반복 무작위 7:3 검증 노트북, 회차별 결과(csv), 회차별·배합별 예측(parquet), 보고서 생성 스크립트 |
| `모델성능_1000회검증.pdf` | 반복 무작위 7:3 검증 1,000회 결과로 정리한 모델 성능 보고서 |
| `src/이전모델_공식결합/` | 이전 버전 (공식 값을 ANN 입력에 넣은 모델)과 그 엑셀·설명서 |
| `계산기_사용설명서.pdf` | 이전 버전 기준 사용 설명서 (화면 구성은 같고, 숫자와 일부 칸 이름이 바뀜) |

## 계산기 다시 만들기

```bash
python3 src/html_model/build_html.py
cp src/public/index.html index.html
```

모델을 다시 학습했다면 Colab 노트북이 만든 `model_export.json`을 `src/`에 덮어쓴 뒤 위 명령을 실행합니다.
