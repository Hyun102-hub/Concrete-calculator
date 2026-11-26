# Concrete-calculator

콘크리트 배합(재료 7종 단위량)과 목표 강도, 재령을 넣으면 그 재령에서 압축강도가 **목표 강도 이상일 확률**을 계산하는 웹 계산기입니다.

- 사이트: https://hyun102-hub.github.io/Concrete-calculator/ (GitHub Pages)
- 모델: 인공신경망(ANN)에 Abrams 법칙(물-결합재비)과 ACI 209 재령 계수를 결합해 강도를 예측하고, 예측 오차를 정규분포로 보아 확률을 계산
- 데이터: I-Cheng Yeh, *Concrete Compressive Strength*, UCI Machine Learning Repository, DOI 10.24432/C5PK67 (CC BY 4.0), 1,030건
- 확률은 모델의 예측 오차를 바탕으로 한 값이며, 실제 강도 시험을 대신하지 않습니다.

## 폴더

| 경로 | 내용 |
|---|---|
| `index.html` | 사이트로 보이는 계산기 (파일 하나로 동작) |
| `src/concrete_target_prob.ipynb` | Google Colab 학습 노트북 |
| `src/concrete_target_prob_실행결과.ipynb` | 노트북 실행 결과 |
| `src/model_export.json` | 학습된 모델 (가중치·공식 계수·σ) |
| `src/html_model/` | 계산기 화면(`template.html`), 계산 코드(`model_core.js`), 생성 스크립트(`build_html.py`) |
| `src/콘크리트_목표강도_확률_데이터.xlsx` | 데이터·공식(수식)·결과 정리 엑셀 |
| `src/README_목표강도확률.md` | 결과, 검수 내용, 한계점 |
| `src/data_build/` | 엑셀 생성 스크립트, UCI 원자료 |

## 계산기 다시 만들기

```bash
python3 src/html_model/build_html.py
cp src/public/index.html index.html
```

모델을 다시 학습했다면 Colab 노트북이 만든 `model_export.json`을 `src/`에 덮어쓴 뒤 위 명령을 실행합니다.
