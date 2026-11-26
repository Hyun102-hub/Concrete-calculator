"""model_export.json + template.html + model_core.js → concrete_probability.html (파일 하나로 열리는 HTML 모델)"""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
e = json.loads((H.parent / "model_export.json").read_text(encoding="utf-8"))
if e["chosen"].startswith("A"):          # 회귀 방식이면 분류 모델 가중치는 넣지 않음 (파일 크기 절약)
    e = {k: v for k, v in e.items() if k not in ("clf", "iso", "t_train", "platt")}
blob = json.dumps(e, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
html = (H / "template.html").read_text(encoding="utf-8").replace("__MODEL_JSON__", blob)
html = html.replace("__CORE_JS__", (H / "model_core.js").read_text(encoding="utf-8"))
# 입력칸과 예시 버튼을 HTML에 미리 써 넣음 (자바스크립트가 막힌 미리보기에서도 보이게)
MATS = ["시멘트", "고로슬래그", "플라이애시", "물", "고성능감수제", "굵은골재", "잔골재"]
EX = {"시멘트": 350, "고로슬래그": 0, "플라이애시": 0, "물": 180, "고성능감수제": 0, "굵은골재": 1000, "잔골재": 780}
fmtn = lambda v: f"{v:g}"
mix_html = "".join(
    f'<label for="m-{j}">{k}<em>{fmtn(e["ranges"][k][0])}–{fmtn(e["ranges"][k][1])}</em></label>'
    f'<div class="field"><input type="number" id="m-{j}" step="0.1" min="0" value="{EX[k]}" data-key="{k}" inputmode="decimal">'
    f'<span class="unit">kg</span></div>' for j, k in enumerate(MATS))
btns = "".join(f'<button type="button" class="chip-btn" data-i="{i}">{n}</button>' for i, n in enumerate(["보통 콘크리트", "혼화재 + 감수제", "슬래그 다량"]))
html = html.replace("__MIX_INPUTS__", mix_html).replace("__PRESET_BUTTONS__", btns)
assert "__" not in html.split("<script")[0], "치환 안 된 자리표시자"
out = H / "concrete_probability.html"          # Artifact 게시용 (게시할 때 문서 머리말이 자동으로 붙음)
out.write_text(html, encoding="utf-8")
print(out, f"{len(html.encode()) / 1024:.0f} KB", "| 방법", e["chosen"])

# 내려받아 PC에서 바로 여는 단독 파일: 문서 형식·문자 인코딩·기본 스타일을 직접 넣음
head, body = html.split("</style>", 1)
title = head.split("</title>", 1)[0] + "</title>"
rest_head = head.split("</title>", 1)[1].replace('<meta name="claude:viewport-fit-sides" content="cover">', "")
standalone = (
    '<!DOCTYPE html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    + title + rest_head +
    "\nhtml { -webkit-text-size-adjust: 100%; }\nbody { margin: 0; }\nimg { max-width: 100%; }\n</style>\n</head>\n<body>\n"
    + body.strip() + "\n</body>\n</html>\n")
out2 = H.parent / "콘크리트_목표강도_확률_계산기.html"
out2.write_text(standalone, encoding="utf-8")
print(out2, f"{len(standalone.encode()) / 1024:.0f} KB (단독 실행용)")

# 공개 웹사이트(GitHub Pages 등)에 올릴 index.html: claude.ai 주소 대신 일반 안내 문구
pub = standalone.replace(
    '<span class="nojs-how">Safari·Chrome 같은 <b>웹 브라우저</b>에서 아래 주소로 여세요.<br><span class="nojs-url">https://claude.ai/artifact/4J3pgnPkU2eFXL6fKkE1zq</span></span>',
    '<span class="nojs-how">이 페이지의 주소를 복사해 Safari·Chrome·Edge 같은 <b>웹 브라우저</b>에서 여세요.</span>')
assert "claude.ai/artifact" not in pub
(H.parent / "public").mkdir(exist_ok=True)
(H.parent / "public" / "index.html").write_text(pub, encoding="utf-8")
print(H.parent / "public" / "index.html", "(공개 사이트용)")
