# KIISE 양식 LaTeX 패키지 — Overleaf 업로드 안내

## 구성 파일
```
kiise_paper/
├── main.tex            # 본문 (KIISE 양식: 제목 1단·본문 2단, A4, 여백 위30/아래20/좌우10mm)
├── refs.bib            # 참고문헌 12편 (ieeetr 스타일)
└── figures/            # 그림 8개 (그림1은 본문 내 블록도)
    ├── fig2_backbone.png      (그림 2)
    ├── fig3_handbag.png       (그림 3)
    ├── fig4_handheld.png      (그림 4)
    ├── fig5_trolley.png       (그림 5)
    ├── fig6_align.png         (그림 6)
    ├── fig7_sensitivity.png   (그림 7)
    ├── fig8_drifttraj.png     (그림 8)
    └── fig9_ekfvsrotvec.png   (그림 9)
```

## Overleaf 업로드 방법 (둘 중 택1)
**방법 A — 새 프로젝트 업로드 (가장 간단)**
1. `kiise_paper` 폴더를 통째로 ZIP 압축.
2. Overleaf → **New Project → Upload Project** → ZIP 선택.
3. **Menu → Settings → Compiler = XeLaTeX** 로 변경 (한글 kotex 필수).
4. **Recompile**.

**방법 B — 기존 프로젝트(6a278c18...)에 넣기**
1. 해당 프로젝트 열기 → 좌측 파일 패널 상단 **Upload** 아이콘.
2. `main.tex`, `refs.bib`, 그리고 `figures/` 안 PNG 8개를 업로드
   (figures 폴더 구조 유지 — Overleaf에서 폴더 드래그 가능).
3. **Settings → Compiler = XeLaTeX**, Main document = `main.tex`.
4. Recompile.

> ⚠️ **반드시 XeLaTeX**로 컴파일하세요. pdfLaTeX로는 한글(kotex)이 깨집니다.
> (`% !TEX program = xelatex` 주석을 main.tex 첫 줄에 추가하면 자동 선택됨)

## 채워야 할 항목
- 제목부 `［저자명］`, `［소속/학과］`, 영문 `［Author Name］`, `［Affiliation］` — 실제 정보로 교체.
- 발표자 표시(`◦`)는 공동저자와 구분용. 단독이면 그대로 두거나 제거.

## 주의 / 알려진 한계
- 로컬에 LaTeX 엔진이 없어 **컴파일 검증은 못 했습니다.** Overleaf에서 첫 컴파일 시
  사소한 경고/오류가 날 수 있으며, 대부분 그림 경로·폰트 관련입니다.
- 분량이 KIISE 권장(2~4쪽)을 초과할 수 있습니다. 표/그림 일부를 줄이거나
  `\small`→`\footnotesize` 조정으로 맞추세요.
- 표 캡션은 상단, 그림 캡션은 하단(KIISE 규정)으로 이미 설정돼 있습니다.
- 심사용 익명본이 필요하면 제목부의 저자/소속/이메일을 주석 처리하세요.
- `ieeetr` 참고문헌 스타일은 KIISE 권장(IEEE 계열) 중 하나입니다. 학회 지정이
  다르면 `\bibliographystyle{}`만 교체하세요.
