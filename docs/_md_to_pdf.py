"""
_md_to_pdf.py — Markdown → PDF 변환 (한글 폰트 안전).
==================================================
APPLICATION_PRESENTATION.md 같은 한국어 마크다운을 reportlab Platypus 로 변환.
Malgun Gothic (Windows 기본 한글 폰트) 사용으로 한글 깨짐 방지.

지원:
  - H1/H2/H3 헤딩 (크기 + 간격 위계)
  - 본문 paragraph (자동 줄바꿈)
  - 코드 블록 ``` (monospace 박스)
  - 인라인 코드 `code` (회색 배경)
  - **bold**, *italic*
  - bulleted list (-, *)
  - numbered list (1. 2. ...)
  - 표 | a | b | (기본 격자, 헤더 강조)
  - 수평선 ---
  - blockquote >

사용:
  python _md_to_pdf.py <input.md> <output.pdf>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib.colors import HexColor, black, white, grey
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── 한글 폰트 등록 (Windows 기본 Malgun Gothic) ───────────────────────
WIN_FONTS = Path(r"C:\Windows\Fonts")
pdfmetrics.registerFont(TTFont("KR",     str(WIN_FONTS / "malgun.ttf")))
pdfmetrics.registerFont(TTFont("KR-Bold", str(WIN_FONTS / "malgunbd.ttf")))
# monospace 는 Consolas (Windows 기본, 한글 fallback 가능)
try:
    pdfmetrics.registerFont(TTFont("MonoKR", str(WIN_FONTS / "consola.ttf")))
    MONO_FONT = "MonoKR"
except Exception:
    MONO_FONT = "Courier"

# ── 스타일 ─────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

H1 = ParagraphStyle("H1", parent=styles["Heading1"],
    fontName="KR-Bold", fontSize=20, leading=26,
    textColor=HexColor("#1565C0"), spaceBefore=18, spaceAfter=10)

H2 = ParagraphStyle("H2", parent=styles["Heading2"],
    fontName="KR-Bold", fontSize=15, leading=20,
    textColor=HexColor("#0277BD"), spaceBefore=14, spaceAfter=6)

H3 = ParagraphStyle("H3", parent=styles["Heading3"],
    fontName="KR-Bold", fontSize=12, leading=16,
    textColor=HexColor("#37474F"), spaceBefore=10, spaceAfter=4)

BODY = ParagraphStyle("Body", parent=styles["BodyText"],
    fontName="KR", fontSize=10, leading=15,
    textColor=black, alignment=TA_LEFT, spaceBefore=2, spaceAfter=6)

LIST_ITEM = ParagraphStyle("ListItem", parent=BODY,
    leftIndent=18, bulletIndent=4, spaceAfter=2)

QUOTE = ParagraphStyle("Quote", parent=BODY,
    leftIndent=14, rightIndent=8,
    textColor=HexColor("#546E7A"), borderColor=HexColor("#B0BEC5"),
    borderPadding=4, fontSize=10)

CODE = ParagraphStyle("Code", parent=styles["Code"],
    fontName=MONO_FONT, fontSize=9, leading=12,
    textColor=HexColor("#263238"), backColor=HexColor("#ECEFF1"),
    leftIndent=8, rightIndent=8, spaceBefore=4, spaceAfter=8,
    borderColor=HexColor("#CFD8DC"), borderWidth=0.5, borderPadding=6)


# ── 인라인 마크다운 → reportlab XML ────────────────────────────────────
def _esc(s: str) -> str:
    """reportlab Paragraph 가 안전하게 처리하도록 XML 이스케이프."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_inline(text: str) -> str:
    """**bold** *italic* `code` 를 reportlab font 태그로 변환. 이미 이스케이프된 입력 가정."""
    # `inline code` — 회색 배경 + monospace
    text = re.sub(
        r"`([^`]+)`",
        lambda m: f'<font name="{MONO_FONT}" color="#C2185B">{m.group(1)}</font>',
        text,
    )
    # **bold**
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    # *italic* (단 ** 와 충돌 주의 — ** 먼저 처리됐으므로 여기선 단일 *)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", text)
    return text


def md_paragraph(text: str, style: ParagraphStyle = BODY) -> Paragraph:
    return Paragraph(render_inline(_esc(text)), style)


# ── 표 변환 ────────────────────────────────────────────────────────────
def md_table_to_flowable(rows: list[list[str]]) -> Table:
    """| a | b | 형식 → reportlab Table. rows[0] = header."""
    data = []
    for r_idx, row in enumerate(rows):
        line = []
        for cell in row:
            line.append(Paragraph(render_inline(_esc(cell.strip())), BODY))
        data.append(line)

    n_cols = max(len(r) for r in data)
    # 페이지 폭 - 마진 = 사용 가능 폭
    page_w = A4[0] - 4 * cm
    col_w = [page_w / n_cols] * n_cols
    tbl = Table(data, colWidths=col_w, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#E3F2FD")),
        ("FONTNAME",   (0, 0), (-1, 0), "KR-Bold"),
        ("TEXTCOLOR",  (0, 0), (-1, 0), HexColor("#0D47A1")),
        ("GRID",       (0, 0), (-1, -1), 0.4, HexColor("#90A4AE")),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return tbl


# ── 마크다운 파서 (라인 기반, 본 용도 충분) ────────────────────────────
def parse_markdown(md_text: str) -> list:
    flowables = []
    lines = md_text.splitlines()
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # 빈 줄
        if not stripped:
            i += 1
            continue

        # 코드 블록 ```
        if stripped.startswith("```"):
            i += 1
            code_lines = []
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # 닫는 ```
            code = "\n".join(code_lines)
            # Preformatted 가 XML 처리 안 함 — 그대로 폰트만 적용
            flowables.append(Preformatted(code, CODE))
            continue

        # 헤딩
        if stripped.startswith("### "):
            flowables.append(md_paragraph(stripped[4:], H3))
            i += 1
            continue
        if stripped.startswith("## "):
            flowables.append(md_paragraph(stripped[3:], H2))
            i += 1
            continue
        if stripped.startswith("# "):
            flowables.append(md_paragraph(stripped[2:], H1))
            i += 1
            continue

        # 수평선 ---
        if re.fullmatch(r"-{3,}|_{3,}|\*{3,}", stripped):
            flowables.append(Spacer(1, 4))
            flowables.append(HRFlowable(width="100%", thickness=0.6,
                                        color=HexColor("#CFD8DC")))
            flowables.append(Spacer(1, 4))
            i += 1
            continue

        # blockquote >
        if stripped.startswith("> "):
            block = []
            while i < n and lines[i].strip().startswith("> "):
                block.append(lines[i].strip()[2:])
                i += 1
            flowables.append(md_paragraph("<br/>".join(_esc(b) for b in block), QUOTE))
            # 위에서 _esc 후 render_inline 미적용 — 다시:
            flowables[-1] = Paragraph(
                render_inline("<br/>".join(_esc(b) for b in block)), QUOTE
            )
            continue

        # 표 |
        if stripped.startswith("|") and stripped.endswith("|"):
            table_rows = []
            while i < n and lines[i].strip().startswith("|"):
                row_line = lines[i].strip()
                # 구분선 |---|---| 은 skip
                if re.fullmatch(r"\|[\s:\-|]+\|", row_line):
                    i += 1
                    continue
                cells = [c.strip() for c in row_line.strip("|").split("|")]
                table_rows.append(cells)
                i += 1
            if table_rows:
                flowables.append(md_table_to_flowable(table_rows))
                flowables.append(Spacer(1, 6))
            continue

        # 리스트 (- 또는 *)
        if re.match(r"^[-*]\s+", stripped):
            list_items = []
            while i < n and re.match(r"^\s*[-*]\s+", lines[i]):
                m = re.match(r"^(\s*)[-*]\s+(.*)$", lines[i])
                if m is None:
                    break
                indent_spaces = len(m.group(1))
                item_text = m.group(2)
                indent_level = indent_spaces // 2  # 2 칸 = 1 단계
                bullet = "• " if indent_level == 0 else "◦ "
                style = ParagraphStyle(
                    f"L{indent_level}",
                    parent=LIST_ITEM,
                    leftIndent=14 + 16 * indent_level,
                )
                list_items.append(
                    Paragraph(bullet + render_inline(_esc(item_text)), style)
                )
                i += 1
            flowables.extend(list_items)
            flowables.append(Spacer(1, 4))
            continue

        # 번호 리스트 1. 2.
        if re.match(r"^\d+\.\s+", stripped):
            num_items = []
            while i < n and re.match(r"^\s*\d+\.\s+", lines[i]):
                m = re.match(r"^(\s*)(\d+)\.\s+(.*)$", lines[i])
                if m is None:
                    break
                num = m.group(2)
                txt = m.group(3)
                num_items.append(
                    Paragraph(f"{num}. " + render_inline(_esc(txt)), LIST_ITEM)
                )
                i += 1
            flowables.extend(num_items)
            flowables.append(Spacer(1, 4))
            continue

        # 그 외 = 본문 paragraph (연속 라인은 한 paragraph 로 합침)
        para_lines = [stripped]
        i += 1
        while i < n:
            nxt = lines[i].strip()
            if not nxt:
                break
            if (nxt.startswith("#") or nxt.startswith("```") or
                nxt.startswith("|") or nxt.startswith(">") or
                re.match(r"^[-*]\s+", nxt) or re.match(r"^\d+\.\s+", nxt) or
                re.fullmatch(r"-{3,}", nxt)):
                break
            para_lines.append(nxt)
            i += 1
        flowables.append(md_paragraph(" ".join(para_lines)))

    return flowables


# ── 메인 ───────────────────────────────────────────────────────────────
def md_to_pdf(md_path: Path, pdf_path: Path) -> None:
    md_text = md_path.read_text(encoding="utf-8")
    flowables = parse_markdown(md_text)

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title=md_path.stem, author="imu_android",
    )

    def _on_page(canvas, _doc):
        canvas.saveState()
        canvas.setFont("KR", 8)
        canvas.setFillColor(HexColor("#90A4AE"))
        canvas.drawRightString(
            A4[0] - 2 * cm, 1.2 * cm, f"— {_doc.page} —"
        )
        canvas.restoreState()

    doc.build(flowables, onFirstPage=_on_page, onLaterPages=_on_page)
    print(f"[OK] {pdf_path}  ({pdf_path.stat().st_size / 1024:.1f} KB)")


def main() -> int:
    if len(sys.argv) != 3:
        print("사용: python _md_to_pdf.py <input.md> <output.pdf>")
        return 1
    md_path = Path(sys.argv[1])
    pdf_path = Path(sys.argv[2])
    if not md_path.is_file():
        print(f"입력 파일 없음: {md_path}")
        return 1
    md_to_pdf(md_path, pdf_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
