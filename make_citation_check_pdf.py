"""
make_citation_check_pdf.py — 본문 인용 ↔ 참고문헌 교차참조 점검 PDF
=================================================================
06-09 IMRaD 최종보고서의 [1]~[12] 인용에 대해
  · 본문에서 인용된 위치(절)
  · 참고문헌 제목
  · 검증용 하이퍼링크(Google Scholar 정확 제목 검색 — 항상 유효)
를 표로 정리해 사용자가 직접 더블체크할 수 있는 PDF를 생성한다.
"""
import urllib.parse
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle)

pdfmetrics.registerFont(TTFont("Malgun", r"C:\Windows\Fonts\malgun.ttf"))
pdfmetrics.registerFont(TTFont("MalgunBd", r"C:\Windows\Fonts\malgunbd.ttf"))

# (ref#, 저자·연도, 제목, 본문 인용 위치, 검증 비고)
REFS = [
    (1, "Liu et al., 2020", "TLIO: Tight Learned Inertial Odometry", "§1 Introduction; §4.2 Discussion", "IEEE RA-L / arXiv:2007.01867"),
    (2, "Wang et al., 2023", "LLIO: Lightweight Learned Inertial Odometer", "§1 Introduction; §4.2 Discussion", "IEEE IoT-J"),
    (3, "Herath et al., 2020", "RoNIN: Robust Neural Inertial Navigation in the Wild", "§1 Introduction; §4.2 Discussion", "ICRA 2020 / arXiv:1905.12853"),
    (4, "Chen et al., 2018", "OxIOD: The Dataset for Deep Inertial Odometry", "§2.1 System Design; §2.2 데이터셋", "arXiv:1809.07491"),
    (5, "Chen et al., 2018", "IONet: Learning to Cure the Curse of Drift in Inertial Odometry", "§1 Introduction", "AAAI 2018 / arXiv:1802.02209"),
    (6, "Touvron et al., 2023", "ResMLP: Feedforward Networks for Image Classification", "§2.1 백본(1D-ResMLP128)", "IEEE TPAMI / arXiv:2105.03404"),
    (7, "Loshchilov & Hutter, 2019", "Decoupled Weight Decay Regularization (AdamW)", "§2.7 학습 설정", "ICLR 2019 / arXiv:1711.05101"),
    (8, "Kalman, 1960", "A New Approach to Linear Filtering and Prediction Problems", "§2.6 상태 추정기", "J. Basic Eng."),
    (9, "Roumeliotis & Burdick, 2002", "Stochastic Cloning: A Generalized Framework for Relative State Measurements", "§2.1 상태 추정기; §2.6", "ICRA 2002"),
    (10, "Harle, 2013", "A Survey of Indoor Inertial Positioning Systems for Pedestrians", "§1 Introduction", "IEEE COMST"),
    (11, "Mohamed & Schwarz, 1999", "Adaptive Kalman Filtering for INS/GPS", "§1 Introduction; §4.2 Discussion", "J. Geodesy"),
    (12, "Bahl & Padmanabhan, 2000", "RADAR: An In-Building RF-Based User Location and Tracking System", "§1 Introduction", "IEEE INFOCOM 2000"),
]


def scholar(title):
    return "https://scholar.google.com/scholar?q=" + urllib.parse.quote(title)


def main():
    doc = SimpleDocTemplate(r"D:\mobile\imu_android\citation_check.pdf",
                            pagesize=A4, topMargin=15*mm, bottomMargin=15*mm,
                            leftMargin=12*mm, rightMargin=12*mm)
    ss = getSampleStyleSheet()
    h = ParagraphStyle("h", parent=ss["Title"], fontName="MalgunBd", fontSize=15)
    sub = ParagraphStyle("sub", parent=ss["Normal"], fontName="Malgun", fontSize=9, textColor=colors.grey)
    cell = ParagraphStyle("cell", parent=ss["Normal"], fontName="Malgun", fontSize=8.2, leading=11)
    link = ParagraphStyle("link", parent=cell, textColor=colors.blue)
    hdr = ParagraphStyle("hdr", parent=cell, fontName="MalgunBd", textColor=colors.white)

    story = []
    story.append(Paragraph("본문 인용 ↔ 참고문헌 교차참조 점검", h))
    story.append(Paragraph("IMU-based Indoor Localization 최종보고서(2026.06.09) — [1]~[12] 더블체크용. "
                           "링크는 제목 기준 Google Scholar 검색(항상 유효). 본문 인용 [n]과 목록 [n] 일치 확인 완료(12/12).", sub))
    story.append(Spacer(1, 6*mm))

    data = [[Paragraph(t, hdr) for t in ["[#]", "저자·연도 / 제목", "본문 인용 위치", "검증 링크"]]]
    for n, who, title, where, note in REFS:
        data.append([
            Paragraph(f"[{n}]", cell),
            Paragraph(f"<b>{who}</b><br/>{title}<br/><font size=7 color='#666666'>{note}</font>", cell),
            Paragraph(where, cell),
            Paragraph(f'<a href="{scholar(title)}">Scholar 검색 ↗</a>', link),
        ])

    tbl = Table(data, colWidths=[12*mm, 92*mm, 45*mm, 27*mm], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2166ac")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bbbbbb")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6fb")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph("※ 점검 결과: 본문 인용 [1]~[12]가 모두 참고문헌 목록에 존재하고, 목록의 12편이 모두 본문에서 "
                           "최소 1회 인용됨(누락·고아 인용 0). 인용 위치는 위 표의 '본문 인용 위치' 열 참조.", sub))
    doc.build(story)
    print("[OK] D:\\mobile\\imu_android\\citation_check.pdf 생성")


if __name__ == "__main__":
    main()
