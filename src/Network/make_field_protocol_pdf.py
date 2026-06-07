# -*- coding: utf-8 -*-
"""현장 측정 프로토콜(절대 GT, 5경로) → 1~2쪽 PDF. 맑은 고딕(한글)."""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

pdfmetrics.registerFont(TTFont("Malgun", r"C:/Windows/Fonts/malgun.ttf"))
pdfmetrics.registerFont(TTFont("MalgunB", r"C:/Windows/Fonts/malgunbd.ttf"))
pdfmetrics.registerFontFamily("Malgun", normal="Malgun", bold="MalgunB", italic="Malgun", boldItalic="MalgunB")

OUT = Path(r"D:\mobile\imu_android\현장측정_프로토콜_GT.pdf")

H1 = ParagraphStyle("H1", fontName="MalgunB", fontSize=15, spaceAfter=4, textColor=colors.HexColor("#6A1B9A"))
SUB = ParagraphStyle("SUB", fontName="Malgun", fontSize=8.5, spaceAfter=8, textColor=colors.HexColor("#666666"))
H2 = ParagraphStyle("H2", fontName="MalgunB", fontSize=11, spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#1565C0"))
BODY = ParagraphStyle("BODY", fontName="Malgun", fontSize=9.5, leading=14, spaceAfter=2)
CELL = ParagraphStyle("CELL", fontName="Malgun", fontSize=8.5, leading=11)
CELLB = ParagraphStyle("CELLB", fontName="MalgunB", fontSize=8.5, leading=11)


def b(t):  # bullet
    return Paragraph("• " + t, BODY)


story = []
story.append(Paragraph("IMU 측위 — 절대 GT 현장 측정 프로토콜", H1))
story.append(Paragraph("5경로 · 마크(볼륨키) 기반 · 회별 개별 측정", SUB))
story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CCCCCC"), spaceAfter=6))

story.append(Paragraph("0. 사전 체크 (한 번)", H2))
story.append(b("앱 메뉴 → <b>단위보정(A): OFF → 탭 → ON</b> 확인 (필수)"))
story.append(b("[시작] 후 <b>걷지 말고 10초</b> 정지 → 드리프트 &lt;1m면 OK. 5~10m 튀면 측정 중단·보고."))

story.append(Paragraph("1. 매 회 공통", H2))
story.append(b("단위보정(A) <b>ON</b> · 매 회 <b>[초기화]</b>로 누적 리셋"))
story.append(b("속도: <b>편안한 일정 보행 ~1.0 m/s</b> (달리기·초저속 금지, 회마다 같게)"))
story.append(b("자세: handheld, 폰을 <b>진행방향</b>으로 안정되게"))

story.append(Paragraph("2. 시작 → 마크 → 끝", H2))
story.append(b("<b>시작</b>: W1에 서서 [초기화] → [시작] → <b>2초 완전정지(보정, 절대 안 움직임)</b> → 그 자리서 <b>볼륨키(마크 #1)</b> → 출발"))
story.append(b("<b>각 웨이포인트</b>: ~1초 정지 → <b>볼륨키 1회</b> (토스트 “마크 #N” 확인). 화면 위치 무관."))
story.append(b("<b>끝</b>: 마지막 지점 마크 → [정지] → 메뉴 <b>마크 내보내기 + 경로 내보내기</b>"))
story.append(b("<b>루프/왕복(R3·R4)</b>: 시작점 W1으로 <b>정확히 복귀</b> 후 마지막 마크"))

story.append(Paragraph("3. 5개 경로 (기하 다양화)", H2))
rows = [
    [Paragraph("경로", CELLB), Paragraph("유형", CELLB), Paragraph("목적", CELLB), Paragraph("마크", CELLB), Paragraph("예시 동선", CELLB)],
    [Paragraph("R1", CELL), Paragraph("직선", CELL), Paragraph("순수 직진 스케일", CELL), Paragraph("3", CELL), Paragraph("W1→W2→W3 (복도)", CELL)],
    [Paragraph("R2", CELL), Paragraph("L자(1턴)", CELL), Paragraph("단일 90° 회전", CELL), Paragraph("3", CELL), Paragraph("복도 직진 후 방 진입", CELL)],
    [Paragraph("R3", CELL), Paragraph("왕복(180°)", CELL), Paragraph("회전+복귀(끝=시작)", CELL), Paragraph("3", CELL), Paragraph("W1→W3→W1", CELL)],
    [Paragraph("R4", CELL), Paragraph("루프(ㅁ/ㄷ)", CELL), Paragraph("누적 회전·루프클로저", CELL), Paragraph("4~5", CELL), Paragraph("방 블록 한 바퀴(끝=시작)", CELL)],
    [Paragraph("R5", CELL), Paragraph("복합", CELL), Paragraph("다중 회전", CELL), Paragraph("5~6", CELL), Paragraph("방 진입·복도 여러 꺾임", CELL)],
]
t = Table(rows, colWidths=[18*mm, 22*mm, 42*mm, 12*mm, 58*mm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDE7F6")),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BBBBBB")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(t)

story.append(Paragraph("4. 웨이포인트·기록", H2))
story.append(b("웨이포인트 = <b>좌표를 아는 물리 지점</b>(문틀·복도 모퉁이·기둥). 평면도 치수선으로 (x,y) 읽기."))
story.append(b("원점=W1, X=복도방향, Y=복도→실내. <b>5경로 공유</b>(좌표 1회만 측정)."))
story.append(b("기록: 각 경로의 <b>웨이포인트 좌표(순서대로)</b> + 회당 <b>marks_*.csv</b> · <b>track_PATH_B_*.csv</b>"))

story.append(Paragraph("5. 주의", H2))
story.append(b("마크는 <b>반드시 그 지점에 서서</b> 누름 (GT=실제 위치). 마크 수 = 웨이포인트 수."))
story.append(b("<b>회별로 따로</b> 측정·확인 (한 번에 몰아서 X) — 추출 순서: R1(파일럿) → 확인 → R2~R5."))

doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                        leftMargin=16*mm, rightMargin=16*mm, topMargin=14*mm, bottomMargin=12*mm)
doc.build(story)
print(f"[OK] {OUT}")
