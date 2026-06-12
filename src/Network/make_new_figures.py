"""make_new_figures.py — 보고서 그림 보강: 블록도 + 헤드라인 막대그래프."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

for fn in ("Malgun Gothic", "맑은 고딕", "Gulim"):
    try:
        font_manager.findfont(fn, fallback_to_default=False)
        plt.rcParams["font.family"] = fn; break
    except Exception: continue
plt.rcParams["axes.unicode_minus"] = False
LOG = Path(r"D:\mobile\imu_android\logs")

# ---------- 1) 블록도 ----------
def block_diagram():
    fig, ax = plt.subplots(figsize=(8.6, 4.0)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.10",
                    linewidth=1.3, edgecolor="#33485e", facecolor=fc))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=9.2)
    def arr(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                    mutation_scale=13, linewidth=1.3, color="#33485e"))
    box(0.3, 4.7, 9.4, 0.9, "IMU 센서  ·  Accelerometer · Gyroscope · RotationVector  (100 Hz)", "#eaf2fb")
    box(0.3, 3.2, 9.4, 1.1, "전처리(Preprocessing)\n2초 정지구간 바이어스 영점보정 · per-sample 중력정렬 · 채널 정규화 · 1초(100×6) 윈도우", "#fff3e0")
    box(0.3, 1.95, 9.4, 0.85, "1D-ResMLP128 백본   patch embed → ResMLP×6 → mean pool → feature [B,128]", "#e8f5e9")
    box(0.3, 0.95, 3.0, 0.7, "회귀 헤드(변위)\n변위 μ (3차원)", "#f3e8fd")
    box(3.5, 0.95, 3.0, 0.7, "회귀 헤드(공분산)\nlog σ² (3차원)", "#f3e8fd")
    box(6.7, 0.95, 3.0, 0.7, "분류 헤드\np (7클래스, softmax)", "#f3e8fd")
    box(0.3, 0.05, 9.4, 0.6, "상태 추정기   경로 A: SC-EKF(15-dim)   |   경로 B: RotVec dead-reckoning + PDR-hybrid (단말 기본)", "#eef2f7")
    arr(5, 4.7, 5, 4.32); arr(5, 3.2, 5, 2.82); arr(5, 1.95, 5, 1.67); arr(5, 0.95, 5, 0.67)
    fig.tight_layout()
    fig.savefig(LOG/"fig1_pipeline_new.png", dpi=170, bbox_inches="tight")
    print("saved fig1_pipeline_new.png")

# ---------- 2) 헤드라인 막대 (Net vs EKF) ----------
def headline_bar():
    cats = ["trolley","handbag","handheld","pocket","running","slow_walk","large_scale"]
    net  = [1.9764,1.2018,1.5706,1.3682,1.3139,1.3399,2.3559]
    ekf  = [1.6424,3.0852,7.7783,4.8744,4.6835,3.8677,19.3046]
    x = np.arange(len(cats)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    b1 = ax.bar(x-w/2, net, w, label="네트워크 단독 (dead-reckoning)", color="#2166ac")
    b2 = ax.bar(x+w/2, ekf, w, label="EKF 결합", color="#d6604d")
    ax.set_yscale("log")
    ax.set_ylabel("RMSE$_{XY}$ (m, log)")
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=15, ha="right", fontsize=9)
    ax.set_title("카테고리별 네트워크 단독 vs EKF 결합 측위 오차", fontsize=11)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.legend(fontsize=9, loc="upper left")
    # trolley(유일 EKF 우위) 강조
    ax.annotate("trolley만\nEKF 우위", xy=(0+w/2, 1.64), xytext=(0.1, 0.55),
                fontsize=8.5, color="#1a7a3a", ha="center",
                arrowprops=dict(arrowstyle="->", color="#1a7a3a", lw=1))
    for i,(n,e) in enumerate(zip(net,ekf)):
        r = e/n
        ax.text(i+w/2, e*1.06, f"{r:.1f}×" if r>=1 else f"+{(1-r)*100:.0f}%",
                ha="center", va="bottom", fontsize=7.5,
                color="#1a7a3a" if r<1 else "#9b2226")
    fig.tight_layout()
    fig.savefig(LOG/"fig_headline_bar.png", dpi=170, bbox_inches="tight")
    print("saved fig_headline_bar.png")

# ---------- 3) 온디바이스 앱 합성 (작품사진1 실시간궤적 + 작품사진3 평면도, 이름 블러) ----------
def app_composite():
    from PIL import Image, ImageFilter, ImageDraw
    SRC = Path(r"D:\mobile\imu_android\작품사진")
    a = Image.open(SRC/"작품사진1.jpg").convert("RGB")   # 실시간 궤적 + 위치/지연/컨트롤
    b = Image.open(SRC/"작품사진3.jpg").convert("RGB")   # 평면도 오버레이
    # 작품사진3: 312~316호 실명 영역 블러 (분수 좌표 — 우중앙)
    W, H = b.size
    bx0, by0, bx1, by1 = int(0.55*W), int(0.25*H), int(0.86*W), int(0.52*H)
    region = b.crop((bx0, by0, bx1, by1)).filter(ImageFilter.GaussianBlur(9))
    b.paste(region, (bx0, by0))
    # 같은 높이로 리사이즈 후 나란히
    TH = 1400
    ar = a.resize((int(a.width*TH/a.height), TH)); br = b.resize((int(b.width*TH/b.height), TH))
    gap, pad, top = 40, 30, 60
    cw = ar.width + br.width + gap + pad*2
    canvas = Image.new("RGB", (cw, TH+top+pad), "white")
    canvas.paste(ar, (pad, top)); canvas.paste(br, (pad+ar.width+gap, top))
    d = ImageDraw.Draw(canvas)
    try:
        from PIL import ImageFont
        f = ImageFont.truetype("malgun.ttf", 34)
    except Exception:
        f = None
    d.text((pad+ar.width//2-160, 16), "(a) 실시간 측위 궤적 · 추론 13ms", fill="black", font=f)
    d.text((pad+ar.width+gap+br.width//2-150, 16), "(b) 평면도 오버레이 (미래관 3층)", fill="black", font=f)
    canvas.save(LOG/"fig_app.png")
    print("saved fig_app.png", canvas.size)

block_diagram(); headline_bar(); app_composite()
