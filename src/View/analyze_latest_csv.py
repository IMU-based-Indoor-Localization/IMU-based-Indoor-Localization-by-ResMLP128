"""
analyze_latest_csv.py — latest.csv raw IMU 구간/자세 분석
==========================================================
목적: "캘리브레이션 - 정지 - 5m 이동 - 정지 - 180° 회전 - 5m 이동" 의 5m 왕복
구조가 raw 데이터에 실제로 담겨 있는지, rotVec yaw 가 180° 회전을 잡는지 확인.
모델/DR 파이프라인 문제와 데이터 자체 문제를 분리한다.

numpy 만 사용 (torch/scipy 불필요). 실행:
    python src/View/analyze_latest_csv.py latest.csv
"""
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FS = 100.0


def read_long(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        f.readline()
        for ln in f:
            p = ln.strip().split(",")
            if len(p) != 6:
                continue
            try:
                rows.append((p[0], int(p[1]), float(p[2]), float(p[3]),
                             float(p[4]), float(p[5])))
            except ValueError:
                continue
    out = {}
    for s in ("acc", "gyr", "linAcc", "rotVec"):
        sub = [r for r in rows if r[0] == s]
        if not sub:
            continue
        ts = np.array([r[1] for r in sub], dtype=np.int64)
        v = np.array([[r[2], r[3], r[4], r[5]] for r in sub], dtype=np.float64)
        o = np.argsort(ts)
        out[s] = (ts[o], v[o])
    return out


def quat_to_yaw(q):
    """q[:, (x,y,z,w)] → yaw(rad).  yaw = atan2(2(xy+wz), 1-2(yy+zz))."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "latest.csv"
    raw = read_long(path)
    print(f"=== {path} ===")
    for s, (ts, v) in raw.items():
        dur = (ts[-1] - ts[0]) / 1e9
        print(f"  {s:7s} n={len(ts):6d}  dur={dur:5.1f}s  rate={len(ts)/dur:5.0f}Hz")

    la_ts, la = raw["linAcc"][0], raw["linAcc"][1][:, :3]
    gy_ts, gy = raw["gyr"][0],    raw["gyr"][1][:, :3]
    rv_ts, rv = raw["rotVec"][0], raw["rotVec"][1]

    # 100Hz 균일 그리드
    t0 = max(la_ts[0], gy_ts[0], rv_ts[0])
    t1 = min(la_ts[-1], gy_ts[-1], rv_ts[-1])
    n = int((t1 - t0) / 1e9 * FS)
    grid = (t0 + (np.arange(n) / FS * 1e9)).astype(np.float64)
    gf = grid

    def interp3(src_ts, src_v):
        s = src_ts.astype(np.float64)
        return np.stack([np.interp(gf, s, src_v[:, k]) for k in range(3)], 1)

    linacc = interp3(la_ts, la)
    gyr    = interp3(gy_ts, gy)

    # rotVec yaw — 샘플별 yaw 계산 후 unwrap, 그리드로 보간
    yaw_rv = np.unwrap(quat_to_yaw(rv))
    yaw = np.interp(gf, rv_ts.astype(np.float64), yaw_rv)
    yaw_deg = np.degrees(yaw)

    t_s = (grid - grid[0]) / 1e9   # 초

    # ── 0.5초 윈도우 motion 에너지 ──────────────────────────────
    win = int(0.5 * FS)
    print(f"\n=== 0.5초 구간별 motion 프로파일 (총 {t_s[-1]:.1f}s) ===")
    print(f"  {'t(s)':>8s}  {'gyrRMS':>8s}  {'accRMS':>8s}  {'yaw(°)':>8s}  상태")
    STAT = 0.08   # rad/s — 프로젝트 STATIC_GYR_RMS_THRESHOLD
    seg_rows = []
    for i in range(0, n - win, win):
        g = gyr[i:i + win]
        a = linacc[i:i + win]
        grms = float(np.sqrt((g ** 2).sum(1).mean()))
        arms = float(np.sqrt((a ** 2).sum(1).mean()))
        ymid = yaw_deg[i + win // 2]
        state = "정지" if grms < STAT else "이동"
        seg_rows.append((t_s[i], grms, arms, ymid, state))
        print(f"  {t_s[i]:8.1f}  {grms:8.4f}  {arms:8.3f}  {ymid:8.1f}  {state}")

    # ── 구간 병합 (연속 같은 상태) ───────────────────────────────
    print(f"\n=== 구간 요약 ===")
    segs = []
    cur = seg_rows[0][4]
    start = seg_rows[0][0]
    for r in seg_rows[1:]:
        if r[4] != cur:
            segs.append((cur, start, r[0]))
            cur = r[4]
            start = r[0]
    segs.append((cur, start, t_s[-1]))
    for st, a, b in segs:
        print(f"  {a:5.1f}s ~ {b:5.1f}s  ({b-a:4.1f}s)  {st}")

    # ── yaw 프로파일 (180° 회전 검출) ────────────────────────────
    print(f"\n=== 자세(yaw) 프로파일 ===")
    print(f"  시작 yaw      : {yaw_deg[0]:8.1f}°")
    print(f"  종료 yaw      : {yaw_deg[-1]:8.1f}°")
    print(f"  누적 yaw 변화 : {yaw_deg[-1] - yaw_deg[0]:8.1f}°  (180° 회전이면 ±180 근처)")

    # 0.5초 윈도우별 yaw 변화율 → 회전(turn) 구간 = |rate| 가 큰 연속 구간
    TURN_RATE = 40.0   # °/0.5s — 보행 중 yaw 흔들림(~20°)과 의도적 회전 구분
    samp_yaw = yaw_deg[::win][:len(seg_rows)]
    rates = np.diff(samp_yaw)
    turn_idx = [i for i, r in enumerate(rates) if abs(r) > TURN_RATE]
    if turn_idx:
        ta, tb = turn_idx[0], turn_idx[-1] + 1
        turn_ang = samp_yaw[tb] - samp_yaw[ta]
        print(f"  회전 구간     : t={t_s[ta*win]:.1f}~{t_s[tb*win]:.1f}s "
              f"({(tb-ta)*0.5:.1f}s 동안 {turn_ang:.0f}° 회전)")
    else:
        print(f"  회전 구간     : 뚜렷한 급회전 미검출 (보행 중 흔들림만)")

    # ── 해석 ────────────────────────────────────────────────────
    print(f"\n=== 해석 ===")
    moving = [s for s in segs if s[0] == "이동"]
    static = [s for s in segs if s[0] == "정지"]
    print(f"  정지 구간 {len(static)}개, 이동 구간 {len(moving)}개")
    print(f"  기대 구조 [정지·5m이동·정지·180°회전·5m이동] 대비:")
    print(f"   - 데이터에 정지/이동 구간이 분리되어 나타나는가")
    print(f"   - rotVec yaw 누적 변화가 ±180° 근처인가 (회전이 데이터에 잡히는가)")


if __name__ == "__main__":
    main()
