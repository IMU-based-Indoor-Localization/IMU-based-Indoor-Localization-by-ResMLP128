"""
imu_oxiod_vs_android.py
=======================
HANDOFF P40 §5 단계 1 — OxIOD raw IMU (학습 분포) 와 Android raw IMU CSV (실행 분포) 의
channel-wise 통계 비교.

목적
----
"학습 데이터 (iPhone Core Motion 출력) 와 Android raw IMU 사이에 *어느 채널의*
*어떤 통계가* 다른가" 를 정량 파악 → P40 §5 단계 2 보정 선택의 근거.

사용 예
-------
1) OxIOD 만 (Android 측정 전 점검):
   python src/View/imu_oxiod_vs_android.py \
       --oxiod_dir src/TLIO_Oxford_Dataset \
       --oxiod_scenario oxford_handheld_1

2) Android 만 (단말 CSV 만 확인):
   python src/View/imu_oxiod_vs_android.py \
       --android_csv csv/imu_csv/imu_record_1717830000000.csv

3) 양쪽 비교 (+ plot 저장):
   python src/View/imu_oxiod_vs_android.py \
       --oxiod_dir src/TLIO_Oxford_Dataset \
       --oxiod_scenario oxford_handheld_1 \
       --android_csv csv/imu_csv/imu_record_xxx.csv \
       --plot_out logs/cmp_still.png

입력 형식
---------
- OxIOD (TLIO 변환): {scenario}/imu0_resampled.npy
    채널 (description.json 기준):
      col[0]      ts_us
      col[1:4]    gyr(3)        rad/s
      col[4:7]    acc(3)        g 단위 (gravity 제거)
      col[7:10]   gravity(3)    g 단위 (정지 시 ~1g norm)
      col[10:13]  attitude(3)   Euler (rad)
      col[13]     label
      col[14:18]  qxyzw_World_Device(4)
      col[18:21]  pos_World_Device(3) m
      col[21:24]  vel_World(3) m/s
    Sampling: 100 Hz

- Android CSV (ImuTestActivity P40 CSV 기능):
    long-format: sensor,ts_ns,x,y,z,w
      sensor ∈ {acc, gyr, linAcc, rotVec}
      acc/gyr/linAcc: m/s² 또는 rad/s, w=0.0
      rotVec: 단위 quaternion (x,y,z,w)
    Sampling: SENSOR_DELAY_FASTEST (~100-200 Hz 단말 의존)

의존성
------
- numpy (필수)
- matplotlib (옵션 — plot 출력 시만 필요)

출력
----
- 콘솔: 채널별 mean / std / RMS / range / gravity-norm-check (단위 자동 추론) 표
- 옵션 plot (--plot_out 지정 시): violin / FFT / 시계열 / 윈도우별 std
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple, Dict

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. 데이터 로딩
# ─────────────────────────────────────────────────────────────────────────────

def load_oxiod(scenario_dir: str) -> Dict[str, np.ndarray]:
    """TLIO 변환 OxIOD 시나리오 폴더에서 imu0_resampled.npy 를 읽어 채널별로 분리한다.

    반환: dict
      'ts_us'    [N]    timestamps (us)
      'gyr'      [N, 3] body-frame gyroscope (rad/s)
      'acc'      [N, 3] body-frame linear acc (g 단위, gravity 제거됨)
      'gravity'  [N, 3] body-frame gravity 벡터 (g 단위, norm ~1)
      'attitude' [N, 3] Euler (rad)
      'quat'     [N, 4] qxyzw World→Device
    """
    npy = os.path.join(scenario_dir, 'imu0_resampled.npy')
    if not os.path.exists(npy):
        raise FileNotFoundError(f"imu0_resampled.npy 없음: {npy}")
    data = np.load(npy)
    if data.shape[1] < 18:
        raise ValueError(f"npy 컬럼 부족: shape={data.shape}, 24 컬럼 기대")
    return {
        'ts_us'   : data[:, 0],
        'gyr'     : data[:, 1:4].astype(np.float32),
        'acc'     : data[:, 4:7].astype(np.float32),
        'gravity' : data[:, 7:10].astype(np.float32),
        'attitude': data[:, 10:13].astype(np.float32),
        'quat'    : data[:, 14:18].astype(np.float32),
    }


def load_android_csv(csv_path: str) -> Dict[str, np.ndarray]:
    """Android ImuTestActivity 의 long-format CSV 를 읽어 센서별로 pivot.

    long-format: sensor,ts_ns,x,y,z,w
    반환: dict (센서가 없으면 키 누락)
      'acc'    {'ts_ns': [Na], 'xyz': [Na, 3]}
      'gyr'    {'ts_ns': [Ng], 'xyz': [Ng, 3]}
      'linAcc' {'ts_ns': [Nl], 'xyz': [Nl, 3]}
      'rotVec' {'ts_ns': [Nr], 'xyzw': [Nr, 4]}
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Android CSV 없음: {csv_path}")

    # numpy genfromtxt 는 dtype mixed 처리. dtype='object' 후 분리.
    rows = []
    with open(csv_path, encoding='utf-8') as f:
        header = f.readline().strip()
        if not header.startswith('sensor,'):
            raise ValueError(f"CSV 헤더 형식 다름: {header!r}")
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            parts = ln.split(',')
            if len(parts) != 6: continue
            try:
                rows.append((parts[0], int(parts[1]),
                            float(parts[2]), float(parts[3]),
                            float(parts[4]), float(parts[5])))
            except ValueError:
                continue

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for sensor in ('acc', 'gyr', 'linAcc', 'rotVec'):
        sub = [r for r in rows if r[0] == sensor]
        if not sub: continue
        ts  = np.array([r[1] for r in sub], dtype=np.int64)
        xyz = np.array([[r[2], r[3], r[4]] for r in sub], dtype=np.float32)
        if sensor == 'rotVec':
            xyzw = np.array([[r[2], r[3], r[4], r[5]] for r in sub], dtype=np.float32)
            out[sensor] = {'ts_ns': ts, 'xyzw': xyzw}
        else:
            out[sensor] = {'ts_ns': ts, 'xyz': xyz}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. 채널 통계 + 단위 추론
# ─────────────────────────────────────────────────────────────────────────────

def channel_stats(arr: np.ndarray, name: str) -> dict:
    """[N, D] 배열의 채널별 mean/std/min/max/RMS 계산."""
    if arr.ndim != 2:
        raise ValueError(f"{name}: 2D 배열 기대, got {arr.shape}")
    return {
        'name'   : name,
        'N'      : arr.shape[0],
        'D'      : arr.shape[1],
        'mean'   : arr.mean(axis=0),
        'std'    : arr.std(axis=0),
        'min'    : arr.min(axis=0),
        'max'    : arr.max(axis=0),
        'rms'    : np.sqrt((arr ** 2).mean(axis=0)),
        'abs_max': np.abs(arr).max(axis=0),
    }


def infer_acc_unit_from_gravity_norm(gravity_or_acc_at_rest: np.ndarray) -> str:
    """gravity 벡터 (또는 정지 시 acc raw) 의 norm 평균으로 단위 추론.
    norm ≈ 1.0 → [g], norm ≈ 9.81 → [m/s²]"""
    norms = np.linalg.norm(gravity_or_acc_at_rest, axis=1)
    m = norms.mean()
    if abs(m - 1.0) < 0.2:
        return f'[g] (norm mean = {m:.4f})'
    if abs(m - 9.81) < 1.5:
        return f'[m/s²] (norm mean = {m:.4f})'
    return f'unknown (norm mean = {m:.4f})'


def print_stats(stats: dict, ch_labels=('x', 'y', 'z', 'w')) -> None:
    """채널별 stats 표 인쇄."""
    print(f"\n  {stats['name']}  (N={stats['N']:,}, D={stats['D']})")
    print(f"    {'ch':>3s}  {'mean':>10s}  {'std':>10s}  {'min':>10s}  {'max':>10s}  {'rms':>10s}  {'|max|':>10s}")
    for i in range(stats['D']):
        lbl = ch_labels[i] if i < len(ch_labels) else f'c{i}'
        print(f"    {lbl:>3s}  {stats['mean'][i]:>10.4f}  {stats['std'][i]:>10.4f}  "
              f"{stats['min'][i]:>10.4f}  {stats['max'][i]:>10.4f}  "
              f"{stats['rms'][i]:>10.4f}  {stats['abs_max'][i]:>10.4f}")


def print_compare(ox_stats: dict, ad_stats: dict, label: str,
                  unit_conv_ox_to_ad: float = 1.0) -> None:
    """OxIOD vs Android 같은 채널 통계 비교 (mean / std / RMS 비율).
    unit_conv_ox_to_ad: OxIOD 단위 → Android 단위 변환 계수 (예: g → m/s² = 9.81)."""
    print(f"\n  === 비교: {label} (OxIOD ↔ Android) ===")
    print(f"  unit conversion ox→ad: ×{unit_conv_ox_to_ad}")
    print(f"    {'ch':>3s}  {'mean_ox*c':>12s}  {'mean_ad':>10s}  {'Δmean':>10s}  "
          f"{'std_ox*c':>10s}  {'std_ad':>10s}  {'std ratio (ad/ox)':>18s}")
    D = min(ox_stats['D'], ad_stats['D'])
    for i in range(D):
        ch = ('x', 'y', 'z', 'w')[i] if i < 4 else f'c{i}'
        mox = ox_stats['mean'][i] * unit_conv_ox_to_ad
        mad = ad_stats['mean'][i]
        sox = ox_stats['std'][i] * abs(unit_conv_ox_to_ad)
        sad = ad_stats['std'][i]
        ratio = (sad / sox) if abs(sox) > 1e-9 else float('inf')
        print(f"    {ch:>3s}  {mox:>12.4f}  {mad:>10.4f}  {mad-mox:>+10.4f}  "
              f"{sox:>10.4f}  {sad:>10.4f}  {ratio:>18.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. 윈도우별 std / FFT
# ─────────────────────────────────────────────────────────────────────────────

def windowed_std(arr: np.ndarray, win: int, stride: int) -> np.ndarray:
    """[N, D] → [Nw, D] 윈도우별 std (보행 신호 진폭 분포 비교용)."""
    Nw = max(0, (len(arr) - win) // stride + 1)
    out = np.empty((Nw, arr.shape[1]), dtype=np.float32)
    for i in range(Nw):
        s = i * stride
        out[i] = arr[s:s+win].std(axis=0)
    return out


def fft_power(arr_ch: np.ndarray, fs: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
    """1D 채널 시계열의 single-sided power spectrum (0..fs/2).
    arr_ch: [N], fs Hz. 반환 (freqs, power)."""
    arr_ch = arr_ch - arr_ch.mean()      # DC 제거
    N = len(arr_ch)
    F = np.fft.rfft(arr_ch)
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    power = (np.abs(F) ** 2) / max(N, 1)
    return freqs, power


def print_fft_peaks(arr: np.ndarray, name: str, fs: float = 100.0, top_n: int = 3) -> None:
    """채널별 FFT power peak 주파수 출력 (보행 cadence 식별용)."""
    print(f"\n  FFT peaks ({name}, fs={fs}Hz, top {top_n}):")
    for i in range(arr.shape[1]):
        freqs, power = fft_power(arr[:, i], fs=fs)
        # DC 근방 (0~0.5Hz) 제외
        mask = freqs >= 0.5
        if not mask.any(): continue
        f_sub, p_sub = freqs[mask], power[mask]
        idx = np.argsort(-p_sub)[:top_n]
        peaks = [f'{f_sub[k]:5.2f}Hz({p_sub[k]:.2e})' for k in idx]
        ch = ('x', 'y', 'z', 'w')[i] if i < 4 else f'c{i}'
        print(f"    {ch}: {' | '.join(peaks)}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. 옵션 plot (matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def make_plots(ox: Optional[dict], ad: Optional[dict], out_path: str) -> None:
    """matplotlib 있을 때만. 4개 subplot: 시계열 / violin / FFT / windowed std."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\n  [SKIP] matplotlib 없음 → plot 생성 안 함. 콘솔 통계만 사용.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) 시계열 — acc x (앞 1000 샘플)
    ax = axes[0, 0]
    if ox is not None:
        n = min(1000, len(ox['acc']))
        ax.plot(np.arange(n) / 100.0, ox['acc'][:n, 0], 'b-', label='OxIOD acc_x [g]', alpha=0.7)
    if ad is not None and 'linAcc' in ad:
        n = min(1000, len(ad['linAcc']['xyz']))
        ax.plot(np.arange(n) / 100.0, ad['linAcc']['xyz'][:n, 0], 'r-', label='Android linAcc_x [m/s²]', alpha=0.7)
    ax.set_xlabel('time (s)'); ax.set_ylabel('acc_x')
    ax.legend(); ax.set_title('Time series — acc x')
    ax.grid(True, alpha=0.3)

    # (2) Violin — acc 채널별 분포
    ax = axes[0, 1]
    data, labels = [], []
    if ox is not None:
        for i, ch in enumerate(['x', 'y', 'z']):
            data.append(ox['acc'][:, i]); labels.append(f'Ox.acc_{ch}')
    if ad is not None and 'linAcc' in ad:
        for i, ch in enumerate(['x', 'y', 'z']):
            data.append(ad['linAcc']['xyz'][:, i]); labels.append(f'And.linAcc_{ch}')
    if data:
        ax.violinplot(data, showmeans=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title('Distribution per channel')
    ax.grid(True, alpha=0.3)

    # (3) FFT — gyr_z (회전 cadence 식별)
    ax = axes[1, 0]
    if ox is not None:
        freqs, power = fft_power(ox['gyr'][:, 2], fs=100.0)
        ax.semilogy(freqs[freqs < 50], power[freqs < 50], 'b-', label='OxIOD gyr_z', alpha=0.7)
    if ad is not None and 'gyr' in ad:
        freqs, power = fft_power(ad['gyr']['xyz'][:, 2], fs=100.0)
        ax.semilogy(freqs[freqs < 50], power[freqs < 50], 'r-', label='Android gyr_z', alpha=0.7)
    ax.set_xlabel('Hz'); ax.set_ylabel('power')
    ax.legend(); ax.set_title('FFT power — gyr z')
    ax.grid(True, alpha=0.3)

    # (4) Windowed std — acc magnitude (1초 윈도우)
    ax = axes[1, 1]
    if ox is not None:
        wstd = windowed_std(ox['acc'], win=100, stride=50)
        mag = np.linalg.norm(wstd, axis=1)
        ax.hist(mag, bins=50, alpha=0.5, color='b', label='OxIOD')
    if ad is not None and 'linAcc' in ad:
        wstd = windowed_std(ad['linAcc']['xyz'], win=100, stride=50)
        mag = np.linalg.norm(wstd, axis=1)
        ax.hist(mag, bins=50, alpha=0.5, color='r', label='Android')
    ax.set_xlabel('1s window std (acc magnitude)'); ax.set_ylabel('count')
    ax.legend(); ax.set_title('Windowed std distribution — acc')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=110)
    print(f"\n  [OK] plot 저장: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--oxiod_dir',      default=None, help='TLIO_Oxford_Dataset 루트')
    ap.add_argument('--oxiod_scenario', default=None, help='시나리오 폴더명 (예: oxford_handheld_1)')
    ap.add_argument('--android_csv',    default=None, help='Android ImuTestActivity CSV 경로')
    ap.add_argument('--plot_out',       default=None, help='plot PNG 출력 경로 (옵션)')
    args = ap.parse_args()

    ox = None
    if args.oxiod_dir and args.oxiod_scenario:
        scen = os.path.join(args.oxiod_dir, args.oxiod_scenario)
        print(f"\n[OxIOD] 로딩: {scen}")
        ox = load_oxiod(scen)
        unit = infer_acc_unit_from_gravity_norm(ox['gravity'])
        print(f"  단위 추론: acc/gravity 는 {unit}")
        print_stats(channel_stats(ox['acc'],     'OxIOD acc'))
        print_stats(channel_stats(ox['gyr'],     'OxIOD gyr'))
        print_stats(channel_stats(ox['gravity'], 'OxIOD gravity'))
        print_fft_peaks(ox['acc'], 'OxIOD acc', fs=100.0)
        print_fft_peaks(ox['gyr'], 'OxIOD gyr', fs=100.0)

    ad = None
    if args.android_csv:
        print(f"\n[Android] 로딩: {args.android_csv}")
        ad = load_android_csv(args.android_csv)
        for sensor in ('acc', 'gyr', 'linAcc', 'rotVec'):
            if sensor not in ad: continue
            key = 'xyzw' if sensor == 'rotVec' else 'xyz'
            arr = ad[sensor][key]
            print_stats(channel_stats(arr, f'Android {sensor}'))
            # 단위 추론 — acc 는 raw gravity 포함 → norm 으로 g/m/s² 추론
            if sensor == 'acc':
                # 정지 가정 — 첫 200 sample 평균이 gravity 벡터
                rest = arr[:min(200, len(arr))]
                unit = infer_acc_unit_from_gravity_norm(rest)
                print(f"    단위 추론 (rest 200 sample): {unit}")
        if 'gyr' in ad:
            print_fft_peaks(ad['gyr']['xyz'], 'Android gyr', fs=100.0)

    # 양쪽 있으면 비교
    if ox is not None and ad is not None:
        print(f"\n{'='*70}\n  비교 (OxIOD ↔ Android)\n{'='*70}")
        # OxIOD acc 는 [g], Android linAcc 는 [m/s²] — 단위 변환 ×9.81
        if 'linAcc' in ad:
            print_compare(channel_stats(ox['acc'], 'OxIOD acc'),
                          channel_stats(ad['linAcc']['xyz'], 'Android linAcc'),
                          'acc (gravity 제거)', unit_conv_ox_to_ad=9.81)
        if 'gyr' in ad:
            print_compare(channel_stats(ox['gyr'], 'OxIOD gyr'),
                          channel_stats(ad['gyr']['xyz'], 'Android gyr'),
                          'gyr', unit_conv_ox_to_ad=1.0)

    # 옵션 plot
    if args.plot_out:
        make_plots(ox, ad, args.plot_out)

    print()


if __name__ == '__main__':
    main()
