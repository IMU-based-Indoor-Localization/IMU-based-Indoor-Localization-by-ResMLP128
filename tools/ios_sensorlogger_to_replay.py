"""
ios_sensorlogger_to_replay.py — iOS Sensor Logger CSV → Android replay CSV 변환
==============================================================================
입력  : Sensor Logger (iOS) 앱이 출력한 폴더 (4개 센서별 CSV)
출력  : Android ImuCollector.startReplay 가 읽는 단일 CSV (sensor,ts_ns,x,y,z,w)

iOS 폴더 구조 (Sensor Logger v1.59 기준):
  - Accelerometer.csv   : user acceleration (gravity 제거된 linear accel, m/s²)
  - Gravity.csv         : gravity 벡터 (device frame, m/s²)
  - Gyroscope.csv       : 회전 각속도 (rad/s)
  - Orientation.csv     : 디바이스 자세 (quaternion qx/qy/qz/qw + yaw/roll/pitch)

iOS 컬럼 순서 주의: time,seconds_elapsed,z,y,x  ← Z,Y,X 역순!
Orientation 컬럼 순서: time,seconds_elapsed,yaw,qx,qz,roll,qw,qy,pitch  ← 섞임

좌표계 / gravity 부호 변환:
  iOS Gravity (face-up rest)  = (0, 0, -9.81)  ← Z out of screen = -gravity 방향
  Android acc (face-up rest)  = (0, 0, +9.81)  ← raw accelerometer "specific force"
  따라서:
    Android acc      = iOS Accelerometer - iOS Gravity   (부호 반전 후 합산)
    Android linAcc   = iOS Accelerometer                 (그대로 — gravity 이미 제거됨)
    Android gyr      = iOS Gyroscope                     (그대로)
    Android rotVec   = iOS Orientation (qx, qy, qz, qw)  (그대로)

사용:
  python tools/ios_sensorlogger_to_replay.py <ios_dir> <output_csv>

예:
  python tools/ios_sensorlogger_to_replay.py \\
    "C:/Users/zihyun2777/Downloads/2026-05-25_05-54-49" \\
    csv/imu_csv/ios_replay_5_25.csv

이후:
  D:/SDK/platform-tools/adb.exe -s adb-... push <output_csv> \\
    /sdcard/Android/data/com.imulocal/files/imu_csv/replay/latest.csv
"""
from __future__ import annotations

import sys
from pathlib import Path


def convert(ios_dir: Path, out_path: Path) -> int:
    """4개 iOS CSV → Android 단일 replay CSV. 반환: 작성된 이벤트 수."""

    def read_columns(path: Path):
        """헤더 첫 줄 + 데이터 라인들 반환. 컬럼명 → 인덱스 dict 도 같이."""
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        header = lines[0].split(",")
        col_idx = {name: i for i, name in enumerate(header)}
        return col_idx, lines[1:]

    acc_idx,  acc_lines  = read_columns(ios_dir / "Accelerometer.csv")
    grav_idx, grav_lines = read_columns(ios_dir / "Gravity.csv")
    gyr_idx,  gyr_lines  = read_columns(ios_dir / "Gyroscope.csv")
    ori_idx,  ori_lines  = read_columns(ios_dir / "Orientation.csv")

    if not (len(acc_lines) == len(grav_lines) == len(gyr_lines) == len(ori_lines)):
        raise SystemExit(
            f"row count mismatch: acc={len(acc_lines)} grav={len(grav_lines)} "
            f"gyr={len(gyr_lines)} ori={len(ori_lines)}"
        )

    N = len(acc_lines)
    print(f"[1] iOS 입력: {N} 샘플 (Accelerometer/Gravity/Gyroscope/Orientation 각각)")

    # 컬럼 인덱스 미리 추출
    ax_i, ay_i, az_i = acc_idx["x"], acc_idx["y"], acc_idx["z"]
    gx_i, gy_i, gz_i = grav_idx["x"], grav_idx["y"], grav_idx["z"]
    wx_i, wy_i, wz_i = gyr_idx["x"], gyr_idx["y"], gyr_idx["z"]
    ts_i = acc_idx["time"]
    ori_qx, ori_qy, ori_qz, ori_qw = (
        ori_idx["qx"], ori_idx["qy"], ori_idx["qz"], ori_idx["qw"],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_events = 0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("sensor,ts_ns,x,y,z,w\n")
        for i in range(N):
            ap = acc_lines[i].split(",")
            gp = grav_lines[i].split(",")
            wp = gyr_lines[i].split(",")
            op = ori_lines[i].split(",")

            ts = int(ap[ts_i])
            ax, ay, az = float(ap[ax_i]), float(ap[ay_i]), float(ap[az_i])
            gx, gy, gz = float(gp[gx_i]), float(gp[gy_i]), float(gp[gz_i])
            wx, wy, wz = float(wp[wx_i]), float(wp[wy_i]), float(wp[wz_i])

            # Android raw acc = iOS user accel - iOS gravity (gravity 부호 반전)
            raw_ax = ax - gx
            raw_ay = ay - gy
            raw_az = az - gz

            qx = float(op[ori_qx])
            qy = float(op[ori_qy])
            qz = float(op[ori_qz])
            qw = float(op[ori_qw])

            # 4 이벤트 동일 ts 로 출력 (Android replay 가 시간 순으로 처리)
            f.write(f"acc,{ts},{raw_ax:.6f},{raw_ay:.6f},{raw_az:.6f},0.0\n")
            f.write(f"gyr,{ts},{wx:.6f},{wy:.6f},{wz:.6f},0.0\n")
            f.write(f"linAcc,{ts},{ax:.6f},{ay:.6f},{az:.6f},0.0\n")
            f.write(f"rotVec,{ts},{qx:.6f},{qy:.6f},{qz:.6f},{qw:.6f}\n")
            n_events += 4

    duration_sec = (int(acc_lines[-1].split(",")[ts_i])
                    - int(acc_lines[0].split(",")[ts_i])) / 1e9
    print(f"[2] 변환 완료: {n_events} 이벤트, 기록 길이 {duration_sec:.1f} s")
    print(f"[3] 출력: {out_path}  ({out_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print()
    print("단말 push 명령 (예):")
    print(f'  D:/SDK/platform-tools/adb.exe -s adb-R5CWC2B9J3D-ZQ8Ju0._adb-tls-connect._tcp \\')
    print(f'    push "{out_path}" \\')
    print(f"    /sdcard/Android/data/com.imulocal/files/imu_csv/replay/latest.csv")
    return n_events


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        print(f"\n사용: python {sys.argv[0]} <ios_dir> <output_csv>")
        return 1
    ios_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    if not ios_dir.is_dir():
        print(f"iOS 폴더가 디렉토리가 아님: {ios_dir}")
        return 1
    convert(ios_dir, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
