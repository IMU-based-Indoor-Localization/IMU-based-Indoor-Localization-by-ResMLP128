# EKF Basic — IMU 기반 보행자 궤적 추정

TLIO 구조의 신경망(TwoLayerModel)과 IMU-MSCKF(Sliding-Window EKF)를 결합한 보행자 위치 추정 프로젝트.

---

## 디렉터리 구조

```
src/
├── Network/          # 모델 정의 및 학습 (train.py)
├── Trans/            # Dataset / DataLoader
├── tracker/          # EKF 코어 (scekf.py, imu_buffer.py)
├── View/
│   └── visualize_comparison.py   # 궤적 시각화 메인 스크립트
├── outputs/          # 학습 결과물 (체크포인트, 정규화 통계)
└── cls_split_7way/   # Oxford 데이터셋 (train/val/test 분할)
```

---

## 시각화 사용법 — `visualize_comparison.py`

### 기본 실행

```bash
cd src
python View/visualize_comparison.py \
  --data_path  <시퀀스.npy 경로> \
  --model_path <체크포인트.pth 경로> \
  --norm_mean  <norm_mean.npy 경로> \
  --norm_std   <norm_std.npy 경로>
```

**출력:**
- Figure 1 — 전체 궤적 (XY 궤적 + 시간별 오차)
- 콘솔에 `RMSE_XY Network / EKF` 출력

---

### 옵션 목록

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--window_len` | `100` | 네트워크 입력 윈도우 길이 (샘플 수, 100Hz 기준) |
| `--stride` | `100` | 윈도우 슬라이딩 간격 |
| `--save` | `comparison_<시퀀스명>.png` | Figure 1 저장 경로 |
| `--lap_start` | `None` | Figure 2 구간 시작 (초) |
| `--lap_end` | `None` | Figure 2 구간 끝 (초) |
| `--gt_meas` | `False` | GT 변위를 EKF 측정값으로 사용 (디버그용) |
| `--mahalanobis_factor` | `1.0` | Mahalanobis 게이팅 계수 (`0` = 완전 비활성) |
| `--meascov_scale` | `10.0` | 측정 공분산 스케일 배율 |
| `--sigma_na` | `sqrt(1e-3)` | 가속도 노이즈 표준편차 |
| `--sigma_ng` | `sqrt(1e-4)` | 자이로 노이즈 표준편차 |
| `--ita_ba` | `1e-4` | 가속도 바이어스 랜덤워크 |
| `--ita_bg` | `1e-6` | 자이로 바이어스 랜덤워크 |
| `--init_vel_sigma` | `1.0` | 초기 속도 불확실성 |

---

### Figure 2: 특정 구간(바퀴) 확대 보기

같은 공간을 반복 순회하는 시퀀스에서 특정 바퀴만 따로 확인할 때 사용합니다.

```bash
# 예시: 총 188초 시퀀스에서 2번째 바퀴 (63~126s) 확대
python View/visualize_comparison.py \
  --data_path  cls_split_7way/val/oxford_handheld_3/imu0_resampled.npy \
  --model_path outputs/out_classifier2/checkpoints/best.pth \
  --norm_mean  outputs/out_classifier2/norm_mean.npy \
  --norm_std   outputs/out_classifier2/norm_std.npy \
  --lap_start 63 --lap_end 126 \
  --save result/handheld3.png

# 3번째 바퀴 (126~189s)
  --lap_start 126 --lap_end 189
```

`--lap_start`/`--lap_end`를 지정하면 Figure 1(전체) + Figure 2(구간) 두 창이 함께 열립니다.  
저장 파일은 Figure 1 → `handheld3.png`, Figure 2 → `handheld3_lap63-126.png` 로 자동 분리됩니다.

---

### 디버그: GT 측정값으로 EKF 상한 확인

```bash
python View/visualize_comparison.py \
  --data_path  cls_split_7way/val/oxford_handheld_3/imu0_resampled.npy \
  --model_path outputs/out_classifier2/checkpoints/best.pth \
  --norm_mean  outputs/out_classifier2/norm_mean.npy \
  --norm_std   outputs/out_classifier2/norm_std.npy \
  --gt_meas \
  --mahalanobis_factor 0
```

> 네트워크 대신 GT 변위를 EKF 입력으로 사용하여 propagation/update 자체의 정상 동작을 검증합니다.

---

## 데이터 형식

### Oxford 24컬럼 포맷 (`imu0_resampled.npy`)

| 컬럼 | 내용 |
|------|------|
| 0 | timestamp (µs) |
| 1:4 | gyroscope body (rad/s) |
| 4:7 | linear acceleration body (m/s², 중력 제거) |
| 7:10 | gravity unit vector (body frame) |
| 13 | 동작 레이블 (1~7) |
| 14:18 | quaternion xyzw (R\_BW, world→body) |
| 18:21 | position GT (m) |
| 21:24 | velocity GT (m/s) |

> **주의:** Oxford 쿼터니언은 `R_BW` (world→body)를 저장합니다.  
> `scipy.spatial.transform.Rotation.from_quat(q).as_matrix()` = R\_BW  
> EKF body→world 회전 초기화 시 반드시 `.as_matrix().T` 를 사용해야 합니다.

---

## 모델 학습

```bash
cd src
python Network/train.py --config <config.json>
```

학습 결과는 `outputs/<run_name>/checkpoints/` 에 저장됩니다.
