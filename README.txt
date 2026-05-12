python View\visualize_comparison.py --data_path TLIO_Oxford_Dataset\oxford_large_scale_6\imu0_resampled.npy --model_path outputs\out_classifier2\checkpoints\best.pth --norm_mean outputs\out_classifier2\norm_mean.npy --norm_std outputs\out_classifier2\norm_std.npy

python View\visualize_comparison.py --data_path TLIO_Oxford_Dataset\oxford_handheld_21\imu0_resampled.npy --model_path outputs\out_classifier2\checkpoints\best.pth --norm_mean outputs\out_classifier2\norm_mean.npy --norm_std outputs\out_classifier2\norm_std.npy --meascov_scale 0.1 --init_vel_sigma 0.01 --sigma_na 0.2 --sigma_ng 0.01



상태         meascov_scale   sigma_na   sigma_ng   ita_ba   ita_bg
─────────────────────────────────────────────────────────────────
handbag  (1) 크게 ↑         크게 ↑     보통        보통     보통   ← 흔들림 많음
handheld (2) 보통            보통       보통        보통     보통
pocket   (3) 작게 ↓         작게 ↓     작게 ↓      보통     보통   ← 안정적
running  (4) 크게 ↑         매우 크게↑ 크게 ↑      보통     보통   ← 충격 심함
slow-walk(5) 작게 ↓         작게 ↓     작게 ↓      보통     보통   ← 매우 안정적
trolley  (6) 보통            보통       보통        크게 ↑   보통   ← 바퀴 진동


View -> visualize_comparison.py -> 48 줄

STATE_EKF_PARAMS = {
    # state_id: dict(meascov_scale, sigma_na, sigma_ng, ita_ba, ita_bg)
    # None 값은 run_ekf_imutracker 의 전역 파라미터를 그대로 사용함
    -1: dict(meascov_scale=0.001,  sigma_na=None, sigma_ng=None, ita_ba=None, ita_bg=None),  # unknown
    1:  dict(meascov_scale=0.05,   sigma_na=None, sigma_ng=None, ita_ba=None, ita_bg=None),  # handbag
    2:  dict(meascov_scale=0.01,   sigma_na=None, sigma_ng=None, ita_ba=None, ita_bg=None),  # handheld
    3:  dict(meascov_scale=0.005,  sigma_na=None, sigma_ng=None, ita_ba=None, ita_bg=None),  # pocket
    4:  dict(meascov_scale=0.05,   sigma_na=None, sigma_ng=None, ita_ba=None, ita_bg=None),  # running
    5:  dict(meascov_scale=0.005,  sigma_na=None, sigma_ng=None, ita_ba=None, ita_bg=None),  # slow-walking
    6:  dict(meascov_scale=0.02,   sigma_na=None, sigma_ng=None, ita_ba=None, ita_bg=None),  # trolley
}
이 부분 수정하면 됨