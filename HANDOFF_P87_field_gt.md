# HANDOFF — IMU 실내측위 논문 · 현장 절대 GT (P84–P87) · 2026-06-08

다른 세션에서 **논의를 이어가기 위한** 핸드오프. 콜드 스타트로 읽어도 되도록 정리.

## 0. 프로젝트 컨텍스트
- 스마트폰 IMU 단독 실내측위. 모델 **1D-ResMLP128**(변위 μ + 공분산 + 분류 헤드), **OxIOD(iPhone)** 학습, **Android(Galaxy S23 FE)** 배포.
- 배포 경로 **PATH_B = RotVec(자력계 융합 절대 yaw) dead-reckoning + PDR-hybrid**. (EKF 경로 A는 도메인갭으로 발산해 미사용)
- **논문 thesis**: 측위 품질을 좌우하는 건 EKF/필터 튜닝이 아니라 **입력 전처리(중력정렬 + 절대 yaw 안정성)**. yaw-drift는 변위 *방향* 예측을 선택적으로 붕괴(7.7×)시키되 *크기*는 보존(1.0×); OOD 임계 ≈ 누적 yaw 10°.
- 앱: `D:\mobile\imu_android\android` (pkg `com.imulocal`). 분석: `D:\mobile\imu_android\src\Network\`.

## 1. 논문 (Notion)
- 06-03본: `373760ab-b925-81dc-9f63-fa040fb775d3` (§4.8 추가 완료)
- **06-04본(현재 작업본)**: `375760abb9258029afe1c525d54442bd` (§4.8 동일 정렬 완료)
- §4.6 전처리 ablation + 합성 yaw-drift(OxIOD) / §4.7 논의(선행연구 차별·한계) / **§4.8 실측 Android 검증** = 4.8.1 in-domain ATE(~1% drift), 4.8.2 전처리 정렬 재현(body 붕괴·ga≈window-start), 4.8.3 yaw-drift 재현+RotVec 정당화, 4.8.4 종합·한계(절대 GT는 future work).
- **규칙(엄수)**: Notion 반영·git **push 전 사용자 확인 필수**(doc 이미지 추가만 예외). 로컬 **커밋은 진행 OK**.

## 2. 앱 추가 기능
- **P84** FloorPlanView: 평면도 오버레이 + 2점 보정 + 체크포인트(시각화). 메뉴 토글.
- **P85** 단위보정(A) 런타임 토글 `InferenceEngine.USE_OOD_FIX`(@Volatile var) — ON: linAcc÷9.81(g단위 매칭) + (ViewModel) 적응스케일 1.0×. **현장측정은 반드시 A-ON.** (OFF면 P67 적응스케일 2.5~7×로 과대)
- **P87** GT 마크 버튼(볼륨키 eyes-free + 화면) → `marks_*.csv`(idx,est_x,est_y,t_ms). **마크=모델 추정위치(est)**, GT(실좌표)는 평면도에서 별개로 와야 함.
- 빌드: `JAVA_HOME=D:\Android\jbr`, gradle 8.7 캐시(`~/.gradle/wrapper/dists/gradle-8.7-bin/.../bin/gradle.bat`), `:app:assembleDebug`. APK 설치됨(SM-S711N, 무선 adb `D:\SDK\platform-tools\adb.exe`).

## 3. 현장 절대 GT 측정 (4 walk · 종료)
- 데이터: `csv/walks/` (track_PATH_B_* + marks_*). 모두 **A-ON·handheld·단일 피험자·door-to-door**.
- 매핑(inspect 라벨): **A=경로4**(321-1→323~324→324→327-1→321-1 루프, 마크5), **B=경로3**(311-2→309→311-2 왕복; **제외=heading-flip**), **C=경로2**(311-2→309→후문), **D=경로1**(311-2→309).
- 평면도: **미래관 3층**. 격자(치수선 검증): col5=0, col간 11.2/5.6m 교대, col1→5=38.75m, 총 139.55m ✓. X(m): col6=5.6·7=16.8·8=22.4·9=33.6·10=39.2·11=50.4·12=56.0·13=67.2. **후문은 309 근처**(top-row 338 아님). 단 베이 내 소형방 칸막이 치수 없음 → 문 중심 **±2m**.
- 결과(door GT): **경로2 ATE≈2.6m**(후문 0.6m 정확·311-2→309 잔차 3.4m) / **경로1 ~44% 과소** / **경로4 루프복귀 14.1m·62m≈23%**, 구간 ATE≈6.2m(불확실: 323~324·324 모호, 327 추정) / 경로3 제외.
- **해석**: 전처리(프레임/방향)는 실측 정확, **잔여 오차 = 학습-배포 스케일 갭(~40% 과소) + 장거리·다중턴 누적**. = 논문 frame-vs-magnitude 분리 재확인.
- 도구: `android_gt_ate.py`(SE2 Procrustes ATE; self-test 통과), `android_walk_inspect.py`(track+marks), `android_preproc_compare.py`, `android_yaw_drift.py`, `oxiod_model_ate.py`. 추정 wp: `csv/walks/wp_C2.csv`,`wp_A.csv`.

## 4. 스코프 결정 (합의)
- **논문은 메커니즘 검증에서 정지.** fine-tune·스케일 재보정·대규모 절대GT·ARCore = **논문 밖 future work**(정확도/도메인적응·데이터셋용).
- 현장 예비결과는 **본문 미반영(메모만)**. §4.8.4의 (ii)루프복귀·(iii)heading-flip 사례는 **추가측정 후 정식 재작성**.

## 5. 오픈 항목 / 향후
- 정밀 GT: ① 소형방 문 **줄자 측량**(±2m 제거, 기존 마크버튼 재사용) ② **ARCore VIO 연속 pose**(cm~dm, 카메라 가시 자세만) + **N 확대(다보행·다자세·다기기)**.
- 경로4 정식 ATE: **327(-1)·"323~324" 정확 좌표** 필요.
- fine-tune: RoNIN 트랙(`D:\mobile\ronin_finetune`) = 정확도/도메인 적응.
- 데이터셋화: "Android 실내 절대GT 벤치(교차기기 도메인적응)" = ARCore GT + 규모 확대 시 **독립 기여 가능**.

## 6. 핵심 수치 (참조)
- in-domain(OxIOD) ATE ~1% drift, DR경로≈GT경로(스케일 정확) → Android 스케일 문제는 **도메인시프트**(모델 아님).
- Android 단위갭 **9.8×**(OxIOD g vs Android m/s²) → A-ON(÷9.81)로 보정, 잔여 ~30-40% 과소.
- 이 기기 자이로-only yaw drift **72-113°/20-33s**(자력계 없으면) → RotVec 절대 yaw 채택 정당화.
- 경로4 루프 23%, 경로2 ~2.6m.

## 7. 메모리/커밋
- 메모리: `memory/project_field_gt.md`(이 캠페인 상세) + `MEMORY.md` 인덱스. RoNIN: `project_ronin_finetune.md`.
- 최근 커밋(android 브랜치, **미push**): P84+P85, Android 분석 스크립트, P86(OxIOD ATE), P87(마크버튼)/P87b(ATE분석)/P87c(프로토콜PDF)/P87d(점검도구).

## 8. 미해결 논점 (이어서 논의)
- 경로D(311-2→309)가 ~44% 과소인 이유: 스케일 갭 + **느린 보행(~0.84 m/s 실측 vs 0.47 추정)**의 출력 압축(saturation) + 짧은 보행의 시작 transient + GT 추정 ±2m(실제 과소는 ~30-40%일 수). → 정밀 GT·속도 통제로 분리 필요.
- §4.8에 현장 예비결과를 넣을지(현재 b안=미반영), 정식화 시점.
