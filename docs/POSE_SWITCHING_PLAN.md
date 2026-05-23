# 자세별 EKF/PDR 분기 구상안

> 2026-05-23 작성. 본 문서는 *구상 단계* 다 — 실제 구현은 추후 단계별 결정.
> 관련: `docs/PATH_B_RATIONALE.md`, `memory/project_offline_harness.md`.

---

## 0. 배경 — Notion 2026-05-11 의 결론

Python 트랙 (트랙 A) 의 `src/View/ekf_tune.py` 그리드서치 결과(Notion
2026-05-11):

| state | 클래스 | Winner |
|---|---|---|
| 1 | handbag | NET (PDR) |
| 2 | handheld | NET (PDR) |
| 3 | pocket | NET (PDR) |
| 4 | running | NET (PDR) |
| 5 | slow | NET (PDR) |
| **6** | **trolley** | **EKF** ✓ (유일) |
| 7 | multi_devices | NET |
| 8 | multi_users | NET |
| 9 | large_scale | NET |

핵심: **trolley 만 EKF, 나머지 8 자세는 PDR (dead-reckoning)**.

직관적 이유:
- trolley = 기계적으로 안정한 손잡이 + 일정한 진행 방향 → EKF 의 IMU 적분
  모델이 잘 동작. measurement (모델 disp) 신뢰도 가장 높음 (GT-Meas
  0.37 m).
- 나머지 자세 = 보행 시 손/주머니/가방의 진자/스윙 운동 → IMU 적분으로 잡기
  어려움. PDR (모델 disp 누적) 이 더 우수.

본 단말에서도 같은 분기를 적용할 수 있는가? → 본 문서가 그 구상.

---

## 1. 핵심 도전 — 분류기 OoD 문제

본 단말의 휴대모드 분류기가 OoD 라는 점이 이미 진단됨:

- 학습 = OxIOD 7-class (handbag/handheld/pocket/running/slow_walk/trolley/unknown)
- Android 단말 = 분류 결과 대부분 *unknown* (도메인 갭)
- 결과: 분류 확률이 trolley 인지 다른 자세인지 신뢰 못 함

→ Notion 의 "trolley 만 EKF" 결론을 그대로 단말에 적용하려면 *분류기를 신뢰
가능한 수준* 으로 만들거나, *분류기 우회의 trolley 감지 방법* 이 필요.

---

## 2. 자세 자동 판별 — 후보 방법

### (A) 학습 분류기 그대로 사용 (현재)
- 출력: `result.topClass`, `result.clsProb`
- 단말 결과: 대부분 unknown, trolley 확률 거의 0
- 장점: 추가 구현 0
- 단점: 신뢰도 사실상 0 → 자세 분기 못 함
- 판정: **단독 사용 불가**

### (B) IMU 통계 기반 휴리스틱 trolley 감지
trolley 자세의 IMU 특징:
- 보행 진자 운동 *없음* (가속도 진폭 작음)
- 일정한 진행 방향 (rotVec yaw 변화 작음, 좌우 swing 없음)
- 진동 패턴 = 바퀴 회전 노이즈 (고주파, 작은 진폭)

휴리스틱:
```
가속도 std (1초 window) < TROLLEY_ACC_STD_THRESH     # 진자 운동 없음
AND  rotVec yaw 변화 std < TROLLEY_YAW_STD_THRESH    # 일정 진행 방향
AND  속도 > TROLLEY_MIN_SPEED                         # 정지 아님
⇒ trolley 자세로 판정
```

- 장점: 분류기 우회, 단순한 통계
- 단점: 임계값 튜닝 필요. handheld + 일정한 도보 시 오판 가능
- 판정: **별도 측정 데이터로 임계 결정 후 시도 가능**

### (C) 보행 step detection 부재로 trolley 추정
보행 자세는 발걸음마다 가속도 peak (1~2 Hz 보행 케이던스). trolley 는 발걸음 peak 없음.

```
가속도 magnitude 의 1~2 Hz 대역 power < THRESH  ⇒  trolley
```

- 장점: 명확한 물리적 특징
- 단점: FFT 또는 band-pass filter 필요 (계산 비용)
- 판정: **(B) 와 결합 가능**

### (D) UI 자세 명시 선택 (사용자 입력)
시작 화면에서 사용자가 "걸어다님 / 카트 / 가방" 등 선택 후 측위.

- 장점: 자동 판별 오류 0
- 단점: 데모 흐름 끊김, 자세 변경 시 재선택 필요
- 판정: **MVP 단계에서 가장 안전** — 자동 판별 충분히 검증 전엔 권장

### (E) 분류기 fine-tuning (모델 측 작업)
RoNIN 트랙 실패 경험으로 *해당 단말 데이터* 없이는 모델 수정 효과 보장 안 됨.

- 장점: 본질적 해결
- 단점: Samsung 자체 데이터 수집 + 라벨링 필요 (작업량 큼)
- 판정: **장기 트랙** — 단기 데모용 아님

---

## 3. 단계적 구현 로드맵

### Phase 0: 데모는 PATH_B (HANDHELD only) 그대로
- 현재 상태 유지 (P57/P58)
- 자세 분기 미적용
- 데모 시연 자세 = HANDHELD 한정 (README §8.1)
- ⇒ 다음 작업 진행 동안 데모 안정성 확보

### Phase 1: trolley 자동 감지 휴리스틱 측정
- 사용자가 trolley 자세로 보행하며 ImuTestActivity 로 IMU CSV 기록
- Python 도구로 acc/yaw std + 1~2 Hz band power 분석 → 임계값 결정
- *측정만* — 코드 변경 없음
- ⇒ trolley 자세의 IMU 특징 정량화

### Phase 2: 단말에 trolley 감지 + EKF 분기 추가
- `LocalizationViewModel.runRotVecDrStep` 끝에 trolley 감지 휴리스틱 추가
- trolley 로 판정된 윈도우 = EKF 경로 (현재 EKF_CURRENT 와 같은 흐름) 로
  분기, 나머지는 PATH_B 그대로
- 새 모드: `EkfMode.AUTO_POSE_SWITCH` 추가 (PATH_B 와 별도, 기본 비활성)
- 메뉴에서 선택 시 활성
- 단말 ZUPT/freeze 헬퍼는 EKF 분기에서 활용
- ⇒ Notion 2026-05-11 결론을 단말에 부분 적용

### Phase 3: 자세별 PDR 변형
trolley 외 자세도 자세별로 다른 PDR 처리 가능:

| 자세 | heading 처리 | 크기 처리 |
|---|---|---|
| HANDHELD | rotVec heading 그대로 | × 1.5 |
| HANDBAG | rotVec heading 의 저주파 성분만 (진자 운동 제거) | × 1.3 ? |
| POCKET | rotVec heading 의 평균 보행 방향 (다리 흔들림 제거) | × 1.2 ? |
| RUNNING | rotVec heading 그대로 | × 2.0 ? |
| TROLLEY | rotVec heading 그대로 (or EKF 경로) | × 1.0 (EKF) |

- 자세 자동 감지 + 자세별 처리 분기
- 각 자세 보정 계수는 *해당 자세 측정 데이터* 필요

### Phase 4: 분류기 fine-tuning + 자세별 모델
- Samsung 단말 자체 데이터 수집 (각 자세별 ~10-30 분)
- 분류기 재학습 → unknown 비율 감소
- 회귀기도 데이터 확장 가능 → 모델 방향 채널 OoD 해소 시 EKF 경로 활용성 ↑
- (장기 트랙)

---

## 4. 가장 빠른 다음 단계 (Phase 1 추천)

### 작업
1. ImuTestActivity 로 trolley 자세 보행 CSV 기록 (5~10 분)
2. 같은 동선을 HANDHELD 자세로 기록
3. Python 도구 `src/View/analyze_pose_features.py` (신설) 로 비교 분석:
   - 가속도 std (1초 window)
   - rotVec yaw 변화 std
   - 가속도 magnitude 의 1~2 Hz band power
   - 두 자세에서 각 통계의 분포 + 임계값 후보
4. 임계값 표 도출 → 다음 Phase 2 구현에 활용

### 결과
- 임계값이 명확히 분리되면 → Phase 2 진행 (단말에 휴리스틱 + EKF 분기 추가)
- 분리가 불명확하면 → (D) UI 명시 선택 또는 (E) fine-tuning 으로 우회

---

## 5. 학술적 기여 측면

### 단순 단말 데모 → 학술 contribution 으로 격상 가능

자세별 분기 구현이 성공하면:
- "도메인 갭 환경에서 자세별 EKF/PDR 자동 분기로 신뢰도 향상" 이 **본 프로젝트의 단말 contribution**
- thesis 의 Context-Aware Adaptive EKF 정신 (자세별 처리) 의 *단말 실측 적용*
- TLIO 의 baseline (PATH_B = 3D-RONIN) 과 단말 적용 EKF 의 자세별 hybrid → 단말 환경에서 의미 있는 차별성

### 부정적 결과도 가치
- trolley 자동 감지 휴리스틱이 실패하면 → "단말 도메인 갭 + 자세 OoD 의 *복합* 한계" 정량 진단
- 후속 연구의 데이터 수집 전제 (각 자세별 단말 데이터 필요량) 정량화

---

## 6. 위험 / 주의

### 위험 1: trolley 감지 휴리스틱 임계 튜닝 불안정
- 자세 측정 데이터가 적으면 임계 신뢰 못 함
- 사용자 → 보조 자세 (예: 양손에 든 가방) 같은 중간 케이스에서 오판
- **완화**: Phase 1 측정 시 다양한 보행 속도 + 회전 패턴 포함

### 위험 2: EKF 분기에서 단말 EKF 발산 재현
- trolley 자세에서 EKF 가 잘 동작한다는 보장이 OxIOD/Python 트랙 기준
- 본 단말의 EKF 가 trolley 자세에서도 발산할 가능성 존재 (단말 도메인 갭이 자세에 무관할 수 있음)
- **완화**: Phase 2 진행 전 trolley 자세 IMU 로 offline_eval.py 진단 먼저
  - 모델 방향 채널이 OxIOD 처럼 정상이면 EKF 분기 의미 ↑
  - 여전히 OoD 면 trolley 분기도 PATH_B 변형으로

### 위험 3: 코드 복잡도 증가
- 자세별 분기 = 코드 경로 다양화
- 데모 안정성 ↓ 가능
- **완화**: Phase 2 는 *별도 ekfMode 모드* 로 격리. PATH_B 기본 데모는 변경 0.

---

## 7. 결정 사항 (현재)

- Phase 0 유지: 데모는 PATH_B (HANDHELD only), 코드 변경 없음
- Phase 1 진행 권장: trolley 자세 측정 데이터 수집 + Python 분석
- Phase 2 이상은 Phase 1 결과 보고 결정

본 문서는 진행 단계에 따라 갱신한다.
