# SO-101 졸업과제 — Task1: 블록 정렬

## 개요
- **로봇**: SO-101 (leader-follower), 제어 기기: Jetson Orin Nano
- **Task1**: 블록 5개를 지정구역(20x10cm, 빨간 테이프)으로 이동
  - 블록 크기: 4x4x2cm
  - 평가: 3분 제한시간 내 지정구역 블록 개수로 점수 산정, 5개 전체 완료 시 시간 보너스
  - 성공 기준: 선 안에 위치(2cm까지 걸침 허용), 세우기/겹치기 허용 (쌓기만 불허)
- **카메라**: 탑뷰 + 손목뷰 (640x480, 30fps)
- **정책**: ACT (Action Chunking Transformer)

전체 진행 히스토리는 [`docs/changelog_summary.md`](./docs/changelog_summary.md), 모델 버전별 비교는 [`experiments/EXPERIMENTS.md`](./experiments/EXPERIMENTS.md) 참고.

## 파이프라인

```
[Jetson Orin Nano]                [학교 GPU 서버 (RTX 5090)]         [Jetson Orin Nano]
 데이터 수집 (teleop)  --push-->    데이터셋 병합 및 ACT 학습  --push-->  실물 로봇 배포/평가
```

## 현재 데이터셋 수집 현황 
Task 1 (목표 300 + 80 episodes)
| 태그 | 목표 | 완료 | 설명 |
|---|---|---|---|
| fixed | 60 | ✅ 60 | 고정 위치 |
| random | 120 | ✅ 60 | 랜덤 위치 (골고루 분산) |
| left | 45 | ✅ 45 | 지정구역 왼쪽 |
| right | 45 | ✅ 45 | 지정구역 오른쪽 |
| abnormal | 30 | ⬜ 미착수 | 비정상 출발 자세 |
| DAgger | 40 | 🔄 진행중 | 경로 수정 |

Task 2 (목표 400 + 120 episodes)
| 태그 | 목표 | 완료 | 설명 |
|---|---|---|---|
| fixed2 | 60 | ✅ 60 | 고정 위치 |
| random2 | 160 | ⬜ 미착수 | 랜덤 위치 (골고루 분산) |
| left2 | 50 | 🔄 40 | 지정구역 왼쪽 |
| right2 | 50 | ⬜ 미착수 | 지정구역 오른쪽 |
| abnormal2 | 80 | ⬜ 미착수 | 비정상 출발 자세 |
| DAgger2 | 120 | ⬜ 미착수 | 경로 수정 |

## 사용법

### 1. 환경 세팅 (학교 GPU 서버, 최초 1회)
```bash
bash scripts/setup_env.sh
```

### 2. 데이터 수집 (Jetson Orin Nano)
```bash
export HF_USER=s1eepypillow
bash scripts/record.sh
```

### 3. 데이터셋 병합
```bash
export HF_USER=s1eepypillow
bash scripts/merge_dataset.sh
```

### 4. ACT 학습 (학교 GPU 서버)
```bash
tmux new -s train
export HF_USER=s1eepypillow
bash scripts/train.sh
# Ctrl+B, D 로 세션에서 빠져나오기 (백그라운드로 계속 진행됨)
```

### 5. 실물 로봇 배포 및 평가 (Jetson Orin Nano)
```bash
export HF_USER=s1eepypillow
bash scripts/rollout.sh
```

## 문서

- [`docs/changelog_summary.md`](./docs/changelog_summary.md) — 전체 진행 히스토리 요약
- [`experiments/EXPERIMENTS.md`](./experiments/EXPERIMENTS.md) — 모델 버전별 비교표
- [`docs/troubleshooting.md`](./docs/troubleshooting.md) — 지금까지 겪은 에러/해결법 모음
- [`docs/act_architecture_notes.md`](./docs/act_architecture_notes.md) — ACT 내부 구조 분석 (posterior collapse 진단)

## TODO

Task 1
- [ ] DAgger 및 비정상 출발 데이터셋 추가
- [ ] chunk_size=30 / kl_weight=1 실험 결과 비교
- [ ] 최종 데이터셋(~270 episodes)으로 재학습
- [ ] 지정구역-블록 오인 문제 대응 데이터 보강
- [ ] abnormal 시나리오 착수
Task 2
- [ ] 데이터셋 추가 (오른쪽, 랜덤 위치)
- [ ] 학교 평가 서버용 Docker 이미지 빌드 절차 확정
