# 진행 히스토리 (핵심 요약)

## Phase 1 — 초기 구축 (~07-24)
- SO-101 + Jetson Orin Nano 세팅, teleop으로 100 episodes 수집 (15세션, 그립 방식 미통일·3인 분담으로 스타일 편차 존재)
- 병합·정리 후 75 episodes 확정 → 학교 서버(RTX 5090) ACT 1차 학습 (chunk=60, kl_weight=10, 100k steps) → loss 0.042 수렴
- 모델: `task1_act_policy`

## Phase 2 — 첫 실물 테스트 (07-28)
- 환경 이슈 다수 해결 (Jetson↔서버 LeRobot 버전, PyTorch/NumPy 호환 → `docs/troubleshooting.md`)
- 1차 rollout(N=10): 6회 블록 1개 이상 성공, 완전성공 1회 / 문제: 그립 헛짚음, 완료블록 재시도, 구역 위치 바뀌면 급격히 저하
- 캘리브레이션 재설정 중 손목 나사 풀림 사고 → 복구
- 재캘리브레이션 전후 teleop 비교 결과, **캘리브레이션 자체는 원인 아님**으로 결론 (성능 수준 비슷)

## Phase 3 — 데이터 재설계 및 태그 수집 (08-03 ~ 08-07)
- 평가기준 재확인: 배치 자유도 높음(2cm 허용, 겹침/세우기 OK) → **그립 정밀도가 핵심 병목**
- 그립 방식 통일(완전 하강 후 닫기), 태그 기반 수집 체계 도입 (목표 300ep: fixed 60 / random 120 / left 45 / right 45 / abnormal 30)
- fixed+random75(135ep) 학습 → 그립/인식 개선, 10회 중 1회 완전성공, 랜덤 위치는 약함(과적합 추정)
- +left45(180ep) 학습 → 왼쪽 편향 발생 (right 비중 부족 추정)

## Phase 4 — ACT 내부 분석 및 하이퍼파라미터 실험 (08-07~)
- `modeling_act.py` 분석 → **posterior collapse 확인** (`kl_weight=10`이 과도, latent가 그립 스타일 정보를 못 담음) → `docs/act_architecture_notes.md`
- 대응 실험 병렬 진행: `chunk_size 60→30`, `kl_weight 10→1`
- klw1 초기 관찰: kld_loss가 0이 아닌 값(0.206) 확인 — collapse 완화 가능성, 검증 중 (batch_size 16으로 축소 실행되어 재검증 필요)

## 다음 계획
- [ ] random/right 데이터 목표치 채우기
- [ ] chunk30 / klw1 결과 비교 후 최종 데이터셋(~270ep) 재학습
- [ ] 지정구역-블록 오인 문제 대응 데이터 보강
- [ ] abnormal 시나리오, Task2 착수
- [ ] 평가 서버 Docker 이미지화 절차 확정
