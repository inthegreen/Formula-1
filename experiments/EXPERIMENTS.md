# 실험/모델 버전 비교표

| ID | 날짜 | 데이터셋 (episodes) | 주요 변경사항 | Hyperparameters | 결과 요약 | 모델 repo |
|---|---|---|---|---|---|---|
| v1 | 07-24 | grad_block_merged (75) | 최초 baseline | chunk=60, kl_weight=10, batch=64 | loss 0.042 수렴. 실물: 그립 헛짚음 잦음, 성공률 낮음 | `s1eepypillow/task1_act_policy` |
| v2 | 08-05 | fixed(60)+random(75) = 135 | 그립 방식 통일(완전 하강 후 닫기)로 데이터 재수집 | chunk=60, kl_weight=10, batch=64 | 그립/인식 개선. 10회 중 1회 완전성공, 9회는 부분성공. 랜덤 위치 약함(과적합 추정) | `s1eepypillow/task1_act_policy_fixed` |
| v3 | 08-05 | +left(45) = 180 | 지정구역 왼쪽 위치 데이터 추가 | chunk=60, kl_weight=10, batch=64 | v2와 비슷하나 "왼쪽 편향" 발생 (블록 찾을 때 왼쪽으로만 이동하려는 습관) | `s1eepypillow/task1_act_policy_fixedleft` |
| v4 | 08-07 | 180 (v3와 동일) | chunk_size 60→30 실험 | chunk=30, kl_weight=10, batch=64 | 진행 중. 학습 속도 대폭 향상(2.95→11.48 step/s) | `s1eepypillow/task1_act_policy_chunk30` (예정) |
| v5 | 08-07 | 180 (v3와 동일) | kl_weight 10→1 실험 (posterior collapse 대응) | chunk=60, kl_weight=1, batch=16 (GPU 병렬 실행으로 축소) | **완료.** posterior collapse 재발 확인 (100k step 종료 시 `mu_hat`≈0.001, kl_weight=10 때보다 오히려 더 붕괴). kl_weight 조정만으로는 해결 안 됨 → `docs/act_architecture_notes.md` 참고 | `s1eepypillow/task1_act_policy_klw1` |
| v6 | 08-11 | 180 (v3와 동일) | chunk_size=50, n_action_steps=15 (Trelis Research 자료 기반, 길게 예측·짧게 실행) | chunk=50, n_action_steps=15, kl_weight=10(기본), batch=64, wandb 연동 | 진행 중 (W&B: grad-so101-task1 프로젝트) | `s1eepypillow/task1_act_policy_chunk50_step15` (예정) |
| v7 | (예정) | fixed+random+left+right (~270) | 최종 데이터 규모 확보 + 유효했던 설정 반영 | TBD | - | - |

## Posterior Collapse 관련 결론 업데이트 (08-11)

kl_weight를 10→1로 낮춰도 collapse가 재발한 것에 대해, 외부 자료(Trelis Research) 조사 결과 **"작은/단순한 데이터셋에서는 ACT의 스타일 인코딩(latent z)이 애초에 큰 효과를 내지 못할 수 있다"**는 내용을 확인함. 즉 collapse가 kl_weight 크기보다는 데이터 규모·다양성 부족에 기인할 가능성이 높음. 우선순위를 kl_weight 추가 튜닝에서 **데이터 규모 확대(270개 목표)** 및 **chunk_size/n_action_steps 조합 최적화**로 전환.

## 비교 시 참고사항

- v1→v2 개선은 **데이터 품질(그립 방식 통일)** 효과, v3의 편향은 **데이터 비중 불균형** 문제로 별개 이슈
- v4, v5는 **서로 다른 변수를 독립적으로 검증**하기 위해 분리 실행 (같은 학습에 섞지 않음)
- v5는 GPU 메모리 제약으로 batch_size가 기준(64)과 다름(16) → kl_weight 효과와 batch_size 효과가 섞여있을 수 있어, 최종 결론 전에 batch=64로 단독 재검증 권장

## 평가 프로토콜 (버전 간 공정 비교를 위해 고정)

- 동일한 블록 배치 3~5세트를 사전에 고정
- 각 버전마다 N≥10 반복 테스트
- 기록 항목: 그립 성공/실패(헛짚음 여부), 지정구역 안착 개수, 소요 시간, 관찰된 실패 패턴
