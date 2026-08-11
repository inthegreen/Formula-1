# 외부 참고자료 정리

## 1. Trelis Research — "Train an ACT Policy for an SO-101 Robot using LeRobot"
https://trelis.substack.com/p/train-an-act-policy-for-an-so-101

### 참고한 설정
- Chunk size: 50 steps (~1.66초 앞을 예측)
- Action steps: 15 (예측한 것 중 처음 0.5초 분량만 실제 실행, 나머지는 버리고 재예측)
  → 저희는 지금까지 `chunk_size == n_action_steps`로만 실험(예측한 걸 전부 실행)했는데, "더 길게 보되 짧게 실행"하는 이 방식은 최신 관측을 더 자주 반영할 수 있어 그립처럼 타이밍이 중요한 동작에 유리할 수 있음 → v6 실험에 반영
- Batch size: 소규모 데이터셋은 4, 대규모는 최대 32 (저희 64보다 작음, 참고만)

### KL weight 관련 (중요)
> "Small datasets may not benefit from style encoding" / "Style encoding may be unnecessary for simple, consistent demonstrations"

→ 저희 kl_weight=1 실험에서도 posterior collapse가 재발한 것에 대한 설명이 될 수 있음. kl_weight 크기보다 데이터 규모/복잡도 자체가 원인일 가능성 시사.

---

## 2. Sherry Chen — "How I Trained ACT on SO-101: My Journey, Gotchas, and Lessons"
https://huggingface.co/blog/sherryxychen/train-act-on-so-101

저희 프로젝트와 겹치는 문제가 많아 특히 유용했던 사례.

### 겹치는 문제 & 확인된 원인
| 문제 | Sherry Chen 사례 | 저희 프로젝트와의 관련성 |
|---|---|---|
| 캘리브레이션 사고 | 전원 사이클 중 캘리브레이션 파일 유실, 재캘리브레이션 시 모든 관절을 중간범위로 안 가져가서 homing offset이 틀어짐 | 저희가 겪은 캘리브레이션 재설정 사고와 동일한 실패 메커니즘 |
| 블록 위쪽을 집으려는 문제 | 데이터셋을 검토해보니 본인이 블록 위쪽 가까이 잡은 시연이 많았음 (원인: 데이터 품질) | 저희 그립 헛짚음의 원인 가설(3인 그립 스타일 편차)과 같은 계열의 문제 |
| 실패 후 회복 불가 | 그립 실패 시 재시도는 하지만 전혀 회복 못 함 | 저희의 "완료 블록 재시도/헛짚음 반복" 패턴과 유사 |
| 고정 위치 과적합 | 5곳×10개 고정 위치로만 학습 시 궤적을 암기, 새 위치 일반화 실패 | 저희 fixed vs random 결과(고정은 잘하나 랜덤은 약함)와 동일 |
| "랜덤" 샘플링의 함정 | 균등하게 랜덤이라 생각해도 실제로는 위치가 뭉쳐있는 경우가 많음(시각화로 확인) | 저희 random 태그 데이터도 실제 분포 확인이 필요할 수 있음 |

### 데이터 양·다양성에 따른 성능 변화 (참고 수치)
| 시도 | 데이터 | In-distribution | Out-of-distribution |
|---|---|---|---|
| Try 1 | 50 (5곳×10) | 사실상 실패 | - |
| Try 2 | 50 (구조화된 6 bin) | 60% | 10% |
| Try 3 | 150 (25/bin) + 회전 다양성 | **90%** | **75%** |

→ 150개 규모 + 위치뿐 아니라 회전 다양성까지 추가하면 극적 개선 가능. 저희 목표(270개)가 합리적인 방향임을 뒷받침. **회전(yaw) perturbation은 저희가 아직 시도 안 한 부분으로 향후 고려 가능.**

### 평가 방법론 참고
- 재현 가능한 고정 eval episode(블록/컨테이너 위치를 고정 config로 관리) → 저희가 계획한 "고정 배치 세트, N≥10" 프로토콜과 동일한 아이디어
- 이분법적 성공/실패 대신 **단계별 progress score** 사용: 블록 접근(0.2) → 그립(0.4) → 목적지 접근(0.7) → 릴리즈(0.8) → 완료(1.0)
  → 향후 평가 시 도입 검토 가능. 실패 지점을 더 정교하게 진단 가능

### 하드웨어 유의사항
- teleop 중 블록을 놓칠까봐 너무 세게 그립하다가 그리퍼 모터가 마모되어 고장난 사례 있음
- `max_relative_target` 설정으로 관절 속도를 제한하는 것도 예방책으로 언급됨
- 여분 서보모터 확보 권장
