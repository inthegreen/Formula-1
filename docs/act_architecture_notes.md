# ACT 모델 구조 분석 — Posterior Collapse 진단 (2026-08-07)

## 배경

Fixed+Random+Left(180 episodes) 학습 모델 테스트 결과, 그립 타이밍이 여전히 불안정(블록 아래 헛짚음 / 블록 위쪽에서 성급하게 닫으려는 현상)했음. 이 문제가 "데이터 부족" 때문인지 "모델 학습 방식 자체의 문제" 때문인지 구분하기 위해 ACT 소스코드(`src/lerobot/policies/act/modeling_act.py`)를 직접 분석함.

## ACT의 기본 구조 요약

- 입력: 카메라 이미지(top, wrist) + 로봇 관절 상태(observation.state)
- 출력: 향후 `chunk_size` 스텝만큼의 액션 시퀀스
- CVAE(Conditional VAE) 구조 사용:
  - **학습 시**: 인코더(`vae_encoder`)가 실제 시연 액션 시퀀스를 입력받아, latent 분포의 파라미터 `mu_hat`(평균), `log_sigma_x2_hat`(로그분산)을 산출. 이 분포에서 샘플링한 z를 디코더에 전달
  - **추론 시**: 정답 액션이 없으므로 인코더가 작동하지 않고, z는 고정값(보통 0)을 사용

## Loss 구성

```python
actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)
l1_loss = L1(실제 액션, 예측 액션)
mean_kld = (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp())).sum(-1).mean()
loss = l1_loss + mean_kld * kl_weight
```

- `mean_kld`: latent 분포가 표준정규분포 N(0,1)에서 얼마나 벗어났는지 재는 KL divergence
- `kl_weight`: 이 규제를 얼마나 강하게 걸지 결정하는 계수 (기존 설정: 10)

## 실험 1 (실패): 잘못된 진단 방법

- 방법: 학습된 모델을 `eval()` 모드로 로드, 동일 입력에 대해 `select_action()`을 반복 호출하여 latent 샘플링에 따른 예측 분산 확인
- 결과: 분산 = 0.0000
- 문제: 추론(eval) 모드에서는 VAE 인코더 자체가 작동하지 않고 z가 고정값이므로, 애초에 분산이 나올 수 없는 조건에서 실험한 것. 실험 설계 오류.
- 교훈: posterior collapse 진단은 **학습(training) 모드에서 인코더가 실제로 산출하는 `mu_hat`, `log_sigma_x2_hat` 값**을 직접 봐야 함

## 실험 2: 올바른 진단 (디버그 로그 삽입)

`modeling_act.py`에 아래 디버그 코드를 삽입하여 학습 중 값 변화를 직접 관찰:

```python
actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)
if mu_hat is not None:
    print(f"[DEBUG] mu_hat abs mean: {mu_hat.abs().mean().item():.6f} | log_sigma_x2_hat mean: {log_sigma_x2_hat.mean().item():.6f}")
```

원본 파일은 `modeling_act.py.backup`으로 백업 후 진행. (수기 편집 시 tab/space 혼용 에러 발생 → `sed`로 재삽입, 상세는 `troubleshooting.md` 참고)

### 결과: kl_weight=10 (기존 설정), 50 steps

| Step | mu_hat abs mean | log_sigma_x2_hat mean |
|---|---|---|
| 1 | 0.523 | -0.061 |
| 10 | 0.168 | -0.054 |
| 25 | 0.119 | -0.029 |
| 50 | 0.097 | -0.004 |

→ 두 값 모두 학습 진행에 따라 지속적으로 0에 수렴. `log_sigma_x2_hat → 0`은 분산이 `exp(0)=1`, 즉 표준정규분포와 동일해진다는 뜻.

**결론: 인코더가 입력(실제 시연 스타일)과 무관하게 항상 표준정규분포를 출력하도록 붕괴(posterior collapse) 확인.** 기존 100k step 학습 로그에서 계속 `kld_loss: 0.000`이었던 현상과 정합됨.

### 결과: kl_weight=1, 초반 관찰 (진행 중 실험, batch=16)

| Step | kld_loss |
|---|---|
| 600 | 0.206 |

→ kl_weight=10 실험과 달리 초반부터 kld_loss가 0이 아닌 값을 보임. Collapse가 완화되고 있을 가능성 있는 긍정적 신호. 다만 학습 초반(전체의 0.6%)이라 추가 관찰 필요.

## 해석 / 가설

Posterior collapse가 일어나면, 디코더는 latent z로부터 아무 정보도 받지 못한 채 순수하게 이미지+관절상태만으로 액션을 예측해야 함. 3명이 나눠 촬영하며 그립 타이밍/스타일이 조금씩 달랐던 학습 데이터의 특성상, 이 경우 디코더는 여러 스타일을 뭉뚱그린 "평균적이고 애매한 그립 타이밍"을 예측하는 쪽으로 수렴할 수 있음. 이것이 실물 테스트에서 관찰된 그립 헛짚음(너무 낮게/너무 높게)의 원인 중 하나일 가능성.

## 대응 실험

1. **chunk_size 60→30**: posterior collapse와는 독립적인 문제(정밀 동작 반응성)를 다루는 실험. kl_weight=10 상태에서 chunk_size만 변경한 결과도 mu_hat, log_sigma_x2_hat이 여전히 0으로 수렴 → chunk_size 조정만으로는 collapse가 해결되지 않음을 재확인 (두 문제가 독립적임을 시사).
2. **kl_weight 10→1**: collapse 직접 대응. 결과 진행 중.

## 주의사항 / 한계

- kl_weight=1 실험은 GPU 병렬 실행 제약으로 batch_size=16 (기준 64 대비 축소)으로 진행됨. 최종 결론 전에 batch_size=64 단독 재검증 필요.
- posterior collapse가 실제로 그립 실패의 "주 원인"인지, 여러 원인 중 하나인지는 아직 인과관계까지 확정된 것은 아님 (상관관계 수준의 강한 정황 증거). 최종 판단은 실물 rollout 테스트 결과로 검증 예정.
