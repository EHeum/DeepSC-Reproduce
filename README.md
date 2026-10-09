# DeepSC — Deep Learning Enabled Semantic Communication (PyTorch 재구현)

> H. Xie, Z. Qin, G. Y. Li, B.-H. Juang, **"Deep Learning Enabled Semantic Communication Systems,"**
> *IEEE Trans. Signal Processing*, vol. 69, pp. 2663–2675, 2021.

이 저장소는 위 논문의 시스템 모델을 분석하고, 핵심 기여인 **DeepSC**(Transformer 기반 joint
semantic-channel coding 송수신기)를 Python/PyTorch로 처음부터 구현한 뒤 논문의 주요 실험
(Fig. 6, 8, 9, 10, 13, Table II, III)을 축소된 규모로 재현·검증한 결과를 정리한 것입니다.

[![CI](https://github.com/EHeum/DeepSC-PyTorch/actions/workflows/ci.yml/badge.svg)](https://github.com/EHeum/DeepSC-PyTorch/actions/workflows/ci.yml)

> 관련 저장소: [DeepSC-Digital](https://github.com/EHeum/DeepSC-Digital) — 이 구현에 M-QAM constellation
> 제약(DeepJSCC-Q 방식)을 더해 DeepSC를 디지털화하고 cliff 현상을 분석한 확장 프로젝트

---

## 1. 시스템 모델 분석

### 1.1 두 단계(level)로 본 통신 시스템 (Sec. III, Fig. 1)

| 단계 | 역할 | 논문의 대응 블록 |
|---|---|---|
| **Semantic level** | 문장의 *의미*를 추출/복원. 송·수신기는 공통의 *background knowledge*(학습 데이터)를 공유 | semantic encoder `S_β`, semantic decoder `S⁻¹_χ` |
| **Transmission level** | 의미 특징이 물리 채널을 통해 올바르게 전달되도록 보장 | channel encoder `C_α`, channel decoder `C⁻¹_δ` |

논문은 두 종류의 잡음을 구분합니다.

* **Semantic noise (Def. 1)** – 단어/문장의 모호성 때문에 해석이 어긋나는 교란
* **Physical channel noise (Def. 2)** – AWGN, fading, multipath 등에 의한 신호 왜곡

### 1.2 수식 모델 (eq. (1)–(3))

입력 문장 `s = [w_1, …, w_L]`에 대해

```
x  = C_α( S_β(s) )                 x ∈ C^{M×1}         (1)  송신기
y  = h·x + n,  n ~ CN(0, σ_n² I)                       (2)  채널 (AWGN: h=1, Rayleigh: h ~ CN(0,1))
ŝ  = S⁻¹_χ( C⁻¹_δ(y) )                                 (3)  수신기
```

* 코히어런스 시간이 `M`(=한 문장의 심볼 수)이라고 가정 → **문장 단위 block fading**
* 수신기는 **완벽한 CSI**를 가진다고 가정 (Sec. V)
* 채널은 학습 시 역전파가 가능해야 하므로 신경망의 한 "층"으로 구현

### 1.3 손실 함수

**(a) Cross-entropy (eq. (4))** – 의미 수준 오차를 줄이기 위한 문장 복원 손실

```
L_CE(s, ŝ) = − Σ_l q(w_l) log p(w_l) + (1 − q(w_l)) log(1 − p(w_l))
```

**(b) Mutual information (eq. (5)–(10))** – 전송률(용량)을 높이기 위한 손실.
`I(x;y) = D_KL(p(x,y) ‖ p(x)p(y))` 를 Donsker–Varadhan 표현(Theorem 1)으로 하한을 구하고,
그 하한을 신경망 `T`(MINE)로 근사합니다.

```
L_MI(x, y; T) = E_{p(x,y)}[f_T] − log E_{p(x)p(y)}[e^{f_T}]  ≤  I(x; y)        (9),(10)
```

**(c) 전체 손실 (eq. (14))**

```
L_total = L_CE(s, ŝ; α, β, χ, δ) − λ · L_MI(x, y; T, α, β),   0 ≤ λ ≤ 1
```

MI 항은 **송신기(α, β)**에만 그래디언트를 주며, 같은 전력 제약 하에서 채널 입력 분포를
용량에 가깝게 만드는 정규화 항 역할을 합니다.

### 1.4 신경망 구조 (Fig. 2, Table I)

```
S (B×L) ─Embedding→ E (B×L×128) ─3×Transformer Enc(8 heads)→ M (B×L×128)
  ─Dense 256 ReLU → Dense 2N→ reshape → X (B×NL×2 = 복소 NL개) ─Power norm→ Channel
  ─→ Y ─reshape→ (B×L×2N) ─Dense 256 ReLU → Dense 128 ReLU→ M̂ (B×L×128)
  ─3×Transformer Dec(8 heads)→ Dense(Vocab) + Softmax → Ŝ
MI model T : Dense 256 ReLU → Dense 256 ReLU → Dense 1
```

* `N` = 단어(토큰)당 복소 심볼 수. Table I의 channel encoder 출력 16 unit = **단어당 8개 복소 심볼**
  (Fig. 9의 "8 symbols per word"와 일치).
* 송신 심볼 수는 문장 길이에 비례 → **가변 길이 입력/가변 길이 심볼열**.

### 1.5 학습 절차 (Algorithm 1–4, Fig. 4–5)

* **Phase 1 (Alg. 2)** – 송신기를 고정하고 `−L_MI`를 최소화하여 MI 추정 모델 `T` 학습
* **Phase 2 (Alg. 3)** – `T`를 고정하고 `L_total`로 송수신기 전체(α, β, χ, δ) 학습
* 매 미니배치마다 두 단계를 번갈아 수행 (Alg. 1)
* **Transfer learning (Alg. 4, Fig. 5)**
  * 채널이 바뀌면: semantic encoder/decoder를 **고정**하고 channel encoder/decoder만 재학습
  * 배경지식이 바뀌면: channel encoder/decoder를 **고정**하고 semantic encoder/decoder만 재학습
  * 마지막에 전체를 풀어 몇 epoch 미세조정

### 1.6 성능 지표 (Sec. III-C)

* **BLEU** (eq. (11),(12)) – n-gram 정밀도 + brevity penalty. 논문 그림은 각 n에 대한
  *개별* n-gram 점수(1-gram … 4-gram)를 사용
* **Sentence similarity** (eq. (13)) – BERT 문장 임베딩의 코사인 유사도

---

## 2. 구현 (Python / PyTorch)

### 2.1 저장소 구성

```
deepsc/
  data.py        Europarl 전처리(4–30 단어 문장), 어휘집, Dataset
  models.py      DeepSC 송수신기(Fig. 2, Table I), MINE(T 네트워크), eq. (9) MI 하한
  channels.py    AWGN / Rayleigh / Rician / Erasure 채널, 전력 정규화 (eq. (2))
  engine.py      Algorithm 1–4 (MI 모델 학습 ↔ 전체 네트워크 학습, transfer learning), 평가
  metrics.py     BLEU (eq. (11)–(12)), BERT sentence similarity (eq. (13))
  baselines.py   Huffman / 5-bit 소스 부호 + RS / Turbo(log-MAP) 채널 부호 + Gray M-QAM
scripts/
  prepare_data.py  eval_baselines.py  train.py  evaluate.py  estimate_mi.py  plot_results.py  run_all.sh
tests/test_deepsc.py   단위 테스트 (shape/전력/SNR/MINE/BLEU/부호기 round-trip)
results/               실험 결과 JSON + 그림
```

### 2.2 논문 ↔ 코드 대응

| 논문 | 코드 |
|---|---|
| `S_β` semantic encoder (Embedding + 3×Transformer enc, 128 dim, 8 heads) | `DeepSC.embedding`, `DeepSC.semantic_encoder` |
| `C_α` channel encoder (Dense 256 ReLU → Dense 16) | `DeepSC.channel_encoder` |
| 전력 제약 / 복소 심볼 변환 (reshape layer) | `DeepSC.transmit`, `channels.power_normalize` |
| 채널 layer `y = h x + n` (eq. (2)) | `channels.awgn / rayleigh / rician / erasure` |
| `C⁻¹_δ` channel decoder (Dense 256 ReLU → Dense 128 ReLU) | `DeepSC.channel_decoder` |
| `S⁻¹_χ` semantic decoder (3×Transformer dec + softmax) | `DeepSC.semantic_decoder`, `DeepSC.prediction` |
| MI 모델 `T` (Dense 256 → 256 → 1) | `models.MINE` |
| `L_MI` (eq. (9)) | `models.mi_lower_bound` |
| `L_total = L_CE − λ L_MI` (eq. (14)) | `engine.train` (phase 2) |
| Algorithm 2 (MI 모델 학습) / Algorithm 3 (전체 학습) | `engine.train` phase 1 / phase 2 |
| Algorithm 4 (transfer learning, 일부 모듈 고정) | `train.py --init … --freeze semantic|channel --unfreeze-epoch k` |
| BLEU / sentence similarity | `metrics.bleu_scores`, `metrics.SentenceSimilarity` |

### 2.3 구현상의 선택과 논문과의 차이 (중요)

| 항목 | 논문 | 본 구현 | 이유 |
|---|---|---|---|
| 데이터셋 | Europarl v7 전체 (≈2.0M 문장) | NLTK `europarl_raw` 영어 샘플 10개 세션 → 4–30 단어, 중복 제거 후 **12,985 문장** (train 11,687 / test 1,298, 어휘 10,200) | 실행 환경에서 statmt.org 접근 불가. `--raw`로 전체 Europarl 사용 가능 |
| 학습량 | 미기재(GPU) | CPU 4코어, 40 epoch, batch 64, Adam lr 1e-3, warm-up 200 step | 계산 자원 |
| 학습 SNR | 미기재 (Fig. 10/11은 12 dB) | 배치마다 SNR ~ U[0, 18] dB | 한 모델로 전 SNR 구간 평가 |
| 심볼 출력층 활성화 | Table I: ReLU | **선형** (이후 전력 정규화) | ReLU면 심볼이 1사분면에 갇혀 신호 공간의 3/4을 버림 |
| MI 모델 마지막 층 | Table I: ReLU | **선형** | DV 하한의 `T`는 실수 전체 값을 가져야 함 |
| 수신단 위치 정보 | 명시 없음 | 채널 디코더 출력에 positional encoding을 더함 | 수신기는 각 N-심볼 블록이 몇 번째 단어 슬롯인지 이미 앎(전송 비용 0). 작은 데이터에서 수렴 속도 개선 |
| 패딩 | – | 패딩 토큰은 전송하지 않음(심볼=0, 전력 계산 제외) | 심볼 수를 공정하게 계산 |
| λ (eq. (14)) | 0 ≤ λ ≤ 1 (값 미기재) | λ = 0.01 | MI 하한(≈2–3 nat)과 CE의 크기 차이를 고려 |
| MI 단위 | 미기재 | 복소 심볼 1개당 (x, y) 쌍, bit | |
| Rayleigh | `h ~ CN(0,1)` | 문장 단위 block fading, 완전 CSI, ZF 등화 `y/h` | 논문 가정(코히어런스 시간 M, perfect CSI) |
| 베이스라인 | Huffman/5-bit/Brotli × RS/Turbo, 8/64/128-QAM, JSCC[22] | Huffman/5-bit × RS(42,30)·RS(54,42)/Turbo(1/3, log-MAP 5회), **64-QAM** | 128-QAM(cross), Brotli(128문장 묶음 압축), BiLSTM JSCC는 생략 |
| Sentence similarity | BERT | 코드만 구현(`metrics.SentenceSimilarity`), **이번 실험에서는 미측정** | 실행 환경에서 BERT 가중치(HuggingFace) 다운로드 불가 |

---

## 3. 검증 결과

모든 수치는 test set 1,298 문장(베이스라인은 처음 500 문장)에 대한 결과입니다.
BLEU는 논문과 같이 **개별 n-gram BLEU**(brevity penalty 포함)입니다.

### 3.1 BLEU vs SNR — DeepSC vs 기존 분리 부호화 (Fig. 6 재현)

![AWGN](results/fig6_bleu_awgn.png)
![Rayleigh](results/fig6_bleu_rayleigh.png)

**AWGN** (BLEU-1 / BLEU-4)

| 방식 | 심볼/문장 | 0 dB | 3 dB | 6 dB | 9 dB | 12 dB | 15 dB | 18 dB |
|---|---|---|---|---|---|---|---|---|
| **DeepSC** | 162 | **0.61** / **0.17** | **0.79** / **0.43** | **0.85** / **0.54** | 0.86 / 0.58 | 0.87 / 0.59 | 0.87 / 0.60 | 0.87 / 0.60 |
| Huffman + RS(42,30) | 106 | 0.00 / 0.00 | 0.00 / 0.00 | 0.01 / 0.00 | 0.01 / 0.00 | 0.03 / 0.00 | 0.11 / 0.00 | 0.59 / 0.40 |
| 5-bit + RS(54,42) | 115 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.01 / 0.00 | 0.03 / 0.00 | 0.15 / 0.00 | 0.58 / 0.26 |
| Huffman + Turbo | 208 | 0.01 / 0.00 | 0.01 / 0.00 | 0.15 / 0.04 | 0.99 / 0.99 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| 5-bit + Turbo | 248 | 0.00 / 0.00 | 0.00 / 0.00 | 0.25 / 0.09 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |

**Rayleigh fading** (BLEU-1 / BLEU-4)

| 방식 | 심볼/문장 | 0 dB | 3 dB | 6 dB | 9 dB | 12 dB | 15 dB | 18 dB |
|---|---|---|---|---|---|---|---|---|
| **DeepSC** | 162 | **0.49** / **0.13** | **0.59** / **0.22** | **0.67** / 0.29 | **0.73** / 0.36 | 0.76 / 0.39 | 0.79 / 0.43 | 0.80 / 0.44 |
| Huffman + RS(42,30) | 106 | 0.00 / 0.00 | 0.00 / 0.00 | 0.01 / 0.00 | 0.02 / 0.00 | 0.07 / 0.02 | 0.21 / 0.15 | 0.44 / 0.37 |
| 5-bit + RS(54,42) | 115 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.02 / 0.00 | 0.07 / 0.02 | 0.23 / 0.12 | 0.45 / 0.35 |
| Huffman + Turbo | 208 | 0.02 / 0.00 | 0.12 / 0.09 | 0.35 / 0.31 | 0.55 / 0.53 | 0.74 / 0.72 | 0.86 / 0.85 | 0.90 / 0.89 |
| 5-bit + Turbo | 248 | 0.01 / 0.01 | 0.12 / 0.09 | 0.35 / 0.32 | 0.55 / 0.52 | 0.74 / 0.73 | 0.88 / 0.87 | 0.94 / 0.93 |

**해석**

* **논문의 핵심 주장(저 SNR 영역에서의 강건성)이 재현됨.** AWGN 0 dB에서 DeepSC는 BLEU-1 0.61
  (논문 Fig. 6(a) ≈ 0.62)인 반면, 기존 방식은 모두 ≈ 0입니다. 기존 방식은 SNR이 문턱을 넘으면
  갑자기 완벽해지는 **cliff effect**를 보이고, DeepSC는 **완만하게 열화(graceful degradation)**됩니다.
* **고 SNR에서는 기존 방식(특히 Turbo)이 더 좋음** — 논문도 "SNR > 12 dB에서는 전통적 방식이
  DNN 기반보다 낫다"고 보고합니다. 본 구현의 DeepSC 포화값(AWGN BLEU-1 0.87)은 논문(≈0.92)보다
  낮은데, 학습 문장 수가 논문의 약 0.6%에 불과해 test에만 나오는 단어(test 토큰의 2%)와 희귀어를
  복원하지 못하기 때문입니다(아래 3.7).
* 심볼 수: DeepSC(단어당 8 복소 심볼 → 문장당 162개)는 RS 방식보다 많고 Turbo 방식보다 적은
  심볼을 씁니다. RS보다 적은 심볼을 쓰는 경우는 3.3의 "단어당 4 심볼(≈81개)" 결과를 참고하세요.
* Rayleigh에서 본 구현의 Turbo 베이스라인은 논문 그림보다 훨씬 좋습니다. 본 구현은 문장 단위
  block fading + 완전 CSI + 정확한 soft LLR(잡음분산 `σ²/|h|²`)을 사용하므로, 깊은 페이딩이 걸린
  문장만 실패하기 때문입니다(논문의 베이스라인 구현 세부는 공개되지 않음). 그럼에도 **BLEU-1 기준 9 dB 이하,
  BLEU-4 기준 3 dB 이하에서는 DeepSC가 모든 베이스라인보다 우수**합니다.

### 3.2 복원 문장 예시 (Table II 재현, Rayleigh, SNR = 18 dB)

| 방식 | 문장 |
|---|---|
| 송신 문장 | i would like to finish off by making a general comment on interreg . |
| **DeepSC** | i would like to finish off by making a general comment on interreg . |
| Huffman + RS | i utmra neckeiinscs irh aft hbe rhhuttr a general comms pln interreg . |
| 5-bit + RS | m womdc .mim xo vihi pk!fv!aykea.yng!a geperi. doement!knkintu.rwg z |
| Huffman + Turbo | i would like to finish off by making a general comment on interreg . |

저 SNR(AWGN 0 dB)에서도 DeepSC는 *"but as you will also be aware , we are waiting for a commission
communication on coherence , really , serve the strategic development policy ."* 처럼 문법과
상당 부분의 의미를 유지하는 반면, 기존 방식은 문자 단위로 깨져 읽을 수 없습니다.

### 3.3 단어당 심볼 수의 영향 (Fig. 8)

Fig. 8처럼 단어당 복소 심볼 수 N을 줄이면 BLEU가 떨어집니다. 단어당 4 심볼(문장당 평균 **81개**)은
가장 적은 심볼을 쓰는 RS 베이스라인(106–115개)보다도 적지만, 0–18 dB 전 구간에서 RS 방식보다 우수합니다.

| AWGN BLEU-1 / BLEU-4 | 심볼/문장 | 0 dB | 3 dB | 6 dB | 9 dB | 12 dB | 15 dB | 18 dB |
|---|---|---|---|---|---|---|---|---|
| DeepSC, 8 심볼/단어 | 162 | 0.61 / 0.17 | 0.79 / 0.43 | 0.85 / 0.54 | 0.86 / 0.58 | 0.87 / 0.59 | 0.87 / 0.60 | 0.87 / 0.60 |
| DeepSC, 4 심볼/단어 | 81 | 0.39 / 0.03 | 0.59 / 0.15 | 0.73 / 0.33 | 0.79 / 0.42 | 0.81 / 0.46 | 0.82 / 0.48 | 0.82 / 0.49 |
| Huffman + RS(42,30) | 106 | 0.00 / 0.00 | 0.00 / 0.00 | 0.01 / 0.00 | 0.01 / 0.00 | 0.03 / 0.00 | 0.11 / 0.00 | 0.59 / 0.40 |

![symbols per word](results/fig8_symbols_per_word.png)

### 3.4 Mutual information 손실의 효과 (Fig. 9, 10)

같은 구조·같은 seed로 λ = 0.01(MI 항 사용)과 λ = 0(CE만 사용) 모델을 각각 40 epoch 학습했습니다.
MI는 학습이 끝난 송신기를 고정하고 SNR마다 새 MINE을 1,500 step 학습해 eq. (9)의 하한으로 추정했습니다.

| AWGN | 0 dB | 3 dB | 6 dB | 9 dB | 12 dB | 15 dB | 18 dB |
|---|---|---|---|---|---|---|---|
| I(X;Y) with MI term [bit/심볼] | 0.99 | 1.57 | 2.29 | 3.11 | 4.00 | 4.85 | 5.67 |
| I(X;Y) without MI term | 0.99 | 1.56 | 2.27 | 3.07 | 3.94 | 4.77 | 5.61 |
| Gaussian 입력 용량 log2(1+SNR) | 1.00 | 1.58 | 2.32 | 3.16 | 4.08 | 5.03 | 6.00 |
| BLEU-1 with MI term | **0.61** | **0.79** | **0.85** | **0.86** | **0.87** | **0.87** | **0.87** |
| BLEU-1 without MI term | 0.59 | 0.77 | 0.83 | 0.84 | 0.85 | 0.85 | 0.85 |
| BLEU-4 with MI term | **0.17** | **0.43** | **0.54** | **0.58** | **0.59** | **0.60** | **0.60** |
| BLEU-4 without MI term | 0.15 | 0.38 | 0.49 | 0.53 | 0.54 | 0.55 | 0.55 |

![MI](results/fig9_mutual_information.png)
![MI ablation BLEU](results/fig_mi_ablation_bleu.png)
![MI during training](results/fig10_mi_during_training.png)

(마지막 그림: 학습 중 phase 1에서 추정한 MI 하한의 epoch 평균. 배치마다 SNR이 무작위이므로 절대값보다 추세를 보는 용도입니다.)

* MI 항을 넣은 모델이 모든 SNR에서 MI가 조금 더 높고(최대 +0.09 bit/심볼, 15 dB), BLEU-1은 약 +0.02,
  BLEU-4는 +0.02–0.05 높습니다. 방향은 논문과 같습니다.
* 다만 **차이는 논문 Fig. 9(같은 MI를 얻는 데 SNR 약 5 dB 차이)보다 훨씬 작습니다.** 본 구현에서는 MI 항이
  없어도 송신 심볼 분포가 이미 Gaussian 용량에 근접하기 때문입니다. 본 구현은 심볼 출력층을 선형으로
  두었는데(2.3절), Table I처럼 ReLU를 쓰면 심볼이 음이 아닌 값으로 제한되어 MI가 크게 떨어지고, 이때
  MI 항의 효과가 커질 것으로 추정합니다(검증하지 않은 가설). 단일 seed라 +0.02 수준의 BLEU 차이는
  seed 간 편차 안에 들 수도 있습니다.

### 3.5 Transfer learning (Algorithm 4, Fig. 13)

AWGN에서 학습된 모델을 불러와 **semantic encoder/decoder를 고정**하고 channel encoder/decoder만
8 epoch 재학습한 뒤, 9 epoch부터 전체를 풀어 4 epoch 미세조정(총 12 epoch)했습니다.

| 새 채널 | 방법 | 1 epoch 후 train CE | 12 epoch 후 train CE | BLEU-1 (평가 조건) |
|---|---|---|---|---|
| Rayleigh | TL (AWGN 모델에서) | **1.06** | 1.00 | 0.51 / 0.72 / 0.81 / **0.85** (0 / 6 / 12 / 18 dB) |
| Rayleigh | 처음부터 (12 epoch) | 6.83 | 2.40 | – |
| Rayleigh | 처음부터 (**40 epoch**) | 6.83 | 1.02 (40 ep) | 0.49 / 0.67 / 0.76 / 0.80 |
| Erasure (p = 0.1) | TL (AWGN 모델에서) | **0.55** | 0.69 | **0.83 / 0.82 / 0.79 / 0.77** (p = 0.05 / 0.1 / 0.15 / 0.2, 12 dB) |
| Erasure (p = 0.1) | 처음부터 (12 epoch) | 6.84 | 2.13 | 0.65 / 0.64 / 0.63 / 0.61 (p = 0.05 / 0.1 / 0.15 / 0.2, 12 dB) |

![TL Rayleigh](results/fig13_tl_rayleigh_loss.png)
![TL erasure](results/fig13_tl_erasure_loss.png)
![TL Rayleigh BLEU](results/fig13_tl_rayleigh_bleu.png)
![TL erasure BLEU](results/fig13_tl_erasure_bleu.png)

* 논문 Fig. 13(a),(b)처럼 사전학습 모델을 쓰면 **수렴에 필요한 epoch이 크게 줄어듭니다.** 첫 epoch
  부터 CE가 처음부터 학습한 모델의 40 epoch 수준에 도달합니다.
* 같은 12 epoch에서 TL 모델의 BLEU가 훨씬 높고, Rayleigh에서는 40 epoch 동안 처음부터 학습한
  모델보다도 좋았습니다(18 dB BLEU-1 0.85 vs 0.80). AWGN에서 학습된 semantic 부분이 페이딩
  잡음에 덜 오염된 표현을 학습했기 때문으로 보입니다.

### 3.6 계산량 (Table III)

| | DeepSC (본 구현, CPU, 배치 1, greedy) | 논문 DeepSC (GTX 2060) |
|---|---|---|
| 문장당 처리 시간 | 약 62 ms | 3.27 ms |

CPU에서 다른 학습과 동시에 측정한 값이라 절대값 비교는 의미가 없습니다(참고용).

### 3.7 한계와 해석 시 주의점

* **데이터 규모**: 1.3만 문장/1만 단어 어휘로, 어휘의 절반 가까이가 1회만 등장합니다. 학습 CE(0.49)와
  test CE(1.22, 6 dB)의 차이가 보여주듯 과적합 경향이 있으며, 고 SNR 포화값이 논문보다 낮은 주원인입니다.
  전체 Europarl을 `scripts/prepare_data.py --raw <dir>`로 넣으면 그대로 확장 가능합니다.
* **Sentence similarity(eq. (13))**는 코드만 있고 BERT 가중치를 받을 수 없어 측정하지 못했습니다.
  `pip install transformers` 후 `deepsc.metrics.SentenceSimilarity()`로 계산할 수 있습니다.
* 베이스라인은 논문의 구성을 그대로 따르지 못한 부분(128-QAM, Brotli, JSCC[22])이 있습니다.
* 단일 seed 실험입니다.

---

## 개발 환경

```bash
git clone https://github.com/EHeum/DeepSC-PyTorch.git && cd DeepSC-PyTorch
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # GPU라면 PyTorch 공식 안내에 맞는 wheel
pip install -r requirements-dev.txt                                  # numpy, nltk, matplotlib, reedsolo, pytest, ruff …
ruff check deepsc scripts tests                                      # lint
pytest -q                                                            # 단위 테스트
```

* **CI** (`.github/workflows/ci.yml`): `main` push, PR, 수동 실행 시 GitHub Actions(ubuntu, Python 3.11,
  CPU PyTorch)에서 ruff lint, pytest, 그리고 256문장 1 epoch 학습·평가 스모크 테스트를 돌립니다.
* **Claude Code 클라우드 세션** (`.claude/hooks/session-start.sh`): 원격 세션이 시작될 때 의존성을
  자동으로 설치합니다(동기 실행. 설치가 끝나야 세션이 시작됨).
* 학습된 체크포인트(`checkpoints/*.pt`, 개당 약 22 MB)는 저장소에 포함하지 않습니다. 학습 곡선은
  `results/*_history.json`, 로그는 `logs/`에 있습니다.

## 4. 실행 방법

```bash
pip install -r requirements.txt
python3 scripts/prepare_data.py                 # Europarl 샘플 다운로드 + 전처리 (data/europarl_en.json 포함됨)
python3 -m pytest -q tests                      # 단위 테스트

# 학습 (Algorithm 1)
python3 scripts/train.py --channel awgn --epochs 40 --out checkpoints/deepsc_awgn.pt
python3 scripts/train.py --channel rayleigh --epochs 40 --out checkpoints/deepsc_rayleigh.pt
# MI 손실 없이 / 단어당 4 심볼
python3 scripts/train.py --channel awgn --mi-lambda 0 --out checkpoints/deepsc_awgn_nomi.pt
python3 scripts/train.py --channel awgn --symbols-per-word 4 --out checkpoints/deepsc_awgn_sym4.pt
# Transfer learning (Algorithm 4): AWGN 모델 → Rayleigh, semantic 부분 고정 후 9 epoch에 전체 해제
python3 scripts/train.py --channel rayleigh --epochs 12 --init checkpoints/deepsc_awgn.pt \
    --freeze semantic --unfreeze-epoch 9 --out checkpoints/deepsc_rayleigh_tl.pt

# 평가
python3 scripts/evaluate.py --ckpt checkpoints/deepsc_awgn.pt --out results/deepsc_awgn.json
python3 scripts/eval_baselines.py --out results/baselines.json
python3 scripts/estimate_mi.py --ckpts "with MI term=checkpoints/deepsc_awgn.pt,without MI term=checkpoints/deepsc_awgn_nomi.pt"
python3 scripts/plot_results.py
```

전체 실험은 `scripts/run_all.sh` 한 번으로 재현됩니다(CPU 4코어 기준 약 6시간). CUDA GPU가 있으면 자동으로 사용합니다.

