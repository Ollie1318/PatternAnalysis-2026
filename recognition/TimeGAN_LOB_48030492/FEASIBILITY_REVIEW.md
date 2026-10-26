# Feasibility Review: Conditional TimeGAN for Limit Order Book Simulation

**Project 2.7 (Time-Series Generation), Hard difficulty.** Draft for the midpoint check-off, 27 October 2026.

## 1. User need, scope and acceptance criteria

The intended user of this prototype is a quantitative finance firm which requires large amounts of realistic market data to allow them to develop and back-test trading algorithms. Since access to high volumes of real historical data is expensive it is beneficial to develop a generative model that can produce synthetic LOB data which appears realistic. The prototype will be a single generator conditioned on the stock ticker, trained on the five LOBSTER samples (AMZN, AAPL, GOOG, INTC, MSFT; 2012-06-21). The purpose of the prototype is to see if the TimeGAN model is capable of learning the structural constraints of a LOB.

Acceptance criteria (on 500-step generations, per ticker, scored against the test hour):

| Criterion | Definition | Target |
|---|---|---|
| **1. Validity** | Share of snapshots breaking an invariant: crossed or locked spread, non-monotonic ladder, negative size | < 1%, and lower than the RCGAN baseline *(real: 0%)* |
| **2. Conditioning** | Per-ticker VWAP variance and VWAP–mid distance; spread in ticks | VWAP stats within ±30% of real and distinct across tickers; INTC/MSFT one-tick spread in ≥ 90% of steps *(real ≈ 99%)*; AMZN/AAPL/GOOG keep wide spreads *(real mean 13–27 ticks)* |
| **3. Volatility clustering** | ACF of returns and of \|returns\| | \|ACF r, lag 1\| < 0.05; ACF \|r\| positive up to 60 s, lag 1 within ±0.05 of real *(real: 0.07–0.16)* |
| **4. Spec metrics** | KL on spread and mid returns (tick-aligned bins); SSIM on depth heatmaps | Reported against the real-vs-real reference as well as the spec's KL ≤ 0.1 and SSIM > 0.6, with a critique of SSIM |
| **5. Resources** | Wall-clock training time and peak VRAM for a full run | < 30 min and < 2 GB on one Tesla T4 |

## 2. Model choice and course concepts

I will implement a TimeGAN (Yoon, Jarrett & van der Schaar, 2019) conditioned on the stock ticker (Mirza & Osindero, 2014), in the Hard difficulty band, and compare it against an RCGAN-style baseline (Esteban et al., 2017). Recurrent networks (GRUs) suit time-series data because their hidden state carries information forward, so each generated step depends on its history. TimeGAN adds three networks to a recurrent GAN. An embedder and a recovery network learn a compact latent space from real books. A supervisor learns real next-step transitions, which gives the generator a direct signal for temporal dynamics that an adversarial loss alone captures poorly.

Books are resampled to a 1 s clock-time grid using the last state at or before each step, so there is no look-ahead. Each ticker is split chronologically into train (09:30–14:00), validation (14:00–15:00) and test (15:00–16:00). Each step has 41 features: the mid log-return, the price offsets from the mid, and log sizes. Min-max scaling is fitted on the training split and shared across tickers, so the model must learn each stock's identity. Models train on 24-step windows, the paper's length, and generate 500 steps. A constrained representation (log spread and log level gaps) is valid by construction and gives a comparison point.

## 3. Preliminary feasibility evidence

**Baseline smoke test.** I ran 300 iterations of the conditional RCGAN baseline, which uses the same GRU backbone as TimeGAN, on all five tickers (80,880 training windows). Training ran end to end on both CPU and GPU and saved losses, loss plots and checkpoints. Runs with the same seed reproduced identical losses.

| Device | Time per iteration | 300 iterations | Peak VRAM | Parameters |
|---|---|---|---|---|
| Tesla T4 | 0.027 s | 8.1 s | 130 MB | 45,866 |
| PC CPU | 0.29 s | 88 s | – | 45,866 |

**Resource estimate.** A TimeGAN iteration involves five networks instead of two, so I expect it to cost roughly 3–4× an RCGAN iteration. With 10,000 iterations per phase, as in the reference code, a full run should take about 15–20 minutes and use under 1 GB of VRAM on one T4. The first TimeGAN run will confirm these estimates.

## 4. Risks, budget and fallback

| Risk | Signal | Mitigation |
|---|---|---|
| **Ticker collapse** | All stocks get similar spreads and VWAP statistics | D also receives the ticker; global scaling; per-ticker metric checks |
| **Memorisation** (one day of overlapping windows) | Generated windows almost identical to training windows | Small model; nearest-neighbour check against training windows; compare with a larger configuration |
| **Collapse to zero returns** (94–96% zeros for INTC/MSFT) | No non-zero moves, so the ACF of \|r\| is flat | Report the share of zero returns and the distribution of non-zero moves, not just KL |
| **GAN instability** | Oscillating or diverging losses; D accuracy stuck near 100% | 2 G steps per D step; D-loss threshold; 2–3 seeds |
| **Test hour out of distribution** (AAPL selloff, end-of-day volume) | Metrics much worse on test than on validation | Report both hours; use it as a failure case in the error analysis |
| **KL sensitive to bins and length** | KL changes with bin count or sample length alone | Tick-aligned bins; compare with real-vs-real KL at the same length |

- **Next experiment:** build the TimeGAN embedder, recovery and supervisor, then compare a joint-training run against RCGAN on the unconstrained representation.
- **Budget:** one T4 VM. At about 0.03 s per iteration, 10,000 iterations take about 5 minutes, so every experiment can be run with multiple seeds.
- **Fallback:** an unconditional TimeGAN on AMZN only.

