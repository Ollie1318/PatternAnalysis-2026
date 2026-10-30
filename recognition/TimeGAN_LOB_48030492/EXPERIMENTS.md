# Experiment Log

Every experiment is recorded here, **including failures**, as requested by
staff. Results are reported honestly: no cherry-picked seeds or hidden runs.

All runs use the same chronological splits (train 09:30–14:00, validation
14:00–15:00, test 15:00–16:00) and the same evaluation code unless stated.

## Entry template

```
### EXP-NNN — <short title>
- **Date:**
- **Commit:** <hash>
- **Model / representation:** e.g. TimeGAN (conditional), unconstrained
- **Config:** hidden size, layers, batch size, L_train, lr, epochs per phase, λ_pen, seed(s)
- **Compute:** GPU, peak VRAM, wall-clock time, parameter count
- **Result:** violation rates, VWAP mean/var per ticker, ACF, KL, SSIM
- **What went wrong / observations:**
- **What was tried next:**
```

## Experiments

### EXP-001 — RCGAN baseline CPU smoke test
- **Date:** 2026-10-25
- **Commit:** run with the first, uncommitted version of `train.py` (on top of 6cc7369); the fix found in this run is included in aee667b, the first commit of `train.py`
- **Model / representation:** RCGAN-style (conditional, 3-layer GRU, bidirectional D), unconstrained, global min-max (0.5/99.5% bounds, winsorised)
- **Config:** hidden 24, layers 3, cond dim 8, z dim 41 (U(0,1)), batch 128, seq len 24, stride 1, Adam lr 1e-3, 2 G steps per D step, D threshold 0.15, 300 iterations, seed 0
- **Compute:** local CPU (no GPU); 88 s total, 0.29 s/iteration; params G 13,665 / D 32,201; peak VRAM n/a on CPU
- **Result:** pipeline runs end to end (data, training, losses.csv, loss plot, checkpoint); seeded runs reproduce identical losses. Losses oscillate (raw G 0.67-4.10, raw D 0.07-1.39; 25-iteration means G 0.7-3.6, D 0.25-1.3) with D winning around iterations 60-110 and again near 300. D was updated on 98.7% of iterations, so the 0.15 threshold rarely bites at this stage. Far too short to judge sample quality.
- **What went wrong / observations:** first checkpoint could not be loaded with `torch.load(weights_only=True)` because the config stored a TorchVersion object; fixed by storing it as a string.
- **What was tried next:** the same smoke test on a GPU for peak VRAM and time per iteration (EXP-002).

### EXP-002 — RCGAN baseline GPU smoke test (Tesla T4)
- **Date:** 2026-10-26
- **Commit:** aee667b
- **Model / representation:** identical to EXP-001 (RCGAN-style, unconstrained, global robust min-max)
- **Config:** identical to EXP-001 (hidden 24, layers 3, cond dim 8, z dim 41 U(0,1), batch 128, seq len 24, stride 1, Adam lr 1e-3, 2 G steps per D step, D threshold 0.15, 300 iterations, seed 0); `--device auto` selected CUDA
- **Compute:** Virtual GPU with one NVIDIA Tesla T4 (15 GB, driver 595.71.05, CUDA 13.2 supported), Python 3.10.21, PyTorch 2.14.1+cu126. Training loop 8.1 s for 300 iterations = **0.027 s/iteration** (~11x faster than the 0.29 s/iteration CPU run in EXP-001; ~12 s wall-clock including data loading). **Peak VRAM 130 MB** (under 1% of the T4). Params G 13,665 / D 32,201 (45,866 total). D updated on 98.7% of iterations.
- **Result:** GPU path works end to end (CUDA selected automatically, all run outputs saved). Losses match the CPU run to ~1e-5 for the first iterations, then drift apart (first difference > 1e-4 at iteration 25; G loss at iteration 100: CPU 3.618, GPU 3.510; final G 2.25 vs 1.22). Same qualitative pattern: early D dominance then oscillation. Still far too short to judge sample quality.
- **What went wrong / observations:** CPU and GPU runs with the same seed are not bit-identical. CPU and cuDNN GRU kernels round floating-point operations differently, and the adversarial feedback loop amplifies these ~1e-5 differences over iterations. This is expected rather than a bug: same-device reruns are reproducible (verified on CPU in EXP-001), but results must be compared within one device, and all reported results should come from the GPU.
- **What was tried next:** resource figures feed the feasibility review: the model is tiny relative to the hardware, so a single T4 is ample and multi-GPU (`torchrun`) is not needed.

### EXP-003 — TimeGAN phase 1 (embedder + recovery autoencoder), Tesla T4
- **Date:** 2026-10-27
- **Commit:** 223e66f
- **Model / representation:** conditional TimeGAN Embedder + Recovery only (phase 1), unconstrained, global min-max (0.5/99.5% bounds, winsorised), all five tickers
- **Config:** hidden 24, latent 24 (= hidden), layers 3, cond dim 8, batch 128, seq len 24, stride 1, Adam lr 1e-3, loss 10·sqrt(MSE) as in the reference code, 10,000 iterations, seed 0; validation MSE tracked every 50 iterations on 2,048 evenly spread validation windows; run name `train-autoencoder-test1`
- **Compute:** Tesla T4 (same VM as EXP-002), PyTorch 2.14.1+cu126. 55.6 s for 10,000 iterations = **0.0056 s/iteration** (~8x faster than a 0.044 s/iteration CPU test of the same code). **Peak VRAM 275 MB.** Params E 13,240 / R 12,441 (25,681 total).
- **Result:** reconstruction learned quickly and stably. Validation MSE fell from 0.105 at iteration 1 to 0.0055 by iteration 1,000, then improved only slowly to **0.0045** at 10,000 (best 0.00449 at 9,950). That is about **18x below the 0.081 predict-the-mean baseline** on the same windows. Mean train MSE per block: 0.0060 (1k-3k), 0.0055 (3k-6k), 0.0050 (6k-10k). Validation stayed at or slightly below train throughout, so there is no sign of overfitting.
- **What went wrong / observations:** nothing failed. Most of the gain happens in the first ~500 iterations; the last 8,000 iterations bought only ~0.001 MSE, so phase 1 could be shortened if time matters. Caveats: the baseline uses one mean across all tickers, so a per-ticker mean would be a stricter reference; and an average MSE does not show *which* features reconstruct poorly (e.g. the rare non-zero returns, which matter most for volatility clustering, could be smoothed away while sizes dominate the average).

### EXP-004 — Diagnostics of the EXP-003 autoencoder: conditioning, per-ticker and per-feature error
- **Date:** 2026-10-27
- **Commit:** analysis of the EXP-003 checkpoint (`runs/train-autoencoder-test1`, trained at 223e66f); analysis scripts run locally on CPU, not yet in the repo; raw results in `runs/train-autoencoder-test1/autoencoder_analysis.json`
- **Model / representation:** as EXP-003 (conditional Embedder + Recovery, hidden = latent = 24, unconstrained, global scaling, 5 tickers)
- **Config:** full validation set (17,885 windows). Three checks: (1) per-ticker reconstruction MSE vs predicting that ticker's own feature means; (2) ticker swap: reconstruct each ticker's windows with every ticker id; (3) per-feature-group error and explained variance (1 - MSE / variance), pooled and within each ticker, plus mid log-return detail on non-zero moves. Extra reference: predicting every step of a window by that window's own mean (a "static book over 24 s").
- **Compute:** CPU, inference only (seconds).
- **Result:** the low pooled MSE in EXP-003 is mostly the ticker's *level*, not its dynamics.
  - Per ticker, reconstruction beats the ticker's own mean by only **1.1-1.3x** (e.g. AMZN 0.0069 vs 0.0088). The 18x figure in EXP-003 came from a pooled baseline that includes the large differences *between* tickers.
  - Reconstruction is **worse than the static-book baseline** for every ticker (AMZN 0.0069 vs 0.0052; INTC 0.00056 vs 0.00020).
  - Within-ticker explained variance: **log sizes 0.08-0.18 for AMZN/AAPL/GOOG and slightly negative for INTC/MSFT**, and sizes make up 75-90% of each ticker's error. Price offsets 0.22-0.68. Mid log-return 0.09-0.39 for the wide-spread stocks, 0.65 for INTC/MSFT.
  - Mid log-return on non-zero moves: correlation 0.37-0.74 (AMZN/AAPL/GOOG) and 0.91-0.95 (INTC/MSFT); reconstructed moves are only 36-73% of the true size, so returns are damped towards zero.
  - Ticker swap: the networks do use the condition. A wrong id raises MSE ~7x for AMZN/AAPL/GOOG and 170-230x for INTC/MSFT, mostly when swapping between the two groups (wide-spread vs one-tick). Within the wide-spread group a wrong id costs only 8-33%; INTC<->MSFT costs 4-21x.
- **What went wrong / observations:** phase 1 underfits. The autoencoder has learned each ticker's average book (helped by the ticker id) plus part of the price ladder, but very little of how sizes or returns move within a ticker. This matters for generation: every synthetic book is decoded by R, so if R cannot express size variation the generated depth will be close to each ticker's average, and damped returns weaken volatility clustering. It is also a fairness issue: the RCGAN baseline outputs features directly and has no such bottleneck. Likely causes: a 24-dim latent and 24-unit GRUs for 41 features (the paper's hidden size would be 4 x 41 = 164), and the loss plateauing after ~1,000 iterations at lr 1e-3. EXP-003's conclusion that reconstruction was "about 18x below the mean" is superseded by this per-ticker view.
- **What was tried next:** a phase 1 capacity sweep on the T4 (hidden 24 / 64 / 164, latent 24 vs 41), judged on per-ticker MSE against the static-book baseline and within-ticker explained variance of sizes and returns. `train.py` now logs per-ticker validation MSE against each ticker's own mean at the end of phase 1.

### EXP-005 — Unconditional autoencoder ablation (`--cond-dim 0`), Tesla T4
- **Date:** 2026-10-27
- **Commit:** f25f1ac (explicit unconditional mode in `ConditionalGRU`). The first attempt ran on 05591ba and crashed (see below).
- **Model / representation:** as EXP-003, but with ticker conditioning switched off in both networks (`--cond-dim 0`): conditional TimeGAN Embedder + Recovery (phase 1 only), unconstrained, global min-max, all five tickers
- **Config:** identical to EXP-003 apart from cond dim 0: hidden 24, latent 24, layers 3, batch 128, seq len 24, stride 1, Adam lr 1e-3, loss 10·sqrt(MSE), 10,000 iterations, seed 0; run name `ae-uncond-test1`
- **Compute:** Tesla T4 VM (as EXP-002/003), PyTorch 2.14.1+cu126. 51.8 s = 0.0052 s/iteration; peak VRAM 261 MB; E and R each have 616 fewer parameters than EXP-003 (smaller GRU input).
- **Result:** analysed with the same EXP-004 diagnostics on the full validation set (17,885 windows), so the two runs are directly comparable.
  - Overall error is only 10-20% worse than the conditional model: pooled val MSE 0.0052 vs 0.0045; per ticker AMZN 0.0079 vs 0.0069, AAPL 0.0078 vs 0.0072, GOOG 0.0089 vs 0.0075, INTC 0.00066 vs 0.00056, MSFT 0.00052 vs 0.00047. Like EXP-003, it is worse than the static-book baseline for every ticker.
  - **Mid log-returns are lost for the wide-spread stocks.** On non-zero moves, correlation with the true return is -0.01 / 0.00 / -0.02 for AMZN / AAPL / GOOG (conditional: 0.74 / 0.37 / 0.56), and reconstructed moves are only 6-10% of their true size (conditional: 36-55%). Within-ticker explained variance of returns is about 0 (conditional: 0.09-0.39). INTC/MSFT keep some return information (correlation 0.76 / 0.86) but weaker than conditional (0.95 / 0.91).
  - Log sizes are barely captured, as in EXP-004 (within-ticker explained variance -0.21 to 0.12).
  - Ticker-swap test: identical error for every id, confirming the conditioning is fully off.
- **What went wrong / observations:**
  - The first attempt crashed on the T4 with `CUDA error: an illegal memory access was encountered`. `--cond-dim 0` then created a zero-width `nn.Embedding`, which works on CPU (where it had been tested) but not on CUDA. Fixed in f25f1ac by not creating an embedding at all when cond dim is 0; this run used the fix.
  - The pooled average hides the main difference: returns are 1 of 41 features and mostly zero, so losing them barely moves the overall MSE. In the autoencoder the ticker condition is therefore not just a label: without it, return dynamics for AMZN/AAPL/GOOG are not reconstructed at all. A plausible but untested explanation is that, with only 24 dimensions, the unconditional model must spend capacity identifying the stock from the features.
  - Both models still underfit (EXP-004), so this comparison is at the undersized configuration.
- **What was tried next:** the phase 1 capacity sweep (EXP-006). The conditional vs unconditional comparison could be repeated at the chosen size if needed for the README.

### EXP-006 — Phase 1 capacity sweep (hidden 24 / 64 / 164 x latent 24 / 41), Kaggle T4
- **Date:** 2026-10-29
- **Commit:** f9c3f5b (cloned from GitHub into a Kaggle notebook)
- **Model / representation:** conditional TimeGAN Embedder + Recovery (phase 1), unconstrained, global min-max (0.5/99.5% bounds, winsorised), all five tickers
- **Config:** six runs, hidden dim {24, 64, 164} x latent dim {24, 41}; otherwise as EXP-003: layers 3, cond dim 8, batch 128, seq len 24, stride 1, Adam lr 1e-3, 10,000 phase 1 iterations, seed 0. Phase 2 was cut to 1 iteration (`--sup-iterations 1`) because only phase 1 was being tested; the phase 2 numbers in these runs are meaningless. Run names `ae_sweep_h<H>_k<K>`.
- **Compute:** Kaggle notebook with 2 x NVIDIA Tesla T4 (15 GB each; the code uses only the first). Python 3.13.15, PyTorch 2.11.0+cu128 (CUDA 12.8), NumPy 2.1.3, pandas 2.3.3, matplotlib 3.10.0. This differs from the T4 VM's Python 3.10.21 / PyTorch 2.14.1+cu126.

  | Hidden / latent | Phase 1 time | Peak VRAM | E + R params |
  |---|---|---|---|
  | 24 / 24 | 66 s | 275 MB | 25,681 |
  | 24 / 41 | 65 s | 283 MB | 27,330 |
  | 64 / 24 | 68 s | 581 MB | 145,041 |
  | 64 / 41 | 66 s | 588 MB | 149,410 |
  | 164 / 24 | 391 s | 1,363 MB | 863,441 |
  | 164 / 41 | 393 s | 1,371 MB | 874,610 |

- **Result:** EXP-004 diagnostics on all six checkpoints, full validation set; raw numbers in `runs/ae_sweep_diagnostics.json`.
  - **Reproducibility check:** 24 / 24 repeats EXP-003 and gives val MSE 0.00450 vs 0.00450, on a different PyTorch version and machine.
  - **Against the static-book baseline** (each window predicted by its own mean): 24 / x loses for every ticker; 64 / 41 beats it for AMZN, AAPL, GOOG and MSFT; **164 / 41 beats it by 4.3x / 4.5x / 5.0x / 2.3x for AMZN / AAPL / GOOG / MSFT and ties for INTC** (0.00021 vs 0.00020).
  - **Within-ticker explained variance** (24 / 24 -> 64 / 41 -> 164 / 41): log sizes -0.07-0.14 -> 0.30-0.43 -> **0.57-0.89** (0.86-0.89 for AMZN/AAPL/GOOG); mid log-return 0.09-0.66 -> 0.46-0.95 -> **0.51-0.98**. Reconstructed size of non-zero return moves 36-73% -> 66-93% -> 61-98%. AAPL returns remain the weakest (0.51).
  - **Hidden size matters far more than latent size.** At hidden 24 a 41-dim latent barely helps; at hidden 164 it halves the error (val 0.00151 -> 0.00083) and raises size explained variance from 0.52-0.75 to 0.57-0.89.
  - **No overfitting:** mean train and val MSE over the last 1,000 iterations are equal for every configuration (164 / 41: 0.00098 / 0.00101).
  - **Hidden 164 had not converged:** its val MSE fell from 0.00211 to 0.00083 between iterations 7,500 and 10,000 and was still dropping steeply; hidden 24 had levelled off (EXP-003).
- **What went wrong / observations:**
  - The EXP-003/004 underfitting was mainly a capacity problem: the networks (hidden 24), more than the latent bottleneck, were too small.
  - Cost rises sharply: hidden 164 is about 6x slower in phase 1 (391 s vs 66 s), uses 1.37 GB of VRAM for E + R alone, and has 34x the parameters. With all five networks in phase 3, a full TimeGAN run at this size will probably exceed feasibility criterion 5 (< 30 min, < 2 GB on one T4); to be measured and reported.
  - The original T4 VM was unavailable, so this sweep ran on Kaggle; its PyTorch version differs, so timings are not strictly comparable with EXP-002/003/005.
- **What was tried next:** adopt hidden 164 / latent 41 for TimeGAN. Hidden 164 is the paper's 4 x features; latent 41 departs from the paper's ½ x features ≈ 20, justified by this sweep. RCGAN must use the same hidden size so both keep the same backbone. Candidates: a 20,000-iteration phase 1 at 164 / 41 to check the gain from training longer; moving the diagnostics into the repo; then phase 2 at the new size.

### EXP-007 — TimeGAN phases 1 + 2 at hidden 164 / latent 41 (20,000 + 10,000 iterations), Kaggle T4
- **Date:** 2026-10-30
- **Commit:** 52175f2 (`train.py` identical to f9c3f5b; phase 3 not yet implemented, so the run stopped after phase 2)
- **Model / representation:** conditional TimeGAN Embedder + Recovery (phase 1), then Supervisor on frozen embeddings (phase 2); unconstrained, global min-max (0.5/99.5% bounds, winsorised), all five tickers
- **Config:** hidden 164, latent 41 (chosen in EXP-006), layers 3 (supervisor 2), cond dim 8, batch 128, seq len 24, stride 1, Adam lr 1e-3, **20,000 phase 1 iterations** (double EXP-006, because its loss was still falling at 10,000), 10,000 phase 2 iterations, seed 0; run name `timegan_h164_k41_p12`
- **Compute:** Kaggle T4, same environment as EXP-006 (Python 3.13.15, PyTorch 2.11.0+cu128). Phase 1: 782 s (0.039 s/iteration), peak VRAM 1,371 MB. Phase 2: 261 s (0.026 s/iteration), peak VRAM 1,135 MB. Params E 437,305 / R 437,305 / S 274,945.
- **Result:** diagnostics on the full validation set (17,885 windows), saved to `runs/timegan_h164_k41_p12/diagnostics.json`.
  - **Phase 1: training longer paid off.** Val MSE kept falling well past 10,000 iterations (0.00083 at 10k, 0.00030 at 12.5k, 0.00016 at 15k) and was levelling off by 20,000 (0.00011 at 19k, **0.00010** at 20k). Per ticker it is now **18-40x better than the static-book baseline** (AMZN 0.00013 vs 0.0052; INTC 0.00003 vs 0.0002), compared with at most 5x at 10,000 iterations (EXP-006). Within-ticker explained variance: log sizes 0.94-0.99, price offsets 0.88-0.95, mid log-return 0.96-1.00; non-zero return moves reconstructed with correlation 0.99-1.00 at 95-102% of their true size. Train and val MSE stay together (last 1,000 iterations 0.00014 / 0.00011): no overfitting.
  - **Phase 2: the supervisor beats persistence for every ticker**, by 1.12x (AMZN), 1.20x (AAPL), 1.13x (GOOG), 1.53x (INTC) and 1.49x (MSFT) (val next-step MSE vs "next latent = current latent"). In the small-model test (hidden 24, 600 iterations) it had been ~4x *worse* than persistence for INTC/MSFT. Val MSE fell quickly (0.0123 at iteration 1, 0.0022 at 500) and reached its best of 0.00196 at iteration 4,450.
- **What went wrong / observations:**
  - Phase 2 starts to overfit after ~4,500 iterations: train MSE keeps falling (0.00239 at 2-3k -> 0.00221 at 9-10k) while val drifts up slightly (0.00197 -> 0.00199). The effect is small, but ~5,000 phase 2 iterations would be enough.
  - The supervisor's margin over persistence is modest for AMZN/AAPL/GOOG (12-20%): most of what can be predicted one second ahead is "the book stays roughly the same".
  - With latent 41 = number of features and near-perfect reconstruction, the latent space is not compressed; TimeGAN's "compact latent space" argument (paper Section 4.1) therefore carries less weight here, and the latent is close to a smooth re-coding of the features. Worth noting in the README when comparing with RCGAN.
  - Persistence MSE (0.0024 pooled) is not comparable with earlier runs, because each run learns a different latent space; only the supervisor-to-persistence ratio is comparable.
  - Phase 1 at this length is the most expensive phase so far (13 min); every TimeGAN run currently retrains phases 1 and 2 from scratch.
- **What was tried next:** phase 3 (joint training) implemented in 4dbdbd9; first a short phase 3 smoke test at this size to measure time per iteration and VRAM before a full run. Settings for later runs: 20,000 phase 1 iterations, ~5,000 phase 2 iterations.

### EXP-008 — TimeGAN phase 3 smoke test at hidden 164 / latent 41, Kaggle T4
- **Date:** 2026-10-30
- **Commit:** 4dbdbd9 (first version of phase 3 joint training)
- **Model / representation:** full conditional TimeGAN (all five networks), unconstrained, global min-max (0.5/99.5% bounds, winsorised), all five tickers
- **Config:** hidden 164, latent 41, layers 3 (supervisor 2), cond dim 8, z dim 41 (U(0,1)), batch 128, seq len 24, stride 1, Adam lr 1e-3. Phases 1 and 2 deliberately short (1,000 and 500 iterations), because this run only measures phase 3's cost and early behaviour; phase 3: 300 iterations, 2 G steps per D step, D threshold 0.15, reference-code loss weights (gamma 1, 100·sqrt(L_S), 100·L_V, 0.1·L_S for E); seed 0; run name `timegan_h164_k41_p3smoke`
- **Compute:** Kaggle T4, same environment as EXP-006/007. Phase 3: 143 s = **0.48 s/iteration**, **peak VRAM 526 MB**. Phases 1/2: 46 s / 14 s. Params E 437,305 / R 437,305 / S 274,945 / G 437,305 / D 1,184,121 (2.77M in total).
- **Result:** phase 3 runs end to end on the GPU with all outputs saved; no NaN or infinite losses. Mean losses per 100 iterations stay roughly flat: D loss 1.76-1.85, G adversarial loss 1.11-1.23 (on supervised fakes) and 1.28-1.50 (on raw fakes), supervised loss ~0.003, moment loss 0.027-0.030, reconstruction MSE ~0.006. Far too short to judge sample quality.
- **What went wrong / observations:**
  - **Cost:** at 0.48 s/iteration, 10,000 phase 3 iterations would take ~80 min; with phase 1 at 20,000 iterations (13 min, EXP-007), a full run is far above feasibility criterion 5 (< 30 min). Each phase 3 iteration runs two G+S steps and two E+R steps through all five networks, plus a D step, at hidden 164.
  - **VRAM is not a problem:** phase 3 peaks at 526 MB. The 1,371 MB reported for phase 1 comes from computing validation loss on 2,048 windows in one batch, not from training.
  - **Discriminator behaviour:** D classifies almost everything as fake (accuracy on real 0.09-0.23, on fake 0.94-0.96, per 100 iterations). Its loss never comes near the 0.15 threshold, so it was updated on 100% of iterations and the threshold never acted. With weak phases 1/2 and only 300 iterations this is not yet a failure, but to be watched in longer runs.
  - Phases 1 and 2 were re-run from scratch only for this test; every full run would otherwise repeat ~17 min of phases 1 and 2.
- **What was tried next:** `--init-from` (3b22cd0) to start phase 3 from the trained EXP-007 checkpoint; then a 2,000-iteration phase 3 run (~16 min) before any 10,000-iteration run, and a `predict.py` to evaluate generated books.

### EXP-009 — TimeGAN phase 3, 2,000 iterations from the EXP-007 checkpoint, Kaggle T4
- **Date:** 2026-10-30
- **Commit:** 3b22cd0 (`--init-from`; training code unchanged up to e60f560)
- **Model / representation:** full conditional TimeGAN, unconstrained, global min-max (0.5/99.5% bounds, winsorised), all five tickers
- **Config:** `--init-from runs/timegan_h164_k41_p12` (phases 1 and 2 from EXP-007: 20,000 + 10,000 iterations), then **2,000 phase 3 iterations**; hidden 164, latent 41, layers 3 (supervisor 2), cond dim 8, z dim 41 (U(0,1)), batch 128, seq len 24, Adam lr 1e-3 for all networks, 2 G steps per D step, D threshold 0.15, reference-code loss weights (gamma 1, 100·sqrt(L_S), 100·L_V, 0.1·L_S for E), seed 0; run name `timegan_h164_k41_p3_2k`
- **Compute:** Kaggle T4, same environment as EXP-006/007. Phase 3: 949 s (15.8 min) = 0.47 s/iteration; peak VRAM 526 MB. Params G 437,305 / D 1,184,121.
- **Result:** training ran to completion with no NaN or infinite losses, but **the discriminator never learned**, so there was effectively no adversarial training.
  - Mean losses per 250 iterations were flat for the whole run: D loss 1.86-1.89, G adversarial loss 1.10-1.15 (supervised fakes) and 1.15-1.20 (raw fakes). D's accuracy was 3-7% on real and 98-100% on fake latents; it was updated on 100% of iterations because its loss never approached the 0.15 threshold.
  - These values are exactly those of a discriminator that ignores its input. D's loss has one "real" term and two "fake" terms, so the best *constant* logit is log(1/2) ≈ -0.69, giving D loss ln 3 + 2 ln 1.5 ≈ 1.91 and G loss ln 3 ≈ 1.10. Post-hoc check on the checkpoint (2,048 validation windows): the trained D gives mean logits -0.72 (real), -0.69 (supervised fake) and -0.77 (raw fake), each with std ≤ 0.11, i.e. P(real) ≈ 1/3 for everything.
  - The generated latents *are* distinguishable: a fresh discriminator trained for 400 steps on train-split latents separates real from supervised-fake latents on held-out validation windows with 99% balanced accuracy (hidden 24, 2 layers; a quick version of the paper's post-hoc discriminative score, ≈ 0.49 where 0 is best).
  - What did train: the moment loss fell from 0.022 to 0.017-0.018, the supervised loss from 0.0026 to 0.0021, and the reconstruction MSE stayed at ~0.0001 (E/R kept their EXP-007 quality). So the generator was shaped mainly by the moment loss and by passing through the supervisor, not by the discriminator.
- **What went wrong / observations:**
  - **Likely cause: the learning rate is too high for a discriminator of this size.** In an offline test on the same real and generated latents (300 Adam steps, batch 128), a fresh hidden-164 discriminator at lr 1e-3 only reached loss 0.96 (held-out balanced accuracy 0.90), whereas at lr 1e-4 it reached loss 0.008 and 0.999. A hidden-24 discriminator at lr 1e-3 learned (loss 0.14). The reference code's lr 1e-3 was tuned for its small (hidden 24) networks. This offline test uses fixed latents, so it suggests rather than proves the cause inside the GAN loop.
  - The same pattern was already visible in the EXP-008 smoke test (D accuracy on real 0.09-0.23), so it is not caused by `--init-from`.
  - The D-loss threshold (0.15) and the 2:1 schedule never acted: they only help when D is too *strong*.
  - At 0.47 s/iteration, the 10,000 iterations of the reference schedule would take ~80 min.
- **What was tried next:** rerun phase 3 from the same checkpoint with a lower learning rate (1e-4), watching D's real/fake accuracy; if D then dominates, the threshold and 2:1 schedule become relevant. Any learning-rate change must also be applied to the RCGAN baseline for a fair comparison.

### EXP-010 — TimeGAN phase 3 at lr 1e-4 (2,000 iterations from EXP-007), Kaggle T4
- **Date:** 2026-10-30
- **Commit:** training code as in 3b22cd0 (unchanged up to 3562a18)
- **Model / representation:** full conditional TimeGAN, unconstrained, global min-max (0.5/99.5% bounds, winsorised), all five tickers
- **Config:** identical to EXP-009 except **Adam lr 1e-4** for all networks (G, S, E, R, D): `--init-from runs/timegan_h164_k41_p12`, 2,000 phase 3 iterations, hidden 164, latent 41, batch 128, 2 G steps and 1 D step per iteration, D threshold 0.15, reference-code loss weights, seed 0; run name `timegan_h164_k41_p3_2k_lr1e-4`
- **Compute:** Kaggle T4, same environment as EXP-006 to 009. Phase 3: 950 s = 0.47 s/iteration; peak VRAM 526 MB.
- **Result:** **the discriminator is still stuck**; lowering the learning rate did not fix EXP-009.
  - Mean losses per 250 iterations stay at the constant-output values: D loss 1.82-1.92, G adversarial loss 1.07-1.17. D accuracy on real latents 1-15%, on fake 98-100%; D updated on 100% of iterations. Same pattern as EXP-009 (lr 1e-3).
  - What did train: moment loss 0.021 -> 0.017, supervised loss 0.0023 -> 0.0017, reconstruction MSE ~0.0001.
  - Generated latents are less varied than real ones (on 4,096 training windows): std 0.081 for S(G(z)) vs 0.100 for real E(x), a mild sign of collapse.
- **What went wrong / observations:** offline checks on this checkpoint (4,096 fixed training-split latents, Adam lr 1e-4, batch 128, the loop's 3-term D loss) show that D *can* learn, so the cause is not the learning rate, a bug in D, or a saturated network:
  - a **fresh** hidden-164 D reached loss 1.65 / real accuracy 0.15 after 100 steps and 0.60 / 0.74 after 300;
  - **this run's own trained D**, trained further on the same fixed latents, reached 1.05 / 0.57 after 100 steps and 0.78 / 0.73 after 300; its first GRU layer is far from saturation (mean |input pre-activation| 0.25, none above 5) and its first-step gradient norm is small (0.28), consistent with sitting on a flat region.
  - (These accuracies are measured on latents from the training batches, so they show D's ability to learn, not generalisation; EXP-009's held-out check showed 99% for a small D.)
  - The constant-output state is a slow plateau even on fixed targets (100-300 updates to leave it). In the loop, D gets one update per iteration while G and E each get two, so both the fake and the real latents move under it and it never escapes. The reference schedule (2 G steps per D step, skip D when its loss is below 0.15) is designed for a discriminator that is too *strong*, so here it works against training. This explanation fits all the evidence but is not yet confirmed inside the loop.
  - Neither EXP-009 nor EXP-010 is a valid test of TimeGAN's adversarial training, because the adversarial part never worked.
- **What was tried next:** `--d-steps` (discriminator updates per iteration; 1 reproduces the reference code exactly) added to both the TimeGAN and RCGAN loops and copied by `scripts/train_rcgan_matched.sh`; next run: phase 3 from the same checkpoint with more D steps per iteration.

### EXP-011 — TimeGAN phase 3 with 5 D steps per iteration (lr 1e-4, 2,000 iterations from EXP-007), Kaggle T4
- **Date:** 2026-10-30
- **Commit:** 13f04bb (`--d-steps`)
- **Model / representation:** full conditional TimeGAN, unconstrained, global min-max (0.5/99.5% bounds, winsorised), all five tickers
- **Config:** as EXP-010 but **`--d-steps 5`**: per iteration 2 G+S / E+R steps and 5 D steps (each on a fresh batch); `--init-from runs/timegan_h164_k41_p12`, 2,000 phase 3 iterations (10,000 D updates in total), Adam lr 1e-4 for all networks, hidden 164, latent 41, batch 128, D threshold 0.15, reference-code loss weights, seed 0; run name `timegan_h164_k41_p3_2k_lr1e-4_d5`
- **Compute:** Kaggle T4, same environment as EXP-006 to 010. Phase 3: 1,963 s (32.7 min) = **0.98 s/iteration** (about double EXP-010, from the extra D steps); peak VRAM 526 MB.
- **Result:** a small improvement, but the discriminator still does not learn to separate real from generated latents.
  - Mean D loss per 200 iterations 1.71-1.83 (EXP-010: 1.82-1.92; constant-output value 1.91); D accuracy on real latents 8-18% (EXP-010: 1-15%), on fake 93-100%; every D step applied (loss never below 0.15). G adversarial loss 1.03-1.20 (supervised fakes) and 1.24-1.70 (raw fakes), slightly above the constant-output value ln 3 ≈ 1.10, so D now gives fakes slightly lower scores than before.
  - Other losses as in EXP-010: moment 0.023 -> 0.017, supervised 0.0024 -> 0.0018, reconstruction ~0.0001.
- **What went wrong / observations:**
  - **EXP-010's explanation was incomplete.** D received 10,000 updates here, far more than the 100-300 it needed in the offline tests, so "too few D updates on stable targets" cannot be the whole cause.
  - **The generator undoes what D learns.** Offline test on this checkpoint (D and G+S trained in turns with the loop's exact losses, lr 1e-4, batch 128; separation measured on 512 held-out training windows as the logit gap mean D(real) - mean D(S(G(z))), 0 = no separation):

    | Step | Logit gap | Balanced accuracy* |
    |---|---|---|
    | checkpoint as saved | +0.26 | 0.51 |
    | after 50 D steps (G frozen) | +1.98 | 0.66 |
    | after 150 D steps (G frozen) | +2.14 | 0.68 |
    | then 1 G step (D frozen) | +2.09 | 0.63 |
    | then 2 G steps | +1.59 | 0.60 |
    | then 5 G steps | +1.09 | 0.55 |
    | then 20 G steps | +0.70 | 0.54 |

    \*threshold at the median generated logit, so a rough measure; the logit gap is the reliable number.

    D builds separation at about +0.04 per update, while each generator update removes about 0.25: one G step undoes roughly what 5-10 D steps build. With 2 G steps per 5 D steps the generator still wins every round, so D never holds a lasting advantage.
  - This also corrects an earlier claim (EXP-009/010 discussion): the generator does *not* receive "almost no" adversarial gradient. It receives enough to escape whatever direction D has just learned, yet a fresh discriminator still separates its output from real data (EXP-009: 99%). The generator chases the current discriminator's weak spots instead of matching the real distribution, a classic GAN non-convergence pattern. The reduced variety of generated latents (EXP-010) fits this: nothing in the current loop rewards diversity.
  - The imbalance is per update, in favour of the generator; more generator steps would make it worse.
- **What was tried next:** `--d-lr` (separate discriminator learning rate, default `--lr`), so D can learn on a faster time scale than G, the two time-scale update rule (Heusel et al., 2017); also copied by `scripts/train_rcgan_matched.sh`. Next run: e.g. `--lr 1e-5 --d-lr 1e-4 --d-steps 5` from the same checkpoint (note that `--lr` also slows E, R and S, which are only fine-tuned in phase 3).

### EXP-012 — TimeGAN phase 3 with a faster discriminator (TTUR) + first full benchmark, Kaggle T4
- **Date:** 2026-10-30
- **Commit:** 8065fb4 (`--d-lr`)
- **Model / representation:** full conditional TimeGAN, unconstrained, global min-max (0.5/99.5% bounds, winsorised), all five tickers
- **Config:** as EXP-011 but with the two time-scale update rule (Heusel et al., 2017): **`--lr 1e-5` for G, S, E and R, `--d-lr 1e-4` for D**, `--d-steps 5`, 2 G steps per iteration; `--init-from runs/timegan_h164_k41_p12`, 2,000 phase 3 iterations, hidden 164, latent 41, batch 128, D threshold 0.15, reference-code loss weights, seed 0; run name `timegan_h164_k41_p3_2k_ttur`. Benchmark: `predict.py` with defaults (32 sequences x 500 steps per ticker, scored against the real test hour; reference = real validation hour vs test hour, 7 segments of 500 steps).
- **Compute:** training on the Kaggle T4: phase 3 1,961 s (32.7 min) = 0.98 s/iteration, peak VRAM 783 MB; with phases 1 and 2 (EXP-007, 17.4 min) the full model took ~50 min. Generation (on a CPU): 2.2 s for all 160 sequences = 73 sequences/s.
- **Result (training):** **the adversarial game works for the first time.** Per 200 iterations, D loss fell from 1.46 to 0.41-0.70 (constant-output value 1.91) and D's accuracy on real latents rose from 39% to 78-88% while staying at 94-99% on fakes. G's adversarial loss rose from 1.05 to 2.5-3.0, i.e. G now gets real feedback and D is ahead. The D threshold acted for the first time (99.3% of D steps applied; some skipped around iterations 600-800). Costs of the slower generator: the supervised loss stayed at 0.0025-0.0029 (EXP-011: 0.0018) and the moment loss at 0.021-0.024 (EXP-011: 0.017). Reconstruction ~0.0001. No NaNs.
- **Result (generated books): poor, and far from the acceptance criteria.**

  | Ticker | Violations (crossed / ladder) | Spread ticks gen / real | Return std gen / real | ACF r lag 1 gen / real | ACF \|r\| lag 1 / 60 gen (real lag 1) | KL spread gen (ref) | KL mid change gen (ref) | SSIM gen (ref) |
  |---|---|---|---|---|---|---|---|---|
  | AMZN | 0% / **80%** | 15.3 / 9.9 | 2.4e-6 / 8.9e-5 | +0.90 / +0.07 | 0.90 / 0.60 (0.13) | 16.0 (0.26) | 5.7 (0.17) | 0.20 (0.10) |
  | AAPL | 0% / 0% | 7.3 / 11.1 | 3.6e-6 / 6.3e-5 | +0.90 / +0.09 | 0.91 / 0.10 (0.11) | 12.2 (0.12) | 8.2 (0.14) | 0.06 (0.04) |
  | GOOG | 0% / **26%** | 26.0 / 21.6 | 1.7e-6 / 6.0e-5 | +0.80 / +0.02 | 0.80 / 0.10 (0.10) | 12.7 (0.14) | 19.1 (0.17) | 0.18 (0.12) |
  | INTC | 0% / 0% | 1.00 / 1.01 | 6.6e-6 / 8.1e-5 | +0.96 / +0.01 | 0.96 / 0.66 (0.04) | 0.10 (0.00) | 0.81 (0.02) | 0.26 (0.27) |
  | MSFT | 0% / 0% | 1.00 / 1.01 | 4.6e-6 / 9.0e-5 | +0.93 / -0.01 | 0.94 / 0.61 (0.05) | 0.21 (0.00) | 1.18 (0.04) | 0.46 (0.43) |

  - **Smooth deterministic drift instead of price dynamics.** The VWAP-path figure shows each generated mid moving in a smooth, almost straight line (AMZN -$1.1, GOOG -$4.6, AAPL +$4.8 over 500 s), with the three plotted samples per ticker practically on top of each other. Returns are 13-35x smaller in standard deviation than real, have excess kurtosis of -0.9 to 4.4 (real 18-41) and lag-1 autocorrelation of +0.80 to +0.96 (real about 0). The high ACF of |returns| is therefore an artefact of the steady drift, not volatility clustering. Within-sequence VWAP variance is 3-23x too high for the same reason.
  - **The generator largely ignores its noise:** different noise gives almost the same sequence (mode collapse to one trajectory per ticker).
  - **Static books:** depth heatmaps are smooth horizontal bands close to each ticker's average book, with none of the real order-flow texture.
  - **Validity degrades with horizon:** no crossed books and no negative sizes anywhere, but ladder violations for AMZN and GOOG rise from 1% / 0% in the first 100 steps to 100% in the last 100, as the drifting offsets eventually break the ordering.
  - **Prices are off the tick grid:** mean distance from the nearest tick 0.23-0.25 ticks (real 0; uniformly random prices 0.25).
  - **What works:** the ticker condition separates the two market regimes. INTC and MSFT have a one-tick spread on 100% of steps (real 99%) and AMZN / AAPL / GOOG keep wide spreads (7-26 ticks), so there is no ticker collapse, though AAPL's spread is too narrow and AMZN's too wide.
  - **Spec metrics:** KL is 15-100x the real-vs-real reference, mainly because continuous, drifting prices never reproduce the tick-discrete spread and mid-change distributions. SSIM is *at or above* the real-vs-real reference for every ticker even though the books are clearly unrealistic: smooth average heatmaps resemble any real heatmap at least as much as two real hours resemble each other. This is direct evidence for the README's critique of SSIM.
  - **Against the acceptance criteria:** 1 (violations < 1%) fails for AMZN and GOOG; 2 (conditioning) partly met (regimes distinct, levels off); 3 (volatility clustering) fails; 4 reported, far from reference; 5 (< 30 min) fails (~50 min).
  - Compared with EXP-009 (same benchmark, constant D): EXP-009's returns jittered from step to step (lag-1 ACF -0.17 to -0.74) around a drift; here they are smooth. Ladder violations: AMZN 66% -> 80%, GOOG 60% -> 26%, AAPL 6% -> 0%. Neither run reproduces real price dynamics.
- **What went wrong / observations (hypotheses, not yet tested):**
  - **Long horizon:** training windows are 24 steps, generation is 500. A small constant bias in the generated return, compounded over 500 steps, gives exactly this kind of steady drift; validity is fine early and breaks late. Benchmarking at 24 steps would separate "cannot generate" from "cannot extrapolate"; the planned 128-step training windows are directly relevant.
  - **Heavy tails removed by preprocessing:** returns are winsorised at the 0.5/99.5% training quantiles, so the largest moves never appear in the training data, and a sigmoid generator cannot exceed them. This alone limits kurtosis (the open decision from the normalisation step).
  - **Undertrained generator:** at lr 1e-5 for 2,000 iterations, G (and S) moved little; its adversarial feedback only became meaningful in this run, and D is ahead at the end.
  - **No reward for diversity:** the moment loss matches only per-step means and spreads across the batch, and a discriminator that is ahead but sees one sequence at a time does not directly penalise every sample being the same.
  - **Continuous prices:** the unconstrained decoder does not round to the tick grid; tick-aware decoding (or the constrained representation for the ladder) would address the off-grid prices and part of the KL.
- **Follow-up: horizon check.** The same checkpoint benchmarked at its 24-step training length (`predict.py --seq-len 24 --n-samples 128`, CPU; the real-vs-real reference is then cut into 24-step segments, so these numbers are not comparable with the 500-step table): **0% violations for every ticker** (500 steps: AMZN 80%, GOOG 26%), but return std still 7-35x smaller than real (e.g. AMZN 2.7e-6 vs 8.9e-5). So the broken ladders are a long-horizon effect, while the damped price dynamics are already present within the training length.
- **What was tried next:** one final run set, with a longer training window (targets the horizon failure) and a faster generator learning rate (targets undertraining), followed by the RCGAN baseline trained with identical settings (`scripts/train_rcgan_matched.sh`). Return clipping was left unchanged so the final run changes as little as possible.

### EXP-013 — Final TimeGAN: 64-step windows, TTUR, trained from scratch, Kaggle T4
- **Date:** 2026-10-30
- **Commit:** 50e3dbf
- **Model / representation:** full conditional TimeGAN, unconstrained, global min-max (0.5/99.5% bounds, winsorised), all five tickers
- **Config:** **seq len 64** (80,680 / 17,685 / 17,685 train/val/test windows), hidden 164, latent 41, layers 3 (supervisor 2), cond dim 8, z dim 41 (U(0,1)), batch 128; 20,000 phase 1 + 5,000 phase 2 + 2,000 phase 3 iterations; **`--lr 3e-5`** (G, S, E, R) and **`--d-lr 1e-4`** (D); 2 G steps and 5 D steps per iteration, D threshold 0.15, reference-code loss weights, seed 0; run name `final_timegan`. Benchmark: `predict.py` on the GPU, 32 x 500-step sequences per ticker against the test hour, plus the same at the 64-step training length (`eval_64`).
- **Compute:** Kaggle T4 (Python 3.13.15, PyTorch 2.11.0+cu128). Phase 1 1,931 s (0.097 s/it, peak VRAM 3,509 MB), phase 2 319 s (0.064 s/it, 2,916 MB), phase 3 4,980 s (2.49 s/it, 1,213 MB): **120.5 min in total**. Params E 437,305 / R 437,305 / S 274,945 / G 437,305 / D 1,184,121. The phase 1 and 2 peaks come mostly from computing validation loss on 2,048 64-step windows in one batch. Generation: 0.84 s for 160 x 500 steps (190 sequences/s, 224 MB).
- **Result (training):**
  - **Phase 1 underfitted: val reconstruction 0.0045 after 20,000 iterations** (EXP-007: 0.0001), per ticker only 1.1-1.3x better than each ticker's own mean (AMZN 0.0069 vs 0.0088), i.e. back at EXP-003/004 quality. Cause: `--lr` sets the learning rate of every phase, so phases 1 and 2 ran at 3e-5 instead of the 1e-3 used in EXP-007 (33x slower). The intent was to slow only the phase 3 fine-tuning of G/S/E/R; that is only true with `--init-from`. This was an error in the command (suggested by the AI assistant, not caught before the run).
  - Phase 2: supervisor beats persistence for every ticker by 1.8-2.6x (val 0.00119 vs 0.00218 pooled).
  - Phase 3: adversarial training active throughout. Per 250 iterations, D loss 0.89-1.53, D accuracy on real 33-74% and on fake 84-95%, G adversarial loss 1.38-2.04; every D step applied. Moment loss 0.041 -> 0.020, reconstruction MSE ~0.0047. No NaNs.
- **Result (generated books, 500 steps):**

  | Ticker | Violations | Spread ticks gen / real | VWAP var gen / real ($²) | VWAP-mid ticks gen / real | Return std gen / real | ACF r lag 1 gen / real | Kurtosis gen / real | KL spread (ref) | KL mid (ref) | SSIM (ref) |
  |---|---|---|---|---|---|---|---|---|---|---|
  | AMZN | **0.0%** | 16.5 / 9.9 | 0.023 / 0.023 | -0.64 / -1.41 | 8.2e-6 / 8.9e-5 | +0.61 / +0.07 | 47.6 / 41.4 | 15.5 (0.26) | 2.9 (0.17) | 0.14 (0.10) |
  | AAPL | **0.0%** | 12.1 / 11.1 | 0.052 / 0.081 | -2.44 / -1.77 | 7.8e-6 / 6.3e-5 | +0.79 / +0.09 | 67.4 / 38.5 | 12.6 (0.12) | 6.4 (0.14) | 0.11 (0.04) |
  | GOOG | **0.0%** | 24.1 / 21.6 | 0.049 / 0.125 | -0.26 / -3.04 | 6.5e-6 / 6.0e-5 | +0.92 / +0.02 | 2.0 / 18.5 | 13.3 (0.13) | 3.3 (0.17) | 0.19 (0.12) |
  | INTC | **0.0%** | 1.00 / 1.01 | 0.018 / 0.0004 | -0.42 / -0.34 | 7.4e-6 / 8.1e-5 | +0.93 / +0.01 | 7.6 / 23.8 | 0.10 (0.00) | 0.81 (0.02) | 0.34 (0.27) |
  | MSFT | **0.0%** | 1.00 / 1.01 | 0.044 / 0.0003 | -0.01 / +0.03 | 1.0e-5 / 8.9e-5 | +0.81 / -0.01 | 6.8 / 20.2 | 0.21 (0.00) | 1.11 (0.04) | 0.54 (0.43) |

  - **No invariant violations at all**, for every ticker and at every horizon (0% in both the first and the last 100 steps; also 0% at 64 steps). INTC/MSFT one-tick spread on 100% of steps (real 99%); AMZN/AAPL/GOOG wide spreads, AAPL and GOOG close to real, AMZN 67% too wide. VWAP variance matches real for AMZN, is 36-61% too low for AAPL/GOOG and 40-140x too high for INTC/MSFT (drift).
  - **Still no realistic price dynamics:** mid prices follow smooth curves (VWAP-path figure), returns are 8-11x too small in standard deviation with lag-1 ACF of +0.61 to +0.93, and different noise gives near-identical paths. The ACF of |returns| (0.79-0.95 at lag 1) is again an artefact of smooth drift. Prices are off the tick grid (0.25 ticks). KL is 15-100x the reference; SSIM at or above it for every ticker.
- **What went wrong / observations:**
  - The learning-rate slip means the final TimeGAN is confounded: its latent space is much weaker than EXP-007's, and every generated book is decoded through that weaker recovery network. Several changes happened at once compared with EXP-012 (window 24 -> 64, G lr 1e-5 -> 3e-5, weaker autoencoder, trained from scratch), so the improvement in validity cannot be attributed to one of them. The 24-step check in EXP-012 points to the longer window for the long-horizon ladder breakage.
  - At 120 min and a measured 3.5 GB peak, the run fails feasibility criterion 5 (< 30 min, < 2 GB), though phase 3 training itself peaked at 1.2 GB.
- **What was tried next:** the matched RCGAN baseline (EXP-014). Not done for lack of time: phases 1 and 2 at lr 1e-3 with phase 3 at the final settings (would remove the confound), and repeated seeds.

### EXP-014 — Final RCGAN baseline, settings matched to EXP-013, Kaggle T4
- **Date:** 2026-10-30
- **Commit:** 50e3dbf (`scripts/train_rcgan_matched.sh runs/final_timegan`)
- **Model / representation:** conditional RCGAN-style baseline (generator and discriminator on features), unconstrained, global min-max, all five tickers
- **Config:** every shared setting copied from EXP-013's `config.json`: seq len 64, hidden 164, layers 3, cond dim 8, z dim 41, batch 128, 2,000 iterations (= TimeGAN phase 3), `--lr 3e-5`, `--d-lr 1e-4`, 2 G steps and 5 D steps per iteration, D threshold 0.15, seed 0; run name `rcgan_matched_final_timegan`. Benchmark as EXP-013.
- **Compute:** Kaggle T4, same environment. 2,053 s (34.2 min) = 1.03 s/iteration, peak VRAM 1,150 MB. Params G 437,305 / D 1,184,121 (identical to TimeGAN's G and D). Generation 0.36 s for 160 x 500 steps (448 sequences/s).
- **Result (training):** **the discriminator dominated from the start.** Per 250 iterations, D loss 0.075-0.18, D accuracy 96-99% on real and 99-100% on fake; G loss 3.2-3.6. Because D's loss was mostly below the 0.15 threshold, only **10.7% of D steps were applied** (the threshold doing its job). No NaNs.
- **Result (generated books):** unusable for every ticker.
  - **100% ladder violations**, from the first step to the last (no crossed books); also 100% at 64 steps.
  - Spreads far too wide: 35-93 ticks for AMZN/AAPL/GOOG (real 10-22) and 4-5 ticks for INTC/MSFT (real 1; one-tick share 0%), so the regime difference is only partly kept. VWAP sits 36-95 ticks *above* the mid (real -3 to 0).
  - Inspected book (AAPL, step 250): 2-7 shares at most levels and one order of ~3,800 shares at the third ask level; bid prices out of order; the mid had drifted from $581.9 to about $626 (+$44). Mid paths are smooth and near-identical across noise (lag-1 return ACF +0.96-0.97, return std 65-99x too small).
  - Depth heatmaps are almost empty except one band; SSIM is negative or near 0 (-0.04 to 0.04). KL 18-22.
- **What went wrong / observations:**
  - With TimeGAN's settings, the baseline's discriminator (which sees raw features rather than learned latents) wins easily, and the adversarial loss alone gives the generator no route to valid books. The settings were tuned (EXP-009 to 012) for TimeGAN's latent-space discriminator, and the faster D learning rate (TTUR) suited that case; for the baseline it makes D overwhelming. The comparison is fair in the sense of identical settings and tuning effort, but not tuned separately for the baseline, which is a limitation to report.
  - Comparison with EXP-013 on identical splits and settings: TimeGAN 0% violations vs RCGAN 100%; TimeGAN keeps the one-tick regime for INTC/MSFT, RCGAN does not; neither reproduces realistic returns.
- **What was tried next:** results written up in the README (Sections 7, 10 and 11).
