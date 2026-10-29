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
- **Compute:** Kaggle notebook, NVIDIA Tesla T4, PyTorch 2.11.0+cu128 (a different version from the T4 VM's 2.14.1+cu126; other package versions to be recorded).

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
  - The diagnostics script is not yet in the repo.
- **What was tried next:** adopt hidden 164 / latent 41 for TimeGAN. Hidden 164 is the paper's 4 x features; latent 41 departs from the paper's ½ x features ≈ 20, justified by this sweep. RCGAN must use the same hidden size so both keep the same backbone. Candidates: a 20,000-iteration phase 1 at 164 / 41 to check the gain from training longer; moving the diagnostics into the repo; then phase 2 at the new size.
