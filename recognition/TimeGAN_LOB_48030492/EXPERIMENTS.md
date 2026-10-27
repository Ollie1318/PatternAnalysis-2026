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
- **What went wrong / observations:** CPU and GPU runs with the same seed are not bit-identical. CPU and cuDNN GRU kernels round floating-point operations differently, and the adversarial feedback loop amplifies these ~1e-5 differences over iterations. This is expected rather than a bug: same-device reruns are reproducible (verified on CPU in EXP-001), but results must be compared within one device, and all reported results should come from the GPU. Package versions on the VM differ from the laptop (Python 3.10 cannot install pandas 3.x / latest NumPy); exact versions to be recorded with `pip freeze`.
- **What was tried next:** resource figures feed the feasibility review: the model is tiny relative to the hardware, so a single T4 is ample and multi-GPU (`torchrun`) is not needed.
