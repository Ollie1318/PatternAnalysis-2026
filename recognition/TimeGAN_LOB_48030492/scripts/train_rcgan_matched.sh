#!/usr/bin/env bash
# Train the RCGAN-style baseline with exactly the settings of a TimeGAN run,
# then benchmark it with predict.py.
#
# Fairness (CLAUDE.md "Main baseline"): the baseline must get the same data
# preparation, GRU backbone, optimiser, learning rate, stabilisation tricks
# (G and D steps per iteration, D-loss threshold) and seed as TimeGAN, so that any
# difference in results comes from TimeGAN's extra components (embedder,
# recovery, supervisor, joint training), not from tuning. Copying the settings
# from the TimeGAN run's own config.json removes the chance of a mismatch.
#
# The baseline gets as many adversarial iterations as TimeGAN's phase 3
# (--iterations). TimeGAN additionally spends its phase 1 and 2 iterations on
# pre-training; report both iteration counts and wall-clock times in the README.
#
# Usage (from anywhere):
#   scripts/train_rcgan_matched.sh <timegan-run-dir> <data-dir> [extra train.py args]
# Extra arguments are appended last, so they override the copied ones, e.g.
#   scripts/train_rcgan_matched.sh runs/timegan_h164_k41_p3_2k data --device cuda
# Environment variables:
#   PYTHON   interpreter to use (default: python)
#   OUT_DIR  where runs are written (default: runs)
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <timegan-run-dir> <data-dir> [extra train.py args]" >&2
  exit 1
fi
TIMEGAN_RUN=$(cd "$1" && pwd)  # absolute, before changing directory
DATA_DIR=$(cd "$2" && pwd)
shift 2
PYTHON=${PYTHON:-python}
OUT_DIR=${OUT_DIR:-runs}
cd "$(dirname "$0")/.."  # project folder, where train.py and predict.py live

# Shared settings from the TimeGAN run, as train.py arguments.
MATCHED=$("$PYTHON" - "$TIMEGAN_RUN" <<'PY'
import json
import os
import sys

cfg = json.load(open(os.path.join(sys.argv[1], "config.json")))
if cfg.get("model") != "timegan":
    sys.exit(f"{sys.argv[1]} is not a TimeGAN run")
keys = (
    # data preparation
    "tickers", "representation", "step", "scope", "quantile", "seq_len", "stride",
    # backbone and noise
    "hidden_dim", "num_layers", "cond_dim", "z_dim", "noise",
    # optimisation and stabilisation
    "iterations", "batch_size", "lr", "d_lr", "g_steps", "d_steps", "d_threshold", "balanced",
    "seed",
)
args = []
for key in keys:
    value = cfg.get(key)
    flag = "--" + key.replace("_", "-")
    if value is None or value is False:
        continue  # argument left at its default (None) or flag not set
    if value is True:
        args.append(flag)
    elif isinstance(value, list):
        args += [flag] + [str(v) for v in value]
    else:
        args += [flag, str(value)]
print(" ".join(args))
PY
)

RUN_NAME="rcgan_matched_$(basename "$TIMEGAN_RUN")"
echo "matched settings: $MATCHED"
# $MATCHED is unquoted on purpose: it holds several arguments (no spaces inside values).
"$PYTHON" train.py --model rcgan --data-dir "$DATA_DIR" $MATCHED \
  --out-dir "$OUT_DIR" --run-name "$RUN_NAME" "$@"
echo "$TIMEGAN_RUN" > "$OUT_DIR/$RUN_NAME/matched_to.txt"  # record which run it mirrors

"$PYTHON" predict.py --run-dir "$OUT_DIR/$RUN_NAME" --data-dir "$DATA_DIR"
