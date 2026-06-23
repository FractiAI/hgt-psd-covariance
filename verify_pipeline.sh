#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements_lock.txt
else
  source .venv/bin/activate
fi

export PYTHONPATH="$ROOT/src/python"

python tools/fetch_gm12878_hic.py
python tools/train.py --epochs 60 --ablation-rank
python tools/verify_audit.py

echo "=== HGT-PSD PIPELINE COMPLETE ==="
