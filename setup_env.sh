#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/matrices raw_outputs checkpoints
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_lock.txt
echo "Environment ready."
