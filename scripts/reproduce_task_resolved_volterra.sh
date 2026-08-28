#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"

python -m pytest -q tests/test_task_resolved.py
python experiments/run_task_resolved_volterra.py --stage theorem
python experiments/run_task_resolved_volterra.py --stage theory-n3
python experiments/run_task_resolved_volterra.py --stage exact-n3
python experiments/run_task_resolved_volterra.py --stage finite-shot
python experiments/run_task_resolved_volterra.py --stage amplitude
python experiments/run_task_resolved_volterra.py --stage audit
python experiments/run_task_resolved_volterra.py --stage theory-n4
python experiments/run_task_resolved_volterra.py --stage exact-n4

for seed in 0 1 2 3 4; do
  python experiments/run_task_resolved_volterra.py --stage orientation-search --seed "$seed"
  python experiments/run_task_resolved_volterra.py --stage orientation-exact --seed "$seed"
done

python experiments/run_task_resolved_volterra.py --stage orientation-merge
python experiments/analyze_task_resolved_volterra.py
