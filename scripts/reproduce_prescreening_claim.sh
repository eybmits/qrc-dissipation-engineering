#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m pytest -q tests/test_task_resolved.py tests/test_prescreen.py

python experiments/analyze_dissipator_prescreening.py \
  --input results/task_resolved_volterra/prescreening_candidates.csv \
  --output results/task_resolved_volterra/prescreening

python paper/make_prescreening_figures.py \
  --results results/task_resolved_volterra/prescreening \
  --candidates results/task_resolved_volterra/prescreening_candidates.csv \
  --output paper/figures

if command -v latexmk >/dev/null 2>&1; then
  (
    cd paper
    latexmk -pdf -interaction=nonstopmode -halt-on-error \
      certified_qrc_dissipator_prescreening.tex
  )
else
  echo "latexmk not found; numerical reproduction completed, PDF build skipped." >&2
fi
