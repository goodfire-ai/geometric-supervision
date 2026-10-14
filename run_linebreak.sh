#!/bin/bash
# The linebreaking task (Section 4). Uses every visible GPU (NUM_GPUS=n to use fewer):
#   bash run_linebreak.sh          # main text: Figures 4-6, ~2 hours
#   bash run_linebreak.sh --all    # as in the paper: Fig. 4 on all 3 seeds and every width, plus the
#                                  # hyperparameter sweeps of Appendix B (141 LoRAs), ~1-2 days
# Outputs go to outputs/linebreak. Finished steps are skipped, so an interrupted run can be restarted.
set -euo pipefail
cd "$(dirname "$0")"
source scripts/gpus.sh
S=scripts/linebreak

if [ -f "${GEOSUP_OUTPUTS:-outputs}/linebreak/corpus/manifest.json" ]; then
  $PY $S/01_build_corpus.py --check
else
  $PY $S/01_build_corpus.py   # streams FineWeb from the Hugging Face Hub
fi
$PY $S/02_fit_probes.py
$PY $S/03_count_means.py

# the task and geometry LoRAs, 3 seeds each
sharded $PY $S/04_train.py

# activation clouds, per-layer probe R^2, ablations, and the linebreaking error of every model
EVAL=""
[ "${1:-}" = "--all" ] && EVAL="--paper"   # Fig. 4 on all seeds and every width
commands=("$PY $S/05_capture_clouds.py" "$PY $S/06_probe_r2_by_layer.py" "$PY $S/08_ablate.py")
for ((i = 0; i < NUM_GPUS; i++)); do
  commands+=("$PY $S/07_evaluate.py $EVAL --shard $i --of $NUM_GPUS")
done
queued "${commands[@]}"
$PY $S/07_evaluate.py --merge

if [ "${1:-}" = "--all" ]; then
  sharded $PY $S/11_sweep.py
  $PY $S/11_sweep.py --summarize
fi

# figures and the headline numbers (CPU)
$PY $S/09_make_figures.py
$PY $S/10_summarize.py
