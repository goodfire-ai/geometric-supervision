#!/bin/bash
# The months task (Section 3). Uses every visible GPU (NUM_GPUS=n to use fewer):
#   bash run_months.sh          # main text: Figures 2-3 and the patching numbers, ~20 minutes
#   bash run_months.sh --all    # also every appendix experiment (1,700 LoRAs), ~1 day
# Outputs go to outputs/months. Finished steps are skipped, so an interrupted run can be restarted.
set -euo pipefail
cd "$(dirname "$0")"
source scripts/gpus.sh
S=scripts/months
MAIN="base task_L03_r08 geometry_e04_s18"

$PY $S/02_fit_probes.py --stage both              # Fourier probes on a+b= prompts, every layer
$PY $S/03_build_planes.py --set canonical         # the supervised planes at layer 18

if [ "${1:-}" != "--all" ]; then
  # the task and geometry LoRAs, 3 seeds each
  sharded $PY $S/04_train.py --config-ids task_L03_r08 geometry_e04_s18
  $PY $S/05_build_ladder_json.py                  # base-model accuracy
  $PY $S/06_capture_manifolds.py --conditions $MAIN --pairs 18,resid --carriers sum --periods 2 5 10
  $PY $S/07_period_sweep.py --conditions $MAIN --layers 18
  queued "$PY $S/08_patch_planes.py --donor task_L03_r08" "$PY $S/08_patch_planes.py --donor geometry_e04_s18"
  $PY $S/13_summarize_runs.py                     # prints the main-text numbers
  $PY $S/14_make_figures.py --only behavioral_main projections_main r2_lines_main
  exit 0
fi

# plane sets for the appendix sweeps and the random / in-span controls
queued "$PY $S/03_build_planes.py --set all" \
  "$PY $S/03_build_planes.py --set random --seed 42 && $PY $S/03_build_planes.py --set span --seed 42" \
  "$PY $S/03_build_planes.py --set random --seed 43 && $PY $S/03_build_planes.py --set span --seed 43" \
  "$PY $S/03_build_planes.py --set random --seed 44 && $PY $S/03_build_planes.py --set span --seed 44"

# every configuration in configs/months, 3 seeds each
sharded $PY $S/04_train.py --all

PERPROMPT="perprompt_perprompt_lam000 perprompt_perprompt_lam030 perprompt_perprompt_lam100 perprompt_perprompt_lam300 perprompt_pervalue_lam000 perprompt_pervalue_lam100 perprompt_pervalue_lam300"
commands=("$PY $S/05_build_ladder_json.py && $PY $S/06_capture_manifolds.py && $PY $S/06_capture_manifolds.py --conditions $PERPROMPT --pairs 18,resid --carriers input-month offset --periods 2 5 10 20 && $PY $S/07_period_sweep.py")
for donor in task_L03_r08 geometry_e04_s18; do
  commands+=("$PY $S/08_patch_planes.py --donor $donor")
  for layer in 18 20 22; do
    commands+=("$PY $S/08_patch_planes.py --donor $donor --layer $layer --carriers sum --periods 2 5 10 20 50")
  done
done
for donor in task_L03_r08 geometry_e04_s18; do
  for layers in "18" "$(seq -s ' ' 4 18)" "$(seq -s ' ' 5 22)" "$(seq -s ' ' 4 31)"; do
    for positions in all last; do
      [ "$layers" = "18" ] && [ "$positions" = "all" ] && continue
      commands+=("$PY $S/09_patch_multilayer.py --donor $donor --layers $layers --positions $positions")
    done
  done
done
for donor in task_L03_r08 geometry_e04_s18; do   # Appendix A.4
  for seed in 42 43 44; do
    commands+=("$PY $S/10_diff_rank_sweep.py --donor $donor --layer 18 --seed $seed"
               "$PY $S/11_difference_subspace.py --donors $donor --seed $seed")
  done
done
commands+=("$PY $S/12_train_multilayer.py --method task --upto 16"
           "$PY $S/12_train_multilayer.py --method task --upto 31"
           "$PY $S/12_train_multilayer.py --method task --upto 31 --attention"
           "$PY $S/12_train_multilayer.py --method geometry --upto 17"
           "$PY $S/12_train_multilayer.py --method geometry --upto 17 --attention")
queued "${commands[@]}"

$PY $S/13_summarize_runs.py
$PY $S/14_make_figures.py
