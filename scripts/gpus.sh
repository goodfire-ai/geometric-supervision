# Helpers for spreading work over the GPUs of one machine; sourced by run_months.sh and
# run_linebreak.sh. Uses every visible GPU unless NUM_GPUS is set.
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
PY="uv run --frozen python"

_wait_all() {
  local pid status=0
  for pid in "$@"; do wait "$pid" || status=1; done
  return $status
}

# sharded CMD...: run "CMD --shard i --of NUM_GPUS" on every GPU i at once.
sharded() {
  local i pids=()
  for ((i = 0; i < NUM_GPUS; i++)); do
    (set -o pipefail; CUDA_VISIBLE_DEVICES=$i "$@" --shard "$i" --of "$NUM_GPUS" 2>&1 | sed -u "s/^/[gpu$i] /") &
    pids+=($!)
  done
  _wait_all "${pids[@]}"
}

# queued CMD...: each argument is one shell command; they are dealt round-robin to the
# GPUs, and each GPU runs its share in order. Fails if any command fails.
queued() {
  local cmds=("$@") i pids=()
  for ((i = 0; i < NUM_GPUS; i++)); do
    (
      set -o pipefail
      status=0
      for ((k = i; k < ${#cmds[@]}; k += NUM_GPUS)); do
        echo "[gpu$i] ${cmds[k]}"
        CUDA_VISIBLE_DEVICES=$i bash -c "${cmds[k]}" 2>&1 | sed -u "s/^/[gpu$i] /" || status=1
      done
      exit $status
    ) &
    pids+=($!)
  done
  _wait_all "${pids[@]}"
}
