# Neural Geometry as a Learning Signal

Companion code for *Neural Geometry as a Learning Signal*.

We fine-tune Llama-3.1-8B with LoRAs that are trained only to make an internal representation match an idealized geometry (no loss on the model's outputs), and compare them to LoRAs trained on the task directly. The geometric loss (Equations 2-5) is in `src/common/geometry.py`.

## Reproducing the paper

```bash
bash run_months.sh       # Section 3 (~20 min on one GPU)
bash run_linebreak.sh    # Section 4 (~2 hours on one GPU)
```

Figures and results are written to `outputs/months/` and `outputs/linebreak/`. Add `--all` to also run the appendix experiments, which train many more LoRAs and use all visible GPUs (set `NUM_GPUS` to use fewer). The off-target benchmarks (Appendix A.5) and the CCA, full fine-tuning and few-shot baselines (Appendix A.6) are not included.

`artifacts/` has the four trained LoRAs from the main text, with an example of loading them.

### Alternate forms of supervision (Appendices A.3.3 and B.3.2)

These compare the idealized geometry to other training targets (direct regression of the task variables through linear probes, evenly spaced lines, the base model's own centroids, and readouts in other subspaces). They are not part of `run_months.sh` or `run_linebreak.sh`; run them after the main reproduction of each task, which provides the probes, planes and per-count means they build on.

```bash
# months: build the alternate readouts and targets, train every configuration (504 configs x 3 seeds), print the table
uv run --frozen python scripts/months/15_alternate_planes.py
uv run --frozen python scripts/months/04_train.py --configs configs/months_alternate --all   # add --shard i --of n to split over GPUs
uv run --frozen python scripts/months/16_summarize_alternate.py

# linebreaking: each target at the paper's configuration, then the PCA-subspace targets over alpha_r x tau
ALT="configs/linebreak/alternate_targets.jsonl configs/linebreak/alternate_targets_pca_grid.jsonl"
uv run --frozen python scripts/linebreak/11_sweep.py --configs $ALT                         # add --shard i --of n to split over GPUs
uv run --frozen python scripts/linebreak/11_sweep.py --configs $ALT --summarize
```

The tuned helix that the PCA-subspace targets are compared with is the α_r × τ grid of `configs/linebreak/hparam_sweep.jsonl` (`run_linebreak.sh --all`).

## Table of Contents

### Section 3: Months

All under `scripts/months/`.

- `02_fit_probes.py` trains Fourier probes on `a+b=` prompts, following Feucht et al. (Appendix A.2).
- `03_build_planes.py` builds the supervised Fourier planes and their target rings.
- `04_train.py` trains task and geometry LoRAs. Every configuration we ran is in `configs/months/`.
- `06_capture_manifolds.py` and `07_period_sweep.py` project activations onto the Fourier planes and measure how circular they are (Figure 3).
- `08_patch_planes.py` and `09_patch_multilayer.py` patch fine-tuned activations into the base model (Section 3.3).
- `10_diff_rank_sweep.py` and `11_difference_subspace.py` look at what the fine-tuned models change at layer 18 (Appendix A.4).
- `12_train_multilayer.py` trains LoRAs on many layers at once (Tables in Appendix A.1 and A.3).
- `13_summarize_runs.py` prints the main-text numbers, and `14_make_figures.py` draws the figures.
- `15_alternate_planes.py` builds the readouts and targets for the alternate forms of supervision, and `16_summarize_alternate.py` prints their table (Appendix A.3.3; configurations in `configs/months_alternate/`).

### Section 4: Linebreaking

All under `scripts/linebreak/`.

- `01_build_corpus.py` downloads FineWeb documents and strips their newlines.
- `02_fit_probes.py` fits the helix probes at layer 5, and `03_count_means.py` computes the per-count mean activations (Figure 5b).
- `04_train.py` trains the task and geometry LoRAs.
- `05_capture_clouds.py` fits the helix to each model's activations (Figure 6), and `06_probe_r2_by_layer.py` fits count probes at every layer (Figure 5a).
- `07_evaluate.py` measures linebreaking error (Figure 4). By default it scores one seed of each LoRA at every 5th width, which is much faster; the paper (and `--all`) scores three seeds at every width.
- `08_ablate.py` ablates the count subspace (Figure 5c).
- `09_make_figures.py` draws the figures, and `10_summarize.py` prints the main numbers.
- `11_sweep.py` runs the rank, layer, α_r, τ and λ sweeps (Appendix B.3.1), and the alternate supervision targets (Appendix B.3.2; `configs/linebreak/alternate_targets*.jsonl`, targets defined in `src/linebreak/targets.py`).

## Setup

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and a GPU with 80 GB of memory.

```bash
git clone https://github.com/goodfire-ai/geometric-supervision.git
cd geometric-supervision
uv sync --extra dev
uv run pytest
```

`torch` is pinned to the CUDA 12.8 build, so you need an NVIDIA driver that supports CUDA ≥ 12.8. The model is [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B), which is gated: accept the license on Hugging Face and run `hf auth login` first.
