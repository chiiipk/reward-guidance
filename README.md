# Are we really tilting?

Experiment code for the paper *"Are we really tilting? The mechanics of reward guidance in flow and diffusion models"*. The repository implements the plug-in (Doob *h*-transform) sampler, the reward damping schedule, best-of-*n* selection, and flow map reward guidance (FMRG), and contains the runnable scripts that produce every figure and table in the paper.

The code is organized into four self-contained experiments:

| Directory | Experiment |
|---|---|
| `gaussian_mixture/` | 2D Gaussian and Gaussian-mixture targets with quadratic and quartic rewards |
| `mode_selection/` | 1D symmetric Gaussian-mixture mode selection with step and Gaussian rewards |
| `checkerboard/` | 2D checkerboard target with a Gaussian-bump reward |
| `flux/` | FLUX.1-dev text-to-image with blueness, masked-brightness, ImageReward, and VLM rewards |

Each experiment has its own `sample.py` for generating samples and one or more figure scripts (`make_*.py` or `figures/*/regenerate.py`) for rendering the corresponding figures. Generated artifacts land in `results/` (sampling outputs) and `figures/` (rendered figures), both gitignored.

---

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # skip if uv is installed
uv sync
source .venv/bin/activate
```

`uv sync` creates `.venv` and installs the exact environment described by
`pyproject.toml` and `uv.lock`. Use Python 3.10–3.12; Python 3.11 is recommended.

The Gaussian-mixture, checkerboard, and 1D mode-selection experiments run on CPU or a single small GPU. The FLUX experiments require a single GPU with at least 48 GB of memory (we used an NVIDIA RTX A6000 or L40S); 24 GB is not sufficient even with gradient checkpointing.

The FLUX scripts download:
- `black-forest-labs/FLUX.1-dev` (FLUX.1 [dev] Non-Commercial License)
- `Qwen/Qwen2.5-VL-3B-Instruct` (Qwen Research License Agreement) for the VLM reward
- ImageReward (Apache-2.0) for the learned preference reward

You will need a Hugging Face access token to download FLUX.1-dev:

```bash
huggingface-cli login
```

---

## Quick start

After installation, the four end-to-end pipelines are:

```bash
# Gaussian / Gaussian-mixture grids and the FMRG crossover figure
cd gaussian_mixture/ && python make_grid_figures.py && python make_fmrg_figure.py

# 1D mode-selection figures
cd mode_selection/
python sample.py --reward step     --record-trajectories --output-dir results/step_lam5.0
python sample.py --reward gaussian --record-trajectories --output-dir results/gaussian_lam5.0
python make_overview_figure.py
python make_trajectory_figures.py

# Checkerboard
cd checkerboard/
python train.py --num-steps 500000               # ~60 min on one GPU
python sample.py --analytic-tilt --lam 10.0
python sample.py --k 1 --lam 10.0 --num-samples 20000
python sample.py --k 8 --lam 10.0
python sample.py --k 1 --lam 10.0 --sigma-damp 0.2 --num-samples 20000
python make_main_figure.py
python plot.py --plots bon_paper --lam 10.0       # produces bon_reward_vs_n.pdf

# FLUX (one figure at a time; see "FLUX figures" section below)
cd flux/
python sample.py --reward blue_minus_rg \
    --prompt "Artist painting in the center of a cluttered room lit by candlelight, rococo" \
    --num-images 20 --output-dir ../data/blueness_rococo/unguided --reward-scale 0
# ... (additional conditions, then:)
cd ../figures/blueness_rococo/ && python regenerate.py
```

The FLUX commands are repetitive across conditions, so each figure folder under `figures/` has a fixed list of conditions it expects to find under `data/<figure_name>/<condition>/`. The exact commands per figure are listed below.

---

## Repository layout

```
release/
├── README.md
├── pyproject.toml                    # dependency source for uv
├── uv.lock                           # reproducible resolved environment
├── assets/
│   ├── default.mplstyle              # used by sample/plot scripts
│   ├── paper.mplstyle                # used by paper-quality figure scripts
│   └── fonts/                        # Lato (used by paper.mplstyle)
├── gaussian_mixture/
│   ├── model.py                      # GMM target, interpolant, reward functions
│   ├── sample.py                     # GuidedSampler (analytic, exact-h, plug-in, FMRG)
│   ├── make_grid_figures.py          # the 6 GMM 4-panel grids
│   └── make_fmrg_figure.py           # the FMRG vs. plug-in regime-crossover figure
├── mode_selection/
│   ├── model.py                      # 1D symmetric Gaussian mixture, reward functions
│   ├── sample.py                     # 1D plug-in sampler with best-of-n
│   ├── make_overview_figure.py       # the step-reward best-of-n overview
│   └── make_trajectory_figures.py    # particle-trajectory figures (step + Gaussian rewards)
├── checkerboard/
│   ├── model.py                      # VelocityMLP, checkerboard density, reward
│   ├── train.py                      # flow-matching training loop
│   ├── sample.py                     # GLASS-flow plug-in sampler
│   ├── plot.py                       # diagnostic plots and best-of-n / softmax-of-n curve
│   └── make_main_figure.py           # the four-panel mode-selection figure
├── flux/
│   ├── pipeline.py                   # FLUX pipeline wrapper with hooks for guidance
│   ├── dual_time_embedder.py         # Diamond-map time embedder
│   ├── rewards.py                    # blueness, masked-brightness, ImageReward, VLM rewards
│   └── sample.py                     # FLUX guided sampler
└── figures/                          # one folder per text-to-image figure
    ├── blueness_fox/regenerate.py
    ├── blueness_rococo/regenerate.py
    ├── masked_brightness_welder/regenerate.py
    ├── imagereward_archaeologist/regenerate.py
    ├── imagereward_market/regenerate.py
    ├── imagereward_miner/regenerate.py
    ├── vlm_diner_eclipse/regenerate.py
    ├── vlm_subway_mars/regenerate.py
    └── fmrg_blueness_dragon/regenerate.py
```

---

## Gaussian and Gaussian-mixture experiments

These figures are produced entirely from closed-form analytics plus short Heun integrators; no external data is needed.

```bash
cd gaussian_mixture/
python make_grid_figures.py         # writes ../figures/gaussian_mixture/{gaussian,quadratic,double_well,noniso,unequal,uncentered}.{pdf,png}
python make_fmrg_figure.py          # writes ../figures/gaussian_mixture/fmrg_crossover.{pdf,png}
```

If you want to inspect intermediate samples, use `python sample.py --help` for the full CLI (variant selection, reward scale `--lam`, particle count `--k`, damping `--sigma-damp`, method `--method {analytic, exact, plugin, fmrg}`, etc.).

---

## 1D mode-selection experiments

```bash
cd mode_selection/

# Sample the step- and Gaussian-reward 1D mixtures (each takes a few CPU-seconds).
python sample.py --reward step     --num-samples 2000 --max-n 16 --lam 5.0 \
    --record-trajectories --output-dir results/step_lam5.0
python sample.py --reward gaussian --num-samples 2000 --max-n 16 --lam 5.0 \
    --record-trajectories --output-dir results/gaussian_lam5.0

# Render figures.
python make_overview_figure.py       # ../figures/mode_selection/best_of_n_overview.{pdf,png}
python make_trajectory_figures.py    # ../figures/mode_selection/{step,gaussian}_trajectories.{pdf,png}
```

`--record-trajectories` stores intermediate-time particle positions so the trajectory figure can be drawn; without it, only the terminal samples are saved.

---

## Checkerboard experiments

The checkerboard base velocity field is a 4-layer MLP trained with flow matching. Training takes about 60 minutes on a single GPU; if you want a quick sanity check first, drop `--num-steps` to 10000 and the model will still produce visually recognizable samples.

```bash
cd checkerboard/

# (1) Train the base velocity field.
python train.py --num-steps 500000

# (2) Sample the conditions used by the paper figures (lambda = 10).
python sample.py --analytic-tilt --lam 10.0
python sample.py --k 1 --lam 10.0 --num-samples 20000
python sample.py --k 8 --lam 10.0 --num-samples 5000
python sample.py --k 1 --lam 10.0 --sigma-damp 0.2 --num-samples 20000

# (3) Render the four-panel mode-selection figure (paper Fig. 5).
python make_main_figure.py           # ../figures/checkerboard/main_figure.{pdf,png}

# (4) Render the best-of-n vs softmax-of-n curve (paper appendix figure).
python plot.py --plots bon_paper --lam 10.0       # ../figures/checkerboard/bon_reward_vs_n.pdf
```

`checkerboard/plot.py` is a multi-mode diagnostic plotting script with many other modes (per-method scatter, lambda sweeps, damping sweeps, etc.). Run `python plot.py --help` for the full list.

---

## FLUX text-to-image experiments

Each FLUX figure expects per-condition image samples under `data/<figure_name>/<condition>/<seed_dir>/<index>.png` and a `rewards.npy` of per-image reward values in the same `<seed_dir>`. The `flux/sample.py` script writes exactly this layout when given `--output-dir`.

The skeleton command is:

```bash
cd flux/
python sample.py \
    --reward {blue_minus_rg | masked_brightness | imagereward | skywork} \
    --prompt "<prompt text>" \
    [--ir-prompt "<ImageReward scoring prompt>"] \
    [--skywork-question "<yes/no question>" --skywork-model-id Qwen/Qwen2.5-VL-3B-Instruct] \
    --gradient-norm-scale <gns>          # paper's lambda (gradient is unit-normalized)
    [--num-particles 8 --lam 1.0]        # for k = 8 conditions
    [--sigma-damp 0.1]                   # for damped conditions
    --reward-scale 1                     # use 0 for the unguided baseline
    --snr-factor 5 --num-guidance-steps 5 --guidance-start-step 1 \
    --num-images 20 --num-steps 28 --height 512 --width 512 \
    --output-dir ../data/<figure_name>/<condition>
```

After producing all conditions for a figure, render it with:

```bash
cd ../figures/<figure_name>/
python regenerate.py                  # writes <figure_name>.{pdf,png} alongside the script
```

The exact (figure, condition) → command mapping for every paper figure follows.

### Figure: `blueness_fox` (appendix)

Prompt: *"a baby fox wearing a cozy knitted sweater"*. Reward: `blue_minus_rg`.

| Condition (subdir under `data/blueness_fox/`) | sample.py flags |
|---|---|
| `unguided`         | `--reward-scale 0` |
| `gns100`           | `--gradient-norm-scale 100` |
| `gns50`            | `--gradient-norm-scale 50` |
| `gns50_k8`         | `--gradient-norm-scale 50 --num-particles 8 --lam 1.0` |
| `gns100_damp0.1`   | `--gradient-norm-scale 100 --sigma-damp 0.1` |

### Figure: `blueness_rococo` (main text)

Prompt: *"Artist painting in the center of a cluttered room lit by candlelight, rococo"*. Reward: `blue_minus_rg`.

Same five conditions as `blueness_fox`, but the displayed conditions in the figure are `gns50`, `gns30`, `gns50_k8`, `gns100_damp0.1` (plus unguided). Run the same five commands with `--gradient-norm-scale 30` substituted for the lower-lambda baseline.

### Figure: `masked_brightness_welder` (main text)

Prompt: *"Photorealistic worm's-eye view of a welder mid-spark inside a rusted ship hull, sweat, smoke, orange backlight"*. Reward: `masked_brightness` with `--mask-region topright_circle`.

| Condition | sample.py flags |
|---|---|
| `unguided`         | `--reward-scale 0` |
| `gns100`           | `--gradient-norm-scale 100` |
| `gns50`            | `--gradient-norm-scale 50` |
| `gns50_k8`         | `--gradient-norm-scale 50 --num-particles 8 --lam 1.0` |
| `gns100_damp0.1`   | `--gradient-norm-scale 100 --sigma-damp 0.1` |

### Figures: `imagereward_archaeologist`, `imagereward_miner`, `imagereward_market`

Reward: `imagereward`. Each figure uses one prompt; pass it both as `--prompt` and as `--ir-prompt` (ImageReward scores against the latter).

| Figure | Prompt | Damped sigma |
|---|---|---|
| `imagereward_archaeologist` | *"a young archaeologist gently brushing dust from an ancient ceramic vase, soft museum lighting, intricate details, cinematic composition"* | 0.15 |
| `imagereward_miner`         | *"a coal miner pausing for a moment underground, hard hat lamp glowing, dust in the air, painterly chiaroscuro"* | 0.10 |
| `imagereward_market`        | *"a vibrant Indian outdoor market with colorful stalls and produce"* (and `--prompt` set to a duller scene to make the reward signal visible) | 0.05 |

The condition keys mirror `blueness_fox`. The `imagereward_market` figure uses `gns50` and `gns30` for the guided / lower-lambda rows.

### Figures: `vlm_diner_eclipse`, `vlm_subway_mars`

Reward: `skywork` with `--skywork-model-id Qwen/Qwen2.5-VL-3B-Instruct`.

Pass the question via `--skywork-question`. Each condition produces 20 images that are then grouped into best-of-1, best-of-2, best-of-4 cells inside the figure.

| Figure | Prompt | Question |
|---|---|---|
| `vlm_diner_eclipse` | *"A roadside American diner in the Nevada desert, shot at twilight, a neon sign on the roof glowing ECLIPSE DINER in cherry-red and cream tubes, a long empty highway behind it, painterly warm light on chrome surfaces"* | *"Does this image clearly show a neon sign with the word 'ECLIPSE' as the main readable text? Answer Yes or No."* |
| `vlm_subway_mars`   | *"cyberpunk subway platform with a holographic display that says NEXT TRAIN MARS, teal neon, commuters in silhouette"* | *"Does this image clearly show a display with the text 'NEXT TRAIN MARS' as the main readable text? Answer Yes or No."* |

Conditions per figure:

| Condition | sample.py flags |
|---|---|
| `unguided`         | `--reward-scale 0` |
| `gns100`           | `--gradient-norm-scale 100` |
| `gns50_k8`         | `--gradient-norm-scale 50 --num-particles 8 --lam 1.0` |
| `gns100_damp0.1`   | `--gradient-norm-scale 100 --sigma-damp 0.1` |

### Figure: `fmrg_blueness_dragon` (appendix)

Three rows on the same prompt. Reward: `blue_minus_rg`.

Prompt: *"a massive dragon perched on basalt cliffs above lava waterfalls, volcanic ash, crimson sunset, ultra-detailed fantasy"*

| Subdir | sample.py flags |
|---|---|
| `dragon_unguided`         | `--reward-scale 0` |
| `dragon_plugin`           | `--gradient-norm-scale 50 --snr-factor 5` |
| `dragon_fmrg`             | `--gradient-norm-scale 50 --snr-factor 1` |

(`--snr-factor 1` makes the lookahead deterministic, which is the FMRG scheme; `--snr-factor 5` gives the Diamond-map plug-in scheme used for the rest of the paper.)

---

## How the code maps to the paper

- **Plug-in (Doob *h*-transform) guidance.** Implemented in each `sample.py`. For the closed-form experiments, see `gaussian_mixture/sample.py::GuidedSampler` and `mode_selection/sample.py`. For the high-dimensional experiments, see `flux/sample.py::compute_plugin_guidance_flux` and the `--num-particles` flag for the *k*-particle estimator.
- **Reward damping.** The `--sigma-damp` flag in every `sample.py` activates the time-dependent reward scale schedule from the paper. In `gaussian_mixture/sample.py`, set `apply_damping_scale=True` when constructing the sampler.
- **Best-of-*n*.** Implemented in `mode_selection/sample.py::best_of_n` and `checkerboard/sample.py::best_of_n`. For FLUX figures, best-of-*n* is performed inside the `figures/vlm_*/regenerate.py` scripts directly from the per-image reward arrays.
- **Flow map reward guidance (FMRG).** Implemented as the `"fmrg"` method in `gaussian_mixture/sample.py`; for FLUX it is reproduced by setting `--snr-factor 1` (deterministic flow-map endpoint) instead of the default `--snr-factor 5` (Diamond-map renoise lookahead).

---

## License

Released under the MIT License (see `LICENSE`). Third-party components carry their own licenses, which apply when those components are used:

- The FLUX text-to-image experiments depend on `black-forest-labs/FLUX.1-dev`, distributed under the FLUX.1 [dev] Non-Commercial License.
- The VLM-reward experiments depend on `Qwen/Qwen2.5-VL-3B-Instruct`, distributed under the Qwen Research License Agreement.
- The ImageReward dependency is distributed under Apache-2.0.
- `flux/dual_time_embedder.py` is reproduced verbatim from the Diamond Maps repository (https://github.com/PeterHolderrieth/diamond_maps), and `flux/pipeline.py` follows its structure; please cite their work when using these components.
- The checkerboard training and sampling code follows conventions from `nmboffi/jax-interpolants` (https://github.com/nmboffi/jax-interpolants).
