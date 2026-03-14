# DATA 37100 — Final Project: Diffusion Parameterization & Capacity on MNIST

Chenqi Wang  

---

## Core Question

> How do target parameterization (eps vs x0) and model capacity (base_ch) affect diffusion sample quality on MNIST?

## Models

| Family | Role | Source |
|--------|------|--------|
| Diffusion| Primary — two-knob controlled experiment | `final/starter/src/diffusion_baseline.py` |
| GAN| Second baseline | `final/starter/src/gan_baseline.py` |

## Experiment Design

Two knobs, six runs (2 × 3 grid), all other hyperparameters fixed:

| target \ base_ch | 32 | 64 | 128 |
|-------------------|----|----|-----|
| eps | run 1 | run 2 | run 3 |
| x0  | run 4 | run 5 | run 6 |

Fixed: T=200, beta1=1e-4, beta2=0.02, lr=2e-4, epochs=1, batch_size=128, seed=42.

---

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd data37100_final_chenqi_wang

# 2. Create a conda environment (Python 3.11)
conda create -n data37100 python=3.11 -y
conda activate data37100

# 3. Install dependencies
pip install -r requirements.txt
```

MNIST is downloaded automatically by `torchvision` on the first run.

---

## Run Commands

### GAN baseline

```bash
python final/starter/src/gan_baseline.py --dataset mnist --epochs 1
```

### Diffusion baseline

```bash
python final/starter/src/diffusion_baseline.py --dataset mnist --epochs 1 --T 200
```

### Controlled experiment (6-run grid)

```bash
python final/starter/src/diffusion_baseline.py --dataset mnist --epochs 1 --T 200 --grid "target=eps,x0;base_ch=32,64,128"
```

All outputs are saved under `./untrack/outputs/final/`.

---

## Expected Runtime

All experiments were run on **CPU** (no GPU required).

| Run | Time |
|-----|------|
| GAN baseline (1 epoch) | ~42 s |
| Diffusion, base_ch=32 | ~38 s |
| Diffusion, base_ch=64 | ~90 s |
| Diffusion, base_ch=128 | ~250 s |
| **Full grid (6 diffusion runs)** | **~12 min total** |

---

## Repository Structure

```
.
├── README.md                          ← this file
├── requirements.txt                   ← Python dependencies
├── final/
│   ├── FINAL_PROJECT.md               ← project requirements
│   ├── PROJECT_PROPOSAL.md            ← approved proposal
│   ├── STUDENT_CHECKLIST.md           ← submission checklist
│   ├── DATASETS.md                    ← approved datasets
│   ├── starter/src/
│   │   ├── diffusion_baseline.py      ← diffusion training + grid experiment
│   │   ├── gan_baseline.py            ← GAN training
│   │   └── utils_data.py              ← data loading utilities
│   └── analysis/
│       └── final_project_analysis_template.ipynb  ← analysis notebook
├── week7/src/
│   └── lab07_diffusion_core.py        ← diffusion core (schedule, model, sampling)
└── untrack/                           ← all outputs (not committed to git)
    └── outputs/final/
        ├── diffusion/
        │   ├── results.csv            ← experiment manifest
        │   └── ds-mnist_T-200_target-{eps,x0}_b2-0.02_ch-{32,64,128}/
        │       ├── run_args.json
        │       ├── summary.json
        │       ├── checkpoints/
        │       └── samples/           ← sample grids + denoising steps
        └── gan/
            └── ds-mnist_ep-1_.../
                ├── summary.json
                ├── train_log.csv
                └── samples/
```

---

## Analysis Notebook

The analysis notebook is at `final/analysis/final_project_analysis_template.ipynb`.

To run it:

```bash
cd final/analysis
jupyter notebook final_project_analysis_template.ipynb
```

The notebook loads results from `untrack/` and produces:
- Baseline comparison (GAN vs Diffusion)
- Side-by-side sample grids for all 6 experiment runs
- Runtime and diversity proxy tables
- Failure mode analysis and conclusions

---

## Random Seeds

All runs use `seed=42`. Both `diffusion_baseline.py` and `gan_baseline.py` set:
- `torch.manual_seed`
- `torch.cuda.manual_seed_all`
- `random.seed`
- `np.random.seed`

ai acknoledge: use chatgpt for some debuging, manage the code readbility and check and fix garmmar.
