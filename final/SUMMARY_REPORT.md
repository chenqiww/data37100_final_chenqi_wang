# DATA 37100 — Final Project Summary Report

**Author:** Chenqi Wang  
**Course:** DATA 37100, Winter 2026  
**Dataset:** MNIST  
**Date:** March 13, 2026

---

## 1. Question & Motivation

**Core question:** How do target parameterization (`eps` vs `x0`) and model capacity (`base_ch`) affect diffusion sample quality on MNIST?

In DDPM-style diffusion models, the denoiser network can be trained to predict either the added noise (`eps`) or the clean image (`x0`). The original DDPM paper (Ho et al., 2020) uses `eps` and argues that it simplifies the loss landscape and improves gradient behavior. However, `x0` prediction is also valid and may behave differently when the training budget is limited and neither parameterization fully converges.

Separately, the base channel width (`base_ch`) controls the capacity of the U-Net denoiser. A wider network has more parameters and representational power but requires proportionally more compute.

This project investigates how these two knobs interact under a deliberately short 1-epoch training budget on MNIST. The dataset is chosen for its simplicity: each run completes in minutes on CPU, allowing a full 2 × 3 experiment grid in under 15 minutes, while producing visually interpretable outputs where success and failure are easy to distinguish.

A GAN (DCGAN) baseline is included as a second required model family, providing a contrast in how different generative paradigms fail under the same short-training constraint.

---

## 2. Methods

### 2.1 Models

| Family | Architecture | Role |
|--------|-------------|------|
| **Diffusion (DDPM-lite)** | U-Net with GroupNorm, SiLU, sinusoidal time embedding | Primary model for controlled two-knob experiment |
| **GAN (DCGAN)** | Convolutional generator + discriminator, BCEWithLogitsLoss | Second baseline for cross-family comparison |

### 2.2 Controlled Experiment (Two-Knob Grid)

Two knobs are varied; all other hyperparameters are held fixed:

- **Knob 1 — `target`** ∈ {`eps`, `x0`}: the denoiser's training objective (predict noise vs. predict clean image)
- **Knob 2 — `base_ch`** ∈ {32, 64, 128}: the U-Net's base channel width (controls model capacity)

Fixed hyperparameters: T = 200 diffusion steps, linear noise schedule (β₁ = 1×10⁻⁴, β₂ = 0.02), AdamW (lr = 2×10⁻⁴), batch size = 128, epochs = 1, seed = 42.

This produces a 2 × 3 grid of **6 runs**:

| target \ base_ch | 32 | 64 | 128 |
|-------------------|----|----|-----|
| **eps** | run 1 | run 2 | run 3 |
| **x0**  | run 4 | run 5 | run 6 |

### 2.3 GAN Baseline

DCGAN with lr = 0.0002, z_dim = 128, base_ch = 64, d_steps = 1, epochs = 1, seed = 42 (401 training steps, ~42 s on CPU).

### 2.4 Evaluation

- **Qualitative:** 8×8 sample grids for all 6 diffusion runs; sample grids for GAN.
- **Quantitative:** Training runtime per configuration; pixel-space pairwise L2 diversity proxy (average pairwise L2 distance on a 64-sample batch).
- **Reproducibility:** All runs fix `torch.manual_seed`, `torch.cuda.manual_seed_all`, `random.seed`, `np.random.seed` with seed = 42.

---

## 3. Results

### 3.1 Baseline Comparison — GAN vs. Diffusion

Both baselines trained for 1 epoch on MNIST:

| Model | Runtime | Steps | Device |
|-------|---------|-------|--------|
| GAN (DCGAN, ch=64) | 42.3 s | 401 | CPU |
| Diffusion (eps, ch=64) | 86.7 s | 468 | CPU |

**GAN baseline (final samples, step 401):**

![GAN baseline samples](../untrack/outputs/final/gan/ds-mnist_ep-1_bs-128_lr-0.0002_dsteps-1_z-128_ch-64_seed-42/samples/grid_step000401.png)

*Figure 1: DCGAN samples after 1 epoch (401 steps). Digits are recognizable but show limited variety.*

**Diffusion baseline (final samples, step 468):**

![Diffusion baseline samples](../untrack/outputs/final/diffusion/ds-mnist_T-200_target-eps_b2-0.02_ch-64/samples/samples_step000468.png)

*Figure 2: Diffusion baseline (eps, ch=64) samples after 1 epoch. Noisier than GAN, with some emerging digit structure.*

The two families exhibit qualitatively different behaviors under short training: the GAN produces recognizable but low-diversity digits, while the diffusion model generates noisier, less structured outputs.

---

### 3.2 Two-Knob Experiment — Sample Grids

The figures below show the final (step 468) 8×8 sample grids for all 6 configurations, arranged as a 2 × 3 comparison.

#### eps parameterization (top row)

| base_ch = 32 | base_ch = 64 | base_ch = 128 |
|:---:|:---:|:---:|
| ![eps-32](../untrack/outputs/final/diffusion/ds-mnist_T-200_target-eps_b2-0.02_ch-32/samples/samples_step000468.png) | ![eps-64](../untrack/outputs/final/diffusion/ds-mnist_T-200_target-eps_b2-0.02_ch-64/samples/samples_step000468.png) | ![eps-128](../untrack/outputs/final/diffusion/ds-mnist_T-200_target-eps_b2-0.02_ch-128/samples/samples_step000468.png) |

*Figure 3: eps parameterization at increasing model capacity. At ch=32, the model produces white blob-like masses with no digit structure. Quality gradually improves with capacity but remains noisy.*

#### x0 parameterization (bottom row)

| base_ch = 32 | base_ch = 64 | base_ch = 128 |
|:---:|:---:|:---:|
| ![x0-32](../untrack/outputs/final/diffusion/ds-mnist_T-200_target-x0_b2-0.02_ch-32/samples/samples_step000468.png) | ![x0-64](../untrack/outputs/final/diffusion/ds-mnist_T-200_target-x0_b2-0.02_ch-64/samples/samples_step000468.png) | ![x0-128](../untrack/outputs/final/diffusion/ds-mnist_T-200_target-x0_b2-0.02_ch-128/samples/samples_step000468.png) |

*Figure 4: x0 parameterization at increasing model capacity. Even at ch=32, identifiable (though distorted) digits emerge. Quality improves monotonically with capacity.*

**Key observation:** x0 consistently outperforms eps at every capacity level. The contrast is most dramatic at ch=32, where eps completely fails (noise blobs) while x0 already produces recognizable digits.

---

### 3.3 Training Runtime

Runtime depends almost entirely on `base_ch`; the choice of target has negligible impact:

| target \ base_ch | 32 | 64 | 128 |
|-------------------|------|------|--------|
| **eps** | 38.2 s | 86.7 s | 252.3 s |
| **x0**  | 38.1 s | 86.5 s | 250.1 s |

*Table 1: Training time per run (CPU, 1 epoch). Runtime grows super-linearly with base_ch (~6.5× from ch=32 to ch=128), reflecting quadratic parameter-count growth.*

---

### 3.4 Diversity Proxy

Average pairwise L2 distance in pixel space over 64 generated samples:

| target \ base_ch | 32 | 64 | 128 |
|-------------------|-------|-------|-------|
| **eps** | **13.45** | 11.62 | 12.69 |
| **x0**  | 11.22 | 11.50 | 12.24 |

*Table 2: Pixel-space diversity proxy. Higher ≈ more varied outputs, but this metric is unreliable for underfitted models (see Section 4).*

For x0, diversity increases monotonically with capacity (11.22 → 11.50 → 12.24), which is interpretable: a larger model produces more varied digit outputs. For eps, the pattern is non-monotonic — eps/ch=32 has the **highest** diversity score (13.45) despite producing the worst samples. This is because random, unstructured noise blobs are all different from each other in pixel space, inflating the L2 metric.

---

## 4. Failure Modes & Limitations

### Failure Mode 1: eps Produces Noise Blobs at Low Capacity

With `target=eps` and `base_ch=32`, the model outputs large white blob-like masses with no recognizable digit structure (Figure 3, left). In contrast, `target=x0` at the same capacity already produces identifiable, though distorted, digits (Figure 4, left). This pattern persists at higher capacities: eps samples appear noisier and less structured than x0 at every `base_ch` level tested.

**Likely cause:** In the eps parameterization, the network predicts the noise component at each step, and the reverse process reconstructs the image by iteratively subtracting the predicted noise across 200 sequential denoising steps. Under short training and limited capacity, errors in noise prediction may propagate and compound through these steps, leading to visibly degraded final samples. By contrast, x0 parameterization directly predicts the clean image, which may provide a more accessible learning target in this underfitted regime: even an approximate x0 prediction can produce digit-like structure, whereas approximate eps predictions translate into progressively noisier reconstructions.

| eps, ch=32 (failure) | x0, ch=32 (success) |
|:---:|:---:|
| ![eps-32-fail](../untrack/outputs/final/diffusion/ds-mnist_T-200_target-eps_b2-0.02_ch-32/samples/samples_step000468.png) | ![x0-32-ok](../untrack/outputs/final/diffusion/ds-mnist_T-200_target-x0_b2-0.02_ch-32/samples/samples_step000468.png) |

*Figure 6: Direct comparison at ch=32 — eps (left) produces noise blobs; x0 (right) produces recognizable digits.*

### Failure Mode 2: Misleading Diversity Proxy for Underfitted Models

The pixel-space diversity proxy assigns eps/ch=32 the **highest** diversity score (13.45), yet visual inspection (Figure 5, left) shows these are random noise blobs, not meaningfully diverse outputs. Meanwhile, x0/ch=32 has lower diversity (11.22) but produces actual recognizable digits.

**Likely cause:** The L2 diversity metric measures pairwise distance in pixel space. Random, unstructured outputs are all different from each other (high L2), while recognizable digits share common visual structures — strokes, curves, consistent backgrounds — that reduce pairwise L2 distance. This metric fundamentally cannot distinguish "diverse because noise" from "diverse because rich generation." A perceptual metric such as FID would better separate quality from randomness.

### Failure Mode 3: GAN Limited Diversity at Short Training

The DCGAN baseline (Figure 1), trained for only 1 epoch (401 steps), produces recognizable digits but shows signs of limited diversity. Some samples cluster around similar digit styles. The discriminator loss drops quickly while the generator loss rises early in training, consistent with the discriminator learning faster than the generator.

**Likely cause:** With very short training, the adversarial equilibrium is never reached. The generator has not yet explored the full data distribution.

### Limitations

- **Single-epoch training:** All diffusion runs use only 1 epoch. This is sufficient for MNIST to reveal trends but may not represent converged behavior, especially for larger models that need more passes.
- **Diversity proxy is unreliable for underfitted models:** As demonstrated above, pixel-space L2 reports artificially high diversity for noisy/failed outputs. A perceptual metric (e.g., FID) would be more appropriate.
- **MNIST simplicity:** MNIST is low-complexity. The strong advantage of x0 over eps in this short-training regime may not generalize to harder datasets where x0 regression targets are more difficult to learn.

---

## 5. Conclusions

1. **x0 parameterization visually outperforms eps in this short-training regime.** At every `base_ch` level, x0 produces more recognizable and structured digits, while eps remains noisier and less stable — especially at low capacity. This is the clearest qualitative finding of the experiment.

2. **Model capacity (base_ch) improves sample quality for both parameterizations,** with a larger apparent benefit for eps. At ch=32, eps largely fails while x0 already produces digits; by ch=128, both improve, although x0 still appears visually stronger.

3. **The eps/ch=32 failure is consistent with error propagation through the reverse process.** Noise-prediction errors may compound across 200 sequential denoising steps under underfitting, while direct x0 prediction provides a more accessible learning target that yields digit-like structure even before convergence.

4. **The diversity proxy is unreliable for underfitted models.** eps/ch=32 receives the highest diversity score despite producing the worst samples, because random unstructured outputs are highly separated in pixel space. This highlights a fundamental limitation of pixel-space diversity metrics.

5. **Runtime scales super-linearly with base_ch** (~38 s at ch=32, ~87 s at ch=64, ~250 s at ch=128), while the choice of target parameterization has negligible impact on training time.

6. **Different generative paradigms fail differently under short training.** The GAN produces recognizable but low-diversity digits, while diffusion (especially eps) tends toward noisy or blob-like outputs. This contrast illustrates how the training objective shapes the failure mode.

7. **Next step:** Extending training beyond 1 epoch would test whether eps eventually catches up to x0 — as suggested by the DDPM literature, which favors eps for longer training — or whether x0 retains its advantage in short, simple settings like MNIST.
