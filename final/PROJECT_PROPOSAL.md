# DATA 37100 — Final Project Proposal (Required)

Length: ~1 page

## 1) Model families (pick at least two)

Check **at least two**:

☑ GAN (DCGAN)  
☐ Transformer  
☑ Diffusion  



---

## 2) Dataset

**MNIST** 


---

## 3) Your core question (one sentence)

> How do target parameterization (eps vs x0) and model capacity (base_ch) affect diffusion sample quality on MNIST?

---

## 4) Controlled variables (two-knob study)

- Knob 1: **target** ∈ {eps, x0} — whether the model predicts the added noise or the clean image directly
- Knob 2: **base_ch** ∈ {32, 64, 128} — the base channel width of the U-Net, controlling model capacity

All other hyperparameters are held fixed: T=200, beta1=1e-4, beta2=0.02, lr=2e-4, epochs=1, bs=128, seed=42.

Planned grid (2 × 3 = 6 runs):

| target \ base_ch | 32 | 64 | 128 |
|-------------------|----|----|-----|
| eps               | run 1 | run 2 | run 3 |
| x0                | run 4 | run 5 | run 6 |

---

## 5) Evidence & evaluation plan

Quantitative signals:
- Training runtime comparison across configurations 
- Pixel-space pairwise diversity proxy (average L2 distance between generated samples)

Qualitative inspection:
- Side-by-side sample grids (8×8) for each of the 6 runs
- Denoising-step intermediate visualizations to compare convergence behavior

---

## 6) Risks & fallback plan

- **Risk:** One-epoch training may not be enough for larger models (base_ch=128) to converge, producing uninformative blurry samples across all runs.
  **Fallback:** Increase to 2 epochs if needed (still under 10 minutes per run on CPU).

- **Risk:** eps vs x0 may produce negligibly different results on simple MNIST digits.
  **Fallback:** Shift Knob 1 to T ∈ {100, 400} instead, which is expected to produce more visible quality differences.

---

