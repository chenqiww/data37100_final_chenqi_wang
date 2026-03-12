# Example of a Strong Final Project (Template)

This is a *template example* showing what a strong, well-scoped **DATA 37100** final project could look like.

Do **not** copy this directly. Use it as a structure checklist.

---

## 1. Title

**“Small Models, Big Behaviors: Decoding Controls vs. Diffusion Steps on Fashion-MNIST”**

---

## 2. Core question (one sentence)

> How do *inference-time controls* (transformer decoding) compare to *generation-time controls* (diffusion step count) in the kinds of failures they produce?

Tight, behavior-focused, and answerable in ~20 hours.

---

## 3. Models used (at least two)

- Transformer (tiny character LM) — demonstrates decoding knobs
- Diffusion (MNIST/FashionMNIST) — demonstrates schedule/step knobs

(You could swap one for **DCGAN** if that fits your question better.)

---

## 4. Dataset choice

- **Fashion-MNIST** (fast, simple, lots of “confusable” classes)
- Chosen because it quickly shows artifacts, ambiguity, and mode coverage issues.

---

## 5. Baselines (required)

### Transformer baseline
- command used
- runtime
- a few short sample outputs at default decoding

### Diffusion baseline
- command used
- runtime
- one saved grid at default `T=200`

---

## 6. Controlled experiment (exactly two knobs)

A good two-knob study is small and clean.

### Example experiment (Diffusion)
Knob 1: `T ∈ {100, 200, 400}`  
Knob 2: `target ∈ {eps, x0}`  

Total runs: 3 × 2 = 6 (fits scope)

For each run:
- show a sample grid
- show a loss curve (or basic training log)
- write a short interpretation

---

## 7. Evidence and interpretation

Good projects tie claims to evidence.

### What changed?
- At low `T`, samples look blurrier / miss fine structure.
- `x0` target might produce different sharpness vs stability tradeoffs.

### Why (mechanistically)?
- Fewer steps reduces iterative refinement.
- Parameterization changes what the network is directly optimizing.

### Failure mode (must include at least one)
- “Class confusion” and “texture collapse” at low `T`
- Occasional artifacts / repeated motifs

---

## 8. Report outline (~3–5 pages)

1. Question + motivation  
2. Methods (models, dataset, knobs)  
3. Results (figures + sample grids)  
4. Failure modes + limitations  
5. Conclusion  

---

## 9. Repo checklist

- `README.md` with exact run commands
- outputs under `./untrack/`
- no data/checkpoints committed
- one analysis notebook that reproduces the figures/grids

---

## 10. What makes it “A-level”

- one tight question
- two model families demonstrated
- ≤6-run controlled experiment
- evidence-based interpretation
- explicit failure mode analysis
- reproducible repo
