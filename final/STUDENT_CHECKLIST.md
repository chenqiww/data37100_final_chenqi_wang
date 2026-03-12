# DATA 37100 — Final Project Student Checklist

Use this checklist **before submitting**. Most point losses come from small omissions.

---

## Scope & Framing

- [ ] My project investigates **one clear question** about model behavior
- [ ] I ran **at least two model families** (GAN/DCGAN, Transformer, Diffusion)
- [ ] I ran **one controlled experiment** varying **exactly two knobs**
- [ ] My total experiment runs are small (recommended: ≤ 6)
- [ ] My training runs are fast (target: minutes; avoid multi-hour runs)

---

## Code & Reproducibility

From a clean clone, a TA should be able to run:

- [ ] Baseline run command(s) work
- [ ] Controlled experiment command works
- [ ] All outputs write under `./untrack/`
- [ ] Random seeds are fixed (where applicable)
- [ ] No large data or checkpoints committed to git

**README must include:**

- [ ] Exact run commands
- [ ] Expected runtime + hardware assumptions (CPU/MPS/CUDA)
- [ ] Any special setup notes

---

## Required Artifacts

### Repo

- [ ] Runnable code
- [ ] README with instructions
- [ ] Saved sample outputs (images/text) under `./untrack/`

### Analysis notebook

- [ ] Baseline results shown for **two** model families
- [ ] Two-knob comparison shown (logs + samples)
- [ ] Side-by-side sample visualization
- [ ] Short written interpretations tied to evidence
- [ ] At least one failure mode + likely cause

### Summary report (~3–5 pages)

- [ ] Question + motivation
- [ ] Methods (what models, what knobs)
- [ ] Results (figures / grids / text samples)
- [ ] Failure modes + limitations
- [ ] Conclusions

> No video is required for DATA 37100.

---

## Common Pitfalls (avoid these)

- Changing more than two knobs in the controlled experiment
- Running huge models instead of analyzing behavior
- Showing samples without interpretation
- Missing seeds → non-reproducible results
- Repo that only runs on your machine
- Claims not supported by outputs

---

## What an A-level project typically shows

- Tight question
- Clean ≤6-run experiment grid
- Clear side-by-side comparisons
- Specific failure mode analysis
- Fully reproducible repo

---

If in doubt: simplify and analyze more deeply.
