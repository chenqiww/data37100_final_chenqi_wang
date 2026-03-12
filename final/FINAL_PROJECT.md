# DATA 37100 — Final Project (Winter 2026)

**Expected effort:** ~20 hours per student  
**Work mode:** Individual or pairs  
**Due:** March 13, 2026

---

## Goal

Build a *small*, *fast*, and *interpretable* generative AI study.

You will use the models we covered in DATA 37100 to answer a focused question about **model behavior**:
- What does the model learn easily vs. struggle with?
- How do decoding / schedules / objectives change outputs?
- What failure modes show up, and why?

This is not a “train the biggest thing” contest. It’s a “learn the most from the smallest thing” contest.

---

## Required model coverage (NEW)

You must run **at least two** of the three model families:

- **Transformer** (Week 5)  
- **Diffusion** (Week 7)  
- **GAN (DCGAN)** (Week 4)

Starter baselines are provided in `final/starter/src/`.

---

## Required components

### 1) Two working baselines (required)

Show that you can run **two** model families end-to-end and produce samples.

Minimum:
- one baseline run per chosen family
- saved outputs (images and/or text)
- brief notes on runtime + what “good enough” looks like

### 2) One controlled experiment (required)

Design **one** controlled experiment that varies **exactly two** meaningful factors (a “two‑knob” study).

You may do this:
- within a single model family, **or**
- as a comparison across two families (recommended only if you keep it tight)

Examples of knobs:
- **Transformer:** temperature, top‑p, top‑k, repetition penalty, context length  
- **Diffusion:** timesteps `T`, noise schedule, target (`eps` vs `x0`), model width  
- **GAN/DCGAN:** learning rate, discriminator steps, label smoothing, latent dim, model width  

Keep it fast:
- recommend **≤ 6 total runs** for the experiment
- each run should be reasonably short (**target: minutes**, not hours)

### 3) Analysis + failure modes (most important)

Your write-up must explain:
- what changed (evidence: logs + samples)
- why it changed (mechanistic reasoning, not vibes)
- at least **one failure mode** you observed (and what likely caused it)

### 4) Deliverables (match syllabus)

Submit a Git repo (or a repo folder you zip) containing:

**A. Technical analysis + code (required)**
- your runnable training / sampling code
- saved sample outputs under `./untrack/`

**B. Repository hygiene (required)**
- clear `README.md` with:
  - setup notes (minimal)
  - exact run commands
  - expected runtime and hardware assumptions
- no large data committed

**C. Summary report (required) — ~3–5 pages**
- question + motivation
- methods (what you ran, what knobs)
- results (figures / sample grids / text samples)
- failure modes + limitations
- conclusion

> No video is required for DATA 37100.

---

## Constraints (please respect your future self)

- Use **approved datasets** (see `final/DATASETS.md`)
- Keep training fast and small
- Focus on *behavioral insight* over scale
- If something breaks, simplify before adding features

---

## Grading emphasis

Highest weight:
- depth of reasoning and interpretation
- clean controlled experimentation
- clear evidence (plots / grids / samples)

Lower weight:
- model size
- fancy UI / polish

---

If you’re unsure: pick two models, run small baselines, then do one tight two‑knob experiment and explain it well.
