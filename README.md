**▶ Live dashboard:** https://akankshaswarnim.github.io/qcommerce-pricing-experimentation-engine/reports/dashboard.html
**▶ Interactive simulator:** https://akankshaswarnim.github.io/qcommerce-pricing-experimentation-engine/reports/simulator.html


# Q-Commerce Pricing &amp; Demand Experimentation Engine

**An end-to-end pricing analytics project for a UAE quick-commerce (grocery delivery) platform — price elasticity, A/B experimentation, and margin guardrails that turn a price decision into a defensible, P&amp;L-aware recommendation.**

> This model does not just report numbers. It acts as a decision gate: it tells a pricing team *whether to ship a discount, to whom, and how deep* — and it catches the statistical traps (Simpson's Paradox, under-powered reads, margin dilution) that would otherwise turn a "win" into a quiet loss.

Built for the kind of pricing / product-analytics work done at platforms like **talabat, Noon, Careem and Deliveroo** in the UAE. All figures are AED-denominated and the scenarios (delivery zones, daypart demand, Ramadan evening surge) reflect the local market.

---

## TL;DR — what this project demonstrates

| Capability | What I did | Result on the synthetic data |
|---|---|---|
| **Causal experiment reading** | Detected & corrected **Simpson's Paradox** caused by uneven treatment allocation across zones | Naive read: **−0.09pp** (CI crosses zero, "looks dead"). True stratified lift: **+4.46pp** |
| **Variance reduction** | Applied **CUPED** to the revenue-per-exposure metric using a pre-period spend covariate | **5.1%** variance cut → tighter CIs, faster reads |
| **Price elasticity** | Logistic demand model + segment demand curves | New customers **elastic (e≈−2.3)**, loyal **inelastic (e≈−0.5)** |
| **Margin guardrails** | Stratified margin/exposure, fairness, statistical power (MDE) | Blanket promo **fails** the margin guardrail (−AED 3.90/exposure) |
| **Pricing optimisation** | Margin-optimal discount per segment from a randomised natural experiment | Projected **~AED 2.0M/month** margin recovered by right-sizing discounts |

**Headline recommendation:** *Do not ship the flat 10% promo.* It genuinely lifts conversion, but on thin grocery margins it does not pay for itself. Treat discounting as a **targeted acquisition lever** for the elastic *new* segment (justified by LTV), and **stop discounting inelastic loyal customers**.

---

## Why this problem matters (the business context)

Quick-commerce in the UAE runs on **thin gross margins (~20–25%)** and **fierce promo competition**. Pricing and growth teams constantly run discount experiments, and two failure modes are everywhere:

1. **Reading experiments wrong.** Promotions are rarely allocated perfectly evenly. When allocation correlates with a zone's baseline conversion, the *aggregate* result can point the opposite way to the truth — **Simpson's Paradox**. Teams kill working levers, or ship dead ones.
2. **Winning conversion while losing money.** A discount almost always lifts conversion. The real question is whether the incremental volume covers the margin given away — and *for which customers*. A blanket discount over-pays the customers who would have bought anyway.

This project builds the machinery to answer both correctly.

---

## What the engine does (the three analytical blocks)

### 1. Price elasticity — *how much does demand move when price moves?*
A logistic demand model (`converted ~ log(price) + zone + segment + daypart + ramadan`) yields an elasticity of conversion. **Why control for zone/segment/daypart?** Because richer zones see higher prices *and* convert better — without controls, elasticity comes out with the wrong sign. Segment-level demand curves then expose that **new customers are ~5× more price-sensitive than loyal ones**, which is the entire justification for differentiated pricing.

### 2. Experiment readout — *did the promo work, and by how much?*
- **Naive aggregate** read (deliberately shown as the trap).
- **Simpson's-Paradox correction** via post-stratification: re-weight each zone's lift by its exposure share to strip out the allocation bias.
- **CUPED** variance reduction on the revenue metric using a pre-period covariate, so the effect can be called with less traffic.

### 3. Guardrails — *is the win real, safe, and margin-accretive?*
- **G1 Margin** — *stratified* incremental gross margin per exposure (the naive group means are themselves confounded by allocation — a subtle trap most candidates miss).
- **G2 Targeting** — per-segment margin delta → who the promo actually pays off for.
- **G3 Fairness** — lift positive in every zone, no zone harmed.
- **G4 Power** — minimum detectable effect (MDE) vs. the observed lift, so we don't over-read noise.

Plus a **pricing-optimisation** layer: since organic discounts were randomly assigned (a clean natural experiment), the margin-optimal discount depth per segment is recoverable, with discounts capped at the observed range (**no extrapolation beyond support**).

---

## Edge cases &amp; guardrails handled (and why)

| Risk | How the project handles it |
|---|---|
| **Simpson's Paradox / confounded allocation** | Post-stratification by zone; never trust the raw aggregate. |
| **Confounded elasticity** | Control for zone, segment, daypart, Ramadan in the demand model. |
| **High-variance revenue metric** | CUPED with a pre-period covariate to tighten CIs. |
| **Margin dilution hidden by a conversion win** | Stratified margin/exposure guardrail; report margin delta, not just conversion. |
| **Cannibalisation** | Flagged: high base conversion ⇒ many promo recipients would convert anyway ⇒ target, don't blanket. |
| **Under-powered reads** | Compute MDE; only call effects comfortably above it. |
| **Extrapolating a price curve** | Optimisation capped to the tested 0–20% discount range. |
| **Data privacy / API deprecation** | 100% synthetic, deterministic-seed data — no real customer data, no dependency on deprecated platform APIs. |

---

## Results &amp; visuals

The interactive dashboard (`reports/dashboard.html`) has KPI cards, hoverable charts, guardrail chips, and the final recommendation. Static figures live in `reports/figures/`:

- `01_elasticity_by_segment.png` — price sensitivity per segment
- `02_simpsons_paradox.png` — naive vs. stratified lift
- `03_margin_guardrail.png` — margin delta overall vs. by segment
- `04_discount_optimization.png` — margin-optimal discount per segment

---

## How to run

```bash
pip install -r requirements.txt
python src/main.py          # generates data, runs analysis, builds dashboard
# then open reports/dashboard.html
```

Reproducible end to end: a fixed random seed means every run yields identical numbers.

## Repo structure

```
qcommerce-pricing-experimentation-engine/
├── README.md
├── requirements.txt
├── src/
│   ├── generate_data.py     # synthetic UAE q-commerce orders (documented DGP)
│   ├── analysis.py          # elasticity, experiment (Simpson + CUPED), guardrails, optimisation
│   ├── build_dashboard.py   # self-contained interactive Plotly dashboard
│   └── main.py              # one-command pipeline
├── data/                    # generated orders.csv (git-ignored)
└── reports/
    ├── results.json         # all computed metrics
    ├── dashboard.html       # interactive dashboard
    └── figures/             # static PNGs
```

## Tech stack
Python · pandas · NumPy · statsmodels · scikit-learn · SciPy · Plotly · Matplotlib

---

*All data is synthetic and generated locally; metrics and AED figures are illustrative and intended to demonstrate methodology, not to represent any real company's results.*
