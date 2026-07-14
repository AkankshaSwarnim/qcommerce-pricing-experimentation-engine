"""
analysis.py
===========
The analytical core. Three blocks, each answering a real pricing question:

  1. PRICE ELASTICITY  - "How much does demand move when we change price?"
  2. EXPERIMENT READOUT - "Did the 10%-off promo actually work, and by how much?"
                          (incl. Simpson's-Paradox correction + CUPED variance reduction)
  3. GUARDRAILS         - "Is the win real, safe, and margin-accretive - or are we
                          buying conversions we'd have gotten anyway?"

Every function is commented with WHAT it does, WHY it matters commercially, and
the EDGE CASES / GUARDRAILS it protects against.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats

PALETTE = {"control": "#9aa0a6", "treatment": "#ff5a36", "accent": "#1f3864", "ok": "#1a936f"}
DISCOUNT_LEVELS = [0.00, 0.05, 0.10, 0.15, 0.20]


# ---------------------------------------------------------------------------
# 1. PRICE ELASTICITY
# ---------------------------------------------------------------------------
def estimate_elasticity(df: pd.DataFrame) -> dict:
    """
    WHAT: Fit a logistic demand model and convert it into a price elasticity of
          conversion - the % change in conversion for a 1% change in price.
    WHY:  Elasticity is the single number pricing teams live by. Elastic segments
          (|e|>1) respond strongly to discounts; inelastic ones (|e|<1) do not, so
          discounting them just burns margin.
    HOW:  Logit(convert) ~ log(price) + zone + segment + daypart + ramadan.
          The coefficient on log(price) is the semi-elasticity in log-odds; we
          convert it to an elasticity of the conversion probability at the mean.
    EDGE CASE: price and conversion are confounded by zone/segment/daypart (richer
          zones see higher prices AND convert more). We MUST control for them, or
          elasticity comes out with the wrong sign. That control is the whole point.
    """
    model = smf.logit(
        "converted ~ np.log(effective_price_aed) + C(zone) + C(segment) + C(daypart) + is_ramadan",
        data=df,
    ).fit(disp=False)

    b = model.params["np.log(effective_price_aed)"]
    p_bar = df["converted"].mean()
    elasticity = b * (1 - p_bar)  # elasticity of the conversion probability at the mean

    # Segment-level demand curves (observed conversion at each organic discount depth).
    seg_curves = {}
    seg_elasticity = {}
    for seg, g in df[df["promo_discount"] == 0].groupby("segment"):  # exclude promo to isolate organic price
        curve = g.groupby("organic_discount")["converted"].mean().reindex(DISCOUNT_LEVELS)
        seg_curves[seg] = curve.round(4).to_dict()
        # crude arc elasticity between 0% and 20% discount
        p0, p20 = curve.get(0.0), curve.get(0.20)
        if p0 and p20 and p0 > 0:
            pct_q = (p20 - p0) / p0
            pct_p = (-0.20)  # 20% lower price
            seg_elasticity[seg] = round(pct_q / pct_p, 2)

    return {
        "logit_log_price_coef": round(b, 3),
        "elasticity_at_mean": round(elasticity, 3),
        "interpretation": ("demand is ELASTIC (discounts move volume)" if abs(elasticity) >= 1
                           else "demand is INELASTIC (discounting mostly erodes margin)"),
        "segment_curves": seg_curves,
        "segment_arc_elasticity": seg_elasticity,
    }


# ---------------------------------------------------------------------------
# 2. EXPERIMENT READOUT
# ---------------------------------------------------------------------------
def _prop_ci(p, n, z=1.96):
    se = np.sqrt(p * (1 - p) / n)
    return p - z * se, p + z * se


def run_experiment(df: pd.DataFrame) -> dict:
    c = df[df.experiment_group == "control"]
    t = df[df.experiment_group == "treatment"]

    # --- 2a. NAIVE aggregate read (the trap) -------------------------------
    pc, pt = c.converted.mean(), t.converted.mean()
    naive_lift = pt - pc
    se = np.sqrt(pc * (1 - pc) / len(c) + pt * (1 - pt) / len(t))
    naive_ci = (naive_lift - 1.96 * se, naive_lift + 1.96 * se)

    # --- 2b. SIMPSON'S PARADOX correction via post-stratification ----------
    # WHY: treatment was over-allocated to low-converting zones, dragging the
    #      aggregate down. Re-weighting each zone's lift by its share of all
    #      exposures removes the allocation bias.
    zone_rows = []
    weighted_lift, weight_total = 0.0, 0.0
    for z, g in df.groupby("zone"):
        gc, gt = g[g.experiment_group == "control"], g[g.experiment_group == "treatment"]
        lift = gt.converted.mean() - gc.converted.mean()
        w = len(g)
        weighted_lift += lift * w
        weight_total += w
        zone_rows.append({"zone": z, "control": round(gc.converted.mean(), 4),
                          "treatment": round(gt.converted.mean(), 4),
                          "lift_pp": round(lift * 100, 2),
                          "treat_share": round((g.experiment_group == "treatment").mean(), 2)})
    stratified_lift = weighted_lift / weight_total

    # --- 2c. CUPED variance reduction --------------------------------------
    # WHAT: use pre-period SPEND (measured BEFORE the test) to strip baseline
    #       customer-value noise out of the revenue-per-exposure metric.
    # WHY:  revenue/exposure is high-variance (most exposures = 0, converts vary
    #       60-95 AED). CUPED tightens its CI sharply, so we can read the revenue
    #       effect with far less traffic - critical on a fast promo calendar.
    # NOTE: CUPED on a ~50% binary metric barely helps (variance reduction = rho^2,
    #       and rho is capped for a near-even coin); applying it to the continuous
    #       revenue metric is where it earns its keep.
    x = df["pre_period_spend"].values
    y = df["revenue_aed"].values
    theta = np.cov(y, x, bias=True)[0, 1] / np.var(x)
    y_cuped = y - theta * (x - x.mean())
    df = df.assign(_y_cuped=y_cuped)
    var_before = df.groupby("experiment_group")["revenue_aed"].var().mean()
    var_after = df.groupby("experiment_group")["_y_cuped"].var().mean()
    var_reduction = 1 - var_after / var_before

    return {
        "n_control": len(c), "n_treatment": len(t),
        "control_rate": round(pc, 4), "treatment_rate": round(pt, 4),
        "naive_lift_pp": round(naive_lift * 100, 2),
        "naive_ci_pp": [round(naive_ci[0] * 100, 2), round(naive_ci[1] * 100, 2)],
        "stratified_lift_pp": round(stratified_lift * 100, 2),
        "cuped_variance_reduction_pct": round(var_reduction * 100, 1),
        "cuped_metric": "revenue per exposure (AED)",
        "zone_table": zone_rows,
        "verdict": ("Naive read understates the true effect ~"
                    f"{stratified_lift/naive_lift:.0f}x due to allocation bias (Simpson's Paradox)."),
    }


# ---------------------------------------------------------------------------
# 3. GUARDRAILS  (the part that separates an analyst from a junior)
# ---------------------------------------------------------------------------
def optimize_discount(df: pd.DataFrame) -> dict:
    """
    WHAT: Find the margin-maximising discount depth for EACH segment.
    WHY:  Elasticity says new customers respond to price and loyal customers do
          not. So a single company-wide discount is always wrong - it over-discounts
          the loyal (pure margin loss) and may under-discount the elastic new.
    HOW:  organic_discount was RANDOMLY assigned in the data (a clean natural
          experiment), so mean gross-margin-per-exposure at each discount level is
          an unbiased estimate of that policy's payoff. Pick the argmax per segment.
    EDGE CASE / GUARDRAIL: we optimise on the control population only (no promo
          overlay) to keep the price signal clean, and we cap discounts at the
          tested range (0-20%) - never extrapolate a curve beyond observed support.
    """
    base = df[df.promo_discount == 0]  # clean organic-price population
    curves, optimal = {}, {}
    seg_share = df.segment.value_counts(normalize=True).to_dict()

    baseline_margin, optimal_margin = 0.0, 0.0
    for seg, g in base.groupby("segment"):
        m = g.groupby("organic_discount")["gross_margin_aed"].mean()
        curves[seg] = {round(float(k), 2): round(float(v), 2) for k, v in m.items()}
        best_disc = float(m.idxmax())
        optimal[seg] = {"optimal_discount": best_disc, "margin_at_optimal": round(float(m.max()), 2)}
        # status-quo = margin under the current observed discount mix for this segment
        sq = g["gross_margin_aed"].mean()
        baseline_margin += sq * seg_share.get(seg, 0)
        optimal_margin += m.max() * seg_share.get(seg, 0)

    uplift_per_exposure = optimal_margin - baseline_margin
    monthly_exposures = 1_000_000
    return {
        "segment_margin_curves": curves,
        "segment_optimal_policy": optimal,
        "baseline_margin_per_exposure_aed": round(baseline_margin, 3),
        "optimal_margin_per_exposure_aed": round(optimal_margin, 3),
        "uplift_per_exposure_aed": round(uplift_per_exposure, 3),
        "projected_monthly_margin_uplift_aed": int(uplift_per_exposure * monthly_exposures),
        "policy_summary": {s: f"{int(o['optimal_discount']*100)}% discount" for s, o in optimal.items()},
    }


def _stratified_delta(df, value_col):
    """Zone-weighted treatment-minus-control delta for a per-exposure metric.
    Removes the allocation bias that contaminates the naive group means."""
    num, den = 0.0, 0.0
    for _, g in df.groupby("zone"):
        gc, gt = g[g.experiment_group == "control"], g[g.experiment_group == "treatment"]
        num += (gt[value_col].mean() - gc[value_col].mean()) * len(g)
        den += len(g)
    return num / den


def evaluate_guardrails(df: pd.DataFrame, exp: dict) -> dict:
    """
    A conversion lift is necessary but NOT sufficient. Before shipping a price cut
    we must check it does not quietly destroy the P&L. Four guardrails:
      G1 MARGIN      - is the STRATIFIED incremental gross margin positive?
      G2 TARGETING   - which segments is the promo actually margin-positive for?
      G3 FAIRNESS    - is the lift positive in EVERY zone (no zone harmed)?
      G4 POWER       - did we have enough sample to trust the call?
    """
    c = df[df.experiment_group == "control"]
    t = df[df.experiment_group == "treatment"]

    # Naive (biased) group means - reported only to show the trap.
    rev_c, rev_t = c.revenue_aed.mean(), t.revenue_aed.mean()
    mar_c, mar_t = c.gross_margin_aed.mean(), t.gross_margin_aed.mean()

    # G1: STRATIFIED deltas (the honest read). Revenue/exposure and margin/exposure
    # are confounded by zone allocation exactly like conversion was.
    strat_rev_delta = _stratified_delta(df, "revenue_aed")
    strat_margin_delta = _stratified_delta(df, "gross_margin_aed")
    g1_pass = strat_margin_delta >= 0

    # G2: per-segment margin delta (zone-weighted within each segment) -> targeting.
    seg_margin = {}
    for seg, gseg in df.groupby("segment"):
        seg_margin[seg] = round(_stratified_delta(gseg, "gross_margin_aed"), 3)
    margin_positive_segments = [s for s, d in seg_margin.items() if d >= 0]

    # G3: fairness - lift positive in every zone?
    g3_pass = all(r["lift_pp"] > 0 for r in exp["zone_table"])

    # G4: power - minimum detectable effect at this sample size (~80% power, 5% alpha)
    p = df.converted.mean()
    n_per_arm = min(exp["n_control"], exp["n_treatment"])
    mde_pp = 2.8 * np.sqrt(2 * p * (1 - p) / n_per_arm) * 100

    # Projected monthly AED impact under a TARGETED rollout (promo only to the
    # segments where it is margin-positive). Assumptions stated openly.
    monthly_exposures = 1_000_000
    target_mask = df.segment.isin(margin_positive_segments) if margin_positive_segments else df.index < 0
    target_share = float(target_mask.mean())
    targeted_margin_delta = (_stratified_delta(df[target_mask], "gross_margin_aed")
                             if target_share > 0 else 0.0)
    incr_margin_aed = monthly_exposures * target_share * targeted_margin_delta

    return {
        "naive_rev_per_exposure": {"control": round(rev_c, 2), "treatment": round(rev_t, 2)},
        "naive_margin_per_exposure": {"control": round(mar_c, 2), "treatment": round(mar_t, 2)},
        "stratified_revenue_delta_aed": round(strat_rev_delta, 3),
        "stratified_margin_delta_aed": round(strat_margin_delta, 3),
        "G1_blanket_margin_accretive": bool(g1_pass),
        "G2_segment_margin_delta_aed": seg_margin,
        "G2_margin_positive_segments": margin_positive_segments,
        "G3_fairness_all_zones_positive": bool(g3_pass),
        "G4_min_detectable_effect_pp": round(mde_pp, 2),
        "projected_monthly_incremental_margin_aed_TARGETED": int(incr_margin_aed),
        "projection_assumptions": ("1,000,000 monthly exposures; promo restricted to "
                                   "margin-positive segments; stratified deltas held. "
                                   "SYNTHETIC DATA - illustrative only."),
    }


# ---------------------------------------------------------------------------
# FIGURES (static PNGs for the README)
# ---------------------------------------------------------------------------
def save_figures(df, ela, exp, guard, outdir="reports/figures"):
    # Fig 1: elasticity / demand curves by segment
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for seg, curve in ela["segment_curves"].items():
        xs = [k * 100 for k in curve]
        ax.plot(xs, list(curve.values()), marker="o", label=f"{seg} (e={ela['segment_arc_elasticity'].get(seg,'n/a')})")
    ax.set_xlabel("Discount depth (%)"); ax.set_ylabel("Conversion rate")
    ax.set_title("Price sensitivity by customer segment"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{outdir}/01_elasticity_by_segment.png", dpi=130); plt.close(fig)

    # Fig 2: Simpson's paradox
    zt = pd.DataFrame(exp["zone_table"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(zt.zone, zt.lift_pp, color=PALETTE["treatment"], alpha=.85, label="Per-zone lift")
    ax.axhline(exp["naive_lift_pp"], color=PALETTE["control"], ls="--", lw=2, label=f"Naive aggregate ({exp['naive_lift_pp']}pp)")
    ax.axhline(exp["stratified_lift_pp"], color=PALETTE["accent"], ls="-", lw=2, label=f"Stratified true ({exp['stratified_lift_pp']}pp)")
    ax.set_ylabel("Conversion lift (pp)"); ax.set_title("Simpson's Paradox: naive read hides a real win")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{outdir}/02_simpsons_paradox.png", dpi=130); plt.close(fig)

    # Fig 3: guardrail - stratified margin delta overall vs by segment (targeting)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    seg_d = guard["G2_segment_margin_delta_aed"]
    labels = ["ALL (blanket)"] + list(seg_d.keys())
    vals = [guard["stratified_margin_delta_aed"]] + list(seg_d.values())
    colors = [PALETTE["ok"] if v >= 0 else PALETTE["treatment"] for v in vals]
    ax.bar(labels, vals, color=colors)
    ax.axhline(0, color="#333", lw=1)
    ax.set_ylabel("Incremental gross margin / exposure (AED)")
    ax.set_title("Guardrail: blanket promo dilutes margin; target the elastic segments")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{outdir}/03_margin_guardrail.png", dpi=130); plt.close(fig)


def analyze(df: pd.DataFrame) -> dict:
    ela = estimate_elasticity(df)
    exp = run_experiment(df)
    guard = evaluate_guardrails(df, exp)
    opt = optimize_discount(df)
    save_figures(df, ela, exp, guard)
    # Fig 4: margin-vs-discount optimisation by segment
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for seg, curve in opt["segment_margin_curves"].items():
        xs = [k * 100 for k in curve]
        ax.plot(xs, list(curve.values()), marker="o", label=seg)
        best = opt["segment_optimal_policy"][seg]["optimal_discount"] * 100
        ax.axvline(best, ls=":", alpha=.4)
    ax.set_xlabel("Discount depth (%)"); ax.set_ylabel("Gross margin / exposure (AED)")
    ax.set_title("Margin-optimal discount differs by segment"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("reports/figures/04_discount_optimization.png", dpi=130); plt.close(fig)

    results = {"elasticity": ela, "experiment": exp, "guardrails": guard, "optimization": opt}
    with open("reports/results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    df = pd.read_csv("data/orders.csv")
    r = analyze(df)
    print(json.dumps(r, indent=2)[:1500])
