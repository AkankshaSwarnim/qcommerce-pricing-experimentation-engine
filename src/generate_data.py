"""
generate_data.py
================
Synthetic data generator for a UAE q-commerce (quick-commerce grocery) pricing
problem. Everything here is *simulated* with a fixed random seed so the entire
analysis is 100% reproducible and contains NO real customer data.

WHY SYNTHETIC?
--------------
Real q-commerce pricing logs are confidential and, for talabat-style apps, the
public APIs that would expose order-level price/conversion data are not
available. A deterministic synthetic generator lets us demonstrate the full
methodology (elasticity, experimentation, guardrails) on data whose ground truth
we control - which is exactly what lets us *prove* the model recovers the right
answer.

WHAT WE MODEL (the data-generating process, or "DGP")
-----------------------------------------------------
Each row is one "order opportunity" (a customer opening the app with intent).
The customer converts (places the order) with a probability that depends on:
  * the effective price they see (base price minus any discount),
  * how price-sensitive their segment is,
  * the delivery zone (Downtown converts better than Deira, etc.),
  * the daypart (dinner peak converts better than mid-afternoon),
  * whether it is Ramadan (evening demand surges after iftar).

We then layer an A/B EXPERIMENT on top: a promotional treatment that grants an
extra discount. Critically, we allocate the treatment UNEVENLY across zones -
this deliberately manufactures Simpson's Paradox so the analysis can show how to
detect and correct it (a classic interview talking point).

OUTPUT
------
data/orders.csv  - one row per order opportunity.
"""

import numpy as np
import pandas as pd

SEED = 42
N = 60_000  # order opportunities; large enough for stable A/B statistics

# --- Business primitives -------------------------------------------------------
# Delivery zones with a baseline "intent-to-convert" propensity. These encode the
# real-world fact that affluent, dense areas convert at a higher base rate.
ZONES = {
    "Downtown":     {"base": 0.55, "basket_aed": 95},
    "Marina":       {"base": 0.50, "basket_aed": 90},
    "Business Bay": {"base": 0.45, "basket_aed": 85},
    "JLT":          {"base": 0.38, "basket_aed": 75},
    "Deira":        {"base": 0.30, "basket_aed": 60},
}

# Customer segments differ in how strongly price moves their decision.
# price_sensitivity is the weight on the relative price gap in the utility model:
# higher = more elastic (discounts move them more).
SEGMENTS = {
    "loyal":  {"share": 0.30, "price_sensitivity": 1.2, "base_adj":  0.35},
    "casual": {"share": 0.45, "price_sensitivity": 2.4, "base_adj":  0.00},
    "new":    {"share": 0.25, "price_sensitivity": 3.6, "base_adj": -0.30},
}

# Daypart demand multipliers (additive in log-odds).
DAYPARTS = {"morning": -0.10, "midday": -0.30, "evening_peak": 0.45, "late_night": -0.20}

# Organic discount depth offered by merchandising (independent of the experiment).
# This natural variation is what lets us estimate price elasticity cleanly.
DISCOUNT_LEVELS = np.array([0.00, 0.05, 0.10, 0.15, 0.20])

MARGIN_RATE = 0.25  # gross margin on the *undiscounted* basket; the guardrail anchor


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    zone_names = np.array(list(ZONES.keys()))
    zone = rng.choice(zone_names, size=n, p=[0.24, 0.22, 0.20, 0.18, 0.16])

    seg_names = np.array(list(SEGMENTS.keys()))
    segment = rng.choice(seg_names, size=n, p=[SEGMENTS[s]["share"] for s in seg_names])

    daypart = rng.choice(list(DAYPARTS.keys()), size=n, p=[0.20, 0.25, 0.40, 0.15])
    is_ramadan = rng.random(n) < 0.18  # ~18% of the window falls in Ramadan

    base_price = np.array([ZONES[z]["basket_aed"] for z in zone]) * rng.normal(1.0, 0.08, n)
    base_price = np.clip(base_price, 35, 200)

    organic_discount = rng.choice(DISCOUNT_LEVELS, size=n, p=[0.45, 0.22, 0.18, 0.10, 0.05])

    # --- A/B EXPERIMENT DESIGN -------------------------------------------------
    # Treatment = an extra 10% promotional discount on top of any organic discount.
    # We assign treatment with a probability that DEPENDS ON ZONE, over-weighting
    # the low-converting zones. This imbalance is what produces Simpson's Paradox.
    treat_prob_by_zone = {"Downtown": 0.30, "Marina": 0.35, "Business Bay": 0.45,
                          "JLT": 0.62, "Deira": 0.70}
    p_treat = np.array([treat_prob_by_zone[z] for z in zone])
    group = np.where(rng.random(n) < p_treat, "treatment", "control")
    promo_discount = np.where(group == "treatment", 0.10, 0.00)

    total_discount = np.clip(organic_discount + promo_discount, 0, 0.45)
    effective_price = base_price * (1 - total_discount)

    # Reference price per zone (what a customer "expects" to pay) - the anchor for
    # the relative price gap that drives elasticity.
    ref_price = np.array([ZONES[z]["basket_aed"] for z in zone])
    rel_price_gap = (effective_price - ref_price) / ref_price  # negative = cheaper than expected

    # --- Utility / conversion model -------------------------------------------
    base_logit = np.log(np.array([ZONES[z]["base"] for z in zone]) /
                        (1 - np.array([ZONES[z]["base"] for z in zone])))
    seg_adj = np.array([SEGMENTS[s]["base_adj"] for s in segment])
    seg_sens = np.array([SEGMENTS[s]["price_sensitivity"] for s in segment])
    day_adj = np.array([DAYPARTS[d] for d in daypart])
    ramadan_adj = np.where(is_ramadan & (daypart == "evening_peak"), 0.55, 0.0)

    # Customer-value latent: real customer bases are highly heterogeneous (some
    # convert ~10%, some ~90%). Widening this spread is what makes a pre-period
    # covariate genuinely useful for CUPED.
    z_value = rng.normal(0, 0.9, n)

    logit = (base_logit + seg_adj + day_adj + ramadan_adj
             + 0.8 * z_value                                 # high-value customers convert more
             - seg_sens * rel_price_gap                      # cheaper -> higher conversion
             + rng.normal(0, 0.15, n))                        # irreducible noise
    p_convert = _sigmoid(logit)
    converted = (rng.random(n) < p_convert).astype(int)

    # --- Pre-period covariate for CUPED ---------------------------------------
    # A user-level metric measured BEFORE the experiment, correlated with the
    # outcome. Using it as a covariate cuts variance without introducing bias.
    user_quality = _sigmoid(base_logit + seg_adj + 0.8 * z_value)
    pre_period_orders = rng.poisson(lam=2 + 6 * user_quality)
    # Pre-period SPEND (AED, prior 30 days) loads on the same value latent as the
    # in-experiment revenue, so it is a strong CUPED covariate for the revenue KPI.
    pre_period_spend = np.exp(3.4 + 0.85 * z_value) * rng.lognormal(0, 0.20, n)

    # --- Economics -------------------------------------------------------------
    cost = base_price * (1 - MARGIN_RATE)                      # unit cost is fixed
    revenue = converted * effective_price
    gross_margin = converted * (effective_price - cost)        # can go negative on deep discounts

    df = pd.DataFrame({
        "order_id": np.arange(n),
        "zone": zone,
        "segment": segment,
        "daypart": daypart,
        "is_ramadan": is_ramadan,
        "base_price_aed": base_price.round(2),
        "organic_discount": organic_discount,
        "promo_discount": promo_discount,
        "total_discount": total_discount.round(3),
        "effective_price_aed": effective_price.round(2),
        "experiment_group": group,
        "pre_period_orders": pre_period_orders,
        "pre_period_spend": pre_period_spend.round(2),
        "converted": converted,
        "revenue_aed": revenue.round(2),
        "gross_margin_aed": gross_margin.round(2),
        "true_p_convert": p_convert.round(4),  # ground truth - for validation only
    })
    return df


if __name__ == "__main__":
    out = generate()
    out.to_csv("data/orders.csv", index=False)
    print(f"Wrote data/orders.csv  ({len(out):,} rows)")
    print(out.groupby("experiment_group")["converted"].mean())
