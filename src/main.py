"""
main.py - run the whole pipeline end to end.

    python src/main.py

Steps:
  1. generate synthetic order data           -> data/orders.csv
  2. run elasticity + experiment + guardrails -> reports/results.json, reports/figures/*.png
  3. build the interactive dashboard          -> reports/dashboard.html
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, "src")
import pandas as pd
import generate_data, analysis, build_dashboard


def run():
    # Ensure output folders exist (git does not track empty folders, so a fresh
    # clone has no data/ or reports/figures/ until we create them here).
    for d in ("data", "reports", "reports/figures"):
        os.makedirs(d, exist_ok=True)

    print("1/3  generating synthetic q-commerce orders ...")
    df = generate_data.generate()
    df.to_csv("data/orders.csv", index=False)

    print("2/3  analysing (elasticity, experiment, guardrails, optimisation) ...")
    results = analysis.analyze(df)

    print("3/3  building interactive dashboard ...")
    build_dashboard.build()

    e, o = results["experiment"], results["optimization"]
    print("\nHEADLINES")
    print(f"  naive lift        : {e['naive_lift_pp']} pp (CI {e['naive_ci_pp']})  <- the trap")
    print(f"  true (stratified) : +{e['stratified_lift_pp']} pp")
    print(f"  CUPED var cut     : {e['cuped_variance_reduction_pct']}% on revenue/exposure")
    print(f"  margin policy     : {o['policy_summary']}")
    print(f"  projected upside  : AED {o['projected_monthly_margin_uplift_aed']:,}/month")
    print("\nOpen reports/dashboard.html in a browser.")


if __name__ == "__main__":
    run()
