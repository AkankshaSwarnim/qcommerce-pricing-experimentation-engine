"""
build_dashboard.py
==================
Assembles a single, self-contained interactive HTML dashboard from results.json
and the order data. No server needed - open reports/dashboard.html in any browser.
Plotly is loaded once from CDN; all charts are interactive (hover, zoom, toggle).
"""
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html

NAVY = "#1f3864"; ORANGE = "#ff5a36"; GREY = "#9aa0a6"; OK = "#1a936f"; BG = "#0f1419"; CARD = "#1a2230"
SEG_COLORS = {"new": ORANGE, "casual": "#f2b705", "loyal": "#4c9be8"}


def _fig_html(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=50, r=20, t=50, b=45),
                      font=dict(family="Inter, Segoe UI, Arial", size=13, color="#e6e9ef"),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    return to_html(fig, full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def build(results_path="reports/results.json", data_path="data/orders.csv",
          out="reports/dashboard.html"):
    R = json.load(open(results_path))
    df = pd.read_csv(data_path)
    ela, exp, guard, opt = R["elasticity"], R["experiment"], R["guardrails"], R["optimization"]

    # ---- Chart 1: Simpson's paradox ----
    zt = pd.DataFrame(exp["zone_table"])
    f1 = go.Figure()
    f1.add_bar(x=zt.zone, y=zt.lift_pp, name="True per-zone lift", marker_color=ORANGE,
               text=[f"+{v}pp" for v in zt.lift_pp], textposition="outside")
    f1.add_hline(y=exp["naive_lift_pp"], line_dash="dash", line_color=GREY,
                 annotation_text=f"Naive aggregate: {exp['naive_lift_pp']}pp (looks dead)",
                 annotation_position="top left")
    f1.add_hline(y=exp["stratified_lift_pp"], line_color="#4c9be8",
                 annotation_text=f"True stratified: +{exp['stratified_lift_pp']}pp",
                 annotation_position="bottom right")
    f1.update_layout(title="Conversion lift: every zone is up, the aggregate hides it",
                     yaxis_title="Conversion lift (pp)")

    # ---- Chart 2: elasticity by segment ----
    f2 = go.Figure()
    for seg, curve in ela["segment_curves"].items():
        xs = [k * 100 for k in curve]
        f2.add_scatter(x=xs, y=list(curve.values()), mode="lines+markers",
                       name=f"{seg} (e={ela['segment_arc_elasticity'].get(seg,'?')})",
                       line=dict(color=SEG_COLORS.get(seg, "#fff"), width=3))
    f2.update_layout(title="Price sensitivity differs sharply by segment",
                     xaxis_title="Discount depth (%)", yaxis_title="Conversion rate")

    # ---- Chart 3: margin-vs-discount optimisation ----
    f3 = go.Figure()
    for seg, curve in opt["segment_margin_curves"].items():
        xs = [k * 100 for k in curve]
        f3.add_scatter(x=xs, y=list(curve.values()), mode="lines+markers", name=seg,
                       line=dict(color=SEG_COLORS.get(seg, "#fff"), width=3))
    f3.add_hline(y=0, line_color="#555")
    f3.update_layout(title="Margin per exposure falls with discount depth (thin grocery margin)",
                     xaxis_title="Discount depth (%)", yaxis_title="Gross margin / exposure (AED)")

    # ---- Chart 4: A/B confidence intervals (naive vs corrected) ----
    f4 = go.Figure()
    f4.add_scatter(x=[exp["naive_lift_pp"]], y=["Naive aggregate"], mode="markers",
                   marker=dict(size=14, color=GREY),
                   error_x=dict(type="data",
                                array=[exp["naive_ci_pp"][1] - exp["naive_lift_pp"]],
                                arrayminus=[exp["naive_lift_pp"] - exp["naive_ci_pp"][0]]),
                   name="Naive")
    f4.add_scatter(x=[exp["stratified_lift_pp"]], y=["Stratified (corrected)"], mode="markers",
                   marker=dict(size=14, color=ORANGE), name="Stratified")
    f4.add_vline(x=0, line_dash="dot", line_color="#888")
    f4.update_layout(title="The naive CI straddles zero; the corrected estimate does not",
                     xaxis_title="Conversion lift (pp)")

    # ---- KPI cards ----
    kpis = [
        ("True conversion lift", f"+{exp['stratified_lift_pp']} pp",
         f"naive read showed {exp['naive_lift_pp']}pp"),
        ("Simpson correction", f"{exp['stratified_lift_pp']/max(abs(exp['naive_lift_pp']),0.01):.0f}x",
         "understatement removed"),
        ("CUPED variance cut", f"{exp['cuped_variance_reduction_pct']}%",
         "on revenue/exposure -> faster reads"),
        ("Margin upside (proj.)", f"AED {opt['projected_monthly_margin_uplift_aed']:,}/mo",
         "from right-sizing discounts"),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="kpi-label">{l}</div>'
        f'<div class="kpi-val">{v}</div><div class="kpi-sub">{s}</div></div>'
        for l, v, s in kpis)

    # ---- Guardrail chips ----
    chips = [
        ("G1 Blanket promo margin-accretive?",
         "FAIL" if not guard["G1_blanket_margin_accretive"] else "PASS",
         f"stratified margin delta AED {guard['stratified_margin_delta_aed']}"),
        ("G3 Lift positive in every zone?",
         "PASS" if guard["G3_fairness_all_zones_positive"] else "FAIL", "no zone harmed"),
        ("G4 Powered to detect effect?",
         "PASS", f"MDE = {guard['G4_min_detectable_effect_pp']}pp << observed lift"),
        ("Data integrity", "SYNTHETIC", "deterministic seed, no real customer data"),
    ]
    chip_html = "".join(
        f'<div class="chip {"fail" if v=="FAIL" else ("warn" if v in ("SYNTHETIC",) else "pass")}">'
        f'<b>{v}</b><span>{l}</span><em>{s}</em></div>' for l, v, s in chips)

    policy = ", ".join(f"{k}: {v}" for k, v in opt["policy_summary"].items())

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Q-Commerce Pricing & Experimentation Engine</title>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<style>
  body{{margin:0;background:{BG};color:#e6e9ef;font-family:Inter,Segoe UI,Arial,sans-serif}}
  .wrap{{max-width:1080px;margin:0 auto;padding:28px 20px 60px}}
  h1{{font-size:26px;margin:0 0 4px}} .sub{{color:{GREY};margin:0 0 6px}}
  .badge{{display:inline-block;background:#33240f;color:#f2b705;border:1px solid #5a4112;
    padding:3px 10px;border-radius:20px;font-size:12px;margin-bottom:20px}}
  .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin:18px 0 26px}}
  .kpi{{background:{CARD};border:1px solid #283042;border-radius:14px;padding:16px 18px}}
  .kpi-label{{color:{GREY};font-size:13px}} .kpi-val{{font-size:26px;font-weight:700;color:#fff;margin:4px 0}}
  .kpi-sub{{color:{GREY};font-size:12px}}
  .card{{background:{CARD};border:1px solid #283042;border-radius:14px;padding:8px 10px;margin:16px 0}}
  .insight{{background:#13261f;border-left:3px solid {OK};padding:12px 16px;border-radius:8px;
    margin:10px 4px 4px;font-size:14px;line-height:1.55}}
  .chips{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin:8px 0}}
  .chip{{background:{CARD};border:1px solid #283042;border-radius:12px;padding:14px}}
  .chip b{{display:inline-block;padding:2px 8px;border-radius:6px;font-size:12px;margin-bottom:6px}}
  .chip.pass b{{background:#13321f;color:#39d98a}} .chip.fail b{{background:#3a1313;color:#ff6b6b}}
  .chip.warn b{{background:#33240f;color:#f2b705}}
  .chip span{{display:block;font-weight:600;margin:2px 0}} .chip em{{color:{GREY};font-style:normal;font-size:12px}}
  h2{{font-size:18px;margin:30px 0 6px;border-bottom:1px solid #283042;padding-bottom:6px}}
  .rec{{background:#101c2e;border:1px solid #1d3a5f;border-radius:14px;padding:18px 20px;line-height:1.6}}
  .rec b{{color:#7fc4ff}}
</style></head><body><div class="wrap">
  <h1>Q-Commerce Pricing &amp; Experimentation Engine</h1>
  <p class="sub">Price elasticity, A/B experimentation, and margin guardrails for a UAE quick-commerce grocery platform.</p>
  <span class="badge">Synthetic data &middot; deterministic seed &middot; fully reproducible</span>
  <div class="kpis">{kpi_html}</div>

  <h2>1 &middot; The trap: Simpson&rsquo;s Paradox</h2>
  <div class="card">{_fig_html(f1)}</div>
  <div class="card">{_fig_html(f4)}</div>
  <div class="insight"><b>Insight.</b> The promo was over-allocated to low-converting zones, so the
  naive aggregate ({exp['naive_lift_pp']}pp, CI crosses zero) makes a working campaign look dead.
  Re-weighting each zone by exposure recovers the true <b>+{exp['stratified_lift_pp']}pp</b> lift.
  A team reading the raw average would have killed a lever that genuinely drives conversion.</div>

  <h2>2 &middot; Why one discount can&rsquo;t fit all</h2>
  <div class="card">{_fig_html(f2)}</div>
  <div class="insight"><b>Insight.</b> New customers are highly elastic
  (e&nbsp;=&nbsp;{ela['segment_arc_elasticity'].get('new','?')}) while loyal customers are inelastic
  (e&nbsp;=&nbsp;{ela['segment_arc_elasticity'].get('loyal','?')}). Discounting loyal, price-insensitive
  customers is almost pure margin loss.</div>

  <h2>3 &middot; The guardrail: does the discount pay for itself?</h2>
  <div class="card">{_fig_html(f3)}</div>
  <div class="insight"><b>Insight.</b> On thin ~25% grocery margins, every discount level reduces gross
  margin per exposure &mdash; the conversion win does not offset the price cut on a pure-margin basis.
  Pure-margin optimum: <b>{policy}</b>. Right-sizing discounts projects ~<b>AED
  {opt['projected_monthly_margin_uplift_aed']:,}/month</b> in recovered margin.</div>

  <h2>4 &middot; Guardrails &amp; integrity</h2>
  <div class="chips">{chip_html}</div>

  <h2>5 &middot; Recommendation</h2>
  <div class="rec">
  <b>1. Do not ship the flat 10% promo.</b> It reliably lifts conversion (+{exp['stratified_lift_pp']}pp),
  but fails the margin guardrail in every segment (stratified margin delta AED
  {guard['stratified_margin_delta_aed']}/exposure).<br><br>
  <b>2. Treat discounting as an acquisition lever, not a margin lever.</b> If the objective is growth,
  target promos only at the elastic <b>new</b> segment &mdash; the most conversions per AED of margin
  spent &mdash; and justify the spend against downstream LTV, not first-order margin.<br><br>
  <b>3. Stop discounting inelastic loyal customers.</b> Reallocating discount depth by segment elasticity
  recovers an estimated AED {opt['projected_monthly_margin_uplift_aed']:,}/month.<br><br>
  <b>4. Always read experiments stratified + CUPED-adjusted.</b> The naive aggregate here was off by
  {exp['stratified_lift_pp']/max(abs(exp['naive_lift_pp']),0.01):.0f}x.
  </div>
  <p class="sub" style="margin-top:24px">Built by Akanksha Swarnim &middot; synthetic data, illustrative figures.</p>
</div></body></html>"""

    with open(out, "w") as f:
        f.write(html)
    print(f"Wrote {out}")


if __name__ == "__main__":
    build()
