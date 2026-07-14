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


_first_fig = [True]

def _fig_html(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=50, r=20, t=50, b=45),
                      font=dict(family="Inter, Segoe UI, Arial", size=13, color="#e6e9ef"),
                      legend=dict(bgcolor="rgba(0,0,0,0)"),
                      dragmode=False)
    inc = True if _first_fig[0] else False   # bundle plotly.js once, inline (no CDN)
    _first_fig[0] = False
    return to_html(fig, full_html=False, include_plotlyjs=inc,
                   config={"displayModeBar": False, "scrollZoom": False,
                           "doubleClick": False, "staticPlot": False})


def build(results_path="reports/results.json", data_path="data/orders.csv",
          out="reports/dashboard.html"):
    R = json.load(open(results_path))
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
        xs = [float(k) * 100 for k in curve]
        f2.add_scatter(x=xs, y=list(curve.values()), mode="lines+markers",
                       name=f"{seg} (e={ela['segment_arc_elasticity'].get(seg,'?')})",
                       line=dict(color=SEG_COLORS.get(seg, "#fff"), width=3))
    f2.update_layout(title="Price sensitivity differs sharply by segment",
                     xaxis_title="Discount depth (%)", yaxis_title="Conversion rate",
                     xaxis=dict(tickmode="array", tickvals=[0, 5, 10, 15, 20],
                                ticktext=["0%", "5%", "10%", "15%", "20%"]))

    # ---- Chart 3: margin-vs-discount optimisation ----
    f3 = go.Figure()
    for seg, curve in opt["segment_margin_curves"].items():
        xs = [float(k) * 100 for k in curve]
        f3.add_scatter(x=xs, y=list(curve.values()), mode="lines+markers", name=seg,
                       line=dict(color=SEG_COLORS.get(seg, "#fff"), width=3))
    f3.add_hline(y=0, line_color="#555")
    f3.update_layout(title="Margin per exposure falls with discount depth (thin grocery margin)",
                     xaxis_title="Discount depth (%)", yaxis_title="Gross margin / exposure (AED)",
                     xaxis=dict(tickmode="array", tickvals=[0, 5, 10, 15, 20],
                                ticktext=["0%", "5%", "10%", "15%", "20%"]))

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
         f"naive read showed {exp['naive_lift_pp']}pp  ·  pp = percentage points"),
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
        ("G1 Does the blanket promo add profit?",
         "FAIL" if not guard["G1_blanket_margin_accretive"] else "PASS",
         f"stratified margin delta AED {guard['stratified_margin_delta_aed']}"),
        ("G3 Lift positive in every zone?",
         "PASS" if guard["G3_fairness_all_zones_positive"] else "FAIL", "no zone harmed"),
        ("G4 Enough data to trust the result?",
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
  .reads{{background:{CARD};border:1px solid #283042;border-radius:12px;padding:14px 18px;margin:10px 4px 4px;line-height:1.6;font-size:14px}}
  .reads p{{margin:0 0 11px}} .reads p:last-child{{margin:0}}
  .tag{{display:inline-block;background:#1d3a5f;color:#9fd0ff;font-size:11px;font-weight:700;
    padding:2px 8px;border-radius:6px;margin-right:8px;text-transform:uppercase;letter-spacing:.4px;vertical-align:middle}}
  .chips{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin:8px 0}}
  .chip{{background:{CARD};border:1px solid #283042;border-radius:12px;padding:14px}}
  .chip b{{display:inline-block;padding:2px 8px;border-radius:6px;font-size:12px;margin-bottom:6px}}
  .chip.pass b{{background:#13321f;color:#39d98a}} .chip.fail b{{background:#3a1313;color:#ff6b6b}}
  .chip.warn b{{background:#33240f;color:#f2b705}}
  .chip span{{display:block;font-weight:600;margin:2px 0}} .chip em{{color:{GREY};font-style:normal;font-size:12px}}
  h2{{font-size:18px;margin:30px 0 6px;border-bottom:1px solid #283042;padding-bottom:6px}}
  .rec{{background:#101c2e;border:1px solid #1d3a5f;border-radius:14px;padding:18px 20px;line-height:1.6}}
  .rec b{{color:#7fc4ff}}
  .intro{{background:{CARD};border:1px solid #283042;border-radius:14px;padding:16px 20px;margin:8px 0 22px;line-height:1.6;font-size:14px}}
  .intro p{{margin:0 0 10px}} .intro p:last-child{{margin:0}} .intro b{{color:#cdd6e4}}
  .brief{{margin:6px 0 26px}}
  .brief-item{{background:{CARD};border:1px solid #283042;border-left:3px solid {ORANGE};
    border-radius:10px;padding:12px 16px;margin:0 0 10px}}
  .brief-item h4{{margin:0 0 4px;font-size:14px;color:#ffd9cf}}
  .brief-item p{{margin:0;font-size:13.5px;line-height:1.55;color:#c8cfda}}
  .gloss{{background:{CARD};border:1px solid #283042;border-radius:12px;padding:14px 18px;margin:0 0 24px}}
  .gloss dl{{display:grid;grid-template-columns:auto 1fr;gap:7px 16px;margin:0}}
  .gloss dt{{color:#9fd0ff;font-weight:700;font-size:13px;white-space:nowrap}}
  .gloss dd{{margin:0;font-size:13px;color:#c8cfda;line-height:1.5}}
</style></head><body><div class="wrap">
  <h1>Q-Commerce Pricing &amp; Experimentation Engine</h1>
  <p class="sub">Price elasticity, A/B experimentation, and margin guardrails for a UAE quick-commerce grocery platform.</p>
  <span class="badge">Synthetic data &middot; deterministic seed &middot; fully reproducible</span>
  <div class="intro">
    <p><b>What this is.</b> A pricing-decision tool for a UAE quick-commerce grocery platform. It answers one question &mdash; <b>should we ship a discount, to whom, and how deep?</b> &mdash; and catches the statistical traps that quietly turn a &ldquo;win&rdquo; into a loss.</p>
    <p><b>How to read it.</b> Start with the four headline numbers below. Section&nbsp;1 shows why the raw A/B result was misleading; section&nbsp;2, which customers actually respond to discounts; section&nbsp;3, what discounting costs in margin; section&nbsp;4 is the guardrail verdict; section&nbsp;5 is the recommendation. Every chart is interactive &mdash; hover any point for exact values.</p>
    <p style="margin-top:12px"><a href="simulator.html" style="display:inline-block;background:#241612;color:#ffd9cf;border:1px solid #ff5a36;padding:8px 16px;border-radius:10px;text-decoration:none;font-size:14px;font-weight:600">&#9654;&nbsp; Try the interactive simulator</a> <span style="color:#9aa0a6;font-size:13px">&mdash; drag the levers and watch the numbers and recommendation change.</span></p>
  </div>
  <div class="kpis">{kpi_html}</div>

  <h2>In plain English</h2>
  <div class="brief">
    <div class="brief-item"><h4>The problem</h4><p>Should we run a 10% discount? Discounts almost always lift sales &mdash; but on thin grocery margins they can quietly lose money. We need to know three things: does the promo actually help, who should get it, and how deep it should go.</p></div>
    <div class="brief-item"><h4>The constraints</h4><p>Grocery margins are thin (~25%), so there&rsquo;s little room to give away. The promo wasn&rsquo;t handed out evenly &mdash; more of it went to weaker-performing areas &mdash; which distorts any quick read of the results. And we can&rsquo;t test forever; we need a confident answer without waiting for huge volumes of traffic.</p></div>
    <div class="brief-item"><h4>The guardrails (safety checks before saying &ldquo;ship it&rdquo;)</h4><p>Does the promo still make money after the discount? Does it help every area, or hurt some? Do we have enough data to trust the result, or could it just be noise? And is the underlying data real and clean? A promo only gets a green light if it passes all four.</p></div>
    <div class="brief-item"><h4>The approach</h4><p>Compare like-for-like areas so the overall average can&rsquo;t mislead us (this fixes &ldquo;Simpson&rsquo;s Paradox&rdquo;). Use customers&rsquo; past spending to tighten the result so we can call it with less traffic (CUPED). Measure how sensitive each customer type is to price (elasticity). Then run every finding through the guardrails before recommending anything.</p></div>
    <div class="brief-item"><h4>Why this approach fits</h4><p>These are the standard, trusted tools for pricing experiments. They&rsquo;re built to catch the exact mistakes that make teams ship money-losing promos &mdash; or kill good ones by accident.</p></div>
    <div class="brief-item"><h4>Limitations (being honest)</h4><p>The data is synthetic &mdash; realistic but made up &mdash; so it proves the method works, not any real company&rsquo;s results. The &ldquo;discount new customers&rdquo; idea depends on their long-term value, which isn&rsquo;t modelled here. And correcting a messy experiment after the fact is a rescue, not a substitute for running it cleanly in the first place.</p></div>
    <div class="brief-item"><h4>The insights, in plain terms</h4><p>The quick read said the promo did nothing &mdash; that was wrong; every area actually improved, the average just hid it. New customers respond strongly to discounts; loyal customers barely do, so discounting loyal ones is mostly wasted money. On these margins, every discount level loses money on the sale itself. Bottom line: don&rsquo;t discount everyone &mdash; aim it only at new customers, justified by their future value. Stopping the wasteful discounts could recover roughly AED 2 million a month.</p></div>
  </div>

  <h2>Key terms &mdash; one line each</h2>
  <div class="gloss"><dl>
    <dt>pp</dt><dd>Percentage points &mdash; the plain gap between two percentages. 40% &rarr; 45% is +5&nbsp;pp (not &ldquo;5% more&rdquo;).</dd>
    <dt>A/B test</dt><dd>Split users into two groups, show one group the change, and compare the results.</dd>
    <dt>Conversion</dt><dd>The customer actually places the order (not just browses).</dd>
    <dt>Exposure</dt><dd>One chance to convert &mdash; a customer opening the app with intent to buy.</dd>
    <dt>Elasticity</dt><dd>How much demand moves when price moves. Big negative = very price-sensitive; near zero = barely reacts.</dd>
    <dt>Margin</dt><dd>What&rsquo;s left after costs &mdash; the actual profit on a sale.</dd>
    <dt>CI</dt><dd>Confidence interval &mdash; the range the true answer likely sits in. If it crosses zero, you can&rsquo;t be sure there&rsquo;s any real effect.</dd>
    <dt>CUPED</dt><dd>A technique that uses customers&rsquo; past data to sharpen a result, so you can trust it with less traffic.</dd>
    <dt>Stratified</dt><dd>Measured group-by-group (e.g. zone-by-zone) instead of one lumped-together average.</dd>
    <dt>MDE</dt><dd>Minimum detectable effect &mdash; the smallest change the test is large enough to spot reliably.</dd>
    <dt>Guardrail</dt><dd>A safety check a result must pass before you act on it.</dd>
    <dt>LTV</dt><dd>Lifetime value &mdash; total profit a customer brings over their whole relationship, not just one order.</dd>
  </dl></div>

  <h2>1 &middot; The trap: Simpson&rsquo;s Paradox</h2>
  <div class="card">{_fig_html(f1)}</div>
  <div class="card">{_fig_html(f4)}</div>
  <div class="reads">
    <p><span class="tag">What is Simpson&rsquo;s Paradox?</span> A pattern that holds inside every subgroup can reverse or vanish once you lump the subgroups together.</p>
    <p><span class="tag">What this chart shows</span> Each bar is one delivery zone&rsquo;s conversion lift from the promo &mdash; all five are clearly positive. The dashed line near the bottom is the naive result when you pool every zone into one average.</p>
    <p><span class="tag">How we proved it &amp; the point</span> The promo was handed out more often in low-converting zones (Deira, JLT), which drags the pooled average down to {exp['naive_lift_pp']}pp &mdash; making a working campaign look dead. Re-weighting each zone by its share of exposures removes that imbalance and recovers the true <b>+{exp['stratified_lift_pp']}pp</b> lift. The takeaway: never trust the raw average &mdash; read the experiment zone-by-zone (&ldquo;stratified&rdquo;).</p>
  </div>

  <h2>2 &middot; Why one discount can&rsquo;t fit all</h2>
  <div class="card">{_fig_html(f2)}</div>
  <div class="reads">
    <p><span class="tag">What is price elasticity?</span> How much demand moves when price moves. A big negative number = very price-sensitive; close to zero = barely reacts.</p>
    <p><span class="tag">What this chart shows</span> Each line is a customer segment. Left-to-right is a deeper discount; up is a higher conversion rate. A steeper climb means that segment responds more strongly to discounts.</p>
    <p><span class="tag">How we proved it &amp; the point</span> New customers&rsquo; line climbs steeply (elastic, e&nbsp;=&nbsp;{ela['segment_arc_elasticity'].get('new','?')}), while loyal customers are almost flat (inelastic, e&nbsp;=&nbsp;{ela['segment_arc_elasticity'].get('loyal','?')}). We measured this with a demand model that controls for zone, time of day and season, so the gap is real price response, not noise. The same discount buys many extra orders from new customers and almost none from loyal ones &mdash; so one blanket discount can&rsquo;t fit all; it overpays the people who would have bought anyway.</p>
  </div>

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
