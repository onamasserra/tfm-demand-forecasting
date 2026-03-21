"""
05_Economic_Impact.py — Safety Stock & Newsvendor Cost Analysis
TFM: Demand Forecasting for Inventory Management Using Machine Learning
Author: Ona Mas i Serra

Computes:
  1. Safety stock per product using σ_e from each model
  2. Newsvendor cost: TC = Σ[ Co·(F-D)⁺ + Cu·(D-F)⁺ ]
  3. Comparison table: ML models vs Naive baseline
  4. Sensitivity analysis on service level

Reads: output/models/predictions_last_fold.parquet,
       output/evaluation/shap_importance.csv (optional),
       Datos.xlsx (stock sheet for lead time proxy),
       EstimacionCostesStock.docx (cost parameters reference)
Outputs: output/economic/ (tables, plots)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error
from style_config import (apply_style, PRIMARY, SECONDARY, ACCENT, NEUTRAL,
                          PROMO_YES, PROMO_NO, DARK_BLUE, AIR_BLUE, TEAL,
                          AMBER, CORAL, SLATE, LIGHT_BLUE,
                          get_model_color, get_model_colors)

# ── Configuration ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PREDS_PATH = BASE_DIR / "output" / "models" / "predictions_last_fold.parquet"
OUTPUT_DIR = BASE_DIR / "output" / "economic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()

TARGET = "udsVenta"
MODELS = ["Naive_lag7", "RandomForest", "XGBoost", "LightGBM"]

# ── Cost parameters (from thesis State of Art & EstimacionCostesStock) ──
# These are configurable — adjust if more precise data becomes available.
# Co = overage cost (cost per unit of excess inventory per day)
# Cu = underage cost (cost per unit of unmet demand — lost margin + penalty)
# Sebastian used: holding cost = 5% × stock × avg_price (annualized)
# We use a newsvendor framework instead.

# Default cost ratio: Cu/Co determines optimal service level
# Cu/(Cu+Co) = critical ratio → service level
# Typical retail: Cu >> Co (stockout is more expensive than holding)
COST_OVERAGE_PER_UNIT = 0.10    # € per unit per day (holding/obsolescence)
COST_UNDERAGE_PER_UNIT = 0.50   # € per unit per day (lost sale margin)

# Safety stock parameters
# Service level z-scores: 90%→1.28, 95%→1.64, 99%→2.33
SERVICE_LEVELS = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
DEFAULT_SERVICE_LEVEL = 0.95

# Lead time + review period (days) — proxy from Sebastian's thesis
# Sebastian used diasEntrePedidos + diasLeadtime from a separate file
# We use a reasonable default; can be updated with actual data
LEAD_TIME_DAYS = 3
REVIEW_PERIOD_DAYS = 7
TOTAL_PROTECTION_DAYS = LEAD_TIME_DAYS + REVIEW_PERIOD_DAYS

def save_fig(name):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}.png", bbox_inches="tight", dpi=120)
    plt.close()
    print(f"   → Saved {name}.png")


# ══════════════════════════════════════════════════════════════════
# 1. LOAD PREDICTIONS
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. LOADING PREDICTIONS")
print("=" * 60)

pf = pd.read_parquet(PREDS_PATH)
print(f"Predictions: {len(pf):,} rows, {pf['producto'].nunique()} products")
print(f"Date range: {pf['fecha'].min().date()} → {pf['fecha'].max().date()}")
print(f"Days: {pf['fecha'].nunique()}")


# ══════════════════════════════════════════════════════════════════
# 2. PER-PRODUCT FORECAST ERROR (σ_e) FOR EACH MODEL
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. PER-PRODUCT FORECAST ERROR (σ_e)")
print("=" * 60)

sigma_e = {}
for model in MODELS:
    per_prod = pf.groupby("producto").apply(
        lambda g: np.std(g[TARGET].values - g[model].values), include_groups=False
    )
    sigma_e[model] = per_prod

sigma_df = pd.DataFrame(sigma_e)
print(f"\n── σ_e Summary (across {len(sigma_df)} products) ──")
print(sigma_df.describe().round(3))


# ══════════════════════════════════════════════════════════════════
# 3. SAFETY STOCK CALCULATION
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. SAFETY STOCK CALCULATION")
print("=" * 60)

# SS = z × σ_e × √(L + R)
# where z = service level z-score, L = lead time, R = review period
z = SERVICE_LEVELS[DEFAULT_SERVICE_LEVEL]
protection_factor = np.sqrt(TOTAL_PROTECTION_DAYS)

print(f"Service level: {DEFAULT_SERVICE_LEVEL*100:.0f}% (z = {z:.3f})")
print(f"Protection period: {TOTAL_PROTECTION_DAYS} days (lead={LEAD_TIME_DAYS} + review={REVIEW_PERIOD_DAYS})")
print(f"√(L+R) = {protection_factor:.3f}")

ss = {}
for model in MODELS:
    ss[model] = z * sigma_df[model] * protection_factor

ss_df = pd.DataFrame(ss)
print(f"\n── Safety Stock (units) Summary ──")
print(ss_df.describe().round(2))

# Total safety stock across all products
total_ss = ss_df.sum()
print(f"\n── Total Safety Stock (all products) ──")
for model in MODELS:
    print(f"  {model:15s}: {total_ss[model]:,.0f} units")


# ══════════════════════════════════════════════════════════════════
# 4. NEWSVENDOR COST ANALYSIS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. NEWSVENDOR COST ANALYSIS")
print("=" * 60)

# TC = Σ[ Co × max(F-D, 0) + Cu × max(D-F, 0) ]
# Co = overage cost (predicted too high → excess inventory)
# Cu = underage cost (predicted too low → lost sales)

print(f"Co (overage):  €{COST_OVERAGE_PER_UNIT:.2f}/unit/day")
print(f"Cu (underage): €{COST_UNDERAGE_PER_UNIT:.2f}/unit/day")
print(f"Critical ratio Cu/(Cu+Co) = {COST_UNDERAGE_PER_UNIT/(COST_UNDERAGE_PER_UNIT+COST_OVERAGE_PER_UNIT):.2f}")

cost_results = {}
for model in MODELS:
    forecast = pf[model].values
    actual = pf[TARGET].values

    overage = np.maximum(forecast - actual, 0)   # predicted too high
    underage = np.maximum(actual - forecast, 0)   # predicted too low

    cost_over = COST_OVERAGE_PER_UNIT * overage.sum()
    cost_under = COST_UNDERAGE_PER_UNIT * underage.sum()
    total_cost = cost_over + cost_under

    cost_results[model] = {
        "overage_units": overage.sum(),
        "underage_units": underage.sum(),
        "cost_overage": cost_over,
        "cost_underage": cost_under,
        "total_cost": total_cost,
        "avg_daily_cost": total_cost / pf["fecha"].nunique(),
    }

cost_df = pd.DataFrame(cost_results).T
cost_df["annual_cost_est"] = cost_df["avg_daily_cost"] * 365

print(f"\n── Newsvendor Cost (test period: {pf['fecha'].nunique()} days) ──")
print(cost_df[["cost_overage", "cost_underage", "total_cost", "annual_cost_est"]].round(2).to_string())

# Savings vs Naive
naive_cost = cost_df.loc["Naive_lag7", "total_cost"]
print(f"\n── Savings vs Naive Baseline ──")
for model in MODELS:
    if model == "Naive_lag7":
        continue
    saving = naive_cost - cost_df.loc[model, "total_cost"]
    pct = (saving / naive_cost) * 100
    print(f"  {model:15s}: €{saving:,.2f} saved ({pct:.1f}%)")

cost_df.to_csv(OUTPUT_DIR / "newsvendor_costs.csv")
print(f"\n✓ Saved newsvendor_costs.csv")


# ══════════════════════════════════════════════════════════════════
# 5. SENSITIVITY ANALYSIS — SERVICE LEVEL
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. SENSITIVITY ANALYSIS — SERVICE LEVEL")
print("=" * 60)

sensitivity_rows = []
for sl, z_val in SERVICE_LEVELS.items():
    for model in MODELS:
        total_ss_val = (z_val * sigma_df[model] * protection_factor).sum()
        # Holding cost proxy: assume €0.10/unit/day for safety stock
        daily_holding = COST_OVERAGE_PER_UNIT * total_ss_val
        annual_holding = daily_holding * 365
        sensitivity_rows.append({
            "service_level": f"{sl*100:.0f}%",
            "model": model,
            "z_score": z_val,
            "total_safety_stock": total_ss_val,
            "annual_holding_cost": annual_holding,
        })

sens_df = pd.DataFrame(sensitivity_rows)
sens_pivot = sens_df.pivot(index="model", columns="service_level",
                           values="total_safety_stock")
print("\n── Total Safety Stock by Service Level ──")
print(sens_pivot.round(0).to_string())

sens_cost_pivot = sens_df.pivot(index="model", columns="service_level",
                                values="annual_holding_cost")
print("\n── Annual Holding Cost (€) by Service Level ──")
print(sens_cost_pivot.round(0).to_string())

sens_df.to_csv(OUTPUT_DIR / "sensitivity_service_level.csv", index=False)


# ══════════════════════════════════════════════════════════════════
# 6. COMBINED COST: NEWSVENDOR + SAFETY STOCK HOLDING
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. COMBINED COST SUMMARY")
print("=" * 60)

z_default = SERVICE_LEVELS[DEFAULT_SERVICE_LEVEL]
combined_rows = []
for model in MODELS:
    nv_annual = cost_df.loc[model, "annual_cost_est"]
    ss_total = (z_default * sigma_df[model] * protection_factor).sum()
    ss_annual = COST_OVERAGE_PER_UNIT * ss_total * 365
    combined_rows.append({
        "model": model,
        "newsvendor_annual": nv_annual,
        "safety_stock_units": ss_total,
        "safety_stock_holding_annual": ss_annual,
        "total_annual_cost": nv_annual + ss_annual,
    })

combined_df = pd.DataFrame(combined_rows).set_index("model")
print(combined_df.round(2).to_string())

# Savings
naive_total = combined_df.loc["Naive_lag7", "total_annual_cost"]
print(f"\n── Total Annual Savings vs Naive ──")
for model in MODELS:
    if model == "Naive_lag7":
        continue
    saving = naive_total - combined_df.loc[model, "total_annual_cost"]
    pct = (saving / naive_total) * 100
    print(f"  {model:15s}: €{saving:,.2f} ({pct:.1f}%)")

combined_df.to_csv(OUTPUT_DIR / "combined_cost_summary.csv")


# ══════════════════════════════════════════════════════════════════
# 7. PLOTS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. GENERATING PLOTS")
print("=" * 60)

# 7.1 Newsvendor cost comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(MODELS))
width = 0.35
bars_over = ax.bar(x - width/2, [cost_df.loc[m, "cost_overage"] for m in MODELS],
                   width, label="Overage Cost (Co)", color=DARK_BLUE)
bars_under = ax.bar(x + width/2, [cost_df.loc[m, "cost_underage"] for m in MODELS],
                    width, label="Underage Cost (Cu)", color=LIGHT_BLUE)
ax.set_xticks(x)
ax.set_xticklabels(MODELS, rotation=15)
ax.set_ylabel("Cost (€)")
ax.set_title("Newsvendor Cost Breakdown by Model (Test Period)")
ax.legend()
save_fig("20_newsvendor_cost_breakdown")

# 7.2 Safety stock comparison across service levels
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.2
x = np.arange(len(MODELS))
sl_colors = [DARK_BLUE, LIGHT_BLUE, TEAL]
for i, (sl, z_val) in enumerate(SERVICE_LEVELS.items()):
    vals = [(z_val * sigma_df[m] * protection_factor).sum() for m in MODELS]
    ax.bar(x + i * bar_width, vals, bar_width, label=f"{sl*100:.0f}%", color=sl_colors[i])
ax.set_xticks(x + bar_width)
ax.set_xticklabels(MODELS, rotation=15)
ax.set_ylabel("Total Safety Stock (units)")
ax.set_title("Safety Stock by Model and Service Level")
ax.legend(title="Service Level")
save_fig("21_safety_stock_by_service_level")

# 7.3 Combined annual cost
fig, ax = plt.subplots(figsize=(10, 6))
nv_vals = [combined_df.loc[m, "newsvendor_annual"] for m in MODELS]
ss_vals = [combined_df.loc[m, "safety_stock_holding_annual"] for m in MODELS]
ax.bar(MODELS, nv_vals, label="Newsvendor Cost", color=DARK_BLUE)
ax.bar(MODELS, ss_vals, bottom=nv_vals, label="Safety Stock Holding", color=LIGHT_BLUE)
ax.set_ylabel("Annual Cost (€)")
ax.set_title(f"Total Annual Cost by Model (Service Level {DEFAULT_SERVICE_LEVEL*100:.0f}%)")
ax.legend()
plt.xticks(rotation=15)
save_fig("22_combined_annual_cost")

# 7.4 Per-product safety stock distribution (LightGBM vs Naive)
fig, ax = plt.subplots(figsize=(12, 6))
ss_compare = pd.DataFrame({
    "Naive_lag7": z_default * sigma_df["Naive_lag7"] * protection_factor,
    "LightGBM": z_default * sigma_df["LightGBM"] * protection_factor,
})
ss_compare.plot(kind="hist", bins=50, alpha=0.6, ax=ax, edgecolor="white",
                color=[PRIMARY, SECONDARY])
ax.set_title("Per-Product Safety Stock Distribution: Naive vs LightGBM (95% SL)")
ax.set_xlabel("Safety Stock (units)")
ax.set_ylabel("Number of Products")
save_fig("23_safety_stock_distribution")

print("\n" + "=" * 60)
print("ECONOMIC IMPACT ANALYSIS COMPLETE")
print("=" * 60)
