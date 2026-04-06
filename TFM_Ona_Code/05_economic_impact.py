"""
05_Economic_Impact.py — Safety Stock & Cost Analysis (Lorena's methodology)
TFM: Demand Forecasting for Inventory Management Using Machine Learning
Author: Ona Mas i Serra

Methodology (from EstimacionCostesStock.docx + AnalisisEscenariosStock.xlsx):
  SS_i = z × RMSE_i × √(T_i + L_i)
  DailyCost_i = 5% × price_i × SS_i
  AnnualCost_i = 365 × DailyCost_i

  Compare Scenario 1 (Naive) vs Scenario 2 (Best ML model)

Reads: output/models/predictions_last_fold.parquet,
       Datos/DatosCicloAprovisionamiento.xlsx,
       Datos/DatosPrecioMedio.xlsx
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
                          DARK_BLUE, AIR_BLUE, TEAL, LIGHT_BLUE,
                          AMBER, CORAL, SLATE,
                          get_model_color, get_model_colors)

# ── Configuration ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PREDS_PATH = BASE_DIR / "output" / "models" / "predictions_last_fold.parquet"
DATA_DIR = BASE_DIR.parent / "Datos"
OUTPUT_DIR = BASE_DIR / "output" / "economic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()

TARGET = "udsVenta"
MODELS = ["Naive_lag7", "RandomForest", "XGBoost", "LightGBM"]

# ── Cost parameters (from Lorena's documents) ──
# Service level 95% → z = NORMSINV(0.95) = 1.6449
SERVICE_LEVEL = 0.95
Z_SCORE = 1.6449

# Daily holding cost rate: 5% of (price × units in stock)
# From EstimacionCostesStock.docx and AnalisisEscenariosStock Sheet 1
HOLDING_COST_RATE = 0.05

def save_fig(name):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}.png", bbox_inches="tight", dpi=120)
    plt.close()
    print(f"   → Saved {name}.png")


# ══════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

pf = pd.read_parquet(PREDS_PATH)
print(f"Predictions: {len(pf):,} rows, {pf['producto'].nunique()} products")
print(f"Date range: {pf['fecha'].min().date()} → {pf['fecha'].max().date()}")
print(f"Days: {pf['fecha'].nunique()}")

# Load per-product cost parameters
ciclo = pd.read_excel(DATA_DIR / "DatosCicloAprovisionamiento.xlsx")
precio = pd.read_excel(DATA_DIR / "DatosPrecioMedio.xlsx")

print(f"\nCiclo aprovisionamiento: {len(ciclo)} products")
print(f"  diasEntrePedidos: mean={ciclo['diasEntrePedidos'].mean():.1f}, range=[{ciclo['diasEntrePedidos'].min()}, {ciclo['diasEntrePedidos'].max()}]")
print(f"  diasLeadtime: mean={ciclo['diasLeadtime'].mean():.1f}, range=[{ciclo['diasLeadtime'].min()}, {ciclo['diasLeadtime'].max()}]")
print(f"\nPrecio medio: {len(precio)} products")
print(f"  eurPrecioMedio: mean=€{precio['eurPrecioMedio'].mean():.2f}, range=[€{precio['eurPrecioMedio'].min():.2f}, €{precio['eurPrecioMedio'].max():.2f}]")

# Merge cost parameters
# Protection period = diasEntrePedidos + diasLeadtime (from Yamazaki / Sebastian)
cost_params = ciclo.merge(precio, on="producto", how="inner")
cost_params["protection_days"] = cost_params["diasEntrePedidos"] + cost_params["diasLeadtime"]
cost_params["sqrt_protection"] = np.sqrt(cost_params["protection_days"])

print(f"\nProtection period (T+L): mean={cost_params['protection_days'].mean():.1f} days, "
      f"range=[{cost_params['protection_days'].min()}, {cost_params['protection_days'].max()}]")

# Filter to products present in predictions
pred_products = pf["producto"].unique()
cost_params = cost_params[cost_params["producto"].isin(pred_products)]
print(f"Products with cost data & predictions: {len(cost_params)}")


# ══════════════════════════════════════════════════════════════════
# 2. PER-PRODUCT RMSE FOR EACH MODEL
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. PER-PRODUCT RMSE")
print("=" * 60)

rmse_per_product = {}
for model in MODELS:
    per_prod = pf.groupby("producto").apply(
        lambda g: np.sqrt(mean_squared_error(g[TARGET], g[model])), include_groups=False
    )
    rmse_per_product[model] = per_prod

rmse_df = pd.DataFrame(rmse_per_product)
print(f"\n── Per-product RMSE Summary ({len(rmse_df)} products) ──")
print(rmse_df.describe().round(3))


# ══════════════════════════════════════════════════════════════════
# 3. SAFETY STOCK CALCULATION (Lorena's formula)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. SAFETY STOCK CALCULATION")
print("=" * 60)

# SS_i = z × RMSE_i × √(T_i + L_i)
print(f"Service level: {SERVICE_LEVEL*100:.0f}% (z = {Z_SCORE:.4f})")
print(f"Formula: SS = z × RMSE × √(diasEntrePedidos + diasLeadtime)")

ss_results = {}
for model in MODELS:
    # Merge RMSE with cost params
    model_df = cost_params[["producto", "sqrt_protection"]].copy()
    model_df = model_df.merge(rmse_df[[model]].rename(columns={model: "rmse"}),
                               left_on="producto", right_index=True, how="inner")
    model_df["safety_stock"] = Z_SCORE * model_df["rmse"] * model_df["sqrt_protection"]
    ss_results[model] = model_df.set_index("producto")["safety_stock"]

ss_df = pd.DataFrame(ss_results)
print(f"\n── Safety Stock (units) Summary ──")
print(ss_df.describe().round(2))

total_ss = ss_df.sum()
print(f"\n── Total Safety Stock (all products) ──")
for model in MODELS:
    print(f"  {model:15s}: {total_ss[model]:,.0f} units")


# ══════════════════════════════════════════════════════════════════
# 4. DAILY & ANNUAL COST (Lorena's formula)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. COST CALCULATION")
print("=" * 60)

# DailyCost_i = 5% × price_i × SS_i
# AnnualCost_i = 365 × DailyCost_i
print(f"Daily cost formula: {HOLDING_COST_RATE*100:.0f}% × price × SS")

price_map = cost_params.set_index("producto")["eurPrecioMedio"]

cost_daily = {}
cost_annual = {}
for model in MODELS:
    daily = HOLDING_COST_RATE * price_map * ss_df[model]
    cost_daily[model] = daily
    cost_annual[model] = daily * 365

cost_daily_df = pd.DataFrame(cost_daily)
cost_annual_df = pd.DataFrame(cost_annual)

print(f"\n── Annual Safety Stock Cost per Product (€) ──")
print(cost_annual_df.describe().round(2))

total_annual = cost_annual_df.sum()
print(f"\n── Total Annual Safety Stock Cost ──")
for model in MODELS:
    print(f"  {model:15s}: €{total_annual[model]:,.2f}")

# Savings vs Naive
naive_total = total_annual["Naive_lag7"]
print(f"\n── Annual Savings vs Naive Baseline ──")
for model in MODELS:
    if model == "Naive_lag7":
        continue
    saving = naive_total - total_annual[model]
    pct = (saving / naive_total) * 100
    print(f"  {model:15s}: €{saving:,.2f} saved ({pct:.1f}%)")


# ══════════════════════════════════════════════════════════════════
# 5. SENSITIVITY ANALYSIS — SERVICE LEVEL
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. SENSITIVITY ANALYSIS — SERVICE LEVEL")
print("=" * 60)

SERVICE_LEVELS = {0.90: 1.2816, 0.95: 1.6449, 0.99: 2.3263}

sensitivity_rows = []
for sl, z_val in SERVICE_LEVELS.items():
    for model in MODELS:
        ss_total = (z_val * rmse_df[model] * cost_params.set_index("producto")["sqrt_protection"]).dropna().sum()
        annual_cost = HOLDING_COST_RATE * (price_map * z_val * rmse_df[model] *
                      cost_params.set_index("producto")["sqrt_protection"]).dropna().sum() * 365
        sensitivity_rows.append({
            "service_level": f"{sl*100:.0f}%",
            "model": model,
            "z_score": z_val,
            "total_safety_stock": ss_total,
            "annual_cost": annual_cost,
        })

sens_df = pd.DataFrame(sensitivity_rows)
sens_pivot = sens_df.pivot(index="model", columns="service_level", values="total_safety_stock")
print("\n── Total Safety Stock by Service Level ──")
print(sens_pivot.round(0).to_string())

sens_cost_pivot = sens_df.pivot(index="model", columns="service_level", values="annual_cost")
print("\n── Annual Cost (€) by Service Level ──")
print(sens_cost_pivot.round(0).to_string())

sens_df.to_csv(OUTPUT_DIR / "sensitivity_service_level.csv", index=False)


# ══════════════════════════════════════════════════════════════════
# 6. SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. SUMMARY TABLE")
print("=" * 60)

summary_rows = []
for model in MODELS:
    summary_rows.append({
        "model": model,
        "avg_rmse": rmse_df[model].mean(),
        "total_safety_stock_units": total_ss[model],
        "annual_cost_eur": total_annual[model],
        "saving_vs_naive_eur": naive_total - total_annual[model],
        "saving_vs_naive_pct": (naive_total - total_annual[model]) / naive_total * 100,
    })

summary = pd.DataFrame(summary_rows).set_index("model")
print(summary.round(2).to_string())
summary.to_csv(OUTPUT_DIR / "economic_summary.csv")
print(f"\n✓ Saved economic_summary.csv")


# ══════════════════════════════════════════════════════════════════
# 7. PLOTS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. GENERATING PLOTS")
print("=" * 60)

# 7.1 Annual cost comparison (Naive vs ML models)
fig, ax = plt.subplots(figsize=(10, 6))
colors = [SLATE if m == "Naive_lag7" else DARK_BLUE for m in MODELS]
bars = ax.bar(MODELS, [total_annual[m] for m in MODELS], color=colors, edgecolor="white")
ax.set_ylabel("Annual Safety Stock Cost (€)")
ax.set_title(f"Annual Safety Stock Cost by Model (Service Level {SERVICE_LEVEL*100:.0f}%)")
# Add value labels
for bar, model in zip(bars, MODELS):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f"€{total_annual[model]:,.0f}", ha="center", va="bottom", fontsize=9)
plt.xticks(rotation=15)
save_fig("20_annual_cost_comparison")

# 7.2 Safety stock comparison across service levels
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.2
x = np.arange(len(MODELS))
sl_colors = [DARK_BLUE, LIGHT_BLUE, TEAL]
for i, (sl, z_val) in enumerate(SERVICE_LEVELS.items()):
    vals = []
    for m in MODELS:
        v = (z_val * rmse_df[m] * cost_params.set_index("producto")["sqrt_protection"]).dropna().sum()
        vals.append(v)
    ax.bar(x + i * bar_width, vals, bar_width, label=f"{sl*100:.0f}%", color=sl_colors[i])
ax.set_xticks(x + bar_width)
ax.set_xticklabels(MODELS, rotation=15)
ax.set_ylabel("Total Safety Stock (units)")
ax.set_title("Safety Stock by Model and Service Level")
ax.legend(title="Service Level")
save_fig("21_safety_stock_by_service_level")

# 7.3 Per-product safety stock distribution (Naive vs XGBoost)
fig, ax = plt.subplots(figsize=(12, 6))
ss_compare = pd.DataFrame({
    "Naive_lag7": ss_df["Naive_lag7"],
    "XGBoost": ss_df["XGBoost"],
})
ss_compare.plot(kind="hist", bins=50, alpha=0.6, ax=ax, edgecolor="white",
                color=[SLATE, DARK_BLUE])
ax.set_title("Per-Product Safety Stock Distribution: Naive vs XGBoost (95% SL)")
ax.set_xlabel("Safety Stock (units)")
ax.set_ylabel("Number of Products")
save_fig("22_safety_stock_distribution")

# 7.4 Per-product annual cost distribution
fig, ax = plt.subplots(figsize=(12, 6))
cost_compare = pd.DataFrame({
    "Naive_lag7": cost_annual_df["Naive_lag7"],
    "XGBoost": cost_annual_df["XGBoost"],
})
cost_compare.plot(kind="hist", bins=50, alpha=0.6, ax=ax, edgecolor="white",
                  color=[SLATE, DARK_BLUE])
ax.set_title("Per-Product Annual Cost Distribution: Naive vs XGBoost")
ax.set_xlabel("Annual Safety Stock Cost (€)")
ax.set_ylabel("Number of Products")
save_fig("23_annual_cost_distribution")

# 7.5 Top 20 products by cost savings
savings_per_product = cost_annual_df["Naive_lag7"] - cost_annual_df["XGBoost"]
top20_savings = savings_per_product.sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(top20_savings.index.astype(str), top20_savings.values, color=DARK_BLUE, edgecolor="white")
ax.set_title("Top 20 Products by Annual Cost Savings (Naive → XGBoost)")
ax.set_xlabel("Annual Savings (€)")
ax.set_ylabel("Product ID")
ax.invert_yaxis()
save_fig("24_top20_savings_by_product")

print("\n" + "=" * 60)
print("ECONOMIC IMPACT ANALYSIS COMPLETE")
print("=" * 60)
