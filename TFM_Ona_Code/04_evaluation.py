"""
04_Evaluation.py — SHAP Analysis, Detailed Model Comparison, Error Analysis
TFM: Demand Forecasting for Inventory Management Using Machine Learning
Author: Ona Mas i Serra

Reads: output/processed/features.parquet, output/models/predictions_last_fold.parquet,
       output/models/tuned_params.json
Outputs: output/evaluation/ (SHAP plots, error analysis, comparison tables)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import shap
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
from style_config import (apply_style, PRIMARY, SECONDARY, NEUTRAL, ALERT,
                          get_model_color, get_model_colors)

# ── Configuration ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
FEATURES_PATH = BASE_DIR / "output" / "processed" / "features.parquet"
PREDS_PATH = BASE_DIR / "output" / "models" / "predictions_last_fold.parquet"
PARAMS_PATH = BASE_DIR / "output" / "models" / "tuned_params.json"
OUTPUT_DIR = BASE_DIR / "output" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()

TARGET = "udsVenta"
EXCLUDE = ["producto", "idSecuencia", "fecha", "day_name", TARGET,
           "prod_mean_sales", "prod_std_sales", "prod_median_sales"]

def save_fig(name):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}.png", bbox_inches="tight", dpi=120)
    plt.close()
    print(f"   → Saved {name}.png")

def calc_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    errors = y_true - y_pred
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true > 0
    mape = np.mean(np.abs(errors[mask] / y_true[mask])) * 100 if mask.sum() > 0 else np.nan
    sigma_e = np.std(errors)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sigma_e": sigma_e}


# ══════════════════════════════════════════════════════════════════
# 1. LOAD DATA & RETRAIN BEST MODEL
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. LOADING DATA & RETRAINING LightGBM")
print("=" * 60)

df = pd.read_parquet(FEATURES_PATH)
preds = pd.read_parquet(PREDS_PATH)
with open(PARAMS_PATH) as f:
    params = json.load(f)

feature_cols = [c for c in df.columns if c not in EXCLUDE]

# Use last 30 days as test, rest as train (matching fold 3)
all_dates = sorted(df["fecha"].unique())
cutoff = all_dates[-30]
train = df[df["fecha"] < cutoff]
test = df[df["fecha"] >= cutoff]

X_train, y_train = train[feature_cols], train[TARGET]
X_test, y_test = test[feature_cols], test[TARGET]

# Retrain LightGBM (best model) for SHAP
lgb_params = params["lgb"]
model = lgb.LGBMRegressor(**lgb_params)
model.fit(X_train, y_train)
print(f"Train: {len(train):,} | Test: {len(test):,}")

# ══════════════════════════════════════════════════════════════════
# 2. SHAP ANALYSIS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. SHAP FEATURE IMPORTANCE")
print("=" * 60)

# Use a sample for SHAP (full dataset is too large)
np.random.seed(42)
shap_sample_idx = np.random.choice(len(X_test), size=min(3000, len(X_test)), replace=False)
X_shap = X_test.iloc[shap_sample_idx]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)

# 2.1 SHAP summary (beeswarm)
print("Generating SHAP summary plot...")
fig, ax = plt.subplots(figsize=(12, 10))
shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
save_fig("14_shap_summary")

# 2.2 SHAP bar plot (mean absolute SHAP) — override colour to UOC dark blue
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, max_display=20,
                  color=PRIMARY)
save_fig("15_shap_bar")

# 2.3 SHAP values table
shap_importance = pd.DataFrame({
    "feature": feature_cols,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

print("\n── Top 15 Features by Mean |SHAP| ──")
print(shap_importance.head(15).to_string(index=False))
shap_importance.to_csv(OUTPUT_DIR / "shap_importance.csv", index=False)

# 2.4 SHAP dependence plots for top 4 features
top4 = shap_importance["feature"].head(4).tolist()
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for ax, feat in zip(axes.flat, top4):
    shap.dependence_plot(feat, shap_values, X_shap, ax=ax, show=False)
    ax.set_title(f"SHAP Dependence: {feat}")
save_fig("16_shap_dependence_top4")


# ══════════════════════════════════════════════════════════════════
# 3. ERROR ANALYSIS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. ERROR ANALYSIS")
print("=" * 60)

pf = preds.copy()
models = ["Naive_lag7", "RandomForest", "XGBoost", "LightGBM"]

# 3.1 Error distribution per model
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, m in zip(axes.flat, models):
    errors = pf[TARGET] - pf[m]
    ax.hist(errors, bins=80, edgecolor="white", alpha=0.7, density=True, color=get_model_color(m))
    ax.axvline(x=0, color=ALERT, linewidth=1, linestyle="--")
    ax.set_title(f"{m} — Error Distribution")
    ax.set_xlabel("Error (Actual - Predicted)")
    ax.set_ylabel("Density")
    ax.set_xlim(-15, 15)
save_fig("17_error_distributions")

# 3.2 Per-product RMSE distribution
print("\nPer-product RMSE distribution:")
prod_rmse = {}
for m in models:
    per_prod = pf.groupby("producto").apply(
        lambda g: np.sqrt(mean_squared_error(g[TARGET], g[m])), include_groups=False
    )
    prod_rmse[m] = per_prod

prod_rmse_df = pd.DataFrame(prod_rmse)
print(prod_rmse_df.describe().round(3))

fig, ax = plt.subplots(figsize=(12, 6))
prod_rmse_df[["Naive_lag7", "LightGBM"]].plot(
    kind="hist", bins=50, alpha=0.6, ax=ax, edgecolor="white",
    color=[PRIMARY, SECONDARY]
)
ax.set_title("Per-Product RMSE Distribution: Naive vs LightGBM")
ax.set_xlabel("RMSE")
ax.set_ylabel("Number of Products")
save_fig("18_per_product_rmse_dist")

# 3.3 Products where ML improves most / least over naive
improvement = prod_rmse_df["Naive_lag7"] - prod_rmse_df["LightGBM"]
improvement = improvement.sort_values(ascending=False)

print(f"\n── Top 10 products with most improvement (Naive→LGBM) ──")
print(improvement.head(10).round(3))
print(f"\n── 10 products with least improvement ──")
print(improvement.tail(10).round(3))

# 3.4 Error by day of week
pf_with_dow = pf.merge(
    df[["producto", "fecha", "day_of_week"]].drop_duplicates(),
    on=["producto", "fecha"], how="left"
)

dow_errors = {}
for m in ["LightGBM"]:
    pf_with_dow["_err"] = np.abs(pf_with_dow[TARGET] - pf_with_dow[m])
    dow_errors[m] = pf_with_dow.groupby("day_of_week")["_err"].mean()

dow_df = pd.DataFrame(dow_errors)
dow_df.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

fig, ax = plt.subplots(figsize=(10, 5))
dow_df.plot(kind="bar", ax=ax, color=PRIMARY, edgecolor="white", legend=False)
ax.set_title("LightGBM Mean Absolute Error by Day of Week")
ax.set_ylabel("MAE")
ax.set_xlabel("")
plt.xticks(rotation=45)
save_fig("19_mae_by_day_of_week")

# 3.5 Promo vs Non-Promo detailed comparison table
print("\n── Detailed Promo vs Non-Promo Metrics ──")
comparison_rows = []
for label, mask in [("Overall", pf[TARGET] >= 0), ("Non-Promo", pf["en_promo"] == 0), ("Promo", pf["en_promo"] == 1)]:
    subset = pf[mask]
    for m in models:
        metrics = calc_metrics(subset[TARGET], subset[m])
        metrics["segment"] = label
        metrics["model"] = m
        comparison_rows.append(metrics)

comp_df = pd.DataFrame(comparison_rows)
comp_pivot = comp_df.pivot_table(index=["segment", "model"], values=["MAE", "RMSE", "MAPE", "sigma_e"])
print(comp_pivot.round(3).to_string())
comp_df.to_csv(OUTPUT_DIR / "promo_comparison.csv", index=False)

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
