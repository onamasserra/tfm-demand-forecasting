"""Build 04_evaluation.ipynb from 04_evaluation.py"""
import json
from uuid import uuid4

def uid():
    return uuid4().hex[:8]

cells = []

# ── Title ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": [
        "# 04 — Evaluation: SHAP Analysis, Error Analysis & Model Comparison\n",
        "\n",
        "**TFM: Demand Forecasting for Inventory Management Using Machine Learning**\n",
        "\n",
        "Author: Ona Mas i Serra\n",
        "\n",
        "This notebook performs SHAP feature importance analysis on the best model (LightGBM),\n",
        "detailed error analysis across models, per-product RMSE distributions,\n",
        "and promo vs non-promo performance comparison.\n"
    ]
})

# ── Setup ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": ["## Setup & Configuration\n"]
})

cells.append({
    "cell_type": "code", "id": uid(), "metadata": {}, "execution_count": None,
    "outputs": [],
    "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import json\n",
        "import shap\n",
        "from pathlib import Path\n",
        "from sklearn.metrics import mean_absolute_error, mean_squared_error\n",
        "import lightgbm as lgb\n",
        "import sys\n",
        "sys.path.insert(0, '/Users/omasiser/Library/CloudStorage/OneDrive-amazon.com/UOC/TFM forecasting/knowledge base/TFM_Ona_Code')\n",
        "from style_config import (apply_style, PRIMARY, SECONDARY, NEUTRAL, ALERT,\n",
        "                          get_model_color, get_model_colors)\n",
        "\n",
        "%matplotlib inline\n",
        "\n",
        "BASE_DIR = Path('/Users/omasiser/Library/CloudStorage/OneDrive-amazon.com/UOC/TFM forecasting/knowledge base/TFM_Ona_Code')\n",
        "FEATURES_PATH = BASE_DIR / 'output' / 'processed' / 'features.parquet'\n",
        "PREDS_PATH = BASE_DIR / 'output' / 'models' / 'predictions_last_fold.parquet'\n",
        "PARAMS_PATH = BASE_DIR / 'output' / 'models' / 'tuned_params.json'\n",
        "OUTPUT_DIR = BASE_DIR / 'output' / 'evaluation'\n",
        "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "apply_style()\n",
        "\n",
        "TARGET = 'udsVenta'\n",
        "EXCLUDE = ['producto', 'idSecuencia', 'fecha', 'day_name', TARGET,\n",
        "           'prod_mean_sales', 'prod_std_sales', 'prod_median_sales']\n",
        "\n",
        "def save_fig(name):\n",
        "    plt.tight_layout()\n",
        "    plt.savefig(OUTPUT_DIR / f'{name}.png', bbox_inches='tight', dpi=120)\n",
        "    plt.show()\n",
        "    print(f'   \\u2192 Saved {name}.png')\n",
        "\n",
        "def calc_metrics(y_true, y_pred):\n",
        "    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)\n",
        "    errors = y_true - y_pred\n",
        "    mae = mean_absolute_error(y_true, y_pred)\n",
        "    rmse = np.sqrt(mean_squared_error(y_true, y_pred))\n",
        "    mask = y_true > 0\n",
        "    mape = np.mean(np.abs(errors[mask] / y_true[mask])) * 100 if mask.sum() > 0 else np.nan\n",
        "    sigma_e = np.std(errors)\n",
        "    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'sigma_e': sigma_e}\n"
    ]
})

# ── 1. Load Data & Retrain ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": [
        "## 1. Load Data & Retrain LightGBM\n",
        "\n",
        "Load the feature-engineered data and predictions from the modelling step,\n",
        "then retrain LightGBM on the train split for SHAP analysis.\n"
    ]
})

cells.append({
    "cell_type": "code", "id": uid(), "metadata": {}, "execution_count": None,
    "outputs": [],
    "source": [
        "print('=' * 60)\n",
        "print('1. LOADING DATA & RETRAINING LightGBM')\n",
        "print('=' * 60)\n",
        "\n",
        "df = pd.read_parquet(FEATURES_PATH)\n",
        "preds = pd.read_parquet(PREDS_PATH)\n",
        "with open(PARAMS_PATH) as f:\n",
        "    params = json.load(f)\n",
        "\n",
        "feature_cols = [c for c in df.columns if c not in EXCLUDE]\n",
        "\n",
        "all_dates = sorted(df['fecha'].unique())\n",
        "cutoff = all_dates[-30]\n",
        "train = df[df['fecha'] < cutoff]\n",
        "test = df[df['fecha'] >= cutoff]\n",
        "\n",
        "X_train, y_train = train[feature_cols], train[TARGET]\n",
        "X_test, y_test = test[feature_cols], test[TARGET]\n",
        "\n",
        "lgb_params = params['lgb']\n",
        "model = lgb.LGBMRegressor(**lgb_params)\n",
        "model.fit(X_train, y_train)\n",
        "print(f'Train: {len(train):,} | Test: {len(test):,}')\n"
    ]
})

# ── 2. SHAP Analysis ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": [
        "## 2. SHAP Feature Importance\n",
        "\n",
        "Compute SHAP values using TreeExplainer on a sample of the test set.\n",
        "Generate beeswarm summary, bar plot, importance table, and dependence plots for top 4 features.\n"
    ]
})

cells.append({
    "cell_type": "code", "id": uid(), "metadata": {}, "execution_count": None,
    "outputs": [],
    "source": [
        "print('\\n' + '=' * 60)\n",
        "print('2. SHAP FEATURE IMPORTANCE')\n",
        "print('=' * 60)\n",
        "\n",
        "np.random.seed(42)\n",
        "shap_sample_idx = np.random.choice(len(X_test), size=min(3000, len(X_test)), replace=False)\n",
        "X_shap = X_test.iloc[shap_sample_idx]\n",
        "\n",
        "explainer = shap.TreeExplainer(model)\n",
        "shap_values = explainer.shap_values(X_shap)\n",
        "\n",
        "# 2.1 SHAP summary (beeswarm)\n",
        "print('Generating SHAP summary plot...')\n",
        "fig, ax = plt.subplots(figsize=(12, 10))\n",
        "shap.summary_plot(shap_values, X_shap, show=False, max_display=20)\n",
        "save_fig('14_shap_summary')\n",
        "\n",
        "# 2.2 SHAP bar plot\n",
        "fig, ax = plt.subplots(figsize=(10, 8))\n",
        "shap.summary_plot(shap_values, X_shap, plot_type='bar', show=False, max_display=20,\n",
        "                  color=PRIMARY)\n",
        "save_fig('15_shap_bar')\n",
        "\n",
        "# 2.3 SHAP values table\n",
        "shap_importance = pd.DataFrame({\n",
        "    'feature': feature_cols,\n",
        "    'mean_abs_shap': np.abs(shap_values).mean(axis=0)\n",
        "}).sort_values('mean_abs_shap', ascending=False)\n",
        "\n",
        "print('\\n\\u2500\\u2500 Top 15 Features by Mean |SHAP| \\u2500\\u2500')\n",
        "print(shap_importance.head(15).to_string(index=False))\n",
        "shap_importance.to_csv(OUTPUT_DIR / 'shap_importance.csv', index=False)\n",
        "\n",
        "# 2.4 SHAP dependence plots for top 4 features\n",
        "top4 = shap_importance['feature'].head(4).tolist()\n",
        "fig, axes = plt.subplots(2, 2, figsize=(16, 12))\n",
        "for ax, feat in zip(axes.flat, top4):\n",
        "    shap.dependence_plot(feat, shap_values, X_shap, ax=ax, show=False)\n",
        "    ax.set_title(f'SHAP Dependence: {feat}')\n",
        "save_fig('16_shap_dependence_top4')\n"
    ]
})

# ── 3. Error Analysis ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": [
        "## 3. Error Analysis\n",
        "\n",
        "Error distributions per model, per-product RMSE distributions,\n",
        "products with most/least improvement over naive, and error by day of week.\n"
    ]
})

cells.append({
    "cell_type": "code", "id": uid(), "metadata": {}, "execution_count": None,
    "outputs": [],
    "source": [
        "print('\\n' + '=' * 60)\n",
        "print('3. ERROR ANALYSIS')\n",
        "print('=' * 60)\n",
        "\n",
        "pf = preds.copy()\n",
        "models = ['Naive_lag7', 'RandomForest', 'XGBoost', 'LightGBM']\n",
        "\n",
        "# 3.1 Error distribution per model\n",
        "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
        "for ax, m in zip(axes.flat, models):\n",
        "    errors = pf[TARGET] - pf[m]\n",
        "    ax.hist(errors, bins=80, edgecolor='white', alpha=0.7, density=True, color=get_model_color(m))\n",
        "    ax.axvline(x=0, color=ALERT, linewidth=1, linestyle='--')\n",
        "    ax.set_title(f'{m} \\u2014 Error Distribution')\n",
        "    ax.set_xlabel('Error (Actual - Predicted)')\n",
        "    ax.set_ylabel('Density')\n",
        "    ax.set_xlim(-15, 15)\n",
        "save_fig('17_error_distributions')\n",
        "\n",
        "# 3.2 Per-product RMSE distribution\n",
        "print('\\nPer-product RMSE distribution:')\n",
        "prod_rmse = {}\n",
        "for m in models:\n",
        "    per_prod = pf.groupby('producto').apply(\n",
        "        lambda g: np.sqrt(mean_squared_error(g[TARGET], g[m])), include_groups=False\n",
        "    )\n",
        "    prod_rmse[m] = per_prod\n",
        "\n",
        "prod_rmse_df = pd.DataFrame(prod_rmse)\n",
        "print(prod_rmse_df.describe().round(3))\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(12, 6))\n",
        "prod_rmse_df[['Naive_lag7', 'LightGBM']].plot(\n",
        "    kind='hist', bins=50, alpha=0.6, ax=ax, edgecolor='white',\n",
        "    color=[PRIMARY, SECONDARY]\n",
        ")\n",
        "ax.set_title('Per-Product RMSE Distribution: Naive vs LightGBM')\n",
        "ax.set_xlabel('RMSE')\n",
        "ax.set_ylabel('Number of Products')\n",
        "save_fig('18_per_product_rmse_dist')\n",
        "\n",
        "# 3.3 Products where ML improves most / least\n",
        "improvement = prod_rmse_df['Naive_lag7'] - prod_rmse_df['LightGBM']\n",
        "improvement = improvement.sort_values(ascending=False)\n",
        "print(f'\\n\\u2500\\u2500 Top 10 products with most improvement (Naive\\u2192LGBM) \\u2500\\u2500')\n",
        "print(improvement.head(10).round(3))\n",
        "print(f'\\n\\u2500\\u2500 10 products with least improvement \\u2500\\u2500')\n",
        "print(improvement.tail(10).round(3))\n"
    ]
})

# ── 3.4 Error by day of week + 3.5 Promo comparison ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": [
        "### Error by Day of Week & Promo vs Non-Promo Comparison\n"
    ]
})

cells.append({
    "cell_type": "code", "id": uid(), "metadata": {}, "execution_count": None,
    "outputs": [],
    "source": [
        "# 3.4 Error by day of week\n",
        "pf_with_dow = pf.merge(\n",
        "    df[['producto', 'fecha', 'day_of_week']].drop_duplicates(),\n",
        "    on=['producto', 'fecha'], how='left'\n",
        ")\n",
        "\n",
        "dow_errors = {}\n",
        "for m in ['LightGBM']:\n",
        "    pf_with_dow['_err'] = np.abs(pf_with_dow[TARGET] - pf_with_dow[m])\n",
        "    dow_errors[m] = pf_with_dow.groupby('day_of_week')['_err'].mean()\n",
        "\n",
        "dow_df = pd.DataFrame(dow_errors)\n",
        "dow_df.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(10, 5))\n",
        "dow_df.plot(kind='bar', ax=ax, color=PRIMARY, edgecolor='white', legend=False)\n",
        "ax.set_title('LightGBM Mean Absolute Error by Day of Week')\n",
        "ax.set_ylabel('MAE')\n",
        "ax.set_xlabel('')\n",
        "plt.xticks(rotation=45)\n",
        "save_fig('19_mae_by_day_of_week')\n",
        "\n",
        "# 3.5 Promo vs Non-Promo detailed comparison\n",
        "print('\\n\\u2500\\u2500 Detailed Promo vs Non-Promo Metrics \\u2500\\u2500')\n",
        "comparison_rows = []\n",
        "for label, mask in [('Overall', pf[TARGET] >= 0), ('Non-Promo', pf['en_promo'] == 0), ('Promo', pf['en_promo'] == 1)]:\n",
        "    subset = pf[mask]\n",
        "    for m in models:\n",
        "        metrics = calc_metrics(subset[TARGET], subset[m])\n",
        "        metrics['segment'] = label\n",
        "        metrics['model'] = m\n",
        "        comparison_rows.append(metrics)\n",
        "\n",
        "comp_df = pd.DataFrame(comparison_rows)\n",
        "comp_pivot = comp_df.pivot_table(index=['segment', 'model'], values=['MAE', 'RMSE', 'MAPE', 'sigma_e'])\n",
        "print(comp_pivot.round(3).to_string())\n",
        "comp_df.to_csv(OUTPUT_DIR / 'promo_comparison.csv', index=False)\n",
        "\n",
        "print('\\n' + '=' * 60)\n",
        "print('EVALUATION COMPLETE')\n",
        "print('=' * 60)\n"
    ]
})

# ── Write notebook ──
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": cells
}

with open("TFM_Ona_Code/04_evaluation.ipynb", "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✓ Built 04_evaluation.ipynb")
