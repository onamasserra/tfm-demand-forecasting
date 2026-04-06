"""Build 05_economic_impact.ipynb from 05_economic_impact.py"""
import json
from uuid import uuid4

def uid():
    return uuid4().hex[:8]

cells = []

# ── Title ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": [
        "# 05 — Economic Impact: Safety Stock & Cost Analysis\n",
        "\n",
        "**TFM: Demand Forecasting for Inventory Management Using Machine Learning**\n",
        "\n",
        "Author: Ona Mas i Serra\n",
        "\n",
        "This notebook applies the safety stock methodology (SS = z × RMSE × √(T+L)) to compare\n",
        "the economic impact of using ML-based forecasts vs the naive baseline.\n",
        "It uses real per-product prices and lead times provided by the thesis supervisor.\n"
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
        "from pathlib import Path\n",
        "from sklearn.metrics import mean_squared_error\n",
        "import sys\n",
        "sys.path.insert(0, '/Users/omasiser/Library/CloudStorage/OneDrive-amazon.com/UOC/TFM forecasting/knowledge base/TFM_Ona_Code')\n",
        "from style_config import (apply_style, PRIMARY, SECONDARY, ACCENT, NEUTRAL,\n",
        "                          DARK_BLUE, AIR_BLUE, TEAL, LIGHT_BLUE,\n",
        "                          AMBER, CORAL, SLATE,\n",
        "                          get_model_color, get_model_colors)\n",
        "\n",
        "%matplotlib inline\n",
        "\n",
        "BASE_DIR = Path('/Users/omasiser/Library/CloudStorage/OneDrive-amazon.com/UOC/TFM forecasting/knowledge base/TFM_Ona_Code')\n",
        "PREDS_PATH = BASE_DIR / 'output' / 'models' / 'predictions_last_fold.parquet'\n",
        "DATA_DIR = Path('/Users/omasiser/Library/CloudStorage/OneDrive-amazon.com/UOC/TFM forecasting/knowledge base') / 'Datos'\n",
        "OUTPUT_DIR = BASE_DIR / 'output' / 'economic'\n",
        "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "apply_style()\n",
        "\n",
        "TARGET = 'udsVenta'\n",
        "MODELS = ['Naive_lag7', 'RandomForest', 'XGBoost', 'LightGBM']\n",
        "\n",
        "SERVICE_LEVEL = 0.95\n",
        "Z_SCORE = 1.6449\n",
        "HOLDING_COST_RATE = 0.05\n",
        "\n",
        "def save_fig(name):\n",
        "    plt.tight_layout()\n",
        "    plt.savefig(OUTPUT_DIR / f'{name}.png', bbox_inches='tight', dpi=120)\n",
        "    plt.show()\n",
        "    print(f'   \\u2192 Saved {name}.png')\n"
    ]
})

# ── 1. Load Data ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": [
        "## 1. Load Data\n",
        "\n",
        "Load predictions, per-product lead times (ciclo de aprovisionamiento), and average prices.\n"
    ]
})

cells.append({
    "cell_type": "code", "id": uid(), "metadata": {}, "execution_count": None,
    "outputs": [],
    "source": [
        "print('=' * 60)\n",
        "print('1. LOADING DATA')\n",
        "print('=' * 60)\n",
        "\n",
        "pf = pd.read_parquet(PREDS_PATH)\n",
        "print(f\"Predictions: {len(pf):,} rows, {pf['producto'].nunique()} products\")\n",
        "print(f\"Date range: {pf['fecha'].min().date()} \\u2192 {pf['fecha'].max().date()}\")\n",
        "print(f\"Days: {pf['fecha'].nunique()}\")\n",
        "\n",
        "ciclo = pd.read_excel(DATA_DIR / 'DatosCicloAprovisionamiento.xlsx')\n",
        "precio = pd.read_excel(DATA_DIR / 'DatosPrecioMedio.xlsx')\n",
        "\n",
        "print(f\"\\nCiclo aprovisionamiento: {len(ciclo)} products\")\n",
        "print(f\"  diasEntrePedidos: mean={ciclo['diasEntrePedidos'].mean():.1f}, range=[{ciclo['diasEntrePedidos'].min()}, {ciclo['diasEntrePedidos'].max()}]\")\n",
        "print(f\"  diasLeadtime: mean={ciclo['diasLeadtime'].mean():.1f}, range=[{ciclo['diasLeadtime'].min()}, {ciclo['diasLeadtime'].max()}]\")\n",
        "print(f\"\\nPrecio medio: {len(precio)} products\")\n",
        "print(f\"  eurPrecioMedio: mean=\\u20ac{precio['eurPrecioMedio'].mean():.2f}, range=[\\u20ac{precio['eurPrecioMedio'].min():.2f}, \\u20ac{precio['eurPrecioMedio'].max():.2f}]\")\n",
        "\n",
        "cost_params = ciclo.merge(precio, on='producto', how='inner')\n",
        "cost_params['protection_days'] = cost_params['diasEntrePedidos'] + cost_params['diasLeadtime']\n",
        "cost_params['sqrt_protection'] = np.sqrt(cost_params['protection_days'])\n",
        "\n",
        "print(f\"\\nProtection period (T+L): mean={cost_params['protection_days'].mean():.1f} days, \"\n",
        "      f\"range=[{cost_params['protection_days'].min()}, {cost_params['protection_days'].max()}]\")\n",
        "\n",
        "pred_products = pf['producto'].unique()\n",
        "cost_params = cost_params[cost_params['producto'].isin(pred_products)]\n",
        "print(f'Products with cost data & predictions: {len(cost_params)}')\n"
    ]
})

# ── 2. Per-Product RMSE ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": ["## 2. Per-Product RMSE\n"]
})

cells.append({
    "cell_type": "code", "id": uid(), "metadata": {}, "execution_count": None,
    "outputs": [],
    "source": [
        "print('\\n' + '=' * 60)\n",
        "print('2. PER-PRODUCT RMSE')\n",
        "print('=' * 60)\n",
        "\n",
        "rmse_per_product = {}\n",
        "for model in MODELS:\n",
        "    per_prod = pf.groupby('producto').apply(\n",
        "        lambda g: np.sqrt(mean_squared_error(g[TARGET], g[model])), include_groups=False\n",
        "    )\n",
        "    rmse_per_product[model] = per_prod\n",
        "\n",
        "rmse_df = pd.DataFrame(rmse_per_product)\n",
        "print(f'\\n\\u2500\\u2500 Per-product RMSE Summary ({len(rmse_df)} products) \\u2500\\u2500')\n",
        "print(rmse_df.describe().round(3))\n"
    ]
})

# ── 3. Safety Stock ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": [
        "## 3. Safety Stock Calculation\n",
        "\n",
        "SS_i = z × RMSE_i × √(T_i + L_i), where z = 1.6449 for 95% service level.\n"
    ]
})

cells.append({
    "cell_type": "code", "id": uid(), "metadata": {}, "execution_count": None,
    "outputs": [],
    "source": [
        "print('\\n' + '=' * 60)\n",
        "print('3. SAFETY STOCK CALCULATION')\n",
        "print('=' * 60)\n",
        "\n",
        "print(f'Service level: {SERVICE_LEVEL*100:.0f}% (z = {Z_SCORE:.4f})')\n",
        "print(f'Formula: SS = z \\u00d7 RMSE \\u00d7 \\u221a(diasEntrePedidos + diasLeadtime)')\n",
        "\n",
        "ss_results = {}\n",
        "for model in MODELS:\n",
        "    model_df = cost_params[['producto', 'sqrt_protection']].copy()\n",
        "    model_df = model_df.merge(rmse_df[[model]].rename(columns={model: 'rmse'}),\n",
        "                               left_on='producto', right_index=True, how='inner')\n",
        "    model_df['safety_stock'] = Z_SCORE * model_df['rmse'] * model_df['sqrt_protection']\n",
        "    ss_results[model] = model_df.set_index('producto')['safety_stock']\n",
        "\n",
        "ss_df = pd.DataFrame(ss_results)\n",
        "print(f'\\n\\u2500\\u2500 Safety Stock (units) Summary \\u2500\\u2500')\n",
        "print(ss_df.describe().round(2))\n",
        "\n",
        "total_ss = ss_df.sum()\n",
        "print(f'\\n\\u2500\\u2500 Total Safety Stock (all products) \\u2500\\u2500')\n",
        "for model in MODELS:\n",
        "    print(f'  {model:15s}: {total_ss[model]:,.0f} units')\n"
    ]
})

# ── 4. Cost Calculation ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": [
        "## 4. Daily & Annual Cost\n",
        "\n",
        "DailyCost = 5% × price × SS; AnnualCost = 365 × DailyCost.\n"
    ]
})

cells.append({
    "cell_type": "code", "id": uid(), "metadata": {}, "execution_count": None,
    "outputs": [],
    "source": [
        "print('\\n' + '=' * 60)\n",
        "print('4. COST CALCULATION')\n",
        "print('=' * 60)\n",
        "\n",
        "print(f'Daily cost formula: {HOLDING_COST_RATE*100:.0f}% \\u00d7 price \\u00d7 SS')\n",
        "\n",
        "price_map = cost_params.set_index('producto')['eurPrecioMedio']\n",
        "\n",
        "cost_daily = {}\n",
        "cost_annual = {}\n",
        "for model in MODELS:\n",
        "    daily = HOLDING_COST_RATE * price_map * ss_df[model]\n",
        "    cost_daily[model] = daily\n",
        "    cost_annual[model] = daily * 365\n",
        "\n",
        "cost_daily_df = pd.DataFrame(cost_daily)\n",
        "cost_annual_df = pd.DataFrame(cost_annual)\n",
        "\n",
        "print(f'\\n\\u2500\\u2500 Annual Safety Stock Cost per Product (\\u20ac) \\u2500\\u2500')\n",
        "print(cost_annual_df.describe().round(2))\n",
        "\n",
        "total_annual = cost_annual_df.sum()\n",
        "print(f'\\n\\u2500\\u2500 Total Annual Safety Stock Cost \\u2500\\u2500')\n",
        "for model in MODELS:\n",
        "    print(f'  {model:15s}: \\u20ac{total_annual[model]:,.2f}')\n",
        "\n",
        "naive_total = total_annual['Naive_lag7']\n",
        "print(f'\\n\\u2500\\u2500 Annual Savings vs Naive Baseline \\u2500\\u2500')\n",
        "for model in MODELS:\n",
        "    if model == 'Naive_lag7':\n",
        "        continue\n",
        "    saving = naive_total - total_annual[model]\n",
        "    pct = (saving / naive_total) * 100\n",
        "    print(f'  {model:15s}: \\u20ac{saving:,.2f} saved ({pct:.1f}%)')\n"
    ]
})

# ── 5. Sensitivity Analysis ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": [
        "## 5. Sensitivity Analysis — Service Level\n",
        "\n",
        "Compare safety stock and costs across 90%, 95%, and 99% service levels.\n"
    ]
})

cells.append({
    "cell_type": "code", "id": uid(), "metadata": {}, "execution_count": None,
    "outputs": [],
    "source": [
        "print('\\n' + '=' * 60)\n",
        "print('5. SENSITIVITY ANALYSIS \\u2014 SERVICE LEVEL')\n",
        "print('=' * 60)\n",
        "\n",
        "SERVICE_LEVELS = {0.90: 1.2816, 0.95: 1.6449, 0.99: 2.3263}\n",
        "\n",
        "sensitivity_rows = []\n",
        "for sl, z_val in SERVICE_LEVELS.items():\n",
        "    for model in MODELS:\n",
        "        ss_total = (z_val * rmse_df[model] * cost_params.set_index('producto')['sqrt_protection']).dropna().sum()\n",
        "        annual_cost = HOLDING_COST_RATE * (price_map * z_val * rmse_df[model] *\n",
        "                      cost_params.set_index('producto')['sqrt_protection']).dropna().sum() * 365\n",
        "        sensitivity_rows.append({\n",
        "            'service_level': f'{sl*100:.0f}%',\n",
        "            'model': model,\n",
        "            'z_score': z_val,\n",
        "            'total_safety_stock': ss_total,\n",
        "            'annual_cost': annual_cost,\n",
        "        })\n",
        "\n",
        "sens_df = pd.DataFrame(sensitivity_rows)\n",
        "sens_pivot = sens_df.pivot(index='model', columns='service_level', values='total_safety_stock')\n",
        "print('\\n\\u2500\\u2500 Total Safety Stock by Service Level \\u2500\\u2500')\n",
        "print(sens_pivot.round(0).to_string())\n",
        "\n",
        "sens_cost_pivot = sens_df.pivot(index='model', columns='service_level', values='annual_cost')\n",
        "print('\\n\\u2500\\u2500 Annual Cost (\\u20ac) by Service Level \\u2500\\u2500')\n",
        "print(sens_cost_pivot.round(0).to_string())\n",
        "\n",
        "sens_df.to_csv(OUTPUT_DIR / 'sensitivity_service_level.csv', index=False)\n"
    ]
})

# ── 6. Summary Table ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": ["## 6. Summary Table\n"]
})

cells.append({
    "cell_type": "code", "id": uid(), "metadata": {}, "execution_count": None,
    "outputs": [],
    "source": [
        "print('\\n' + '=' * 60)\n",
        "print('6. SUMMARY TABLE')\n",
        "print('=' * 60)\n",
        "\n",
        "summary_rows = []\n",
        "for model in MODELS:\n",
        "    summary_rows.append({\n",
        "        'model': model,\n",
        "        'avg_rmse': rmse_df[model].mean(),\n",
        "        'total_safety_stock_units': total_ss[model],\n",
        "        'annual_cost_eur': total_annual[model],\n",
        "        'saving_vs_naive_eur': naive_total - total_annual[model],\n",
        "        'saving_vs_naive_pct': (naive_total - total_annual[model]) / naive_total * 100,\n",
        "    })\n",
        "\n",
        "summary = pd.DataFrame(summary_rows).set_index('model')\n",
        "print(summary.round(2).to_string())\n",
        "summary.to_csv(OUTPUT_DIR / 'economic_summary.csv')\n",
        "print(f'\\n\\u2713 Saved economic_summary.csv')\n"
    ]
})

# ── 7. Plots ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": [
        "## 7. Plots\n",
        "\n",
        "Annual cost comparison, safety stock by service level, per-product distributions,\n",
        "and top 20 products by cost savings.\n"
    ]
})

cells.append({
    "cell_type": "code", "id": uid(), "metadata": {}, "execution_count": None,
    "outputs": [],
    "source": [
        "print('\\n' + '=' * 60)\n",
        "print('7. GENERATING PLOTS')\n",
        "print('=' * 60)\n",
        "\n",
        "# 7.1 Annual cost comparison\n",
        "fig, ax = plt.subplots(figsize=(10, 6))\n",
        "colors = [SLATE if m == 'Naive_lag7' else DARK_BLUE for m in MODELS]\n",
        "bars = ax.bar(MODELS, [total_annual[m] for m in MODELS], color=colors, edgecolor='white')\n",
        "ax.set_ylabel('Annual Safety Stock Cost (\\u20ac)')\n",
        "ax.set_title(f'Annual Safety Stock Cost by Model (Service Level {SERVICE_LEVEL*100:.0f}%)')\n",
        "for bar, model in zip(bars, MODELS):\n",
        "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,\n",
        "            f'\\u20ac{total_annual[model]:,.0f}', ha='center', va='bottom', fontsize=9)\n",
        "plt.xticks(rotation=15)\n",
        "save_fig('20_annual_cost_comparison')\n",
        "\n",
        "# 7.2 Safety stock by service level\n",
        "fig, ax = plt.subplots(figsize=(10, 6))\n",
        "bar_width = 0.2\n",
        "x = np.arange(len(MODELS))\n",
        "sl_colors = [DARK_BLUE, LIGHT_BLUE, TEAL]\n",
        "for i, (sl, z_val) in enumerate(SERVICE_LEVELS.items()):\n",
        "    vals = []\n",
        "    for m in MODELS:\n",
        "        v = (z_val * rmse_df[m] * cost_params.set_index('producto')['sqrt_protection']).dropna().sum()\n",
        "        vals.append(v)\n",
        "    ax.bar(x + i * bar_width, vals, bar_width, label=f'{sl*100:.0f}%', color=sl_colors[i])\n",
        "ax.set_xticks(x + bar_width)\n",
        "ax.set_xticklabels(MODELS, rotation=15)\n",
        "ax.set_ylabel('Total Safety Stock (units)')\n",
        "ax.set_title('Safety Stock by Model and Service Level')\n",
        "ax.legend(title='Service Level')\n",
        "save_fig('21_safety_stock_by_service_level')\n",
        "\n",
        "# 7.3 Per-product safety stock distribution\n",
        "fig, ax = plt.subplots(figsize=(12, 6))\n",
        "ss_compare = pd.DataFrame({\n",
        "    'Naive_lag7': ss_df['Naive_lag7'],\n",
        "    'XGBoost': ss_df['XGBoost'],\n",
        "})\n",
        "ss_compare.plot(kind='hist', bins=50, alpha=0.6, ax=ax, edgecolor='white',\n",
        "                color=[SLATE, DARK_BLUE])\n",
        "ax.set_title('Per-Product Safety Stock Distribution: Naive vs XGBoost (95% SL)')\n",
        "ax.set_xlabel('Safety Stock (units)')\n",
        "ax.set_ylabel('Number of Products')\n",
        "save_fig('22_safety_stock_distribution')\n",
        "\n",
        "# 7.4 Per-product annual cost distribution\n",
        "fig, ax = plt.subplots(figsize=(12, 6))\n",
        "cost_compare = pd.DataFrame({\n",
        "    'Naive_lag7': cost_annual_df['Naive_lag7'],\n",
        "    'XGBoost': cost_annual_df['XGBoost'],\n",
        "})\n",
        "cost_compare.plot(kind='hist', bins=50, alpha=0.6, ax=ax, edgecolor='white',\n",
        "                  color=[SLATE, DARK_BLUE])\n",
        "ax.set_title('Per-Product Annual Cost Distribution: Naive vs XGBoost')\n",
        "ax.set_xlabel('Annual Safety Stock Cost (\\u20ac)')\n",
        "ax.set_ylabel('Number of Products')\n",
        "save_fig('23_annual_cost_distribution')\n",
        "\n",
        "# 7.5 Top 20 products by cost savings\n",
        "savings_per_product = cost_annual_df['Naive_lag7'] - cost_annual_df['XGBoost']\n",
        "top20_savings = savings_per_product.sort_values(ascending=False).head(20)\n",
        "fig, ax = plt.subplots(figsize=(12, 6))\n",
        "ax.barh(top20_savings.index.astype(str), top20_savings.values, color=DARK_BLUE, edgecolor='white')\n",
        "ax.set_title('Top 20 Products by Annual Cost Savings (Naive \\u2192 XGBoost)')\n",
        "ax.set_xlabel('Annual Savings (\\u20ac)')\n",
        "ax.set_ylabel('Product ID')\n",
        "ax.invert_yaxis()\n",
        "save_fig('24_top20_savings_by_product')\n",
        "\n",
        "print('\\n' + '=' * 60)\n",
        "print('ECONOMIC IMPACT ANALYSIS COMPLETE')\n",
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

with open("TFM_Ona_Code/05_economic_impact.ipynb", "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✓ Built 05_economic_impact.ipynb")
