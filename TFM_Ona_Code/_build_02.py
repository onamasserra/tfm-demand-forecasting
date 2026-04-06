"""Build 02_preprocessing.ipynb from 02_preprocessing.py"""
import json
from uuid import uuid4

def uid():
    return uuid4().hex[:8]

cells = []

# ── Title markdown ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": [
        "# 02 — Preprocessing: Data Cleaning, Transformation & Feature Engineering\n",
        "\n",
        "**TFM: Demand Forecasting for Inventory Management Using Machine Learning**\n",
        "\n",
        "Author: Ona Mas i Serra\n",
        "\n",
        "This notebook cleans the merged raw data from the EDA step, engineers lag/rolling/promotional features,\n",
        "handles stock-break imputation and outlier capping, and produces the final `features.parquet` ready for modelling.\n"
    ]
})

# ── Setup & Configuration ──
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
        "import sys\n",
        "sys.path.insert(0, '/Users/omasiser/Library/CloudStorage/OneDrive-amazon.com/UOC/TFM forecasting/knowledge base/TFM_Ona_Code')\n",
        "from style_config import apply_style, PRIMARY\n",
        "\n",
        "%matplotlib inline\n",
        "\n",
        "BASE_DIR = Path('/Users/omasiser/Library/CloudStorage/OneDrive-amazon.com/UOC/TFM forecasting/knowledge base/TFM_Ona_Code')\n",
        "INPUT_PATH = BASE_DIR / 'output' / 'processed' / 'merged_raw.parquet'\n",
        "OUTPUT_DIR = BASE_DIR / 'output' / 'processed'\n",
        "PLOT_DIR = BASE_DIR / 'output' / 'preprocessing'\n",
        "PLOT_DIR.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "apply_style()\n",
        "plt.rcParams.update({'figure.figsize': (14, 6)})\n",
        "\n",
        "def save_fig(name):\n",
        "    plt.tight_layout()\n",
        "    plt.savefig(PLOT_DIR / f'{name}.png', bbox_inches='tight')\n",
        "    plt.show()\n",
        "    print(f'   → Saved {name}.png')\n"
    ]
})

# ── 1. Load Merged Data ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": [
        "## 1. Load Merged Data\n",
        "\n",
        "Load the merged raw parquet file produced by the EDA notebook.\n"
    ]
})

cells.append({
    "cell_type": "code", "id": uid(), "metadata": {}, "execution_count": None,
    "outputs": [],
    "source": [
        "print('=' * 60)\n",
        "print('1. LOADING MERGED DATA')\n",
        "print('=' * 60)\n",
        "\n",
        "df = pd.read_parquet(INPUT_PATH)\n",
        "print(f'Shape: {df.shape}')\n",
        "print(f\"Date range: {df['fecha'].min().date()} → {df['fecha'].max().date()}\")\n",
        "print(f\"Products: {df['producto'].nunique()}\")\n"
    ]
})

# ── 2. Data Cleaning ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": [
        "## 2. Data Cleaning\n",
        "\n",
        "Handle missing calendar values, missing stock, negative values, stock-break imputation,\n",
        "and outlier treatment (3σ cap per product/month).\n"
    ]
})

cells.append({
    "cell_type": "code", "id": uid(), "metadata": {}, "execution_count": None,
    "outputs": [],
    "source": [
        "print('\\n' + '=' * 60)\n",
        "print('2. DATA CLEANING')\n",
        "print('=' * 60)\n",
        "\n",
        "n_before = len(df)\n",
        "\n",
        "# 2.1 Handle missing calendario values\n",
        "missing_cal = df['bolOpen'].isnull().sum()\n",
        "print(f'Missing calendario rows: {missing_cal}')\n",
        "df['bolOpen'] = df['bolOpen'].fillna(1).astype(int)\n",
        "df['bolHoliday'] = df['bolHoliday'].fillna(0).astype(int)\n",
        "\n",
        "# 2.2 Handle missing stock\n",
        "missing_stock = df['udsStock'].isnull().sum()\n",
        "print(f'Missing stock rows: {missing_stock}')\n",
        "df = df.sort_values(['producto', 'fecha'])\n",
        "df['udsStock'] = df.groupby('producto')['udsStock'].ffill().bfill()\n",
        "remaining_null = df['udsStock'].isnull().sum()\n",
        "print(f'Stock nulls after ffill/bfill: {remaining_null}')\n",
        "if remaining_null > 0:\n",
        "    df['udsStock'] = df['udsStock'].fillna(0)\n",
        "\n",
        "# 2.3 Negative values\n",
        "neg_sales = (df['udsVenta'] < 0).sum()\n",
        "neg_stock = (df['udsStock'] < 0).sum()\n",
        "print(f'Negative sales: {neg_sales} → clipped to 0')\n",
        "print(f'Negative stock: {neg_stock} → clipped to 0')\n",
        "df['udsVenta'] = df['udsVenta'].clip(lower=0)\n",
        "df['udsStock'] = df['udsStock'].clip(lower=0)\n",
        "\n",
        "# 2.4 Stock break imputation\n",
        "stock_break_mask = (df['udsVenta'] == 0) & (df['udsStock'] == 0)\n",
        "n_breaks = stock_break_mask.sum()\n",
        "print(f'\\nStock breaks before imputation: {n_breaks:,}')\n",
        "\n",
        "df['_dow'] = df['fecha'].dt.dayofweek\n",
        "df['_impute'] = df.groupby(['producto', '_dow'])['udsVenta'].transform(\n",
        "    lambda x: x.rolling(window=4, min_periods=1).mean().shift(1)\n",
        ")\n",
        "product_dow_mean = df.groupby(['producto', '_dow'])['udsVenta'].transform('mean')\n",
        "df['_impute'] = df['_impute'].fillna(product_dow_mean)\n",
        "\n",
        "df.loc[stock_break_mask, 'udsVenta'] = df.loc[stock_break_mask, '_impute'].round().clip(lower=0)\n",
        "df.drop(columns=['_dow', '_impute'], inplace=True)\n",
        "\n",
        "n_breaks_after = ((df['udsVenta'] == 0) & (df['udsStock'] == 0)).sum()\n",
        "print(f'Stock breaks after imputation: {n_breaks_after:,}')\n",
        "\n",
        "# 2.5 Outlier treatment\n",
        "print('\\n── Outlier Treatment (3σ cap per product/month) ──')\n",
        "df['_ym'] = df['fecha'].dt.to_period('M')\n",
        "grp = df.groupby(['producto', '_ym'])['udsVenta']\n",
        "upper = grp.transform('mean') + 3 * grp.transform('std')\n",
        "outlier_mask = df['udsVenta'] > upper\n",
        "n_outliers = outlier_mask.sum()\n",
        "print(f'Outliers detected: {n_outliers:,} ({100*n_outliers/len(df):.2f}%)')\n",
        "df.loc[outlier_mask, 'udsVenta'] = upper[outlier_mask].round()\n",
        "df.drop(columns=['_ym'], inplace=True)\n",
        "\n",
        "print(f'\\nCleaning complete. Shape: {df.shape}')\n"
    ]
})

# ── 3. Feature Engineering ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": [
        "## 3. Feature Engineering\n",
        "\n",
        "Create calendar features, lag features (1, 7, 14, 28 days), rolling statistics,\n",
        "EWMA, promotional features, and product-level static features.\n"
    ]
})

cells.append({
    "cell_type": "code", "id": uid(), "metadata": {}, "execution_count": None,
    "outputs": [],
    "source": [
        "print('\\n' + '=' * 60)\n",
        "print('3. FEATURE ENGINEERING')\n",
        "print('=' * 60)\n",
        "\n",
        "df = df.sort_values(['producto', 'fecha']).reset_index(drop=True)\n",
        "\n",
        "# 3.1 Calendar features\n",
        "print('3.1 Calendar features...')\n",
        "df['week_of_month'] = (df['fecha'].dt.day - 1) // 7 + 1\n",
        "df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)\n",
        "df['is_month_start'] = (df['fecha'].dt.day <= 3).astype(int)\n",
        "df['is_month_end'] = (df['fecha'].dt.day >= 28).astype(int)\n",
        "\n",
        "# 3.2 Lag features\n",
        "print('3.2 Lag features (1, 7, 14, 28 days)...')\n",
        "for lag in [1, 7, 14, 28]:\n",
        "    df[f'lag_{lag}'] = df.groupby('producto')['udsVenta'].shift(lag)\n",
        "\n",
        "# 3.3 Rolling statistics\n",
        "print('3.3 Rolling statistics (7d, 14d, 28d)...')\n",
        "for window in [7, 14, 28]:\n",
        "    rolled = df.groupby('producto')['udsVenta'].shift(1).groupby(df['producto'])\n",
        "    df[f'roll_mean_{window}'] = rolled.transform(lambda x: x.rolling(window, min_periods=1).mean())\n",
        "    df[f'roll_std_{window}'] = rolled.transform(lambda x: x.rolling(window, min_periods=1).std())\n",
        "    df[f'roll_min_{window}'] = rolled.transform(lambda x: x.rolling(window, min_periods=1).min())\n",
        "    df[f'roll_max_{window}'] = rolled.transform(lambda x: x.rolling(window, min_periods=1).max())\n",
        "\n",
        "# 3.4 EWMA\n",
        "print('3.4 EWMA (spans: 7, 28)...')\n",
        "shifted = df.groupby('producto')['udsVenta'].shift(1)\n",
        "for span in [7, 28]:\n",
        "    df[f'ewma_{span}'] = shifted.groupby(df['producto']).transform(\n",
        "        lambda x: x.ewm(span=span, min_periods=1).mean()\n",
        "    )\n"
    ]
})

# ── 3.5 Promotional features ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": [
        "### 3.5 Promotional & Product-Level Features\n",
        "\n",
        "Compute days since last promo ended, post-promo lag effects, promo×lag interaction,\n",
        "and per-product static aggregates (mean, std, median sales).\n"
    ]
})

cells.append({
    "cell_type": "code", "id": uid(), "metadata": {}, "execution_count": None,
    "outputs": [],
    "source": [
        "# 3.5 Promotional features\n",
        "print('3.5 Promotional features...')\n",
        "df['_promo_change'] = df.groupby('producto')['en_promo'].diff().fillna(0)\n",
        "df['_promo_end'] = (df['_promo_change'] == -1).astype(int)\n",
        "\n",
        "def days_since_event(series):\n",
        "    result = pd.Series(np.nan, index=series.index)\n",
        "    last_event = np.nan\n",
        "    for i, val in enumerate(series):\n",
        "        if val == 1:\n",
        "            last_event = 0\n",
        "        elif not np.isnan(last_event):\n",
        "            last_event += 1\n",
        "        result.iloc[i] = last_event\n",
        "    return result\n",
        "\n",
        "df['days_since_promo_end'] = df.groupby('producto')['_promo_end'].transform(days_since_event)\n",
        "df['days_since_promo_end'] = df['days_since_promo_end'].fillna(-1)\n",
        "\n",
        "df['post_promo_7d'] = ((df['days_since_promo_end'] >= 0) & (df['days_since_promo_end'] <= 7)).astype(int)\n",
        "df['promo_x_lag7'] = df['en_promo'] * df['lag_7'].fillna(0)\n",
        "df.drop(columns=['_promo_change', '_promo_end'], inplace=True)\n",
        "\n",
        "# 3.6 Product-level static features\n",
        "print('3.6 Product-level features...')\n",
        "prod_agg = df.groupby('producto')['udsVenta'].agg(\n",
        "    prod_mean_sales='mean',\n",
        "    prod_std_sales='std',\n",
        "    prod_median_sales='median'\n",
        ").reset_index()\n",
        "df = df.merge(prod_agg, on='producto', how='left')\n"
    ]
})

# ── 4. Handle NaNs ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": [
        "## 4. Handle NaNs from Feature Engineering\n",
        "\n",
        "Drop the first 28 days per product (warmup period for lag_28) and fill any remaining NaNs.\n"
    ]
})

cells.append({
    "cell_type": "code", "id": uid(), "metadata": {}, "execution_count": None,
    "outputs": [],
    "source": [
        "print('\\n' + '=' * 60)\n",
        "print('4. HANDLING NaNs FROM FEATURE ENGINEERING')\n",
        "print('=' * 60)\n",
        "\n",
        "warmup_days = 28\n",
        "df['_day_rank'] = df.groupby('producto')['fecha'].rank(method='dense').astype(int)\n",
        "n_before_warmup = len(df)\n",
        "df = df[df['_day_rank'] > warmup_days].copy()\n",
        "df.drop(columns=['_day_rank'], inplace=True)\n",
        "n_after_warmup = len(df)\n",
        "\n",
        "print(f'Rows before warmup removal: {n_before_warmup:,}')\n",
        "print(f'Rows after warmup removal:  {n_after_warmup:,}')\n",
        "print(f'Removed: {n_before_warmup - n_after_warmup:,} ({100*(n_before_warmup-n_after_warmup)/n_before_warmup:.1f}%)')\n",
        "\n",
        "remaining_nans = df.isnull().sum()\n",
        "has_nans = remaining_nans[remaining_nans > 0]\n",
        "if len(has_nans) > 0:\n",
        "    print(f'\\nRemaining NaNs:\\n{has_nans}')\n",
        "    df = df.fillna(0)\n",
        "else:\n",
        "    print('No remaining NaNs.')\n"
    ]
})

# ── 5. Feature Summary & Visualisation ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": [
        "## 5. Feature Summary & Visualisation\n",
        "\n",
        "Summarise the engineered features and plot their correlation with the target variable (udsVenta).\n"
    ]
})

cells.append({
    "cell_type": "code", "id": uid(), "metadata": {}, "execution_count": None,
    "outputs": [],
    "source": [
        "print('\\n' + '=' * 60)\n",
        "print('5. FEATURE SUMMARY')\n",
        "print('=' * 60)\n",
        "\n",
        "feature_cols = [c for c in df.columns if c not in ['producto', 'idSecuencia', 'fecha', 'day_name']]\n",
        "print(f'Total features: {len(feature_cols)}')\n",
        "print(f'Feature list: {feature_cols}')\n",
        "print(f'\\nFinal shape: {df.shape}')\n",
        "print(f\"Date range: {df['fecha'].min().date()} → {df['fecha'].max().date()}\")\n",
        "print(f\"Products: {df['producto'].nunique()}\")\n",
        "\n",
        "corr_with_target = df[feature_cols].corr()['udsVenta'].drop('udsVenta').sort_values(ascending=False)\n",
        "print(f'\\n── Top 15 features correlated with udsVenta ──')\n",
        "print(corr_with_target.head(15))\n",
        "print(f'\\n── Bottom 5 ──')\n",
        "print(corr_with_target.tail(5))\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(10, 12))\n",
        "corr_with_target.plot(kind='barh', ax=ax, color=PRIMARY)\n",
        "ax.set_title('Feature Correlation with udsVenta (Sales)')\n",
        "ax.set_xlabel('Pearson Correlation')\n",
        "ax.axvline(x=0, color='black', linewidth=0.5)\n",
        "save_fig('10_feature_correlations')\n"
    ]
})

# ── 6. Save Processed Data ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": [
        "## 6. Save Processed Data\n",
        "\n",
        "Export the final feature-engineered dataset to parquet and save the feature name list.\n"
    ]
})

cells.append({
    "cell_type": "code", "id": uid(), "metadata": {}, "execution_count": None,
    "outputs": [],
    "source": [
        "print('\\n' + '=' * 60)\n",
        "print('6. SAVING PROCESSED DATA')\n",
        "print('=' * 60)\n",
        "\n",
        "df.to_parquet(OUTPUT_DIR / 'features.parquet', index=False)\n",
        "print(f\"✓ Saved features.parquet ({df.shape[0]:,} rows × {df.shape[1]} cols)\")\n",
        "\n",
        "feature_names = [c for c in feature_cols if c != 'udsVenta']\n",
        "pd.Series(feature_names).to_csv(OUTPUT_DIR / 'feature_names.csv', index=False, header=False)\n",
        "print(f'✓ Saved feature_names.csv ({len(feature_names)} features)')\n",
        "\n",
        "print('\\n' + '=' * 60)\n",
        "print('PREPROCESSING COMPLETE')\n",
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

with open("TFM_Ona_Code/02_preprocessing.ipynb", "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✓ Built 02_preprocessing.ipynb")
