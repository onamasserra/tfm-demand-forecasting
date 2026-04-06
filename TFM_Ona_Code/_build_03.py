"""Build 03_modelling.ipynb from 03_modelling.py"""
import json
from uuid import uuid4

def uid():
    return uuid4().hex[:8]

cells = []

# ── Title ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": [
        "# 03 — Model Training & Evaluation\n",
        "\n",
        "**TFM: Demand Forecasting for Inventory Management Using Machine Learning**\n",
        "\n",
        "Author: Ona Mas i Serra\n",
        "\n",
        "This notebook trains Naive, Random Forest, XGBoost, and LightGBM models with Optuna hyperparameter tuning,\n",
        "using rolling forecast origin (expanding window) validation with 3 folds of 30 days each.\n"
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
        "import warnings\n",
        "import json\n",
        "import time\n",
        "from pathlib import Path\n",
        "from sklearn.ensemble import RandomForestRegressor\n",
        "from sklearn.metrics import mean_absolute_error, mean_squared_error\n",
        "import xgboost as xgb\n",
        "import lightgbm as lgb\n",
        "import optuna\n",
        "import sys\n",
        "sys.path.insert(0, '/Users/omasiser/Library/CloudStorage/OneDrive-amazon.com/UOC/TFM forecasting/knowledge base/TFM_Ona_Code')\n",
        "from style_config import (apply_style, PRIMARY, ACCENT, NEUTRAL, SECONDARY, ALERT,\n",
        "                          get_model_color, get_model_colors)\n",
        "\n",
        "%matplotlib inline\n",
        "\n",
        "optuna.logging.set_verbosity(optuna.logging.WARNING)\n",
        "warnings.filterwarnings('ignore', category=UserWarning)\n",
        "\n",
        "BASE_DIR = Path('/Users/omasiser/Library/CloudStorage/OneDrive-amazon.com/UOC/TFM forecasting/knowledge base/TFM_Ona_Code')\n",
        "INPUT_PATH = BASE_DIR / 'output' / 'processed' / 'features.parquet'\n",
        "MODEL_DIR = BASE_DIR / 'output' / 'models'\n",
        "PLOT_DIR = BASE_DIR / 'output' / 'models' / 'plots'\n",
        "MODEL_DIR.mkdir(parents=True, exist_ok=True)\n",
        "PLOT_DIR.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "apply_style()\n",
        "\n",
        "def save_fig(name):\n",
        "    plt.tight_layout()\n",
        "    plt.savefig(PLOT_DIR / f'{name}.png', bbox_inches='tight', dpi=120)\n",
        "    plt.show()\n",
        "    print(f'   \\u2192 Saved {name}.png')\n",
        "\n",
        "TARGET = 'udsVenta'\n",
        "EXCLUDE = ['producto', 'idSecuencia', 'fecha', 'day_name', TARGET,\n",
        "           'prod_mean_sales', 'prod_std_sales', 'prod_median_sales']\n",
        "\n",
        "N_FOLDS = 3\n",
        "FOLD_DAYS = 30\n",
        "N_OPTUNA_TRIALS = 30\n",
        "N_TUNE_PRODUCTS = 50\n"
    ]
})

# ── 1. Load Data ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": ["## 1. Load Data\n"]
})

cells.append({
    "cell_type": "code", "id": uid(), "metadata": {}, "execution_count": None,
    "outputs": [],
    "source": [
        "print('=' * 60)\n",
        "print('1. LOADING DATA')\n",
        "print('=' * 60)\n",
        "\n",
        "df = pd.read_parquet(INPUT_PATH)\n",
        "df = df.sort_values(['producto', 'fecha']).reset_index(drop=True)\n",
        "\n",
        "feature_cols = [c for c in df.columns if c not in EXCLUDE]\n",
        "print(f'Shape: {df.shape}')\n",
        "print(f'Features: {len(feature_cols)}')\n",
        "print(f\"Date range: {df['fecha'].min().date()} \\u2192 {df['fecha'].max().date()}\")\n"
    ]
})

# ── 2. Rolling Forecast Origin Splits ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": [
        "## 2. Rolling Forecast Origin Splits\n",
        "\n",
        "Create 3 test windows of 30 days each, working backwards from the end of the dataset.\n"
    ]
})

cells.append({
    "cell_type": "code", "id": uid(), "metadata": {}, "execution_count": None,
    "outputs": [],
    "source": [
        "print('\\n' + '=' * 60)\n",
        "print('2. ROLLING FORECAST ORIGIN SPLITS')\n",
        "print('=' * 60)\n",
        "\n",
        "all_dates = sorted(df['fecha'].unique())\n",
        "max_date = all_dates[-1]\n",
        "\n",
        "folds = []\n",
        "for i in range(N_FOLDS):\n",
        "    test_end = all_dates[-(i * FOLD_DAYS + 1)] if i > 0 else max_date\n",
        "    test_end_idx = all_dates.index(test_end) if test_end in all_dates else len(all_dates) - 1 - i * FOLD_DAYS\n",
        "    test_start_idx = max(0, test_end_idx - FOLD_DAYS + 1)\n",
        "    test_dates = all_dates[test_start_idx:test_end_idx + 1]\n",
        "    train_dates = all_dates[:test_start_idx]\n",
        "    if len(train_dates) > 0 and len(test_dates) > 0:\n",
        "        folds.append({\n",
        "            'fold': N_FOLDS - i,\n",
        "            'train_end': train_dates[-1],\n",
        "            'test_start': test_dates[0],\n",
        "            'test_end': test_dates[-1],\n",
        "            'n_train_days': len(train_dates),\n",
        "            'n_test_days': len(test_dates)\n",
        "        })\n",
        "\n",
        "folds = list(reversed(folds))\n",
        "for f in folds:\n",
        "    print(f\"  Fold {f['fold']}: train\\u2192{pd.Timestamp(f['train_end']).date()} | \"\n",
        "          f\"test {pd.Timestamp(f['test_start']).date()}\\u2192{pd.Timestamp(f['test_end']).date()} \"\n",
        "          f\"({f['n_test_days']}d)\")\n"
    ]
})

# ── 3. Metrics + 4. Naive Baseline ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": [
        "## 3–4. Metrics Definition & Naive Baseline\n",
        "\n",
        "Define the evaluation metrics (MAE, RMSE, MAPE, σ_e) and compute the naive lag-7 baseline.\n"
    ]
})

cells.append({
    "cell_type": "code", "id": uid(), "metadata": {}, "execution_count": None,
    "outputs": [],
    "source": [
        "def calc_metrics(y_true, y_pred):\n",
        "    y_true = np.array(y_true, dtype=float)\n",
        "    y_pred = np.array(y_pred, dtype=float)\n",
        "    errors = y_true - y_pred\n",
        "    mae = mean_absolute_error(y_true, y_pred)\n",
        "    rmse = np.sqrt(mean_squared_error(y_true, y_pred))\n",
        "    mask = y_true > 0\n",
        "    mape = np.mean(np.abs(errors[mask] / y_true[mask])) * 100 if mask.sum() > 0 else np.nan\n",
        "    sigma_e = np.std(errors)\n",
        "    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'sigma_e': sigma_e}\n",
        "\n",
        "print('\\n' + '=' * 60)\n",
        "print('4. NAIVE BASELINE (lag_7 = same weekday last week)')\n",
        "print('=' * 60)\n",
        "\n",
        "naive_results = []\n",
        "for fold in folds:\n",
        "    test_mask = (df['fecha'] >= fold['test_start']) & (df['fecha'] <= fold['test_end'])\n",
        "    test = df[test_mask]\n",
        "    y_true = test[TARGET].values\n",
        "    y_pred = test['lag_7'].values\n",
        "    m = calc_metrics(y_true, y_pred)\n",
        "    m['fold'] = fold['fold']\n",
        "    m['model'] = 'Naive_lag7'\n",
        "    naive_results.append(m)\n",
        "    print(f\"  Fold {fold['fold']}: MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  MAPE={m['MAPE']:.1f}%  \\u03c3_e={m['sigma_e']:.3f}\")\n"
    ]
})

# ── 5. Optuna Tuning ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": [
        "## 5. Hyperparameter Tuning (Optuna)\n",
        "\n",
        "Tune LightGBM, XGBoost, and Random Forest using Optuna (30 trials each) on a subset of 50 products.\n",
        "This step takes approximately 5–10 minutes.\n"
    ]
})

cells.append({
    "cell_type": "code", "id": uid(), "metadata": {}, "execution_count": None,
    "outputs": [],
    "source": [
        "print('\\n' + '=' * 60)\n",
        "print('5. HYPERPARAMETER TUNING (Optuna)')\n",
        "print('=' * 60)\n",
        "\n",
        "np.random.seed(42)\n",
        "tune_products = np.random.choice(df['producto'].unique(), size=N_TUNE_PRODUCTS, replace=False)\n",
        "tune_fold = folds[0]\n",
        "\n",
        "tune_train_mask = df['fecha'] <= tune_fold['train_end']\n",
        "tune_test_mask = (df['fecha'] >= tune_fold['test_start']) & (df['fecha'] <= tune_fold['test_end'])\n",
        "\n",
        "tune_train = df[tune_train_mask & df['producto'].isin(tune_products)]\n",
        "tune_test = df[tune_test_mask & df['producto'].isin(tune_products)]\n",
        "\n",
        "X_tune_train = tune_train[feature_cols]\n",
        "y_tune_train = tune_train[TARGET]\n",
        "X_tune_test = tune_test[feature_cols]\n",
        "y_tune_test = tune_test[TARGET]\n",
        "\n",
        "print(f'Tuning on {N_TUNE_PRODUCTS} products, fold 1')\n",
        "print(f'Tune train: {len(tune_train):,} rows | Tune test: {len(tune_test):,} rows')\n",
        "\n",
        "# ── LightGBM tuning ──\n",
        "print('\\n── Tuning LightGBM ──')\n",
        "\n",
        "def lgb_objective(trial):\n",
        "    params = {\n",
        "        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),\n",
        "        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),\n",
        "        'num_leaves': trial.suggest_int('num_leaves', 15, 127),\n",
        "        'max_depth': trial.suggest_int('max_depth', 3, 12),\n",
        "        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),\n",
        "        'subsample': trial.suggest_float('subsample', 0.6, 1.0),\n",
        "        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),\n",
        "        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),\n",
        "        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),\n",
        "        'verbose': -1, 'n_jobs': -1, 'random_state': 42\n",
        "    }\n",
        "    model = lgb.LGBMRegressor(**params)\n",
        "    model.fit(X_tune_train, y_tune_train)\n",
        "    preds = model.predict(X_tune_test).clip(0)\n",
        "    return np.sqrt(mean_squared_error(y_tune_test, preds))\n",
        "\n",
        "lgb_study = optuna.create_study(direction='minimize')\n",
        "lgb_study.optimize(lgb_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)\n",
        "lgb_best = lgb_study.best_params\n",
        "lgb_best.update({'verbose': -1, 'n_jobs': -1, 'random_state': 42})\n",
        "print(f'Best RMSE: {lgb_study.best_value:.4f}')\n",
        "\n",
        "# ── XGBoost tuning ──\n",
        "print('\\n── Tuning XGBoost ──')\n",
        "\n",
        "def xgb_objective(trial):\n",
        "    params = {\n",
        "        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),\n",
        "        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),\n",
        "        'max_depth': trial.suggest_int('max_depth', 3, 10),\n",
        "        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),\n",
        "        'subsample': trial.suggest_float('subsample', 0.6, 1.0),\n",
        "        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),\n",
        "        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),\n",
        "        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),\n",
        "        'n_jobs': -1, 'random_state': 42, 'verbosity': 0\n",
        "    }\n",
        "    model = xgb.XGBRegressor(**params)\n",
        "    model.fit(X_tune_train, y_tune_train)\n",
        "    preds = model.predict(X_tune_test).clip(0)\n",
        "    return np.sqrt(mean_squared_error(y_tune_test, preds))\n",
        "\n",
        "xgb_study = optuna.create_study(direction='minimize')\n",
        "xgb_study.optimize(xgb_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)\n",
        "xgb_best = xgb_study.best_params\n",
        "xgb_best.update({'n_jobs': -1, 'random_state': 42, 'verbosity': 0})\n",
        "print(f'Best RMSE: {xgb_study.best_value:.4f}')\n",
        "\n",
        "# ── Random Forest tuning ──\n",
        "print('\\n── Tuning Random Forest ──')\n",
        "\n",
        "def rf_objective(trial):\n",
        "    params = {\n",
        "        'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),\n",
        "        'max_depth': trial.suggest_int('max_depth', 5, 20),\n",
        "        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 20),\n",
        "        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),\n",
        "        'max_features': trial.suggest_float('max_features', 0.3, 1.0),\n",
        "        'n_jobs': -1, 'random_state': 42\n",
        "    }\n",
        "    model = RandomForestRegressor(**params)\n",
        "    model.fit(X_tune_train, y_tune_train)\n",
        "    preds = model.predict(X_tune_test).clip(0)\n",
        "    return np.sqrt(mean_squared_error(y_tune_test, preds))\n",
        "\n",
        "rf_study = optuna.create_study(direction='minimize')\n",
        "rf_study.optimize(rf_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)\n",
        "rf_best = rf_study.best_params\n",
        "rf_best.update({'n_jobs': -1, 'random_state': 42})\n",
        "print(f'Best RMSE: {rf_study.best_value:.4f}')\n",
        "\n",
        "tuned_params = {'lgb': lgb_best, 'xgb': xgb_best, 'rf': rf_best}\n",
        "with open(MODEL_DIR / 'tuned_params.json', 'w') as f:\n",
        "    json.dump(tuned_params, f, indent=2, default=str)\n",
        "print(f'\\n\\u2713 Saved tuned_params.json')\n"
    ]
})

# ── 6. Train & Evaluate ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": [
        "## 6. Rolling Validation — All Models\n",
        "\n",
        "Train Random Forest, XGBoost, and LightGBM on each fold using the tuned hyperparameters.\n"
    ]
})

cells.append({
    "cell_type": "code", "id": uid(), "metadata": {}, "execution_count": None,
    "outputs": [],
    "source": [
        "print('\\n' + '=' * 60)\n",
        "print('6. ROLLING VALIDATION \\u2014 ALL MODELS')\n",
        "print('=' * 60)\n",
        "\n",
        "all_results = list(naive_results)\n",
        "all_predictions = []\n",
        "\n",
        "for fold in folds:\n",
        "    print(f\"\\n\\u2500\\u2500 Fold {fold['fold']} \\u2500\\u2500\")\n",
        "    train_mask = df['fecha'] <= fold['train_end']\n",
        "    test_mask = (df['fecha'] >= fold['test_start']) & (df['fecha'] <= fold['test_end'])\n",
        "    train = df[train_mask]\n",
        "    test = df[test_mask]\n",
        "    X_train = train[feature_cols]\n",
        "    y_train = train[TARGET]\n",
        "    X_test = test[feature_cols]\n",
        "    y_test = test[TARGET]\n",
        "    print(f'  Train: {len(train):,} | Test: {len(test):,}')\n",
        "\n",
        "    # Random Forest\n",
        "    t0 = time.time()\n",
        "    rf_model = RandomForestRegressor(**rf_best)\n",
        "    rf_model.fit(X_train, y_train)\n",
        "    rf_preds = rf_model.predict(X_test).clip(0)\n",
        "    rf_time = time.time() - t0\n",
        "    m = calc_metrics(y_test, rf_preds)\n",
        "    m.update({'fold': fold['fold'], 'model': 'RandomForest', 'time_s': rf_time})\n",
        "    all_results.append(m)\n",
        "    print(f\"  RF:   MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  MAPE={m['MAPE']:.1f}%  \\u03c3_e={m['sigma_e']:.3f}  ({rf_time:.0f}s)\")\n",
        "\n",
        "    # XGBoost\n",
        "    t0 = time.time()\n",
        "    xgb_model = xgb.XGBRegressor(**xgb_best)\n",
        "    xgb_model.fit(X_train, y_train)\n",
        "    xgb_preds = xgb_model.predict(X_test).clip(0)\n",
        "    xgb_time = time.time() - t0\n",
        "    m = calc_metrics(y_test, xgb_preds)\n",
        "    m.update({'fold': fold['fold'], 'model': 'XGBoost', 'time_s': xgb_time})\n",
        "    all_results.append(m)\n",
        "    print(f\"  XGB:  MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  MAPE={m['MAPE']:.1f}%  \\u03c3_e={m['sigma_e']:.3f}  ({xgb_time:.0f}s)\")\n",
        "\n",
        "    # LightGBM\n",
        "    t0 = time.time()\n",
        "    lgb_model = lgb.LGBMRegressor(**lgb_best)\n",
        "    lgb_model.fit(X_train, y_train)\n",
        "    lgb_preds = lgb_model.predict(X_test).clip(0)\n",
        "    lgb_time = time.time() - t0\n",
        "    m = calc_metrics(y_test, lgb_preds)\n",
        "    m.update({'fold': fold['fold'], 'model': 'LightGBM', 'time_s': lgb_time})\n",
        "    all_results.append(m)\n",
        "    print(f\"  LGBM: MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  MAPE={m['MAPE']:.1f}%  \\u03c3_e={m['sigma_e']:.3f}  ({lgb_time:.0f}s)\")\n",
        "\n",
        "    if fold['fold'] == N_FOLDS:\n",
        "        pred_df = test[['producto', 'fecha', TARGET, 'en_promo']].copy()\n",
        "        pred_df['Naive_lag7'] = test['lag_7'].values\n",
        "        pred_df['RandomForest'] = rf_preds\n",
        "        pred_df['XGBoost'] = xgb_preds\n",
        "        pred_df['LightGBM'] = lgb_preds\n",
        "        all_predictions.append(pred_df)\n"
    ]
})

# ── 7. Results Summary ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": [
        "## 7. Results Summary\n",
        "\n",
        "Average metrics across all folds and per-fold detail.\n"
    ]
})

cells.append({
    "cell_type": "code", "id": uid(), "metadata": {}, "execution_count": None,
    "outputs": [],
    "source": [
        "print('\\n' + '=' * 60)\n",
        "print('7. RESULTS SUMMARY')\n",
        "print('=' * 60)\n",
        "\n",
        "results_df = pd.DataFrame(all_results)\n",
        "summary = results_df.groupby('model')[['MAE', 'RMSE', 'MAPE', 'sigma_e']].mean()\n",
        "summary = summary.sort_values('RMSE')\n",
        "print('\\n\\u2500\\u2500 Average Metrics Across Folds \\u2500\\u2500')\n",
        "print(summary.round(4).to_string())\n",
        "\n",
        "print('\\n\\u2500\\u2500 Per-Fold Detail \\u2500\\u2500')\n",
        "pivot = results_df.pivot(index='fold', columns='model', values='RMSE')\n",
        "print(pivot.round(4).to_string())\n",
        "\n",
        "results_df.to_csv(MODEL_DIR / 'results_all_folds.csv', index=False)\n",
        "summary.to_csv(MODEL_DIR / 'results_summary.csv')\n",
        "print(f'\\n\\u2713 Saved results_all_folds.csv and results_summary.csv')\n",
        "\n",
        "if all_predictions:\n",
        "    pred_final = pd.concat(all_predictions, ignore_index=True)\n",
        "    pred_final.to_parquet(MODEL_DIR / 'predictions_last_fold.parquet', index=False)\n",
        "    print(f'\\u2713 Saved predictions_last_fold.parquet ({len(pred_final):,} rows)')\n"
    ]
})

# ── 8. Promo vs Non-Promo ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": [
        "## 8. Promo vs Non-Promo Performance\n"
    ]
})

cells.append({
    "cell_type": "code", "id": uid(), "metadata": {}, "execution_count": None,
    "outputs": [],
    "source": [
        "print('\\n' + '=' * 60)\n",
        "print('8. PROMO vs NON-PROMO PERFORMANCE')\n",
        "print('=' * 60)\n",
        "\n",
        "if all_predictions:\n",
        "    pf = pred_final.copy()\n",
        "    models = ['Naive_lag7', 'RandomForest', 'XGBoost', 'LightGBM']\n",
        "    for label, mask in [('Non-Promo', pf['en_promo'] == 0), ('Promo', pf['en_promo'] == 1)]:\n",
        "        subset = pf[mask]\n",
        "        if len(subset) == 0:\n",
        "            continue\n",
        "        print(f'\\n\\u2500\\u2500 {label} ({len(subset):,} rows) \\u2500\\u2500')\n",
        "        for model in models:\n",
        "            m = calc_metrics(subset[TARGET], subset[model])\n",
        "            print(f\"  {model:15s}: MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  MAPE={m['MAPE']:.1f}%  \\u03c3_e={m['sigma_e']:.3f}\")\n"
    ]
})

# ── 9. Plots ──
cells.append({
    "cell_type": "markdown", "id": uid(), "metadata": {},
    "source": [
        "## 9. Plots\n",
        "\n",
        "Model comparison bar chart, LightGBM feature importance, and actual vs predicted scatter.\n"
    ]
})

cells.append({
    "cell_type": "code", "id": uid(), "metadata": {}, "execution_count": None,
    "outputs": [],
    "source": [
        "print('\\n' + '=' * 60)\n",
        "print('9. GENERATING PLOTS')\n",
        "print('=' * 60)\n",
        "\n",
        "# 9.1 Model comparison\n",
        "fig, axes = plt.subplots(1, 4, figsize=(20, 5))\n",
        "for ax, metric in zip(axes, ['MAE', 'RMSE', 'MAPE', 'sigma_e']):\n",
        "    vals = summary[metric].sort_values()\n",
        "    colors = [ACCENT if v == vals.min() else PRIMARY for v in vals]\n",
        "    vals.plot(kind='barh', ax=ax, color=colors, edgecolor='white')\n",
        "    ax.set_title(metric)\n",
        "    ax.set_xlabel(metric)\n",
        "plt.suptitle('Model Comparison \\u2014 Average Across Folds', fontsize=14)\n",
        "save_fig('11_model_comparison')\n",
        "\n",
        "# 9.2 Feature importance\n",
        "importance = pd.Series(lgb_model.feature_importances_, index=feature_cols)\n",
        "importance = importance.sort_values(ascending=False).head(20)\n",
        "fig, ax = plt.subplots(figsize=(10, 8))\n",
        "importance.sort_values().plot(kind='barh', ax=ax, color=PRIMARY)\n",
        "ax.set_title('LightGBM Feature Importance (Top 20)')\n",
        "ax.set_xlabel('Importance (split count)')\n",
        "save_fig('12_lgbm_feature_importance')\n",
        "\n",
        "# 9.3 Actual vs Predicted\n",
        "if all_predictions:\n",
        "    sample = pred_final.sample(min(5000, len(pred_final)), random_state=42)\n",
        "    fig, ax = plt.subplots(figsize=(8, 8))\n",
        "    ax.scatter(sample[TARGET], sample['LightGBM'], alpha=0.2, s=5, color=get_model_color('LightGBM'))\n",
        "    lim = max(sample[TARGET].max(), sample['LightGBM'].max())\n",
        "    ax.plot([0, lim], [0, lim], color=ALERT, linestyle='--', linewidth=1, label='Perfect prediction')\n",
        "    ax.set_xlabel('Actual Sales')\n",
        "    ax.set_ylabel('Predicted Sales (LightGBM)')\n",
        "    ax.set_title('Actual vs Predicted \\u2014 LightGBM (Last Fold)')\n",
        "    ax.legend()\n",
        "    save_fig('13_actual_vs_predicted_lgbm')\n",
        "\n",
        "print('\\n' + '=' * 60)\n",
        "print('MODELLING COMPLETE')\n",
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

with open("TFM_Ona_Code/03_modelling.ipynb", "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✓ Built 03_modelling.ipynb")
