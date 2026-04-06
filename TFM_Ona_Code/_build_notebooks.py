"""
Build .ipynb notebooks from .py scripts with aligned markdown commentary.
Run once to generate notebooks; then execute them in Jupyter to capture outputs.
"""
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent

def make_nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.14.0"}
        },
        "cells": cells
    }

def md(lines):
    """Create a markdown cell. lines can be a string or list of strings."""
    if isinstance(lines, str):
        lines = lines.rstrip("\n").split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}

def code(lines):
    """Create a code cell. lines can be a string or list of strings."""
    if isinstance(lines, str):
        lines = lines.rstrip("\n").split("\n")
    return {"cell_type": "code", "metadata": {}, "source": [l + "\n" for l in lines],
            "outputs": [], "execution_count": None}

def save_nb(nb, path):
    with open(path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  ✓ {path.name}")

def read_py(name):
    src = (BASE / name).read_text()
    # Fix for notebook compatibility: replace __file__ paths
    src = src.replace('Path(__file__).resolve().parent.parent', 'Path("..")')
    src = src.replace('Path(__file__).resolve().parent', 'Path(".")')
    # Remove matplotlib Agg backend (notebooks render inline)
    src = src.replace("matplotlib.use('Agg')\n", "")
    src = src.replace('matplotlib.use(\'Agg\')\n', '')
    return src


# ══════════════════════════════════════════════════════════════════
# Helper: split a .py file on section comment blocks
# ══════════════════════════════════════════════════════════════════
def split_sections(src):
    """Split source on section markers (print('=' * 60) blocks)."""
    lines = src.split("\n")
    sections = []
    current_header = None
    current_code = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        # Detect section: print("=" * 60) followed by print("N. TITLE") followed by print("=" * 60)
        if 'print("=" * 60)' in line or "print('=' * 60)" in line:
            # Save previous section
            if current_code:
                sections.append((current_header, "\n".join(current_code).strip()))
                current_code = []
            # Read the title from the next print statement
            i += 1
            title = None
            while i < len(lines):
                if 'print("=' in lines[i] or "print('=" in lines[i]:
                    i += 1
                    break
                m = re.search(r'print\(["\'](.+?)["\']\)', lines[i])
                if m:
                    title = m.group(1)
                i += 1
            current_header = title
            continue
        current_code.append(line)
        i += 1
    # Last section
    if current_code:
        sections.append((current_header, "\n".join(current_code).strip()))
    return sections


# ══════════════════════════════════════════════════════════════════
# NOTEBOOK 1: EDA
# ══════════════════════════════════════════════════════════════════
print("Building 01_eda.ipynb...")

eda_src = read_py("01_eda.py")
# Extract the docstring and imports (everything before first section)
parts = split_sections(eda_src)

cells_eda = [
    md([
        "# 01 — Exploratory Data Analysis",
        "",
        "**TFM: Demand Forecasting for Inventory Management Using Machine Learning**",
        "",
        "Author: Ona Mas i Serra",
        "",
        "This notebook performs the initial exploration of the dataset provided by the thesis supervisor. "
        "The dataset corresponds to real historical data from a company operating in the automotive sector, "
        "containing daily demand records for 861 products over 731 days (March 2024 to March 2026), "
        "along with calendar events, promotional campaigns, and stock levels.",
        "",
        "All product references are anonymised using numerical identifiers, ensuring compliance with data privacy requirements."
    ]),
]

# Add each section as markdown header + code
for header, code_block in parts:
    if not code_block.strip():
        continue
    # Skip standalone module docstring blocks (no actual code)
    stripped = code_block.strip()
    if stripped.startswith('"""') and stripped.endswith('"""') and stripped.count('"""') == 2:
        continue
    if header:
        cells_eda.append(md(f"## {header}"))
    # Add contextual markdown for key sections
    if header and "LOADING DATA" in header.upper():
        cells_eda.append(md([
            "The dataset is loaded from `Datos.xlsx`, which contains four sheets: **Ventas** (sales), "
            "**Calendario** (calendar), **Promociones** (promotions), and **Stock** (stock levels). "
            "All four sheets are merged into a single dataframe on the shared keys `(producto, fecha)`.",
            "",
            "Promotional periods are transformed from date ranges into a daily binary indicator (`en_promo`)."
        ]))
    elif header and "DESCRIPTIVE" in header.upper():
        cells_eda.append(md([
            "The target variable (`udsVenta`) exhibits a highly skewed distribution with 65.3% zero-sales observations. "
            "This is characteristic of automotive parts demand, where most product-day combinations register zero sales."
        ]))
    elif header and "PLOT" in header.upper() or (header and "VISUAL" in header.upper()):
        cells_eda.append(md([
            "The following visualisations examine demand patterns across multiple dimensions: "
            "daily distribution, day-of-week cycles, monthly seasonality, promotional effects, "
            "and the top-selling products."
        ]))
    elif header and "AUTOCORRELATION" in header.upper():
        cells_eda.append(md([
            "Autocorrelation analysis confirms significant positive autocorrelation at lags 1, 7, 14, and 28, "
            "with the strongest signal at lag 7 (weekly periodicity). This validates the choice of lag-7 as the naive baseline "
            "and supports the creation of lag features at these intervals."
        ]))
    elif header and "SEASONALITY" in header.upper():
        cells_eda.append(md([
            "Weekly and monthly patterns are examined to identify seasonal demand behaviour "
            "that can be captured through calendar-based features."
        ]))
    elif header and "QUALITY" in header.upper():
        cells_eda.append(md([
            "The data quality assessment identifies stock breaks (2.0% of observations), "
            "absence of negative values, and missing calendar entries for one day. "
            "These issues are addressed in the preprocessing step."
        ]))
    cells_eda.append(code(code_block))

nb_eda = make_nb(cells_eda)
save_nb(nb_eda, BASE / "01_eda.ipynb")


# ══════════════════════════════════════════════════════════════════
# NOTEBOOK 2: PREPROCESSING
# ══════════════════════════════════════════════════════════════════
print("Building 02_preprocessing.ipynb...")

pre_src = read_py("02_preprocessing.py")
parts = split_sections(pre_src)

cells_pre = [
    md([
        "# 02 — Data Preprocessing & Feature Engineering",
        "",
        "**TFM: Demand Forecasting for Inventory Management Using Machine Learning**",
        "",
        "Author: Ona Mas i Serra",
        "",
        "This notebook covers data cleaning, transformation, and feature engineering. "
        "The cleaning process addresses stock break imputation and outlier capping. "
        "Feature engineering produces 36 features capturing temporal dependencies, "
        "demand dynamics, and promotional effects.",
    ]),
]

for header, code_block in parts:
    if not code_block.strip():
        continue
    stripped = code_block.strip()
    if stripped.startswith('"""') and stripped.endswith('"""') and stripped.count('"""') == 2:
        continue
    if header:
        cells_pre.append(md(f"## {header}"))
    if header and "CLEANING" in header.upper():
        cells_pre.append(md([
            "Two data quality issues are addressed:",
            "",
            "1. **Stock break imputation:** When both sales and stock are zero, the sales value is replaced "
            "with the rolling mean for that product on the same weekday (4-week window). This assumes zero sales "
            "with zero stock reflects a supply-side constraint rather than genuine zero demand.",
            "",
            "2. **Outlier capping:** Values exceeding 3σ above the mean for each product-month combination "
            "are capped at the threshold. A total of 8,895 values (1.41%) were capped."
        ]))
    elif header and "FEATURE ENGINEERING" in header.upper():
        cells_pre.append(md([
            "A comprehensive set of 36 features is engineered across five categories:",
            "",
            "- **Calendar features** (8): year, month, day_of_week, week_of_year, etc.",
            "- **Lag features** (4): lag_1, lag_7, lag_14, lag_28",
            "- **Rolling statistics** (14): mean/std/min/max at 7, 14, 28-day windows + EWMA",
            "- **Promotional features** (3): days_since_promo_end, post_promo_7d, promo×lag7",
            "- **Product-level statistics** (3): mean, std, median sales per product",
            "",
            "All lag and rolling features use strictly past data to prevent information leakage."
        ]))
    elif header and "NaN" in header.upper():
        cells_pre.append(md([
            "The first 28 days of each product's history are dropped (warmup period for lag_28), "
            "reducing the dataset from 629,391 to 605,283 rows."
        ]))
    elif header and "SUMMARY" in header.upper():
        cells_pre.append(md([
            "The feature correlation analysis reveals that the 28-day EWMA and rolling mean "
            "are the most strongly correlated with the target variable, confirming the importance "
            "of smoothed demand history for prediction."
        ]))
    cells_pre.append(code(code_block))

nb_pre = make_nb(cells_pre)
save_nb(nb_pre, BASE / "02_preprocessing.ipynb")


# ══════════════════════════════════════════════════════════════════
# NOTEBOOK 3: MODELLING
# ══════════════════════════════════════════════════════════════════
print("Building 03_modelling.ipynb...")

mod_src = read_py("03_modelling.py")
parts = split_sections(mod_src)

cells_mod = [
    md([
        "# 03 — Model Training & Evaluation",
        "",
        "**TFM: Demand Forecasting for Inventory Management Using Machine Learning**",
        "",
        "Author: Ona Mas i Serra",
        "",
        "This notebook trains and evaluates four forecasting models using a rolling forecast origin "
        "validation strategy with three 30-day test folds:",
        "",
        "- **Naive baseline (lag-7):** same weekday last week",
        "- **Random Forest:** ensemble of decision trees with bagging",
        "- **XGBoost:** gradient boosting with L1/L2 regularisation",
        "- **LightGBM:** histogram-based gradient boosting with leaf-wise growth",
        "",
        "Hyperparameters are optimised using Optuna (30 trials per model) on a subset of 50 products."
    ]),
]

for header, code_block in parts:
    if not code_block.strip():
        continue
    stripped = code_block.strip()
    if stripped.startswith('"""') and stripped.endswith('"""') and stripped.count('"""') == 2:
        continue
    if header:
        cells_mod.append(md(f"## {header}"))
    if header and "ROLLING" in header.upper():
        cells_mod.append(md([
            "Three validation folds are defined, each with a 30-day test window:",
            "",
            "| Fold | Training end | Test period | Days |",
            "|------|-------------|-------------|------|",
            "| 1 | 2025-12-22 | 2025-12-23 → 2026-01-21 | 30 |",
            "| 2 | 2026-01-21 | 2026-01-22 → 2026-02-20 | 30 |",
            "| 3 | 2026-02-20 | 2026-02-21 → 2026-03-22 | 30 |",
        ]))
    elif header and "HYPERPARAMETER" in header.upper():
        cells_mod.append(md([
            "Optuna's Tree-structured Parzen Estimator (TPE) is used to optimise hyperparameters. "
            "Tuning is performed on 50 sampled products using fold 1, with RMSE as the objective."
        ]))
    elif header and "RESULTS" in header.upper():
        cells_mod.append(md([
            "The three tree-based models achieve very similar performance, with XGBoost holding a marginal edge. "
            "All ML models substantially outperform the naive baseline (24–28% RMSE reduction)."
        ]))
    elif header and "PROMO" in header.upper():
        cells_mod.append(md([
            "Performance is compared across promotional and non-promotional demand segments "
            "to assess whether the models maintain their advantage under different demand conditions."
        ]))
    cells_mod.append(code(code_block))

nb_mod = make_nb(cells_mod)
save_nb(nb_mod, BASE / "03_modelling.ipynb")


# ══════════════════════════════════════════════════════════════════
# NOTEBOOK 3b: ARIMA + LSTM
# ══════════════════════════════════════════════════════════════════
print("Building 03b_modelling_arima_lstm.ipynb...")

arima_src = read_py("03b_modelling_arima_lstm.py")
parts = split_sections(arima_src)

cells_arima = [
    md([
        "# 03b — ARIMA & LSTM Models",
        "",
        "**TFM: Demand Forecasting for Inventory Management Using Machine Learning**",
        "",
        "Author: Ona Mas i Serra",
        "",
        "This notebook adds two additional models to the comparison:",
        "",
        "- **ARIMA (Seasonal):** Classical time series approach fitted per product using `auto_arima` "
        "with weekly periodicity (m=7). Fitted to all 861 products.",
        "- **LSTM:** Recurrent neural network (PyTorch) with a single LSTM layer (32 hidden units), "
        "trained on 14-day input sequences of standardised features.",
        "",
        "These models complement the tree-based approaches from notebook 03, providing a comprehensive "
        "comparison across classical, machine learning, and deep learning paradigms."
    ]),
]

for header, code_block in parts:
    if not code_block.strip():
        continue
    stripped = code_block.strip()
    if stripped.startswith('"""') and stripped.endswith('"""') and stripped.count('"""') == 2:
        continue
    if header:
        cells_arima.append(md(f"## {header}"))
    if header and "ARIMA" in header.upper():
        cells_arima.append(md([
            "ARIMA is fitted per product on the raw sales series without engineered features. "
            "Each product's demand is modelled with seasonal ARIMA (SARIMA) with weekly periodicity, "
            "automatically selecting optimal orders via stepwise search. "
            "Running on all 861 products takes approximately 2.5 hours."
        ]))
    elif header and "LSTM" in header.upper():
        cells_arima.append(md([
            "The LSTM model processes the same 33 features as the tree models, but as ordered sequences "
            "rather than independent observations. The internal gating mechanism allows it to selectively "
            "retain or discard information from previous time steps.",
            "",
            "Architecture: 1 LSTM layer (32 hidden units) → Linear head → clipped to ≥0"
        ]))
    elif header and "COMBINED" in header.upper():
        cells_arima.append(md([
            "All six models are compared on the same test period. The results show that tree-based models "
            "form a tight cluster at the top, with LSTM and ARIMA close behind, all substantially "
            "outperforming the naive baseline."
        ]))
    cells_arima.append(code(code_block))

nb_arima = make_nb(cells_arima)
save_nb(nb_arima, BASE / "03b_modelling_arima_lstm.ipynb")


# ══════════════════════════════════════════════════════════════════
# NOTEBOOK 4: EVALUATION
# ══════════════════════════════════════════════════════════════════
print("Building 04_evaluation.ipynb...")

eval_src = read_py("04_evaluation.py")
parts = split_sections(eval_src)

cells_eval = [
    md([
        "# 04 — Model Evaluation & SHAP Analysis",
        "",
        "**TFM: Demand Forecasting for Inventory Management Using Machine Learning**",
        "",
        "Author: Ona Mas i Serra",
        "",
        "This notebook provides a detailed evaluation of the trained models, including:",
        "",
        "- **SHAP feature importance:** Understanding which features drive predictions",
        "- **Error analysis:** Distribution of forecast errors, per-product RMSE, day-of-week patterns",
        "- **Promotional vs non-promotional performance:** Segment-level comparison",
        "",
        "The SHAP analysis reveals that the 28-day EWMA is the single most important predictor, "
        "confirming that smoothed demand history drives forecast quality."
    ]),
]

for header, code_block in parts:
    if not code_block.strip():
        continue
    stripped = code_block.strip()
    if stripped.startswith('"""') and stripped.endswith('"""') and stripped.count('"""') == 2:
        continue
    if header:
        cells_eval.append(md(f"## {header}"))
    if header and "SHAP" in header.upper():
        cells_eval.append(md([
            "SHAP (SHapley Additive exPlanations) values are computed for the LightGBM model "
            "to understand the contribution of each feature to individual predictions.",
            "",
            "The top features by mean |SHAP| value are:",
            "1. `ewma_28` (0.473) — 28-day exponentially weighted moving average",
            "2. `day_of_week` (0.260) — weekly cycle",
            "3. `roll_mean_28` (0.108) — 28-day rolling mean"
        ]))
    elif header and "ERROR" in header.upper():
        cells_eval.append(md([
            "Error analysis examines the distribution of forecast errors across models, "
            "per-product RMSE variation, and error patterns by day of week. "
            "The ML models produce tighter, more symmetric error distributions compared to the naive baseline."
        ]))
    cells_eval.append(code(code_block))

nb_eval = make_nb(cells_eval)
save_nb(nb_eval, BASE / "04_evaluation.ipynb")


# ══════════════════════════════════════════════════════════════════
# NOTEBOOK 5: ECONOMIC IMPACT
# ══════════════════════════════════════════════════════════════════
print("Building 05_economic_impact.ipynb...")

econ_src = read_py("05_economic_impact.py")
parts = split_sections(econ_src)

cells_econ = [
    md([
        "# 05 — Economic Impact Analysis",
        "",
        "**TFM: Demand Forecasting for Inventory Management Using Machine Learning**",
        "",
        "Author: Ona Mas i Serra",
        "",
        "This notebook translates forecast accuracy improvements into monetary terms using the "
        "methodology provided by the thesis supervisor (EstimacionCostesStock.docx).",
        "",
        "**Methodology:**",
        "- Safety stock: SS_i = z × RMSE_i × √(T_i + L_i)",
        "- Daily cost: 5% × price_i × SS_i",
        "- Annual cost: 365 × daily cost",
        "",
        "**Data sources:**",
        "- `DatosCicloAprovisionamiento.xlsx`: replenishment cycle and lead time per product",
        "- `DatosPrecioMedio.xlsx`: average unit price per product",
        "",
        "The analysis compares two scenarios: Naive forecast vs best ML model (XGBoost), "
        "demonstrating annual savings of approximately €3.1 million (27.1%)."
    ]),
]

for header, code_block in parts:
    if not code_block.strip():
        continue
    stripped = code_block.strip()
    if stripped.startswith('"""') and stripped.endswith('"""') and stripped.count('"""') == 2:
        continue
    if header:
        cells_econ.append(md(f"## {header}"))
    if header and "LOAD" in header.upper():
        cells_econ.append(md([
            "Predictions from the last validation fold are loaded alongside the per-product "
            "cost parameters (replenishment cycle, lead time, and average price) provided by the company."
        ]))
    elif header and "RMSE" in header.upper():
        cells_econ.append(md([
            "Per-product RMSE is computed for each model. This is the key input to the safety stock formula, "
            "as it captures the forecast uncertainty for each individual product."
        ]))
    elif header and "SAFETY STOCK" in header.upper():
        cells_econ.append(md([
            "Safety stock is computed using the formula from Yamazaki et al. (2016):",
            "",
            "**SS = z × RMSE × √(diasEntrePedidos + diasLeadtime)**",
            "",
            "where z = 1.6449 for a 95% service level."
        ]))
    elif header and "COST" in header.upper() and "SENSITIVITY" not in header.upper():
        cells_econ.append(md([
            "The daily holding cost for each product's safety stock is computed as:",
            "",
            "**DailyCost = 5% × price × SS**",
            "",
            "This rate represents the combined cost of warehousing, insurance, and capital opportunity cost."
        ]))
    elif header and "SENSITIVITY" in header.upper():
        cells_econ.append(md([
            "The sensitivity analysis examines how safety stock requirements and costs change "
            "across different service levels (90%, 95%, 99%), confirming that the ML advantage "
            "is robust regardless of the risk tolerance adopted."
        ]))
    elif header and "SUMMARY" in header.upper():
        cells_econ.append(md([
            "The final summary shows that XGBoost saves approximately **€3.1 million annually** (27.1%) "
            "compared to the naive baseline, with all three ML models performing nearly identically."
        ]))
    cells_econ.append(code(code_block))

nb_econ = make_nb(cells_econ)
save_nb(nb_econ, BASE / "05_economic_impact.ipynb")


print("\n✅ All notebooks built successfully!")
