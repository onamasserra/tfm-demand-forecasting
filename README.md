# TFM — Demand Forecasting for Safety Stock Optimisation

Master's Thesis (TFM) for the MSc in Data Science at Universitat Oberta de Catalunya (UOC).

## Overview

This project applies machine learning models to forecast daily product demand in a retail/distribution context, and translates those forecasts into safety stock optimisation and holding-cost analysis using a service-level-based safety stock methodology (SS = z × RMSE × √(T + L)).

## Repository Structure

- `TFM_Ona_Code/` — Jupyter notebook pipeline (EDA → Preprocessing → Modelling → Evaluation → Economic Impact)
- `TFM_DS_Ona_Mas_i_Serra/` — LaTeX thesis source files
- `Datos/` — Raw input data (Excel files)

## Setup

```bash
git clone https://github.com/onamasserra/tfm-demand-forecasting.git
cd tfm-demand-forecasting
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open `TFM_Ona_Code/01_eda.ipynb` in VS Code or Jupyter and run notebooks in order (01 → 05).

## Pipeline

| Notebook | Description |
|----------|-------------|
| `01_eda.ipynb` | Exploratory data analysis |
| `02_preprocessing.ipynb` | Data cleaning and feature engineering |
| `03_modelling.ipynb` | Tree-based models (RF, XGBoost, LightGBM) with Optuna |
| `03b_modelling_arima_lstm.ipynb` | ARIMA and LSTM baselines |
| `04_evaluation.ipynb` | Model evaluation and SHAP interpretability |
| `05_economic_impact.ipynb` | Safety stock cost analysis and service-level optimisation |
