# TFM — Demand Forecasting for Safety Stock Optimisation

Master's Thesis (TFM) for the MSc in Data Science at Universitat Oberta de Catalunya (UOC).

## Overview

This project applies machine learning models to forecast daily product demand in a retail/distribution context, and translates those forecasts into safety stock optimisation and holding-cost analysis using a service-level-based safety stock methodology (SS = z × RMSE × √(T + L)).

## Repository Structure

- `TFM_Ona_Code/` — Python analysis pipeline (EDA → Preprocessing → Modelling → Evaluation → Economic Impact)
- `TFM_DS_Ona_Mas_i_Serra/` — LaTeX thesis source files

## Pipeline

| Script | Description |
|--------|-------------|
| `01_eda.py` | Exploratory data analysis |
| `02_preprocessing.py` | Data cleaning and feature engineering |
| `03_modelling.py` | Tree-based models (RF, XGBoost, LightGBM) with Optuna |
| `03b_modelling_arima_lstm.py` | ARIMA and LSTM baselines |
| `04_evaluation.py` | Model evaluation and SHAP interpretability |
| `05_economic_impact.py` | Safety stock cost analysis and service-level optimisation |
