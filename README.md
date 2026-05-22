# TFM — Demand Forecasting for Inventory Management Using Machine Learning

Master's Final Project for the MSc in Data Science at Universitat Oberta de Catalunya (UOC).

**Author:** Ona Mas i Serra  
**Date:** May 2026

## Overview

This project develops machine learning models for daily demand forecasting in the automotive sector and evaluates their impact on safety stock costs. The models achieve a 28% reduction in forecast error (RMSE) compared to a naive baseline, translating into annual safety stock cost savings of approximately €3.0 million (27%).

## Repository Contents

This public repository contains the Jupyter notebooks (source code) and their HTML exports for viewing without running the code:

| Notebook | Description |
|----------|-------------|
| `01_eda.ipynb` | Exploratory data analysis |
| `02_preprocessing.ipynb` | Data cleaning and feature engineering |
| `03_modelling.ipynb` | Tree-based models (RF, XGBoost, LightGBM) with Optuna tuning |
| `03b_modelling_arima_lstm.ipynb` | ARIMA and LSTM models |
| `04_evaluation.ipynb` | Model evaluation, cross-model SHAP analysis |
| `05_economic_impact.ipynb` | Safety stock cost analysis and service-level optimisation |

Each notebook is available in two formats:
- `.ipynb` — Interactive Jupyter notebook (requires Python environment to run)
- `.html` — Static rendered version (viewable in any browser)

## Key Results

- **Best models:** Random Forest, XGBoost, LightGBM, and LSTM achieve nearly identical performance (RMSE 1.634–1.643)
- **Improvement over baseline:** ~28% RMSE reduction vs naive lag-7
- **Economic impact:** ~€3.0M annual safety stock cost savings (26–27%)
- **Most important feature:** 28-day EWMA (consistent across all model architectures)

## Note

The raw data files, LaTeX thesis source, and auxiliary files are not included in this public repository.
