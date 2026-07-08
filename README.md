# Spatio-Temporal Ride Demand Prediction in NYC


Predicting hourly taxi ride demand across 261 NYC pickup zones using January 2025 Yellow Taxi trip records. Fourteen supervised ML models (linear, tree-based, gradient boosting, instance-based, SVR) are benchmarked against a historical-average baseline.

## Results

| Model | MAE | RMSE | R² | Time (s) |
|---|---|---|---|---|
| **XGBoost (Tuned)**  | **3.0949** | 10.0165 | 0.9624 | 115.2 |
| Stacking (XGB+Ridge+KNN) | 3.1116 | 10.0151 | 0.9624 | 160.7 |
| XGBoost (early stop) | 3.1368 | 10.1694 | 0.9613 | 3.3 |
| LightGBM (early stop) | 3.1657 | 10.2880 | 0.9604 | 6.9 |
| Random Forest | 3.2549 | 10.6323 | 0.9577 | 49.2 |
| Historical Baseline | 4.6018 | 13.5004 | 0.9318 | — |

Tuned XGBoost delivers a **32.7% MAE improvement** over the historical baseline. LightGBM reaches near-identical accuracy in ~17× less training time, making it the strongest candidate for real-world deployment. Full comparison of all 14 models is in the report.

**Key drivers of demand:** `lag_1` (short-term momentum, importance 0.667) and `lag_168` (weekly seasonality, importance 0.189) account for over 91% of feature importance in the tuned XGBoost model.

## Repository structure

```
nyc-taxi-demand-prediction/
├── notebooks/
│   ├── ML_ready_dataset_notebook.ipynb        # Data cleaning, feature engineering, VIF/PCA
│   └── Spatio_Temporal_Ride_Demand_Prediction.ipynb  # Modeling, tuning, evaluation
├── report/
│   └── Report_group4.pdf                      # Full write-up (IEEE format)
├── requirements.txt
└── README.md
```

## Pipeline

1. **`ML_ready_dataset_notebook.ipynb`**
   - Loads raw NYC TLC Yellow Taxi parquet + zone lookup CSV
   - Aggregates trips to hourly zone-level demand counts
   - Engineers 18 features: cyclical hour/day encodings, weekend/holiday flags, per-zone lag features (`lag_1`, `lag_2`, `lag_24`, `lag_168`), rolling mean, borough one-hot encoding
   - Runs a VIF analysis (multicollinearity check) and applies PCA for linear-model inputs
   - Outputs the ML-ready dataset used by the modeling notebook

2. **`Spatio_Temporal_Ride_Demand_Prediction.ipynb`**
   - Chronological 80/20 train/test split (no shuffling, to avoid leakage)
   - Trains 14 models across 4 tiers: linear (OLS/Ridge/Lasso/ElasticNet), distance/margin-based (KNN, Linear SVR), tree-based (Decision Tree, Random Forest), gradient boosting (LightGBM, XGBoost, Gradient Boosting, tuned XGBoost) + a stacking ensemble
   - Hyperparameter tuning via `RandomizedSearchCV` with `TimeSeriesSplit`
   - 5-fold `TimeSeriesSplit` cross-validation on top-3 models with bootstrap 95% CIs
   - Feature importance, error analysis by hour/zone, residual diagnostics

## Data

[NYC TLC Yellow Taxi Trip Records — January 2025](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) (not included in this repo due to size — download separately and place under `data/raw/`).

## Setup

```bash
git clone https://github.com/<your-username>/nyc-taxi-demand-prediction.git
cd nyc-taxi-demand-prediction
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Run `notebooks/ML_ready_dataset_notebook.ipynb` first to produce the ML-ready dataset, then `notebooks/Spatio_Temporal_Ride_Demand_Prediction.ipynb` for modeling and evaluation.

## Authors

Group 4 — CSAI-801, Artificial Intelligence & Machine Learning

## License

MIT (or update as preferred)
