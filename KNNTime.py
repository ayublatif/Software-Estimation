import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import pairwise_distances
from scipy.stats import wilcoxon

# Load dataset
df = pd.read_csv('c:\\ietcode\\augmented_project_data1.csv')

# Features and target
feature_cols = ['Effort', 'TeamSalary', 'EstimatedCost']
X_all = df[feature_cols].values
y_all = df['ActualTime'].values
original_estimates = df['EstimatedTime'].values

# Initialize result storage
lr_preds, svr_preds, rf_preds, xgb_preds = [], [], [], []
actuals, original_preds = [], []

lr_mres, svr_mres, rf_mres, xgb_mres, original_mres = [], [], [], [], []

# Iterate over each project (like Leave-One-Out)
for i in range(len(df)):
    # Test sample
    X_test = X_all[i].reshape(1, -1)
    y_test = y_all[i]
    original_pred = original_estimates[i]

    # Exclude test project
    X_train_pool = np.delete(X_all, i, axis=0)
    y_train_pool = np.delete(y_all, i, axis=0)

    # Compute distances from test project to others
    distances = pairwise_distances(X_test, X_train_pool, metric='euclidean').flatten()

    # Get indices of 7 nearest neighbors
    neighbor_indices = np.argsort(distances)[:7]

    # Select neighbors
    X_train = X_train_pool[neighbor_indices]
    y_train = y_train_pool[neighbor_indices]

    # Train models on 7 neighbors
    lr = LinearRegression().fit(X_train, y_train)
    svr = SVR().fit(X_train, y_train)
    rf = RandomForestRegressor().fit(X_train, y_train)
    xgb = XGBRegressor(verbosity=0).fit(X_train, y_train)

    # Predict
    lr_pred = lr.predict(X_test)[0]
    svr_pred = svr.predict(X_test)[0]
    rf_pred = rf.predict(X_test)[0]
    xgb_pred = xgb.predict(X_test)[0]

    # Save predictions
    actuals.append(y_test)
    original_preds.append(original_pred)
    lr_preds.append(lr_pred)
    svr_preds.append(svr_pred)
    rf_preds.append(rf_pred)
    xgb_preds.append(xgb_pred)

    # Calculate MREs
    original_mres.append(abs(original_pred - y_test) / y_test)
    lr_mres.append(abs(lr_pred - y_test) / y_test)
    svr_mres.append(abs(svr_pred - y_test) / y_test)
    rf_mres.append(abs(rf_pred - y_test) / y_test)
    xgb_mres.append(abs(xgb_pred - y_test) / y_test)

# Build results DataFrame
results_df = pd.DataFrame({
    'EstimatedTime': original_preds,
    'ActualTime': actuals,
    'LR_PredictedTime': lr_preds,
    'SVR_PredictedTime': svr_preds,
    'RF_PredictedTime': rf_preds,
    'XGB_PredictedTime': xgb_preds,
    'Original_MRE': original_mres,
    'LR_MRE': lr_mres,
    'SVR_MRE': svr_mres,
    'RF_MRE': rf_mres,
    'XGB_MRE': xgb_mres
})

# MMRE row
mmre_row = {
    'EstimatedTime': '',
    'ActualTime': 'MMRE',
    'LR_PredictedTime': '',
    'SVR_PredictedTime': '',
    'RF_PredictedTime': '',
    'XGB_PredictedTime': '',
    'Original_MRE': np.mean(original_mres),
    'LR_MRE': np.mean(lr_mres),
    'SVR_MRE': np.mean(svr_mres),
    'RF_MRE': np.mean(rf_mres),
    'XGB_MRE': np.mean(xgb_mres)
}

# Pred(25) row
pred25_row = {
    'EstimatedTime': '',
    'ActualTime': 'Pred(25)',
    'LR_PredictedTime': '',
    'SVR_PredictedTime': '',
    'RF_PredictedTime': '',
    'XGB_PredictedTime': '',
    'Original_MRE': sum(np.array(original_mres) <= 0.25),
    'LR_MRE': sum(np.array(lr_mres) <= 0.25),
    'SVR_MRE': sum(np.array(svr_mres) <= 0.25),
    'RF_MRE': sum(np.array(rf_mres) <= 0.25),
    'XGB_MRE': sum(np.array(xgb_mres) <= 0.25)
}

# Append rows
results_df.loc[len(results_df)] = mmre_row
results_df.loc[len(results_df)] = pred25_row

# Set index to start from 1
results_df.index = np.arange(1, len(results_df) + 1)

# Save to CSV
results_df.to_csv('c:\\ietcode\\model_predictions_with_knn7.csv', index_label='Index')
print("✅ Model predictions saved to: c:\\ietcode\\model_predictions_with_knn7.csv")

# ---------------------
# Statistical Testing
# ---------------------
stat_results = []

# Original vs Actual (Wilcoxon)
try:
    stat, p = wilcoxon(original_preds, actuals)
    interpretation = '✅ Significant' if p < 0.05 else '❌ Not Significant'
    print(f"Original vs Actual - p-value: {p:.5f} -> {interpretation}")
    stat_results.append({'Model': 'Original', 'p-value': p, 'Interpretation': interpretation})
except Exception as e:
    print(f"Original vs Actual Wilcoxon test failed: {e}")
    stat_results.append({'Model': 'Original', 'p-value': np.nan, 'Interpretation': '❌ Test Failed'})

# Models vs Original (Wilcoxon on MREs)
models = {
    'LR': lr_mres,
    'SVR': svr_mres,
    'RF': rf_mres,
    'XGBoost': xgb_mres
}

for model_name, model_mre in models.items():
    try:
        stat, p = wilcoxon(model_mre, original_mres)
        interpretation = '✅ Significant' if p < 0.05 else '❌ Not Significant'
        print(f"{model_name} vs Original - p-value: {p:.5f} -> {interpretation}")
        stat_results.append({'Model': model_name, 'p-value': p, 'Interpretation': interpretation})
    except Exception as e:
        print(f"{model_name} Wilcoxon test failed: {e}")
        stat_results.append({'Model': model_name, 'p-value': np.nan, 'Interpretation': '❌ Test Failed'})

# Save Wilcoxon test results
stat_df = pd.DataFrame(stat_results)
stat_df.to_csv('c:\\ietcode\\wilcoxon_test_results_knn7.csv', index=False)
print("📊 Wilcoxon test results saved to: c:\\ietcode\\wilcoxon_test_results_knn7.csv")
