import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import wilcoxon

# Load dataset
df = pd.read_csv('enhanced_effort_dataset_subset.csv')

# Define features and target
X_cols = [
    'Size', 'Complexity', 'Noftasks',
    'developmenttype', 'relatedtechnologies', 'externalhardware',
    'TasksPerComplexity', 'SizeToTasksRatio'
]
y_col = 'effort'

X = df[X_cols].copy()
y = df[y_col].copy()

# Identify categorical and numeric columns
categorical_cols = ['developmenttype', 'relatedtechnologies', 'externalhardware']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipeline (✅ Fixed the encoder)
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
])

# ML Models
models = {
    'LR': LinearRegression(),
    'SVR': SVR(),
    'RF': RandomForestRegressor(random_state=42),
    'XGB': XGBRegressor(random_state=42)
}

# KNN configuration
K = 7
loo = LeaveOneOut()

# Store results
results = []
model_predictions = {name: [] for name in models}

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Scale features for KNN distance calculation
    scaled = StandardScaler().fit(X_train[numeric_cols])
    X_train_scaled = scaled.transform(X_train[numeric_cols])
    X_test_scaled = scaled.transform(X_test[numeric_cols])

    # Find KNN using Euclidean distance
    distances = np.linalg.norm(X_train_scaled - X_test_scaled[0], axis=1)
    nearest_indices = distances.argsort()[:K]

    X_knn = X_train.iloc[nearest_indices]
    y_knn = y_train.iloc[nearest_indices]

    row_result = {'Project': test_idx[0] + 1, 'ActualEffort': y_test.values[0]}

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocess', preprocessor),
            ('regressor', model)
        ])
        pipeline.fit(X_knn, y_knn)
        y_pred = pipeline.predict(X_test)[0]
        mre = abs(y_pred - y_test.values[0]) / y_test.values[0]
        row_result[f'{name}_Prediction'] = y_pred
        row_result[f'{name}_MRE'] = mre
        model_predictions[name].append(y_pred)

    results.append(row_result)

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Calculate MMRE and Pred(25)
metrics = []
for name in models:
    mmre = results_df[f'{name}_MRE'].mean()
    pred_25 = (results_df[f'{name}_MRE'] <= 0.25).mean() * 100
    metrics.append((name, mmre, pred_25))

metrics_df = pd.DataFrame(metrics, columns=['Model', 'MMRE', 'Pred(25)'])
metrics_df.to_csv('effort_prediction_results.csv', index=False)

# Save detailed per-project predictions as well
results_df.to_csv('detailed_effort_predictions.csv', index=False)

# Wilcoxon p-values between models
p_values = []
model_names = list(models.keys())

for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        model_i = model_names[i]
        model_j = model_names[j]
        preds_i = np.array(model_predictions[model_i])
        preds_j = np.array(model_predictions[model_j])
        try:
            stat, p = wilcoxon(preds_i, preds_j)
        except ValueError:
            p = np.nan
        p_values.append({
            'Model_1': model_i,
            'Model_2': model_j,
            'p_value': p
        })

pval_df = pd.DataFrame(p_values)
pval_df.to_csv('wilcoxon_p_values.csv', index=False)

print("✔️ All done! Results saved to 'effort_prediction_results.csv' and 'wilcoxon_p_values.csv'")
