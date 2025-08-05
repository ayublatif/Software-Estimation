import pandas as pd
import numpy as np
from tkinter import *
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# ----------------------
# Load Dataset
# ----------------------
df = pd.read_csv('c:\\ietcode\\augmented_project_data1.csv')
feature_cols = ['Effort', 'TeamSalary', 'EstimatedCost']
X_all = df[feature_cols].values
y_all = df['ActualTime'].values

# ✅ Global scalers (fit once for LR)
global_scaler_X = StandardScaler().fit(X_all)
global_scaler_y = StandardScaler().fit(y_all.reshape(-1, 1))

# ----------------------
# Prediction Function
# ----------------------
def predict_time():
    try:
        # Clear previous predictions and lines
        for widget in prediction_frame.winfo_children():
            widget.destroy()
        canvas.delete("chain")

        # Get user inputs
        effort = float(entry_effort.get())
        salary = float(entry_salary.get())
        cost = float(entry_cost.get())
        X_test = np.array([[effort, salary, cost]])

        # KNN (7 nearest neighbors)
        distances = pairwise_distances(X_test, X_all, metric='euclidean').flatten()
        neighbor_indices = np.argsort(distances)[:7]
        X_train = X_all[neighbor_indices]
        y_train = y_all[neighbor_indices]

        # ✅ Linear Regression (scaled with bound)
        X_train_scaled = global_scaler_X.transform(X_train)
        y_train_scaled = global_scaler_y.transform(y_train.reshape(-1, 1)).ravel()
        X_test_scaled = global_scaler_X.transform(X_test)

        lr = LinearRegression().fit(X_train_scaled, y_train_scaled)
        lr_pred_scaled = lr.predict(X_test_scaled)
        lr_pred = global_scaler_y.inverse_transform(lr_pred_scaled.reshape(-1, 1))[0][0]

        # ✅ Bound LR prediction to realistic range
        neighbor_min = min(y_train)
        neighbor_max = max(y_train)
        lower_bound = neighbor_min * 0.8  # 20% below neighbor min
        upper_bound = neighbor_max * 1.2  # 20% above neighbor max
        lr_pred = max(min(lr_pred, upper_bound), lower_bound)

        # Train other models (raw values)
        svr = SVR().fit(X_train, y_train)
        rf = RandomForestRegressor().fit(X_train, y_train)
        xgb = XGBRegressor(verbosity=0).fit(X_train, y_train)

        # Predictions
        predictions = {
            "Linear Regression": lr_pred,
            "Support Vector Regression": svr.predict(X_test)[0],
            "Random Forest": rf.predict(X_test)[0],
            "XGBoost": xgb.predict(X_test)[0]
        }

        # Display predictions in blockchain blocks
        prev_y_center = None
        prediction_frame.update()
        for model_name, value in predictions.items():
            block = create_block(prediction_frame, model_name, value)
            block.pack(pady=8, padx=50, fill="x")
            prediction_frame.update()

            # Blockchain chaining line
            block_y = block.winfo_rooty() - prediction_frame.winfo_rooty() + block.winfo_height() // 2
            if prev_y_center is not None:
                canvas.create_line(260, prev_y_center + 5, 260, block_y - 5, width=2, fill="#007ACC", dash=(4, 3), tags="chain")
            prev_y_center = block_y

    except ValueError:
        for widget in prediction_frame.winfo_children():
            widget.destroy()
        canvas.delete("chain")
        error_lbl = Label(prediction_frame, text="❌ Please enter valid numeric inputs!",
                          font=("Segoe UI", 11, "bold"), fg="red", bg="#EAF4FF")
        error_lbl.pack(pady=6)

# ----------------------
# Blockchain Block Creator
# ----------------------
def create_block(parent, model_name, prediction):
    block = Frame(parent, bg="white", highlightbackground="#007ACC", highlightthickness=2, bd=1)
    Label(block, text=model_name, font=("Segoe UI", 11, "bold"), bg="white", fg="#007ACC").pack(pady=(4, 1))
    Label(block, text=f"Predicted Time: {prediction:.2f} hrs", font=("Segoe UI", 10), bg="white").pack(pady=(0, 4))
    return block

# ----------------------
# GUI Setup
# ----------------------
root = Tk()
root.title("Software Time Estimation")
root.geometry("520x600")
root.configure(bg="#EAF4FF")

# Title
title_lbl = Label(root, text="Software Time Estimation", font=("Segoe UI", 18, "bold"), bg="#EAF4FF", fg="#004080")
title_lbl.pack(pady=12)

# Input Frame
input_frame = Frame(root, bg="#EAF4FF")
input_frame.pack(pady=5)

Label(input_frame, text="Effort:", font=("Segoe UI", 11), bg="#EAF4FF").grid(row=0, column=0, padx=8, pady=6, sticky=W)
entry_effort = Entry(input_frame, width=22, font=("Segoe UI", 11))
entry_effort.grid(row=0, column=1, pady=6)

Label(input_frame, text="Team Salary:", font=("Segoe UI", 11), bg="#EAF4FF").grid(row=1, column=0, padx=8, pady=6, sticky=W)
entry_salary = Entry(input_frame, width=22, font=("Segoe UI", 11))
entry_salary.grid(row=1, column=1, pady=6)

Label(input_frame, text="Estimated Cost:", font=("Segoe UI", 11), bg="#EAF4FF").grid(row=2, column=0, padx=8, pady=6, sticky=W)
entry_cost = Entry(input_frame, width=22, font=("Segoe UI", 11))
entry_cost.grid(row=2, column=1, pady=6)

# Predict Button
btn_predict = Button(root, text="Calculate Time", command=predict_time,
                     bg="#007ACC", fg="white", font=("Segoe UI", 12, "bold"), width=25, height=1)
btn_predict.pack(pady=15)

# Prediction Display (Canvas + Frame)
container = Frame(root, bg="#EAF4FF")
container.pack(fill="both", expand=True)

canvas = Canvas(container, bg="#EAF4FF", highlightthickness=0, width=520, height=300)
canvas.pack(fill="both", expand=True)

prediction_frame = Frame(canvas, bg="#EAF4FF")
canvas.create_window((0, 0), window=prediction_frame, anchor="nw", width=520)

root.mainloop()
