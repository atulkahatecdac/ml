# pip install xgboost lightgbm
# If LightGBM fails: pip install lightgbm --prefer-binary

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("customer_loan_default_dataset.csv")

# Drop ID column
X = df.drop(columns=["customer_id", "default"])
y = df["default"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# XGBoost
from xgboost import XGBClassifier

# Define model
# Builds many small decision trees, one after another, each new tree fixes the mistakes of the previous ones
xgb_model = XGBClassifier(
    n_estimators=100, # Number of trees to build = 100 small improvements
    max_depth=3, # Simple trees (10 would be very complex, might overfit)
    learning_rate=0.1, # Step size/shrinkage ... Slow and careful
    eval_metric="logloss", # Evaluate performance during training ... Lower the better
    random_state=42
)

# Train
xgb_model.fit(X_train, y_train)

# Predict
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

# LightGMB
from lightgbm import LGBMClassifier

# Define model
lgbm_model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_data_in_leaf=5,   
    min_data_in_bin=1,  
    random_state=42
)

# Train
lgbm_model.fit(X_train, y_train)

# Predict
y_pred_lgbm = lgbm_model.predict(X_test)

# Evaluate
print("LightGBM Accuracy:", accuracy_score(y_test, y_pred_lgbm))
print(classification_report(y_test, y_pred_lgbm))

