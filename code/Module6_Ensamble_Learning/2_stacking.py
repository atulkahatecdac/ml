import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------
# 1. Load data
# ----------------------------
df = pd.read_csv("ezybank_churn_dataset.csv")

# Drop ID column
X = df.drop(columns=["customer_id", "churned"])
y = df["churned"]

# ----------------------------
# 2. Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------------------
# 3. Scaling (important for LR & KNN)
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 4. Base learners
# ----------------------------
# A list of models that will be trained in parallel on the same dataset
# Each model 
# (1) Learns the problem in a different way
# (2) Makes its own predictions
# (3) Feeds those predictions to a meta-learner
base_learners = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]

# ----------------------------
# 5. Meta learner
# ----------------------------
# Does not see the original features 
# Only sees the predictions of the base models - Learns how to combine them optimally
meta_learner = GradientBoostingClassifier(
    n_estimators=100, # Number of stages it will go through, each will improve on the previous one
    random_state=42
)

# ----------------------------
# 6. Stacking model
# ----------------------------
# Builds a 2-level model
# Level-0: Base model (Logistic Regression, Random Forest, KNN)
# Level-1: Meta learner (GradientBoostingClassifier)

stacking_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    # passthrough=False -> Meta_learner only gets [ LR_pred , RF_pred , KNN_pred ]
    # passthrough=False -> Meta_learner gets [ original_features + base_model_predictions ]
    passthrough=False   
)

# ----------------------------
# 7. Train
# ----------------------------
stacking_model.fit(X_train_scaled, y_train)

# ----------------------------
# 8. Evaluate
# ----------------------------
y_pred = stacking_model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
