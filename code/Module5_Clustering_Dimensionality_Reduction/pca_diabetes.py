import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("diabetes.csv")

X = df.drop(columns="Outcome")
y = df["Outcome"]

# Handle missing values + scale
X = SimpleImputer(strategy="mean").fit_transform(X)
X = StandardScaler().fit_transform(X)

# PCA
pca = PCA()

# X_pca is a new dataset where rows = data points (patients / samples), Columns = principal components
X_pca = pca.fit_transform(X)
print(f"Shape of X_pca: {X_pca.shape}")

# Cumulative variance
# pca.explained_variance_ratio_ gives something like [0.40, 0.25, 0.15, 0.10, 0.05, ...]
# Meaning: PC1 explains 0.40, PC2 explains 0.25 variance etc
# Now we are printing cumulative explained variance at each additional component
print(pca.explained_variance_ratio_)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
print(cumulative_variance)

# Plot
plt.plot(cumulative_variance, marker='o')
plt.axhline(0.95, linestyle='--')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance")
plt.show()

# Choose components for 95% variance
# Suppose cumulative variance = [0.40, 0.65, 0.80, 0.90, 0.96, 0.99]
# Now create a boolean array = [False, False, False, False, True, True] ... since we check for >= 95%
# np.argmax() finds the first "true" from the above, which is 4, since counting starts at 0
# +1 tells us at what point we have reached 95%
n_components = np.argmax(cumulative_variance >= 0.95) + 1

# Take all rows, columns, and required number of components from X_pca dataset
X_reduced = X_pca[:, :n_components]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluation
accuracy = accuracy_score(y_test, model.predict(X_test))
print("Components used:", n_components)
print("Accuracy:", accuracy)
