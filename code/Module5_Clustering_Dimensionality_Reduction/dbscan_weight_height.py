import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np

# ----------------------------
# 1. Load dataset
# ----------------------------
df = pd.read_csv("weight_height.csv")

print(df.head())

# ----------------------------
# 2. Features only (DBSCAN is UNSUPERVISED)
# ----------------------------
X = df[["Weight", "Height"]]

# ----------------------------
# 3. Feature Scaling (MANDATORY)
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# 4. Apply DBSCAN
# ----------------------------
# Note: We have scaled dataset
# If two points are within 0.6 standard deviations (Euclidean distance) 
# of each other in the (Weight, Height) plane, treat them as neighbors
# Almost like Z-score of (Weight, Height) combined
dbscan = DBSCAN(
    eps=0.6,        # points within this value
    min_samples=3   # minimum points to form a dense region
)

# Run DBSCAN to predict cluster ID/label
labels = dbscan.fit_predict(X_scaled)

# ----------------------------
# 5. Attach cluster labels
# ----------------------------
# Add a new column to indicate cluster for each data point
df["Cluster"] = labels

print("\nClustered Data:")
print(df)

# ----------------------------
# 6. Number of clusters and noise points
# ----------------------------
# labels might contain [ 0,  0,  1,  1,  1, -1,  0, -1 ]
# set(labels) will keep unique values only, set([0, 0, 1, 1, 1, -1, 0, -1]) → {0, 1, -1}
# len({0, 1, -1}) = 3
# But we have also considered noise as a cluster
# So, if -1 is in our cluster, reduce 1 from cluster count, else reduce 0, meaning keep cluster count unchanged
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# How many noise clusters?
n_noise = np.sum(labels == -1)

print(f"\nNumber of clusters found: {n_clusters}")
print(f"Number of noise points: {n_noise}")

# ----------------------------
# 7. Visualization
# ----------------------------
plt.figure(figsize=(8, 6))

# Plot clusters
unique_labels = set(labels)

for label in unique_labels:
    if label == -1:
        # Noise points displayed in red colour
        plt.scatter(
            df.loc[df["Cluster"] == label, "Weight"],
            df.loc[df["Cluster"] == label, "Height"],
            c="red",
            label="Noise",
            s=80,
            marker="x"
        )
    else:
        plt.scatter(
            df.loc[df["Cluster"] == label, "Weight"],
            df.loc[df["Cluster"] == label, "Height"],
            label=f"Cluster {label}",
            s=80
        )

plt.xlabel("Weight")
plt.ylabel("Height")
plt.title("DBSCAN Clustering (Weight vs Height)")
plt.legend()
plt.grid(True)
plt.show()