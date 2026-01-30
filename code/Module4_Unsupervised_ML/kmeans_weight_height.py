import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 1. Load dataset
# ----------------------------
df = pd.read_csv("weight_height.csv")

# ----------------------------
# 2. Features only (UNSUPERVISED)
# ----------------------------
X = df[["Weight", "Height"]]

# ----------------------------
# 3. Feature Scaling (MANDATORY for K-Means)
# ----------------------------
# Why scale? K-Means uses distance (Euclidean distance)
# If we do not scale:
# One feature (e.g., weight) dominates
# Other features (e.g., height) become irrelevant
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# 4. Train K-Means model
# ----------------------------
kmeans = KMeans(
    n_clusters=2,      # assume 2 body-type groups
    random_state=42,
    n_init=10 # number of times K-Means is run with different random starting centroids, use the best
)

# train the K-Means model on X_scaled and immediately assigns a cluster label to every data point

# kmeans.fit(X_scaled)          # learn cluster centers
# There will be one center per cluster
# Since our data is weight-height, each center has two values: [ weight_center , height_center ]
# So, kmeans.cluster_centers_ will look like this and used later in plotting:
#array([
#    [ 0.12, -0.45 ],   # cluster 0 center (SCALED), i.e not in kg/cm
#    [ 1.08,  0.92 ]    # cluster 1 center (SCALED), i.e not in kg/cm
#])

# labels = kmeans.predict(X_scaled)  # assign cluster IDs
# labels: [0, 1, 1, 0, 0, 1, ...] # NOTE: THESE ARE NOT CLASS LABELS - THIS IS UNSUPERVISED LEARNING
labels = kmeans.fit_predict(X_scaled)

# ----------------------------
# 5. Attach cluster labels
# ----------------------------
# Add a new column to the df, containing label for each row
df["Cluster"] = labels

print(df)

# ----------------------------
# 6. Visualization
# ----------------------------
plt.figure(figsize=(8, 6))

plt.scatter(
    df["Weight"],
    df["Height"],
    c=df["Cluster"], # Same cluster points should have the same colour
    s=80 # Size of points (s=20 -> small, s=80 -> normal)
)

# Plot centroids
# First convert scaled data back to original (unscaled) format
centers = scaler.inverse_transform(kmeans.cluster_centers_)

# centers will be a 2D array as explained earlier, but in scaled format - now unscale it
# centers =
#[
#  [63.2, 170.8],   # centroid of cluster 0
#  [67.9, 176.3]    # centroid of cluster 1
#]

plt.scatter(
    centers[:, 0], # Take all rows, column 0, i.e. weights (e.g. [63.2, 67.9]) -> x-coordinates -> One point
    centers[:, 1], # Take all rows, column 1, i.e. heights (e.g. [170.8. 176.3]) -> y-coordinates -> One point
    marker="X",
    s=200,
    linewidths=3,
    label="Centroids"
)

plt.xlabel("Weight")
plt.ylabel("Height")
plt.title("K-Means Clustering (Weight vs Height)")
plt.legend()
plt.grid(True)
plt.show()
