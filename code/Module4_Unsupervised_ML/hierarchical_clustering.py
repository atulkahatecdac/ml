import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Data (Marks)
students = ['A', 'B', 'C', 'D', 'E', 'F']
marks = np.array([35, 38, 40, 75, 78, 82]).reshape(-1, 1)

# Hierarchical clustering (Ward linkage)
Z = linkage(marks, method='ward')

# Plot dendrogram
plt.figure(figsize=(6, 4))
dendrogram(Z, labels=students)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Student")
plt.ylabel("Distance")
plt.show()
