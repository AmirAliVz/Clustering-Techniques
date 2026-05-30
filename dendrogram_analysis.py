from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import gower
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import fcluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 14

os.makedirs("Figures", exist_ok=True)

df = pd.read_csv('data/cleaned_churn_data.csv')

# =======================================
#   Part E. Optimal number of clusters
# =======================================
print(50*"=")
print("Hierarchical Clustering ...")
# Compute Gower Distance Matrix
gower_dist = gower.gower_matrix(df)

# Perform hierarchical clustering
linked = linkage(gower_dist, method='average')
print("Dendrogram Created.")

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendro = dendrogram(linked, truncate_mode='level', p=5)
plt.title('Dendrogram with Key Merge Distances')
plt.xlabel('Clusters')
plt.ylabel('Distance')

# Get Top Distances
distances = linked[:, 2]

# Compute differences between consecutive merges
diffs = np.diff(distances)

# Get largest jumps
largest_jumps_idx = np.argsort(diffs)[-5:]

print(50*"=")
print("Finding the best cutting threshold...")
print("Top 5 biggest jumps in distance on the dendrogram:")
for i in largest_jumps_idx:
    print(f"Jump from {distances[i]:.2f} to {distances[i+1]:.2f} = {diffs[i]:.2f}")

# Finding the cutting threshold for Optimal Number of Clusters
print(50*"=")
print("Finding the Optimal Number of Clusters...")
candidate_thresholds = distances[1:]  # possible cut points

results = []

for t in candidate_thresholds:
    k = len(set(fcluster(linked, t=t, criterion='distance')))
    results.append((t, k))

# Keep only reasonable cluster counts
filtered = [(t, k) for t, k in results if 2 <= k <= 6]

# Choose threshold with largest jump in that range
best_t = None
best_jump = -1

for i in range(len(distances) - 1):
    jump = diffs[i]
    t = distances[i + 1]

    k = len(set(fcluster(linked, t=t, criterion='distance')))

    if 2 <= k <= 6 and jump > best_jump:
        best_jump = jump
        best_t = t
        t_idx = i

print(f"Selected threshold: {best_t}, Jump: {best_jump}")
cut_threshold = best_t

# Draw horizontal line at the upper merge distance
plt.axhline(y=best_t, linestyle='--', lw=2, color='green', alpha=0.4)

# Add text label
plt.text(
    x=0,
    y=best_t + 0.2,
    s=f"Δ={best_jump:.2f}\n({distances[t_idx]:.2f}→{best_t:.2f})",
    fontsize=9
)

plt.savefig("Figures/DendrogramPlot.jpg", dpi=300, bbox_inches="tight")
plt.show()

clusters = fcluster(linked, t=cut_threshold, criterion='distance')

df['Cluster'] = clusters
print(50*"=")
print(f"Total Number of Clusters: {len(np.unique(clusters)):.4f}")


# =======================================
#   F1. Visualization + Cluster Quality
# =======================================
print(50*"=")
print("Visualizing Clusters...")
# Convert categorical to numeric for PCA (temporary encoding)
df_encoded = pd.get_dummies(df.drop('Cluster', axis=1))

# Reduce to 2D
pca = PCA(n_components=2)
df_scaled = StandardScaler().fit_transform(df_encoded)
pca_result = pca.fit_transform(df_scaled)

# Plot
unique_labels = np.unique(clusters)
cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

color_map = {label: cycle_colors[i % len(cycle_colors)] for i, label in enumerate(unique_labels)}
point_colors = [color_map[label] for label in clusters]

plt.figure(figsize=(10,6))
scatter = plt.scatter(
    pca_result[:, 0],
    pca_result[:, 1],
    c=point_colors,
)

plt.title('Customer Clusters (PCA Projection)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')


plt.savefig("Figures/Clusters.jpg", dpi=300, bbox_inches="tight")
plt.show()

# Data-Driven Interpretation
print(50*"=")
print("Cluster Summary:")
cluster_summary = df.groupby('Cluster').mean(numeric_only=True)
print(cluster_summary)

for col in df.select_dtypes(include='object').columns:
    print(df.groupby('Cluster')[col].value_counts(normalize=True))
    print(50*"=")