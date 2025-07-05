import numpy as np
import matplotlib.pyplot as plt

points = np.array([
    [1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6],
    [9, 11], [8, 2], [10, 2], [9, 3], [4, 9]
])

num_clusters = 2
iterations = 100
np.random.seed(0)

initial_idxs = np.random.permutation(len(points))[:num_clusters]
centers = points[initial_idxs]

for i in range(iterations):
    dists = np.array([[np.linalg.norm(p - c) for c in centers] for p in points])
    labels = np.argmin(dists, axis=1)
    updated_centers = np.array([points[labels == c].mean(axis=0) for c in range(num_clusters)])
    if np.allclose(centers, updated_centers):
        print(f"Algorithm converged at iteration {i + 1}")
        break
    centers = updated_centers

print("\nCluster Results:")
for idx, pt in enumerate(points):
    print(f"{pt} assigned to Cluster {labels[idx]}")

print("\nFinal Centroid Positions:")
print(centers)

total_wcss = sum(np.sum((points[labels == c] - centers[c]) ** 2) for c in range(num_clusters))
print(f"\nTotal WCSS: {total_wcss:.2f}")

cluster_colors = ['orange', 'green']
for c in range(num_clusters):
    plt.scatter(points[labels == c][:, 0], points[labels == c][:, 1], color=cluster_colors[c], label=f'Group {c}')

plt.scatter(centers[:, 0], centers[:, 1], color='black', s=150, marker='X', label='Centroids')
plt.title("Modified k-Means Clustering (k=2)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid(True)
plt.show()
