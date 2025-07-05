import numpy as np
import matplotlib.pyplot as plt

samples = np.array([
    [1.0, 1.5, 0.8, 1.2],
    [1.2, 1.7, 0.9, 1.1],
    [1.1, 1.4, 0.7, 1.3],
    [1.3, 1.6, 0.85, 1.15],
    [4.0, 4.5, 3.8, 4.2],
    [4.2, 4.7, 3.9, 4.1],
    [4.1, 4.4, 3.7, 4.3],
    [4.3, 4.6, 3.85, 4.15],
    [2.0, 2.2, 1.8, 2.1],
    [2.1, 2.3, 1.9, 2.0],
    [4.5, 4.8, 4.0, 4.4],
    [4.4, 4.9, 3.95, 4.35]
])

clusters = 3
steps = 100
np.random.seed(10)

chosen = np.random.choice(len(samples), clusters, replace=False)
centers = samples[chosen]

for step in range(steps):
    d = np.array([[np.linalg.norm(x - c) for c in centers] for x in samples])
    tags = np.argmin(d, axis=1)
    new_centers = np.array([
        samples[tags == i].mean(axis=0) if np.any(tags == i) else centers[i]
        for i in range(clusters)
    ])
    if np.allclose(centers, new_centers):
        print(f"Stopped at step {step + 1}")
        break
    centers = new_centers

print("Labels:")
for i, tag in enumerate(tags):
    print(f"Point {i} → Cluster {tag}")

print("\nCenters:")
print(centers)

loss = sum(np.sum((samples[tags == i] - centers[i]) ** 2) for i in range(clusters))
print(f"\nTotal WCSS: {loss:.4f}")

shade = ['orange', 'purple', 'cyan']
for i in range(clusters):
    pts = samples[tags == i]
    plt.scatter(pts[:, 0], pts[:, 1], color=shade[i], label=f'Group {i}')

plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=150, label='Centroids')
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.title('KMeans (k=3) on First 2 Features')
plt.legend()
plt.grid(True)
plt.show()
