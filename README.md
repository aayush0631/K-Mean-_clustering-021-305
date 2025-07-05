# K-Mean-_clustering-021-305
# K-Means Clustering with NumPy and Matplotlib

This repository contains simple Python implementations of the **K-Means Clustering Algorithm** using only NumPy and Matplotlib. Two different datasets are used to demonstrate clustering on 2D data and 4D feature data (visualized in 2D using the first two features).

---

## 📁 Contents

1. **`kmeans_2d.py`** – K-Means clustering on 2D points (visualized in a scatter plot).
2. **`kmeans_4d_projection.py`** – K-Means clustering on 4D samples with projection of the first two features for visualization.
3. **`README.md`** – This file.

---

## 🔍 Overview

### 1️⃣ 2D Point Clustering (`k=2`)

- Clusters a small set of 2D points into 2 groups.
- Visualizes clustering using scatter plots with centroid markers.
- Calculates **WCSS** (Within-Cluster Sum of Squares) to measure compactness.

### 2️⃣ 4D Sample Clustering (`k=3`)

- Uses higher-dimensional (4D) sample data.
- Projects and visualizes results using the first two features.
- Handles empty clusters gracefully.
- Also computes **WCSS** as clustering performance metric.

---

## 📊 Visualization

- Clustered data points are shown with different colors.
- Final centroids are marked with black 'X'.
- The graphs include legends, grid lines, and axis labels.

---

## 🧠 Concepts Demonstrated

- Distance calculation using `numpy.linalg.norm`.
- Cluster assignment via closest centroid.
- Updating centroids by computing mean of assigned points.
- Convergence detection using `np.allclose`.
- WCSS computation for evaluation.

---

## ▶️ Run Instructions

1. Make sure you have Python 3.x installed.
2. Install the required libraries:
   ```bash
   pip install numpy matplotlib
