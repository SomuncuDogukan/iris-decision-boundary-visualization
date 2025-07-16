"""
Project: Decision Boundary Visualization with Decision Tree on Iris Dataset
Author: Dogukan Somuncu
Date: 2025

Description:
    This script visualizes decision boundaries created by a Decision Tree classifier
    trained on all possible 2D feature combinations from the Iris dataset.
    For each pair of features, a separate model is trained and its decision surface is plotted.
    This helps understand how different feature pairs contribute to class separation.
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()

# Number of output classes (Setosa, Versicolor, Virginica)
n_classes = len(iris.target_names)

# Color map for each class: red, yellow, blue
plot_colors = "ryb"

# Iterate over all 6 possible 2D feature combinations
for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
    # Select two features (columns) for the current pair
    X = iris.data[:, pair]
    y = iris.target

    # Train a Decision Tree Classifier on the 2D data
    clf = DecisionTreeClassifier().fit(X, y)

    # Create subplot (2 rows × 3 columns)
    ax = plt.subplot(2, 3, pairidx + 1)
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    # Display the decision boundary
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        ax=ax,
        xlabel=iris.feature_names[pair[0]],
        ylabel=iris.feature_names[pair[1]]
    )

    # Overlay the actual data points on the plot
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(
            X[idx, 0], X[idx, 1],
            c=color,
            label=iris.target_names[i],
            cmap=plt.cm.RdYlBu,
            edgecolors="black"
        )

# Show legend
plt.legend()
plt.show()
