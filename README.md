# Decision Boundary Visualization with Decision Tree (Iris Dataset)

This project visualizes how a Decision Tree classifier separates the three Iris flower species using different combinations of features. For each 2D feature pair, a separate classifier is trained and the resulting decision surface is plotted.

## Dataset

- Source: `sklearn.datasets.load_iris()`
- Features:
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)
- Classes:
  - Setosa
  - Versicolor
  - Virginica

## Description

- All six combinations of the four features are used to train separate models.
- A `DecisionTreeClassifier` is fitted for each pair.
- Decision boundaries are displayed using `DecisionBoundaryDisplay.from_estimator`.
- Class regions and actual data points are plotted for comparison.

## Output

The following figure shows the decision boundaries for each feature pair:

<img width="799" height="521" alt="Figure_1" src="https://github.com/user-attachments/assets/499bfbe3-c954-410d-9fa1-178fc2ccd37e" />
