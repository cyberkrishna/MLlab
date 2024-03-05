print('Lab 5 code Here")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
import random

# Load dataset
dataset = pd.read_csv(r'C:\Users\Murari\Downloads\code_comm.csv')

# Handling missing values if any
dataset.dropna(inplace=True)

# A1. Classification evaluation
# Assuming 'features' are columns other than the target column
X_class = dataset.drop('score', axis=1)
y_class = dataset['score']  # Assuming 'score' is a continuous target variable

# Convert continuous target variable into discrete classes
# Example: Bin into 3 classes (low, medium, high)
y_class_bins = pd.cut(y_class, bins=3, labels=['low', 'medium', 'high'])

# Splitting into train and test sets
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class_bins, test_size=0.4, random_state=42)

# A3. Generating and visualizing training data
# Visualizing decision boundaries for any two features

# Function to randomly select two features from the dataset
def select_two_features(X):
    num_features = X.shape[1]
    feature_indices = random.sample(range(num_features), 2)
    return X.iloc[:, feature_indices]

h = .02  # Step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

k_values = [1, 3, 5, 7]  # Example values for k
for k in k_values:
    # Select two features for visualization
    X_visualize = select_two_features(X_train_class)

    # Train kNN classifier
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_visualize, y_train_class)

    # Plot the decision boundary
    x_min, x_max = X_visualize.iloc[:, 0].min() - 1, X_visualize.iloc[:, 0].max() + 1
    y_min, y_max = X_visualize.iloc[:, 1].min() - 1, X_visualize.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = knn_temp.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X_visualize.iloc[:, 0], X_visualize.iloc[:, 1], c=y_train_class, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f"2-Class classification (k = {k})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
