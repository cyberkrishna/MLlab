import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import minkowski
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Load data from CSV file
data = pd.read_csv(r'C:\Users\Murari\Downloads\code_comm.csv')  # Update 'your_dataset.csv' with your CSV file name

# Assuming your dataset has features in columns and the last column as labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# A1. Evaluate intraclass spread and interclass distances
# Assuming you have two classes in your dataset
class1_label = 9
class2_label = 6
class1_indices = np.where(y == class1_label)[0]  # Assuming class labels are encoded as integers
class2_indices = np.where(y == class2_label)[0]

class1_data = X[class1_indices]
class2_data = X[class2_indices]

centroid_class1 = np.mean(class1_data, axis=0)
centroid_class2 = np.mean(class2_data, axis=0)

spread_class1 = np.std(class1_data, axis=0)
spread_class2 = np.std(class2_data, axis=0)

interclass_distance = np.linalg.norm(centroid_class1 - centroid_class2)

print("Centroid for class 1:", centroid_class1)
print("Centroid for class 2:", centroid_class2)
print("Spread for class 1:", spread_class1)
print("Spread for class 2:", spread_class2)
print("Interclass distance:", interclass_distance)

# A2. Density pattern observation for a feature
feature_index = 0  # Choose the index of the feature you want to analyze
feature_vector = X[:, feature_index]

plt.hist(feature_vector, bins=10)  # Adjust bins as necessary
plt.title('Histogram of Feature')
plt.xlabel('Feature Values')
plt.ylabel('Frequency')
plt.show()

mean_feature = np.mean(feature_vector)
variance_feature = np.var(feature_vector)
print("Mean of feature:", mean_feature)
print("Variance of feature:", variance_feature)

# A3. Minkowski distance calculation
vector1_index = 0  # Index of first vector
vector2_index = 1  # Index of second vector
vector1 = X[vector1_index]
vector2 = X[vector2_index]

r_values = list(range(1, 11))
distances = []

for r in r_values:
    distance = minkowski(vector1, vector2, r)
    distances.append(distance)

plt.plot(r_values, distances)
plt.title('Minkowski Distance vs. r')
plt.xlabel('r')
plt.ylabel('Minkowski Distance')
plt.show()

# A4. Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# A5. Train a kNN classifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

# A6. Test the accuracy
accuracy = neigh.score(X_test, y_test)
print("Accuracy:", accuracy)

# A7. Use predict() function
predictions = neigh.predict(X_test)
print("Predictions:", predictions)

# A8. Comparison of kNN (k=3) with kNN (k=1) for varying k
accuracy_scores = []
k_values = range(1, 12)

for k in k_values:
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    accuracy = neigh.score(X_test, y_test)
    accuracy_scores.append(accuracy)

plt.plot(k_values, accuracy_scores)
plt.title('Accuracy vs. k')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()

# A9. Confusion matrix and performance metrics
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
