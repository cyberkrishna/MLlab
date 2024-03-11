import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import minkowski
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    # Replace NaN values in y with a default value or impute them using appropriate techniques
    y = np.nan_to_num(y, nan=-1)  # Replace NaN with -1, you can choose another default value as needed

    return X, y

def evaluate_intraclass_interclass(X, y, class1_label, class2_label):
    class1_indices = np.where(y == class1_label)[0]
    class2_indices = np.where(y == class2_label)[0]
    class1_data = X[class1_indices]
    class2_data = X[class2_indices]
    centroid_class1 = np.mean(class1_data, axis=0)
    centroid_class2 = np.mean(class2_data, axis=0)
    spread_class1 = np.std(class1_data, axis=0)
    spread_class2 = np.std(class2_data, axis=0)
    interclass_distance = np.linalg.norm(centroid_class1 - centroid_class2)
    return centroid_class1, centroid_class2, spread_class1, spread_class2, interclass_distance

def observe_feature(X, feature_index):
    feature_vector = X[:, feature_index]
    plt.hist(feature_vector, bins=10)
    plt.title('Histogram of Feature')
    plt.xlabel('Feature Values')
    plt.ylabel('Frequency')
    plt.show()
    mean_feature = np.mean(feature_vector)
    variance_feature = np.var(feature_vector)
    return mean_feature, variance_feature

def calculate_minkowski_distance(X, vector1_index, vector2_index):
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

def train_knn_classifier(X_train, X_test, y_train, y_test, k=3):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    accuracy = neigh.score(X_test, y_test)
    predictions = neigh.predict(X_test)
    return accuracy, predictions

def compare_knn_models(X_train, X_test, y_train, y_test, k_values):
    accuracy_scores = []
    for k in k_values:
        accuracy, _ = train_knn_classifier(X_train, X_test, y_train, y_test, k)
        accuracy_scores.append(accuracy)
    plt.plot(k_values, accuracy_scores)
    plt.title('Accuracy vs. k')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.show()

def evaluate_performance(y_test, predictions):
    conf_matrix = confusion_matrix(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    return conf_matrix, precision, recall, f1

# Main code
file_path = r'C:\Users\Murari\Downloads\code_comm.csv'
X, y = load_data(file_path)

# A1. Evaluate intraclass spread and interclass distances
centroid_class1, centroid_class2, spread_class1, spread_class2, interclass_distance = evaluate_intraclass_interclass(X, y, 9, 6)

# A2. Density pattern observation for a feature
mean_feature, variance_feature = observe_feature(X, 0)

# A3. Minkowski distance calculation
calculate_minkowski_distance(X, 0, 1)

# A4. Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# A5. Train a kNN classifier
accuracy, predictions = train_knn_classifier(X_train, X_test, y_train, y_test)

# A6. Test the accuracy
print("Accuracy:", accuracy)

# A7. Use predict() function
print("Predictions:", predictions)

# A8. Comparison of kNN (k=3) with kNN (k=1) for varying k
compare_knn_models(X_train, X_test, y_train, y_test, range(1, 12))

# A9. Confusion matrix and performance metrics
conf_matrix, precision, recall, f1 = evaluate_performance(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
