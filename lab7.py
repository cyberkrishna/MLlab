import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

# Load data from CSV file
data = pd.read_csv(r'C:\Users\Downloads\code_comm.csv')

# Assuming the last column is the target variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifiers
classifiers = {
    'Perceptron': Perceptron(),
    'MLP': MLPClassifier(),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'CatBoost': CatBoostClassifier(),
    'XGBoost': XGBClassifier(),
    'Naive Bayes': GaussianNB()
}

# Define hyperparameter grids for Perceptron and MLP
perceptron_param_grid = {
    'alpha': np.linspace(0.0001, 0.01, 100),
    'max_iter': np.arange(100, 1000, 100)
}

mlp_param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
    'activation': ['logistic', 'relu'],
    'solver': ['sgd', 'adam'],
    'learning_rate_init': np.linspace(0.001, 0.01, 10)
}

# Define evaluation metrics
metrics = {
    'Accuracy': accuracy_score,
    'Precision': precision_score,
    'Recall': recall_score,
    'F1 Score': f1_score
}

results = {}

# Function to perform cross-validation and tune hyperparameters
def tune_and_evaluate(classifier_name, classifier, param_grid):
    print(f"Tuning hyperparameters for {classifier_name}...")
    random_search = RandomizedSearchCV(estimator=classifier, param_distributions=param_grid, n_iter=10, cv=5, scoring='accuracy', random_state=42)
    random_search.fit(X_train, y_train)
    
    best_params = random_search.best_params_
    classifier.set_params(**best_params)
    
    print(f"Evaluating {classifier_name}...")
    scores = {}
    for metric_name, metric_func in metrics.items():
        score = cross_val_score(classifier, X_train, y_train, cv=5, scoring=metric_func)
        scores[metric_name] = np.mean(score)
    results[classifier_name] = scores

# Tune and evaluate each classifier
for classifier_name, classifier in classifiers.items():
    if classifier_name == 'Perceptron':
        tune_and_evaluate(classifier_name, classifier, perceptron_param_grid)
    elif classifier_name == 'MLP':
        tune_and_evaluate(classifier_name, classifier, mlp_param_grid)
    else:
        print(f"Evaluating {classifier_name}...")
        scores = {}
        for metric_name, metric_func in metrics.items():
            score = cross_val_score(classifier, X_train, y_train, cv=5, scoring=metric_func)
            scores[metric_name] = np.mean(score)
        results[classifier_name] = scores

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Print results
print(results_df)

# Plot ROC curves
plt.figure(figsize=(10, 8))
for classifier_name in ['Perceptron', 'MLP']:
    classifier = classifiers[classifier_name]
    classifier.fit(X_train, y_train)
    y_pred_proba = classifier.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label='%s (AUC=%.2f)' % (classifier_name, roc_auc_score(y_test, y_pred_proba)))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True)
plt.show()
