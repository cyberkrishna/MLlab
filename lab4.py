import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski

# Step 1: Load the CSV file
file_path = r"C:\Users\Murari\Downloads\code_comm.csv"
data = pd.read_csv(file_path)

# Step 2: Select feature and target class
feature_name = input("Enter the name of the feature you want to analyze: ")
target_class_name = input("Enter the name of the target class column: ")

# Step 3: Extract feature data
feature_data = data[feature_name]

# Step 4: Plot histogram
plt.hist(feature_data, bins=10, color='skyblue', edgecolor='black')  # Adjust the number of bins as needed
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of {}'.format(feature_name))
plt.show()

# Step 5: Calculate mean and variance
mean = np.mean(feature_data)
variance = np.var(feature_data)

print("Mean:", mean)
print("Variance:", variance)

# Step 6: Calculate class-wise mean and variance
class_means = data.groupby(target_class_name)[feature_name].mean()
class_variances = data.groupby(target_class_name)[feature_name].var()

print("\nClass-wise Mean:")
print(class_means)
print("\nClass-wise Variance:")
print(class_variances)

# Step 7: Take two example vectors for calculating Minkowski distance
vector1 = data.iloc[0][feature_name]  # Example: taking the first data point
vector2 = data.iloc[1][feature_name]  # Example: taking the second data point

# Step 8: Calculate Minkowski distance with r from 1 to 10
r_values = range(1, 11)
distances = [minkowski([vector1], [vector2], r) for r in r_values]

# Step 9: Plot Minkowski distance
plt.plot(r_values, distances)
plt.xlabel('r')
plt.ylabel('Minkowski Distance')
plt.title('Minkowski Distance vs. r')
plt.show()
