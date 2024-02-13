import pandas as pd
import numpy as np

df = pd.read_excel(r"C:\Users\Murari\Downloads\Lab Session1 Data.xlsx")
# Example: Extracting specific columns into matrices
matrix1 = df[['Candies (#)','Mangoes (Kg)','Milk Packets (#)']].values  
matrix2 = df[['Payment (Rs)']].values
print(matrix1)
print(matrix2)
num_rows, num_columns = matrix1.shape
print("Dimensionality of the vector space:", num_columns)
print("The number of vectors that exist in the vector space:", num_rows)
np_matrix = df.to_numpy()
rank = np.linalg.matrix_rank(matrix1)
print("Rank of the matrix:", rank)
pseudo_inv=np.linalg.pinv(matrix1)
X=pseudo_inv@matrix2
print(X)
print("The individual cost of a candy is: ",round(X[0][0]))
print("The individual cost of a mango is: ",round(X[1][0]))
print("The individual cost of a milk packet is: ",round(X[2][0]))