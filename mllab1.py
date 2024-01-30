import math
def  numberofvowcon(a):
    vowels=['a','e','i','o','u']
    vowelcount=0
    consonantcount=0
    for i in a:
        if i.lower() in vowels:#lower() is used to ignore the casing
            vowelcount+=1
        else:
            consonantcount+=1
            
    return vowelcount,consonantcount


string ="Apple"
[a,b]=numberofvowcon(string)
print("Number of vowels:",a,"\nNumber of consonants",b)

#2nd question

def matrixmultiplication(a,b):
    #Check if the matrices are compatible for multiplication
    if len(a[0])!=len(b):
        return "The matrices cannot be multiplied"
    result = [[0 for x in range(len(b))] for y in range(len(a))]
    #Performin multiplication
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                result[i][j] += a[i][k]*b[k][j]
                return result
            
#example of how to execute the function
matrix1 =[[1,2],[3,4]]
matrix2 =[[5,6],[7,8]]
print("Result of Matrix Multiplication:\n",matrixmultiplication(matrix1,matrix2))

#3rd question
def numberofcommonelements(a,b):
    common_elements=[]
    for i in a:
        if i in b and i not in common_elements:
            common_elements.append(i)
            
    return len(common_elements)
    
    
list1=[1,2,3,4,5]
list2=[4,5,6,7,8]
print("\nCommon elements between two lists:")
print(numberofcommonelements(list1,list2))

#4th question
def transposeofmatrix(a):
    transposed_matrix = [[0 for x in range(len(a))] for y in range(len(a[0]))]
    for i in range(len(a)):
        for j in range(len(a[0])):
            transposed_matrix[j][i] = a[i][j]
            return transposed_matrix
        
#example
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print("\nTransposed matrix:\n",transposeofmatrix(matrix))

#Question 5 euclidian distance
def euclidian_distance(a,b):
    dist=0 
    for i in range(len(a)):
        dist+= (a[i]-b[i])**2
    return math.sqrt(dist)
    
pointA = [4,9,6]
pointB = [4,5,4]
print("\nEuclidian Distance Between Two Points:",euclidian_distance(pointA, pointB))

        
