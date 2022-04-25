import numpy as np
from random import randint as Rint, random
import random
def PCA(A):
    print("INPUT MATRIX : ")
    print(A)
    #Get the mean values of each column
    Mean = np.mean(A.T,axis = 1)
    print("MEAN OF EACH COLUMN : "  )
    print(Mean)
    #Centering the columns
    Center = A - Mean
    print("CENTERING THE MATRIX : ")
    print(Center)
    #Calculating the covaraince matrix
    V = np.cov(Center.T)
    print("COVARIANCE MATRIX : ")
    print(V)
    #Eigendecomposition of covariance matrix
    values,vectors = np.linalg.eig(V)
    print("EIGENVECTORS")
    print(vectors)
    print("EIGENVALUES")
    print(values)
    print("TAKING DOT PRODUCT AND PROJECTION")
    P = vectors.T.dot(Center.T)
    proj = (vectors.T[:][:2]).T
    print(proj)
    # projecting the data
    P = proj.T.dot(Center.T)
    return P

def main():
    # Making a random matrix
    M = np.random.randint(-10,10,size = (random.randint(1,4),random.randint(1,4)))
    P = PCA(M)
    print("RESULT : ")
    print(P)
    print("ROUNDING : ")
    for i in P :
        for j in i:
            print(round(j,2),end="\t")
        print("\n")
main()