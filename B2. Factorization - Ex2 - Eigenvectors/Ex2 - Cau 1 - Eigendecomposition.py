"""=============================================================================
Ex2: EIGENDECOMPOSITION
    Câu 1:
        a) Tạo ma trận A(5 x 5) chứa các giá trị ngẫu nhiên trong khoảng 1 - 10
        b) Phân tích eigenvalues và eigenvectors
        c) Kiểm tra eigenvector đầu tiên theo dot và theo eigenvalue có bằng nhau?
           Nếu bằng nhau thì tái tạo A từ các eigenvalues và eigenvectors
============================================================================="""
    
import numpy  as np
import random       

from numpy.linalg import eig, inv
from numpy import diag

##------------------------------------------------------------------------------
## Hàm tạo 1 ma trận A[mxn] với giá trị ngẫu nhiên thuộc [start, end]
##------------------------------------------------------------------------------
def create_matrix_random(m, n, start, end):
    mtr = []
    for i in range(m):
        row = []
        for j in range(n):
            a = random.randint(start, end + 1)
            
            # Thêm giá trị vào dòng hiện hành 
            row.append(a)
            
        # Thêm dòng vào ma trận    
        mtr.append(row)
        
    return np.array(mtr)
##------------------------------------------------------------------------------
    
## a) Tạo ma trận A(5 x 5) chứa các giá trị ngẫu nhiên trong khoảng 1 - 10

## Test cases
m, n, min, max = 5, 5, 1, 10
A1 = create_matrix_random(m, n, min, max)

A2 = np.array([[10,  8, 11,  9,  6],
               [ 3,  9,  3,  8,  4], 
               [ 3,  7,  7, 11,  2], 
               [ 9,  2, 11, 10, 11], 
               [11,  6, 10,  2,  7]])

A3 = np.array([[ 8,  5,  4, 11,  8],
               [ 3,  9, 10,  8,  8],
               [ 1,  7,  6,  4, 10],
               [ 5,  1,  9,  8, 11],
               [ 8,  9,  3,  2,  9]])

A = A3
print('Ma trận A', A.shape, ':\n', A)

## b) Phân tích eigenvalues và eigenvectors
values, vectors = eig(A)

for j in range(len(values)):
    eigenvalue = values[j] 
    # print(type(eigenvalue))

    r = values[j].real
    i = values[j].imag
    print('Eigenvalue[%1d]' %j, '= (%f,' %r, '%f)' %i)
    
print('\nEigenvectors', vectors.shape, ':\n', vectors)

## c) Kiểm tra eigenvector đầu tiên theo dot và theo eigenvalue có bằng nhau?
##    Nếu bằng nhau thì tái tạo A từ các eigenvalues và eigenvectors
B = A.dot(vectors[:, 0])
print('\nVectơ B:\n', B.astype(int))

C = vectors[:, 0] * values[0]
print('\nVectơ C:\n', C.astype(int))

Q = vectors
L = diag(values)

print('\nTái tạo ma trận A:\n', Q.dot(L).dot(inv(Q)).astype(int))
