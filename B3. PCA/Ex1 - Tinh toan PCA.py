"""=============================================================================
    Ex1: Tính toán PCA
        a) Tạo một ma trận A(3000, 3) có các giá trị ngẫu nhiên từ 1 đến 255
        b) Áp dụng tính toán PCA
        c) Trực quan hóa kết quả 
============================================================================="""
import numpy as np
from numpy import mean
from numpy import cov
from numpy.linalg import eig

##------------------------------------------------------------------------------
print('\n*** a) Tạo ma trận có các giá trị ngẫu nhiên từ 1 đến 255:')
##------------------------------------------------------------------------------
# set random seed to repeat
np.random.seed(1)
A = np.random.randint(1, 256, (100,3))
print('- Matrix A', A.shape, ': \n', A[0:10])

print('\n*** b) Áp dụng PCA:')
# columns' means
M = mean(A.T, axis = 1)
print('- Mean vectors M:', M, '\n')

# center columns by subtracting column means
C = A - M
print('- Center matrix C', C.shape, ': \n', C[0:10], '\n')

# calculate covariance matrix of centered matrix
# cov(A)  --> rows: features; cols: objects
V = cov(C.T)
print('- Covariance matrix V', V.shape, ': \n', V)

# factorize covariance matrix
eigenvalues, eigenvectors = eig(V)

print('- Eigenvectors P', eigenvectors.shape, ': \n', eigenvectors, '\n')
print('- Eigenvalues Lambda', eigenvalues.shape, ': \n', eigenvalues, '\n')

# project data
P = eigenvectors.T.dot(C.T)
print(P.T)




