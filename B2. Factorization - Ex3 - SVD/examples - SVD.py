"""=============================================================================
Minh họa phương pháp SVD:
       a) Ma trận vuông
       b) Ma trận hình chữ nhật (m > n)
       c) Ma trận hình chữ nhật (m < n)  
============================================================================="""

import numpy as np

from numpy import array
from scipy.linalg import svd

## Các ma trận thử nghiệm
ma_tran_vuong = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
ma_tran_hcn1  = array([[1, 2], [3, 4], [5, 6]])
ma_tran_hcn2  = array([[3, 1, 1], [-1, 3, 1]])

A = ma_tran_hcn2
m = A.shape[0]  # rows
n = A.shape[1]  # cols
print('Ma trận A(', A.shape, ':\n', A)

## SVD
print('\n********** Phương pháp SVD **********')
U, eigenValues, VT = svd(A)

print('\nMa trận U  (left-singular vectors)' , U.shape, ':\n', U)

S = np.zeros(A.shape)
if (m < n):
    S[:m, :m] = np.diag(eigenValues)
else:
    S[:n, :n] = np.diag(eigenValues)
    
print('\nMa trận S  (single values)' , S.shape, ':\n', S)

print('\nMa trận VT (right-singular vectors)', VT.shape, ':\n', VT)

## Tái tạo ma trận ban đầu
print('\n* Tái tạo ma trận A (kiểm chứng lại phép phân rã):', A.shape, '\n', U @ S @ VT)

## Compact SVD
print('\n********** Phương pháp Compact SVD **********')
S = np.diag(eigenValues)
print('\nMa trận S  (single values)' , S.shape, ':\n', S)
S_1 = np.diag(1 / eigenValues)

VT = VT[:m, :n]
print('\nMa trận VT (right-singular vectors)', VT.shape, ':\n', VT)

U = A @ (VT.T) @ S_1

print('\n* Tái tạo ma trận A (kiểm chứng lại phép phân rã):', A.shape, '\n', U @ S @ VT)
