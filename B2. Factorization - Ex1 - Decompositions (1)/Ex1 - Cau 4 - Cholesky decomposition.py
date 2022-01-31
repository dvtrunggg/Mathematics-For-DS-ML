"""=============================================================================
Ex1: DECOMPOSITION
    Câu 4: Cholesky Decomposition
        a) Tạo ma trận A(3 x 3) chứa các giá trị ngẫu nhiên trong khoảng 3 - 9
        b) Chuyển A thành ma trận vuông đối xứng theo tam giác dưới
        c) Thực hiện phân rã Cholesky, nếu có thể
============================================================================="""

import numpy  as np
import random       
from numpy.linalg import cholesky
## numpy.linalg.cholesky: Default --> LOWER
## scipy.linalg.cholesky: Default --> UPPER

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

## a) Tạo ma trận A(3 x 3) chứa các giá trị ngẫu nhiên trong khoảng 3 - 9
m, n, min, max = 3, 3, 3, 9
A = create_matrix_random(m, n, min, max)

## b) Chuyển A thành ma trận vuông đối xứng theo tam giác dưới
## Điều kiện phân rã Cholesky: ma trận ĐỐI XỨNG
for i in range(A.shape[0]):
    for j in range(i):
        A[j][i] = A[i][j]
print('*** Ma trận A', A.shape, ':\n', A)        

## c) Thực hiện phân rã Cholesky, nếu có thể
## Kiểm tra ma trận xác định dương
test = np.linalg.eigvalsh(A)

pos_def = np.all(test > 0)   
if (pos_def == True):
    L = cholesky(A)

    print('Ma trận L', L.shape, ':\n', L)

    print('Tái tạo ma trận A:\n', L.dot(L.T))
else:
    print('\nMa trận A KHÔNG thỏa điều kiện XÁC ĐỊNH DƯƠNG')
    
##------------------------------------------------------------------------------
## Hàm tạo ma trận xác định dương    
##------------------------------------------------------------------------------
def create_matrix_positive_definite(m, n, start, end):
    A       = None
    pos_def = False
    
    while (pos_def == False):
        A = create_matrix_random(m, n, start, end)
        for i in range(A.shape[0]):
            for j in range(i):
                A[j][i] = A[i][j]
        test    = np.linalg.eigvalsh(A)
        pos_def = np.all(test > 0)
    return A
##------------------------------------------------------------------------------
A = create_matrix_positive_definite(3, 3, 3, 9)
print('\n*** Ma trận A', A.shape, 'thỏa điều kiện XÁC ĐỊNH DƯƠNG:\n', A)

print('Áp dụng phân rã Cholesky:')
L = cholesky(A)
print('Ma trận L', L.shape, ':\n', L)

print('\nTái tạo A từ L, L_T (kiểm chứng lại phép phân rã)\n', L.dot(L.T))