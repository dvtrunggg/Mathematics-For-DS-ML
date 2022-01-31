# -*- coding: utf-8 -*-
"""=============================================================================
Ex5: SPARSE MATRIX
    Câu 1:
        a) Tạo ma trận A(5, 5) với các giá trị ngẫu nhiên từ -10 đến 5
        b) Thay thế giá trị 0 cho tất cả các phần tử âm trong ma trận A
        c) Tạo sparse matrix S từ A
        d) Tính sparsity của sparse matrix 
============================================================================="""
    
import numpy  as np
import random
from scipy.sparse     import csr_matrix

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

## a)
A = create_matrix_random(5, 5, -10, 5)
print('A =\n', A)

## b)
A[A < 0] = 0
print('A+ =\n', A)

## c)
print('\nCSR =\n', csr_matrix(A))

## d)
print('\nĐộ thưa = %.f' %(100 * (1.0 - (np.count_nonzero(A) / A.size))) + '%')
