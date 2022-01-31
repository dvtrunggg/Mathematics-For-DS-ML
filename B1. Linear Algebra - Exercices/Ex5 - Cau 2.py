# -*- coding: utf-8 -*-
"""=============================================================================
Ex5: SPARSE MATRIX
    Câu 2:
        a) Tạo ra một ma trận BigA(1000, 1000) phần tử ngẫu nhiên tử -10 đến 5.
        b) Thay thế giá trị 0 cho tất cả các phần tử âm trong ma trận BigA
        c) Tính kích thước của ma trận BigA
        d) Tạo sparse matrix BigS từ BigA
        e) Tích kích thước của BigS
        f) Tính sparsity của sparse matrix
        g) Trực quan hóa BigS
============================================================================="""

from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import numpy             as np
import random

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
BigA = create_matrix_random(1000, 1000, -10, 5)

## b)
BigA[BigA < 0] = 0

## c) Xem kích thước
print("Size of full matrix with zeros =", BigA.nbytes/(1024**2), " MB") 

## d)
BigS = csr_matrix(BigA)

## e)
print("Size of csr_matrix =", BigS.data.nbytes/(1024**2), " MB")

## f)
plt.figure(figsize = (10, 10))
plt.spy(BigS, markersize = 0.1)



