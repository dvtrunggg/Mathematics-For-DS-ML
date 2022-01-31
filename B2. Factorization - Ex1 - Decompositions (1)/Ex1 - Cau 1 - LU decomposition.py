"""=============================================================================
Ex1: DECOMPOSITION
    Câu 1: LU Decomposition
        a) Tạo ma trận A(4 x 4) chứa các giá trị ngẫu nhiên trong khoảng 1 - 10
        b) Phân tích thành các thành phần P, L, U. In kết quả
        c) Tái tạo lại ma trận A từ các thành phần P, L, U
============================================================================="""
    
import numpy  as np
import random       
from scipy.linalg import lu

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

## a) Tạo ma trận A(4 x 4) chứa các giá trị ngẫu nhiên trong khoảng 1 - 10
## Ma trận HÌNH CHỮ NHẬT: A(m, n) = P_T(m, m).L(m, n).U(n, n)
m, n, min, max = 4, 4, 1, 10
A = create_matrix_random(m, n, min, max)
print('Ma trận VUÔNG A', A.shape, ':\n', A)

## b) Phân tích thành các thành phần P, L, U. In kết quả
print('\n*** Áp dụng phân rã PLU:')
P_T, L, U = lu(A)

print('Ma trận P_T', P_T.shape, ':\n', P_T)
print('Ma trận L', L.shape, ':\n', L)
print('Ma trận U', U.shape, ':\n', U)

print('\nTái tạo A từ P, L, U (kiểm chứng lại phép phân rã):\n', P_T.dot(L).dot(U), '\n')

##------------------------------------------------------------------------------
## Mở rộng cho ma trận hình chữ nhật
##------------------------------------------------------------------------------
A = np.array([[1, 2,  4, 3], 
              [3, 8, 14, 0],
              [2, 6, 13, 2]])
P_T, L, U = lu(A)

print('Ma trận HÌNH CHỮ NHẬT A\n', A.shape, A)
print('\n*** Áp dụng phân rã PLU')
print('Ma trận P_T', P_T.shape, ':\n', P_T)
print('Ma trận L', L.shape, ':\n', L)
print('Ma trận U', U.shape, ':\n', U)
print('\nTái tạo A từ P, L, U (kiểm chứng lại phép phân rã):\n', P_T.dot(L).dot(U), '\n')


