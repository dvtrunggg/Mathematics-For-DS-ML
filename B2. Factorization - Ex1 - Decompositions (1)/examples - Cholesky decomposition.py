'''=============================================================================
   a) Slide #14 
      A = L.LT = UT.U
         L: ma trận tam giác DƯỚI, khả nghịch, Lii > 0
         U: ma trận tam giác TRÊN, khả nghịch, Uii > 0
    b) Kiểm tra tính chất POSITIVE DEFINITE, POSITIVE SEMIDEFINITE
============================================================================='''
import numpy as np

from numpy        import linalg   as la
from scipy.linalg import cholesky

# from numpy.linalg import cholesky: Default --> LOWER


##------------------------------------------------------------------------------
## Positive definite (SYMETRIC) matrix
##------------------------------------------------------------------------------
A = np.array([[6, 1, 1], 
              [1, 6, 1], 
              [1, 1, 6]])
print('\n*** Ma trận ĐỐI XỨNG:\n', A)

print('Áp dụng phân rã Cholesky:')

## Cholesky decomposition: Default --> UPPER (U)
U = cholesky(A)
print('\nUPPER   Cholesky factor:\n', U)
print('\nTái tạo A từ U, U_T (kiểm chứng lại phép phân rã)\n', U.T.dot(U))

U = cholesky(A, lower = False)
print('\nUPPER   Cholesky factor:\n', U)

L = cholesky(A, lower = True)
print('\nLOWER   Cholesky factor:\n', L)
print('\nTái tạo A từ L, L_T (kiểm chứng lại phép phân rã)\n', L.dot(L.T))


##------------------------------------------------------------------------------
## Positive definite (NON-SYMETRIC) matrix
##------------------------------------------------------------------------------
A = np.array([[  4,  12, -15], 
              [ 12,  37, -42], 
              [-16, -43,  98]])

print('\n*** Ma trận KHÔNG ĐỐI XỨNG:\n', A)

# Cholesky decomposition
print('Áp dụng phân rã Cholesky:')
U = cholesky(A)
print('UPPER   Cholesky factor:\n', U)
print('\nTái tạo A từ U, U_T (kiểm chứng lại phép phân rã)\n', U.T.dot(U))

U = cholesky(A, lower = False)
print('UPPER   Cholesky factor:\n', U)

L = cholesky(A, lower = True)
print('LOWER   Cholesky factor:\n', L)
print('\nTái tạo A từ L, L_T (kiểm chứng lại phép phân rã)\n', L.dot(L.T))


##------------------------------------------------------------------------------
##   Kiểm tra tính chất POSITIVE DEFINITE của ma trận A
##------------------------------------------------------------------------------
# Positive DEFINITE (symmetric) matrix
A = np.array([[9, 6], 
              [6, 5]])

# Eigenvalues
eigenValues, eigenVectors = la.eig(A)
pos_def = np.all(eigenValues > 0)
if (pos_def == True):
    print('\n*** Ma trận', A, '\nlà ma trận xác định dương.')
    print('Áp dụng phân rã Cholesky:')
    print('UPPER Cholesky factor:\n', cholesky(A))
    print('UPPER Cholesky factor:\n', cholesky(A, check_finite = True))
else:
    print('\n*** Ma trận', A, '\nKHÔNG phải là ma trận xác định dương.')
    
# Positive SEMIDEFINITE (symmetric) matrix
#    A positive semidefinite matrix is a Hermitian matrix all of whose eigenvalues
#    are nonnegative.
A = np.array([[9, 6], [6, 4]])

# Eigenvalues
eigenValues, eigenVectors = la.eig(A)
pos_def = np.all(eigenValues > 0)
if (pos_def == True):
    print('\n*** Ma trận', A, '\nlà ma trận xác định dương.')
    print('Áp dụng phân rã Cholesky:')
    print('UPPER Cholesky factor:\n', cholesky(A, check_finite = True))
else:
    print('\n*** Ma trận', A, '\nKHÔNG phải là ma trận xác định dương.')
    
# NOT Positive SEMIDEFINITE (symmetric) matrix
A = np.array([[9, 6], [6, 3]])

# Eigenvalues
eigenValues, eigenVectors = la.eig(A)
pos_def = np.all(eigenValues > 0)
if (pos_def == True):
    print('\n*** Ma trận', A, '\nlà ma trận xác định dương.')
    print('Áp dụng phân rã Cholesky:')
    print('UPPER Cholesky factor:\n', cholesky(A, check_finite = True))
else:
    print('\n*** Ma trận', A, '\nKHÔNG phải là ma trận xác định dương.')
