# -*- coding: utf-8 -*-
"""=============================================================================
    Minh họa các norms của vectơ và ma trận
============================================================================="""

import numpy as np
import scipy.spatial.distance as dist

from numpy.linalg import norm
    
v = np.array([1, 3, -4, 5, 7, 9, -6])
print('v =', v, '\n')

u = np.zeros(7)

## L1 norm (Manhattan/City block distance)
print('L1 norm            =', norm(v, 1))
print('Manhattan diatance =', dist.cityblock(u, v), '\n')

## L2 norm (Euclidean distance)
print('L2 norm            =', norm(v, 2))
print('L2 norm            =', norm(v))
print('Euclidean distance =', dist.euclidean(u, v), '\n')

## Lmax norm (Chebyshev distance)
print('Lmax norm          =', norm(v, np.inf))
print('Chebyshev distance =', dist.chebyshev(u, v), '\n')

## p-norm (Minskowski distance)
print('p-norm ( p =', 1, ')   =', dist.minkowski(u, v, 1))
print('p-norm ( p =', 2, ')   =', dist.minkowski(u, v, 2))
print('p-norm ( p =', 3, ')   =', dist.minkowski(u, v, 3))
print('p-norm ( p =', 4, ')   =', dist.minkowski(u, v, 4), '\n')

## Frobenius matrix norm
A = np.array([[1, 3, 2, 4], [4, 3, 6, 8], [5, 6, 3, 1]])
print('\nMa trận A', A.shape, '\n', A)
print('Frobenius norm =', norm(A, 'fro'))
print('Frobenius norm =', norm(A))


