"""=============================================================================
Ex1:
    Câu 1: SEM 
        1) Tính SEM của sample distribution với std = 38 và sample size là 45.
        2) Tính SEM của dữ liệu sau: 14, 8, 11, 12, 16, 10
        3) Một số điểm IQ như sau: 96, 104, 126, 134, 140. Tính SEM.
============================================================================="""
# Tham khảo: https://math.tutorvista.com/algebra/standard-error-of-the-mean.html
import numpy  as np

print('=======================================================================')
print('*** 1) Tính SEM của sample distribution: std = 38, sample size = 45.***')
print('=======================================================================')
std = 38
n   = 45
SEM = std / (n**0.5)
print('SEM = %f \n' %SEM)

print('=======================================================================')
print('*** 2) Tính SEM của dữ liệu sau: 14, 8, 11, 12, 16, 10.             ***')
print('=======================================================================')
data = np.array([14, 8, 11, 12, 16, 10])
std  = np.std(data)

SEM = std / (data.size**0.5)
print('SEM = %f \n' %SEM)
      
print('=======================================================================')
print('*** 3) Một số điểm IQ như sau: 96, 104, 126, 134, 140. Tính SEM.    ***')
print('=======================================================================')
IQ  = np.array([96, 104, 126, 134, 140])
std = np.std(IQ)

SEM = std / (data.size**0.5)
print('SEM = %f \n' %SEM)
      
