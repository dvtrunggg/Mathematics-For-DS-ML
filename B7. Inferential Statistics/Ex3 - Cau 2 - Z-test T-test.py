'''=============================================================================
Ex3: Hypothesis testing
    Câu 2: P-test và T-test
        Cho 2 bộ dữ liệu life_battery Chapter 5. Hai bộ dữ liệu này độc lập nhau.
        Đọc dữ liệu và gán cho 2 biến là life1_array và life2_array.
        Áp dụng t-test để kiểm định H0: 'The mean are equal', với alpha = 0.05
============================================================================='''
import numpy as np
import scipy.stats as stats

print('=======================================================================')
print('*** a) Đọc dữ liệu.                                                 ***')
print('=======================================================================')
folder  = 'Data/Bai 5/'

# Tập tin 1
f       = open(folder + 'life_batteries.txt', 'r')
content = f.read()
f.close()
life1 = content.split()
life1 = list(map(int, life1))
life1_array = np.array(life1)
print(life1_array)

# Tập tin 2
f       = open(folder + 'life_batteries_2.txt', 'r')
content = f.read()
f.close()
life2 = content.split()
life2 = list(map(int, life2))
life2_array = np.array(life2)
print(life2_array)

print('------------------------------------------')
print('Các giả thuyết kiểm định                  ')
print('    H0: Mean_1 = Mean_2                   ')
print('    Ha: Mean_1 <> Mean_2                  ')
print('------------------------------------------')
alpha            = .05
confidence_level = 1 - alpha

t, p = stats.ttest_ind(life1_array, life2_array)

##------------------------------------------------------------------------------
print('\n**** Phương pháp CRITICAL VALUE (giá trị tới hạn)')
##------------------------------------------------------------------------------    
df       = len(life1_array) - 1
critical = stats.t.ppf(confidence_level, df)
print('    - critical value = %.4f, statistic = %.4f' % (critical, t))

if (abs(t) >= critical):
    print('    Bác bỏ H0 ==> Mean_1 <> Mean_2')
else:
    print('    KHÔNG bác bỏ H0 ==> Mean_1 = Mean_2')


##------------------------------------------------------------------------------
print('\n**** Phương pháp TRỊ SỐ p (p-value) ----')
##------------------------------------------------------------------------------    
print('    - alpha = %.2f, p = %.5f' % (alpha, p))

if (p <= alpha):
    print('    Bác bỏ H0 ==> Mean_1 <> Mean_2')
else:
    print('    KHÔNG bác bỏ H0 ==> Mean_1 = Mean_2')


