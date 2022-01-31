"""=============================================================================
Ex3: Hypothesis testing
    Câu 4: Chi-squared test

============================================================================="""
import scipy.stats as stats
from scipy.stats import chi2_contingency

table = [[ 6, 35, 15], 
         [ 7, 31,  6]]

print('-------------------------------------------')
print('Các giả thuyết kiểm định                   ')
print('    H0: Trình độ và giới tính là ĐỘC LẬP   ')
print('    Ha: Trình độ và giới tính là PHỤ THUỘC ')
print('-------------------------------------------')
alpha            = .05
confidence_level = (1 - alpha)

stat, p, df, expected = chi2_contingency(table)
print('statistic = %.3f' %stat)
print('p-value   = %.3f' %p)
print('df        = %d' %df)

##------------------------------------------------------------------------------
print('\n**** Phương pháp CRITICAL VALUE (giá trị tới hạn)')
##------------------------------------------------------------------------------    
df       = len(table[0]) - 1
critical = stats.chi2.ppf(confidence_level, df)
print('    - critical value = %.4f, statistic = %.4f' % (critical, stat))

if (abs(stat) >= critical):
    print('    Bác bỏ H0 ==> Trình độ và giới tính là PHỤ THUỘC')
else:
    print('    KHÔNG bác bỏ H0 ==> Trình độ và giới tính là ĐỘC LẬP')


##------------------------------------------------------------------------------
print('\n**** Phương pháp TRỊ SỐ p (p-value) ----')
##------------------------------------------------------------------------------    
print('    - alpha = %.2f, p = %.5f' % (alpha, p))

if (p <= alpha):
    print('    Bác bỏ H0 ==> Trình độ và giới tính là PHỤ THUỘC')
else:
    print('    KHÔNG bác bỏ H0 ==> Trình độ và giới tính là ĐỘC LẬP')


