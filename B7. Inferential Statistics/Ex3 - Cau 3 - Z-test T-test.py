"""=============================================================================
Ex3: Hypothesis testing
    Câu 3: P-test và T-test
        Cho 2 bộ dữ liệu phụ thuộc nhau như sau:
            np.random.seed(11)
            before = stats.norm.rvs(scale=30, loc=250, size=100)
            after  = before + stats.norm.rvs(scale=5, loc=-1.25, size=100) 
        a) Tạo dataframe chứa before, after, và change = after - before.
        b) Áp dụng t-test để kiểm định H0: 'The mean are equal', với alpha = 0.05

============================================================================="""
import numpy       as np
import pandas      as pd
import scipy.stats as stats

print('=======================================================================')
print('*** a) Đọc dữ liệu.                                                 ***')
print('=======================================================================')
np.random.seed(11)
# Dữ liệu tại thời điểm t(i)
before = stats.norm.rvs(scale=30, loc=250, size=100)
# Dữ liệu tại thời điểm t(i+1)
after  = before + stats.norm.rvs(scale=5, loc=-1.25, size=100)
df     = pd.DataFrame({"before":before, "after":after, "change":after-before})
print(df.head())
print('Số liệu thống kê:\n', df.describe())

print('------------------------------------------')
print('Các giả thuyết kiểm định                  ')
print('    H0: Mean_1 = Mean_2                   ')
print('    Ha: Mean_1 <> Mean_2                  ')
print('------------------------------------------')
alpha            = .05
confidence_level = 1 - alpha

t, p = stats.ttest_rel(df.before, df.after)

##------------------------------------------------------------------------------
print('\n**** Phương pháp CRITICAL VALUE (giá trị tới hạn)')
##------------------------------------------------------------------------------    
df       = len(df.before) - 1
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


