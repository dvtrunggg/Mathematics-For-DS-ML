"""=============================================================================
   Sở GTCC muốn kiểm tra sự an toàn của các xe nhỏ, hạng trung và cỡ lớn. 
    1. Tạo dataframe như hình vẽ.
    2. Vẽ boxplot, quan sát kết quả.
    3. Áp dụng ANOVA để xem có sự khác biệt đáng kể giữa 3 loại xe (alpha = 5%).
============================================================================="""
import matplotlib.pyplot as plt
import pandas            as pd
import scipy.stats       as stats
import seaborn           as sns
import statsmodels.api   as sm

from statsmodels.formula.api     import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

alpha      = .05
confidence = (1 - alpha)

# 1. Tạo dataframe 
df = pd.DataFrame({'S_cars': [643, 655, 702],
                   'M_cars': [469, 427, 525],
                   'X_cars': [484, 456, 402]})
print(df.loc[[0]])

# 2. Vẽ boxplot, quan sát kết quả.
sns.boxplot(data=df)
plt.autoscale(enable=True)
plt.show()

# 3. Áp dụng ANOVA
# 3a) Chuẩn bị dữ liệu theo statsmodels
df_melt = pd.melt(df.reset_index(), id_vars = ['index'], 
                  value_vars = ['S_cars', 'M_cars', 'X_cars'])

# Đổi tên các cột
df_melt.columns = ['index', 'cars', 'value']

# Ordinary Least Squares (OLS) model
model = ols('value ~ C(cars)', data = df_melt).fit()

# 3b) Kiểm định Levene: S_cars, M_cars, X_cars có cùng phương sai
print('-------------------------------------------------')
print('* Kiểm định LEVENE:                              ')
print('    H0: VAR(S_cars) = VAR(M_cars) = VAR(X_cars)  ')
print('    Ha: Các phương sai KHÔNG BẰNG NHAU           ')
print('-------------------------------------------------')
levene, pvalue = stats.levene(df.S_cars, df.M_cars, df.X_cars)
print('* Levene-statistic = %.4f, p-value = %.4f' % (levene, pvalue))
# p-value > alpha => không bác bỏ H0: VAR(S_cars) = VAR(M_cars) = VAR(X_cars)

# 3c) Kiểm định Shapiro: S_cars, M_cars, X_cars có pp chuẩn
print('--------------------------------------------------')
print('* Kiểm định SHAPIRO                               ')
print('    H0: S_cars, M_cars, X_cars ~ Norm(Muy, Sigma) ')
print('    Ha: S_cars, M_cars, X_cars KHÔNG pp chuẩn     ')
print('--------------------------------------------------')
shapiro, pvalue = stats.shapiro(model.resid)
print('* Shapiro-statistic = %.4f, p-value = %.4f' % (shapiro, pvalue))
# p-value > alpha => không bác bỏ H0: dữ liệu được rút ra từ phân phối chuẩn.

# 3d) One-way ANOVA
print('\n* Hàm f_oneway() chỉ trả về F-statistic và p-value; KHÔNG tạo ANOVA table')
fvalue, pvalue = stats.f_oneway(df.S_cars, df.M_cars, df.X_cars)
print('   F-statistic = %.4f, p-value = %.4f' %(fvalue, pvalue))
# Giá trị P-value có ý nghĩa về mặt thống kê (P < 0.05),
# do đó, có thể kết luận rằng có sự khác biệt đáng kể giữa các loại xe.

print('\n* Hàm anova_lm() tạo ANOVA table')
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table, '\n')

# 3e) Kiểm định Tukey HSD
m_comp = pairwise_tukeyhsd(endog = df_melt['value'], groups = df_melt['cars'], alpha = 0.05)
print(m_comp)
# ngoại trừ X_cars và M_cars, tất cả các so sánh cặp khác đều bác bỏ H0
# và chỉ ra sự khác biệt đáng kể về mặt thống kê.


