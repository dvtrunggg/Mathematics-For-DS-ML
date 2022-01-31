'''=============================================================================
   1. Cho tập tin data.xlsx. Đọc dữ liệu
   2. Dữ liệu có 2 factors: Detergent (super, best), Temperature (hot, warm, cold).
   3. Sử dụng ANOVA hai chiều, đánh giá chất tẩy rửa và nhiệt độ ảnh hưởng 
      như thế nào đối với chất bẩn bị loại bỏ.
      a) Ảnh hưởng của chất tẩy rửa đến lượng chất bẩn bị loại bỏ 
      b) Ảnh hưởng của nhiệt độ đến lượng chất bẩn bị loại bỏ 
      c) Ảnh hưởng của chất tẩy rửa và nhiệt độ đến lượng chất bẩn bị loại bỏ
      H0D: Lượng chất bẩn bị loại bỏ không phụ thuộc vào loại chất tẩy rửa.
      H0T: Lượng chất bẩn bị loại bỏ không phụ thuộc vào nhiệt độ.
============================================================================='''
import pandas            as pd
import matplotlib.pyplot as plt
import scipy.stats       as stats
import seaborn           as sns
import statsmodels.api   as sm

from statsmodels.formula.api     import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# 1. Đọc tập tin dữ liệu.
folder = 'Data/Bai 7/'
data   = pd.read_excel(folder + 'data.xlsx')
print(data)

# Chuẩn bị dữ liệu theo statsmodels
d_melt = pd.melt(data, id_vars=['type'], value_vars=['cold', 'warm', 'hot'])

# Đổi tên các cột
d_melt.columns = ['type', 'temperature', 'value']
d_melt.head()

# Ordinary Least Squares (OLS) model
model  = ols('value ~ C(type) + C(temperature) + C(type):C(temperature)', data=d_melt).fit()

# 2. Vẽ boxplot, quan sát kết quả.
plt.figure(figsize = (12,10))
sns.boxplot(x='type', y='value', hue='temperature', data=d_melt, palette='Set3')
plt.autoscale(enable=True)
plt.show()

# 3. Áp dụng ANOVA.
# 3a) Kiểm định Levene: Các mẫu dữ liệu có cùng phương sai
print('-------------------------------------------------')
print('* Kiểm định LEVENE:                              ')
print('    H0: Các mẫu dữ liệu có phương sai BẰNG NHAU  ')
print('    Ha: Các phương sai KHÔNG BẰNG NHAU           ')
print('-------------------------------------------------')
levene, pvalue = stats.levene(data['hot'], data['warm'], data['cold'])
print('* Levene-statistic = %.4f, p-value = %.4f' % (levene, pvalue))
# p-value > alpha => không bác bỏ H0: Các mẫu dữ liệu có phương sai bằng nhau

# 3b) Two-way ANOVA
print('\n* Hàm anova_lm() tạo ANOVA table')
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table, '\n')

# Sự khác biệt về chất tẩy rửa và nhiệt độ có ý nghĩa thống kê,
# nhưng ANOVA không cho biết chất tẩy rửa và nhiệt độ khác nhau đáng kể với nhau. 
# Để biết các cặp chất tẩy rửa và nhiệt độ khác nhau đáng kể, 
# thực hiện nhiều phân tích so sánh cặp bằng cách sử dụng Tukey HSD test.

# 3c) Kiểm định Tukey HSD
m_comp = pairwise_tukeyhsd(endog = d_melt['value'], groups = d_melt['type'], alpha=0.05)
print(m_comp)

for name, grouped_df in d_melt.groupby('type'):
    print('type: {}'.format(name), pairwise_tukeyhsd(grouped_df['value'], 
                                                     grouped_df['temperature'], 
                                                     alpha=0.05))
for name, grouped_df in d_melt.groupby('temperature'):
   print('temperature: {}'.format(name), pairwise_tukeyhsd(grouped_df['value'], grouped_df['type'], alpha = 0.05))
    
