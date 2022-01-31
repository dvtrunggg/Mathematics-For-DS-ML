"""=============================================================================
Ex2: Correlation
    Cho dữ liệu baseball trong tập tin Master.csv.
    a) Đọc tập tin vào df => tạo hw từ df, chỉ lấy 2 cột height và weight. 
       Bỏ các dòng có dữ liệu null.
    b) Vẽ biểu đồ để xem xét tính tương quan.
    c) Vẽ boxplot để xác định và loại bỏ các outliers.
    d) Tính correlation của height và weight theo 2 cách Pearson và Spearsman.
============================================================================="""
import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import seaborn           as sns
import scipy.stats       as stats

print('=======================================================================')
print('*** a) Đọc tập tin, chỉ lấy height và weight, loại bỏ dữ liệu null. ***')
print('=======================================================================')
folder    = 'Data/Bai 7/'
df = data = pd.read_csv(folder + 'Master.csv')
df.head()

hw = df[["weight", "height"]].dropna()
hw.info()

print('=======================================================================')
print('*** b) Vẽ biểu đồ để xem xét tính tương quan.                       ***')
print('=======================================================================')
plt.figure(figsize=(8,8))
sns.jointplot(x='weight', y='height', data=hw)
plt.autoscale(enable=True)
plt.show()

print('=======================================================================')
print('*** c) Vẽ boxplot để xác định và loại bỏ các outliers.              ***')
print('=======================================================================')
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
sns.boxplot(data = hw.height)
plt.subplot(1,2,2)
sns.boxplot(data = hw.weight)
plt.autoscale(enable=True)
plt.show()

# IQR - Height
percentiles = np.array([25, 75])
x_h   = np.percentile(hw.height, percentiles)
IQR_h = stats.iqr(hw.height)
print(x_h, IQR_h)

# IQR - Weight
x_w   = np.percentile(hw.weight, percentiles)
IQR_w = stats.iqr(hw.weight)
print(x_w, IQR_w)

hw = hw[(hw.height >= (x_h[0] - 1.5*IQR_h)) & (hw.height <= (x_h[1] + 1.5*IQR_h))]
hw = hw[(hw.weight >= (x_w[0] - 1.5*IQR_w)) & (hw.weight <= (x_w[1] + 1.5*IQR_w))]
print(hw.shape)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
sns.boxplot(data = hw.height)
plt.subplot(1,2,2)
sns.boxplot(data = hw.weight)
plt.autoscale(enable=True)
plt.show()

print('=======================================================================')
print('*** d) Tính correlation (height, weight): Pearson và Spearsman.     ***')
print('=======================================================================')

# Pearson's correlation (default)
print('Pearson corr.  = %.4f' %hw['weight'].corr(hw['height']))

# Spearman's correlation
print('Spearman corr. = %.4f' %hw['weight'].corr(hw['height'], method='spearman'))