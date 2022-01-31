"""=============================================================================
Ex2: PCA - sklearn --> mở rộng thêm phân tích phương sai để xác định k
    a) Đọc tập tin dữ liệu Classification_12f_C.xls vào dataframe.
    b) Áp dụng phương pháp PCA để giảm xuống k chiều (2 < k).
       Giải thích nguyên nhân hay cơ sở về số chiều được giảm.
    c) Giảm chiều xuống còn k = 2 và trực quan hóa dữ liệu. Nhận xét kết quả.
    d) CHUẨN HÓA dữ liệu, sau đó thực hiện PCA.
============================================================================="""
#%%
import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import seaborn           as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print('\n=============================================================')
print('*** a) Đọc tập tin dữ liệu vào dataframe                  ***')
print('=============================================================')
folder = '../Data/Bai 3/'
data   = pd.read_excel(folder + 'Students_12f_C.xls')

## Biến phân lớp (target variable): 'Class' --> cột cuối cùng trong file
target = 'Class'
print('* Biến phân lớp:', target)

## Danh sách các features 
nb_features = data.shape[1] - 1
features    = data.columns[:nb_features]
print('* Số lượng features = %2d' %nb_features)
print('  Các features:', ', '.join(features)) 


print('\n=============================================================')
print('*** b) Áp dụng PCA để giảm xuống còn k chiều (2 < k)      ***')
print('=============================================================')
#   https://stackoverflow.com/questions/32857029/python-scikit-learn-pca-explained-variance-ratio-cutoff
#   - The pca.explained_variance_ratio_ returns a vector of the variance explained by each dimension.
#   - The pca.explained_variance_ratio_[i] gives the variance explained solely by the i+1st dimension.
#   - The pca.explained_variance_ratio_.cumsum() will return a vector x 
#     such that x[i] returns the cumulative variance explained by the first i+1 dimensions.

#   (1) PCA().components_: Chuyển vị của ma trận vectơ riêng EigenVectors.T
#   (2) PCA().explained_variance_: Các giá trị riêng
#   (3) PCA().explained_variance_ratio_: Tỷ lệ phương sai so với dữ liệu gốc
#   (4) Hàm numpy.cumsum()

print('-------------------------------------------------------------')
print('CÁCH 1: Chọn k dựa trên đồ thị biểu diễn phương sai tích lũy ')
print('-------------------------------------------------------------')
pca = PCA().fit(data)

## Vẽ đồ thị biểu diễn % phương sai tích lũy theo số features
print('Phương pháp ELBOW: Chọn k theo điểm gẫy trên đường cong')

# Các điểm dữ liệu
points = np.cumsum(pca.explained_variance_ratio_)
x_i    = np.arange(0, nb_features)
y_i    = (points[-12:])//0.0001/10000

plt.figure(figsize = (10, 8))
plt.plot(points, marker = 'o')
plt.xlabel('Số features (k)')
plt.ylabel('Variance (%)')
plt.title('Đồ thị biểu diễn % phương sai tích lũy theo số features (k)')
plt.xlim([0, nb_features])
plt.grid(axis = 'x')
for i in x_i:
    plt.text(i, y_i[i] + 0.0001, y_i[i])
    
plt.show()

# Kiểm chứng: Tính phương sai tích lũy theo k
var = 0.0
for k in range(1, nb_features + 1):
    pca = PCA(k)
    pca.fit(data)
      
    newVar = pca.explained_variance_ratio_.sum() * 100
    print('   * k = %2d' %k, '--> phương sai tích lũy ~ %.2f%%' %newVar,
          ', độ tăng ~ %.2f%%' %(newVar - var))
    var = newVar


print('-------------------------------------------------------------')
print('CÁCH 2: Chọn k dựa trên ngưỡng phương sai tích lũy mong muốn ')
print('-------------------------------------------------------------')
threshold = .90
percent   = threshold * 100

## Chọn giá trị k
pca = PCA(threshold)
pca.fit_transform(data) 

k   = pca.n_components_
var = sum(pca.explained_variance_ratio_) * 100
print('   * Muốn phương sai tích lũy >= %.2f%%' %percent, 'thì k >= %d' %k, '--> %.2f%%' %var, '\n')

print('   * Kiểm chứng: Phân tích chi tiết theo các ngưỡng phương sai')
A = np.array([.5, .6, .7, .8, .9, .95, .99])
for x in A:
    percent = x * 100
    pca     = PCA(x)

    pca.fit(data)
    k   = pca.n_components_
    var = sum(pca.explained_variance_ratio_) * 100
    print('      - Muốn phương sai tích lũy >= %.2f%%' %percent, 'thì k >= %2d' %k,
          '(var ~ %.2f%%)' %var)

print('\n=============================================================')
print('*** c) Giảm chiều còn k = 2 và trực quan hóa dữ liệu      ***')
print('=============================================================')
k   = 2
pca = PCA(k)
pca.fit(data)

## Gán tên cho các Principal Components
PC_name  = ['Principal Component 1', 'Principal Component 2']

## Transform data
B           = pca.transform(data)
principalDf = pd.DataFrame(data = B, columns = PC_name)

# Trực quan hóa dữ liệu (KHÔNG phân lớp)
plt.figure(figsize = (8, 8))
sns.jointplot(x = PC_name[0], y = PC_name[1], data = principalDf)              
plt.show()

# Lấy cột phân lớp (Class) trong file dữ liệu
y = np.array(data.Class)
y = pd.DataFrame(data = y, columns = [target])

# Ghép cột phân lớp (Class) vào ma trận PCA
finalDf = pd.concat([principalDf, y], axis = 1)
print('\n* Ma trận B_T (có thêm biến phân lớp Class)')
print(finalDf.head(), '\n')

# Trực quan hóa dữ liệu (có PHÂN LỚP)
plt.figure(figsize = (8, 8))
plt.title('Biểu đồ có PHÂN LỚP')
sns.scatterplot(x = PC_name[0], y = PC_name[1], data = finalDf, hue = target, legend = 'full')              
plt.show()
 

print('\n========================================================')
print('*** d) CHUẨN HÓA dữ liệu, sau đó thực hiện PCA       ***')
print('========================================================')
pca_norm  = PCA(k)
data_norm = StandardScaler().fit_transform(data)
pca_norm.fit(data_norm)

# Transform data
B_norm           = pca_norm.transform(data_norm)
principalDf_norm = pd.DataFrame(data = B_norm, columns = PC_name)

# Lấy cột phân lớp (Class) trong file dữ liệu
y = np.array(data.Class)
y = pd.DataFrame(data = y, columns = [target])

# Ghép cột phân lớp (Class) vào ma trận PCA
finalDf_norm = pd.concat([principalDf_norm, y], axis = 1)

# Trực quan hóa dữ liệu (có PHÂN LỚP)
plt.figure(figsize = (8, 8))
plt.title('Biểu đồ sau khi CHUẨN HÓA dữ liệu')
sns.scatterplot(x = PC_name[0], y = PC_name[1], data = finalDf_norm, hue = target, legend = 'full')
plt.show()
