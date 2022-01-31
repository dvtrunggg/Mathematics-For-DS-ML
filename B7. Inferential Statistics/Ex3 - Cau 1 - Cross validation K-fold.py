'''=============================================================================
Ex3: Hypothesis testing
    Câu 1: Cross Validation và K-fold
        Cho dữ liệu fruit_data_with_colors.txt
        a) Đọc dữ liệu vào dataframe.
        b) In thông tin shape, info, head
        c) Bỏ 2 cột là fruit_name và fruit_subtype trong dataframe.
           Tạo X (chứa các cột giá trị), y (là cột fruit_label)
        d) Áp dụng train/test split cho dữ liệu X, y theo tỷ lệ 70:30. 
           In các index của mỗi bộ (train, test).
        e) Áp dụng k-fold với n_splits = 10. In các index của mỗi bộ.
============================================================================='''
import pandas as pd
from sklearn.model_selection import train_test_split, KFold


print('=======================================================================')
print('*** a) Đọc dữ liệu vào dataframe.                                   ***')
print('=======================================================================')
folder = 'Data/Bai 7/'
data   = pd.read_csv(folder + 'fruit_data_with_colors.txt', sep='\t')

print('=======================================================================')
print('*** b) In thông tin shape, info, head.                              ***')
print('=======================================================================')
data.shape
data.info()
data.head()

print('=======================================================================')
print('*** c) Bỏ 2 cột là fruit_name và fruit_subtype trong dataframe.     ***')
print('***    Tạo X (chứa các cột giá trị), y (là cột fruit_label)         ***')                              
print('=======================================================================')
X = data[['mass', 'width', 'height', 'color_score']]
y = data['fruit_label']


print('=======================================================================')
print('*** d) Áp dụng train/test split cho X, y theo tỷ lệ 70:30.          ***')
print('***    In các index của mỗi bộ (train, test).                       ***')
print('=======================================================================')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
X_train.index
X_test.index

print('=======================================================================')
print('*** e) Áp dụng k-fold với n_splits = 10. In các index của mỗi bộ.   ***')
print('=======================================================================')
#
# scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
#    KFold(n_splits=5, *, shuffle=False, random_state=None)
#       shuffle: bool (default=False)
#                Whether to shuffle the data before splitting into batches. 
#                Note that the samples within each split will not be shuffled.
#       random_state: int or RandomState instance (default=None)
#                When shuffle is True, random_state affects the ordering 
#                of the indices, which controls the randomness of each fold.
#                Otherwise, this parameter has no effect.
#                Pass an int for reproducible output across multiple function calls.
#
cv = KFold(n_splits=10, shuffle=True, random_state=1) 
# random_state tương tự seed()
time = 0
for train_index, test_index in cv.split(X):
    time += 1
    print('*** Time:', time)
    print('   - Train Index: ', train_index.tolist())
    print('   - Test Index: ', test_index.tolist(), '\n')
    X_train, X_test= X.iloc[train_index.tolist()], X.iloc[test_index.tolist()]
    y_train, y_test = y.iloc[train_index.tolist()], y.iloc[test_index.tolist()]     
