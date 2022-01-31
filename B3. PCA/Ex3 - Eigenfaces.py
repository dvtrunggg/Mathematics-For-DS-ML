"""=============================================================================
Ex3: PCA - sklearn
    a) Load dữ liệu các khuôn mặt: 'sklearn.datasets.fetch_lfw_people'
       (lấy min_faces_per_person = 60)
    b) Thực hiện giảm chiều dữ liệu từ ~3000 xuống còn 150
    c) Trực quan hóa dữ liệu gốc và sau khi giảm chiều
============================================================================="""
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

##------------------------------------------------------------------------------
print('\n*** a) Load dữ liệu các khuôn mặt: sklearn.datasets.fetch_lfw_people:')
##------------------------------------------------------------------------------
from sklearn.datasets import fetch_lfw_people
m = 40       # 60
faces = fetch_lfw_people(min_faces_per_person = m)
type(faces)

print(faces.target_names)
print(faces.images.shape)

faces.images[0].shape
faces.data[0].size

##------------------------------------------------------------------------------
print('\n*** b) Thực hiện giảm chiều dữ liệu từ ~3000 xuống còn 150:')
##------------------------------------------------------------------------------
k = 40       # 150
pca = PCA(k)
pca.fit(faces.data)

# access values and vectors
# components_ : array, shape (n_components, n_features)
# Các trục chính trong không gian feature, biểu thị
# các hướng của phương sai tối đa trong dữ liệu
# explained_variance_ : array, shape (n_components,)
# Số lượng phương sai được giải thích bởi từng thành phần được chọn.
print('PCA.Components:\n', pca.components_)
print('PCA.Shape: ', pca.components_.shape)
print('PCA.Explained variance: ', pca.explained_variance_)
print('PCA.Explained variance shape: ', pca.explained_variance_.shape)
print(sum(pca.explained_variance_ratio_))

# Transform data
components = pca.transform(faces.data)
projected  = pca.inverse_transform(components)

##------------------------------------------------------------------------------
print('\n*** c) Trực quan hóa dữ liệu:')
##------------------------------------------------------------------------------
# Plot the results
fig, ax = plt.subplots(2, 10, figsize = (10, 2.5), 
                              subplot_kw  = {'xticks':[], 'yticks':[]},
                              gridspec_kw = dict(hspace = 0.1, wspace = 0.1))
for i in range(10):
   ax[0, i].imshow(faces.data[i].reshape(62, 47), cmap = 'binary_r')
   ax[1, i].imshow(projected[i].reshape(62, 47),  cmap='binary_r')
   ax[0, 0].set_ylabel('full-dim\ninput')
   ax[1, 0].set_ylabel('K-dim\nreconstruction')