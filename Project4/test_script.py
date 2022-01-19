import pca
import compress
import numpy as np
from sklearn import decomposition

#test PCA
X = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
Z = pca.compute_Z(X)
COV = pca.compute_covariance_matrix(Z)
L, PCS = pca.find_pcs(COV)
Z_star = pca.project_data(Z, PCS, L, 0, 0.5)

#test compression  using PCA
import compress

X = compress.load_data('Data/Train/') # X shape (2880, 869)
compress.compress_images(X,500)
