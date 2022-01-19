import numpy as np 
import matplotlib.pyplot as plot

def compute_Z(X, centering=True, scailing=False):
        if centering:
                X = X - X.mean(axis=0)
        if scailing:
                X = X / X.std(axis=0)
        return X

def compute_covariance_matrix(Z):
        return np.cov(Z, rowvar=False)

def find_pcs(COV):
        # find eigenvectors and eigenvalues
        eigenVal, eigenVec = np.linalg.eig(COV)
        # make sure the eigenvalues are in descending order
        order = np.argsort(eigenVal)[::-1]
        # return reordered eigenvalues-eigenvectors pairs
        return eigenVal[order], eigenVec[:,order]

def project_data(Z, PCS, L, k, var):
        # make sure the eigenvectors are of unit length
        if (np.linalg.norm(PCS, axis=0)).all() == 1:
                # when var is not 0 (assume k is 0 here)
                if var != 0:
                        needed_eigenvalues_count = 0
                        target_val = L.sum() * var
                        total_sum = 0
                        for val in L:
                                if total_sum >= target_val:
                                        break
                                total_sum += val
                                needed_eigenvalues_count += 1
                        return np.dot(Z, PCS[:,:needed_eigenvalues_count])
                # when k > 0 (assume var is 0)
                else:
                        # use k principal components
                        return np.dot(Z, PCS[:,:k])
