import numpy as np
import matplotlib.pyplot as plt
import os
import pca

file_list = None
original_shape = None
main_path = os.getcwd()

def compress_images(DATA, k):
        # standardize data
        Z = pca.compute_Z(DATA, centering=True, scailing=True)
        # compute covariance matrix
        COV = pca.compute_covariance_matrix(Z)
        # find eigenvectors and eigenvalues
        L, PCS = pca.find_pcs(COV)
        # project data onto k components
        Z_star = pca.project_data(Z, PCS, L, k, 0)
        # reproduce the images by multiplying by k first eigenvectors
        x_compressed = np.dot(Z_star, PCS[:,:k].T)
        # normalize data to be within 0-255 int range
        X_norm = (255*(x_compressed - x_compressed.min(axis=0)) / x_compressed.ptp(axis=0)).astype(np.int)
        # load images back to files located in Data\Output
        directory = "Data/Output"
        dir_path = os.path.join(main_path, directory)
        if not os.path.exists(dir_path):
                os.mkdir(dir_path)       
        os.chdir(dir_path)
        for i, image_name in enumerate(file_list):
                image_name = image_name.replace(".pgm", '.png')
                plt.imsave(image_name, X_norm[:,i].reshape(original_shape, order='F'), cmap = 'gray')        


def load_data(input_dir):
        # get image folder path
        dir_path = os.path.join(main_path, input_dir)
        # collect the filenames for output purposes
        global file_list
        file_list = os.listdir(dir_path)
        # store each vector image in the array
        data = []
        # read in each file
        for file in os.listdir(dir_path):
                with open(os.path.join(dir_path, file), 'rb') as pgmf:
                        # read in an image as numpy 2D array
                        im = plt.imread(pgmf)
                        #save original shape
                        global original_shape
                        original_shape = im.shape
                        # stack columns together
                        y = im.T.flatten()
                        # add to data
                        data.append(y.astype(np.float64))
        # stack all images together (data.shape: (2880, 869))
        data = np.stack(data, axis=1)
        # data is now the matrix where each column is an image and each row is corresponding pixel value
        return data
