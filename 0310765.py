# DataScience hw3 – Dimension Reduction

import sys
import os

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file

def get_data(INPUT_FILE):
    data = load_svmlight_file(INPUT_FILE)
    return data[0], data[1]

def PCA(X):
    # give mean value to non value element
    col_del = []
    for i in range(X[0].size):
        col = X[:,i]
        unique, counts = np.unique(col, return_counts=True)
        #print(unique.size)
        if unique.size==1:
            col_del.append(i)
        if unique.size!=2:
            mean = np.mean(col)
            for j in range(col.size):
                if(col[j]==0):
                    col[j] = mean
    X = np.delete(X,col_del,1)

    # Calaulate Covarianve matix
    Cov = np.cov(X.T)
    # Calaulate Eigenvalues and Eigenvectors of Covariance
    w,v = np.linalg.eigh(Cov.astype(np.float64))

    # sort eigen_pairs by eigenvalue
    eigen_pairs = [(w[i], v[:, i]) for i in range(len(w))]
    eigen_pairs.sort(key=lambda k: k[0], reverse=True)

    # decide the dimensions of reduction features
    eigen_mean = np.mean(w)*0.7
    for i in range(w.size):
        if(eigen_pairs[i][0]<eigen_mean):
            threshold = i
            break

    # extract target eigenvectors
    W = np.hstack((eigen_pairs[i][1][:, np.newaxis] for i in range(threshold)))

    z = X.dot(W)
    return z

if __name__ == '__main__':
    INPUT_FILE = sys.argv[1]
    dataset_name = os.path.splitext(INPUT_FILE)[0]
    X, y = get_data(INPUT_FILE)
    X = np.array(X.toarray(),dtype=np.float64) # transform to type array
    z = PCA(X)
    dump_svmlight_file(z,y,dataset_name+'_out.txt')
