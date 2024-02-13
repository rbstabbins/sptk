# testing the LDA method, using the IRIS dataset,
# with comparison to results computed in this tutorial:
# https://sebastianraschka.com/Articles/2014_python_lda.html
# load in IRIS dataset
# get the iris data set
from sklearn import datasets
import pandas as pd
import numpy as np
import sptk.linear_discriminant_analysis as lda

iris = datasets.load_iris()
# put dataset in a pandas dataframe
df = pd.DataFrame(
        data=np.c_[iris.data, iris.target.astype(int)],
        columns=iris.feature_names + ['class label'])
label_dict = {0: 'Setosa', 1: 'Versicolor', 2:'Virginica'}
class_labels = list(label_dict.values())
df = df.replace({'class label': label_dict})

# format IRIS dataset to [n_entries, n_features] numpy array,
# with accompanying list of classes given by 'categories'
dataset = df[iris.feature_names].to_numpy()
n_features = len(iris.feature_names)
categories = df['class label']

# test within class scatter
S_W = lda.within_class_scatter_matrix(dataset, categories, n_features)
print('Within-Class Scatter Matrix: ')
print(S_W.squeeze())
# expected within-class scatter matrix
print("Expected Within-Class Scatter Matrix:")
print("[[ 38.9562  13.683   24.614    5.6556]")
print("[ 13.683   17.035    8.12     4.9132]")
print("[ 24.614    8.12    27.22     6.2536]")
print("[  5.6556   4.9132   6.2536   6.1756]]")

# test between class scatter
print('Between-Class Scatter Matrix: ')
S_B = lda.between_class_scatter_matrix(dataset, categories, n_features)
print(S_B)
print("Expected Between-Class Scatter Matrix:")
print("[[  63.2121  -19.534   165.1647   71.3631]")
print("[ -19.534    10.9776  -56.0552  -22.4924]")
print("[ 165.1647  -56.0552  436.6437  186.9081]")
print("[  71.3631  -22.4924  186.9081   80.6041]]")

# test computation of projection matrix
tst_n_classes = len(class_labels)
A, tst_eig_vecs, tst_eig_vals = lda.projection_matrix(S_W, S_B, tst_n_classes)
print('Eigenvectors: ')
print(tst_eig_vecs)
print('Eigenvalues: ')
print(tst_eig_vals)
print('Expected Eigenvectors and Eigenvalues:')
print("Eigenvector 1:")
print("[[-0.2049]")
print("[-0.3871]")
print("[ 0.5465]")
print("[ 0.7138]]")
print("Eigenvalue 1: 3.23e+01")

print("Eigenvector 2:")
print("[[-0.009 ]")
print("[-0.589 ]")
print("[ 0.2543]")
print("[-0.767 ]]")
print("Eigenvalue 2: 2.78e-01")

print("Eigenvector 3:")
print("[[ 0.179 ]")
print("[-0.3178]")
print("[-0.3658]")
print("[ 0.6011]]")
print("Eigenvalue 3: -4.02e-17")

print("Eigenvector 4:")
print("[[ 0.179 ]")
print("[-0.3178]")
print("[-0.3658]")
print("[ 0.6011]]")
print("Eigenvalue 4: -4.02e-17")

print('Projection Matrix: ')
print(A)
print("Expected Projection Matrix: ")
print("[[-0.2049 -0.009 ]")
print("[-0.3871 -0.589 ]")
print("[ 0.5465  0.2543]")
print("[ 0.7138 -0.767 ]]")
# test computation of score
tst_score = lda.fisher_ratio(A, S_B, S_W)
print('Score: ')
print(tst_score)
