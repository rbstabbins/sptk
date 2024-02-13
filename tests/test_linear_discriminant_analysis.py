"""Unit tests for the linear_discriminant_analysis module of the
spectral parameters toolkit.

Part of the Spectral Parameters Toolkit
Author: Roger Stabbins, NHM
Date: 28-09-2022
"""
import unittest
import numpy as np
import pandas as pd
from scipy import linalg
from sklearn import datasets
import sptk.linear_discriminant_analysis as lda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import _cov

def load_test_data():
    """_summary_
    """
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

    return dataset, categories, class_labels

class TestLinearDiscriminantAnalysis(unittest.TestCase):
    """Class for testing the functions/methods of the Linear Discriminant
    Analysis module. Uses the iris dataset and results of the scikit_learn
    LDA method. Note scikit_learn method is not parallelised, hence not used
    in sptk.
    """
    def test_within_class_scatter_matrix(self):
        """Testing the within_class_scatter_matrix method.
        """
        dataset, categories, _ = load_test_data()

        lda_scikit = LDA(solver='eigen')
        lda_scikit.fit(dataset, categories)
        n_samples = len(dataset)
        expected = np.array([lda_scikit.covariance_ * n_samples])
        # test within class scatter
        result = lda.within_class_scatter_matrix(dataset,categories)
        np.testing.assert_array_almost_equal(result, expected)

    def test_between_class_scatter_matrix(self):
        """Testing the between_class_scatter_matrix method.
        """
        dataset, categories, _ = load_test_data()

        lda_scikit = LDA(solver='eigen')
        lda_scikit.fit(dataset, categories)
        n_samples = len(dataset)

        Sw = lda_scikit.covariance_ * n_samples  # within scatter
        St = _cov(dataset, None) * n_samples  # total scatter

        expected = np.array([St - Sw])
        # test within class scatter
        result = lda.between_class_scatter_matrix(dataset, categories)
        np.testing.assert_array_almost_equal(result, expected)

    def test_projection_matrix(self):
        """Testing the projection_matrix method.
        """
        dataset, categories, class_labels = load_test_data()
        n_samples = len(dataset)

        lda_scikit = LDA(solver='eigen')
        lda_scikit.fit(dataset, categories)
        Sw = lda_scikit.covariance_ * n_samples  # within scatter
        St = _cov(dataset, None) * n_samples  # total scatter
        Sb = St - Sw  # between scatter
        evals, evecs = linalg.eigh(Sb, Sw)
        evecs = evecs[:, np.argsort(evals)[::-1]]
        evals = evals[np.argsort(evals)[::-1]]

        n_c = len(class_labels)
        wcsm = lda.within_class_scatter_matrix(dataset, categories)
        bcsm = lda.between_class_scatter_matrix(dataset, categories)        
        A, tst_eig_vcs, tst_eig_vls = lda.projection_matrix(wcsm, bcsm, n_c)

        with self.subTest('valid eigen-vectors'):
            for i in range(0, dataset.shape[1]):
                scatter = np.linalg.inv(wcsm[0]).dot(bcsm[0])
                expected = scatter.dot(tst_eig_vcs[0][:, i])
                result = tst_eig_vls[0][i] * tst_eig_vcs[0][:, i]
                np.testing.assert_array_almost_equal(result, expected)
        # with self.subTest('projection'):
        #     result = A
        #     expected = lda_scikit.scalings_
        #     np.testing.assert_array_almost_equal(result, expected)

    def test_fisher_ratio(self):
        pass

    def test_project_data(self):
        pass

    def test_compute_lda_boundaries(self):
        pass