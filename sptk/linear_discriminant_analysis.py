"""Linear Discriminant Analysis Class

Perform key steps of Linear Discriminant Analysis on multiple combinations of
spectral parameter values in parallel.

TODO cythonize this file

Part of the Spectral Parameters Toolkit
Author: Roger Stabbins, NHM
Date: 25-05-2021
"""
from typing import Tuple
import numpy as np
from scipy import linalg
import pandas as pd

def within_class_scatter_matrix(
        k_stack: np.array,
        category_list: pd.Series) -> np.array:
    """Computes the total Within-Category Scatter Matrix S_W, on the dataset
    provided by k_stack, according to category labels.

    Within-category scatter defined as:
        $S_W = Sum(S_c)$
    where S_c is the Within-Class Scatter Matrix of the class 'c', given by:
        $S_c = Sum_{x_in_c}(x - mu_c)^T.(x - mu_c)$
    where x is a vector, given by the dataset, with each element giving the
    value of a feature (or specifically a spectral parameter).
    x_in_c are all of the entries in the dataset that belong to the class 'c',
    and mu_c is the vector that gives the mean feature values in the class 'c'.

    :param k_stack: Data to evaluate. array of spectral parameter
            values of each combination element and each material sample, with
            k_combinations stacked in the 3rd dimension, such that dimensions
            are: [n_combinations x n_samples x k_combinations]
    :type k_stack: np.array
    :param category_list: category label for each sample
    :type category_list: pd.Series
    :return: array of within-class scatter matrices S_w, with spectral parameter
        combination indexed by leading dimension of the array
    :rtype: np.array
    """
    if len(k_stack.shape) != 3:
        k_stack = k_stack.reshape(1,k_stack.shape[0],k_stack.shape[1])
    n_combinations = k_stack.shape[0]
    k_combinations = k_stack.shape[2]

    # initialise within-class scatter matrix (wcsm)
    wcsm = np.zeros((n_combinations, k_combinations, k_combinations))

    # compute wcsm
    for cat in category_list.unique():
        cat_data = k_stack[:, np.where(category_list.to_numpy() == cat)[0], :]
        cat_means = np.mean(cat_data, axis=1, keepdims=True)
        x_centered = cat_data - cat_means
        wcsm += np.matmul((x_centered).transpose(0,2,1), x_centered)
    return wcsm

def between_class_scatter_matrix(
        k_stack: np.array,
        category_list: pd.Series) -> np.array:
    """Computes the total Between-Category Scatter Matrix S_B, on the dataset
    provided by k_stack, according to category labels provided by category_list.

    Between-category scatter matrix defined as:
        $S_B = Sum_{c} n_c.(mu_c - mu).(mu_c - mu)^T$
    where mu_c is the vector that gives the mean feature (spectral parameter)
    values within the class 'c', n_c is the number of entries in the class 'c',
    and mu is the vector that gives the mean feature (spectral parameter) values
     across all classes.

    :param k_stack: Data to evaluate. array of spectral parameter
            values of each combination element and each material sample, with
            k_combinations stacked in the 3rd dimension, such that dimensions
            are: [n_combinations x n_samples x k_combinations]
    :type k_stack: np.array
    :param category_list: category labels for each sample
    :type category_list: pd.Series
    :return: array of between-class scatter matrices S_b, with spectral
        parameter combination indexed by leading dimension of the array
    :rtype: np.array
    """
    # ensure the leading dimension indexes the spectral parameter combinations
    if len(k_stack.shape) != 3:
        k_stack = k_stack.reshape(1,k_stack.shape[0],k_stack.shape[1])
    n_combinations = k_stack.shape[0]
    k_combinations = k_stack.shape[2]

    # initialise between-class scatter matrix (bcsm)
    bcsm = np.zeros((n_combinations, k_combinations, k_combinations))

    # compute bcsm
    for cat in category_list.unique():
        cat_data = k_stack[:, np.where(category_list.to_numpy() == cat)[0], :]
        cat_means = np.mean(cat_data, axis=1, keepdims=True)
        grand_means = np.mean(k_stack, axis=1, keepdims=True)
        n_c = len(np.where(category_list.to_numpy() == cat)[0])
        cntrd_means = cat_means - grand_means
        bcsm += n_c*np.matmul(cntrd_means.transpose(0,2,1), cntrd_means)

    return bcsm

def projection_matrix(
        wcsm: np.array,
        bcsm: np.array,
        n_classes: int = 2,
        singular_spcs: np.array=None) -> Tuple[np.array, np.array, np.array]:
    """Compute the projection matrix A as the solution to the problem
        $S_B a_i = lambda_k S_W a_i$
    where a_i is the projection vector for each LDA axis, of which there are
    n_c - 1 axes, where n_c is the number of classes.

    This is solved by finding the eigenvalues and eigenvectors of $S_W^-1.S_B$,
    where S_W is the within-class scatter and S_B the between-class scatter.

    See Duda, Hart & Stork, Pattern Classification, §4.11, pp. 47 - 51

    S_W and S_B are [..., k_combinations, k_combinations] matrices, where the
    [...] dimension denotes that the operation can act on multiple sets of data
    in parallel.
    The returned projection matrix has dimensions:
    [..., k_combinations, n_classes - 1]

    :param wcsm: array of within-class scatter matrices S_w, with spectral
        parameter combination indexed by leading dimension of the array
    :type wcsm: np.array
    :param bcsm: array of between-class scatter matrices S_w, with spectral
        parameter combination indexed by leading dimension of the array
    :type bcsm: np.array
    :param n_classes: number of categories to classify, defaults to 2
    :type n_classes: int, optional
    :return: array of LDA solutions for projection coordinates,
        and eigenvalues and eigenvectors used to compute these,
        with spectral parameter combination indexed by leading dimension of
        the array.
    :rtype: Tuple[np.array, np.array, np.array]
    """
    # ensure the leading dimensions index the spectral parameter combinations
    if len(wcsm.shape) != 3:
        k_combinations = wcsm.shape[0]
        wcsm = wcsm.reshape(1,k_combinations,k_combinations)
    if len(bcsm.shape) != 3:
        k_combinations = bcsm.shape[0]
        bcsm = bcsm.reshape(1,k_combinations,k_combinations)

    # initialise the output projection matrix
    n_combinations = wcsm.shape[0]
    k_combinations = wcsm.shape[2]
    projection = np.zeros((n_combinations, k_combinations, n_classes-1))

    singular_mask = singular_spcs # new way of filtering spcs
    if singular_mask is None:
        singular_mask = np.zeros(n_combinations, dtype=bool)
    projection[singular_mask,0,:] = 1.0

    # solve eigen-problem
    nonsing_inv_s_w = np.linalg.inv(wcsm[~singular_mask])
    nonsing_s_b = bcsm[~singular_mask]
    eig_vals, eig_vecs = np.linalg.eig(np.matmul(nonsing_inv_s_w, nonsing_s_b))
    eig_vals = eig_vals.real # get real values only
    eig_vecs = eig_vecs.real # get real values only

    # sort the eigenvectors by size of eigenvalue
    idx_srt = np.argsort(-eig_vals, axis=1)
    idx_vec_srt = np.repeat(idx_srt[:, np.newaxis, :], k_combinations, axis=1)
    eig_vals = np.take_along_axis(eig_vals, idx_srt, axis=1)
    eig_vecs = np.take_along_axis(eig_vecs, idx_vec_srt, axis=2)

    # Get the n_classes - 1 largest eigenvalue associated eigenvectors and
    # put in projection matrix
    projection[~singular_mask] = eig_vecs[:,:,0:n_classes-1].copy()

    return projection, eig_vecs, eig_vals

def fisher_ratio(
        projections: np.array,
        bcsm: np.array,
        wcsm: np.array) -> np.array:
    """Computes the Fisher Ratio:

        $J(A) = |A^T S_B A| / |A^T S_W A|$

    where A is the [..., n_features, n_classes - 1] projection matrix,
    S_B is the [..., n_features, n_features] between-class scatter matrix,
    S_W is the [..., n_features, n_features] within-class scatter matrix,
    and '...' denotes that the operation can be performed on multiple sets
    of data in parallel, provided n_features and n_classes are the same for
    each.
    Returns Fisher-Ratio as a [..., 1] vector.

    When n_classes > 2, J(A) ('score') is an [n_classes-1, n_classes-1] matrix.

    :param projections: array of LDA solutions for projection coordinates, with
        spectral parameter combination indexed by leading dimension of the array
    :type projections: np.array
    :param bcsm: array of between-class scatter matrices S_w, with spectral
        parameter combination indexed by leading dimension of the array
    :type bcsm: np.array
    :param wcsm: array of within-class scatter matrices S_w, with spectral
        parameter combination indexed by leading dimension of the array
    :type wcsm: np.array
    :return: Fisher-Ratio score for each spectral parameter combination computed
         on the dataset
    :rtype: np.array
    """
    between_class_scatter = np.matmul(
                                projections.transpose(0,2,1),
                                np.matmul(bcsm, projections))
    within_class_scatter = np.matmul(
                                projections.transpose(0,2,1),
                                np.matmul(wcsm, projections))
    score = np.divide(np.linalg.det(between_class_scatter),
                                        np.linalg.det(within_class_scatter))
    return score

def project_data(
        k_stack: np.array,
        projections: np.array) -> np.array:
    """Project the data onto the LDA axis for each spectral parameter
    combination.

    :param k_stack: Data to evaluate. array of spectral parameter
            values of each combination element and each material sample, with
            k_combinations stacked in the 3rd dimension, such that dimensions
            are: [n_combinations x n_samples x k_combinations]
    :type k_stack: np.array
    :param projections: array of LDA solutions for projection coordinates, with
        spectral parameter combination indexed by leading dimension of the array
    :type projections: np.array
    :return: data for each sample projected on the LDA axis for each spectral
        parameter combination
    :rtype: np.array
    """
    projected_data = np.sum(k_stack * projections[:, np.newaxis,:], axis=2).T
    return projected_data

def compute_lda_boundary(
        projection,
        k_stack: np.array,
        category_list: pd.Series
        ) -> Tuple[pd.Series, pd.Series]:
    """Compute the decision boundary for each spectral parameter combination
        LDA projection, and determine if the target category is greater than
        or less than the background category on the LDA axis.

    :param projections: array of LDA solutions for projection coordinates, with
        spectral parameter combination indexed by leading dimension of the array
    :type projections: np.array
    :param k_stack: 3D array of spectral parameter values, with
        k_combinations stacked in the 3rd dimension, such that dimensions
        are: [n_combinations x n_samples x k_combinations]
    :type k_stack: np.array
    :param category_list: category label for each sample
    :type category_list: pd.Series
    :return: boundary value, and boolean of target>background, for each
        spectral parameter combination.
    :rtype: Tuple[pd.Series, pd.Series]
    """
    projected_data = project_data(k_stack, projection)
    projected_df = pd.DataFrame(data=projected_data, index=category_list)
    cat_means = projected_df.groupby('Category').mean()

    # very simple estimate of boundary for 2 class case, assuming equal variance
    boundaries = cat_means.sum() / 2

    background_mean = cat_means.loc[category_list.cat.categories[0]]
    target_mean = cat_means.loc[category_list.cat.categories[1]]

    target_gt_background_bools = target_mean > background_mean

    return boundaries, target_gt_background_bools