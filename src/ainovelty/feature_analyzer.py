import numpy as np
import os
import pprint
import itertools
import sys

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression

from ml_util import *
from data_io import *

class FeatureSelectionMethod:
    VarianceThreshold, VarianceThresholdPercentile, SelectKBest, SelectPercentile, GiniPercentile = range(0, 5)

class InstanceClusteringFTType:
    Descriptive, Latent, DescriptiveLatent, DescriptiveLatentExt, DescriptiveSubset, DescriptiveSubsetLatent, DescriptiveSubsetLatentExt = range(0, 7)


## TODO : change return for other feature selection methods
def choose_features(norm_ft_matrix, output_matrix, type, k_best_features=10, num_folds_g = 10, num_consc_wrs_steps_g = 5, ft_outlier_threshold_g = 95):
    '''
        Choose a subset of given features

        :param norm_ft_matrix: numpy 2D matrix - normalized feature matrix
        :param output_matrix: numpy 2D matrix - output matrix
        :param type: FeatureSelectionMethod - feature selection method
        :param k_best_features: int - number of features to be selected if applicable
        :param num_folds_g: int - number of folds to evaluate the quality of norm_ft_matrix -> output_matrix mapping (if applicable)
        :param num_consc_wrs_steps_g: number of consecutive worsening steps, used for clustering in GiniPercentile
        :param ft_outlier_threshold_g: percentile threshold level in [0, 100) used to determine number of features to be selected
        :return:
    '''
    num_features = len(norm_ft_matrix.T)

    # f_selector = None
    scores = None
    top_ft_inx_arr = None

    if type == FeatureSelectionMethod.VarianceThreshold:
        f_selector = VarianceThreshold(threshold=(.8 * (1 - .8)))
        f_selector.fit_transform(norm_ft_matrix)
        scores = f_selector.variances_


    elif type == FeatureSelectionMethod.VarianceThresholdPercentile:
        f_selector = VarianceThreshold(threshold=(.8 * (1 - .8)))
        f_selector.fit_transform(norm_ft_matrix)
        scores = f_selector.variances_

        scores, top_ft_inx_arr = percentile_based_outlier(scores, ft_outlier_threshold_g)

    elif type == FeatureSelectionMethod.SelectKBest:
        f_selector = SelectKBest(f_regression, k=k_best_features)
        f_selector.fit_transform(norm_ft_matrix, output_matrix)

    elif type == FeatureSelectionMethod.SelectPercentile: ##TODO - correct this (Nan values for single output ??)
        k_perc = k_best_features * 100 / float(num_features)
        f_selector = SelectPercentile(f_regression, percentile=k_perc)
        f_selector.fit_transform(norm_ft_matrix, output_matrix[:, 0])
        scores = -np.log10(f_selector.pvalues_)
        scores /= scores.max()

    elif type == FeatureSelectionMethod.GiniPercentile:
        centroids, labels, num_clusters = gen_class_arr(output_matrix, ClusteringMethods.KMeansSLHadaptive, num_consc_wrs_steps = num_consc_wrs_steps_g, k_max=10) ## k_max=len(norm_ft_matrix)/4
        clsf = gen_clsf_model(norm_ft_matrix, labels, ClassificationMethods.RFC)
        # TODO: add cv scores to the log file to check the quality of the feature selection model
        # cv_scores = apply_CV(clsf, norm_ft_matrix, labels, num_folds_g)
        # print("CV scores: ", cv_scores)
        ft_importance_arr = get_ft_importance(clsf, FeatureImportanceMethods.Gini)
        scores, top_ft_inx_arr = percentile_based_outlier(ft_importance_arr, ft_outlier_threshold_g)
        sorted_top_ft_inx_arr = np.sort(top_ft_inx_arr)
#         print("top_ft_inx_arr: ", top_ft_inx_arr)

        return scores, top_ft_inx_arr, ft_importance_arr, num_clusters


    return scores, top_ft_inx_arr, None, None

