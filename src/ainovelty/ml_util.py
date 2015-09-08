import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import rankdata

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn import cross_validation
from sklearn.linear_model import BayesianRidge
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
#from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from random import randint
# from pyxmeans.xmeans import  XMeans


__seed__ = 123456

class MLMethodType:
    Regression, Classification, Clustering = range(0, 3)

class RegressionMethods:
    RF, SVR, GBR, BayesRidge = range(0, 4)

class ClassificationMethods:
    RFC, LR, SVM = range(0, 3)

class FeatureImportanceMethods:
    Gini, GBR, RFClsf, BayesRidge  = range(0, 4) ##TODO

class ClusteringEvaluationMetrics:
    SLH, RankBasedPS = range(0, 2)

class ClusteringMethods:
    KMeans, KMeansSLH, KMeansSLHadaptive, XMeans, AFProp, AgglomerativeClustering, MeanShift, DBScan = range(0, 8)

class NormalizationMethods:
    ScikitDef, MinMax = range(0, 2)


def normalize_minmax(np_matrix, min_arr = None, max_arr = None):
    '''
        Normalize a given matrix with
        (np_matrix[i][j] - min_arr[j]) / (max_arr[j] - min_arr[j])

        :param np_matrix: numpy 2D matrix - input matrix to be normalized
        :param min_arr: numpy float array - per column minimum values used for normalization
        :param max_arr: numpy float array - per column maximum values used for normalization
        :return: numpy 2D matrix - normalized matrix (without changing the original matrix)
    '''
    num_rows = len(np_matrix)
    num_cols = len(np_matrix[0])

    pre_norm_info = False
    if min_arr is not None and max_arr is not None:
        pre_norm_info = True
    else:
        min_arr = np.zeros(shape=(num_cols))
        max_arr = np.zeros(shape=(num_cols))

    norm_matrix = np.zeros(shape=(num_rows, num_cols))

    for j in range(num_cols):

        if not pre_norm_info:
            min_arr[j] = np.min(np_matrix.T[j])
            max_arr[j] = np.max(np_matrix.T[j])

        for i in range(num_rows):

            if max_arr[j] == min_arr[j]:
                norm_matrix[i][j] = 0
            else:
                norm_matrix[i][j] = (np_matrix[i][j] - min_arr[j]) / (max_arr[j] - min_arr[j])

    return norm_matrix, min_arr, max_arr


def gen_norm(matrix, type, min_arr = None, max_arr = None):
    '''
        Normalize a given matrix

        :param matrix: numpy 2D matrix - input matrix to be normalized
        :param type: NormalizationMethods - normalization type
        :param min_arr: numpy float array - per column minimum values used for normalization
        :param max_arr: numpy float array - per column maximum values used for normalization
        :return: numpy 2D matrix - normalized matrix
    '''
    # norm_matrix = None
    # min_arr, max_arr = None, None
    if type == NormalizationMethods.MinMax:
        # norm_matrix, min_arr, max_arr = normalize_minmax(matrix, min_arr, max_arr)
        return normalize_minmax(matrix, min_arr, max_arr)
    elif type == NormalizationMethods.ScikitDef:
        ## TODO: return min_arr, max_arr as well as in MinMax
        # norm_matrix = normalize(matrix)
        return normalize(matrix, axis = 0)

    # return norm_matrix, min_arr, max_arr


## TODO: error check - whether rank data's shape is suitable to row data matrix
def eval_clustering(row_data_matrix, clst, type, row_data_rank_matrix = None):
    '''
        Evaluate a clustering result

        :param row_data_matrix: numpy 2D matrix - data used for clustering
        :param clst: clustering model
        :param type: ClusteringEvaluationMetrics - clustering evaluation metric
        :param row_data_rank_matrix: numpy 2D matrix - rank matrix derived from row_data_matrix
        :return: float - clustering score
    '''
    score = 0

    ## to keep the cluster labels / numbers consecutive (starting from zero)
    labels = rankdata(clst.labels_, method='dense')-1

    if type == ClusteringEvaluationMetrics.SLH:
        score = metrics.silhouette_score(row_data_matrix, labels, metric='euclidean', random_state=__seed__)
    elif type == ClusteringEvaluationMetrics.RankBasedPS:
        # num_clusters = len(np.unique(labels))

        if row_data_rank_matrix is None:
            row_data_rank_matrix = rankdata(row_data_matrix, method='dense')
        else:
            if np.array_equal(row_data_matrix, row_data_rank_matrix):
                print("row_data_matrix ", row_data_matrix.shape,  "and row_data_rank_matrix are in different shape", row_data_rank_matrix.shape, "\nExiting...")
                sys.exit()

        list_of_clst_insts_lists = get_clst_info(clst.cluster_centers_, clst.labels_, len(np.unique(labels)))
        avg_clst_rank_matrix = calc_avg_clst_rank_matrix(row_data_rank_matrix, list_of_clst_insts_lists)
        # print("avg_clst_rank_matrix: ", avg_clst_rank_matrix)
        score = calc_clst_rank_score(avg_clst_rank_matrix)

    return score


def get_clst_info(centroids, labels, num_clusters, labels_consecutive = False):
    '''
        Generate a list of lists regarding given clustering results,
        where each list refers to a cluster and includes the indices
        of the member data points

        :param centroids:
        :param labels:
        :param num_clusters: int - number of clusters
        :param labels_consecutive: boolean - whether cluster labels are consecutively numbered
        :return: list - cluster membership list
    '''
    clst_labels = None
    if labels_consecutive:
        clst_labels = rankdata(labels, method='dense')-1  ## cluster indices should start from zero
    else:
        clst_labels = labels

    clst_labels = clst_labels.astype(np.int)

    list_of_list = []
    for inx in range(num_clusters):
        list_of_list.append([])

    data_point_inx = 0
    for label in clst_labels:
        list_of_list[label].append(data_point_inx)
        data_point_inx += 1

    # print(list_of_list)

    return list_of_list


def calc_avg_clst_rank_matrix(ia_rank_matrix, list_of_datapoint_lists):
    '''
        Calculate the average rank of each algorithm for each cluster, composed of instances

        :param ia_rank_matrix: numpy 2D matrix - (instance, algorithm) rank matrix
        :param list_of_datapoint_lists: list - cluster membership list including data points
        :return: numpy 2D matrix - average rank of each algorithm in each cluster
    '''
    avg_clst_rank_matrix = np.zeros(shape=(len(list_of_datapoint_lists), len(ia_rank_matrix.T)))
    clst_inx = 0
    for data_point_list in list_of_datapoint_lists:
        for data_point_inx in data_point_list:
            avg_clst_rank_matrix[clst_inx] = np.add(avg_clst_rank_matrix[clst_inx], ia_rank_matrix[data_point_inx])

        avg_clst_rank_matrix[clst_inx] /= len(data_point_list)

        clst_inx += 1

    return avg_clst_rank_matrix


## TODO: check each cluster's quality, change this to compare only top N algorithms, not all, use NDCG like score calculation since first algo is more critical than second etc.
def calc_clst_rank_score(avg_clst_rank_matrix):
    '''
        Calculate the rank score which indicates how the clusters differ
        w.r.t. algorithms' ranks on each cluster of instances
        (this metric is applicable only for instances)

        :param avg_clst_rank_matrix: numpy 2D matrix - average rank of each algorithm in each cluster
        :return: float - clustering rank score
    '''
    score = 0
    cnt = 0
    num_clusters = len(avg_clst_rank_matrix)
    for clst_1_inx in range(num_clusters):
        for clst_2_inx in xrange(clst_1_inx+1, num_clusters):
            score += np.sum(np.absolute(np.subtract(avg_clst_rank_matrix[clst_1_inx], avg_clst_rank_matrix[clst_2_inx])))
            cnt += 1
    score /= float(cnt)

    return score


def kmeans_adaptive(row_matrix, k_max, init_k_max = 10, clst_eval_metric = ClusteringEvaluationMetrics.SLH, sensitivity = 0.01):
    '''
        kmeans clustering with automatically determining right k (number of clusters)

        :param row_matrix: numpy 2D matrix - matrix to be clustered (rows are data points)
        :param k_max: int - maximum k
        :param init_k_max: int - upper bound on k to test kmeans initially
        :param clst_eval_metric: ClusteringEvaluationMetrics - metric used to evaluate clustering results
        :return: clst, - centroids, labels, number of clusters
    '''
    best_score = -1
    best_clst = None
    best_k = -1

    if k_max < 2 or k_max > len(row_matrix):
        k_max = len(row_matrix)

    k_left = 2
    k_right = k_max-1

    score_left = -1
    score_right = -1

    if init_k_max < 2 or init_k_max > len(row_matrix):
        init_k_max = len(row_matrix)
        
    if init_k_max == 2:
        clst = None
        centroids = row_matrix
        labels = range(0, 2)
        num_clusters = 2
        return clst, centroids, labels, num_clusters
    

    # initially try first init_k_max values to speed up
    for k_val in xrange(2, init_k_max+1):
        clst = KMeans(n_clusters=k_val, random_state=__seed__).fit(row_matrix)
        score = eval_clustering(row_matrix, clst, clst_eval_metric)

        if score > best_score:
            best_score = score
            best_clst = clst
            best_k = k_val
            ##print("INIT -> Best k = ", best_k, " - best_score = ", best_score)
            
            ## not to cluster further if clustering is almost perfect
            if best_score >= 1.0 - sensitivity:
                init_k_max = k_max ## just to skip the further clustering 
                break


    ##print("AFTER INIT -> Best k = ", best_k, "k_left = ", k_left, " - score_left = ", score_left, "k_right = ", k_right, " - score_right = ", score_right)

    if init_k_max < k_max: # adaptive

        score_left = best_score
        clst_left = best_clst
        k_left = init_k_max

        clst_right = KMeans(n_clusters=k_right, random_state=__seed__).fit(row_matrix)
        score_right = eval_clustering(row_matrix, clst_right, clst_eval_metric)
        if score_right > best_score:
            best_score = score_right
            best_clst = clst_right
            best_k = k_right

        while True:
            
            ## not to cluster further if clustering is almost perfect
            if best_score >= 1.0 - sensitivity:
                break


            diff = (k_right - k_left) / 2
            if score_left > score_right:
                k_right = k_left + diff

                clst_right = KMeans(n_clusters=k_right, random_state=__seed__).fit(row_matrix)
                score_right = eval_clustering(row_matrix, clst_right, clst_eval_metric)
                if score_right > best_score:
                    best_score = score_right
                    best_clst = clst_right
                    best_k = k_right
            else:
                k_left = k_left + diff

                clst_left = KMeans(n_clusters=k_left, random_state=__seed__).fit(row_matrix)
                score_left = eval_clustering(row_matrix, clst_left, clst_eval_metric)
                if score_left > best_score:
                    best_score = score_left
                    best_clst = clst_left
                    best_k = k_left
                    
                    
            ##print("Best k = ", best_k, "k_left = ", k_left, " - score_left = ", score_left, "k_right = ", k_right, " - score_right = ", score_right)
            

            if k_left >= k_right-1: ## no more to check
                break

            # added to complete clustering earlier (can be removed??)
            if best_score > 0.5 and abs(score_left - score_right) <= sensitivity:
                break
            

    #if best_k == k_max-1: ## TODO: for now
    if best_k == len(row_matrix)-1: ## TODO: for now
        clst = None
        centroids = row_matrix
        labels = range(0, len(row_matrix))
        num_clusters = len(centroids)
    else:
        clst = best_clst
        centroids = best_clst.cluster_centers_
        # rankdata - in case if the cluster indices are not 1-step consecutive
        labels = rankdata(best_clst.labels_, method='dense')-1  ## cluster indexes start from zero
        num_clusters = len(np.unique(labels))


    return clst, centroids, labels, num_clusters




def gen_class_arr(row_matrix, type, k_for_kmeans=300, num_consc_wrs_steps = 0, k_max=-1):
    '''
        Clustering

        :param row_matrix: numpy 2D matrix - matrix to be clustered (rows are data points)
        :param type: ClusteringMethods - clustering method to use
        :param k_for_kmeans: int - number of clusters for the required clustering methods (e.g. kmeans)
        :param num_consc_wrs_steps: int - number of consecutive worsening steps for KMeansSLH
        :param k_max: int - maximum number of clusters
        :return: centroids, labels, number of clusters
    '''
    #cluster_centers_indices = None
    centroids = None
    labels = None

    consc_wrs_steps_cnt = 1

    max_k = len(row_matrix)
    if max_k < k_for_kmeans:
        k_for_kmeans = max_k

    # clst_matrix = row_matrix
    # if to_norm == True:
    #     clst_matrix = normalize_minmax(row_matrix)

    clst = None
    if type == ClusteringMethods.AFProp:
        clst = AffinityPropagation(preference=-50).fit(row_matrix)
        ##cluster_centers_indices = clst.cluster_centers_indices_
    elif type == ClusteringMethods.KMeans:
        clst = KMeans(n_clusters=k_for_kmeans, random_state=__seed__).fit(row_matrix)
    elif type == ClusteringMethods.XMeans:
        # clst = XMeans(2).fit(row_matrix) ##TODO
        print("TODO")
#    elif type == ClusteringMethods.AgglomerativeClustering:
#        clst = AgglomerativeClustering().fit(row_matrix)
    elif type == ClusteringMethods.MeanShift:
        bandwidth = estimate_bandwidth(row_matrix, quantile=0.2, n_samples=max_k)
        clst = MeanShift(bandwidth=bandwidth).fit(row_matrix)
    elif type == ClusteringMethods.DBScan:
        clst = DBSCAN(eps=0.3, min_samples=2).fit(row_matrix)
    elif type == ClusteringMethods.KMeansSLH:
        best_slh_score = -1
        best_clst = None
        for k in range(len(row_matrix)):
            clst = KMeans(n_clusters=k+2, random_state=__seed__).fit(row_matrix)
            labels = clst.labels_
            try:
                slh_score = metrics.silhouette_score(row_matrix, labels, metric='euclidean', random_state=__seed__)
            except: ## to prevent silhouette_score exception for num_samples == num_clusters
                # if k == len(row_matrix):
                #     best_clst = clst
                # break
                centroids = row_matrix
                labels = range(0,len(row_matrix))
                num_clusters = len(centroids)
                return centroids, labels, num_clusters


            print("k = ", (k+2), " - slh_score = ", slh_score)
            if k == 0:
                best_slh_score = slh_score
                best_clst = clst
            else:
                if slh_score > best_slh_score:
                    best_slh_score = slh_score
                    best_clst = clst

                    consc_wrs_steps_cnt = 1
                else:
                    consc_wrs_steps_cnt += 1
                    if consc_wrs_steps_cnt > num_consc_wrs_steps:
                        break

        clst = best_clst

    elif type == ClusteringMethods.KMeansSLHadaptive: ## TODO
        clst, centroids, labels, num_clusters = kmeans_adaptive(row_matrix, k_max)

    else:
        print("Clustering type is inapplicable: ", type, "\nExiting...")
        sys.exit()

    centroids = None ##TODO: Fix this
    # centroids = clst.cluster_centers_
    # labels = clst.labels_ ##TODO: check this
    if clst is None:
        labels = rankdata(labels, method='dense')-1
    else:
        labels = rankdata(clst.labels_, method='dense')-1

    # labels = best_clst.predict(row_matrix)
    # num_clusters = len(centroids)
    num_clusters = len(np.unique(labels))

    return centroids, labels, num_clusters


def gen_multivar_regr_model(ft_matrix, output_matrix, type, ft_selected = None):
    '''
        Generate a multivariate regression model

        :param ft_matrix: numpy 2D matrix - input/feature matrix
        :param output_matrix: numpy 2D matrix - output matrix
        :param type: RegressionMethods - regression method type
        :param ft_selected: numpy array - selected features
        :return multivariate regression model:
    '''
    regr = None
    if type == RegressionMethods.RF:
        regr = RandomForestRegressor(random_state=__seed__)
    elif type == RegressionMethods.SVR:
        regr = SVR()
    elif type == RegressionMethods.GBR: ## cannot do multivariate regression
        regr = GradientBoostingRegressor(n_estimators=3000, max_depth=6, learning_rate=0.04,
                                         loss='huber', random_state=__seed__)
    elif type == RegressionMethods.BayesRidge:
        regr = BayesianRidge()

    input_matrix = ft_matrix
    if ft_selected is not None:
        sorted_ft_selected = np.sort(ft_selected)
        input_matrix = ft_matrix[:, sorted_ft_selected]

    regr.fit(input_matrix, output_matrix)

    return regr



## TODO: apply xmeans to cluster U matrix entries
def gen_clsf_model(ft_matrix, cls_arr, type):
    '''
        Generate a classification model

        :param ft_matrix: numpy 2D matrix - input/feature matrix
        :param cls_arr: numpy array - class labels
        :param type: Classifier type
        :return:
    '''
    clsf = None
    if type == ClassificationMethods.RFC:
        clsf = RandomForestClassifier(n_estimators=10, random_state=__seed__)
        clsf = clsf.fit(ft_matrix, cls_arr)

    return clsf


def apply_CV(model, ft_matrix, output_matrix, num_folds, ft_selected = None):
    '''
        Cross validation

        :param model: regression/classification model used for cross validation
        :param ft_matrix: numpy 2D matrix - feature/input matrix
        :param output_matrix: numpy 2D matrix - output matrix
        :param num_folds: int - number of folds
        :param ft_selected: numpy int array - selected features (columns)
        :return: numpy array - scores on folds
    '''
    input_matrix = ft_matrix
    if ft_selected != None:
        input_matrix = ft_matrix[:, ft_selected]

    scores = cross_validation.cross_val_score(model, input_matrix, output_matrix, cv=num_folds)

    return scores



# def gen_CV_indices(data, num_folds):
#     '''
#         Generate CV validation indices
#     :param data:
#     :return:
#     '''
#     # get the whole index list
#     data_length = len(data)
#     full_data_inx_list = []
#     for i in range(data_length):
#         full_data_inx_list.append(i)
#
#     train_data_inx_list = []
#     test_data_inx_list = []
#
#     test_size = data_length / num_folds
#     for fold_inx in range(num_folds): # for each fold
#         train = []
#         test = []
#         for test_data_inx in range(test_size):
#
#
#         for data_inx in range(data_length):  # for each data point
#             train.append()


def test_alg_sel(data, inst_sel, alg_sel, ft_sel):
    ## TODO
    train_data, test_data = train_test_split(data, test_size=0.5, random_state=__seed__)


def get_ft_importance(mod, type):
    '''
        Get features' importance / scores for a given model

        :param mod: classification / regression model
        :param type: FeatureImportanceMethods - feature importance/scoring method
        :return: numpy array - features' importance levels or scores
    '''
    ft_importance = None
    if type == FeatureImportanceMethods.Gini:
         ft_importance = mod.feature_importances_
    elif type == FeatureImportanceMethods.GBR:
        # sort importances
        indices = np.argsort(mod.feature_importances_)
        # plot as bar chart - TODO : check this part
        plt.barh(mod.feature_importances_[indices])
        _ = plt.xlabel('Relative importance')
    elif type == FeatureImportanceMethods.BayesRidge:
        ft_importance = mod.feature_importances_
    elif type == FeatureImportanceMethods.RFClsf:
        ft_importance = mod.feature_importances_

    return ft_importance