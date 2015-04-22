import warnings

from data_io import *
from ml_util import *


class AlgorithmSubsetSelectionDataType:
    AvgPerf, AvgRank, TotalSolved = range(0, 3)


# @deprecated
def choose_rep_inst_subset(dio, i_latent_matrix, clustering_method = ClusteringMethods.KMeansSLH, num_inst_per_cluster = 1):
    '''
        Choose an instance subset

        :param dio:
        :param i_latent_matrix:
        :param clustering_method:
        :param num_inst_per_cluster:
        :return:
    '''

    # cluster instances wrt latent features
    centroids, labels, num_clusters = gen_class_arr(i_latent_matrix, clustering_method, num_consc_wrs_steps=5)

    # choose X number of instances from each cluster
    best_per_inst_val = np.zeros(shape=(num_clusters))
    inst_per_cls = np.zeros(shape=(num_clusters), dtype=np.int)
    for inst_inx in range(dio.num_insts):
        cls_inx = labels[inst_inx]
        ## TODO: change equal case !!!
        if dio.i_perf_div_std[inst_inx] >= best_per_inst_val[cls_inx]:
            inst_per_cls[cls_inx] = inst_inx
            best_per_inst_val[cls_inx] = dio.i_perf_div_std[inst_inx]

    return inst_per_cls, centroids, labels, num_clusters


def choose_rep_subset(ft_matrix, per_row_perf_criterion_arr, criterion_higher_better, clustering_method = ClusteringMethods.KMeansSLHadaptive, num_inst_per_cluster = 1, k_max=-1):
    '''
        Choose subsets wrt a given norm feature matrix (or latent matrix)
        representing either algorithms or instances

        :param ft_matrix:
        :param per_row_perf_criterion_arr:
        :param criterion_higher_better:
        :param clustering_method:
        :param num_inst_per_cluster:
        :param k_max:
        :return:
    '''
    # cluster rows wrt latent features
    centroids, labels, num_clusters = gen_class_arr(ft_matrix, clustering_method, num_consc_wrs_steps=5, k_max=k_max)

    # choose X number of rows from each cluster
    best_per_row_val = np.zeros(shape=(num_clusters))
    row_per_cls = np.zeros(shape=(num_clusters), dtype=np.int)
    if not criterion_higher_better:
        best_per_row_val.fill(sys.maxint)

    num_rows = len(ft_matrix)

    for pc_inx in range(num_rows):
        cls_inx = labels[pc_inx]
        # print("pc_inx", pc_inx, "cls_inx", cls_inx, "num_clusters", num_clusters)

        if is_better(best_per_row_val[cls_inx], per_row_perf_criterion_arr[pc_inx], criterion_higher_better=criterion_higher_better):
            row_per_cls[cls_inx] = pc_inx
            best_per_row_val[cls_inx] = per_row_perf_criterion_arr[pc_inx]

    return row_per_cls, centroids, labels, num_clusters


def is_better(curr_val, new_val, criterion_higher_better):
    '''
        Check whether is new_val is better than curr_val

        :param curr_val: float - current value
        :param new_val: float - new value
        :param criterion_higher_better: boolean - whether higher is better
        :return: boolean - whether new_val is better than curr_val
    '''
    if criterion_higher_better:
        ## TODO: change equal case !!!
        if new_val >= curr_val:
            return True
        else:
            return False
    else:
        if new_val < curr_val:
            return True
        else:
            return False


def choose_rep_alg_subset(a_latent_matrix, type, a_perf_avg = None, a_rank_avg = None, a_solved_total = None, clustering_method = ClusteringMethods.KMeansSLHadaptive, num_alg_per_cluster = 1):
    '''
        Determine algorithm subset for further training / generating algorithm selection models

        :param a_latent_matrix:
        :param type:
        :param a_perf_avg:
        :param a_rank_avg:
        :param a_solved_total:
        :param clustering_method:
        :param num_alg_per_cluster:
        :return:
    '''

    # cluster instances wrt latent features
    centroids, labels, num_clusters = gen_class_arr(a_latent_matrix, clustering_method, num_consc_wrs_steps=5)

    # choose X number of instances from each cluster
    best_per_alg_val = np.zeros(shape=(num_clusters), dtype=np.int)
    if type != AlgorithmSubsetSelectionDataType.TotalSolved:
        best_per_alg_val.fill(sys.maxint)

    alg_per_cls = np.zeros(shape=(num_clusters), dtype=np.int)

    num_algs = len(a_latent_matrix)

    for alg_inx in range(num_algs):
        cls_inx = labels[alg_inx]
        # if dio.a_perf_div_std[alg_inx] > best_per_alg_val[cls_inx]:
        #     alg_per_cls[cls_inx] = alg_inx
        #     best_per_alg_val[cls_inx] = dio.a_perf_div_std[alg_inx]
        if type == AlgorithmSubsetSelectionDataType.AvgPerf:
            if a_perf_avg[0][alg_inx] < best_per_alg_val[cls_inx]:
                alg_per_cls[cls_inx] = alg_inx
                best_per_alg_val[cls_inx] = a_perf_avg[0][alg_inx]
        elif type == AlgorithmSubsetSelectionDataType.AvgRank:
            if a_rank_avg[0][alg_inx] < best_per_alg_val[cls_inx]:
                alg_per_cls[cls_inx] = alg_inx
                best_per_alg_val[cls_inx] = a_rank_avg[0][alg_inx]
        elif type == AlgorithmSubsetSelectionDataType.TotalSolved:
            # if a_solved_total[alg_inx] > best_per_alg_val[cls_inx]:
            if a_solved_total[0][alg_inx] > best_per_alg_val[cls_inx]:
                alg_per_cls[cls_inx] = alg_inx
                best_per_alg_val[cls_inx] = a_solved_total[0][alg_inx]

    return alg_per_cls, centroids, labels, num_clusters