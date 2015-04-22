'''
    Methods to apply AINovelty for flexfolio
'''

import numpy as np
import math
import copy

from sets import Set
from data_io import *
from experiment import Experiment
from feature_analyzer import choose_features, InstanceClusteringFTType
from ia_analyzer import choose_rep_alg_subset, AlgorithmSubsetSelectionDataType,\
    choose_rep_subset

# change with the train matrix, not with full matrix ??
# or set this from fold info
def pre_process(instance_dic, meta_info):
    '''
        Load and prepare dataset (in the form of flexfolio instance dict)
        
        :param instance_dic
        :param meta_info
        :return DataIO object
    '''
    # set instance list
    inst_list = []
    for key in instance_dic:
        inst_list.append(key)
    
    num_insts = len(inst_list)
        
    # set algorithm list    
    alg_list = []
    for key in instance_dic[instance_dic.keys()[0]]._cost['runtime']:
        alg_list.append(key)
    
    num_algs = len(alg_list)
    
    # set instance feature list
    ft_list = []
    for ft in meta_info.features: # features come as unicode (so convert to str)
        ft_list.append(ft.encode('ascii','ignore'))
        
    
    
    # create instance-algorithm performance matrix
    ia_perf_matrix = np.zeros(shape=(num_insts, num_algs))
    
    fold_id_set = set()
    
    # fill instance-algorithm performance matrix
    inst_inx = 0
    for inst in inst_list:
        ## ia_perf_matrix[inst_inx] = instance_dic[inst]._cost_vec
        alg_inx = 0
        
        # to check folds
        fold_id_set.add(instance_dic[inst]._fold[1])
        
        for alg in alg_list:
            ia_perf_matrix[inst_inx][alg_inx] = instance_dic[inst]._cost['runtime'][alg][0]
            if math.isnan(ia_perf_matrix[inst_inx][alg_inx]):
                ia_perf_matrix[inst_inx][alg_inx] = meta_info.algorithm_cutoff_time
            
            alg_inx += 1
            
        inst_inx += 1
        
    sorted_fold_id_list = sorted(fold_id_set, key=int)
        
    # create instance-fold matrix
    num_folds = len(fold_id_set) # fold indices start from 1
    if_cv_matrix = np.zeros(shape=(num_insts, num_folds), dtype=np.int)
    
    # fill instance-fold matrix
    inst_inx = 0
    for inst in inst_list:
        # -1 to start fold indices from zero
        if_cv_matrix[inst_inx][sorted_fold_id_list.index(instance_dic[inst]._fold[1])] = 1
        inst_inx += 1
        
    # create instance feature matrix
    num_features = len(instance_dic[instance_dic.keys()[0]]._features)
    i_ft_matrix = np.zeros(shape=(num_insts, num_features))
    
    # fill instance-feature matrix
    inst_inx = 0
    for inst in inst_list:
#         i_ft_matrix[inst_inx] = np.asarray(instance_dic[inst]._features)
        i_ft_matrix[inst_inx] = [0 if v is None else v for v in instance_dic[inst]._features] # set 0 for NaN features
        inst_inx += 1
        
    
    # create DataIO object to process given data further
    dio = DataIO()
    dio.num_algs = num_algs
    dio.num_insts = num_insts
    dio.num_features = num_features
    
    dio.alg_cutoff_time = meta_info.algorithm_cutoff_time
    
    dio.inst_list = inst_list
    dio.alg_list = alg_list
    dio.ft_list = ft_list
    
    dio.ia_perf_matrix = ia_perf_matrix
    dio.i_ft_matrix = i_ft_matrix
    dio.if_cv_matrix = if_cv_matrix
    
    dio.gen_rank_matrix()
    dio.gen_issolved_matrix()

    dio.gen_inst_rank_div_arr()
    dio.gen_inst_perf_div_arr()

    dio.gen_alg_rank_div_arr()
    dio.gen_alg_perf_div_arr()

    dio.gen_alg_perf_avg()
    dio.gen_alg_rank_avg()

    dio.gen_alg_solved_total()
    
    return dio
        

def apply_ain(dio):
    '''
        Apply AINovelty to determine instance, algorithm, instance feature subsets
        
        :param dio - DataIO instance
        :return numpy array, numpy array, numpy array - instances, algorithms, features 
    '''
    exp = Experiment()
        
    # extract latent (hidden) features for instances (Ur) and algorithms (Vr.T)
    i_latent_matrix, i_latent_matrix_for_ft, a_latent_matrix, sr_full, svd_dim = extract_latent_matrices(dio.ia_rank_matrix,
                                                                                                         exp.svd_type,
                                                                                                         exp.svd_dim,
                                                                                                         exp.svd_outlier_threshold)
    
    # choose a subset of representative instance features (via classification, TODO: Regression )
    norm_ft_matrix, norm_ft_min_arr, norm_ft_max_arr = gen_norm(dio.i_ft_matrix, NormalizationMethods.MinMax)
    top_ft_importance_arr, top_ft_inx_arr, ft_importance_arr, num_mapped_clusters_Ur = choose_features(norm_ft_matrix,
                                                                                                       i_latent_matrix_for_ft,
                                                                                                       exp.ft_selection_method,
                                                                                                       ft_outlier_threshold_g = exp.ft_outlier_threshold)
    sorted_top_ft_inx_arr = np.sort(top_ft_inx_arr)
    print("(%d out of %d) Features to keep are determined: %s" % (len(sorted_top_ft_inx_arr),
                                                                         dio.num_features,
                                                                         str(sorted_top_ft_inx_arr.tolist()).replace("[","").replace("]","").strip()))
    
    
    # choose a subset of representative algorithms
    a_solved_total_list = []
    a_solved_total_list.append(dio.a_solved_total) # for compatibility, instead of directly using a_solved_total = dio.a_solved_total,
    alg_per_cls, centroids_alg, labels_alg, num_clusters_alg = choose_rep_alg_subset(a_latent_matrix.T,
                                                                                     AlgorithmSubsetSelectionDataType.TotalSolved,
                                                                                     a_perf_avg = dio.a_perf_avg,
                                                                                     a_rank_avg = dio.a_rank_avg,
                                                                                     a_solved_total = a_solved_total_list, 
                                                                                     clustering_method = exp.clst_method)

    sorted_alg_per_cls = np.sort(alg_per_cls)
    print("(%d out of %d) Algorithms/solvers to keep are determined: %s" % (len(sorted_alg_per_cls),
                                                                                   dio.num_algs,
                                                                                   str(sorted_alg_per_cls.tolist()).replace("[","").replace("]","").strip()))
    
    # choose a subset of representative instances
    # inst_per_cls, centroids_inst, labels_inst, num_clusters_inst = choose_rep_inst_subset(dio, i_latent_matrix, clustering_method = clst_method, is_cv_partial_data = True)
    per_inst_perf_criterion_arr = dio.i_perf_div_std

    inst_clst_ft_matrix = None
    if exp.inst_clst_ft_type == InstanceClusteringFTType.Descriptive:
        inst_clst_ft_matrix = norm_ft_matrix
    elif exp.inst_clst_ft_type == InstanceClusteringFTType.Latent:
        inst_clst_ft_matrix = i_latent_matrix
    elif exp.inst_clst_ft_type == InstanceClusteringFTType.DescriptiveLatent:
        inst_clst_ft_matrix = np.concatenate((norm_ft_matrix, i_latent_matrix), 1)
    elif exp.inst_clst_ft_type == InstanceClusteringFTType.DescriptiveSubset:
        inst_clst_ft_matrix = norm_ft_matrix[:, sorted_top_ft_inx_arr]


    inst_per_cls, centroids_inst, labels_inst, num_clusters_inst = choose_rep_subset(inst_clst_ft_matrix,
                                                                                     per_inst_perf_criterion_arr,
                                                                                     criterion_higher_better=True,
                                                                                     clustering_method = exp.clst_method,
                                                                                     k_max=5) ## dio.num_insts/3

    list_of_clst_insts_lists = get_clst_info(centroids_inst,
                                             labels_inst,
                                             num_clusters_inst)
    avg_clst_rank_matrix = calc_avg_clst_rank_matrix(dio.ia_rank_matrix,
                                                     list_of_clst_insts_lists)
    print("avg_clst_rank_matrix: ", avg_clst_rank_matrix)
    clst_rank_score = calc_clst_rank_score(avg_clst_rank_matrix)
    print("clst_rank_score: ", clst_rank_score)



    sorted_inst_per_cls = np.sort(inst_per_cls)
    print("inst_per_cls : ", inst_per_cls)
    print("(%d out of %d) Instances to keep are determined: %s" % (len(sorted_inst_per_cls),
                                                                          len(dio.ia_perf_matrix),
                                                                          str(sorted_inst_per_cls.tolist()).replace("[","").replace("]","").strip()))
    
    return sorted_inst_per_cls, sorted_alg_per_cls, sorted_top_ft_inx_arr


## TODO : update other feature related variables
## TODO : remove from meta_info.features_deterministic and features_stochastic
## TODO : update meta_info feature_group dict !!
def post_process(instance_dic, meta_info, dio, inst_arr, alg_arr, ft_arr):
    '''
        Filter flexfolio dataset dictionaries 
        
        :param instance_dic
        :param meta_info
        :param dio: DataIO object
        :param inst_arr: numpy array - selected instances
        :param alg_arr: numpy array - selected algorithms
        :param ft_arr: numpy array - selected instance features
        :return filtered_instance_dic, filtered_meta_info
    '''
    # generate a new instance_dic
    filtered_instance_dic = {}
    inst_inx = 0
    for inst_inx in inst_arr:
        inst = dio.inst_list[inst_inx]
        filtered_instance_dic[inst] = copy.deepcopy([instance_dic[inst]])[0]
    inst_inx += 1
    
    # generate a new meta_info
    filtered_meta_info = copy.deepcopy([meta_info])[0]
    
        
    # filter algorithms and instance features
    to_remove_alg_list = []
    for alg_inx in range(dio.num_algs):
        if alg_inx not in alg_arr:
            to_remove_alg_list.append(dio.alg_list[alg_inx])
            
    to_remove_ft_list = []
    for ft_inx in range(dio.num_features):
        if ft_inx not in ft_arr:
            to_remove_ft_list.append(ft_inx)
    
    first_step = True
    for inst_key, instance in filtered_instance_dic.iteritems():
        # remove discarded algorithms
        for alg_key in reversed(to_remove_alg_list):
#             instance._cost['runtime'].pop(alg_key)
            del instance._cost['runtime'][alg_key]
            del instance._cost_vec[dio.alg_list.index(alg_key)]
            del instance._transformed_cost_vec[dio.alg_list.index(alg_key)]
            
            if first_step: # just need to go over once
                del filtered_meta_info.algorithms[filtered_meta_info.algorithms.index(alg_key)]
                if alg_key in filtered_meta_info.algorithms_stochastic:
                    del filtered_meta_info.algorithms_stochastic[filtered_meta_info.algorithms_stochastic.index(alg_key)]
                    
                if alg_key in filtered_meta_info.algortihms_deterministics:
                    del filtered_meta_info.algortihms_deterministics[filtered_meta_info.algortihms_deterministics.index(alg_key)]
            
            
            
        # remove discarded features
        for ft_inx in reversed(to_remove_ft_list):
            ft_key = dio.ft_list[ft_inx]
            del instance._features[ft_inx]
            
            if first_step: # just need to go over once
                del filtered_meta_info.features[filtered_meta_info.features.index(ft_key)]
                if ft_key in filtered_meta_info.features_deterministic:
                    del filtered_meta_info.features_deterministic[filtered_meta_info.features_deterministic.index(ft_key)]
    
                if ft_key in filtered_meta_info.features_stochastic:
                    del filtered_meta_info.features_stochastic[filtered_meta_info.features_stochastic.index(ft_key)]                
        
        first_step = False
    
#     return filtered_instance_dic
    return filtered_instance_dic, filtered_meta_info

# TODO: means and stds of selection_dic should be updated 
def update_selection_dic(instance_dic, selection_dic, selected_ft_arr):
    '''
        Update given selection dictionary (coming from training)
        
        :param instance_dic
        :param selection_dic
        :param selected_ft_arr: numpy array - selected features
    '''
    
    num_features = len(instance_dic[instance_dic.keys()[0]]._features)
    
    selection_dic['normalization']['filter'] = []
    for ft_inx in range(num_features):
        if ft_inx in selected_ft_arr:
            selection_dic['normalization']['filter'].append(1)
        else:
            selection_dic['normalization']['filter'].append(0)
            

def filter_instance_test(instance_test, selected_ft_arr):
    '''
        Filter given test instance dictionary
        
        :param instance_test
        :param selected_ft_arr: numpy array - selected features
    '''
    
    filtered_instance_test = {}
    inst_inx = 0
    for inst_key, instance in instance_test.iteritems():
        filtered_instance_test[inst_key] = copy.deepcopy([instance_test[inst_key]])[0]
    inst_inx += 1
    
    num_features = len(filtered_instance_test[filtered_instance_test.keys()[0]]._features)
    
    to_remove_ft_list = []
    for ft_inx in range(num_features):
        if ft_inx not in selected_ft_arr:
            to_remove_ft_list.append(ft_inx)
    
    
     # remove discarded features
    for inst_key, instance in filtered_instance_test.iteritems():
        for ft_inx in reversed(to_remove_ft_list):
            del instance._features[ft_inx]
                
    return filtered_instance_test
    


def filter_data(instance_dic, meta_info):
    '''
        Dataset filtering for flexfolio 
        
        :param instance_dic
        :param meta_info
        :return filtered_instance_dic, filtered_meta_info, ft_arr
    '''
    
    dio = pre_process(instance_dic, meta_info)
    
    inst_arr, alg_arr, ft_arr = apply_ain(dio)
    
    filtered_instance_dic, filtered_meta_info = post_process(instance_dic, meta_info, dio, inst_arr, alg_arr, ft_arr)

    return filtered_instance_dic, filtered_meta_info, ft_arr
    
    

def main():

    ## TODO : pass args (to run from command prompt or easily change)
    print("")


if __name__ == "__main__":
    main()



