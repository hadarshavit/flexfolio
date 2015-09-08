'''
    Methods to apply AINovelty for flexfolio
    with newer ain implementation
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
from ia_analyzer import get_opt_alg_portfolio,\
    get_opt_alg_portfolio_via_aa_superior,\
    get_opt_alg_portfolio_for_num_solved_insts,\
    choose_rep_subset_insts_as_reduction
from src.ainovelty.feature_analyzer import ft_selection_post_process
from src.ainovelty.ia_analyzer import choose_alg_subset_via_hiearchical_clustering,\
    choose_alg_subset_via_hiearchical_clustering_fcluster,\
    choose_alg_subset_via_hiearchical_clustering_fcluster_kthbest,\
    choose_inst_subset_via_hiearchical_clustering_fcluster_kthbest
from src.ainovelty.plot_util import plot_ft_importance_from_clsf,\
    plot_2d_scatter_subset_with_cluster
from src.ainovelty import settings

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
#     for key in instance_dic[instance_dic.keys()[0]]._cost['runtime']:
#         alg_list.append(key)
    for alg_name in meta_info.algorithms: #### ALGORITHM ORDERS USED AS IN meta_info.algorithms
        alg_list.append(alg_name)
    
    
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
        
        
        
        
        
        
    # process feature step/cost details (!! new)
    i_ft_step_list = meta_info.feature_group_dict.keys()
    num_ft_steps = len(i_ft_step_list)
    i_ft_cost_matrix = np.zeros(shape=(num_insts, num_ft_steps))
    
    inst_inx = 0
    for inst in inst_list:
        
        if instance_dic[inst]._feature_group_cost_dict: ## check whether dictionary is empty
        
            for ft_step in i_ft_step_list:
                ft_step_inx = i_ft_step_list.index(ft_step)
                i_ft_cost_matrix[inst_inx][ft_step_inx] = instance_dic[inst]._feature_group_cost_dict[ft_step]
            
                ## http://docs.scipy.org/doc/numpy/reference/generated/numpy.nan_to_num.html
                ## nan feature cost values are converted to zero !!!
                i_ft_cost_matrix[inst_inx] = np.nan_to_num(i_ft_cost_matrix[inst_inx])
            
            inst_inx += 1
        

        
    i_ft_step_dict = meta_info.feature_group_dict
    
    i_ft_step_membership_matrix = np.zeros((num_features, num_ft_steps), dtype=int)
    for ft_name in ft_list:
        ft_inx = ft_list.index(ft_name)
        
        for ft_step_name, step_ft_list in i_ft_step_dict.iteritems():
            ft_step_inx = i_ft_step_list.index(ft_step_name)
            ## print ft_step_inx
            if  ft_name in step_ft_list: 
                i_ft_step_membership_matrix[ft_inx][ft_step_inx] = 1 
        
        
    
    
    
    ####################################################
    # create DataIO object to process given data further
    ####################################################
    dio = DataIO()
        
    dio.dataset_name = meta_info.scenario
    
    dio.num_algs = num_algs
    dio.num_insts = num_insts
    dio.num_features = num_features
    dio.num_ft_steps = num_ft_steps
    
    dio.alg_cutoff_time = meta_info.algorithm_cutoff_time
    
    dio.inst_list = inst_list
    dio.alg_list = alg_list
    dio.ft_list = ft_list
    
    dio.ia_perf_matrix = ia_perf_matrix
    dio.i_ft_matrix = i_ft_matrix
    dio.if_cv_matrix = if_cv_matrix


    ## feature cost related data
    dio.num_ft_steps = num_ft_steps
    dio.i_ft_step_list = i_ft_step_list
    dio.i_ft_step_dict = i_ft_step_dict
    dio.i_ft_cost_matrix = i_ft_cost_matrix
    dio.i_ft_step_membership_matrix = i_ft_step_membership_matrix
    

#     desc_file_path = "/home/misir/Desktop/aslib_data-aslib-v1.1/"+meta_info.scenario+"/description.txt"
#     ft_costs_file_path = "/home/misir/Desktop/aslib_data-aslib-v1.1/"+meta_info.scenario+"/feature_costs.arff"
#     dio.read_desc_file(desc_file_path)
#     dio.read_ft_costs_file(ft_costs_file_path)
    
    dio.det_unsolved_instances() # determine unsolved instances
        
    
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
        


def apply_ain(dio, ain_num_ft_to_remove = 0):
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
    
    
    # determine optimal portfolio
#     opt_portfolio = get_opt_alg_portfolio(dio.ia_rank_matrix, dio.unsolved_inst_list)
#     print "opt_portfolio: ", opt_portfolio
#     gen_aa_superior_matrix(train_rank_matrix)
#     opt_portfolio_set = get_opt_alg_portfolio_via_aa_superior(dio.ia_rank_matrix)
#     print "opt_portfolio v2: ", opt_portfolio_set
    
    
#     train_issolved_matrix = gen_issolved_matrix(exp.data_to_run.train_ia_perf_matrix, dio.alg_cutoff_time)
#     opt_portfolio_set2 = get_opt_alg_portfolio_for_num_solved_insts(dio.ia_rank_matrix, train_issolved_matrix)
#     print "opt_portfolio v3: ", opt_portfolio_set2
    
    
    inst_inx_arr = range(0, dio.num_insts)
    solved_inst_arr = np.delete(inst_inx_arr, np.array(dio.unsolved_inst_list), 0) 
    
    ################## evaluate dataset wrt when kth best per instance algorithm is selected #####################
    # dataset_hardness_dict = evaluate_dataset_hardness(dio.ia_perf_matrix[solved_inst_arr, :], dio.alg_cutoff_time)
    ##############################################################################################################

    
    # choose a subset of representative instance features (via classification, TODO: Regression )
    norm_ft_matrix, norm_ft_min_arr, norm_ft_max_arr = gen_norm(dio.i_ft_matrix, NormalizationMethods.MinMax)
    
    if exp.ft_subset_selection:
        top_ft_importance_arr, top_ft_inx_arr, ft_importance_arr, num_mapped_clusters_Ur = choose_features(norm_ft_matrix,
                                                                                                           i_latent_matrix_for_ft,
                                                                                                           exp.ft_selection_method,
                                                                                                           ft_outlier_threshold_g = exp.ft_outlier_threshold)
    
    
        ## print "top_ft_inx_arr: ", top_ft_inx_arr
        print "% Selected Instance Features (", len(top_ft_inx_arr)," / ",len(norm_ft_matrix[0]),"): ", top_ft_inx_arr
        
        if exp.ft_postprocessing:
            ft_cost_imp_dict, top_ft_inx_arr, top_ft_importance_arr = ft_selection_post_process(dio.i_ft_cost_matrix, dio.i_ft_step_membership_matrix, top_ft_inx_arr, top_ft_importance_arr)
            
            ## print "AFTER POSTPROCESSING: top_ft_inx_arr: ", top_ft_inx_arr
        
        
    else:
        num_fts = len(norm_ft_matrix[0])
        ft_importance_arr = np.zeros(num_fts, dtype=int)
        top_ft_importance_arr = np.zeros(num_fts, dtype=int)
        top_ft_inx_arr = np.zeros(num_fts, dtype=int)
        for ft_inx in range(num_fts):
            ft_importance_arr[ft_inx] = 1
            top_ft_importance_arr[ft_inx] = 1
            top_ft_inx_arr[ft_inx] = ft_inx
        
        num_mapped_clusters_Ur = num_fts
        

    
    if ain_num_ft_to_remove > 0 and ain_num_ft_to_remove < len(ft_importance_arr):
        
        ordered_ft_imp_arr = np.argsort(ft_importance_arr)
        
        top_ft_importance_arr = ft_importance_arr[ordered_ft_imp_arr]
        top_ft_inx_arr = ordered_ft_imp_arr
        
        
        top_ft_importance_arr = top_ft_importance_arr[ain_num_ft_to_remove:]
        top_ft_inx_arr = top_ft_inx_arr[ain_num_ft_to_remove:]
        
        
        ft_cost_imp_dict = calc_total_ft_cost(dio.i_ft_cost_matrix, dio.i_ft_step_membership_matrix, top_ft_inx_arr, top_ft_importance_arr)

        ## http://stackoverflow.com/questions/3179106/python-select-subset-from-list-based-on-index-set
        sel_ft_name_list = [dio.ft_list[i] for i in top_ft_inx_arr]       
        
#         print "$$$$$$$$$$$\t", len(top_ft_importance_arr)
#         print "@@@@@@@@@@@\t", np.sum(top_ft_importance_arr)
#         print "ZZZZZZZZZZZ\t", ft_cost_imp_dict['ft_total_cost']
#         print "FFFFFFFFFFF\t", sel_ft_name_list
    
        
    
    sorted_top_ft_inx_arr = np.sort(top_ft_inx_arr)
#     print("(%d out of %d) Features to keep are determined: %s" % (len(sorted_top_ft_inx_arr),
#                                                                          dio.num_features,
#                                                                          str(sorted_top_ft_inx_arr.tolist()).replace("[","").replace("]","").strip()))
    
    
    if exp.to_plot:
        plot_ft_importance_from_clsf(dio.dataset_name,
                                     ft_importance_arr,
                                     top_ft_inx_arr,
                                     num_mapped_clusters_Ur,
                                     'FS-Gini-Fold-'+str(settings.plot_uniq_inx+1),
                                     output_folder=settings.___output_folder___,
                                     to_show = False)


    
    
    
    # choose a subset of representative algorithms
    a_solved_total_list = []
    a_solved_total_list.append(dio.a_solved_total) # for compatibility, instead of directly using a_solved_total = dio.a_solved_total,
    if exp.alg_subset_selection:
#         alg_per_cls, centroids_alg, labels_alg, num_clusters_alg = choose_rep_alg_subset(a_latent_matrix.T,
#                                                                                          AlgorithmSubsetSelectionDataType.TotalSolved,
#                                                                                          a_perf_avg = dio.a_perf_avg,
#                                                                                          a_rank_avg = dio.a_rank_avg,
#                                                                                          a_solved_total = a_solved_total_list, 
#                                                                                          clustering_method = exp.clst_method)
        
        
        
         
        num_inst_solved, par10 = evaluate_oracle( dio.ia_perf_matrix[solved_inst_arr, :], dio.alg_cutoff_time)
        ## print "Oracle performance before algorithm subset selection: num_inst_solved = ", num_inst_solved, " - par10 = ", par10
        
         
        ### choose_alg_subset_via_hiearchical_clustering
        alg_per_cls, centroids_alg, labels_alg, num_clusters_alg = choose_alg_subset_via_hiearchical_clustering_fcluster_kthbest(dio.ia_perf_matrix, 
                                                                                                                a_latent_matrix.T, 
                                                                                                                dio.alg_cutoff_time, 
                                                                                                                solved_inst_arr,
                                                                                                                unique_name=dio.dataset_name,
                                                                                                                title='Hierarchical-Algorithms-Fold-'+ str(settings.plot_uniq_inx+1),
                                                                                                                output_folder=settings.___output_folder___,
                                                                                                                k = 3,
                                                                                                                )
                 
                 
        ## check the oracle performance of considering the new algorithm (sub)set
        num_inst_solved, par10 = evaluate_oracle( (dio.ia_perf_matrix[solved_inst_arr,:])[:, np.sort(alg_per_cls)], dio.alg_cutoff_time)
        ##print "Oracle performance after algorithm subset selection: num_inst_solved = ", num_inst_solved, " - par10 = ", par10
        
        print "% Selected Algorithms (",len(alg_per_cls)," / ", dio.num_algs,"): ", str(alg_per_cls.tolist()).replace("[","").replace("]","").strip()
        
        
    else:
        ## ignored for now- to only keep optimal portfolio
        ##alg_per_cls = list(opt_portfolio_set)
        alg_per_cls = []
    
        #sorted_alg_per_cls = np.sort(alg_per_cls)
    
        labels_alg = np.zeros(shape=(dio.num_algs), dtype=np.int)
        for alg_inx in range(dio.num_algs):
            labels_alg[alg_inx] = alg_inx
            alg_per_cls.append(alg_inx)
#         print("No algorithm subset selection, just get the optimal portfolio: ",   
#                                                                          str(alg_per_cls).replace("[","").replace("]","").strip())
     



    sorted_alg_per_cls = np.sort(alg_per_cls)
#     print("(%d out of %d) Algorithms/solvers to keep are determined: %s" % (len(sorted_alg_per_cls),
#                                                                                    dio.num_algs,
#                                                                                    str(sorted_alg_per_cls.tolist()).replace("[","").replace("]","").strip()))


    
    if exp.to_plot:
        ## if hierarchical clustering
        dummy_selected_arr = np.zeros(dio.num_algs);
        for alg_inx in alg_per_cls:
            dummy_selected_arr[alg_inx] = 1;
        plot_2d_scatter_subset_with_cluster(dio.dataset_name,
                                            a_latent_matrix.T,
                                            alg_per_cls,
                                            dummy_selected_arr, ###labels_alg,
                                            'Algorithms-Fold-'+str(settings.plot_uniq_inx+1),
                                            plt_annt_list = dio.alg_list,
                                            marker_size = 30,
                                            dim_reduction_type = exp.dim_rd_type,
                                            hide_axis_labels = True,
                                            output_folder=settings.___output_folder___,
                                            to_show = False)
    
    
    
    
    
    # choose a subset of representative instances
    # inst_per_cls, centroids_inst, labels_inst, num_clusters_inst = choose_rep_inst_subset(dio, i_latent_matrix, clustering_method = clst_method, is_cv_partial_data = True)
    per_inst_perf_criterion_arr = dio.i_perf_div_std

    if exp.inst_subset_selection:
        to_change = False
        while True:
            if not to_change:
                inst_clst_ft_matrix = i_latent_matrix
            else:
                inst_clst_ft_matrix = np.concatenate((norm_ft_matrix, i_latent_matrix), 1)
            
            inst_per_cls, centroids_inst, labels_inst, num_clusters_inst = choose_rep_subset_insts_as_reduction(inst_clst_ft_matrix,
                                                                                                                 dio.unsolved_inst_list, 
                                                                                                                 per_inst_perf_criterion_arr,
                                                                                                                 criterion_higher_better=True,
                                                                                                                 clustering_method = exp.clst_method,
                                                                                                                 k_max=dio.num_insts/3) ## dio.num_insts/3  
            
            
#             inst_per_cls, centroids_inst, labels_inst, num_clusters_inst = choose_inst_subset_via_hiearchical_clustering_fcluster_kthbest(dio.ia_perf_matrix, 
#                                                                                                                 i_latent_matrix, 
#                                                                                                                 dio.alg_cutoff_time, 
#                                                                                                                 sorted_alg_per_cls, 
#                                                                                                                 solved_inst_arr,
#                                                                                                                 k = 3)
                        
                        
            
            if num_clusters_inst <= (dio.num_insts*0.01):
                if to_change:
                    break
                else:
                    to_change = True
            else:
                break
            
        print "% Selected Instances (", len(inst_per_cls)," / ", len(dio.ia_perf_matrix),") : ", str(inst_per_cls.tolist()).replace("[","").replace("]","").strip()  
    

    else: # no instance subset selection
        
        inst_per_cls = np.zeros(dio.num_insts)
        for inst_inx in range(dio.num_insts):
            inst_per_cls[inst_inx] = inst_inx
        
        sorted_inst_per_cls = inst_per_cls
        labels_inst = inst_per_cls
        num_clusters_inst = dio.num_insts

    

#     list_of_clst_insts_lists = get_clst_info(centroids_inst,
#                                              labels_inst,
#                                              num_clusters_inst)
#     avg_clst_rank_matrix = calc_avg_clst_rank_matrix(dio.ia_rank_matrix,
#                                                      list_of_clst_insts_lists)
#     print("avg_clst_rank_matrix: ", avg_clst_rank_matrix)
#     clst_rank_score = calc_clst_rank_score(avg_clst_rank_matrix)
#     print("clst_rank_score: ", clst_rank_score)



    sorted_inst_per_cls = np.sort(inst_per_cls)
#     print("inst_per_cls : ", inst_per_cls)
#     print("(%d out of %d) Instances to keep are determined: %s" % (len(sorted_inst_per_cls),
#                                                                           len(dio.ia_perf_matrix),
#                                                                           str(sorted_inst_per_cls.tolist()).replace("[","").replace("]","").strip()))
    
    
    
    if exp.to_plot:     
#         if not inst_clst_ft_matrix:
#             inst_clst_ft_matrix = i_latent_matrix
        
        for inx in range(dio.num_insts):
            labels_inst[inx] = 0
        
        plot_2d_scatter_subset_with_cluster(dio.dataset_name,
                                            i_latent_matrix, #inst_clst_ft_matrix,
                                            inst_per_cls,
                                            labels_inst, ###labels_alg,
                                            'Latent-Instances-Fold-'+str(settings.plot_uniq_inx+1),
                                            #plt_annt_list = dio.alg_list,
                                            marker_size = 30,
                                            dim_reduction_type = exp.dim_rd_type,
                                            hide_axis_labels = True,
                                            output_folder=settings.___output_folder___,
                                            to_show = False)
        
        ## directly plot initial descriptive features matrix
        plot_2d_scatter_subset_with_cluster(dio.dataset_name,
                                    norm_ft_matrix, #inst_clst_ft_matrix,
                                    inst_per_cls,
                                    labels_inst, ###labels_alg,
                                    'Initial-Instances-Fold-'+str(settings.plot_uniq_inx+1),
                                    #plt_annt_list = dio.alg_list,
                                    marker_size = 30,
                                    dim_reduction_type = exp.dim_rd_type,
                                    hide_axis_labels = True,
                                    output_folder=settings.___output_folder___,
                                    to_show = False)
    
    
    return sorted_inst_per_cls, sorted_alg_per_cls, sorted_top_ft_inx_arr




## TODO : update other feature related variables
## TODO : remove from meta_info.features_deterministic and features_stochastic
## TODO : update meta_info feature_group dict !!
def post_process(instance_dic, meta_info, config_dic, dio, inst_arr, alg_arr, ft_arr):
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
    # generate a new instance_dic (only keep selected instances)
    ##### inst_arr = np.arange(0, len(dio.inst_list))  ### just to test effects of different number of instances
    filtered_instance_dic = {}
    inst_inx = 0
    for inst_inx in inst_arr.astype(int):
        inst = dio.inst_list[inst_inx]
        filtered_instance_dic[inst] = copy.deepcopy([instance_dic[inst]])[0]
    inst_inx += 1
    
    # generate a new meta_info
    filtered_meta_info = copy.deepcopy([meta_info])[0]
    
    # generate a new config_dic
    filtered_config_dic = copy.deepcopy([config_dic])[0]
        
    # filter algorithms and instance features
    to_remove_alg_list = []
    to_remove_alg_inx_list = []
    for alg_inx in range(dio.num_algs):
        if alg_inx not in alg_arr:
            to_remove_alg_list.append(dio.alg_list[alg_inx])
            to_remove_alg_inx_list.append(alg_inx)
            
            
    to_remove_ft_list = []
    for ft_inx in range(dio.num_features):
        if ft_inx not in ft_arr:
            to_remove_ft_list.append(ft_inx)
    
    first_step = True
    for inst_key, instance in filtered_instance_dic.iteritems():
        # remove discarded algorithms
        
        ##for alg_key in reversed(to_remove_alg_list):
        for alg_indice in range(len(to_remove_alg_inx_list)-1, 0, -1):
#             instance._cost['runtime'].pop(alg_key)
            ##print to_remove_alg_list, "alg_key: ", alg_key

            alg_key = to_remove_alg_list[alg_indice]
            alg_inx = to_remove_alg_inx_list[alg_indice]

            del instance._cost['runtime'][alg_key]
#             del instance._cost_vec[dio.alg_list.index(alg_key)]
#             del instance._transformed_cost_vec[dio.alg_list.index(alg_key)]
            del instance._cost_vec[alg_inx]
            del instance._transformed_cost_vec[alg_inx]

            
            if first_step: # just need to go over once
                del filtered_config_dic[alg_key]
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
        
        #TODO: update feature step info
        # determine ft steps to be removed
        original_to_remove_ft_step_list = []
        to_remove_ft_step_list = []
        sel_i_ft_step_membership_matrix = dio.i_ft_step_membership_matrix[ft_arr, :]
        for ft_step_name, step_ft_list in dio.i_ft_step_dict.iteritems(): 
            ft_step_inx = dio.i_ft_step_list.index(ft_step_name) 
            
            if np.sum(sel_i_ft_step_membership_matrix.T[ft_step_inx]) == 0:
                to_remove_ft_step_list.append(ft_step_inx)
          
            if np.sum(dio.i_ft_step_membership_matrix.T[ft_step_inx]) == 0:
                original_to_remove_ft_step_list.append(ft_step_inx)
            
        
        ### TODO: prevent unnecessary loops - e.g. close if all diff steps are removed
        
        if set(to_remove_ft_step_list) != set(original_to_remove_ft_step_list): 
#             print "Some feature steps should be ignored / removed !! TODO: make required changes in ain_flexfolio"
            
            all_to_remove_ft_step_name_list = []
            for ft_step_inx in to_remove_ft_step_list:
                all_to_remove_ft_step_name_list.append(dio.i_ft_step_list[ft_step_inx])
                
            
            diff_ft_step_list = list(set(to_remove_ft_step_list) - set(original_to_remove_ft_step_list))
            diff_ft_step_name_list = []
            for ft_step_inx in diff_ft_step_list:
                diff_ft_step_name_list.append(dio.i_ft_step_list[ft_step_inx])
                
            for inst_key, instance in filtered_instance_dic.iteritems():
#                 instance._feature_cost_total = 0

                if not instance._feature_group_cost_dict:
                    continue
                
                for ft_step_name in dio.i_ft_step_list:
                    
                    if ft_step_name in instance._feature_group_cost_dict:
                    
                        if ft_step_name in diff_ft_step_name_list and (instance._feature_group_cost_dict[ft_step_name] is not None):
                            ##print "instance._feature_group_cost_dict[ft_step_name], ", inst_key, " +++ ", ft_step_name, " +++ ", instance._feature_group_cost_dict[ft_step_name]
                            instance._feature_cost_total -= instance._feature_group_cost_dict[ft_step_name]
                            instance._feature_group_cost_dict[ft_step_name] = 0.0
                            
                            if diff_ft_step_name_list.index(ft_step_name) == len(diff_ft_step_name_list)-1: ## all are processed
                                break
                        
    #                     elif ft_step_name not in all_to_remove_ft_step_name_list:
    #                         instance._feature_cost_total += instance._feature_group_cost_dict[ft_step_name]
                       
            
                
        
        
        #TODO: update feature costs
#         total_cost = 0
#         previous_presolved = False
#         for f_step in dio.i_ft_step_list:
#             if instance._feature_group_cost_dict.get(f_step) and not previous_presolved: # feature costs are maybe None
#                 total_cost += instance._feature_group_cost_dict[f_step]
#             if instance._features_status[f_step] == "PRESOLVED":
#                 previous_presolved = True
#         for un_step in unused_steps:        # remove step status if unused 
#             del instance._features_status[un_step]
        
#         instance._feature_cost_total = total_cost
        
        
        # TODO : remove discarded instances 
        
        
        
        
#     return filtered_instance_dic
    return filtered_instance_dic, filtered_meta_info, filtered_config_dic, to_remove_alg_list, to_remove_alg_inx_list



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
            

def filter_instance_test(instance_test, to_remove_alg_list, to_remove_alg_inx_list, selected_ft_arr):
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
            
#         for alg_key in reversed(to_remove_alg_list):
        for alg_indice in range(len(to_remove_alg_inx_list)-1, 0, -1):
            alg_key = to_remove_alg_list[alg_indice]
            alg_inx = to_remove_alg_inx_list[alg_indice]
            del instance._cost['runtime'][alg_key]
            del instance._cost_vec[alg_inx]
            del instance._transformed_cost_vec[alg_inx]            
                
    return filtered_instance_test
    


def filter_data(instance_dic, meta_info, config_dic, ain_num_ft_to_remove = 0):
    '''
        Dataset filtering for flexfolio 
        
        :param instance_dic
        :param meta_info
        :return filtered_instance_dic, filtered_meta_info, ft_arr
    '''
    
    dio = pre_process(instance_dic, meta_info)
    
    inst_arr, alg_arr, ft_arr = apply_ain(dio, ain_num_ft_to_remove)
    
    filtered_instance_dic, filtered_meta_info, filtered_config_dic, to_remove_alg_list, to_remove_alg_inx_list = post_process(instance_dic, meta_info, config_dic, dio, inst_arr, alg_arr, ft_arr)

    return filtered_instance_dic, filtered_meta_info, filtered_config_dic, to_remove_alg_list, to_remove_alg_inx_list, ft_arr
    
    

def main():

    ## TODO : pass args (to run from command prompt or easily change)
    print("")


if __name__ == "__main__":
    main()



