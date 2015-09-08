import sys
import itertools
import logging
import datetime
import warnings

from data_io import *
# from evaluator import *
from ml_util import *
from plot_util import *
from ia_analyzer import *
from feature_analyzer import *
from utils import *
from experiment import *
import ia_analyzer
from src.ml_util import ClusteringMethods
from src.ia_analyzer import gen_aa_superior_matrix,\
    get_opt_alg_portfolio_for_num_solved_insts,\
    parameter_importance
from src.data_io import gen_issolved_matrix, extract_alg_confs


'''
    TODO:
    
    Tests with
    - directly initial descriptive features + latent features
    - using MiniBatchKMeans rather than KMeans under kmeans_adaptive
        * additionally test other clustering methods !!
    - start clustering with num_instances/2
    - fix cases when outliers do not exist
        * for instances and instance features
    - e.g. for instances with small clusters like CSP with only 3 clusters
    it might be hard to have outliers so a very small number of instances are picked
        * propose a solution to overcome this directly
        
    - fix final performance report on the complete matrix for SAT12-RAND
        * where optimal portfolio includes 30 out of 31 algorithms
        * fold results are correct but final result is wrong??
        * check results from here: E:\_____TEST-SNM-KOD-DGR\_____Freiburg-Tests\Ainovelty\13052015121405734242-latent-or-desclat-noalgsel  
'''


##to prevent warnings to be printed to console
##warnings.filterwarnings("ignore")


__output_folder__ = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
# __eps_dpi__ = 200
# plot = True

# InstanceClusteringFTType = enum('Latent','DescriptiveLatent', 'DescriptiveLatentExt')

## to keep a general log
# logging.basicConfig(filename="AIN.log", format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)


logger = logging.getLogger('AIN.log')


# def apply_ain(exp, dio, plot_uniq_inx):
# 
#     # generate rank matrix for extracting latent features
#     train_rank_matrix = gen_rank_matrix(exp.data_to_run.train_ia_perf_matrix, dio.alg_cutoff_time)
#     logger.debug("A rank matrix is generated from the training data/matrix")
# 
#     # extract latent (hidden) features for instances (Ur) and algorithms (Vr.T)
#     i_latent_matrix, i_latent_matrix_for_ft, a_latent_matrix, sr_full, svd_dim = extract_latent_matrices(train_rank_matrix,
#                                                                                                          exp.svd_type,
#                                                                                                          exp.svd_dim,
#                                                                                                          exp.svd_outlier_threshold)
#     logger.debug("Latent (hidden) features are extracted from the rank matrix")
# 
# 
#     # determine optimal portfolio
#     opt_portfolio = get_opt_alg_portfolio(train_rank_matrix, exp.data_to_run.train_unsolved_inst_list)
#     print "opt_portfolio: ", opt_portfolio
# #     gen_aa_superior_matrix(train_rank_matrix)
#     opt_portfolio_set = get_opt_alg_portfolio_via_aa_superior(train_rank_matrix)
#     print "opt_portfolio v2: ", opt_portfolio_set
#     
#     
#     train_issolved_matrix = gen_issolved_matrix(exp.data_to_run.train_ia_perf_matrix, dio.alg_cutoff_time)
#     opt_portfolio_set2 = get_opt_alg_portfolio_for_num_solved_insts(train_rank_matrix, train_issolved_matrix)
#     print "opt_portfolio v3: ", opt_portfolio_set2
# 
# 
#     ##np.savetxt("train_issolved_matrix.csv", X=train_issolved_matrix.astype(int), delimiter=",")
# 
#     # choose a subset of representative instance features (via classification, TODO: Regression )
#     train_norm_ft_matrix, train_norm_ft_min_arr, train_norm_ft_max_arr = gen_norm(exp.data_to_run.train_i_ft_matrix, NormalizationMethods.MinMax)
#     
#     if exp.ft_subset_selection:
#         top_ft_importance_arr, top_ft_inx_arr, ft_importance_arr, num_mapped_clusters_Ur = choose_features(train_norm_ft_matrix,
#                                                                                                            i_latent_matrix_for_ft,
#                                                                                                            exp.ft_selection_method,
#                                                                                                            ft_outlier_threshold_g = exp.ft_outlier_threshold)
#         
#     else:
#         num_fts = len(train_norm_ft_matrix[0])
#         ft_importance_arr = np.zeros(num_fts, dtype=int)
#         top_ft_importance_arr = np.zeros(num_fts, dtype=int)
#         top_ft_inx_arr = np.zeros(num_fts, dtype=int)
#         for ft_inx in range(num_fts):
#             ft_importance_arr[ft_inx] = 1
#             top_ft_importance_arr[ft_inx] = 1
#             top_ft_inx_arr[ft_inx] = ft_inx
#         
#         num_mapped_clusters_Ur = num_fts
# 
#         
#     sorted_top_ft_inx_arr = np.sort(top_ft_inx_arr)
#     logger.debug("(%d out of %d) Features to keep are determined: %s" % (len(sorted_top_ft_inx_arr),
#                                                                          dio.num_features,
#                                                                          str(sorted_top_ft_inx_arr.tolist()).replace("[","").replace("]","").strip()))
#     if exp.to_plot:
#         plot_ft_importance_from_clsf(dio.dataset_name,
#                                      ft_importance_arr,
#                                      top_ft_inx_arr,
#                                      num_mapped_clusters_Ur,
#                                      'FS-Gini-Fold-'+str(plot_uniq_inx+1),
#                                      output_folder=__output_folder__,
#                                      to_show = False)
# 
# 
#     # choose a subset of representative algorithms
#     if exp.alg_subset_selection:
#         alg_per_cls, centroids_alg, labels_alg, num_clusters_alg = choose_rep_alg_subset(a_latent_matrix.T,
#                                                                                          AlgorithmSubsetSelectionDataType.TotalSolved,
#                                                                                          a_perf_avg = exp.data_to_run.a_perf_avg,
#                                                                                          a_rank_avg = exp.data_to_run.a_rank_avg,
#                                                                                          a_solved_total = exp.data_to_run.a_solved_total,
#                                                                                          clustering_method = ClusteringMethods.KMeans) ##default: exp.clst_method
#     
#         #sorted_alg_per_cls = np.sort(alg_per_cls)
#     else:
#         alg_per_cls = list(opt_portfolio_set)
#     
#         #sorted_alg_per_cls = np.sort(alg_per_cls)
#     
#         labels_alg = np.zeros(shape=(dio.num_algs), dtype=np.int)
#         for alg_inx in range(dio.num_algs):
#             labels_alg[alg_inx] = alg_inx
#         print("No algorithm subset selection, just get the optimal portfolio: ",   
#                                                                          str(alg_per_cls).replace("[","").replace("]","").strip())
#         logger.debug("No algorithm subset selection, just get the optimal portfolio: %s" % 
#                                                                          str(alg_per_cls).replace("[","").replace("]","").strip())
#         
#         
#     sorted_alg_per_cls = np.sort(alg_per_cls)
#         
#         
#     logger.debug("(%d out of %d) Algorithms/solvers to keep are determined: %s" % (len(sorted_alg_per_cls),
#                                                                                    dio.num_algs,
#                                                                                    str(sorted_alg_per_cls.tolist()).replace("[","").replace("]","").strip()))
#     print("alg_per_cls : ", alg_per_cls)
#     # print(a_latent_matrix.T)
#     if exp.to_plot:
#         plot_2d_scatter_subset_with_cluster(dio.dataset_name,
#                                             a_latent_matrix.T,
#                                             alg_per_cls,
#                                             labels_alg,
#                                             'Algorithms-Fold-'+str(plot_uniq_inx+1),
#                                             plt_annt_list = dio.alg_list,
#                                             marker_size = 30,
#                                             dim_reduction_type = exp.dim_rd_type,
#                                             hide_axis_labels = True,
#                                             output_folder=__output_folder__,
#                                             to_show = False)
# 
# 
# 
#     # choose a subset of representative instances
#     if exp.inst_subset_selection:
#         
#         # inst_per_cls, centroids_inst, labels_inst, num_clusters_inst = choose_rep_inst_subset(dio, i_latent_matrix, clustering_method = clst_method, is_cv_partial_data = True)
#         train_inx_arr = np.where(dio.if_cv_matrix.T[plot_uniq_inx] == 0)[0]
#         per_inst_perf_criterion_arr = dio.i_perf_div_std[train_inx_arr]
#         #per_inst_perf_criterion_arr = dio.i_orank_sim[train_inx_arr]
#     
#     #     inst_clst_ft_matrix = None
#     #     if exp.inst_clst_ft_type == InstanceClusteringFTType.Descriptive:
#     #         inst_clst_ft_matrix = train_norm_ft_matrix
#     #     elif exp.inst_clst_ft_type == InstanceClusteringFTType.Latent:
#     #         inst_clst_ft_matrix = i_latent_matrix
#     #     elif exp.inst_clst_ft_type == InstanceClusteringFTType.DescriptiveLatent:
#     #         inst_clst_ft_matrix = np.concatenate((train_norm_ft_matrix, i_latent_matrix), 1)
#     #     elif exp.inst_clst_ft_type == InstanceClusteringFTType.DescriptiveSubset:
#     #         inst_clst_ft_matrix = train_norm_ft_matrix[:, sorted_top_ft_inx_arr]
#     # 
#     # 
#     #     inst_per_cls, centroids_inst, labels_inst, num_clusters_inst = choose_rep_subset(inst_clst_ft_matrix,
#     #                                                                                      per_inst_perf_criterion_arr,
#     #                                                                                      criterion_higher_better=True,
#     #                                                                                      clustering_method = exp.clst_method,
#     #                                                                                      k_max=dio.num_insts/3)
#     
#         to_change = False ## False - set True just for test
#         while True:
#             if not to_change:
#                 inst_clst_ft_matrix = i_latent_matrix
#             else:
#                 inst_clst_ft_matrix = np.concatenate((train_norm_ft_matrix, i_latent_matrix), 1)
#             
#             
#             ## earlier, used this function: choose_rep_subset(...)
#             ## new - choose_rep_subset_insts_as_reduction
#             inst_per_cls, centroids_inst, labels_inst, num_clusters_inst = choose_rep_subset_insts_as_reduction(inst_clst_ft_matrix,
#                                                                                                                  exp.data_to_run.train_unsolved_inst_list, ## new parameter
#                                                                                                                  per_inst_perf_criterion_arr,
#                                                                                                                  criterion_higher_better=True,
#                                                                                                                  clustering_method = exp.clst_method,
#                                                                                                                  k_max=dio.num_insts) ## dio.num_insts/3    
#             
#             
#             ### break ## just for test 
#             
#             
#             if num_clusters_inst <= (dio.num_insts*0.01):
#                 if to_change:
#                     break
#                 else:
#                     to_change = True
#             else:
#                 break
#     
#         list_of_clst_insts_lists = get_clst_info(centroids_inst,
#                                                  labels_inst,
#                                                  num_clusters_inst)
#         avg_clst_rank_matrix = calc_avg_clst_rank_matrix(train_rank_matrix,
#                                                          list_of_clst_insts_lists)
#         #print("avg_clst_rank_matrix: ", avg_clst_rank_matrix)
#         clst_rank_score = calc_clst_rank_score(avg_clst_rank_matrix)
#         print("clst_rank_score: ", clst_rank_score)
# 
#     else: # no instance subset selection
#         
#         inst_per_cls = np.zeros(exp.data_to_run.num_insts)
#         for inst_inx in range(exp.data_to_run.num_insts):
#             inst_per_cls[inst_inx] = inst_inx
#         
#         sorted_inst_per_cls = inst_per_cls
#         labels_inst = inst_per_cls
#         num_clusters_inst = exp.data_to_run.num_insts
#         
#     
#     print "(#clusters= ",num_clusters_inst,")# instances are selected: ", len(inst_per_cls), " out of ", dio.num_insts, " (", (100 * len(inst_per_cls) / float(dio.num_insts)), ")"
# 
#     sorted_inst_per_cls = np.sort(inst_per_cls)
#     print("inst_per_cls : ", inst_per_cls)
#     logger.debug("(%d out of %d) Instances to keep are determined: %s" % (len(sorted_inst_per_cls),
#                                                                           len(exp.data_to_run.train_ia_perf_matrix),
#                                                                           str(sorted_inst_per_cls.tolist()).replace("[","").replace("]","").strip()))
#     if exp.to_plot:
#         plot_2d_scatter_subset_with_cluster(dio.dataset_name,
#                                             inst_clst_ft_matrix,
#                                             inst_per_cls,
#                                             labels_inst,
#                                             'Instances-Fold-'+str(plot_uniq_inx+1),
#                                             marker_size = 30,
#                                             dim_reduction_type = exp.dim_rd_type,
#                                             hide_axis_labels = True,
#                                             output_folder=__output_folder__,
#                                             to_show = False)
# 
# 
# 
# 
#     #######################################
#     ####### EVALUATION + COMPARISON #######
#     #######################################
# 
# 
#     ## Performance model generation (training)
#     sel_ia_perf_matrix, sel_i_ft_matrix = extract_perf_ft_data_for_selected(exp.data_to_run.train_ia_perf_matrix,
#                                                                             exp.data_to_run.train_i_ft_matrix,
#                                                                             sorted_inst_per_cls,
#                                                                             sorted_alg_per_cls,
#                                                                             sorted_top_ft_inx_arr) ## , after_norm = after_norm
#     sel_ia_rank_matrix = np.zeros(shape=(len(inst_per_cls), len(alg_per_cls)))
#     for i in range(len(inst_per_cls)):
#         for j in range(len(alg_per_cls)):
#             sel_ia_rank_matrix[i][j] = dio.ia_rank_matrix[inst_per_cls[i], alg_per_cls[j]]
# 
# 
#     norm_sel_i_ft_matrix, norm_sel_i_ft_min_arr, norm_sel_i_ft_max_arr = gen_norm(sel_i_ft_matrix,
#                                                                                   NormalizationMethods.MinMax,
#                                                                                   min_arr=train_norm_ft_min_arr[sorted_top_ft_inx_arr],
#                                                                                   max_arr=train_norm_ft_max_arr[sorted_top_ft_inx_arr])
# 
#     regr_model = gen_multivar_regr_model(norm_sel_i_ft_matrix, sel_ia_perf_matrix, RegressionMethods.RF)
# 
# 
#     ## Prediction via generated performance model (testing)
#     test_norm_ft_matrix, test_norm_ft_min_arr, test_norm_ft_max_arr = gen_norm(exp.data_to_run.test_i_ft_matrix,
#                                                                                NormalizationMethods.MinMax,
#                                                                                min_arr = train_norm_ft_min_arr,
#                                                                                max_arr = train_norm_ft_max_arr)
#     test_norm_sel_ft_matrix = test_norm_ft_matrix[:, sorted_top_ft_inx_arr]
#     fold_pred_matrix = regr_model.predict(test_norm_sel_ft_matrix)
#     
#     
#     ## TODO: evaluate fold performance  
#     test_ia_perf_matrix = exp.data_to_run.test_ia_perf_matrix[:, sorted_alg_per_cls]
#     test_ia_issolved_matrix = np.zeros((len(exp.data_to_run.test_inst_inx_arr), len(sorted_alg_per_cls)))
#     test_ia_rank_matrix = np.zeros((len(exp.data_to_run.test_inst_inx_arr), len(sorted_alg_per_cls)))
#     
#     inx = 0
#     for inst_inx in exp.data_to_run.test_inst_inx_arr:
#         test_ia_issolved_matrix[inx] = dio.ia_issolved_matrix[inst_inx][sorted_alg_per_cls]
#         test_ia_rank_matrix[inx] = dio.ia_rank_matrix[inst_inx][sorted_alg_per_cls]
#         inx += 1
#     
# #     test_ia_issolved_matrix = dio.ia_issolved_matrix[exp.data_to_run.test_inst_inx_arr, sorted_alg_per_cls]
# #     test_ia_rank_matrix = dio.ia_rank_matrix[exp.data_to_run.test_inst_inx_arr, sorted_alg_per_cls]
#     
#     num_solved, par10, avg_rank = evaluate_pred_matrix(test_ia_perf_matrix, test_ia_issolved_matrix, test_ia_rank_matrix, fold_pred_matrix)
#     perc_solved = (num_solved / float(len(test_ia_perf_matrix)))
#     
#     print("FOLD #", (plot_uniq_inx+1)," - # num_solved: ", num_solved, "par10_ss: ", par10, "avg_rank_ss: ", avg_rank)
#     logger.info("FOLD #%d - # solved instances: %d out of %d (%.3f), Par10: %.2f, Avg Rank: %.2f" %((plot_uniq_inx+1), num_solved, len(test_ia_perf_matrix), perc_solved, par10, avg_rank))
# 
# 
# 
#     return fold_pred_matrix, sorted_alg_per_cls





def apply_ain(csv_perf_file, higher_better, ft_file, exp):
    '''
        Apply AINovelty to a given instance-algorithm performance dataset
    '''
    global __output_folder__
    __output_folder__ = os.path.join(__output_folder__, exp.output_folder_name)
    if not os.path.exists(__output_folder__):
        os.makedirs(__output_folder__)

    ## to keep a log file for each run
    # logging.basicConfig(filename=os.path.join(__output_folder__, "AIN.log"), format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logger.propagate = False ## prevent console logs
    logger.setLevel(logging.DEBUG)
    handler = MyFileHandler(__output_folder__, logger, logging.FileHandler)
    ## for formatting: http://stackoverflow.com/questions/11581794/how-do-i-change-the-format-of-a-python-log-message-on-a-per-logger-basis
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # logger.info('run specific logger')


    logger.debug("Run configuration: svd_dim = %s, svd_type = %s, svd_outlier_threshold = %d, dim_rd_type = %s, clustering method = %s, ft_selection_method = %s, inst_clst_ft_type = %s"
                  %(str(exp.svd_dim),
                    str(exp.svd_type),
                    exp.svd_outlier_threshold,
                    str(exp.dim_rd_type),
                    str(exp.clst_method),
                    str(exp.ft_selection_method),
                    str(exp.inst_clst_ft_type)))


    # logging.debug("Output folder %s is created" %(output_folder_name))
    logger.debug("Output folder %s is created" %(exp.output_folder_name))

    

    # load and process ASlib data
    dio = DataIO()
    dio.load_process_csv(csv_perf_file, higher_better, ft_file)
    
    
    conf_list, numeric_conf_list = extract_alg_confs(dio.alg_list)
    conf_matrix = np.array(numeric_conf_list).astype(float)
    
    logger.debug("Started processing dataset %s (num_algs: %d, num_insts: %d, num_features: %d)" %(dio.dataset_name,
                                                                                                   dio.num_algs,
                                                                                                   dio.num_insts,
                                                                                                   dio.num_features))
    
    print "dio.unsolved_inst_list: ", dio.unsolved_inst_list

    if exp.to_report:
        dio.report_eval(os.path.join(__output_folder__, dio.dataset_name+"-report.txt"))
      
      
    '''
        Instance feature processing:
        Determine a representative subset of features
    '''  

    # extract latent (hidden) features for instances (Ur) and algorithms (Vr.T)
    i_latent_matrix, i_latent_matrix_for_ft, a_latent_matrix, sr_full, svd_dim = extract_latent_matrices(dio.ia_rank_matrix, 
                                                                                                         exp.svd_type, 
                                                                                                         exp.svd_dim, 
                                                                                                         exp.svd_outlier_threshold)
    logger.debug("Latent (hidden) features are extracted from the rank matrix")    
        
        
    # choose a subset of representative instance features (via classification, TODO: Regression )
    norm_ft_matrix, norm_ft_min_arr, norm_ft_max_arr = gen_norm(dio.i_ft_matrix, NormalizationMethods.MinMax)
    
    if exp.ft_subset_selection:
        top_ft_importance_arr, top_ft_inx_arr, ft_importance_arr, num_mapped_clusters_Ur = choose_features(norm_ft_matrix, 
                                                                                                           i_latent_matrix_for_ft, 
                                                                                                           exp.ft_selection_method, 
                                                                                                           ft_outlier_threshold_g = exp.ft_outlier_threshold)

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
        
        
    sorted_top_ft_inx_arr = np.sort(top_ft_inx_arr)
    logger.debug("(%d out of %d) Features to keep are determined: %s" % (len(sorted_top_ft_inx_arr), dio.num_features, str(sorted_top_ft_inx_arr.tolist()).replace("[","").replace("]","").strip()))
        
    
    if exp.to_plot:
        plot_ft_importance_from_clsf(dio.dataset_name, 
                                     ft_importance_arr, 
                                     top_ft_inx_arr, 
                                     num_mapped_clusters_Ur, 
                                     'FS-Gini-', 
                                     output_folder=__output_folder__, 
                                     to_show = False)        
        
        
     
    '''
        Algorithm configuration processing:
        Analyze parameter importance and similarity
    '''
     
    # choose a subset of representative algorithms
    alg_per_cls, centroids_alg, labels_alg, num_clusters_alg = choose_rep_alg_subset(a_latent_matrix.T, 
                                                                                     AlgorithmSubsetSelectionDataType.AvgRank,
                                                                                     a_perf_avg=dio.a_perf_avg, 
                                                                                     a_rank_avg=dio.a_rank_avg, 
                                                                                     a_solved_total=dio.a_solved_total, 
                                                                                     clustering_method = exp.clst_method,
                                                                                     k_max = 100)
    sorted_alg_per_cls = np.sort(alg_per_cls)
    logger.debug("(%d out of %d) Algorithms/solvers to keep are determined: %s" % (len(sorted_alg_per_cls), dio.num_algs, str(sorted_alg_per_cls.tolist()).replace("[","").replace("]","").strip()))
    print("alg_per_cls : ", alg_per_cls)
    # print(a_latent_matrix.T)
    if exp.to_plot:
        plot_2d_scatter_subset_with_cluster(dio.dataset_name, 
                                            a_latent_matrix.T, 
                                            alg_per_cls, 
                                            labels_alg,
                                            'Algorithms-',
                                            plt_annt_list = dio.alg_list, 
                                            marker_size = 30, 
                                            dim_reduction_type = exp.dim_rd_type, 
                                            hide_axis_labels = True, 
                                            output_folder=__output_folder__, 
                                            to_show = False)
     
     
    # check parameter importance (for the whole datasets)
    norm_conf_matrix, norm_conf_min_arr, norm_conf_max_arr = gen_norm(conf_matrix, NormalizationMethods.MinMax)
    
    conf_scores, top_conf_inx_arr, conf_importance_arr, conf_num_clusters = parameter_importance(norm_conf_matrix, 
                                                                                                 a_latent_matrix.T, 
                                                                                                 exp.ft_outlier_threshold)
    
    if exp.to_plot:
        plot_conf_importance_from_clsf(dio.dataset_name, 
                                     conf_importance_arr, 
                                     top_conf_inx_arr, 
                                     conf_num_clusters, 
                                     'Conf-Gini-', 
                                     output_folder=__output_folder__, 
                                     to_show = False)        




    # logger.debug("Finished processing dataset %s\n" %(dio.dataset_name))
    logger.debug("Finished processing dataset %s" %(dio.dataset_name))



    '''
        Instance analysis
    '''

    # choose a subset of representative instances
    if exp.inst_subset_selection:
         
        to_change = False ## False - set True just for test
        while True:
            if not to_change:
                inst_clst_ft_matrix = i_latent_matrix
            else:
                inst_clst_ft_matrix = np.concatenate((dio.i_ft_matrix, i_latent_matrix), 1)
             
             
            ## earlier, used this function: choose_rep_subset(...)
            ## new - choose_rep_subset_insts_as_reduction
            inst_per_cls, centroids_inst, labels_inst, num_clusters_inst = choose_rep_subset_insts_as_reduction(inst_clst_ft_matrix,
                                                                                                                 [], ## new parameter
                                                                                                                 dio.i_rank_div_std,
                                                                                                                 criterion_higher_better=False,
                                                                                                                 clustering_method = exp.clst_method,
                                                                                                                 k_max=dio.num_insts) ## dio.num_insts/3    
             
             
            ### break ## just for test 
             
             
            if num_clusters_inst <= (dio.num_insts*0.01):
                if to_change:
                    break
                else:
                    to_change = True
            else:
                break
     
        list_of_clst_insts_lists = get_clst_info(centroids_inst,
                                                 labels_inst,
                                                 num_clusters_inst)
        avg_clst_rank_matrix = calc_avg_clst_rank_matrix(dio.ia_rank_matrix,
                                                         list_of_clst_insts_lists)
        #print("avg_clst_rank_matrix: ", avg_clst_rank_matrix)
        clst_rank_score = calc_clst_rank_score(avg_clst_rank_matrix)
        print("clst_rank_score: ", clst_rank_score)
 
    else: # no instance subset selection
         
        inst_per_cls = np.zeros(dio.num_insts)
        for inst_inx in range(dio.num_insts):
            inst_per_cls[inst_inx] = inst_inx
         
        sorted_inst_per_cls = inst_per_cls
        labels_inst = inst_per_cls
        num_clusters_inst = dio.num_insts
         
     
    print "(#clusters= ",num_clusters_inst,")# instances are selected: ", len(inst_per_cls), " out of ", dio.num_insts, " (", (100 * len(inst_per_cls) / float(dio.num_insts)), ")"
 
    sorted_inst_per_cls = np.sort(inst_per_cls)
    print("inst_per_cls : ", inst_per_cls)
    logger.debug("(%d out of %d) Instances to keep are determined: %s" % (len(sorted_inst_per_cls),
                                                                          len(dio.ia_perf_matrix),
                                                                          str(sorted_inst_per_cls.tolist()).replace("[","").replace("]","").strip()))
    if exp.to_plot:
        plot_2d_scatter_subset_with_cluster(dio.dataset_name,
                                            inst_clst_ft_matrix,
                                            inst_per_cls,
                                            labels_inst,
                                            'Instances-',
                                            plt_annt_list = dio.inst_list, ## for annotation
                                            marker_size = 30,
                                            dim_reduction_type = exp.dim_rd_type,
                                            hide_axis_labels = True,
                                            output_folder=__output_folder__,
                                            to_show = False)







    



def main():

    ## TODO : pass args (to run from command prompt or easily change)

    # E:\_DELLNtb-Offce\_Eclipse Helios\WorkspaceFreiburg\_ASlib-Benchmarks
    # "/home/misir/Desktop/Bitbucket-SRC/ainovelty/_ASlib-Benchmarks"
    # "E:\_FREIBURG/aslib_data-aslib-v1.1"
    # bench_root_folder = "E:\_FREIBURG/aslib_data-aslib-v1.1"
#     csv_file_name = "E:/_DELLNtb-Offce/_Bitbucket SRC/ainovelty/data/openml-bagging-accuracy-for-ARS.csv"
#     ft_file_name = "E:/_DELLNtb-Offce/_Bitbucket SRC/ainovelty/data/openml-features.txt"
#    csv_file_name = "E:/_DELLNtb-Offce/_Bitbucket SRC/ainovelty/data/product_all_withall-error.csv"
#    ft_file_name = "E:/_DELLNtb-Offce/_Bitbucket SRC/ainovelty/data/product_all_withall-features.txt"
    
    csv_file_name = "E:/_DELLNtb-Offce/_Bitbucket SRC/ainovelty/data/SGD-avg-scores.csv"
    ft_file_name = "E:/_DELLNtb-Offce/_Bitbucket SRC/ainovelty/data/SGD-avg-dataset-features.txt"
    
    exp = Experiment()
    
    
    # False for product_all_withall-error etc.
    higher_better = True   ## referring to the given performance data

    apply_ain(csv_file_name, higher_better, ft_file_name, exp)


if __name__ == "__main__":
    main()


