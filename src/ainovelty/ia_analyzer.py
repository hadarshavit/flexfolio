import warnings
import math
import scipy
import scipy.cluster.hierarchy as sch

from sets import Set

from data_io import *
from ml_util import *
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster, fclusterdata
from scipy.stats.stats import itemfreq
from ainovelty import settings


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


def choose_rep_subset_insts_as_reduction(ft_matrix, 
                                         unsolved_inst_list, 
                                         per_row_perf_criterion_arr, 
                                         criterion_higher_better, 
                                         clustering_method = ClusteringMethods.KMeansSLHadaptive, 
                                         num_inst_per_cluster = 1, 
                                         k_max=-1):
    '''
        Choose subsets wrt a given norm feature matrix (or latent matrix)
        representing either only instances
        
        unlike choose_rep_subset, exclude unsolved instances

        :param ft_matrix:
        :param per_row_perf_criterion_arr:
        :param criterion_higher_better:
        :param clustering_method:
        :param num_inst_per_cluster:
        :param k_max:
        :return:
    '''
    ##### cluster rows wrt latent features
    centroids, labels, num_clusters = gen_class_arr(ft_matrix, clustering_method, num_consc_wrs_steps=5, k_max=k_max)
    
    ##print "(@choose_rep_subset_insts_as_reduction) num_clusters= ", num_clusters
    
    num_rows = len(ft_matrix)
    
    ##### determine clustered instance groups
    clst_list = []
    for clst_inx in range(num_clusters):
        inst_list = []
        clst_list.append(inst_list)
        
    for inst_inx in range(num_rows):
        if inst_inx not in unsolved_inst_list: # add only solved instances
            clst_inx = int(labels[inst_inx])
            clst_list[clst_inx].append(inst_inx)

    clst_size_arr = np.zeros(num_clusters, dtype=np.int)
    clst_inx = 0
    for inst_list in clst_list:
        clst_size_arr[clst_inx] = len(inst_list)
        clst_inx += 1
        
        
    #outlier_arr, outlier_inx_arr = percentile_based_outlier(clst_size_arr, threshold=0.9)
    outlier_arr, outlier_inx_arr = percentile_based_except_min_outlier(clst_size_arr, threshold=0.9)
    
    
    ## TODO: might be returned here if True
    if len(outlier_arr) == 0:  ## if there is no any outlier
        ##print "(@choose_rep_subset_insts_as_reduction) outlier_arr = ", outlier_arr
        outlier_arr = np.zeros(num_rows, dtype=int)
        outlier_inx_arr = np.zeros(num_rows, dtype=int)
        for inst_inx in range(num_rows):
            outlier_arr[inst_inx] = 1
            outlier_inx_arr[inst_inx] = inst_inx 
            
        #print "(@choose_rep_subset_insts_as_reduction)-after change- outlier_arr = ", outlier_arr
        #print "(@choose_rep_subset_insts_as_reduction)-after change- outlier_inx_arr = ", outlier_inx_arr
        
    
    min_outlier = np.min(outlier_arr) ## got ValueError: zero-size array to reduction operation minimum which has no identity once???
    
    
    per_clst_sample_size = np.zeros(num_clusters, dtype=np.int)
    inx = 0
    for clst_inx in outlier_inx_arr:
        per_clst_sample_size[clst_inx] = math.ceil(outlier_arr[inx] / min_outlier)
        inx += 1
        
        
    ##### determine instances to include
    # sort instances wrt given criterion
    sorted_inst_inx_arr = np.argsort(per_row_perf_criterion_arr)
    if criterion_higher_better: # reverse sorted instance index list if criterion_higher_better
       sorted_inst_inx_arr = sorted_inst_inx_arr[::-1] 

    # create per cluster selected instance list (of lists)
    sel_inst_list = []
    for clst_inx in range(num_clusters):
        sel_inst_list.append([])
        
    # fill per cluster selected instance list  
    for inst_inx in sorted_inst_inx_arr: # for each instance (from sorted inst list)
        clst_inx = int(labels[inst_inx])
        
        if clst_inx in outlier_inx_arr: # if cluster is chosen for instances
            if len(sel_inst_list[clst_inx]) < per_clst_sample_size[clst_inx]:
                sel_inst_list[clst_inx].append(inst_inx)
        
    # convert list of lists for selected instances to numpy 1D array
    sel_inst_arr = np.hstack(sel_inst_list).astype(int)
    
    return sel_inst_arr, centroids, labels, num_clusters



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



def get_opt_alg_portfolio(ia_rank_matrix, unsolved_inst_list):
    '''
        Determine and return the optimal algorithm portfolio
        (achieving Oracle performance)
        
        :param ia_rank_matrix: numpy 2D array - instance-algorithm rank matrix
        :return alg_incl_arr: numpy array - binary array showing which algorithms to be included
    '''
    
    num_insts = len(ia_rank_matrix)
    num_algs = len(ia_rank_matrix[0])
    
    # 0-1 array for algorithms to be included array
    opt_portfolio = np.zeros((num_algs,), dtype=np.int)
    
    list = [] # per instance best algorithms' list (per-instance top ranked algorithms)
    
    best_alg_portfolio_list = []
    
    non_unique_best_alg_inst = []
    
    for inst_inx in range(num_insts): # for each instance
        
        if inst_inx == 92:
            print ""
        
        min_rank = np.min(ia_rank_matrix[inst_inx])
        pi_best_alg = np.argsort(ia_rank_matrix[inst_inx])
        
        pi_best_alg_list = [pi_best_alg[0]]
        for alg_inx in pi_best_alg[1:]: # for each algorithm from rank-based sorted arr
            if ia_rank_matrix[inst_inx][alg_inx] == min_rank:
                pi_best_alg_list.append(alg_inx)   
            else: ## no need to check rest since algorithms are ordered
                break

        
        ## keep track of algorithms that can be part of oracle portfolio
        for alg_inx in pi_best_alg_list:
            if alg_inx not in best_alg_portfolio_list: ## update best_alg_portfolio_list, if not added
                best_alg_portfolio_list.append(alg_inx)
                
    
        if len(pi_best_alg_list) > 1:
            non_unique_best_alg_inst.append(inst_inx)
    
    if len(non_unique_best_alg_inst) == 0 or set(non_unique_best_alg_inst) == set(unsolved_inst_list):
        opt_portfolio[best_alg_portfolio_list] = 1
    
    return opt_portfolio



# def get_opt_alg_portfolio_via_aai(ia_rank_matrix, unsolved_inst_list):
#     '''
#         Determine and return the optimal algorithm portfolio using aai 3d matrix
#         (achieving Oracle performance)
#          
#         :param ia_rank_matrix: numpy 2D array - instance-algorithm rank matrix
#         :return alg_incl_arr: numpy array - binary array showing which algorithms to be included
#     '''    
#     # generate Algorithm - Algorithm - Instance (aai) 3d comparison matrix
#     aai_3d_matrix = gen_aai_3d_matrix(ia_rank_matrix)
#      
#     num_insts = len(ia_rank_matrix)
#     num_algs = len(ia_rank_matrix[0])
#      
#     # determine non-inferior algorithms
#     for inst_inx in range(num_insts):
#         if inst_inx not in unsolved_inst_list: # for solved instances
#              
#             for alg_inx_1 in range(num_algs): # first algorithm to compare
#                 for alg_inx_2 in range(alg_inx_1+1, num_algs): # second algorithm to compare
#                      
#                     if aai_3d_matrix[alg_inx_1][alg_inx_2][inst_inx] 
       
       
def get_opt_alg_portfolio_via_aa_superior(ia_rank_matrix):
    '''
        Determine optimal set of the algorithm portfolio
    '''
    
    aa_s_matrix = gen_aa_superior_matrix(ia_rank_matrix)
    
    num_algs = len(ia_rank_matrix[0])
    
    opt_portfolio_set = set()
    
    for alg_inx_1 in range(num_algs):
        
        num_algs_inferior = (aa_s_matrix[alg_inx_1] == -1).sum()

#         print alg_inx_1, " - num_algs_inferior = ", num_algs_inferior

        if num_algs_inferior == 0:
            opt_portfolio_set.add(alg_inx_1)
            
#         for alg_inx_2 in range(alg_inx_1+1, num_algs):
#             if aa_s_matrix[alg_inx_1][alg_inx_2] == 1:
#                 opt_portfolio_set.add(alg_inx_1)
#             elif aa_s_matrix[alg_inx_1][alg_inx_2] == -1:
#                 opt_portfolio_set.add(alg_inx_2)
    
    return opt_portfolio_set
            
   
def gen_aa_superior_matrix(ia_rank_matrix):
    '''
       Generate Algorithm x Algorithm superior (1), equal (0), inferior (-1) matrix
        
    '''
    # generate Algorithm - Algorithm - Instance (aai) 3d comparison matrix
    aai_3d_matrix = gen_aai_3d_matrix(ia_rank_matrix)  
    
    num_algs = len(ia_rank_matrix[0])

    aa_s_matrix = np.zeros((num_algs, num_algs)) 
    
    for alg_inx_1 in range(num_algs): # first algorithm to compare
        for alg_inx_2 in range(alg_inx_1+1, num_algs): # second algorithm to compare
            
            num_insts_superior = (aai_3d_matrix[alg_inx_1][alg_inx_2] == 1).sum()
            num_insts_inferior = (aai_3d_matrix[alg_inx_1][alg_inx_2] == -1).sum()

#             print "# superior-inferior ", alg_inx_1, alg_inx_2, num_insts_superior, num_insts_inferior
              
            if num_insts_superior != 0 and num_insts_inferior == 0: # check whether alg_inx_1 is superior than alg_inx_2
                aa_s_matrix[alg_inx_1][alg_inx_2] = 1
                aa_s_matrix[alg_inx_2][alg_inx_1] = -1
            elif num_insts_inferior != 0 and num_insts_superior == 0: # check whether alg_inx_2 is superior than alg_inx_1
                aa_s_matrix[alg_inx_1][alg_inx_2] = -1
                aa_s_matrix[alg_inx_2][alg_inx_1] = 1
                
#                 print("Superior: ", alg_inx_1, alg_inx_2, aa_s_matrix[alg_inx_1][alg_inx_2])
            
    return aa_s_matrix

    
    
def gen_aai_3d_matrix(ia_rank_matrix):
    '''
        Generate Algorithm - Algorithm - Instance (aai) 3d matrix
        indicating whether an algorithm perform better, same, worse on each instance
        1: better, 0: equal, -1: worse 
        
        :param ia_rank_matrix: instance-algorithm rank matrix
        :return aai_3d_matrix: numpy 3d array - aai 3d matrix comparing algorithms (per instance)
    '''
    num_insts = len(ia_rank_matrix)
    num_algs = len(ia_rank_matrix[0])
    aai_3d_matrix = np.zeros((num_algs, num_algs, num_insts))
    
    for alg_inx_1 in range(num_algs): # for each algorithm
        for alg_inx_2 in range(alg_inx_1+1, num_algs): # for each algorithm (from alg_inx_1+1)
            for inst_inx in range(num_insts): # for each instance
                
                # compare two algorithms on instance inst_inx
                if ia_rank_matrix[inst_inx][alg_inx_1] < ia_rank_matrix[inst_inx][alg_inx_2]:
                    aai_3d_matrix[alg_inx_1][alg_inx_2][inst_inx] = 1
                    aai_3d_matrix[alg_inx_2][alg_inx_1][inst_inx] = -1
                elif ia_rank_matrix[inst_inx][alg_inx_1] > ia_rank_matrix[inst_inx][alg_inx_2]:
                    aai_3d_matrix[alg_inx_1][alg_inx_2][inst_inx] = -1
                    aai_3d_matrix[alg_inx_2][alg_inx_1][inst_inx] = 1

    return aai_3d_matrix


## TODO
def get_opt_alg_portfolio_for_num_solved_insts(ia_rank_matrix, ia_issolved_matrix):
    '''
        Determine optimal set of the algorithm portfolios for # solved instances
    '''
    
    opt_portfolio_set = set()

    num_insts = len(ia_rank_matrix)
    num_algs = len(ia_rank_matrix[0])
    
    # algorithm superiority-inferiority count wrt number of solved instances
    aa_si_cnt_matrix = np.zeros((num_algs, num_algs)) 
    
    for alg_inx_1 in range(num_algs): # for each algorithm
        for alg_inx_2 in range(alg_inx_1+1, num_algs): # for each algorithm (from alg_inx_1+1)
            
            for inst_inx in range(num_insts): # for each instance
                
                # compare algorithms whether one solved an instance when the other one couldn't
                if ia_issolved_matrix[inst_inx][alg_inx_1] > ia_issolved_matrix[inst_inx][alg_inx_2]:
                    aa_si_cnt_matrix[alg_inx_1][alg_inx_2] += 1
                elif ia_issolved_matrix[inst_inx][alg_inx_1] < ia_issolved_matrix[inst_inx][alg_inx_2]:
                    aa_si_cnt_matrix[alg_inx_2][alg_inx_1] += 1
                    
                    
    # determine optimal portfolio
    for alg_inx_1 in range(num_algs): # for each algorithm
        any_inferior = False
        for alg_inx_2 in range(alg_inx_1+1, num_algs): # for each algorithm (from alg_inx_1+1)
            
            if aa_si_cnt_matrix[alg_inx_1][alg_inx_2] > 0 and aa_si_cnt_matrix[alg_inx_2][alg_inx_1] == 0:
                opt_portfolio_set.add(alg_inx_1)
            elif aa_si_cnt_matrix[alg_inx_1][alg_inx_2] == 0 and aa_si_cnt_matrix[alg_inx_2][alg_inx_1] > 0:
                opt_portfolio_set.add(alg_inx_2)
                
            if aa_si_cnt_matrix[alg_inx_1][alg_inx_2] == 0:
                any_inferior = True
        
        if not any_inferior:
            opt_portfolio_set.add(alg_inx_1)
    
    return opt_portfolio_set



def choose_rep_alg_subset(a_latent_matrix, type, a_perf_avg = None, a_rank_avg = None, a_solved_total = None, clustering_method = ClusteringMethods.KMeansSLHadaptive, num_alg_per_cluster = 1, k_max=10000):
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

    # cluster algorithms wrt latent features
    centroids, labels, num_clusters = gen_class_arr(a_latent_matrix, clustering_method, num_consc_wrs_steps=5, k_max = k_max)

    # choose X number of algorithms from each cluster
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
            #if a_rank_avg[0][alg_inx] < best_per_alg_val[cls_inx]: ## check here
            if a_rank_avg[alg_inx] < best_per_alg_val[cls_inx]:
                alg_per_cls[cls_inx] = alg_inx
                #best_per_alg_val[cls_inx] = a_rank_avg[0][alg_inx] ## check here
                best_per_alg_val[cls_inx] = a_rank_avg[alg_inx]
        elif type == AlgorithmSubsetSelectionDataType.TotalSolved:
            # if a_solved_total[alg_inx] > best_per_alg_val[cls_inx]:
            if a_solved_total[0][alg_inx] > best_per_alg_val[cls_inx]:
                alg_per_cls[cls_inx] = alg_inx
                best_per_alg_val[cls_inx] = a_solved_total[0][alg_inx]
                

    return alg_per_cls, centroids, labels, num_clusters



## TODO: test this function !!!
def choose_alg_subset_via_hiearchical_clustering(ia_perf_matrix, a_latent_matrix, alg_cutoff_time, solved_inx_arr):
    
    z = sch.linkage(a_latent_matrix)
    
    p = sch.dendrogram(z)
    
    ### to plot hierarchical clustering dendrogram
    plot_tree(p)
    
    oracle_num_inst_solved, oracle_par10 = evaluate_oracle(ia_perf_matrix[solved_inx_arr,:], alg_cutoff_time)

    
    ## all algorithms (before exclusion)
    alg_portfolio = range(0, len(ia_perf_matrix.T))
    
    
    icoord = scipy.array( p['icoord'] )
    dcoord = scipy.array( p['dcoord'] )
    leaves = scipy.array( p['leaves'] )
    
    ## match leaves with icoord
    icoord_sort_inx_arr = np.argsort(icoord[:,0])
    
    print "leaves: ", leaves
    
    inx = 0
    num_leaves_processed = 0
    for ys in zip(dcoord):
        
        print "ys = ", ys
        
        if np.count_nonzero(ys) == 2: ## two leaf nodes
            
            ## get algorithm indices and keep only one 
            alg_pair = np.zeros((2), dtype=int)

            leaf_inx = np.where(icoord_sort_inx_arr == inx)[0][0]
            alg_pair[0] = leaves[leaf_inx]
            alg_pair[1] = leaves[leaf_inx+1]
            
            for alg_inx in alg_pair:
                
                ## remove algorithm from the portfolio
                inx_to_remove = np.where(alg_portfolio == alg_inx)[0][0]
                alg_portfolio = np.delete(alg_portfolio, inx_to_remove, 0)
                
                num_inst_solved, par10 = evaluate_oracle( (ia_perf_matrix[solved_inx_arr,:])[:,alg_portfolio], alg_cutoff_time)
                
#                 if par10 == oracle_par10:
#                     break
#                 else:
                if par10 != oracle_par10:
                    alg_portfolio = np.insert(alg_portfolio, inx_to_remove, alg_inx)
                else:
                    print alg_inx, " is removed from pair"
            
            num_leaves_processed += 2
            
        elif np.count_nonzero(ys) == 4: ## just a connector
            pass
        elif np.count_nonzero(ys) == 3: ## one leaf
            
            leaf_inx = np.where(icoord_sort_inx_arr == inx)[0][0]
            alg_inx = leaves[leaf_inx]
            
             ## remove algorithm from the portfolio
            inx_to_remove = np.where(alg_portfolio == alg_inx)[0][0]
            alg_portfolio = np.delete(alg_portfolio, inx_to_remove, 0)
            
            num_inst_solved, par10 = evaluate_oracle( (ia_perf_matrix[solved_inx_arr,:])[:,alg_portfolio], alg_cutoff_time)
            
            if par10 != oracle_par10:
                alg_portfolio = np.insert(alg_portfolio, inx_to_remove, alg_inx)
            else:
                print alg_inx, " is removed from single"
            
            num_leaves_processed += 1
            
        inx += 1
    
    return alg_portfolio, -1, alg_portfolio, len(alg_portfolio)
    


def choose_alg_subset_via_hiearchical_clustering_fcluster(ia_perf_matrix, a_latent_matrix, alg_cutoff_time, solved_inx_arr):
    
    dist_matrix = pdist(a_latent_matrix)
     
    z = sch.linkage(dist_matrix)
    
    p = sch.dendrogram(z)
    
    ### to plot hierarchical clustering dendrogram
    plot_tree(p)
    
    oracle_num_inst_solved, oracle_par10 = evaluate_oracle(ia_perf_matrix[solved_inx_arr,:], alg_cutoff_time)

    ## all algorithms (before exclusion)
    alg_portfolio = range(0, len(ia_perf_matrix.T))
    
    icoord = scipy.array( p['icoord'] )
    dcoord = scipy.array( p['dcoord'] )
    leaves = scipy.array( p['leaves'] )
    
    ## match leaves with icoord
    icoord_sort_inx_arr = np.argsort(icoord[:,0])
    
    print "leaves: ", leaves
    
    inx = 0
    num_leaves_processed = 0
    threshold_list = []
    for ys in zip(dcoord):
        threshold_list.append(ys[0][1])
    
    
#     assignments = fcluster(z, len(ia_perf_matrix.T), 'distance')
    
    processed_alg_inx_list = []
    for threshold in threshold_list:
        cluster_inxs = fclusterdata(a_latent_matrix, threshold, criterion= 'distance')
        cluster_inxs = cluster_inxs - 1 ##start cluster indices from zero, not 1
        
        ## get number of occurence of each cluster index
        freq_matrix = itemfreq(cluster_inxs)
        
        ## determine common cluster indices
        for (x,y), value in np.ndenumerate(freq_matrix):
            if y == 1 and value > 1:
                alg_inx_arr = np.where(cluster_inxs == freq_matrix[x][0])[0]
            
            
#                 if 5 in alg_inx_arr:
#                     pass
            
                ## keep only un-processed algorithms
                to_be_removed_alg_inx_list = []
                for inter_inx, inter_value in np.ndenumerate(alg_inx_arr):
                    if inter_value in processed_alg_inx_list:
                        to_be_removed_alg_inx_list.append(inter_inx)
                        ##alg_inx_arr = np.delete(alg_inx_arr, inter_inx, 0)
                    else:
                        processed_alg_inx_list.append(inter_value)
                
                if to_be_removed_alg_inx_list:
                    alg_inx_arr = np.delete(alg_inx_arr, to_be_removed_alg_inx_list, 0)       
                
                        
                ## remove each selected algorithm one by one, keep if no oracle par10 performance difference 
                for alg_inx in alg_inx_arr:
#                     if alg_inx == 5:
#                         pass
                    
                    print "alg_portfolio = ", alg_portfolio, " -- alg_inx = ", alg_inx
                    inx_to_remove = np.where(alg_portfolio == alg_inx)[0][0]
                    alg_portfolio = np.delete(alg_portfolio, inx_to_remove, 0)
                
                    num_inst_solved, par10 = evaluate_oracle( (ia_perf_matrix[solved_inx_arr,:])[:,alg_portfolio], alg_cutoff_time)
                    
    #                 if par10 == oracle_par10:
    #                     break
    #                 else:
                    ##if par10 != oracle_par10:
                    ##if par10 >= oracle_par10*1.1:
                    if num_inst_solved < oracle_num_inst_solved:     
                        alg_portfolio = np.insert(alg_portfolio, inx_to_remove, alg_inx)
                    else:
                        print alg_inx, " is removed from pair - new par10: ", par10
                  
                      
    return alg_portfolio, -1, alg_portfolio, len(alg_portfolio)
    
  
  
def choose_alg_subset_via_hiearchical_clustering_fcluster_kthbest(ia_perf_matrix, a_latent_matrix, alg_cutoff_time, solved_inx_arr,
                                                                  unique_name=None,
                                                                  title=None,
                                                                  output_folder=None, 
                                                                  k = 3,
                                                                  alg_subset_criterion = "threholdPAR10"):
    
     
    dist_matrix = pdist(a_latent_matrix)
    z = sch.linkage(dist_matrix)
    
    p = sch.dendrogram(z)
    
    ### to plot hierarchical clustering dendrogram
    # plot_tree(p, unique_name, title, output_folder)
    ####################################################
    
    
    oracle_num_inst_solved, oracle_par10 = evaluate_oracle(ia_perf_matrix[solved_inx_arr,:], alg_cutoff_time)
    kbest_oracle_num_inst_solved, kbest_oracle_par10 = evaluate_oracle_kthbest(ia_perf_matrix[solved_inx_arr,:], alg_cutoff_time, k)

    ## all algorithms (before exclusion)
    alg_portfolio = range(0, len(ia_perf_matrix.T))
    
    icoord = scipy.array( p['icoord'] )
    dcoord = scipy.array( p['dcoord'] )
    leaves = scipy.array( p['leaves'] )
    
    ## match leaves with icoord
    icoord_sort_inx_arr = np.argsort(icoord[:,0])
    
    ##print "leaves: ", leaves
    
    inx = 0
    num_leaves_processed = 0
    threshold_list = []
    for ys in zip(dcoord):
        threshold_list.append(ys[0][1])
    
    
#     assignments = fcluster(z, len(ia_perf_matrix.T), 'distance')
    
    processed_alg_inx_list = []
    for threshold in threshold_list:
        cluster_inxs = fclusterdata(a_latent_matrix, threshold, criterion= 'distance')
        cluster_inxs = cluster_inxs - 1 ##start cluster indices from zero, not 1
        
        ## get number of occurence of each cluster index
        freq_matrix = itemfreq(cluster_inxs)
        
        ## determine common cluster indices
        for (x,y), value in np.ndenumerate(freq_matrix):
            if y == 1 and value > 1:
                alg_inx_arr = np.where(cluster_inxs == freq_matrix[x][0])[0]
            
#                 if 5 in alg_inx_arr:
#                     pass
            
                ## keep only un-processed algorithms
                to_be_removed_alg_inx_list = []
                for inter_inx, inter_value in np.ndenumerate(alg_inx_arr):
                    if inter_value in processed_alg_inx_list:
                        to_be_removed_alg_inx_list.append(inter_inx)
                        ##alg_inx_arr = np.delete(alg_inx_arr, inter_inx, 0)
                    else:
                        processed_alg_inx_list.append(inter_value)
                
                if to_be_removed_alg_inx_list:
                    alg_inx_arr = np.delete(alg_inx_arr, to_be_removed_alg_inx_list, 0)       
                
                        
                ## remove each selected algorithm one by one, keep if no oracle par10 performance difference 
                for alg_inx in alg_inx_arr:
#                     if alg_inx == 5:
#                         pass
                    
                    ##print "alg_portfolio = ", alg_portfolio, " -- alg_inx = ", alg_inx
                    inx_to_remove = np.where(alg_portfolio == alg_inx)[0][0]
                    alg_portfolio = np.delete(alg_portfolio, inx_to_remove, 0)
                
                
                    num_inst_solved, par10 = evaluate_oracle( (ia_perf_matrix[solved_inx_arr,:])[:,alg_portfolio], alg_cutoff_time)
                    kbest_num_inst_solved, kbest_par10 = evaluate_oracle_kthbest( (ia_perf_matrix[solved_inx_arr,:])[:,alg_portfolio], alg_cutoff_time, k )
                    
    #                 if par10 == oracle_par10:
    #                     break
    #                 else:
                    ##if par10 != oracle_par10:
                    ##if par10 >= oracle_par10*1.1:
                    ##if kbest_num_inst_solved < kbest_oracle_num_inst_solved or num_inst_solved < oracle_num_inst_solved:
                    ##if kbest_num_inst_solved < kbest_oracle_num_inst_solved:
                    if check_alg_not_removed(alg_subset_criterion, 
                                          par10, 
                                          num_inst_solved, 
                                          kbest_num_inst_solved, 
                                          kbest_par10, oracle_par10, 
                                          oracle_num_inst_solved, 
                                          kbest_oracle_par10, 
                                          kbest_oracle_num_inst_solved):        
                        alg_portfolio = np.insert(alg_portfolio, inx_to_remove, alg_inx)
                    else:
                        if len(alg_portfolio) == 2: ## if the number of remaining algorithms degrade to 2, then stop removing    
                            return alg_portfolio, -1, alg_portfolio, len(alg_portfolio)
                        ##print alg_inx, " is removed from pair - new par10: ", kbest_par10
                  
                       
    return alg_portfolio, -1, alg_portfolio, len(alg_portfolio)


def check_alg_not_removed(alg_subset_criterion, 
                       par10, 
                       num_inst_solved, 
                       kbest_num_inst_solved, 
                       kbest_par10, 
                       oracle_par10, 
                       oracle_num_inst_solved,
                       kbest_oracle_par10, 
                       kbest_oracle_num_inst_solved):
    '''
        check whether an algorithm should not be removed
    '''
    if alg_subset_criterion == "AthresholdPAR10":
        ##if par10 >= oracle_par10*2:
        if par10 >= kbest_oracle_par10:
            return True
    elif alg_subset_criterion == "thresholdPAR10":
        if par10 >= oracle_par10*1.1:
            return True
    elif alg_subset_criterion == "thresholdNSolved":
        if num_inst_solved <= oracle_num_inst_solved*0.9:  
            return True        
    elif alg_subset_criterion == "NSolved":
        if num_inst_solved < oracle_num_inst_solved:        
            return True    
    elif alg_subset_criterion == "kthBestNSolved":
        if kbest_num_inst_solved < kbest_oracle_num_inst_solved:
            return True
    elif alg_subset_criterion == "kthBestBothNSolved":
        if kbest_num_inst_solved < kbest_oracle_num_inst_solved or num_inst_solved < oracle_num_inst_solved:
            return True
       
    return False 
               


     
    
def plot_tree( P, unique_name, title, output_folder, pos=None ):
    '''
    '''
    icoord = scipy.array( P['icoord'] )
    dcoord = scipy.array( P['dcoord'] )
    color_list = scipy.array( P['color_list'] )
    xmin, xmax = icoord.min(), icoord.max()
    ymin, ymax = dcoord.min(), dcoord.max()
    if pos:
        icoord = icoord[pos]
        dcoord = dcoord[pos]
        color_list = color_list[pos]
    for xs, ys, color in zip(icoord, dcoord, color_list):
        ##print "plt.plot(xs, ys,  color) = ", xs,", ", ys, ", ", color
        plt.plot(xs, ys,  color)
        ##plt.show()
    plt.xlim( xmin-10, xmax + 0.1*abs(xmax) )
    plt.ylim( ymin, ymax + 0.1*abs(ymax) )
#     plt.show()
    #plt.savefig("dendogram.pdf", format='pdf', dpi=100, bbox_inches='tight')
    plt.savefig(output_folder+"/"+title+"-"+unique_name+".pdf", format='pdf', dpi=100, bbox_inches='tight')
    plt.gcf().clear()
      


def parameter_importance(norm_conf_matrix, a_latent_matrix, conf_outlier_threshold_g):
    '''
        Analyze parameter importance for parametric algorithms
        
    '''
    centroids, labels, num_clusters = gen_class_arr(a_latent_matrix, ClusteringMethods.KMeansSLHadaptive, num_consc_wrs_steps = 5, k_max=10) ## k_max=len(norm_ft_matrix)/4
    clsf = gen_clsf_model(norm_conf_matrix, labels, ClassificationMethods.RFC)
    # TODO: add cv scores to the log file to check the quality of the feature selection model
    # cv_scores = apply_CV(clsf, norm_ft_matrix, labels, num_folds_g)
    # print("CV scores: ", cv_scores)
    conf_importance_arr = get_ft_importance(clsf, FeatureImportanceMethods.Gini)
    scores, top_conf_inx_arr = percentile_based_outlier(conf_importance_arr, conf_outlier_threshold_g)
    sorted_top_conf_inx_arr = np.sort(top_conf_inx_arr)
    print("top_ft_inx_arr: ", top_conf_inx_arr)

    return scores, top_conf_inx_arr, conf_importance_arr, num_clusters






def choose_inst_subset_via_hiearchical_clustering_fcluster_kthbest(ia_perf_matrix, i_latent_matrix, alg_cutoff_time, alg_portfolio, solved_inx_arr, k = 3):
    
    dist_matrix = pdist(i_latent_matrix)
     
    z = sch.linkage(dist_matrix)
    
    p = sch.dendrogram(z)
    
    ### to plot hierarchical clustering dendrogram
    plot_tree(p)
    
    num_insts = len(ia_perf_matrix)
    num_solved_insts = len(solved_inx_arr)
    
    
    oracle_num_inst_solved, oracle_par10 = evaluate_oracle(  (ia_perf_matrix[solved_inx_arr,:])[:, alg_portfolio], alg_cutoff_time)
    kbest_oracle_num_inst_solved, kbest_oracle_par10 = evaluate_oracle_kthbest( (ia_perf_matrix[solved_inx_arr,:])[:, alg_portfolio], alg_cutoff_time, k)
    
    oracle_num_inst_solved_ratio = oracle_num_inst_solved / float(num_solved_insts)
    kbest_oracle_num_inst_solved_ratio = kbest_oracle_num_inst_solved / float(num_solved_insts)

    ## all algorithms (before exclusion)
    inst_set = range(0, num_insts)
    ##inst_set = np.copy(solved_inx_arr)
    
    icoord = scipy.array( p['icoord'] )
    dcoord = scipy.array( p['dcoord'] )
    leaves = scipy.array( p['leaves'] )
    
    ## match leaves with icoord
    icoord_sort_inx_arr = np.argsort(icoord[:,0])
    
    print "leaves: ", leaves
    
    inx = 0
    num_leaves_processed = 0
    threshold_list = []
    for ys in zip(dcoord):
        threshold_list.append(ys[0][1])
    
    
#     assignments = fcluster(z, len(ia_perf_matrix.T), 'distance')
    
    processed_inst_inx_list = []
    for threshold in threshold_list:
        cluster_inxs = fclusterdata(i_latent_matrix, threshold, criterion= 'distance')
        cluster_inxs = cluster_inxs - 1 ##start cluster indices from zero, not 1
        
        ## get number of occurence of each cluster index
        freq_matrix = itemfreq(cluster_inxs)
        
        ## determine common cluster indices
        for (x,y), value in np.ndenumerate(freq_matrix):
            if y == 1 and value > 1:
                inst_inx_arr = np.where(cluster_inxs == freq_matrix[x][0])[0]
            
#                 if 5 in alg_inx_arr:
#                     pass
            
                ## keep only un-processed instances
                to_be_removed_inst_inx_list = []
                for inter_inx, inter_value in np.ndenumerate(inst_inx_arr):
                    if inter_value in processed_inst_inx_list:
                        to_be_removed_inst_inx_list.append(inter_inx)
                        ##alg_inx_arr = np.delete(alg_inx_arr, inter_inx, 0)
                    else:
                        processed_inst_inx_list.append(inter_value)
                
                if to_be_removed_inst_inx_list:
                    inst_inx_arr = np.delete(inst_inx_arr, to_be_removed_inst_inx_list, 0)       
                
                        
                ## remove each selected algorithm one by one, keep if no oracle par10 performance difference 
                for inst_inx in inst_inx_arr:
                    
                    
                    if inst_inx == 825:
                        pass
                    
                    print " -- inst_inx = ", inst_inx
                    
                    
                    inx_to_remove = np.where(inst_set == inst_inx)[0][0]
                    inst_set = np.delete(inst_set, inx_to_remove, 0)
                
                
                    if inst_inx not in solved_inx_arr:
                        continue
                
                    ### remove instances
                
                    ##insts_to_be_kept = np.concatenate((inst_set, solved_inx_arr))
                    ##insts_to_be_kept = np.sort(insts_to_be_kept)
                    
                    num_insts_kept = len(inst_set)
                
                    num_inst_solved, par10 = evaluate_oracle( (ia_perf_matrix[inst_set,:])[:, alg_portfolio], alg_cutoff_time)
                    kbest_num_inst_solved, kbest_par10 = evaluate_oracle_kthbest( (ia_perf_matrix[inst_set,:])[:, alg_portfolio], alg_cutoff_time, k )
                    
                    num_inst_solved_ratio = num_inst_solved / float(num_insts_kept)
                    kbest_num_inst_solved_ratio = kbest_num_inst_solved / float(num_insts_kept)

                    
    #                 if par10 == oracle_par10:
    #                     break
    #                 else:
                    ##if par10 != oracle_par10:
                    ##if par10 >= oracle_par10*1.1:
                    if abs(kbest_num_inst_solved_ratio - kbest_oracle_num_inst_solved_ratio) > 0.01 or abs(num_inst_solved_ratio - oracle_num_inst_solved_ratio) > 0.01:    
                        inst_set = np.insert(inst_set, inx_to_remove, inst_inx)
                    else:
                        print inst_inx, " instance is removed from pair - new par10: ", par10, "  +++  new kbest_par10: ", kbest_par10, " ++ num_insts_kept=",num_insts_kept
                        if len(inst_inx_arr) == 2:
                            break ## we should keep one instance at least from each pair ???
                      
        
        
#                 if len(inst_set) < len(solved_inx_arr) - 10: ## just to check whether there is any error from this point
#                     return inst_set, -1, inst_set, len(inst_set)
                      
    return inst_set, -1, inst_set, len(inst_set)
