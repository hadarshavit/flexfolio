'''

    TODO: change experiment parameters/details from class to dictionary
'''

import datetime

from data_io import *
# from evaluator import *
from ml_util import *
from plot_util import *
from ia_analyzer import *
from feature_analyzer import *
from utils import *


#TODO - complete
class DataToRun(object):

    def __init__(self, train_ia_perf_matrix, train_i_ft_matrix, test_ia_perf_matrix, test_i_ft_matrix,
                 train_inst_inx_arr, test_inst_inx_arr, 
                 train_unsolved_inst_list, test_unsolved_inst_list,
                 a_perf_avg, a_rank_avg, a_solved_total):

        self.train_ia_perf_matrix = train_ia_perf_matrix
        self.train_i_ft_matrix = train_i_ft_matrix
        self.test_ia_perf_matrix = test_ia_perf_matrix
        self.test_i_ft_matrix = test_i_ft_matrix
        self.train_inst_inx_arr = train_inst_inx_arr
        self.test_inst_inx_arr = test_inst_inx_arr
        
        self.train_unsolved_inst_list = train_unsolved_inst_list
        self.test_unsolved_inst_list = test_unsolved_inst_list

        self.num_insts = len(train_ia_perf_matrix)
        self.num_algs = len(train_ia_perf_matrix[0])
        self.num_features = len(train_i_ft_matrix[0])

        self.a_perf_avg = a_perf_avg
        self.a_rank_avg = a_rank_avg
        self.a_solved_total = a_solved_total

    def predict(self, pred_matrix):
        print("TODO ??")

    def evaluate(self):
        print("TODO ??")


class Experiment(object):
    
    
    alg_subset_selection = True ## True 
    inst_subset_selection = False  # # True 
    ft_subset_selection = True ## True
    alg_subset_criterion = "threholdPAR10"
    
    
    def __init__(self):
        '''
            data_to_run should be set for further use
        '''

        ## run parameters
        self.inst_clst_ft_type = InstanceClusteringFTType.Latent
 
        self.svd_type = SVDType.kSVD  ## default: kSVD
        self.dim_rd_type = DimReductionType.MDS ##MDS
        self.clst_method = ClusteringMethods.KMeansSLHadaptive
        self.ft_selection_method = FeatureSelectionMethod.RegrGiniPercentileMultiStep  ## default: GiniPercentile - RegrGiniPercentile - RegrGiniPercentileMultiStep

        self.svd_dim = 10 ## 10
        self.svd_outlier_threshold = 20 ## default: 20
        self.ft_outlier_threshold = 0 ## default: 40 - 20
        self.to_report = False
        self.to_plot = False ### False 
        ##TODO: add these options
        
 
        self.alg_subset_criterion = Experiment.alg_subset_criterion
        
        self.alg_subset_selection = Experiment.alg_subset_selection ## True 
        self.inst_subset_selection = Experiment.inst_subset_selection  # # True 
        self.ft_subset_selection = Experiment.ft_subset_selection ## True
        
        self.ft_postprocessing = False ## True


        self.output_folder_name = datetime.datetime.now().strftime("%d%m%Y%H%M%S%f")
        

        self.data_to_run = None


