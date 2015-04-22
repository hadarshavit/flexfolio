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
                 train_inst_inx_arr, test_inst_inx_arr, a_perf_avg, a_rank_avg, a_solved_total):

        self.train_ia_perf_matrix = train_ia_perf_matrix
        self.train_i_ft_matrix = train_i_ft_matrix
        self.test_ia_perf_matrix = test_ia_perf_matrix
        self.test_i_ft_matrix = test_i_ft_matrix
        self.train_inst_inx_arr = train_inst_inx_arr
        self.test_inst_inx_arr = test_inst_inx_arr

        self.num_insts = len(train_ia_perf_matrix)
        self.num_algs = len(train_ia_perf_matrix[0])
        self.num_features = len(train_i_ft_matrix[0])

        self.a_perf_avg = a_perf_avg
        self.a_rank_avg = a_rank_avg
        self.a_solved_total = a_solved_total

    def predict(self):
        print("TODO ??")


    def evaluate(self):
        print("TODO ??")


class Experiment(object):

    def __init__(self):
        '''
            data_to_run should be set for further use
        '''

        ## run parameters
        self.inst_clst_ft_type = InstanceClusteringFTType.Latent

        self.svd_type = SVDType.weightedSVD
        self.dim_rd_type = DimReductionType.MDS ##MDS
        self.clst_method = ClusteringMethods.KMeansSLHadaptive
        self.ft_selection_method = FeatureSelectionMethod.GiniPercentile

        self.svd_dim = 10 ## 10
        self.svd_outlier_threshold = 20
        self.ft_outlier_threshold = 40
        self.to_report = True
        self.to_plot = False

        ##TODO: add these options
        self.alg_subset_selection = True
        self.inst_subset_selection = True
        self.ft_subset_selection = False

        self.output_folder_name = datetime.datetime.now().strftime("%d%m%Y%H%M%S%f")

        self.data_to_run = None







