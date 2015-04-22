'''
    IO and data processing operations compatible with
    ASlib data format (for now)
'''

import numpy as np
import os
import arff
import pprint
import itertools
import sys
import csv

from scipy.stats import rankdata
from sklearn.preprocessing import normalize
from ml_util import *

run_file = "algorithm_runs.arff"
inst_feature_cost_file = "feature_costs.arff"
inst_features_file = "feature_values.arff"
desc_file = "description.txt"
cv_file = "cv.arff"

num_folds = 10

def get_dataset_folder_list(bench_root_folder):
    '''
        Determine the benchmark folders residing a given folder (for ASlib)

        :param bench_root_folder: string - benchmark root folder path
        :return: list - benchmark folders
    '''
    bench_folders = []
    for root, dirs, files in os.walk(bench_root_folder):
        for dir_name in dirs:
            bench_folders.append(os.path.join(bench_root_folder, dir_name))

    ##print(bench_folders)
    return bench_folders


def get_data_file_lists(bench_folder_list):
    '''
        Determine all the data files for a given list of benchmarks (for ASlib)

        :param bench_folder_list: list - benchmark folder path list (can be set via get_dataset_folder_list())
        :return: list, list, list, list, list - performance (runtime) files,
                                                feature calculation cost files,
                                                instance features files,
                                                dataset description files,
                                                cross validation files
    '''

    bench_runtime_files = []
    bench_feature_cost_files = []
    bench_features_files = []
    bench_desc_files = []
    bench_cv_files = []
    for bench_folder in bench_folder_list:
        bench_runtime_files.append(os.path.join(bench_folder, run_file))
        bench_feature_cost_files.append(os.path.join(bench_folder, inst_feature_cost_file))
        bench_features_files.append(os.path.join(bench_folder, inst_features_file))
        bench_desc_files.append(os.path.join(bench_folder, desc_file))
        bench_cv_files.append(os.path.join(bench_folder, cv_file))

    return bench_runtime_files, bench_feature_cost_files, bench_features_files, bench_desc_files, bench_cv_files


def gen_rank_matrix(perf_matrix):
    '''
        Generate a rank matrix from a given performance matrix

        :param perf_matrix:
        :return:
    '''
    rank_matrix = np.empty(perf_matrix.shape, dtype=int)
    for k, row in enumerate(perf_matrix):
        rank_matrix[k] = rankdata(row, method='dense')

    return rank_matrix



def svd(ia_matrix):
    '''
        Apply singular value decomposition (SVD)

        :param ia_matrix: numpy 2D array - (instance, algorithm) matrix
        :return: numpy 2D array, numpy array, numpy 2D array - U matrix representing rows,
                                                                 s matrix (array) for singular values,
                                                                 V matrix representing columns
    '''
    U, s, V = np.linalg.svd(ia_matrix, full_matrices=False)
    return U, s, V


def ksvd(ia_matrix, k):
    '''
        Apply k-singular value decomposition (SVD)
        - Return matrices with k dimensions

        :param ia_matrix: numpy 2D array - (instance, algorithm) matrix
        :param k: int - SVD dimension
        :return: numpy 2D array, numpy array, numpy 2D array - Uk matrix representing rows,
                                                                 sk matrix (array) for singular values,
                                                                 Vk matrix representing columns
    '''
    max_k = min(len(ia_matrix), len(ia_matrix[0]))
    if k > max_k:
        k = max_k

    U, s, V = np.linalg.svd(ia_matrix, full_matrices=False)
    Uk = U[:,0:k]
    sk = s[0:k] #only diagonal values
    Vk = V[0:k,:]

    return Uk, sk, Vk, s


def svd_to_ksvd(U, s, V, k):
    '''
        Get only (first) k dimensions from given singular value decomposition (SVD) resulting matrices

        :param U: numpy 2D array - representing rows
        :param s: numpy array - singular values
        :param V: numpy 2D array - representing columns
        :param k: int - SVD dimension
        :return: numpy 2D array, numpy array, numpy 2D array - Uk matrix representing rows,
                                                                 sk matrix (array) for singular values,
                                                                 Vk matrix representing columns
    '''
    max_k = min(len(U), len(V))
    if k > max_k:
        k = max_k

    Uk = U[:,0:k]
    sk = s[0:k] #only diagonal values
    Vk = V[0:k,:]

    return Uk, sk, Vk


def auto_ksvd(ia_matrix, threshold=95):
    '''
        Automatically determine singular value decomposition (SVD) dimension (k) and apply kSVD

        :param ia_matrix: numpy 2D array - (instance, algorithm) matrix
        :param k: int - SVD dimension
        :return: numpy 2D array, numpy array, numpy 2D array - Uk matrix representing rows,
                                                                 sk matrix (array) for singular values,
                                                                 Vk matrix representing columns
    '''
    max_k = min(len(ia_matrix), len(ia_matrix[0]))
    U, s, V = np.linalg.svd(ia_matrix, full_matrices=False)

    outlier_arr, outlier_inx_arr = percentile_based_outlier(s, threshold)

    k = len(outlier_arr)

    Uk = U[:,0:k]
    sk = s[0:k] #only diagonal values
    Vk = V[0:k,:]

    return Uk, sk, Vk, s


def weighted_svd(ia_matrix):
    '''
        Apply singular value decomposition (SVD) and calculate weighted U and V matrices
        - Weighted matrices are calculated by multiplying each matrix element
          with the square root of its corresponding singular value

        :param ia_matrix: numpy 2D array - (instance, algorithm) matrix
        :return: numpy 2D array, numpy array, numpy 2D array, numpy 2D array, numpy 2D array -
                                                                 U matrix representing rows,
                                                                 s matrix (array) for singular values,
                                                                 V matrix representing columns
                                                                 Weighted U matrix,
                                                                 Weighted V matrix,
    '''
    U, s, V = np.linalg.svd(ia_matrix, full_matrices=False)

    # norm_s = np.zeros(shape=(len(s)))
    # min_s = np.min(s)
    # max_s = np.max(s)
    # for inx in range(len(s)):
    #     if min_s != max_s:
    #         norm_s[inx] = (s[inx] - min_s) / float(max_s - min_s) + 0.01
    #     else:
    #         norm_s[inx] = 1
    norm_s = [float(i)/sum(s) for i in s]
    #norm_s_diag = np.diag(norm_s)
    sqrt_s = np.sqrt(s)

    U_weighted = U * sqrt_s
    #V_weighted = np.dot(V.T, norm_s_diag).T
    V_weighted = (V.T * sqrt_s).T

    return U, s, V, U_weighted, V_weighted


def auto_weighted_svd(ia_matrix, threshold=95):
    print("TODO")


def extract_perf_ft_data_for_selected(ia_perf_matrix, i_ft_matrix, inst_list, alg_list, ft_list):
    '''
        Extract both performance and (instance) feature matrices
        for given subsets of instances and algorithms

        :param ia_perf_matrix: numpy 2D array - (instance, algorithm) performance matrix
        :param i_ft_matrix: numpy 2D array - instance feature matrix
        :param inst_list: numpy int array - selected instances
        :param alg_list: numpy int array - selected algorithms
        :param ft_list: numpy int array - selected instance features
        :return: numpy 2D array, numpy 2D array - performance and instance feature matrices
    '''
    sel_ia_perf_matrix = np.zeros(shape=(len(inst_list), len(alg_list)))
    sel_i_ft_matrix = np.zeros(shape=(len(inst_list), len(ft_list)))

    inst_inx = 0
    # for sel_inst_inx in itertools.izip(inst_list):
    for sel_inst_inx in inst_list:
        alg_inx = 0
        # for sel_alg_inx in itertools.izip(alg_list):
        for sel_alg_inx in alg_list:
            sel_ia_perf_matrix[inst_inx][alg_inx] = ia_perf_matrix[sel_inst_inx][sel_alg_inx]
            alg_inx += 1

        ft_inx = 0
        for sel_ft_inx in ft_list:
            sel_i_ft_matrix[inst_inx][ft_inx] = i_ft_matrix[sel_inst_inx][sel_ft_inx]
            ft_inx += 1

        inst_inx += 1

    return sel_ia_perf_matrix, sel_i_ft_matrix


def extract_latent_matrices(ia_matrix, svd_type, svd_dim, svd_outlier_threshold=95):
    '''
        Extract matrices composed of latent features (by applying singular value decomposition (SVD))
        from a given (instance, algorithm) performance (preferably rank) matrix

        :param ia_matrix: numpy 2D array - (instance, algorithm) matrix
        :param svd_type: SVDType - SVD type
        :param svd_dim: int - SVD dimension (k) (can be ignored if not applicable)
        :param svd_outlier_threshold: float - a value in [0, 100), used to determine
                                              the best SVD dimension (k) if applicable
                                              depending on svd_type
        :return: numpy 2D array, numpy 2D array, numpy 2D array, numpy array, int -
                                                                 a matrix for instances,
                                                                 a matrix for instance features,
                                                                 a matrix for algorithms,
                                                                 singular values,
                                                                 SVD dimension (k)
    '''

    i_latent_matrix = None
    a_latent_matrix = None
    i_latent_matrix_for_ft = None

    # extract latent (hidden) features for instances (Ur) and algorithms (Vr.T)
    Ur, sr, Vr, sr_full = None, None, None, None
    if svd_type == SVDType.SVD:
        Ur, sr, Vr, sr_full = ksvd(ia_matrix, svd_dim)
        i_latent_matrix = i_latent_matrix_for_ft = Ur
        a_latent_matrix = Vr
    elif svd_type == SVDType.kSVD:
        Ur, sr, Vr, sr_full = auto_ksvd(ia_matrix, threshold=svd_outlier_threshold)
        svd_dim = len(sr)
        i_latent_matrix = i_latent_matrix_for_ft = Ur
        a_latent_matrix = Vr
    elif svd_type == SVDType.weightedSVD:
        Ur, sr, Vr, Ur_weighted, Vr_weighted = weighted_svd(ia_matrix)
        i_latent_matrix = i_latent_matrix_for_ft = Ur_weighted
        # i_latent_matrix_for_ft = Ur
        a_latent_matrix = Vr_weighted
        sr_full = sr
        svd_dim = len(sr)
    elif svd_type == SVDType.autoWeightedSVD:
        print("TODO")
        sys.exit(0)

    print("sr: ", sr, " - svd_k: ", svd_dim)

    return i_latent_matrix, i_latent_matrix_for_ft, a_latent_matrix, sr_full, svd_dim


def percentile_based_outlier(data, threshold=95):
    '''
        Determine the top outliers in a given array

        :param data: numpy array -
        :param threshold:
        :return:
    '''
    diff = (100 - threshold) / 2.0
    minval, maxval = np.percentile(data, [diff, 100 - diff])
    # is_outlier_bool_arr = (data < minval) | (data > maxval)
    is_outlier_bool_arr = (data > maxval)

    num_outliers = np.sum(is_outlier_bool_arr)

    inx = 0
    outlier_inx_arr = np.zeros(shape=(num_outliers), dtype=np.int)
    outlier_arr = np.zeros(shape=(num_outliers))
    for i in range(len(is_outlier_bool_arr)):
        if is_outlier_bool_arr[i] == True:
            outlier_arr[inx] = data[i]
            outlier_inx_arr[inx] = i
            inx += 1

    return outlier_arr, outlier_inx_arr


## TODO: handle cases where only ia_perf_matrix is given besides pred_matrix
## TODO: ignore_unsolved_insts = True -> calculate ignoring unsolved instances 
def evaluate_pred_matrix(ia_perf_matrix, ia_issolved_matrix, ia_rank_matrix, pred_matrix):
    '''
        Evaluate a given prediction matrix
        - number of solved instances
        - par10
        - average rank

        :param ia_perf_matrix:
        :param ia_issolved_matrix:
        :param ia_rank_matrix:
        :param pred_matrix: numpy 2D array - runtime / rank prediction matrix
        :return: int, float, float - number of solved instances, par10, average rank
    '''
    num_solved = 0
    avg_rank = 0
    par10 = 0

    num_insts = len(pred_matrix)

    for inst_inx in range(num_insts):
        pi_pred_best_alg_inx = np.argmin(pred_matrix[inst_inx])

        num_solved += ia_issolved_matrix[inst_inx][pi_pred_best_alg_inx]
        avg_rank += ia_rank_matrix[inst_inx][pi_pred_best_alg_inx]
        if ia_issolved_matrix[inst_inx][pi_pred_best_alg_inx] == 1:
            par10 += ia_perf_matrix[inst_inx][pi_pred_best_alg_inx]
        else:
            par10 += (10 * ia_perf_matrix[inst_inx][pi_pred_best_alg_inx])

    avg_rank /= float(num_insts)
    par10 /= float(num_insts)

    return num_solved, par10, avg_rank



class SVDType:
    SVD, kSVD, autoSVD, weightedSVD, autoWeightedSVD  = range(0, 5)


class Performance(object):

    def __init__(self):
        self.num_inst_solved = 0
        self.num_inst_solved_per_alg = None
        self.par10_per_alg = None
        self.oracle_avg_rank = 0
        self.oracle_par10 = 0
        self.avg_rank_per_alg = None
        self.ndcg3_per_alg = None ### TODO

        self.rand_num_inst_solved = 0
        self.rand_avg_rank = 0
        self.rand_par10 = 0
        self.rand_ndcg3 = 0 ### TODO

        self.sb_issolved_algo = None
        self.sb_rank_algo = None
        self.sb_par10_algo = None
        self.sb_ndcg3_algo = None ### TODO

        self.sw_issolved_algo = None
        self.sw_rank_algo = None
        self.sw_par10_algo = None
        self.sw_ndcg3_algo = None ### TODO

    def print_perf(self):
        '''
            Print baseline performance details
        '''
        print("Oracle - num_inst_solved: ", self.num_inst_solved)
        print("Oracle - avg_rank: ", self.oracle_avg_rank)
        print("Oracle - par10: ", self.oracle_par10)

        print("Random - num_inst_solved: ", self.rand_num_inst_solved)
        print("Random - avg_rank: ", self.rand_avg_rank)

        print("SB - issolved algo (", np.max(self.num_inst_solved_per_alg) ,"): ", self.sb_issolved_algo)
        print("SB - rank algo (", np.min(self.avg_rank_per_alg) ,"): ", self.sb_rank_algo)
        print("SB - par10 algo (", np.min(self.par10_per_alg) ,"): ", self.sb_par10_algo)

        print("SW - issolved algo (", np.min(self.num_inst_solved_per_alg) ,"): ", self.sw_issolved_algo)
        print("SW - rank algo (", np.max(self.avg_rank_per_alg) ,"): ", self.sw_rank_algo)
        print("SW - par10 algo (", np.max(self.par10_per_alg) ,"): ", self.sw_par10_algo)


    def write_perf(self, out_perf_file, num_insts):
        '''
            Write baseline performance details to a file

            :param out_perf_file: string - file path
            :param num_insts: int - number of instances
        '''
        file = open(out_perf_file, "w")
        file.write("#Instances:\t%s\n#Algorithms:\t%s\n" % (num_insts, len(self.num_inst_solved_per_alg)))
        file.write("Oracle - num_inst_solved:\t%s\n" % self.num_inst_solved)
        file.write("Oracle - avg_rank:\t%s\n" % self.oracle_avg_rank)
        file.write("Oracle - par10:\t%s\n" % self.oracle_par10)

        file.write("Random - num_inst_solved:\t%s\n" % self.rand_num_inst_solved)
        file.write("Random - avg_rank:\t%s\n" % self.rand_avg_rank)

        file.write("SingleBest - num_inst_solved:\t%s\t%s\n" % (self.sb_issolved_algo, np.max(self.num_inst_solved_per_alg)))
        file.write("SingleBest - avg_rank:\t%s\t%s\n" % (self.sb_rank_algo, np.min(self.avg_rank_per_alg)))
        file.write("SingleBest - par10:\t%s\t%s\n" % (self.sb_par10_algo, np.min(self.par10_per_alg)))

        file.write("SingleWorst - num_inst_solved:\t%s\t%s\n" % (self.sw_issolved_algo, np.min(self.num_inst_solved_per_alg)))
        file.write("SingleWorst - avg_rank:\t%s\t%s\n" % (self.sw_rank_algo, np.max(self.avg_rank_per_alg)))
        file.write("SingleWorst - par10:\t%s\t%s\n" % (self.sw_par10_algo, np.max(self.par10_per_alg)))

        file.close()


class DataIO(object):

    def __init__(self):
    # def __init__(self, run_file, inst_feature_cost_file, inst_features_file, desc_file):
        # self.bench_root_folder = bench_root_folder
        # self.run_file = run_file
        # self.inst_feature_cost_file = inst_feature_cost_file
        # self.inst_features_file = inst_features_file
        # self.desc_file = desc_file
        self.dataset_name = None
        self.inst_list = []
        self.alg_list = []
        self.alg_cutoff_time = 0
        self.ft_cutoff_time = 0
        self.ia_perf_matrix = None
        self.ia_issolved_matrix = None
        self.ia_rank_matrix = None
        self.i_ft_matrix = None
        self.if_cv_matrix = None

        self.i_rank_div_std = None
        self.i_perf_div_std = None
        self.a_rank_div_std = None
        self.a_perf_div_std = None
        self.a_perf_avg = None
        self.a_rank_avg = None

        self.a_solved_total = None

        self.num_insts = -1
        self.num_algs = -1
        self.num_features = -1
        self.perf = Performance()

    def load_process(self, desc_file_path, features_file_path, runtime_file_path, cv_file_path):
        '''
            Load and perform initial data processing for a given dataset by
            specifying its separate dataset files (for ASlib)
            (should be called first)

            :param desc_file_path: string - description file
            :param features_file_path: string - instance features file
            :param runtime_file_path: string - performance (currently runtime) file
            :param cv_file_path: string - cross validation file
        '''
        self.dataset_name = os.path.basename(os.path.dirname(desc_file_path))

        self.read_desc_file(desc_file_path)
        self.read_ft_file(features_file_path)
        self.read_perf_file(runtime_file_path)
        self.read_cv_file(cv_file_path)

        self.gen_rank_matrix()
        self.gen_issolved_matrix()

        self.gen_inst_rank_div_arr()
        self.gen_inst_perf_div_arr()

        self.gen_alg_rank_div_arr()
        self.gen_alg_perf_div_arr()

        self.gen_alg_perf_avg()
        self.gen_alg_rank_avg()

        self.gen_alg_solved_total()


    def add_to_list(self, str_to_add, list):
        '''
            Add a given string to a list if not added before

            :param str_to_add: string
            :param list: string list
            :return: a string list including the given new string
        '''
        if str_to_add not in list:
            list.append(str_to_add)


    def read_perf_file(self, perf_file_path):
        '''
            Read a given algorithm selection data performance file

            :param perf_file_path: string - algorithm selection data performance file path
            :return: numpy 2D array - instance-algorithm performance matrix
        '''

        if self.alg_cutoff_time <= 0:
            print("Algorithm cutoff time is inapplicable: ", self.alg_cutoff_time, "\nExiting...")
            sys.exit()

        runtime_data = arff.load(open(perf_file_path, 'rb'))

        self.num_algs = len(runtime_data["data"]) / self.num_insts

        self.ia_perf_matrix = np.zeros(shape=(self.num_insts, self.num_algs))

        inst_inx = 0
        alg_inx = 0
        for run in runtime_data["data"]:
            #print(run[3])
            #print(inst_inx, " ", alg_inx)
            self.add_to_list(run[0], self.inst_list)
            self.add_to_list(run[2], self.alg_list)

            if run[4] == "ok":
                self.ia_perf_matrix[inst_inx][alg_inx] = run[3]
            else:
                self.ia_perf_matrix[inst_inx][alg_inx] = self.alg_cutoff_time

            inst_inx += 1
            if inst_inx == self.num_insts:
                inst_inx = 0
                alg_inx += 1

        return self.ia_perf_matrix


    def read_cv_file(self, cv_file_path):
        '''
            Read a given cross validation file

            :param cv_file_path: string - cross validation file path
            :return: numpy 2D array - is_fold matrix indicating which instance (row) belongs to which fold (column)
        '''

        cv_data = arff.load(open(cv_file_path, 'rb'))

        self.if_cv_matrix = np.zeros(shape=(self.num_insts, num_folds))

        for cv_line in cv_data["data"]:

            inst_name = cv_line[0].strip()
            inst_inx = self.inst_list.index(inst_name)

            self.if_cv_matrix[inst_inx][int(cv_line[2])-1] = 1

        return self.if_cv_matrix


    def gen_issolved_matrix(self):
        '''
            Generate is_solved matrix from a given performance matrix
            A general performance analysis for the given performance data is also performed
            for oracle, random, single best solver, single worst solver
            - number of solved instances
            - par10
            - average rank

            :return: numpy 2D array: is_solved matrix indicating whether
                                      an instance (row) is solved by an algorithm (column)
        '''

        if self.num_insts <= 0:
            self.num_insts = len(self.ia_perf_matrix)

        if self.num_algs <= 0:
            self.num_algs = len(self.ia_perf_matrix[0])

        self.ia_issolved_matrix = np.zeros(shape=(self.num_insts, self.num_algs))
        self.perf.num_inst_solved_per_alg = np.zeros(shape=(self.num_algs))
        self.perf.par10_per_alg = np.zeros(shape=(self.num_algs))
        self.perf.avg_rank_per_alg = np.zeros(shape=(self.num_algs))

        for inst_inx in range(self.num_insts):

            self.perf.oracle_avg_rank += np.min(self.ia_rank_matrix[inst_inx])

            for alg_inx in range(self.num_algs):
                self.perf.avg_rank_per_alg[alg_inx] += self.ia_rank_matrix[inst_inx][alg_inx]

                if self.ia_perf_matrix[inst_inx][alg_inx] < self.alg_cutoff_time:
                    self.ia_issolved_matrix[inst_inx][alg_inx] = 1

                    self.perf.num_inst_solved_per_alg[alg_inx] += 1
                    self.perf.par10_per_alg[alg_inx] += self.ia_perf_matrix[inst_inx][alg_inx]
                else:
                    self.perf.par10_per_alg[alg_inx] += 10 * self.alg_cutoff_time


            per_inst_num_alg_solved = np.sum(self.ia_issolved_matrix[inst_inx])

            self.perf.rand_num_inst_solved += per_inst_num_alg_solved / self.num_algs

            if per_inst_num_alg_solved > 0:
                self.perf.num_inst_solved += 1
                self.perf.oracle_par10 += np.min(self.ia_perf_matrix)
            else:
                self.perf.oracle_par10 += 10 * self.alg_cutoff_time

        self.perf.rand_avg_rank = self.ia_rank_matrix.sum(dtype='float') / (self.num_insts * self.num_algs)

        self.perf.oracle_avg_rank /= self.num_insts
        self.perf.oracle_par10 /= self.num_insts

        for alg_inx in range(self.num_algs):
            self.perf.avg_rank_per_alg[alg_inx] /= self.num_insts
            self.perf.par10_per_alg[alg_inx] /= self.num_insts

        self.perf.sb_issolved_algo = self.alg_list[np.argmax(self.perf.num_inst_solved_per_alg)]
        self.perf.sb_rank_algo = self.alg_list[np.argmin(self.perf.avg_rank_per_alg)]
        self.perf.sb_par10_algo = self.alg_list[np.argmin(self.perf.par10_per_alg)]

        self.perf.sw_issolved_algo = self.alg_list[np.argmin(self.perf.num_inst_solved_per_alg)]
        self.perf.sw_rank_algo = self.alg_list[np.argmax(self.perf.avg_rank_per_alg)]
        self.perf.sw_par10_algo = self.alg_list[np.argmax(self.perf.par10_per_alg)]

        return self.ia_issolved_matrix


    def read_desc_file(self, desc_file_path):
        '''
            Read a given ASLlib description file

            :param desc_file_path: string - description file path
            :return: float, float - cutoff times for algorithm runs and feature calculation
        '''
        algct_found = False
        ftct_found = False

        #with open(desc_file_path) as f:
            #str_line = f.readlines()
        f = open(desc_file_path,'r')
        for str_line in f.readlines():
            #print(str_line)
            if "algorithm_cutoff_time" in str_line:
                val_str = str_line[str_line.index(':')+1:len(str_line)].strip()
                # print("1", val_str)
                if val_str != '?':
                    self.alg_cutoff_time = float(val_str)
                algct_found = True
            elif "features_cutoff_time" in str_line:
                val_str = str_line[str_line.index(':')+1:len(str_line)].strip()
                # print("2", val_str)
                if val_str != '?':
                    self.ft_cutoff_time = float(val_str)
                ftct_found = True

            if algct_found & ftct_found:
                f.close()
                break

        return self.alg_cutoff_time, self.ft_cutoff_time


    def read_ft_file(self, ft_file_path):
        '''
            Read a given instance features file

            :param ft_file_path: string - instance features file path
            :return: numpy 2D array - instance feature matrix
        '''

        inst_features_data = arff.load(open(ft_file_path, 'rb'))

        self.num_insts = len(inst_features_data["data"])
        self.num_features = len(inst_features_data["data"][0])-2

        self.i_ft_matrix = np.zeros(shape=(self.num_insts, self.num_features))

        inst_inx = 0
        for run in inst_features_data["data"]:
            ft_list = run[2:self.num_features+2]
            #print(ft_list)
            #svda.set_missing_features(ft_list, '0')
            #ft_list = [x for x in ft_list if x is not None]
            ft_list = [0 if v is None else v for v in ft_list]
            #print(ft_list)
            self.i_ft_matrix[inst_inx] = ft_list
            inst_inx += 1

        return self.i_ft_matrix


    ## TODO: chnage for the tie-cases
    ## TODO: add matrix type (e.g. smaller values are better)
    def gen_rank_matrix(self):
        '''
            Generate a rank matrix from a given performance matrix

            :return: numpy 2D array - instance algorithm rank matrix indicating
                                       algorithms' per instance ranks
        '''

        self.ia_rank_matrix = np.empty(self.ia_perf_matrix.shape, dtype=int)
        for k, row in enumerate(self.ia_perf_matrix):
            self.ia_rank_matrix[k] = rankdata(row, method='dense')

        return self.ia_rank_matrix


    def gen_inst_rank_div_arr(self):
        '''
            Calculate standard deviations over ranks for each instance

            :return: numpy float array - array of ranks' standard deviations
        '''
        self.i_rank_div_std = np.zeros(shape=(self.num_insts))
        for inst_inx in range(self.num_insts):
            self.i_rank_div_std[inst_inx] = np.std(self.ia_rank_matrix[inst_inx])

        return self.i_rank_div_std


    def gen_inst_perf_div_arr(self):
        '''
            Calculate performance diversity measure for each instance
            std / average

            :return: numpy float array -
        '''
        self.i_perf_div_std = np.zeros(shape=(self.num_insts))
        for inst_inx in range(self.num_insts):
            # self.i_perf_div_std[inst_inx] = np.std(self.ia_perf_matrix[inst_inx])
            self.i_perf_div_std[inst_inx] = np.std(self.ia_perf_matrix[inst_inx]) / np.average(self.ia_perf_matrix[inst_inx])

        return self.i_perf_div_std


    def gen_alg_rank_div_arr(self):
        '''
            Calculate standard deviations over ranks for each algorithm

            :return: numpy float array - ranks' standard deviations
        '''
        self.a_rank_div_std = np.zeros(shape=(self.num_algs))
        for alg_inx in range(self.num_algs):
            self.a_rank_div_std[alg_inx] = np.std(self.ia_rank_matrix.T[alg_inx])

        return self.a_rank_div_std


    def gen_alg_perf_div_arr(self):
        '''
            Calculate standard deviations over performance (metric) for each algorithm

            :return: numpy float array - standard deviations on algorithms' performance
        '''
        self.a_perf_div_std = np.zeros(shape=(self.num_algs))
        for alg_inx in range(self.num_algs):
            self.a_perf_div_std[alg_inx] = np.std(self.ia_perf_matrix.T[alg_inx])

        return self.a_perf_div_std


    def gen_alg_perf_avg(self):
        '''
            Calculate algorithms' average performance

            :return: numpy float array - algorithms' average performance
        '''
        self.a_perf_avg = np.zeros(shape=(self.num_algs))
        for alg_inx in range(self.num_algs):
            self.a_perf_avg[alg_inx] = np.average(self.ia_perf_matrix.T[alg_inx])

        return self.a_perf_avg


    def gen_alg_perf_avg_for_inst_subset(self, sel_inst_list):
        '''
            Calculate algorithms' average performance
            across a set of selected instances

            :param sel_inst_list: numpy array - selected instances
            :return: numpy float array - algorithms' average performance
        '''
        a_perf_avg = np.zeros(shape=(self.num_algs))
        for alg_inx in range(self.num_algs):
            a_perf_avg[alg_inx] = np.average(self.ia_perf_matrix.T[alg_inx, sel_inst_list])

        return a_perf_avg


    def gen_alg_rank_avg(self):
        '''
            Calculate algorithms' average ranks

            :return: numpy float array - algorithms' average ranks
        '''
        self.a_rank_avg = np.zeros(shape=(self.num_algs))
        for alg_inx in range(self.num_algs):
            self.a_rank_avg[alg_inx] = np.average(self.ia_rank_matrix.T[alg_inx])

        return self.a_rank_avg


    def gen_alg_rank_avg_for_inst_subset(self, sel_inst_list):
        '''
            Calculate algorithms' average ranks
            across a set of selected instances

            :param sel_inst_list: numpy array - selected instances
            :return: numpy float array - algorithms' average ranks
        '''
        a_rank_avg = np.zeros(shape=(self.num_algs))
        for alg_inx in range(self.num_algs):
            a_rank_avg[alg_inx] = np.average(self.ia_rank_matrix.T[alg_inx, sel_inst_list])

        return a_rank_avg


    def gen_alg_solved_total(self):
        '''
            Calculate number of solved instances by each algorithm

            :return: numpy int array - number of solved instances
        '''
        self.a_solved_total = np.zeros(shape=(self.num_algs), dtype=np.int)
        for alg_inx in range(self.num_algs):
            self.a_solved_total[alg_inx] = np.sum(self.ia_issolved_matrix.T[alg_inx])

        return self.a_solved_total


    def gen_alg_solved_total_for_inst_subset(self, sel_inst_list):
        '''
            Calculate number of solved instances by each algorithm
            across a set of selected instances

            :param sel_inst_list: numpy array - selected instances
            :return: numpy int array - number of solved instances
        '''
        a_solved_total = np.zeros(shape=(self.num_algs), dtype=np.int)
        for alg_inx in range(self.num_algs):
            a_solved_total[alg_inx] = np.sum(self.ia_issolved_matrix.T[alg_inx, sel_inst_list])

        return a_solved_total

    
    ## TODO: read csv formed performance files (additional feature to ASlib)
    def read_csv_perf_file(self, csv_file, ignore_first_row = False, ignore_first_col = False):
        '''
            Read a given csv performance file
            :param csv_file: 
        '''
        perf_data_list = []
        with open(csv_file, 'rb') as f:
            reader = csv.reader(f)
             
            if ignore_first_row:
                next(reader, None)  # skip the header
             
            for row in reader:
                if row: # skip if row (list) is empty
                    perf_data_list.append(row[1:len(row)])
                    print row
            
#       self.ia_perf_matrix = np.loadtxt(csv_file, delimiter=',', skiprows=1, usecols=range(1,...))
        self.ia_perf_matrix = np.array(perf_data_list)
        self.num_insts, self.num_algs = self.ia_perf_matrix.shape
        
            
        
        
    def extract_perf_ft_data_for_selected(self, inst_list, alg_list):
        '''
            Extract both performance and (instance) feature matrices
            for given subsets of instances and algorithms

            :param inst_list: numpy int array - selected instances
            :param alg_list: numpy int array - selected algorithms
            :return: numpy 2D array, numpy 2D array - performance and instance feature matrices
        '''

        sel_ia_perf_matrix = np.zeros(shape=(len(inst_list), len(alg_list)))
        sel_i_ft_matrix = np.zeros(shape=(len(inst_list), self.num_features))

        inst_inx = 0
        # for sel_inst_inx in itertools.izip(inst_list):
        for sel_inst_inx in inst_list:
            alg_inx = 0
            # for sel_alg_inx in itertools.izip(alg_list):
            for sel_alg_inx in alg_list:
                sel_ia_perf_matrix[inst_inx][alg_inx] = self.ia_perf_matrix[sel_inst_inx][sel_alg_inx]
                sel_i_ft_matrix[inst_inx] = self.i_ft_matrix[sel_inst_inx]
                alg_inx += 1

            inst_inx += 1

        return sel_ia_perf_matrix, sel_i_ft_matrix
    


    def extract_perf_ft_data_for_selected_v2(self, inst_list, alg_list, ft_list, after_norm = False):
        '''
            Extract both performance and (instance) feature matrices
            for given subsets of instances, algorithms and instance features

            :param inst_list: numpy int array - selected instances
            :param alg_list: numpy int array - selected algorithms
            :param ft_list: numpy int array - selected features
            :param after_norm: boolean - whether normalize feature matrix
            :return: numpy 2D array, numpy 2D array - performance and instance feature matrices
        '''

        ft_matrix = self.i_ft_matrix
        if after_norm == True:
            ft_matrix = gen_norm(self.i_ft_matrix, NormalizationMethods.MinMax)


        sel_ia_perf_matrix = np.zeros(shape=(len(inst_list), len(alg_list)))
        sel_i_ft_matrix = np.zeros(shape=(len(inst_list), len(ft_list)))

        inst_inx = 0
        # for sel_inst_inx in itertools.izip(inst_list):
        for sel_inst_inx in inst_list:
            alg_inx = 0
            # for sel_alg_inx in itertools.izip(alg_list):
            for sel_alg_inx in alg_list:
                sel_ia_perf_matrix[inst_inx][alg_inx] = self.ia_perf_matrix[sel_inst_inx][sel_alg_inx]
                alg_inx += 1

            ft_inx = 0
            for sel_ft_inx in ft_list:
                sel_i_ft_matrix[inst_inx][ft_inx] = ft_matrix[sel_inst_inx][sel_ft_inx]
                ft_inx += 1

            inst_inx += 1

        return sel_ia_perf_matrix, sel_i_ft_matrix


    def extract_fold_data(self, fold_inx):
        '''
            Extract fold data
            - instance - algorithm train matrix
            - instance - algorithm test matrix
            - instance feature train matrix
            - instance feature test matrix

            :param fold_inx: int - fold index
            :return: numpy 2D array, numpy 2D array, numpy 2D array, numpy 2D array, numpy array, numpy array -
                     train performance, train instance feature, test performance, test instance feature matrices,
                     train instances, test instances
        '''

        # if 0 < fold_inx or fold_inx > num_folds-1:
        #     print("Incorrect fold_inx: ", fold_inx, "\nExiting...")
        #     sys.exit(1)

        num_test_insts = np.sum(self.if_cv_matrix.T[fold_inx])
        num_train_insts = self.num_insts - num_test_insts

        # train_ia_perf_matrix = np.zeros(shape=(num_train_insts, self.num_algs))
        # test_ia_perf_matrix = np.zeros(shape=(num_test_insts, self.num_algs))
        #
        # train_i_ft_matrix = np.zeros(shape=(num_train_insts, self.num_features))
        # test_i_ft_matrix = np.zeros(shape=(num_test_insts, self.num_features))

        ## Set train performance matrix
        test_inst_inx_arr = np.where(self.if_cv_matrix.T[fold_inx] == 1)[0]
        train_inst_inx_arr = np.zeros(shape=(num_train_insts), dtype=np.int)
        inx = 0
        for inst_inx in range(self.num_insts):
            if not inst_inx in test_inst_inx_arr:
                train_inst_inx_arr[inx] = inst_inx
                inx += 1

        test_ia_perf_matrix = self.ia_perf_matrix[test_inst_inx_arr, :]
        train_ia_perf_matrix = self.ia_perf_matrix[train_inst_inx_arr, :]

        test_i_ft_matrix = self.i_ft_matrix[test_inst_inx_arr, :]
        train_i_ft_matrix = self.i_ft_matrix[train_inst_inx_arr, :]

        return train_ia_perf_matrix, train_i_ft_matrix, test_ia_perf_matrix, test_i_ft_matrix, train_inst_inx_arr, test_inst_inx_arr


    ## TODO : add other performance metrics
    def evaluate_pred_matrix(self, pred_matrix):
        '''
            Evaluate a given prediction matrix
            - number of solved instances
            - par10
            - average rank

            :param pred_matrix: numpy 2D array - runtime / rank prediction matrix
            :return: int, float, float - number of solved instances, par10, average rank
        '''
        # num_solved = 0
        # avg_rank = 0
        # par10 = 0
        #
        # for inst_inx in range(self.num_insts):
        #     pi_pred_best_alg_inx = np.argmin(pred_matrix[inst_inx])
        #
        #     num_solved += self.ia_issolved_matrix[inst_inx][pi_pred_best_alg_inx]
        #     avg_rank += self.ia_rank_matrix[inst_inx][pi_pred_best_alg_inx]
        #     if self.ia_issolved_matrix[inst_inx][pi_pred_best_alg_inx] == 1:
        #         par10 += self.ia_perf_matrix[inst_inx][pi_pred_best_alg_inx]
        #     else:
        #         par10 += (10 * self.ia_perf_matrix[inst_inx][pi_pred_best_alg_inx])
        #
        # avg_rank /= float(self.num_insts)
        # par10 /= float(self.num_insts)
        #
        # return num_solved, par10, avg_rank

        return evaluate_pred_matrix(self.ia_perf_matrix, self.ia_issolved_matrix, self.ia_rank_matrix, pred_matrix)


    def report_eval(self, out_perf_file, alg_subset = None, inst_subset = None, ft_subset = None, svd_s = None, svd_k = 0, baselines = True):
        '''
            Generate a report file
            - summarizing baselines' performance (Oracle, Random, Single Best Solver, Single Worst Solver)
            - selected instances, algorithms and instance features
            - some additional info, e.g. sorted singular values derived from the related rank matrix

            :param alg_subset: numpy int array - selected algorithms
            :param inst_subset: numpy int array - selected instances
            :param ft_subset: numpy int array - selected features
            :param out_perf_file: string - report file path
            :param svd_s: numpy float array - sorted singular values derived from the corresponding rank matrix
            :param svd_k: int - SVD dimension size
            :param baselines: boolean - whether to report baseslines' performance
        '''
        file = open(out_perf_file, "w")
        file.write("Dataset\t:\t%s\n" % self.dataset_name)
        file.write("#Algorithms\t:\t%s\n" % self.num_algs)
        file.write("#Instances\t:\t%s\n" % self.num_insts)
        file.write("#Features\t:\t%s\n" % self.num_features)

        file.write("\n-----------------------------\n\n")

        if baselines:
            file.write("Oracle - num_inst_solved:\t%s\n" % self.perf.num_inst_solved)
            file.write("Oracle - avg_rank:\t%s\n" % self.perf.oracle_avg_rank)
            file.write("Oracle - par10:\t%s\n" % self.perf.oracle_par10)
            file.write("Random - num_inst_solved:\t%s\n" % self.perf.rand_num_inst_solved)
            file.write("Random - avg_rank:\t%s\n" % self.perf.rand_avg_rank)
            file.write("SingleBest - num_inst_solved:\t%s\t%s\n" % (self.perf.sb_issolved_algo, np.max(self.perf.num_inst_solved_per_alg)))
            file.write("SingleBest - avg_rank:\t%s\t%s\n" % (self.perf.sb_rank_algo, np.min(self.perf.avg_rank_per_alg)))
            file.write("SingleBest - par10:\t%s\t%s\n" % (self.perf.sb_par10_algo, np.min(self.perf.par10_per_alg)))
            file.write("SingleWorst - num_inst_solved:\t%s\t%s\n" % (self.perf.sw_issolved_algo, np.min(self.perf.num_inst_solved_per_alg)))
            file.write("SingleWorst - avg_rank:\t%s\t%s\n" % (self.perf.sw_rank_algo, np.max(self.perf.avg_rank_per_alg)))
            file.write("SingleWorst - par10:\t%s\t%s\n" % (self.perf.sw_par10_algo, np.max(self.perf.par10_per_alg)))
            file.write("\n-----------------------------\n\n")

        if svd_s is not None:
            file.write("SVD full singular values\t:\t%s\n" % (', '.join(map(str, svd_s))).replace(".0", ""))
            file.write("SVD k\t:\t%s\n" % svd_k)
            file.write("\n-----------------------------\n\n")

        if alg_subset is not None and inst_subset is not None:
            file.write("Performance data reduction rate\t:\t%.2f\n" % ( 100 - (( (len(inst_subset) * len(alg_subset)) / float(self.num_insts * self.num_algs) ) * 100) ))

        if ft_subset is not None:
            file.write("Feature reduction rate\t:\t%.2f\n" % ( 100 - (( len(ft_subset) / float(self.num_features) ) * 100)) )

        if alg_subset is not None:
            file.write("Selected Algorithms\t:\t%s\n" % (', '.join(map(str, alg_subset))).replace(".0", ""))

        if inst_subset is not None:
            file.write("Selected Instances\t:\t%s\n" % (', '.join(map(str, inst_subset))).replace(".0", ""))

        if ft_subset is not None:
            file.write("Selected Features\t:\t%s\n" % (', '.join(map(str, ft_subset))).replace(".0", ""))
        file.close()
