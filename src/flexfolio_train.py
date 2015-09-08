#!/usr/bin/env python2.7
##!/home/wv/bin/linux/64/python-2.7/bin/python2.7

'''
Created on Nov 8, 2012

@author: manju

calls:
python src/flexfolio_train.py --aslib <ASLIB PATH>/ASP-POTASSCO/ --model-dir .
'''

import sys
import os
import inspect
import json
import copy
from ainovelty.ain_flexfolio_v2 import pre_process, filter_data
from ainovelty.ain_analyzer import apply_ain
from ainovelty.experiment import Experiment
from pygments.sphinxext import FILTERDOC

# http://stackoverflow.com/questions/279237/python-import-a-module-from-a-folder
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.append(cmd_folder)
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, "trainer"))
if cmd_folder not in sys.path:
    sys.path.append(cmd_folder)
    
from misc.printer import Printer

from trainer.training_parser.cmd_training_parser import TainerParser
#from training_parser.reader import Reader
from trainer.training_parser.coseal_reader import CosealReader

from trainer.selection.classifiers4voting.svmVoting import SVMVoting
from trainer.selection.classifiers4voting.randomforestVoting import RandomForestVoting
from trainer.selection.classifiers4voting.gradientboostingVoting import GradientBoostingVoting

from trainer.selection.regressors.lassoregression import LassoRegression
from trainer.selection.regressors.ridgeregression import RidgeRegression
from trainer.selection.regressors.svr import SVRRegressor
from trainer.selection.regressors.rfregression import RandomForrestRegression

from trainer.selection.regressorsSNNAP.lassoregression import LassoRegressionSNNAP 
from trainer.selection.regressorsSNNAP.ridgeregression import RidgeRegressionSNNAP
from trainer.selection.regressorsSNNAP.svr import SVRRegressorSNNAP
from trainer.selection.regressorsSNNAP.rfregression import RandomForrestRegressionSNNAP

from trainer.selection.classifiersMulti.randomForestMulti import RandomForestMulti
from trainer.selection.classifiersMulti.gradientboostingMulti import GradientBoostingMulti
from trainer.selection.classifiersMulti.svmMulti import SVMMulti

from trainer.selection.clustering.KMeans import KMeansTrainer
from trainer.selection.clustering.GM import GMTrainer
from trainer.selection.clustering.Spectral import SpectralTrainer

from trainer.selection.NN import NearestNeighbourTrainer
from trainer.selection.kNN import KNNTrainer
from trainer.selection.SBS import SBSTrainer
from trainer.selection.Ensemble import Ensemble

from trainer.performancepreprocessing.contributor_filter import ContributorFilter
from trainer.performancepreprocessing.correlator import Correlator
from trainer.performancepreprocessing.instance_weighting import InstanceWeighter
from trainer.performancepreprocessing.performancetransformation import PerformanceTransformator
from trainer.performancepreprocessing.remove_algos import AlgoRemover
from trainer.featurepreprocessing.forwardSelector import ForwardSelector
from trainer.featurepreprocessing.normalizer import Normalizer
from trainer.featurepreprocessing.Imputer import Imputer

from trainer.evalutor.crossValidator import CrossValidator
from trainer.evalutor.classValidator import ClassValidator
from trainer.evalutor.crossValidatorGiven import CrossValidatorGiven
from trainer.evalutor.ttValidator import TrainTestValidator
from trainer.aspeed.aspeedAll import AspeedAll

class Trainer(object):
    '''
        main class for training models for flexfolio
    '''
    
    def __init__(self, ain_num_ft_to_remove = 0):
        '''
            Constructor
        '''
        self.ain_num_ft_to_remove = ain_num_ft_to_remove
        self.selection_methods = {"CLASSVOTER": { "SVM": SVMVoting,
                                                  "RANDOMFOREST": RandomForestVoting,
                                                  "GRADIENTBOOSTING": GradientBoostingVoting
                                                 },
                                  "REGRESSION": {
                                                  "SVR" : SVRRegressor,
                                                  "RIDGE" : RidgeRegression,
                                                  "LASSO" : LassoRegression,
                                                  "RANDOMFOREST" : RandomForrestRegression
                                                 },
                                  "SNNAP": {
                                                  "SVR" : SVRRegressorSNNAP,
                                                  "RIDGE" : RidgeRegressionSNNAP,
                                                  "LASSO" : LassoRegressionSNNAP,
                                                  "RANDOMFOREST" : RandomForrestRegressionSNNAP
                                            },
                                  "CLASSMULTI" : { "SVM": SVMMulti,
                                                   "RANDOMFOREST": RandomForestMulti,
                                                   "GRADIENTBOOSTING": GradientBoostingMulti
                                                  },
                                  "NN":         NearestNeighbourTrainer, 
                                  "kNN":        KNNTrainer,
                                  "CLUSTERING": {"KMEANS" : KMeansTrainer,           
                                                 "GM" : GMTrainer,
                                                 "SPECTRAL": SpectralTrainer
                                                },
                                  "SBS": SBSTrainer(),
                                  "ENSEMBLE": Ensemble
                                  }
        
    def main(self,sys_argv):
        '''
            main method for training
            Parameter:
                sys_argv: command line arguments (sys.argv[1:])
        '''
        Printer.print_c("Command line arguments:")
        Printer.print_c(" ".join(sys_argv))
        #parse command line arguments
        parser = TainerParser()
        args_ = parser.parse_arguments(sys_argv)
            
        # read input files and provide data structures
        #=======================================================================
        # if not args_.coseal:
        #     reader = Reader(meta_info.algorithm_cutoff_time, args_.n_feats, args_.filter_duplicates)
        #     instance_dic, solver_list, config_dic = \
        #         reader.get_data(args_.times, args_.feats, args_.satunsat, args_.configs, args_.feature_time, args_.feature_time_const)
        # else:
        #=======================================================================
        reader = CosealReader()
        instance_dic, meta_info, config_dic = reader.parse_coseal(args_.coseal, args_)
        
        ## Set AIN data filtering parameters
        Experiment.alg_subset_selection = args_.alg_subset
        Experiment.inst_subset_selection = args_.inst_subset
        Experiment.ft_subset_selection = args_.ft_subset
         
        if meta_info.cv_given and meta_info.options.test_set:
            evaluator = TrainTestValidator(args_.update_sup, args_.print_time)
            evaluator.evaluate(self, meta_info, instance_dic, config_dic)
        elif meta_info.cv_given:
            evaluator = CrossValidatorGiven(args_.update_sup, args_.print_time)
            evaluator.evaluate(self, meta_info, instance_dic, config_dic, threads=args_.threads_aspeed, ain_num_ft_to_remove = self.ain_num_ft_to_remove)
            
        #=======================================================================
        # elif args_.smac and args_.crossfold >= 0 and args_.fold > -1: # cross validation with only one evaluated fold (for SMAC)
        #     evaluator = FoldValidator(args_.update_sup, args_.print_time)
        #     evaluator.evaluate(self, meta_info, instance_dic, config_dic)
        #=======================================================================
        
        elif args_.crossfold >= 0: # cross fold validation 
            evaluator = CrossValidator(args_.update_sup, args_.print_time)
#            evalutor.evaluate_invalids(invalid_f_runtime, ranks.index(0), meta_info.algorithm_cutoff_time)
            evaluator.evaluate(self, meta_info, instance_dic, config_dic, threads=args_.threads_aspeed)
        
        elif args_.class_evaluation: #TODO: probably broken!
            evaluator = ClassValidator(args_.update_sup, args_.print_time)
            evaluator.evaluate(self, meta_info, instance_dic, config_dic)
        
#==============================================================================
#        elif args_.test_times: # evaluation with test data
#            instance_test_dic, solver_list, config_dic = \
#                         reader.get_data(args_.test_times, args_.test_feats, args_.test_satunsat, args_.configs)
#            evaluator = TestValidator(args_.update_sup, args_.print_time)
#            evaluator.evaluate(self, args_, instance_dic, instance_test_dic, solver_list, config_dic, args_.seed)
# #           evaluator.evaluate_invalids(invalid_f_runtime, ranks.index(0), meta_info.algorithm_cutoff_time)
#==============================================================================
        else:
            # normal training
            selection_dic = self.train(meta_info, instance_dic, config_dic)
            self.__write_config(selection_dic, meta_info)
            Printer.print_verbose(json.dumps(selection_dic, indent=2))
        
    def train(self, meta_info, instance_dic, config_dic, feature_indicator=None, save_models=True, recursive=False, ain_num_ft_to_remove=0, aspeed_call=False):
        '''
            only train models (no evaluation)
            Parameter:
                meta_info: trainer.trainer_parser.coseal_reader.Metainfo()
                instance_dic: name -> Instance()
                config_dic : solver alias -> command line
                feature_indicator: indicator to set features to 0
                save_models: save models to file system (or not)
                recursive: if it is a recursive call (e.g., via ENSEMBLE) some task shouldn't be performed twice (e.g., feature preprocessing)
            Returns
                dictionary with all generated files and meta informations
            ATTENTION:
                If modified, modify also misc.updater.__retrain()
        '''
        
        # filter algorithms which do not contribute to the oracle performance

        # prevent modification of original objects - important when using CV
        
        original_instance_dic = instance_dic
        original_solver_list = copy.deepcopy(meta_info.algorithms)
        instance_dic = copy.deepcopy(instance_dic)
        meta_info = copy.deepcopy(meta_info)
        solver_list = meta_info.algorithms
        config_dic = copy.deepcopy(config_dic)
        args_ = meta_info.options
        n_feats = len(meta_info.features)
        
        
        # Apply data filtering
        if not aspeed_call and not recursive:
            filtered_instance_dic, _filtered_meta_info, _filtered_config_dic, to_remove_alg_list, _to_remove_alg_inx_list, selected_ft_arr = filter_data(instance_dic, meta_info, config_dic, ain_num_ft_to_remove)
            algos = set(meta_info.algorithms)
            algos = algos.difference(to_remove_alg_list)
            feature_indicator = [0]*n_feats
            for f in selected_ft_arr:
                feature_indicator[f] = 1
            del_insts = set(instance_dic.keys()).difference(filtered_instance_dic.keys())
            for inst_ in del_insts:
                del instance_dic[inst_]
        else:
            algos = meta_info.algorithms
        
        if args_.algorithms:
            args_.algorithms = list(set(args_.algorithms).difference(algos))
        else:
            args_.algorithms = list(algos)
        
        # remove algorithms that are not listed
        if args_.algorithms:
            rem = AlgoRemover()
            instance_dic, solver_list, config_dic = rem.remove_algo(instance_dic, solver_list, config_dic, args_.algorithms)
            
        # find the best solver on the training set
        ranks = self.find_backup_solver(instance_dic, meta_info.algorithm_cutoff_time) # index of best par10 solver
        Printer.print_c("Backup Solver: %s" % (solver_list[ranks.index(0)]))
        
        # contribution filtering wrt VBS
        if args_.contributor_filter > 0:
            Printer.print_verbose("Contribution filtering ...")
            filter_ = ContributorFilter(args_.contributor_filter)
            instance_dic, solver_list, config_dic = filter_.filter(instance_dic, solver_list, config_dic)

        #correlation tests 
        correlation_dict = None
        if args_.correlation > 0:
            Printer.print_verbose("Correlation tests ...")
            tester = Correlator()
            correlation_dict = tester.pairwise_contribution(instance_dic, solver_list, args_.correlation)
            #correlation_dict = tester.correlation_test(instance_dic, solver_list, args_.correlation)

        # pre-solver schedule via aspeed
        if args_.aspeed_opt:
            if args_.aspeed_pre_slice == -10:
                args_.aspeed_pre_slice = int(meta_info.algorithm_cutoff_time / 10)
            scheduler = AspeedAll(clasp=args_.aspeed_clasp,
                                  gringo=args_.aspeed_gringo,
                                  runsolver=args_.aspeed_runsolver,
                                  enc=args_.aspeed_enc,
                                  time_limit=args_.aspeed_time_limit, 
                                  mem_limit=args_.aspeed_mem_limit, 
                                  num_solvers=args_.aspeed_max_solver, 
                                  opt_mode=args_.aspeed_opt_mode,
                                  max_pre_slice=args_.aspeed_pre_slice,
                                  threads=args_.threads_aspeed
                                 )
            pre_solver_dict = scheduler.optimize_schedule(trainer, meta_info, instance_dic, config_dic)

        Printer.print_c("Input Instances for Training: %d (-unsolvable)" %(len(instance_dic)))
        if args_.impute == "none":
            # remove instances with invalid feature status cannot be used to learn ml model
            clean_up_dict = {}
            for inst_ in instance_dic.values():
                if inst_._features and len(inst_._features) == n_feats:
                    clean_up_dict[inst_._name] = inst_
            instance_dic = clean_up_dict
            imp = None
        else: # feature imputation
            imp = Imputer(strategy=args_.impute)
            instance_dic, imp = imp.impute(instance_dic, n_feats)
        Printer.print_c("Used Instances for Training: %d (-unsolvable)" %(len(instance_dic)))         
            
        # apply transformation of performance input
        if not recursive:
            Printer.print_verbose("Performance transformation ...")
            transformer = PerformanceTransformator()
            transformer.transform(instance_dic, meta_info)
        
        # approximate instance weights
        if args_.approx_weights and not recursive:
            Printer.print_verbose("Approximate weigths ...")
            weighter = InstanceWeighter()
            weighter.weight_instances(instance_dic, meta_info)
        
        # normalize features
        if not feature_indicator:
            if args_.feature_indicator:
                feature_indicator = args_.feature_indicator
            else:
                feature_indicator = [1] * n_feats
        
        #if not recursive:
        normalizer = Normalizer()
        instance_dic, norm_stats, feature_indicator = normalizer.normalize(instance_dic, args_.norm, feature_indicator, args_.pca_dims)  
        
        # feature selection
        if args_.feat_sel and not recursive: 
            Printer.print_verbose("Feature selection ...")
            feature_selector = ForwardSelector()
            feature_indicator = feature_selector.select_features(self, instance_dic, feature_indicator, config_dic, meta_info)
        
        selection_dic = None
        trainer_obj = None      
            
        # aggregate normalization dictionary
        if args_.approach == "REGRESSION":
            if args_.regressor == "SVR":
                trainer_obj = self.selection_methods[args_.approach][args_.regressor](args_.svm_gamma,
                                                                                      args_.svm_C,
                                                                                      args_.svm_epsilon,
                                                                                      save_models
                                                                                      )
            if args_.regressor == "RIDGE":
                trainer_obj = self.selection_methods[args_.approach][args_.regressor](args_.ridge_alpha,
                                                                                      save_models
                                                                                      )
            if args_.regressor == "LASSO":
                trainer_obj = self.selection_methods[args_.approach][args_.regressor](args_.lasso_alpha,
                                                                                      save_models
                                                                                      )            
            if args_.regressor == "RANDOMFOREST":
                trainer_obj = self.selection_methods[args_.approach][args_.regressor](args_.rf_max_features,
                                                                                       args_.rf_min_samples_leaf,
                                                                                       save_models
                                                                                       )    
            Printer.print_c("Train with %s" %(str(trainer_obj)))    
            selection_dic = trainer_obj.train(instance_dic, solver_list, config_dic,
                                              meta_info.algorithm_cutoff_time, args_.model_dir, feature_indicator, n_feats)
        if args_.approach == "SNNAP":
            if args_.regressor == "SVR":
                trainer_obj = self.selection_methods[args_.approach][args_.regressor](args_.knn,
                                                                                      args_.best_n,
                                                                                      args_.svm_gamma,
                                                                                      args_.svm_C,
                                                                                      args_.svm_epsilon,
                                                                                      save_models
                                                                                      )
            if args_.regressor == "RIDGE":
                trainer_obj = self.selection_methods[args_.approach][args_.regressor](args_.knn,
                                                                                      args_.best_n,
                                                                                      args_.ridge_alpha,
                                                                                      save_models
                                                                                      )
            if args_.regressor == "LASSO":
                trainer_obj = self.selection_methods[args_.approach][args_.regressor](args_.knn,
                                                                                      args_.best_n,
                                                                                      args_.lasso_alpha,
                                                                                      save_models
                                                                                      )            
            if args_.regressor == "RANDOMFOREST":
                trainer_obj = self.selection_methods[args_.approach][args_.regressor](args_.knn,
                                                                                      args_.best_n,
                                                                                      args_.rf_max_features,
                                                                                      args_.rf_min_samples_leaf,
                                                                                      save_models
                                                                                       )    
            Printer.print_c("Train with %s" %(str(trainer_obj)))
            selection_dic = trainer_obj.train(instance_dic, solver_list, config_dic,
                                              meta_info.algorithm_cutoff_time, args_.model_dir, feature_indicator, n_feats)
        if args_.approach == "CLASSVOTER":
            if args_.classifier == "RANDOMFOREST":
                trainer_obj = self.selection_methods[args_.approach][args_.classifier](args_.rf_max_features,
                                                                                       args_.rf_criterion,
                                                                                       args_.rf_min_samples_leaf,
                                                                                       save_models
                                                                                       )
            if args_.classifier == "SVM":
                trainer_obj = self.selection_methods[args_.approach][args_.classifier](args_.svm_gamma,
                                                                                       args_.svm_C,
                                                                                       save_models
                                                                                       )
            if args_.classifier == "GRADIENTBOOSTING":
                trainer_obj = self.selection_methods[args_.approach][args_.classifier](args_.gb_max_depth,
                                                                                       args_.gb_min_samples_leaf,
                                                                                       args_.gb_max_features,
                                                                                       save_models
                                                                                       )
            Printer.print_c("Train with %s" %(str(trainer_obj)))    
            selection_dic = trainer_obj.train(instance_dic, solver_list, config_dic,
                                              meta_info.algorithm_cutoff_time, args_.model_dir, feature_indicator, n_feats)
        if args_.approach == "CLASSMULTI":
            if args_.classifiermulti == "SVM":
                trainer_obj = self.selection_methods[args_.approach][args_.classifiermulti](args_.svm_gamma,
                                                                                            args_.svm_C,
                                                                                            )
            if args_.classifiermulti == "RANDOMFOREST":
                trainer_obj = self.selection_methods[args_.approach][args_.classifiermulti](args_.rf_max_features,
                                                                                            args_.rf_criterion,
                                                                                            args_.rf_min_samples_leaf
                                                                                            )
            if args_.classifiermulti == "GRADIENTBOOSTING":
                trainer_obj = self.selection_methods[args_.approach][args_.classifiermulti](args_.gb_max_depth,
                                                                                            args_.gb_min_samples_leaf,
                                                                                            args_.gb_max_features
                                                                                            )
            Printer.print_c("Train with %s" %(str(trainer_obj)))
            selection_dic = trainer_obj.train(instance_dic, solver_list, config_dic,
                                                           meta_info.algorithm_cutoff_time, args_.model_dir, feature_indicator, n_feats)                   
        if args_.approach == "NN":
            trainer_obj = self.selection_methods[args_.approach]()
            Printer.print_c("Train with %s" %(str(trainer_obj)))
            selection_dic = trainer_obj.train(instance_dic, solver_list, config_dic,
                                                           meta_info.algorithm_cutoff_time, args_.model_dir, 
                                                           feature_indicator, n_feats)
        if args_.approach == "kNN":
            trainer_obj = self.selection_methods[args_.approach](k=args_.knn, save_models=save_models)
            Printer.print_c("Train with %s" %(str(trainer_obj)))
            selection_dic = trainer_obj.train(instance_dic, solver_list, config_dic,
                                                           meta_info.algorithm_cutoff_time, args_.model_dir, 
                                                           feature_indicator, n_feats, 
                                                           meta_info, trainer)
        if args_.approach == "SBS":
            trainer_obj = self.selection_methods[args_.approach]
            Printer.print_c("Train with %s" %(str(trainer_obj)))
            selection_dic = trainer_obj.train(original_instance_dic, original_solver_list, config_dic,
                                                           meta_info.algorithm_cutoff_time, args_.model_dir, 
                                                           feature_indicator, n_feats)
        if args_.approach == "CLUSTERING":
            if args_.cluster_algo == "KMEANS":
                trainer_obj = self.selection_methods[args_.approach][args_.cluster_algo](max_clusters=args_.clu_max_cluster, 
                                                                                         plot_cluster=args_.clu_plot_cluster)
                
            if args_.cluster_algo == "GM":
                trainer_obj = self.selection_methods[args_.approach][args_.cluster_algo](max_clusters=args_.clu_max_cluster, 
                                                                                         plot_cluster=args_.clu_plot_cluster)
                
            if args_.cluster_algo == "SPECTRAL":
                trainer_obj = self.selection_methods[args_.approach][args_.cluster_algo](max_clusters=args_.clu_max_cluster, 
                                                                                         plot_cluster=args_.clu_plot_cluster)
            Printer.print_c("Train with %s" %(str(trainer_obj)))
            selection_dic = trainer_obj.train(instance_dic, solver_list, config_dic,
                                                           meta_info.algorithm_cutoff_time, args_.model_dir, 
                                                           feature_indicator, n_feats)
        if args_.approach == "ENSEMBLE":
            trainer_obj = self.selection_methods[args_.approach](args_.ensemble_sub_size,
                                                                 args_.ensemble_num_models,
                                                                 args_.ensemble_bootstrapping,
                                                                 args_.ensemble_mfeatures,
                                                                 save_models
                                                                 )
            Printer.print_c("Train with %s" %(str(trainer_obj)))
            trainers = self.selection_methods
            selection_dic = trainer_obj.train(instance_dic, solver_list, config_dic,
                                                           meta_info.algorithm_cutoff_time, args_.model_dir, 
                                                           feature_indicator, n_feats, trainers, meta_info, self)

        selection_dic = trainer_obj.set_backup_solver(selection_dic, ranks)

        if args_.aspeed_opt: 
            self.__add_schedule_2_selection_dict(selection_dic, pre_solver_dict)

        # add feature normalization to selection_dic
        selection_dic["normalization"]["approach"] = { "pca_dims": args_.pca_dims, 
                                                        "name": args_.norm                                                    
                                                    }
        
        selection_dic["normalization"]["impute"] = imp
        d = selection_dic["normalization"]
        d.update(norm_stats)
        
        if correlation_dict:
            selection_dic["approach"]["correlation"] = correlation_dict
        
        return selection_dic
    
    def find_backup_solver(self, instance_dic, cutoff):
        '''
            get backup solver
            backup solver = best solver in valid time dic
            Parameter:
                instance_dic : dictionary  instance -> Instance()
            Returns:
                index of best solver
        '''
        n_solver = len(instance_dic[list(instance_dic.keys())[0]]._cost_vec)
        par10 = [0]*n_solver
        for inst in instance_dic.values():
            times = inst._cost_vec
            times = map(lambda x: cutoff*10 if x >= cutoff else x, times )
            par10 = list(x+y for x,y in zip(par10, times))
        ranks = [sorted(par10).index(x) for x in par10]
        return ranks
        
    
    def __write_config(self, sel_dic, metainfo):
        '''
           write config file for xfolio
        '''
        args_ = metainfo.options
        #=======================================================================
        # n_feats = len(metainfo.features)
        #=======================================================================
        
        config_dic = {
                          "extractor":{
                                   "class" : args_.feat_class,
                                   "path" : args_.feature_path,
                                   "maxTime" : args_.feat_time
                                   },
                          "selector" : sel_dic         
                          }
        
        #=======================================================================
        # if args_.update_sup:
        #     update_dic = {
        #                   "runtime_file" : args_.times,
        #                   "feature_file" : args_.feats,
        #                   "class_file" : args_.satunsat,
        #                   "runtime_update" : os.path.join(args_.model_dir,"runtime_update.txt"),
        #                   "feature_update" : os.path.join(args_.model_dir,"feature_update.txt"),
        #                   "class_update" : os.path.join(args_.model_dir,"class_update.txt"),
        #                   "cutoff" : meta_info.algorithm_cutoff_time,
        #                   "n_feats": n_feats,
        #                   "approach" : args_.approach,
        #                   "norm" : args_.norm,
        #                   "svm_train" : args_.svm_train,
        #                   "model_dir" : args_.model_dir,
        #                   "n_new" : 0
        #                   }
        #     config_dic["update"] = update_dic
        #=======================================================================
        
        config_file = open(os.path.join(args_.model_dir,"config.json"),"w")
        json.dump(config_dic,config_file,indent=2)

    def __add_schedule_2_selection_dict(self, sel_dict, core_solver_time_dict):
        '''
            adds meta information for presolving schedule in meta dictionary sel_dict
            Args:
                sel_dict: dictionary with meta information about learned models
                solver_time_dict: mapping solver to pre-solver time
        '''
        for core, solver_time_dict in core_solver_time_dict.items():
            for solver, time_ in solver_time_dict.items():
                if solver == "claspfolio":
                    continue
                pre_solver_dict = sel_dict["configurations"][solver]
                pre_solver_dict["presolving_time"] = pre_solver_dict.get("presolving_time",{}) 
                pre_solver_dict["presolving_time"][core] = time_    
        sel_dict["approach"]["presolving"] = True   

if __name__ == '__main__':
    
    Printer.print_c("flexfolio Trainer!")
    Printer.print_c("Published under GPLv2")
    Printer.print_c("https://bitbucket.org/mlindauer/flexfolio")
    trainer = Trainer()
    trainer.main(sys.argv[1:])
    
