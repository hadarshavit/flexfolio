'''
Created on July 6, 2015

@author: David Bergdoll
'''
import sys
import math


from trainer.selection.selector import SelectorTrainer

from trainer.evalutor.crossValidator import CrossValidator
from misc.printer import Printer

class SchedulerTrainer(SelectorTrainer):
    '''
        choose a promising scheduling algorithm based on training data
    '''


    def __init__(self, save_models=True):
        '''
            Constructor
        '''
        SelectorTrainer.__init__(self, save_models)
        self._cutoff = 900
        self._max_k_isa = 40
        self._max_k_sunny = 70
        self._norm_approach = "Zscore"

    def __repr__(self):
        return "Scheduling Trainer"
   
    def train(self, instance_dic, config_dic, meta_info, trainer):
        '''
            train model
        '''

        #instance_dic = self.__filter_instances(instance_dic)

        scheduler_settings_dic = self.__initialize_scheduler_settings(len(instance_dic), len(config_dic))

        scheduler = self.__evaluate_schedulers(meta_info,instance_dic,trainer,config_dic, scheduler_settings_dic)

        meta_info = self.__setup_approach(scheduler_settings_dic[scheduler], meta_info)

        Printer.print_nearly_verbose("Chosen scheduler: %s" %scheduler)

        return trainer.train(meta_info, instance_dic, config_dic, save_models=self._save_models, recursive=True)

    def __evaluate_schedulers(self, meta_info, instance_dic, trainer, config_dic, scheduler_settings_dic):
        '''
            compare the schedulers by cross validation
        '''
        evaluator = CrossValidator(meta_info.options.update_sup, None)

        original_folds = meta_info.options.crossfold
        meta_info.options.crossfold = 3

        best_par10 = sys.maxint
        for scheduler in scheduler_settings_dic.keys():
            meta_info = self.__setup_approach(scheduler_settings_dic[scheduler], meta_info)
            Printer.disable_printing = True
            par10, _ = evaluator.evaluate(trainer, meta_info, instance_dic, config_dic)
            Printer.disable_printing = False
            Printer.print_nearly_verbose("Scheduler: %s \t par10: %f" %(scheduler, par10))
            if best_par10 > par10:
                best_par10 = par10
                best_scheduler = scheduler

        meta_info.options.crossfold = original_folds
        return best_scheduler


    def __initialize_scheduler_settings(self, n_instances, n_solvers):
        '''
            write a dictionary of candidate configurations
        '''
        scheduler_settings = {}

        scheduler_settings["aspeed"] = {"approach": "ASPEED",
                                        "max_solver": 10000}
        scheduler_settings["aspeed_l3"] = {"approach": "ASPEED",
                                        "max_solver": 3}

        max_k = min(self._max_k_sunny, n_instances)
        max_limit = 2
        for k in self.__get_candidate_values(1):
            if k > max_k:
                break
            for limit in self.__get_candidate_values(1):
                if limit >= max_limit:
                    break
                label = "sunny_k" + str(k) + "_l" + str(limit)
                scheduler_settings[label] = {"approach": "SUNNY",
                                             "kNN": k,
                                             "max_solver": limit}

        max_k = min(self._max_k_isa, n_instances)
        for k in self.__get_candidate_values(1):
            if k > max_k:
                break
            label = "isa_k" + str(k)
            scheduler_settings[label] = {"approach": "ISA",
                                         "kNN":k,
                                         "max_solver": 10000}

        return scheduler_settings


    def __setup_approach(self, scheduler_settings, meta_info):
        '''
            set up the meta info according to the given configuration
        '''
        if scheduler_settings["approach"] == "SUNNY":
            meta_info.options.approach = "SUNNY"
            meta_info.options.knn = scheduler_settings["kNN"]
            meta_info.options.sunny_max_solver = scheduler_settings["max_solver"]
            meta_info.options.aspeed_opt = False
        if scheduler_settings["approach"] == "ISA":
            meta_info.options.approach = "ISA"
            meta_info.options.aspeed_max_solver = scheduler_settings["max_solver"]
            meta_info.options.knn = scheduler_settings["kNN"]
            meta_info.options.aspeed_opt = False
        if scheduler_settings["approach"] == "ASPEED":
            meta_info.options.approach = "SBS"
            meta_info.options.aspeed_opt = True
            meta_info.options.aspeed_max_solver = scheduler_settings["max_solver"]

        return meta_info

    def __get_candidate_values(self, start):
        x = start
        while 1:
            yield x
            x += max(1, int(math.sqrt(x)))

    def __filter_instances(self, instance_dic):

        new_dic = {}
        for name, instance in instance_dic.items():
            easy = True
            hard = True
            for perf in instance._transformed_cost_vec:
                if perf < self._cutoff:
                    hard = False
                if perf > 1:
                    easy = False
                if not easy and not hard:
                    break
            if not easy and not hard:
                new_dic[name] = instance

        return new_dic
