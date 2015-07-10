'''
Created on July 6, 2015

@author: David Bergdoll
'''
import sys



from trainer.selection.selector import SelectorTrainer
from trainer.selection.sunny import SunnyTrainer
from trainer.selection.instanceSpecificAspeed import ISATrainer
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
        self._norm_approach = "Zscore"

    def __repr__(self):
        return "Scheduling Trainer"
   
    def train(self, instance_dic, config_dic, meta_info, trainer):
        '''
            train model
            Args:
                instance_dic: instance name -> Instance()
                solver_list: list of solvers
                norm_approach: normalization approach
                cutoff: runtime cutoff
                model_dir: directory to save model
                f_indicator: of used features
                n_feats: number of features
        '''

        scheduler_settings_dic = self.__initialize_scheduler_settings()

        scheduler = self.__evaluate_schedulers(meta_info,instance_dic,trainer,config_dic, scheduler_settings_dic)

        meta_info = self.__setup_approach(scheduler_settings_dic[scheduler], meta_info)


        Printer.print_nearly_verbose("Chosen scheduler: %s" %scheduler)

        return scheduler_settings_dic[scheduler]["trainer_obj"],\
               trainer.train(meta_info, instance_dic, config_dic, save_models=self._save_models, recursive=True)

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
        
    def __initialize_scheduler_settings(self):
        scheduler_settings = {}
        scheduler_settings["sunny-basic"] = {"approach": "SUNNY",
                                             "kNN": -1,
                                             "sunny_max_solver": 0,
                                             "trainer_obj": SunnyTrainer(k=-1, save_models=self._save_models)
                                             }
        scheduler_settings["sunny-solver-max"] = {"approach": "SUNNY",
                                                  "kNN": -1,
                                                  "sunny_max_solver": 3,
                                                  "trainer_obj": SunnyTrainer(k=-1, save_models=self._save_models)
                                                  }
        scheduler_settings["isa"] = {"approach": "ISA",
                                     "kNN": -1,
                                     "train_k": False,
                                     "trainer_obj": ISATrainer(k=-1, save_models=self._save_models)
                                     }
        scheduler_settings["aspeed"] = {"approach": "ASPEED"}


        return scheduler_settings

    def __setup_approach(self, scheduler_settings, meta_info):
        if scheduler_settings["approach"] == "SUNNY":
            meta_info.options.approach = "SUNNY"
            meta_info.options.knn = scheduler_settings["kNN"]
            meta_info.options.sunny_max_solver = scheduler_settings["sunny_max_solver"]
            meta_info.options.aspeed_opt = False
        if scheduler_settings["approach"] == "ISA":
            meta_info.options.approach = "ISA"
            meta_info.options.knn = scheduler_settings["kNN"]
            meta_info.options.train_k = scheduler_settings["train_k"]
            meta_info.options.aspeed_opt = False
        if scheduler_settings["approach"] == "ASPEED":
            meta_info.options.approach = "SBS"
            meta_info.options.aspeed_opt = True
            meta_info.options.aspeed_max_solver = 10000

        return meta_info
