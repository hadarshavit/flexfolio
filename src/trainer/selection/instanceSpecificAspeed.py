'''
Created on May 26, 2015

@author: David Bergdoll
'''
import os
import sys
import math


from trainer.selection.selector import SelectorTrainer
from trainer.evalutor.crossValidator import CrossValidator
from misc.printer import Printer

class ISATrainer(SelectorTrainer):
    '''
        trainer for Instance Specific Aspeed scheduling
    '''


    def __init__(self, k=-1, save_models=True):
        '''
            Constructor
        '''
        SelectorTrainer.__init__(self, save_models)
        self._cutoff = 900
        self._norm_approach = "Zscore"
        
        self.k = k
        
        self._UNKNOWN_CODE = -512
        self._MAX_K = 42
   
    def __repr__(self):
        return "Instance Specific Aspeed"
   
    def train(self, instance_dic, solver_list, config_dic, cutoff, model_dir, f_indicator, n_feats, 
                feat_time, meta_info, trainer, clasp, gringo, runsolver, enc, mem_limit,
                max_solver, opt_mode, pre_sclice, threads, train_k, descending):
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
        
        self._cutoff = cutoff

        instance_dic = self.__filter_instances(instance_dic)

        if train_k:
            self.k = self.__guessK(meta_info, instance_dic, trainer, config_dic)

        elif self.k == -1:
            self.k = int(round(math.sqrt(len(instance_dic))))

        Printer.print_nearly_verbose("Chosen k: %d" %(self.k))
        
        if self._save_models:
            model_name = self.__write_arff(instance_dic, solver_list, model_dir, n_feats)
        else:
            model_name = self.__create_save(instance_dic)

        n_normed_feats = len(instance_dic.values()[0]._normed_features)
        
        # build selection_dic
        
        conf_dic = {}
        for solver,cmd in config_dic.items():
                    solver_dic = {
                          "call": cmd,                  
                          "id" : solver_list.index(str(solver))          
                          }
                    conf_dic[solver] = solver_dic
        sel_dic = {
                   "approach": {
                                "approach" : "ISA",
                                "k" : self.k,
                                "model" : model_name,
                                "cutoff": self._cutoff,
                                "feat_time": feat_time,
                                "model_dir": model_dir,
                                "clasp": clasp,
                                "gringo": gringo,
                                "runsolver": runsolver,
                                "enc": enc,
                                "mem_limit": mem_limit,
                                "max_solver": max_solver,
                                "opt_mode": opt_mode,
                                "pre_slice": pre_sclice,
                                "threads": threads,
                                "n_feats": n_normed_feats,
                                "descending": descending

                                },
                   "normalization" : {
                                      "filter"  : f_indicator                           
                                 },
                   "configurations":conf_dic
                   }
        return sel_dic

    def __guessK(self, meta_info, instance_dic, trainer, config_dic):
        '''
            guess k on PAR10 by cross validation
        '''
        evaluator = CrossValidator(meta_info.options.update_sup, None)

        original_folds = meta_info.options.crossfold
        meta_info.options.crossfold = 3

        k = 1
        best_par10 = sys.maxint
        best_k = 1
        max_k = min(self._MAX_K, len(instance_dic) - 1)

        for k in self.__get_candidate_values(1):
            if k > max_k:
                break
            meta_info.options.knn = k
            meta_info.options.train_k = False
            Printer.disable_printing = True
            par10, _ = evaluator.evaluate(trainer, meta_info, instance_dic, config_dic)
            Printer.disable_printing = False
            Printer.print_verbose("k: %d \t par10: %f" %(k, par10))
            if best_par10 > par10:
                best_par10 = par10
                best_k = k

        meta_info.options.crossfold = original_folds

        return best_k
        
    def __write_arff(self, instance_dic, solver_list, model_dir, n_feats):
        '''
            write nn models as arff files
        '''
        model_name = os.path.join(model_dir,"model_isa.arff")
        fp = open(model_name,"w")
        
        fp.write("@relation isa set\n\n")
        
        for i in range(0,n_feats):
            fp.write("@attribute feature_%d numeric\n" % (i))
        
        fp.write("@attribute weight numeric\n")
        
        for s in solver_list:
            fp.write("@attribute %s NUMERIC\n" % (s))
        
        fp.write("@data\n")
        for inst_ in instance_dic.values():
            if inst_._pre_solved:
                continue
            fp.write("%s,%f,%s\n" % (",".join(map(str,inst_._normed_features)), inst_._weight, (",".join(map(str,inst_._transformed_cost_vec)))))
        fp.close()
        
        return model_name 
    
    def __create_save(self, instance_dic):
        save_list =  []
        
        for inst_ in instance_dic.values():
            if inst_._pre_solved:
                continue
            point = {"features" : inst_._normed_features,
                     "weight" : inst_._weight,
                     "perfs" : inst_._transformed_cost_vec
                     }
            save_list.append(point)
            
        return save_list
        
        
    def __get_candidate_values(self, start):
        x = start
        while 1:
            yield x
            x += max(1, int(math.sqrt(x)))


    def __filter_instances(self, instance_dic):
        '''
            remove instances that are either too easy or too hard from the given instance dic
        '''

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
    
    