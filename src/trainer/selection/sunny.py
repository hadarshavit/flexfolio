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

class SunnyTrainer(SelectorTrainer):
    '''
        trainer for SUNNY; does not train anything, merely writes the needed config file
    '''


    def __init__(self, k=-1, max_solvers = 0, save_models=True):
        '''
            Constructor
        '''
        SelectorTrainer.__init__(self, save_models)
        self._cutoff = 900
        self._norm_approach = "Zscore"
        
        self.k = k
        self.max_solvers = max_solvers
        
        self._UNKNOWN_CODE = -512
        self._MAX_K = 32
   
    def __repr__(self):
        return "SUNNY"
   
    def train(self, instance_dic, solver_list, config_dic, cutoff, model_dir, f_indicator, n_feats, 
                meta_info, trainer):
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

        if self.k == -1:
            folds =  meta_info.options.crossfold
            self.k = int(round(math.sqrt(len(instance_dic)*(float(folds-1)/folds))))
        
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
                                "approach" : "SUNNY",
                                "k" : self.k,
                                "max_solvers" : self.max_solvers,
                                "model" : model_name,
                                "cutoff": self._cutoff,
                                "n_feats" : n_normed_feats
                                },
                   "normalization" : {
                                      "filter"  : f_indicator
                                 },
                   "configurations":conf_dic                        
                   }
        return sel_dic
        
    def __write_arff(self, instance_dic, solver_list, model_dir, n_feats):
        '''
            write nn models as arff files
        '''
        model_name = os.path.join(model_dir,"model_sunny.arff")
        fp = open(model_name,"w")
        
        fp.write("@relation sunny set\n\n")
        
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
        
        
        
    
    