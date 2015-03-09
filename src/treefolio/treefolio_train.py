#!/usr/bin/env python2.7
'''
Created on June 20, 2013

@author: CVSH

calls:
--runtimes ./data/avg2/claspfolioTimes-2.0.0-avg2.csv --features ./data/avg2/claspfolio-claspre2-avg2.csv --cutoff 900 --modeldir ./models/ --feature-class claspre --feature-extractor ./binaries/claspre --configurations ./data/avg2/confs.json --approach REGRESSION --crossfold 10 --normalize pca --contr-filter 0.02 --nFeats 132
'''

import copy
import os
import sys
import inspect
import random

# http://stackoverflow.com/questions/279237/python-import-a-module-from-a-folder
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.insert(1,cmd_folder)

from misc.printer import Printer
from tfparser.trparser import TreeTrainerParser
from trainer.parser.reader import Reader
from steps.builder import StepBuilder


class Trainer(object):
    '''
        main class for training models for treefolio
    '''
    
    def __init__(self):
        '''
            Constructor
        '''
        self.builder = None
        
    def main(self,sys_argv):
        '''
            main method for training
            Parameter:
                sys_argv: command line arguments (sys.argv[1:])
        '''
        
        parser = TreeTrainerParser()
        args_ = parser.parse_arguments(sys_argv)
        
        # read input files and provide data structures
        reader = Reader(args_.cutoff, args_.n_feats, args_.filter_duplicates)
        self.instance_dic, self.solver_list, self.config_dic = \
            reader.get_data(args_.times, args_.feats, args_.satunsat, args_.configs)
        
        if args_.crossfold > 1:
            self.cross(args_, self.instance_dic, self.solver_list, self.config_dic)
        else:
            tree, res = self.train(args_, self.instance_dic, self.solver_list, self.config_dic)
            Printer.print_c("\nOverall Result: "+str(res)+"\n")
        
    
    def train(self, args_, instance_dic, solver_list, config_dic):
        '''
            new Treefolio train function
            Parameter:
                args_: parsed arguments
                instance_dic: full instance dictionary for training (does contain runtimes of executors)
                solver_list: List of all solvers
                config_dic: 
        '''
        
        # Build full Structure (scratch) and save as Model config (Placeholders for models!)
        self.builder = StepBuilder()
        
        # Set Builder File, Build Structure!
        self.builder.read_and_build_from_data(args_, args_.treefolio_configfile)
        
        # Print the Tree
        if args_.verbose > 0:
            self.builder.test_print()
            Printer.print_c("")
        
        instance_dic_ = copy.deepcopy(instance_dic)
        
        # Calculate unused Features
        ran = []
        if self.builder.ranges:
            i = 0;
            valt = instance_dic_.values()[0]
            while not valt.get_features():
                i = i+1;
                if (len(instance_dic_)<=i):
                    Printer.print_e("No Features found for ALL! Instances. Make sure they are named correctly.", 4532)
                valt = instance_dic_.values()[i] # if there are no features for any Instance, this will fail!
            n = len(valt.get_features())
            ran = set(range(1,n))
            for thisrange in self.builder.ranges:
                if thisrange and len(thisrange)>1:
                    ran -= set(range(thisrange[0],thisrange[1]+1))
        
        # If used, add starting Features to Dictionary
        if args_.treefolio_start_features:
            for i in instance_dic_:
                feats = instance_dic_[i].get_features()
                feats_n = instance_dic_[i].get_normed_features()
                nf = []
                nf_n = []
                if ran and feats:
                    for r in ran:
                        if len(feats)>r:
                            nf.append(feats[r])
                            if feats_n and len(feats_n)>r:
                                nf_n.append(feats_n[r])
                if not nf:
                    nf = None
                instance_dic_[i].set_features(nf)
                if feats_n:
                    instance_dic_[i].set_normed_features(nf_n)
        else:
            if ran:
                Printer.print_w("Warning: "+str(len(ran))+" Features are not derived : "+str(ran))
            for i in instance_dic_:
                if (instance_dic[i].get_features()):
                    instance_dic_[i].set_features([])
                    instance_dic_[i].set_normed_features([])
                else:
                    instance_dic_[i].set_features(None)
                    instance_dic_[i].set_normed_features(None)
        
        # Train the steps
        if not self.builder.first.is_trained():
            name, res = self.builder.first.train(args_, instance_dic, solver_list, config_dic, instance_dic_)
            return self.builder.first, res
            
        # Save Tree for execution
        self.builder.save_tree(args_,args_.treefolio_modelfile)
        
        return self.builder.first, {}
    
    def cross(self, args_, instance_dic, solver_list, config_dic):
        
        #tcv = TreefolioCrossValidator()
        #tcv.evaluate(self, args_, args_.crossfold, instance_dic, solver_list, config_dic, args_.seed)
        
        random.seed(args_.seed)
        
        self.__n_folds = args_.crossfold
        
        if self.__n_folds > 0:
            l = len(instance_dic)
            index = 0
            self.__instance_parts = []
            parted_inst_dic = {}
            threshold = l / self.__n_folds
            partIndex = 1
            instsList = instance_dic.keys()
            while instsList != []:
                randIndex = random.randint(0, len(instsList)-1)
                inst = instsList.pop(randIndex)
                
                if (index >= threshold): # if the fold full?
                    self.__instance_parts.append(parted_inst_dic)
                    parted_inst_dic = {}
                    l = l - index
                    threshold = l / (self.__n_folds - partIndex)
                    partIndex += 1
                    index = 0
                    
                index += 1
                parted_inst_dic[inst] = instance_dic[inst]
            self.__instance_parts.append(parted_inst_dic)
            #self.__get_cross_fold(instance_dic) # fills __times_parts, self.__feature_parts, self.__status_parts
        
        self._aspeed_opt = args_.aspeed_opt
        
        #global_thread_time_dic = dict((i,0) for i in range(1,2))
        #global_thread_rmse_dic = dict((i,0) for i in range(1,2))
        #global_thread_timeout_dic = dict((i,0) for i in range(1,2))
        #global_spend_time = dict((x,0) for x in solver_list)
        #solver_stats = dict((i,{}) for i in range(1,2))
        
        inst_par10_dict = {}
        #for instance in instance_dic.values():
        #    inst_par10_dict[instance.get_name()] = 0
        timeouts = 0 #[]
        
        for iteration in range(0, self.__n_folds):
            Printer.print_c("\n\n\t>>>\t>>> TREEFOLIO - %d-th iteration TRAINING <<<\t<<<\n\n" %(iteration + 1))
            if self.__n_folds > 1:
                instance_train = {}
                for i in range(0,self.__n_folds):
                    if (iteration == i):
                        continue
                    instance_train.update(self.__instance_parts[i])
                instance_test = self.__instance_parts[iteration]
            else:
                Printer.print_w("NO TRAINING TEST SPLIT!")
                instance_train = instance_dic
                instance_test =  instance_dic
            
            # TRAINING
            tree, res = self.train(args_, instance_train, solver_list, config_dic)
            
            Printer.print_c("\t>>>\t>>> TREEFOLIO - %d-th iteration TEST <<<\t<<<" %(iteration + 1))
            
            # TEST / EVALUATION
            for instance in instance_test.values():
                result_value, time, selected_solvers, status = tree.decidePseudo([],instance,args_, solver_list)
                
                #global_spend_time = dict((solver,global_spend_time.get(solver,0)+ spend_time_dict.get(solver,0)) for solver in solver_list)
                #solver_stats = self._solver_frequency(selected_solvers, solver_stats)
                inst_par10_dict[instance.get_name()] = time # Due to Crossvalidation, each instance is only tested once...
                if time >= args_.cutoff:
                    timeouts += 1
                
            #Printer.print_verbose(str(global_thread_timeout_dic))
            #Printer.print_verbose(str(global_thread_time_dic))
            #Printer.print_nearly_verbose(str(json.dumps(solver_stats, indent=2)))
        
        sum_time = 0
        for instance in instance_dic.values():
            sum_time += inst_par10_dict[instance.get_name()]
        
        Printer.print_verbose(str(inst_par10_dict))
        
        spend_time_dict = dict((solver,0) for solver in solver_list)
        
        avg_time = 0
        for instance in instance_dic.values():
            times = instance.get_runtimes()
            min_time = min(times)
            solver_index = times.index(min_time)
            solver_name = solver_list[solver_index]
            spend_time_dict[solver_name] += min_time
            avg_time += min_time
        
        avg_time = avg_time / len(instance_dic)
        
        oracle_avg_time, oracle_spend_time_dict =  avg_time, spend_time_dict
        #oracle_avg_time, oracle_spend_time_dict = self._oracle_performance(instance_dic, solver_list)
        
        Printer.print_c("\n >>> Oracle Evaluation <<<\n")
        Printer.print_c("average time: %f" %(oracle_avg_time))
        Printer.print_c("(all unsolvable instances were filtered while reading data)")
        
        Printer.print_c("\n >>> Cross Fold Evaluation <<<\n")
        Printer.print_c("Timeouts: %s" %(str(timeouts)))
        Printer.print_c("PAR10 Time: %s" %(str(sum_time)))
        Printer.print_c("Average PAR10 Time: %s" %(str(sum_time/len(inst_par10_dict))))
        
        
        #Printer.print_c("Timeouts per #Thread: %s" %(str(global_thread_timeout_dic)))
        #Printer.print_c("Time (PAR10) per #Thread: %s" %(str(global_thread_time_dic)))
        #Printer.print_c("Selection Positions : %s" %(str(self._selection_stats)))
        #global_thread_avg_dic = {}
        #global_thread_par10_dic = {}
        #for thread, time in global_thread_time_dic.items():
        #    global_thread_par10_dic[thread] = time / len(instance_dic)
        #    global_thread_avg_dic[thread] = self._extract_par1_from_par10(time, global_thread_timeout_dic[thread], args_.cutoff) / len(instance_dic)
        #for thread, squared_error in global_thread_rmse_dic.items():
        #    global_thread_rmse_dic[thread] = math.sqrt(squared_error / len(instance_dic))
        #    #global_thread_rmse_dic[thread] = math.sqrt(math.exp(squared_error / len(instance_train_dic)))
        #Printer.print_c("PAR1 per #Thread: %s" %(str(global_thread_avg_dic)))
        #Printer.print_c("PAR10 per #Thread: %s" %(str(global_thread_par10_dic)))
        #Printer.print_c("RMSE per #Thread: %s" %(str(global_thread_rmse_dic)))
        #Printer.print_nearly_verbose("Solver Selection Frequencies (#Threads -> Solvers):")
        #Printer.print_nearly_verbose(str(json.dumps(solver_stats, indent=2)))
        #Printer.print_c("Time used by each solver: %s" %(str(global_spend_time)))
        #Printer.print_c("Optimal Time used by each solver: %s" %(str(oracle_spend_time_dict)))
        
        #if self._print_file:
        #    self._write_csv_runtimes(self._print_file, instance_dic, inst_par10_dict, solver_list)
        
        #return global_thread_par10_dic[1], inst_par10_dict
    
if __name__ == '__main__':
    
    Printer.print_c("Hello, I am your claspfolio Trainer!")
    Printer.print_c("Please visit us at: http://potassco.sourceforge.net/")
    trainer = Trainer()
    trainer.main(sys.argv[1:])
    
