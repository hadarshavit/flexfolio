'''
Created on July 2, 2013

@author: CVSH
'''

import copy
import os
import re

# algorithm selection methods import
from selector.regression import Regression
from selector.NN import NearestNeighbor
from selector.MajorityVoter import MajorityVoter
from selector.Ensemble import Ensemble
from selector.classVoting import ClassVoter
from selector.classMulti import ClassMulti
from selector.clustering import Clustering

from treefolio.treeex.feature_extractor import TreefolioFeatureExtractor
from treefolio.treeex.instance_translator import TreefolioInstanceTranslator
from trainer.main_train import Trainer
from trainer.evalutor.crossValidator import CrossValidator
from claspfolio import Claspfolio
from misc.printer import Printer

class Decisionstep(object):
    '''
        Class for a Decisionstep
    '''
    
    def __init__(self, oracle, cross):
        
        # Class variables
        self.nextstep = []
        self.feature_extractor = None
        self.translator = None
        self.model = None
        
        self.approach = None
        self.modelpath = ""
        self.activate_features = []
        self.remove_features = []
        self.oracle = oracle #args_.treefolio_oracle #False
        self.crossfold = cross #args_.treefolio_crossfold
        self.selection = {}
        
        self.feature_count = -1
        
        # Other Variables
        self.decision_name = []
        self.default_decision_name = "Default"
        self.default_next_number = 0
        self.extractor = "CLASPRE"
        self.command = ""
        self.feature_extractor_command = ""
        #self.translator_command = ""
        self.depends_on = []
        self.name = "Name"
        self.step_id = -1
        self.step_depth = 0
        self.fullfilled_dependencys = []
    
    def init_from_structure_no_recursion(self, structure):
        '''
            Builds a part of the Structure,
            the rest is done via builder.build_step or during training via builder.read_and_build_from_data
        '''
        self.name = structure["name"]
        self.selection = structure["selection"]
        self.default_next_number = structure["default"]
        self.decision_name = structure["names"]
        self.modelpath = structure["modelpath"]
        self.activate_features = structure["activate_features"]
        self.approach = structure["approach"]
        self.command = structure["command"]
        self.feature_count = structure["feature_count"]
        if self.feature_count <= 0:
            Printer.print_w("No Features in this Step!")
        if self.command:
            self.init_translator()
        self.feature_extractor_command = structure["feature_extractor_command"]
        if self.feature_extractor_command:
            self.init_extractor()
    
    def init_translator(self):
        self.translator = TreefolioInstanceTranslator(self.command)
    
    def init_extractor(self):
        if self.feature_extractor_command:
            if self.feature_extractor_command.upper() in Claspfolio.extractor:
                self.feature_extractor = Claspfolio.extractor[self.feature_extractor_command.upper()]()
            else:
                Printer.print_w("Extractor "+self.feature_extractor_command+" not found.")
        #self.feature_extractor = TreefolioFeatureExtractor(self.feature_extractor_command)
    
    def decide(self, features, instance, args):
        
        new_features = []
        
        Printer.print_c("")
        Printer.print_c("\tStep: \t"+self.name)
        Printer.print_c("")
        
        if self.translator: # Translate the instance, add features if possible
            Printer.print_verbose("-> Translating ...")
            instance, new_features = self.translator.translate(instance)
            Printer.print_verbose("-> ... done")
        else:
            Printer.print_verbose("-> No Translation")
            
        Printer.print_verbose("")
        if not new_features and self.feature_extractor: # Extract features
            Printer.print_verbose("-> Extracting ...")
            new_features = self.feature_extractor.extract(instance)
            Printer.print_verbose("-> ... done")
        else:
            Printer.print_verbose("-> No Extraction")
        
        Printer.print_verbose("")
        if self.activate_features and new_features:
            Printer.print_verbose("-> Adding Features")
            if len(new_features) == self.activate_features[1]-self.activate_features[0]:
                features.extend(new_features)
            else:
                Printer.print_w("Warning! Number of Features does not fit!")
                si = self.activate_features[1]-self.activate_features[0]
                if len(new_features) > si:
                    features.extend(new_features[0:si])
                else:
                    features.extend(new_features)
                    features.extend([0] * (si-len(new_features)))
                    Printer.print_w("Too few Features ("+str(len(new_features))+" of "+str(si)+")!! Appending "+str(si-len(new_features))+" times 0 as Features")
        else:
            Printer.print_verbose("-> No Features to be added")
        
        Printer.print_verbose("")
        for ind in self.remove_features:
            if ind < len(features):
                del features[ind]
        
        Printer.print_c("Possible Choices: "+str(self.decision_name))
        
        if len(self.nextstep) == 1: # If nothing to decide, return default.
            Printer.print_c("Decision: "+str(self.default_next_number)+" -> "+self.decision_name[self.default_next_number])
            resn, time, solver, status = self.nextstep[self.default_next_number].decide(features, instance, args)
            result_value = []
            result_value.append(self.default_decision_name)
            result_value.extend(resn)
            return result_value, time, solver, status
        
        if len(features) == self.feature_count:
            if self.model:
                result = self.eval(features)
            else:
                result = self.load_model_and_eval(features, args)
            result_number = self.decision_name.index(result[0])
            #result_val = result[1]
        else:
            Printer.print_w("Missing Features!!! Using default Choice!")
            result_number = self.default_next_number
            #result_val = -1
            
        result_value = []
        if len(self.decision_name) > result_number:
            result_value.append(self.decision_name[result_number])
        else:
            result_value.append(self.default_decision_name)
        
        if len(self.nextstep) > result_number:
            Printer.print_c("Decision: "+str(result_number)+" -> "+self.decision_name[result_number])
            resn, time, solver, status = self.nextstep[result_number].decide(features, instance, args)
            result_value.extend(resn)
            return result_value, time, solver, status
        else:
            raise Exception("Could not make a Decision: No Decision Steps possible, result would be "+str(result_number))
        
    def decidePseudo(self, features, instance, args, solver_list):
        
        Printer.print_verbose("\tStep: \t"+self.name)
        
        if self.activate_features:
            r = range(self.activate_features[0],self.activate_features[1])
            if r:
                o_feats = instance.get_features()
                if o_feats:
                    if not features:
                        features = []
                    for z in r:
                        if len(o_feats)>z:
                            features.append(o_feats[z])
                    for ind in self.remove_features:
                        if ind < len(features):
                            del features[ind]
                    if not features:
                        features = None
        
        Printer.print_verbose("Possible Choices: "+str(self.decision_name))
        
        if len(self.nextstep) == 1: # If nothing to decide, return default.
            Printer.print_c("Decision: "+str(self.default_next_number)+" -> "+self.decision_name[self.default_next_number])
            resn, time, solver, status = self.nextstep[self.default_next_number].decidePseudo(features, instance, args, solver_list)
            result_value = []
            result_value.append(self.default_decision_name)
            result_value.extend(resn)
            return result_value, time, solver, status
        
        if len(features) == self.feature_count:
            if self.model:
                result = self.eval(features)
            else:
                result = self.load_model_and_eval(features, args)
            #print result;
            result_number = self.decision_name.index(result[0]) # index of *best* decision
        else:
            Printer.print_w("Missing Features!!! Using default Choice!")
            result_number = self.default_next_number
            
        result_value = []
        if len(self.decision_name) > result_number:
            result_value.append(self.decision_name[result_number])
        else:
            result_value.append(self.default_decision_name)
        
        if len(self.nextstep) > result_number:
            Printer.print_c("Decision: "+str(result_number)+" -> "+self.decision_name[result_number])
            resn, time, solver, status = self.nextstep[result_number].decidePseudo(features, instance, args, solver_list)
            result_value.extend(resn)
            return result_value, time, solver, status
        else:
            raise Exception("Could not make a Decision: No Decision Steps possible, result would be "+str(result_number))
        
    def load_model_and_eval(self, features, args):
        #if self.approach and not isinstance(self.approach, str):
        #    print self.approach
        #if isinstance(self.approach, dict):
        #    #print "% approach: "
        #    #print self.approach
        #    #self.approach = self.approach["approach"]
        if self.approach and isinstance(self.approach, str) and self.approach.upper() in Claspfolio.selector:
            self.model = Claspfolio.selector[self.approach.upper()]()
            return self.eval(features)
        else:
            Printer.print_w("Could not make a decision. Using default decision.")
            return [ self.default_decision_name ]
    
    def eval(self, features):
        #list_conf_scores = self.model.select_algorithms(self.selection, features, self.modelpath)
        list_conf_scores = self.model.select_algorithms(self.selection, features, "")
        if not list_conf_scores:
            Printer.print_w("Could not make a decision. Using default decision.")
            return [ self.default_decision_name ]
        return list_conf_scores[0]
    
    def is_trained(self):
        return self.model != None
    
    def update_args(self,args):
        if self.approach and isinstance(self.approach,dict):
            for key in self.approach.keys():
                setattr(args, key, self.approach[key])
        
    def train(self, args, instance_dic, solver_list, config_dic, new_instance_dic):
        if self.is_trained():
            return True
        
        # Copy args
        args_ = copy.deepcopy(args)
        args_.model_dir += "/"+self.name
        self.update_args(args_) # New, Test
        path= unicode(args_.model_dir)
        self.modelpath = path
        if not os.path.exists(path):
            print "Path "+path+" does not exist. Creating."
            os.makedirs(path)
        
        # Copy dics, Modify!
        new_instance_dic_ = copy.deepcopy(new_instance_dic)
        for i in new_instance_dic_:
            new_instance_dic_[i]._set_runtimes([])
            new_instance_dic_[i].set_transformed_runtimes([])
        solver_list_ = []
        config_dic_ = {} #copy.deepcopy(config_dic)
        
        
        if self.activate_features: # If Features are added in this step, add them for training!
            r = range(self.activate_features[0],self.activate_features[1])
            if r:
                for i in new_instance_dic_:
                    feats = new_instance_dic_[i].get_features()
                    feats_n = new_instance_dic_[i].get_normed_features()
                    o_feats = instance_dic[i].get_features()
                    o_feats_n = instance_dic[i].get_normed_features()
                    if o_feats:# and feats:
                        if not feats:
                            feats = []
                        if not feats_n:
                            feats_n = []
                        for z in r:
                            if len(o_feats)>z:
                                feats.append(o_feats[z])
                            if len(o_feats_n)>z:
                                feats_n.append(o_feats_n[z])
                        for ind in self.remove_features:
                            if ind < len(feats):
                                del feats[ind]
                            if ind < len(feats_n):
                                del feats_n[ind]
                        if not feats:
                            feats = None
                        new_instance_dic_[i].set_features(feats)
                        #new_instance_dic_[i].set_normed_features(feats_n)
                    
        
        for step in self.nextstep: # Train next steps, get their results
            if not step.is_trained():
                result_name, result_times = step.train(args_, instance_dic, solver_list, config_dic, new_instance_dic_)
                if result_times:
                    
                    if len(self.nextstep) == 1: # If nothing to decide, return default.
                        self.default_next_number = 0
                        self.default_decision_name = step.name
                        return self.name, result_times
                    
                    solver_list_.append(result_name)
                    if not result_name in config_dic_:
                        if result_name in config_dic:
                            config_dic_[result_name] = config_dic[result_name]
                        config_dic_[result_name] = self.command # TODO
                    for i in result_times:
                        old = new_instance_dic_[i].get_runtimes()
                        if not old:
                            old = []
                        if result_times[i]:
                            old.append(result_times[i])
                        else:
                            Printer.print_e("Error : Empty Times "+str(result_times)+" at index "+i)
                            old.append(-512)
                        new_instance_dic_[i]._set_runtimes(old) # Update your instance_dic with result_stuff
                        new_instance_dic_[i].set_transformed_runtimes(copy.deepcopy(old)) # Update your instance_dic with result_stuff
        
        i = 0;
        hlpr = new_instance_dic_.values()[0]
        while not hlpr.get_features():
            i += 1
            hlpr = new_instance_dic_.values()[i]
        self.feature_count = len(hlpr.get_features())
        args_.n_feats = self.feature_count
        
        Printer.print_c("")
        Printer.print_c("Step "+self.name+": Number of Features: "+str(self.feature_count))
        Printer.print_c("")
        
        tr = Trainer()
        
        if self.oracle: # Find optimal runtime
            self.selection = tr.train(args_, new_instance_dic_, solver_list_, config_dic_, None, True)
            res_dic = {}
            for i in new_instance_dic_:
                tst = new_instance_dic_[i].get_runtimes()
                if tst:
                    res_dic[i] = min(tst)
                else:
                    res_dic[i] = -512
            
            return self.name, res_dic
        
        v = CrossValidator(False,False)
        
        #print "args_"
        #print args_
        #print "new_instance_dic_ "
        #print new_instance_dic_
        #print "solver_list_"
        #print solver_list_
        #print "config_dic_"
        #print config_dic_
         
        somedic, evaluation_result = v.evaluate(tr, args_, self.crossfold, new_instance_dic_, solver_list_, config_dic_, 0)
        self.selection = tr.train(args_, new_instance_dic_, solver_list_, config_dic_, None, True)
        
        conf = self.selection["configurations"]
        for s in conf:
            if conf[s]["backup"] is 0:
                self.default_next_number = self.decision_name.index(s)
                self.default_decision_name = s
                #Printer.print_c("Step Backup Solver: "+s)
                break;
        
        Printer.print_verbose("")
        Printer.print_verbose("Crossvalidation returns: "+str(evaluation_result))
        #Printer.print_c("Training returns: "+str(self.selection))
        Printer.print_c("")
        
        return self.name, evaluation_result
    