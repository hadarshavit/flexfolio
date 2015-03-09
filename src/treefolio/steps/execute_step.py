'''
Created on July 2, 2013

@author: CVSH
'''
from subprocess import Popen
from tempfile import NamedTemporaryFile
import copy

from executor.executor import Executor
from trainer.main_train import Trainer
from misc.printer import Printer

class Executionstep(object):
    '''
        Class for a Executionstep
    '''
    
    def __init__(self):
        self.execcommand = [] # Contains all execution commands. Output File should be appended
        self.configuration_file = None
        self.configuration = {} #{"solvername":{"call":"echo \"run,solver,run!\""}}
        self.fake_score = [] #[[1,1]] #{}
        self.runtime_dictionary = None # Can be run with runtime_dic, then results are searched in the dictionary first!
        self.executor = None #executor()
        self.name = "Missing Name"
        self.result_index = 0
        self.execcommand = ""
        
        self.train_args_ = []
        self.train_instance_dic = {}
        self.train_solver_list = {}
        self.train_config_dic = {}
    
    def init_from_structure_no_recursion(self, structure):
        a = ""
        
    def load_executor(self, command, name, result_index):
        self.execcommand = command
        self.name = name
        self.result_index = result_index
        self.executor = Executor()
        
        self.configuration[name] = {"call": command }
        self.fake_score = [[name,1]]
        #self.fake_score[name] = [1, 1]
        
    def execute(self, instance, args):
        
        if (self.runtime_dictionary):
            if hasattr(instance, 'name'):
                if hasattr(self.runtime_dictionary, instance["name"]):
                    result = self.runtime_dictionary[instance["name"]]
                    if len(result) >= 3:
                        return result[0], result[1], result[2]
        
        if hasattr(instance, 'time'):
            if isinstance(instance.time,(list, tuple)) and (len(instance.time) > self.my_result_index):
                return instance.time[self.my_result_index]
        
        if (self.execcommand or self.executor):
            
            
            if "<instance>" in self.configuration[self.name]["call"]:
                self.configuration[self.name]["call"] = self.configuration[self.name]["call"].replace("<instance>",instance.name)
            
            Printer.print_c("Call: "+self.configuration[self.name]["call"])
            
            time, solver, status = self.executor.execute(1,self.configuration,self.fake_score,instance,None,False)
            
            ###
            #time = 0
            #solver = 0
            #status = "SAT"
            ###
            
            return time, solver, status
            
        else:
            raise Exception("Execution command missing!")
    
    def parse_results(self, outfile):
        return "nope"
    
    def decide(self, features, instance, args):
        # call the solver!
        time, solver, status = self.execute(instance, args)
        return [], time, solver, status
        
    def decidePseudo(self, features, instance, args, solver_list):
        
        ti = -512
        if not self.name in solver_list:
            print "Error: Solver "+self.name+" not found!"
            print "Contained Solvers: "+str(solver_list)
        else:
            index = solver_list.index(self.name)
            ti = instance.get_runtimes()[index]
        
        #return self.name, ti
        return [], ti, self.name, "Unknown"
    
        #time, solver, status = 0,0,0 #self.execute(instance, args)
        #return [], time, solver, status
        
    def is_trained(self):
        #return self.executor != None
        return False # We want to let train() be run!
    
    def train(self, args, instance_dic, solver_list, config_dic, new_instance_dic):
        '''
            Train Model ... Execution Steps do not have models.
        '''
        ti = {}
        if not self.name in solver_list:
            print "Error: Solver "+self.name+" not found!"
            print "Contained Solvers: "+str(solver_list)
        else:
            index = solver_list.index(self.name)
            for i in instance_dic:
                #if len(instance_dic[i].get_runtimes()) > index:
                    ti[i] = instance_dic[i].get_runtimes()[index]
                #else:
                #    print self.name+": index "+str(index)+">="+str(len(instance_dic[i].get_runtimes()))
                #    ti[i] = 20000 #self.cutoff
        # Get your solver and configuration out of dics and return them
        return self.name, ti
    