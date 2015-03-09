'''
Created on June 20, 2013

@author: CVSH
'''

from executor.executor import Executor
from executor.executorClaspmt import ExecutorClaspmt
from featureExtractor.claspre import Claspre
from featureExtractor.claspre2 import Claspre2
from featureExtractor.satzilla import SatZilla
from misc.printer import Printer
from misc.updater import Updater
#from trainer.parser.parser import Parser
from selector.Ensemble import Ensemble
from selector.MajorityVoter import MajorityVoter
from selector.NN import NearestNeighbor
from selector.classMulti import ClassMulti
from selector.classVoting import ClassVoter
from selector.regression import Regression
from treefolio.steps.builder import StepBuilder
from tfparser.exparser import TreeExParser
import operator
import os
import sys
import time
import inspect
import StringIO

# http://stackoverflow.com/questions/279237/python-import-a-module-from-a-folder
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.append(cmd_folder)


# feature extraction imports

# algorithm selection methods import



class Treefolio(object):
    '''
        main class of claspfolio
        execution plan:
            1. parse command line arguments (and config file)
            2. use featureExtractor to compute instance features of given instance
            3. select algorithms to execute (selector)
            4. execute algorithms
    '''
    
    def __init__(self):
        ''' Constructor, empty '''
        #self._parser = "" #Parser()
#         self._extractor = {"CLASPRE" : Claspre(), "CLASPRE2" : Claspre2(), "SATZILLA": SatZilla()}
#         self._selector = {"CLASSVOTER": ClassVoter(),
#                           "CLASSMULTI": ClassMulti(),
#                           "REGRESSION" : Regression(), 
#                           "NN": NearestNeighbor(), 
#                           "KMEANS": NearestNeighbor(), 
#                           "MAJORITY": MajorityVoter(),
#                           "ENSEMBLE": Ensemble()
#                           }
        #self._executor = Executor()
        
    def main(self, sys_argv):
        '''
            main method of claspfolio, is directly called
            Parameter:
                sys_argv: command line arguments
        '''
        parser = TreeExParser()
        args_ = parser._arg_parser.parse_args(sys_argv)
        #instance, ex_dic, se_dic, al_dic, features_stop, values_stop, up_dic, ori_config_file, o_dir, env_, clause_sharing = parser.parse(args_)
        
        if args_.inst == "-": # read from stdin
            memory_file = StringIO.StringIO()
            memory_file.writelines(sys.stdin.readlines())
            args_.inst = memory_file
        else:
            if not os.path.isfile(args_.inst):
                Printer.print_e("File not found: " + args_.inst)
            else:
                args_.inst = open(args_.inst,"r")
        
        Printer.print_c("")
        
        self.builder = StepBuilder()
        
        # Build tree from Structure File (different from Training)
        self.builder.build(args_.config,args_)
        
        if args_.verbose > 0:
            self.builder.test_print()
            Printer.print_c("")
        
        # Run decision and execute!
        print self.builder.first.decide([], args_.inst, args_)
    
if __name__ == '__main__':
    VERSION = "2.0"
    Printer.print_c("Hi, my name is claspfolio-%s. Nice to work for you!" %(VERSION))
    Printer.print_c("Please visit us at: http://potassco.sourceforge.net/")
    treefolio = Treefolio()
    treefolio.main(sys.argv[1:])
    
