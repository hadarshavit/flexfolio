'''
Created on Jul 1, 2015

@author: manju
'''

import sys
import os
import operator
import inspect
import time

# http://stackoverflow.com/questions/279237/python-import-a-module-from-a-folder
global cmd_folder
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.append(cmd_folder)
#print(sys.path)

from input_parser.cmd_parser_aschallenge import Parser
from misc.printer import Printer
from trainer.training_parser.coseal_reader_aschallenge_test import CosealReader 

from selector.selectionApp import SelectionBase

#from misc.updater import Updater

class Flexfolio(object):
    '''
        main class of claspfolio
        execution plan:
            1. parse command line arguments (and config file)
            2. use featureExtractor to compute instance features of given instance
            3. select algorithms to execute (selector)
            4. execute algorithms
    '''

    def __init__(self):
        ''' Constructor '''
        self._parser = Parser()
        
    def main(self,sys_argv):
        '''
            main method of claspfolio, is directly called
            Parameter:
                sys_argv: command line arguments
        '''     
        args_, ex_dic, se_dic, al_dic = \
            self._parser.parse_command_line(sys_argv)
        pwd = os.path.split(args_.config)[0]
            
        # parse input
        reader = CosealReader()
        aslib_data, metainfo, algo_dict = reader.parse_coseal(args_.aslib_dir, args_)
        
        self.normal_mode(aslib_data, ex_dic, se_dic, al_dic, pwd, args_.output)

    def normal_mode(self, aslib_data, ex_dic, se_dic, al_dic, pwd, output_fn):
        '''
            normal mode:
                1. feature prediction
                2. algorithm selection
                3. run solver
            Args:
                aslib_data
                ex_dic: extractor dictionary
                se_dic: selection dictionary
                al_dic: algorithm dictionary 
        '''

        #pwd = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
        #pwd = os.path.realpath(os.path.join(pwd, ".."))
        
        with open(output_fn, "w") as fp:
            fp.write("InstanceID, runID, solver, timeLimit")
            for inst_ in aslib_data.itervalues():
                features = inst_._features 
                selector_name = se_dic["approach"]["approach"].upper()
                list_conf_scores = SelectionBase.select(selector_name, se_dic, features, pwd)
                for id, (solver, _) in enumerate(list_conf_scores): 
                    fp.write("%s, %d, %s, 999999999999999\n" %(inst_._name, id+1, solver))
                fp.flush()
        
              
if __name__ == '__main__':
    Printer.print_c("flexfolio")
    Printer.print_c("published under GPLv2")
    Printer.print_c("https://bitbucket.org/mlindauer/xfolio")
    flexfolio = Flexfolio()
    flexfolio.main(sys.argv[1:])
    