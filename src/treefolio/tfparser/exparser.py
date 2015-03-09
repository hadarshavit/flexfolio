'''
Created on Nov 8, 2012

@author: manju
'''

from parser.parser import Parser
import sys

class TreeExParser(Parser):
    '''
     parse the input of the claspfolio trainer
    '''


    def __init__(self):
        '''
        Constructor
        '''
        Parser.__init__(self);
        self.__init_treefolio_ex_parser()
        
        
    def __init_treefolio_ex_parser(self):
        '''
            init argparse object with all command line arguments
        '''
        
        TREE_GROUP = self._arg_parser.add_argument_group("Treefolio Options")
        TREE_GROUP.add_argument('--treeoracle', dest='treefolio_oracle', action='store_true', default=False, required=False, help='use oracle performance instead of evaluation result')
        TREE_GROUP.add_argument('--treecrossfold', dest='treefolio_crossfold', action='store', type=int, default=5, required=False, help='number of folds for crossfold evaluation')
