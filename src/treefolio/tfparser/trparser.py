'''
Created on Nov 8, 2012

@author: cvsg
'''
from trainer.parser.parser import TainerParser

class TreeTrainerParser(TainerParser):
    '''
     parse the input of the claspfolio trainer
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TainerParser.__init__(self)
        self.__init_treefolio_parser()
        
        
    def __init_treefolio_parser(self):
        '''
            init argparse object with all command line arguments
        '''
        
        TREE_GROUP = self._arg_parser.add_argument_group("Treefolio Options")
        TREE_GROUP.add_argument('--treeoracle', dest='treefolio_oracle', action='store_true', default=False, required=False, help='use oracle performance instead of evaluation result')
        TREE_GROUP.add_argument('--treeconfig', dest='treefolio_configfile', action='store', required=True, help='location of the configuration file')
        TREE_GROUP.add_argument('--treecrossfold', dest='treefolio_crossfold', action='store', type=int, default=5, required=False, help='number of folds for crossfold evaluation')
        TREE_GROUP.add_argument('--treefoliomodel', dest='treefolio_modelfile', action='store', default="./models/treefolio.conf", required=False, help='File to save and load model configuration info')
        TREE_GROUP.add_argument('--treestartfeatures', dest='treefolio_start_features', action='store_true', default=False, required=False, help='use external Features for each instance')