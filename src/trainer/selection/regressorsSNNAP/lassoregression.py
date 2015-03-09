'''
Created on 29.05.2013

@author: manju
'''

from sklearn.linear_model import Lasso
import numpy

from trainer.selection.snnap import SNNAPTrainer
from trainer.selection.metatuner import MetaTuner
from misc.printer import Printer

class LassoRegressionSNNAP(SNNAPTrainer):
    '''
       uses a lasso regression for each algorihm
       sklearn.linear_model.Lasso
       
       http://scikit-learn.org/0.13/modules/generated/sklearn.linear_model.Lasso.html
    '''


    def __init__(self, k,  best_n=3, alpha=None, save_models=True):
        '''
        Constructor
        '''
        SNNAPTrainer.__init__(self, k, best_n, save_models)
        self.__alpha = alpha
    
    def __repr__(self):
        return "LASSO-SNNAP"
    
    def _train_regression(self, y, X, weights):
        '''
            @overrides(ClassifierVoter)
            returns fitted model
        '''
        
        X = numpy.array(X)
        y = numpy.array(y)
        weights = numpy.array(weights)        
        
        Printer.print_verbose("Train Regression Model with Lasso Regression")  
        
        if self.__alpha is None:
            tuner = MetaTuner()
            tuned_parameters = [{'alpha': [16, 8, 4, 2, 1, 0.5, 0.125], 'max_iter': [100]}]
            best_reg = tuner.tune(X, y, Lasso, tuned_parameters)
            alpha = best_reg.alpha
        else:
            alpha = self.__alpha
            
        trainer = Lasso(alpha = alpha)
        try:
            trainer.fit(X,y)
            return trainer
        except ValueError:
            Printer.print_w("zero-size array to reduction operation maximum which has no identity")
            return None
        
        
        
        