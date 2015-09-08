import numpy as np
import itertools as it
import os
import csv

from sklearn.linear_model import SGDClassifier # Stochastic Gradient Descent
from sklearn import cross_validation # cross validation

from datetime import datetime  # to calculate run time
from datetime import timedelta # to calculate run time

from sklearn.preprocessing import normalize


'''
    scores as accuracy
    
    look at
        -- overall parameter importance
        -- parameter importance for different instance clusters
            ** e.g. 12 datasets are similar (grouped/clustered) wrt latent features
               then check parameter importance only for that cluster
    
    TODO
    -- Check how normlization works
    -- Check cross validation randomness??
    -- Check - Need normalization???
    -- Use each fold score as a performence entry, k scores per algorithm + configuration
        ** Compare with the average score (error/accuracy) case
'''

__output_folder__ = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")

__seed__ = 123456789

k_for_CV = 10

data_set_folder = "E:/_DELLNtb-Offce/_Bitbucket SRC/ainovelty/UCI-dataset/"

dataset_files_list = [  
                "iris-numbered.data", 
                ##"adult-numbered.data", 
                ##letter-recognition.data",   
                
               ]
''' 
               ## "adult-numbered.data", ##aslinda calisiyor, cok buyuk diye gecici kapattim
                "australian.dat", 
                "balance-scale.data",
                ##"banding",
                "breast-cancer-wisconsin.data",
                ##"processed.cleveland.data", ##cleveland with 4 classes-use 2 class version
                "cleveland-heart-303-2class.data", ## cleveland 2 class version
                "crx.data", ##credit
                "pima-indians-diabetes.data",
                "german.data-numeric",
                "glass.dat",
                "heart.dat",
                "hepatitis.data",
                "ionosphere.data",
                "iris-numbered.data",
                "letter-recognition.data",
                "monks-1.test",
                "monks-2.test",
                "monks-3.test",
                "agaricus-lepiota.data", ## Mushroom
                "sat.all.data", ## satimage
                "segmentation.data",
                "sonar.all-data",
                "vehicle.dat", 
                "house-votes-84.data", ## votes
                ##"waveform",
                "wine.data"
'''   



## SGDClassifier configuration dictionary
## http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
sgd_dict = {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
            'penalty': ['none', 'l2', 'l1', 'elasticnet'],
            'alpha': np.arange(0.01, 0.5, 0.1),##0.01 - it should be > 0
            'l1_ratio': np.arange(0, 1, 0.1),##0.01
            ##'fit_intercept': [True, False],
            ##'n_iter': np.arange(1, 10),
            ##'shuffle': [True, False],
            ##'epsilon': np.arange(0, 1, 0.1), ##0.01
            ##'power_t': np.arange(0, 1, 0.1)
            }



class Bunch(dict):
    """Container object for datasets: dictionary-like object that
       exposes its keys as attributes."""

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self



def load_dataset(dataSet):
    """Load and return the dataset (classification).

    The iris dataset is a classic and very easy multi-class classification
    dataset.

    =================   ==============
    Classes                          3
    Samples per class               50
    Samples total                  150
    Dimensionality                   4
    Features            real, positive
    =================   ==============

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification labels,
        'target_names', the meaning of the labels, 'feature_names', the
        meaning of the features, and 'DESCR', the
        full description of the dataset.

    Examples
    --------
    Let's say you are interested in the samples 10, 25, and 50, and want to
    know their class name.

    >>> from sklearn.datasets import load_iris
    >>> data = load_iris()
    >>> data.target[[10, 25, 50]]
    array([0, 0, 1])
    >>> list(data.target_names)
    ['setosa', 'versicolor', 'virginica']
    """
 
    data_file_name = dataSet
    print(data_file_name)
    
    data = []
    target = []
    
    with open(os.path.join(data_file_name)) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        
        for ir in reader:
            
            for (i, item) in enumerate(ir[:-1]):
                if item == '?':
                    ir[i] = '0' ##-100
        
            ##print(" >> arr ",ir[:-1])
            data.append(np.asarray(ir[:-1], dtype=np.float))
            target.append(np.asarray(ir[-1], dtype=np.int))
    
    
    n_samples = len(data)
    n_features = len(data[0])
    data_new = np.empty((n_samples, n_features))
    target_new = np.empty((n_samples,), dtype=np.int)

    for (i, values) in enumerate(data):
        data_new[i] = values
        
    for (i, values) in enumerate(target):
        target_new[i] = values
    
    
    #print(data)
    #print(target)
            
    return Bunch(data=data_new, target=target_new,
                 target_names="",
                 DESCR="",
                 feature_names=['sepal length (cm)', 'sepal width (cm)',
                                'petal length (cm)', 'petal width (cm)'])
 
            
## END -------------------------



def write_scores(csv_file_name, dataset, clsf_full_name_list, score_list):
    
    i = 0
    j = 0
    with open(csv_file_name, 'a') as f:
        # Add titles
        if os.stat(csv_file_name).st_size == 0:
            f.write('Datasets')
            for clsName in clsf_full_name_list:
                f.write(','+clsName)
                
            f.write('\n')
            
        # Add values
        for j in range(0, k_for_CV):
            f.write(dataset+'-CV'+str(j+1)+',')
            for i in range(0, len(score_list)):
                f.write(str(score_list[i][j])+',')
        
            f.write('\n')
                
    f.close()
    
def write_avg_scores(csv_file_name, dataset, clsf_full_name_list, avg_score_list):
    
    with open(csv_file_name, 'a') as f:
    # Add titles
        if os.stat(csv_file_name).st_size == 0:
            f.write('Datasets')
            for clsName in clsf_full_name_list:
                f.write(','+clsName)
                
            f.write('\n')
        
        # Add values 
        f.write(dataset+',')
        for i in range(0, len(avg_score_list)):
            f.write(str(avg_score_list[i])+',')
    
        f.write('\n')
            
    f.close()
    

def write_time_elapse(csv_file_name, dataset, clsf_full_name_list, time_list):
    
    with open(csv_file_name, 'a') as f:
    # Add titles
        if os.stat(csv_file_name).st_size == 0:
            f.write('Datasets')
            for clsName in clsf_full_name_list:
                f.write(','+clsName)
                
            f.write('\n')
        
        # Add values 
        f.write(dataset+',')
        for i in range(0, len(time_list)):
            f.write(str(time_list[i])+',')
    
        f.write('\n')
            
    f.close()


def get_clsf_full_name(conf_dict):
    clsf_full_name = ''
    for key,value in conf_dict.iteritems():
        clsf_full_name += '-'+str(key)+ '-' + str(value)
        
    return clsf_full_name




def gen_ml_conf_data():
    
    global __output_folder__
    ##__output_folder__ = os.path.join(__output_folder__, exp.output_folder_name)
    if not os.path.exists(__output_folder__):
        os.makedirs(__output_folder__)
        

    ## generate all configuration combinations
    ## http://stackoverflow.com/questions/3873654/combinations-from-dictionary-with-list-values-using-python
    varNames = sorted(sgd_dict)
    configurations = [dict(zip(varNames, prod)) for prod in it.product(*(sgd_dict[varName] for varName in varNames))]    
    
    print configurations   
    
    clsf_full_name_list = []
    
    
    for dataset_file in dataset_files_list:
        
        score_list = []
        avg_score_list = []
        time_list = [] 

        ds = data_set_folder+dataset_file
        
        ##dataset = np.loadtxt(ds, delimiter=",")
        ##data = normalize(dataset[:,0:(len(dataset[0])-1)], axis = 0)
        ##target = dataset[:,(len(dataset[0])-1)]
        
        dataset = load_dataset(ds) 
        print dataset.data
        data = normalize(dataset.data, axis = 0)
        print data
        target = dataset.target
        
        for conf in configurations:
            
            ## TODO : do only once
            clsf_full_name_list.append("SGD"+get_clsf_full_name(conf))
        
            conf['random_state'] = __seed__
            
            ##cls_full_name_list.append('SVM-g'+str(pr[0])+'-C'+str(pr[1]))
            
            print conf
            
            ## send dictionary as function arguments
            ## http://stackoverflow.com/questions/21986194/how-to-pass-dictionary-items-as-function-arguments-in-python
            clf = SGDClassifier(**conf)
            
            start_time = datetime.now()
            scores = cross_validation.cross_val_score(clf, data, target, cv=k_for_CV)
            end_time = datetime.now()
            
            dt = end_time - start_time
            time_elapsed_in_ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0 # time elapsed in milliseconds
            
            print scores
            
            score_list.append(scores)
            time_list.append(time_elapsed_in_ms)
        
            print("Accuracy: mean,+-std", scores.mean(), scores.std() / 2)
            
            avg_score_list.append(scores.mean())
            
        
        sgd_scores_file = os.path.join(__output_folder__, "SGD-scores.csv")
        sgd_avg_scores_file = os.path.join(__output_folder__, "SGD-avg-scores.csv")
        sgd_time_file = os.path.join(__output_folder__, "SGD-time.csv")
        
        write_scores(sgd_scores_file, dataset_file, clsf_full_name_list, score_list)
        write_avg_scores(sgd_avg_scores_file, dataset_file, clsf_full_name_list, avg_score_list)
        write_time_elapse(sgd_time_file, dataset_file, clsf_full_name_list, time_list)
    
   
def main():
    gen_ml_conf_data()


if __name__ == "__main__":
    main()
 
