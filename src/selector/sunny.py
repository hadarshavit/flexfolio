'''
Created on May 13, 2015

@author: David Bergdoll
'''
from selector import Selector
from misc.printer import Printer

import math
import operator
import os

class Sunny(Selector):
    '''
        Sunny algorithm introduced by R. Amadini, M. Gabrielli, J. Mauro
        k-NN selection followed by heuristic scheduling based on the number of
        solved instances per candidate solver

        largely based on kNN.py
    '''


    def __init__(self):
        '''
        Constructor
        '''
        
    
    def select_algorithms(self, se_dic, features, pwd=""):
        ''' select nearest neighbors, build schedule '''
        
        self._set_base_args(se_dic, features)
        
        self._normalize(self._normalization["approach"], self._normalization["filter"], features)
        
        if not self._features:
            return None
        
        model_file = se_dic["approach"]["model"]
        
        if isinstance(model_file, str):
            data_list = self.__create_data_list(model_file, pwd, len(features))
        else:
            data_list = model_file
        
        k = se_dic["approach"]["k"]
        cutoff = se_dic["approach"]["cutoff"]
        
        schedule = self.__build_schedule(data_list, self._features, k, cutoff, se_dic)

        return schedule
        
    def __create_data_list(self, model_file, pwd, n_feats):
        '''
            read saved performance and feature data
        '''
        fp = open(os.path.join(pwd,model_file),"r")
        
        data_list = []
        for line in fp:
            line = line.replace("\n","")
            if not line.startswith("@") and line != "":
                splitted = line.split(",")
                inst_features = map(float,splitted[:n_feats])
                perfs = map(float,splitted[n_feats+1:])
                weight = float(splitted[n_feats])
                point = {"features" : inst_features,
                         "weight" : weight,
                         "perfs" : perfs
                     }
                data_list.append(point)
        fp.close()
        
        return data_list
    
    def __build_schedule(self, data_list, features, k, cutoff, se_dic):
        '''
            look through all training instances and remember for each encountered solver the nearest distance
        '''
        
        tuples_dist_perfs = []
        for point in data_list:
                inst_features = point["features"]
                perfs = point["perfs"]
                weight = point["weight"]
                # weights are decreased to decrease the influence,
                # hence, 1/weight increases the distance
                distance = self.__get_euclidean_dist(features,inst_features) * (1/weight)

                #if distance < 0.00001:
                    #Printer.print_w("I may encountered the given instance previously!")
                    #Printer.print_w(str(features))
                    #Printer.print_w(str(inst_features))
                    
                tuples_dist_perfs.append((distance,perfs))
        
        tuples_dist_perfs.sort()
        k_nearest_perfs = map(lambda x: x[1], tuples_dist_perfs[:k])
        n_perfs = len(k_nearest_perfs[0])
        average_perfs = reduce(lambda x,y: [sum(pair) for pair in zip(x, y)], k_nearest_perfs, [0]*n_perfs)

        # get the total number of timeslots
        slots = sum(sum(perf < cutoff for perf in perfs) for perfs in k_nearest_perfs) + sum(sum(perf < cutoff for perf in perfs) == 0 for perfs in k_nearest_perfs)
        slots = 0
        for perfs in k_nearest_perfs:
            solved = False
            for perf in perfs:
                if perf <= cutoff:
                    slots += 1
                    solved = True
            if not solved:
                slots += 1

        time_slot = cutoff/slots
        tot_time = 0;

        # assign time to each solver proportional to the number of solved instances
        solver_times = []
        for i in range(0, n_perfs):
            solver_times.append(sum(perfs[i] < cutoff for perfs in k_nearest_perfs) * time_slot)
            tot_time += solver_times[i]

        # save computed solver time slots and their average performances in pairs
        solver_times_scores = zip(average_perfs, solver_times)

        dic_solver_times = self.__map_ids_2_names(solver_times_scores, se_dic["configurations"])
        sorted_times = dic_solver_times.items()
        sorted_times.sort(key = lambda x:x[1][0])

        if tot_time < cutoff:
            best_score = sorted_times[0]
            solver = best_score[0]
            score = best_score[1][0]
            time = best_score[1][1] + cutoff - tot_time
            sorted_times[0] = (solver, (score, time))

        Printer.print_verbose(str(sorted_times))

        for (solver,tuple) in sorted_times:
            Printer.print_verbose("[%s]: %f, %f" %(solver, tuple[0], tuple[1]))
        
        return sorted_times
                
    def __get_euclidean_dist(self,vect1, vect2):
        squares = list( math.pow(f1-f2,2) for f1,f2 in zip(vect1,vect2))
        sum_squares = sum(squares)
        return math.sqrt(sum_squares)
    
    def __map_ids_2_names(self, scores, conf_dic):
        '''
            map id of solver to its name
        '''
        dic_name_score = {}
        for solver_name, meta_dic in conf_dic.items():
            id_ = meta_dic["id"]
            dic_name_score[solver_name] = scores[int(id_)]
        return dic_name_score
            