'''
Created on May 13, 2015

@author: David Bergdoll
'''
from selector import Selector
from misc.printer import Printer

import math
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

        if isinstance(model_file, unicode):
            data_list = self.__create_data_list(model_file, pwd, se_dic["approach"]["n_feats"])
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
            construct a dictionary representing a schedule using the benchmark data
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
        slots = []  # slots per solver
        scores = [] # score sum for each solver (needed for tie breaking if solver count is limited)
        backup_slots = 0
        for i in range(0, n_perfs):
            slots.append(0)
            scores.append(0)

        #  give each solver a slot for every solved instance
        for perfs in k_nearest_perfs:
            solved = False
            for i in range(0, n_perfs):
                if perfs[i] <= cutoff:
                    slots[i] += 1
                    solved = True
                scores[i] += min(perfs[i], cutoff)
            # unsolved instances go to the backup solver
            if not solved:
                backup_slots += 1


        max_solvers = se_dic["approach"]["max_solvers"]
        # if specified, limit the list of solvers that get time slots
        if max_solvers > 0 and max_solvers < n_perfs:
            # get the minimal number of slots needed to stay in the schedule
            slot_threshold = sorted(slots, reverse=True)[max_solvers - 1]
            used_solvers = 0
            tiebreak_list = []
            for i in range(0, len(slots)):
                # erase the slots of filtered solvers
                if slots[i] < slot_threshold:
                    slots[i] = 0
                # solvers that meet the threshold are kept for comparison
                elif slots[i] == slot_threshold:
                    tiebreak_list.append((i, scores[i]))
                # solvers above the threshold stay in
                else:
                    used_solvers += 1
            remaining_slots = max_solvers - used_solvers
            tiebreak_list.sort(key = lambda x:x[1])
            counter = 0
            # decide which of the tied solvers to keep, depending on their scores
            for index,_ in tiebreak_list:
                if counter >= remaining_slots:
                    slots[index] = 0
                counter += 1

        time_slot = cutoff/(sum(slots) + backup_slots)
        tot_time = 0;

        # assign time to each solver proportional to the number of solved instances
        solver_times = []
        for i in range(0, n_perfs):
            solver_times.append(slots[i] * time_slot)
            tot_time += solver_times[i]


        # save computed solver time slots and their average performances in pairs
        solver_times_scores = zip(average_perfs, solver_times)

        dic_solver_times = self.__map_ids_2_names(solver_times_scores, se_dic["configurations"])
        sorted_times = dic_solver_times.items()
        sorted_times.sort(key=lambda x:x[1][1], reverse=True)

        if tot_time < cutoff:
            backup_solver = sorted_times[0]
            solver = backup_solver[0]
            score = backup_solver[1][0]
            time = backup_solver[1][1] + cutoff - tot_time
            sorted_times[0] = (solver, (score, time))

        sorted_times = [element for element in sorted_times if element[1][1] > 0]

        Printer.print_verbose(str(sorted_times))

        return {1: sorted_times}

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
            