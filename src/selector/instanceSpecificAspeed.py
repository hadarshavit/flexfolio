'''
Created on June 9, 2015

@author: David Bergdoll
'''
from selector import Selector
from misc.printer import Printer
from trainer.aspeed.aspeedAll import AspeedAll
from trainer.base.instance import Instance

import math
import operator
import os

class InstanceSpecificAspeed(Selector):
    '''
        Runtime scheduling: run aspeed on a set of nearest neighbors to
        the current instance
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

        k =  se_dic["approach"]["k"]
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
        dic_solver_scores = self.__map_ids_2_names(average_perfs, se_dic["configurations"])


        instance_dic = self.__write_instance_dic(k_nearest_perfs)
        solver_list = dic_solver_scores.keys()

        scheduler = AspeedAll(se_dic["approach"]["clasp"],
                              se_dic["approach"]["gringo"],
                              runsolver = se_dic["approach"]["runsolver"],
                              enc = se_dic["approach"]["enc"],
                              time_limit = cutoff,
                              mem_limit = se_dic["approach"]["mem_limit"],
                              num_solvers = se_dic["approach"]["max_solver"],
                              opt_mode = se_dic["approach"]["opt_mode"],
                              max_pre_slice = se_dic["approach"]["pre_slice"],
                              threads = se_dic["approach"]["threads"]
                              )

        printer_disabled = Printer.disable_printing
        if not printer_disabled:
            Printer.disable_printing = True
        schedule = scheduler.optimize_schedule_online(instance_dic,
                                                      solver_list,
                                                      cutoff,
                                                      se_dic["approach"]["feat_time"],
                                                      se_dic["approach"]["model_dir"])

        if not printer_disabled:
            Printer.disable_printing = False

        dic_thread_schedule = self.transform_schedule(schedule, dic_solver_scores, cutoff)

        # if aspeed delivers an empty schedule, give all time to the solver with the best score
        if not dic_thread_schedule:
            dic_solver_scores = self.__map_ids_2_names(average_perfs, se_dic["configurations"])
            sorted_scores = sorted(dic_solver_scores.iteritems(),key=operator.itemgetter(1))
            dic_thread_schedule = {1:[(sorted_scores[0][0], (sorted_scores[0][1], cutoff))]}

        Printer.print_verbose(str(dic_thread_schedule))

        return dic_thread_schedule

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

    def __write_instance_dic(self, k_nearest_perfs):
        '''
            write a dictionary instance name -> Instance()
            Args:
                k_nearest_perfs: list of performance vectors
        '''
        instance_dic = {}
        id = 0
        for perfs in k_nearest_perfs:
            id += 1
            name = "i%i" %id
            instance = Instance("i%i" %id)
            instance._cost_vec = perfs
            instance_dic[name] = instance

        return instance_dic

    def transform_schedule(self, aspeed_schedule, dic_solver_scores, cutoff):
        '''
            transform the schedule dictionary provided by aspeed into the list form of selectors
        '''
        cores = aspeed_schedule.keys()
        dic_thread_schedule = {}
        core_index = 1
        for core in cores:
            schedule = []
            # count solvers and total time used for this core
            solver_count = 0
            time_sum = 0
            for solver, time in aspeed_schedule[core].iteritems():
                if solver == "claspfolio":
                    continue
                solver_count += 1
                time_sum += time
            # divide the remaining time equally between all solvers
            if solver_count == 0:
                continue
            bonus_time = (cutoff - time_sum)/solver_count

            # write the score list
            for solver, time in aspeed_schedule[core].iteritems():
                if solver == "claspfolio":
                    continue
                schedule.append((solver, (dic_solver_scores[solver], time + bonus_time)))

            dic_thread_schedule[core] = schedule



        return dic_thread_schedule