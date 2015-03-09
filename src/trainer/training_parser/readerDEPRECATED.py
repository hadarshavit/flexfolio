'''
Created on Jul 26, 2012

@author: manju
'''

# standard imports
import sys
import traceback
import math
from misc.printer import Printer
import json
from trainer.base.instance import Instance

class Reader(object):
    '''
        reads in runtime, feature and status data
        has to ensure that all three data types are available for each instance
        
    '''

    def __init__(self, cutoff, n_feats, filter_duplicates):
        '''
        Constructor
        '''
        self.__cutoff = cutoff
        self.__number_feats = n_feats
        self.__filter_duplicates = filter_duplicates
        self.__reset_attributes()
        
        self.__MINIMAL_RUNTIME = 0.005
        self.__PARX = 10 # if you change this, change validator function _extract_par1_from_par10
        
        self.__PRINTERR = True
        self.__DELIMETER = ","
        self.__SAT_UNSAT_TRANSLATION  = {"SAT": "1", "UNSAT": "-1"}
        
        self.__CORRELATION_TEST = False
        
    def __reset_attributes(self):
        self.__feature_dic = {}
        self.__runtime_dic = {}
        self.__status_dic = {}
        self.__ftime_dic = {}
        self.__invalid_f_runtime = {}
        self.__instance_dic = {}
        self.__solver_list = []
        
        self.__available = 0
        self.__presolved = 0
        self.__unsolveable = 0
        self.__too_few_f = 0
        self.__unknown = 0
        self.__n_dubs = 0
        
    def __parse_features(self, feature_file,):
        '''
            reads in the feature_file while maximal __number_feats are processed
            features are internally saved in the dictionary __feature_data_dic
            Args:
                feature_file: csv file with delimeter __delimeter containing all relevant features (string)
                              first coloumn is the instance name
            Returns:
                true if succesfull
                false otherwise
            Raises:
                IOError:   -
        '''
        try:
            filepointer = open(feature_file,"r")
        except IOError:
            if (self.__PRINTERR):
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
            return False
        for line in filepointer:
            line = line.replace("\n","")
            if line == "": # skip empty lines
                continue
            if line.startswith("@"): # arff format: meta information 
                continue
            line = line.split(self.__DELIMETER)
            values = []
            inst_name = line.pop(0) # remove header
            for value in line:
                try:
                    if value == "?":
                        values.append(0) # arff format missing value
                    else:
                        values.append(max([float(value), 0.0])) # minimal value 0; negative values are missing values (e.g. -512 satzilla)
                except ValueError:
                    pass
            if (values != []):    # filter empty lines
                if (len(values) > self.__number_feats):
                    Printer.print_w("Too many features (%d) for %s. TRUNCATE!" %(len(values),inst_name) )
                    values = values[0:self.__number_feats]
                if (len(values) < self.__number_feats):
                    Printer.print_w("Too few features (%d) for %s" %(len(values),inst_name ) ) 
                entry = self.__feature_dic.get(inst_name)
                self.__feature_dic[inst_name] = (values)
                if (entry != None):
                    Printer.print_w("Overwrite: duplication of feature data for %s " %(inst_name))
            else:
                Printer.print_w("No features found: %s" %(inst_name))

        Printer.print_verbose(">>>Feature Data:<<<")
        Printer.print_verbose(str(self.__feature_dic))
        Printer.print_c("Reading Features was successful!")
        try:
            filepointer.close()
        except IOError:
            if (self.__PRINTERR):
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
        
        return True
    
    def __check_dublicates(self, instance_dic):
        '''
            Check whether instances with same features are in the given set
            (gives only warnings)
            Args:
                feature_dic : instance name -> instance Object
            Returns
                list of dublicates tuples
        '''
        Printer.print_nearly_verbose("Searching for duplicates ...")
        range_max = len(instance_dic)
        instances = instance_dic.keys()
        dublicates = {}
        self.__n_dubs = 0
        for index1 in range(0,range_max):
            inst1 = instances[index1]
            features1 = instance_dic[inst1].get_features()
            if not features1:
                continue
            for index2 in range(index1+1, range_max):
                inst2 = instances[index2]
                features2 = instance_dic[inst2].get_features()
                if not features2:
                    continue
                if inst1 != inst2:
                    dist = 0
                    for f1,f2 in zip(features1,features2):
                        dist += math.fabs(f1-f2)
                    if dist < 0.001:
                        Printer.print_verbose("Instances are likely the same (feature distance %f):" %(dist))
                        Printer.print_verbose(inst1)
                        Printer.print_verbose(",".join(map(str,features1)))
                        Printer.print_verbose(inst2)
                        Printer.print_verbose(",".join(map(str,features2)))
                        self.__add_dub_in_dic(inst1, inst2, dublicates)
                        self.__n_dubs += 1
        Printer.print_nearly_verbose(json.dumps(dublicates, indent=2))
        return dublicates
    
    def __add_dub_in_dic(self,inst1, inst2, dub_dic):
        '''
            adds inst1 and inst2 in dub_dic
        '''
        if inst1 in dub_dic:
            dub_dic[inst1].append(inst2)
            return
        if inst2 in dub_dic:
            dub_dic[inst2].append(inst1)
            return
        for inst,list_ in dub_dic.items():
            if inst1 in list_ and inst2 in list_:
                return
            if inst1 in list_:
                dub_dic[inst].append(inst2)
                return
            if inst2 in list_:
                dub_dic[inst].append(inst1)
                return
        dub_dic[inst1] = [inst2]       
            
    def __join_dublicates_data(self, dublicates):
        '''
            join the runtime of same instances, remove features and status for second inst in tuple
            Args:
                dublicates: list of tuples with dublicated instances
        '''
        if dublicates:
            Printer.print_w("I will merge all duplicated data! (use --verbose 1 to see the duplicates)")
        for inst1,dubs in dublicates.items():
            for inst2 in dubs:
                new_runtime = []
                for t1,t2 in zip(self.__instance_dic[inst1].get_runtimes(), self.__instance_dic[inst2].get_runtimes()):
                    if t1 < 0.0 and t2 < 0.0: # unknown runtimes are encoded as -512 ( < 0.0)
                        new_runtime.append(t1)
                        continue
                    if t1 > 0.0 and t2 > 0.0:
                        new_runtime.append(min(t1,t2))
                        continue
                    if t1 < 0.0 or t2 < 0.0:
                        new_runtime.append(max(t1,t2))
                self.__instance_dic.pop(inst2)
                self.__instance_dic[inst1]._set_runtimes(new_runtime)
    
    def __parse_runtimes(self, runtime_file, filter_tos):
        '''
            parse runtime csv file runtime_file and filters timeouts if wished
            runtimes are internally saved in the dictionary ___runtime_data_dic
            Args:
                runtime_file:  csv file with delimeter __DELIMETER (string) 
                filter_tos:    filter timeouts (bool) 
            Returns:
                true if succesfull
                false otherwise
            Raises:
                IOError:   
        '''
        try:
            filepointer = open(runtime_file,"r")
        except IOError:
            if (self.__PRINTERR):
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
            return False
        if runtime_file.endswith(".csv"):
            solver_line = filepointer.readline().replace("\n","")
            self.__solver_list = solver_line.split(",")[1:]
        else:
            self.__solver_list = []
        
        for line in filepointer:
            line = line.replace("\n","")
            if line == "": # skip empty lines
                continue
            if line.startswith("@"): # arff format: meta information 
                line_upper = line.upper()
                if line_upper.startswith("@ATTRIBUTE INSTANCE") or line_upper.startswith("@ATTRIBUTE ID"):
                    continue
                if line_upper.startswith("@ATTRIBUTE"):
                    line = line.strip(" ")
                    line = line.split(" ")
                    solver_name = line[1]
                    #line.replace("@ATTRIBUTE ","").replace(" NUMERIC","")
                    self.__solver_list.append(solver_name)
                    continue
            
            line = line.split(self.__DELIMETER)
            values = []
            instance_name = line.pop(0)
            complete = True
            for value in line:
                if value == "?":
                    value = self.__cutoff
                try:
                    value = float(value)
                except ValueError:
                    #Printer.print_w("Value Error in Runtime Parsing!")
                    continue
                if (value >= self.__cutoff and filter_tos == False):
                    value = self.__PARX * self.__cutoff
                value = max(value,self.__MINIMAL_RUNTIME)
                values.append(value)
                if (value >= self.__cutoff):
                    complete = False
            if (values != [] and (filter_tos == False or complete == True)): # filter empty lines
                entry = self.__runtime_dic.get(instance_name)
                self.__runtime_dic[instance_name] = (values)
                if (entry != None):
                    Printer.print_w("Warning Overwrite: duplication of runtime data for "+str(instance_name)+"")
        try:
            filepointer.close()
        except IOError:
            if (self.__PRINTERR):
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()

        Printer.print_c("Found Solvers: %s" % (", ".join(self.__solver_list)))
        if not self.__solver_list: #empty list
            Printer.print_e("No solver names have been found. Input format wrong!")
        Printer.print_verbose(">>>Runtime Data:<<<")
        Printer.print_verbose(str(self.__runtime_dic))
        Printer.print_verbose("Size: "+str(len(self.__runtime_dic)))
        Printer.print_c("Reading Runtimes was successful!")
        return True
    
    def __parse_sat_unsat(self, sat_unsat):
        '''
            parse sat(1) and unsat(-1) status of the satunsat file
            Args:
                satunsat: csv file with delimeter __DELIMETER 
            Returns:
                true if succesfull
                false otherwise
            Raises:
                IOError:   
        '''
        # if no sat_unsat file is given, assume all instances are from the same class
        if not sat_unsat:
            for instance_name in self.__runtime_dic.keys():
                self.__status_dic[instance_name] = 1
            return True
        
        try:
            filepointer = open(sat_unsat,"r")
        except IOError:
            if (self.__PRINTERR):
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
            return False
        for line in filepointer:
            if line == "\n":
                continue
            if line.startswith("@") or line.startswith("%"): # arff meta information
                continue
            line = line.split(self.__DELIMETER)
            instance_name = line[0]
            status_string = line[1].replace("\n", "") #remove line break
            try:
                status_int = int(self.__SAT_UNSAT_TRANSLATION.get(status_string,status_string)) # translate if nec.
                self.__status_dic[instance_name] = status_int
            except ValueError:
                Printer.print_w("Could not read line status file:\n%s" %(self.__DELIMETER.join(line)))
            
        Printer.print_verbose(">>>Status Data:<<<")
        Printer.print_verbose(str(self.__status_dic))
        Printer.print_c("Reading Status was successful!")
        try:
            filepointer.close()
        except IOError:
            if (self.__PRINTERR):
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
        return True
    
    def __parse_feature_time(self, feature_time_file, const_ftime=3):
        '''
            parse runtime of feature generator
            Args:
                feature_time_file: csv file with delimeter __DELIMETER
                const_ftime: if no feature file is given, use constant runtime assumption
            Returns:
                true if succesfull
                false otherwise
            Raises:
                IOError
        '''
        
        if not feature_time_file:
            for instance_name in self.__runtime_dic.keys():
                self.__ftime_dic[instance_name] = const_ftime
            return True
        
        try:
            filepointer = open(feature_time_file,"r")
        except IOError:
            if (self.__PRINTERR):
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
            return False
        
        for line in filepointer:
            if line == "\n":
                continue
            if line.startswith("@") or line.startswith("%"): # arff meta information
                continue
            line = line.split(self.__DELIMETER)
            instance_name = line[0]
            ftime_string = line[1].replace("\n", "") #remove line break
            try:
                ftime = float(ftime_string) # translate if nec.
                self.__ftime_dic[instance_name] = ftime
            except ValueError:
                Printer.print_w("Could not read line feature time file:\n%s" %(self.__DELIMETER.join(line)))
            
        Printer.print_verbose(">>>Feature Time Data:<<<")
        Printer.print_verbose(str(self.__ftime_dic))
        Printer.print_c("Reading Feature Time was successful!")
        try:
            filepointer.close()
        except IOError:
            if (self.__PRINTERR):
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
        return True
    
    def __join_times_features_status(self, const_ftime):
        '''
            compares the available data of runtimes, features and status 
            and computes a intersection.
            ATTENTION: this methods updates the internal values for __runtime_data_dic and __feature_data_dic
        '''
        n_run_data = len(self.__runtime_dic)
        
        runtime_invalid_features = {}
        missing_ftime = 0
        
        for inst, times in self.__runtime_dic.items():
            if (sum(times) == len(times)*(self.__cutoff * self.__PARX)): # filter instances with only timeouts
                self.__unsolveable += 1
                continue
            features = self.__feature_dic.get(inst)
            status = self.__status_dic.get(inst)
            ftime = self.__ftime_dic.get(inst)
            
            # check valid data, features and status is resetted for invalid data
            if (features is None):
                runtime_invalid_features[inst] = times
                Printer.print_w("Warning: there are runtime data but no features available for "+str(inst)+"")
                self.__presolved += 1 
                features = None
                status = None
            if (features and len(features) < self.__number_feats):
                # warning printed beforehand
                runtime_invalid_features[inst] = times 
                self.__too_few_f += 1
                features = None
                status = None
            if (features and status is None):
                Printer.print_w("Warning: there are runtime data but no status available for "+str(inst)+"")
                self.__unsolveable += 1 # this case should never occur!
                features = None
                status = None
            if (features and sum(features) == 0.0): # all features were <= 0.0 -> interpreted as presolved
                runtime_invalid_features[inst] = times 
                self.__presolved += 1 
                features = None
                status = None
            if (features and (math.isnan(sum(features)) or math.isinf(sum(features)))): #@UndefinedVariable
                runtime_invalid_features[inst] = times
                Printer.print_w("Warning: feature data include NAN or INF in "+str(inst)+"")
                self.__unknown += 1
                features = None
                status = None
            if ftime is None:
                ftime = const_ftime
                Printer.print_w("Warning: there are runtime data but no feature runtime data available for "+str(inst)+"")
                missing_ftime += 1
                            
            # else everything is ok
            #new_feature_dic[inst] = features
            #new_runtime_data_dic[inst] = times
            #new_status_dic[inst] = status
            self.__instance_dic[inst] = Instance(inst, times, features, [], ftime, status)
            self.__available += 1
            
        self.__invalid_f_runtime = runtime_invalid_features
        Printer.print_c(">>> Runtime Data : "+str(n_run_data))
        Printer.print_c(">>> Accepted Data : "+str(self.__available))
        Printer.print_c(">>> Unsolvable : "+str(self.__unsolveable))
        Printer.print_c(">>> No features : "+str(self.__presolved))
        Printer.print_c(">>> Too few features: "+str(self.__too_few_f))
        Printer.print_c(">>> Unknown features (NAN, INF): "+str(self.__unknown))
        
        if self.__available == 0:
            Printer.print_e("No available instance data found (runtime x features x status)!")
    
    def __parse_configurations(self, config_file):
        '''
            parse a json file with solver name -> configuration
        
        '''
        if config_file:
            fp = open(config_file,"r")
            config_dic = json.load(fp)
        else:
            config_dic = {}
            for algorithm in self.__solver_list:
                config_dic[algorithm] = ""
        return config_dic
    
    def get_data(self, runtime_file, feature_file, satunsat_file, config_file, ftime_file=None, const_ftime=3.0):
        '''
            return internal runtime, feature and status data
            If parsing of one file is not sucessful, terminate complete application
            Args:
                runtime_file: csv file with delimeter __DELIMETER (string); first coloumn instance names
                feature file: csv file with delimeter __DELIMETER (string); first coloumn instance names
                satunsat_file csv file with delimeter __DELIMETER (string); first coloumn instance names
            Returns:
                __runtime_data_dic : dictionary from instance name to runtime vector
                __feature_data_dic : dictionary from instance name to feature vector
                __status_data_dic :  dictionary from instance name to sat/unsat status
                __solver_list : headers of runtime csv file
        '''
        self.__reset_attributes()
        
        if not self.__parse_runtimes(runtime_file, False):
            sys.exit(-1)
        if not self.__parse_features(feature_file):
            sys.exit(-1)
        if not self.__parse_sat_unsat(satunsat_file):
            sys.exit(-1)
        if not self.__parse_feature_time(ftime_file, const_ftime):
            sys.exit(-1)
            
        self.__join_times_features_status(const_ftime) # fill self.__instance_dic

        if self.__filter_duplicates:
            dublicates = self.__check_dublicates(self.__instance_dic)
            self.__join_dublicates_data(dublicates) # changes self.__instance_dic
        if self.__n_dubs > 0:
            Printer.print_w("Found  duplicates : "+str(self.__n_dubs))

        if self.__CORRELATION_TEST:
            self.__cor_test()
        
        config_dic = self.__parse_configurations(config_file)

        return self.__instance_dic, self.__solver_list, config_dic
    
    def __cor_test(self):
        '''
            spearman correlation test between runtimes of solvers
        '''
        from scipy.stats import spearmanr
        
        n_solver = len(self.__solver_list)
        
        runtime_matrix = list([] for _ in self.__solver_list)
        for inst_ in self.__instance_dic.values():
            times = inst_.get_runtimes()
            runtime_matrix = map(lambda (x,y): x.append(y) or x, zip(runtime_matrix,times))
        
        correlation_matrix = dict((solver,{}) for solver in self.__solver_list)
        for index1,solver1 in zip(range(0,n_solver),self.__solver_list):
            for index2,solver2 in zip(range(0,n_solver),self.__solver_list):
                cor_coefficient,p_value = spearmanr(runtime_matrix[index1],runtime_matrix[index2])
                correlation_matrix[solver1][solver2] = cor_coefficient

        print(correlation_matrix)
        import numpy as np
        correlation_matrix_np = np.array(list([0.0]*n_solver for _ in self.__solver_list))
        print(correlation_matrix_np.shape)
        sorted_solvers = sorted(self.__solver_list)
        for index1,solver1 in zip(range(0,n_solver),sorted_solvers):
            for index2,solver2 in zip(range(0,n_solver),sorted_solvers):
                correlation_matrix_np[index1][index2] = correlation_matrix[solver1][solver2]
        print(correlation_matrix_np)
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        heatmap = ax.pcolor(correlation_matrix_np)
        
        plt.colorbar(heatmap)
        plt.show()
                
    def get_stats(self):
        ''' return some stats of reading 
            Returns:
                self.__available : number of valid data
                self.__presolved: number of presolved instances
                self.__unsolveable: number of unsolvable instances (only timeouts)
                self.__too_few_f: invalid instances with unknown reason (check them!) 
        '''
        return self.__available, self.__presolved, self.__unsolveable, self.__too_few_f
