import os
import numpy
from numpy import Inf
import tabulate
import copy
import argparse
from stat_tests.PermutationTester import PermutationTester
from coseal_reader import CosealReader
import logging
import sys

ASLIB_PATH = "/home/lindauer/projects/aslib/aslib_data-aslib-v1.0.1/"

oracel_par10s = { "PREMARSHALLING-ASTAR-2013" : 227.6,
                 "ASP-POTASSCO" : 21.3,
                 "MAXSAT12-PMS" : 40.7,
                 "QBF-2011" : 95.9,
                 "CSP-2010" : 107.7,
                 "SAT12-ALL": 93.7,
                 "SAT12-INDU": 88.1,
                 "SAT12-HAND": 113.2,
                 "SAT12-RAND": 46.9,
                 "SAT11-INDU": 419.9,
                 "SAT11-HAND": 478.3,
                 "SAT11-RAND": 227.3,
                 "PROTEUS-2014": 26.3
          }

cutoffs = { "PREMARSHALLING-ASTAR-2013" : 3600,
                 "ASP-POTASSCO" : 600,
                 "MAXSAT12-PMS" : 2100,
                 "QBF-2011" : 3600,
                 "CSP-2010" : 5000,
                 "SAT12-ALL": 1200,
                 "SAT12-INDU": 1200,
                 "SAT12-HAND": 1200,
                 "SAT12-RAND": 1200,
                 "SAT11-INDU": 5000,
                 "SAT11-HAND": 5000,
                 "SAT11-RAND": 5000,
                 "PROTEUS-2014" : 3600 
          }

unsolvable = { "PREMARSHALLING-ASTAR-2013" : 0,
                 "ASP-POTASSCO" : 82,
                 "MAXSAT12-PMS" : 129,
                 "QBF-2011" : 314,
                 "CSP-2010" : 253,
                 "SAT12-ALL": 20,
                 "SAT12-INDU": 209,
                 "SAT12-HAND": 229,
                 "SAT12-RAND": 322,
                 "SAT11-INDU": 47,
                 "SAT11-HAND": 77,
                 "SAT11-RAND": 108,
                 "PROTEUS-2014" : 428 
          }


def get_score(perf_dict, par_factor=1, cutoff=Inf):
    scores = []
    for v in perf_dict.values():
        if v >= cutoff:
            if par_factor is Inf:
                scores.append(1)
            else:
                scores.append(cutoff*par_factor)
        else:
            if par_factor is Inf:
                scores.append(0)
            else:
                scores.append(v)
    
    if par_factor is Inf:
        return numpy.sum(scores)
    else:
        return numpy.mean(scores)
    
def gap_metric(score, sbs, oracle):
    return (sbs - score) / (sbs - oracle)

def get_unsolved_instances(inst_dict):
    '''
        checks the runstatus of each instance and returns the number of instances that was not solved by any solver
    '''
    unsolved_instances = []
    for name, inst_ in inst_dict.iteritems():
        if "ok" not in inst_._status.values():
            unsolved_instances.append(name)
            
    logging.debug("Unsolved Instances: %d" %(len(unsolved_instances)))
    return unsolved_instances

parser = argparse.ArgumentParser(usage="python evaluate.py")
args_ = parser.parse_args()
reader = CosealReader()
args_.feat_time = -1
args_.feature_steps = None

# read coseal data
scen_unsolved = {}
for scen in os.listdir(ASLIB_PATH):
    if not os.path.isdir(os.path.join(ASLIB_PATH, scen)):
        continue
    reader = CosealReader()
    instance_dict, metainfo, algo_dict = reader.parse_coseal(coseal_dir = os.path.join(ASLIB_PATH, scen), args_=args_)        
    scen_unsolved[scen] = get_unsolved_instances(instance_dict)

scen_system_inst = {}
for f in os.listdir("."):
    if f.endswith(".csv"):
        scen, system = f.replace(".csv","").split("_")
        inst_perf = {}
        with open(f) as fp:
            fp.readline()
            for line in fp:
                inst, perf = line.replace("\n","").split(",")
                if not inst in scen_unsolved[scen]:
                    try:
                        inst_perf[inst] = float(perf)
                    except ValueError:
                        sys.path.stderr("[WARNING]: cannot read line: %s" %(line))
                
                
        scen_system_inst[scen] = scen_system_inst.get(scen,{})
        scen_system_inst[scen][system] = {"insts": inst_perf}
        
        n = len(inst_perf)
        
        scen_system_inst[scen][system]["PAR10"] = get_score(inst_perf, par_factor=10, cutoff=cutoffs[scen])
        scen_system_inst[scen][system]["PAR1"] = get_score(inst_perf, par_factor=1, cutoff=cutoffs[scen])
        scen_system_inst[scen][system]["TOs"] = get_score(inst_perf, par_factor=Inf, cutoff=cutoffs[scen])
        
        #scen_system_inst[scen][system]["PAR10"] = (scen_system_inst[scen][system]["PAR10"] * n - (cutoffs[scen] * 10 * unsolvable[scen])) / (n - unsolvable[scen])
        #scen_system_inst[scen][system]["PAR1"] = (scen_system_inst[scen][system]["PAR10"] * n - (cutoffs[scen] * 1 * unsolvable[scen])) / (n - unsolvable[scen])
        #scen_system_inst[scen][system]["TOs"] = (scen_system_inst[scen][system]["PAR10"] * n - (unsolvable[scen])) / (n - unsolvable[scen])

tab = []

scens = scen_system_inst.keys()
systems = scen_system_inst[scens[0]].keys()

header = [""]
header.extend(systems)

METRIC = "PAR10"

for scen in sorted(scens):
    row = [scen_system_inst[scen][sys_][METRIC] for sys_ in systems]
    row = map(lambda x: round(x, 1), row)
    row.insert(0, scen)
    tab.append(row)
    
print(tabulate.tabulate(tab, header, tablefmt="latex"))
    
tab = []
score_rows = []
pt = PermutationTester()
#print(tabulate.LATEX_ESCAPE_RULES)
tabulate.LATEX_ESCAPE_RULES = {}
systems.remove("sbs")
systems = ["default", "autofolio", "aspeed", "sunny-ori", "tsunny", "isa"]
header = [""]
header.extend(systems)
best_systems = dict((sys_, 0) for sys_ in systems)
for scen in sorted(scens):
    row = [gap_metric(scen_system_inst[scen][sys_][METRIC], scen_system_inst[scen]["sbs"][METRIC], oracel_par10s[scen]) for sys_ in systems]
    row = map(lambda x: round(x, 2), row)
    
    score_rows.append(copy.copy(row))
    
    max_v = max(row)
    max_idx = row.index(max_v)
    best_system = systems[max_idx]
    
    best_vec = scen_system_inst[scen][best_system]["insts"]
    for idx, system in enumerate(systems):
        challenger_vec = scen_system_inst[scen][system]["insts"]
        rejected,_,pValue = pt.doTest(best_vec, challenger_vec, permutations=1000)
        if max_idx == idx:
            row[idx] = "$\mathbf{%.2f}^{*}$" %(row[idx])
            best_systems[system] += 1
        elif not rejected:
            row[idx] = "$%.2f^{*}$" %(row[idx])
            best_systems[system] += 1
        else:
            row[idx] = "$%.2f$" %(row[idx])
    row.insert(0, scen)
    tab.append(row)
    
    
row = list(numpy.mean(score_rows, axis=0))
row = map(lambda x: round(x, 2), row)
row.insert(0,"Average")
tab.append(row)

row = [best_systems[sys_] for sys_ in systems]
row.insert(0,"Equal to Best")
tab.append(row)
    
print(tabulate.tabulate(tab, header, tablefmt="latex_booktabs"))        
        
