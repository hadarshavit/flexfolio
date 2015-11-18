import os
import numpy
from numpy import Inf
import tabulate
import copy

import re

num_regex = "[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?"
regex_size = re.compile(r'%% Average schedule size = (?P<size>%s)$' %(num_regex))
regex_pos = re.compile(r"^%% Average schedule position of first successful solver= (?P<pos>%s)$" %(num_regex))

scen_system_size = {}
scen_system_pos = {}
for f in os.listdir("."):
    if f.endswith(".log"):
        scen, system = f.replace(".log","").split("_")
        scen_system_size[scen] = scen_system_size.get(scen,{})
        scen_system_pos[scen] = scen_system_pos.get(scen,{})
        with open(f) as fp:
            for line in fp:
                m = regex_size.match(line)
                if m:
                    size = m.group("size")
                    scen_system_size[scen][system] = size
                m = regex_pos.match(line)
                if m:
                    pos = m.group("pos")
                    scen_system_pos[scen][system] = pos
                    
#print(scen_system_pos)
#print(scen_system_size)

tabulate.LATEX_ESCAPE_RULES = {}
scens = scen_system_size.keys()
systems = scen_system_size[scens[0]].keys()
systems.remove("sbs")
systems.remove("default")

header = [""]
header.extend(systems)

METRIC = "PAR10"
tab = []
for scen in sorted(scens):
    row = ["$%s$ ($%s$)" %(scen_system_size[scen].get(sys_,-1), scen_system_pos[scen].get(sys_,-1)) for sys_ in systems]
    row.insert(0, scen)
    tab.append(row)
    
print(tabulate.tabulate(tab, header, tablefmt="latex"))
