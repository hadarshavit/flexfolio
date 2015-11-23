#!/bin/python

import os

preconfs = ["", "aspeed", "sbs", "isa", "tsunny", "sunny"]

scenarios = "/home/lindauer/projects/aslib_data-aslib-v1.0.1"

flex_path = "/home/lindauer/git/flexfolio/src/flexfolio_train.py"

for scen in os.listdir(scenarios):
    
    scen_path = os.path.join(scenarios, scen)
    
    for conf in preconfs:
        
        if conf:
           conf_string = "--preconf %s" %(conf)
        else:
            conf_string = "--aspeed-opt"
            conf = "default"
            
        print("/home/lindauer/git/flexfolio/virtualenv/bin/python %s --aslib %s --model . %s --print-times %s_%s.csv  1> %s_%s.log 2>&1" %(flex_path, scen_path, conf_string, scen, conf, scen, conf))
    