'''
Created on July 2, 2013

@author: CVSH
'''

import copy
import os
import threading

from subprocess import Popen, PIPE
from tempfile import NamedTemporaryFile

from misc.printer import Printer

class TreefolioFeatureExtractor(object):
    
    def __init__(self, command):
        self.command = command
    
    def extract(self, instance):
        
        features = []
        
        stdout_file = NamedTemporaryFile(prefix="Output",dir=".",delete=True)
        ft_file = None
        
        cmd = self.command
        
        feat = False
        
        if "<instance>" in self.command:
            cmd = cmd.replace("<instance>",instance.name)
        if "<features>" in self.command:
            feat = True
            ft_file = NamedTemporaryFile(prefix="OUT-FEATURES",dir=".",delete=True)
            cmd = cmd.replace("<features>",ft_file.name)
        if "<tempdir>" in self.command:
            cmd = cmd.replace("<tempdir>",".") # TODO: tempdir
            
        Printer.print_c(cmd)
        cmd = cmd.split(" ")
        
        try:
            popen_ = Popen(cmd, stdin=PIPE, stdout = stdout_file)
            t = threading.Timer(self._maxTime, self._timeout, [popen_])
            t.start()
            instance.seek(0)
            popen_.communicate(input=instance.read())
        except OSError:
            Printer.print_e("Feature extractor was not found.")
        
        if feat: # TODO: This is for claspre ... write an general method!
            ft_file.seek(0)            
            for line in ft_file.readlines():
                Printer.print_verbose(line)
                if line.startswith(self.__pattern):
                    line = line.lstrip(self.__pattern).strip(": ")
                    features = map(float,line.split(","))
                    Printer.print_c("Features: "+",".join(map(str,features)))
                    break
            ft_file.close()
        else:
            stdout_file.seek(0)            
            for line in stdout_file.readlines():
                Printer.print_verbose(line)
                if line.startswith(self.__pattern):
                    line = line.lstrip(self.__pattern).strip(": ")
                    features = map(float,line.split(","))
                    Printer.print_c("Features: "+",".join(map(str,features)))
                    break
            stdout_file.close()
        
        try:
            t.cancel()
        except:
            pass
        
        return features
    
    
    
    