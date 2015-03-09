'''
Created on July 2, 2013

@author: CVSH
'''

import copy
import os
import threading
import re

from subprocess import Popen, PIPE
from tempfile import NamedTemporaryFile

from misc.printer import Printer

from misc.printer import Printer

class TreefolioInstanceTranslator(object):
    
    def __init__(self, command):
        self.command = command
        self._maxTime = 200
        self._timeout = 900
    
    def translate(self, instance):
        optfeatures = []
        
        stdout_file = NamedTemporaryFile(prefix="Output",dir=".",delete=True)
        trans_file = NamedTemporaryFile(prefix="Translation",dir=".",delete=True)
        ft_file = None
        
        cmd = self.command
        
        feat = False
        outp = False
        
        if "<instance>" in self.command:
            cmd = cmd.replace("<instance>",instance.name)
        if "<features>" in self.command:
            feat = True
            ft_file = NamedTemporaryFile(prefix="OUT-FEATURES",dir=".",delete=True)
            cmd = cmd.replace("<features>",ft_file.name)
        if "<output>" in self.command:
            outp = True
            cmd = cmd.replace("<output>",trans_file.name)
        if "<tempdir>" in self.command:
            cmd = cmd.replace("<tempdir>",".") # TODO: tempdir
            
        Printer.print_c("Executing: "+cmd)
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
            for line in ft_file.readlines(): # Evil Number Extraction ^o.o^
                Printer.print_verbose(line)
                line = re.sub("[^ ,;:]*[a-zA-Z_][^ ,;:]*","",line) # Remove all Strings no single numbers)
                numbers = [match.group(0) for match in re.finditer("-?\d+(\.\d+)?",line)]
                for i in numbers:
                    optfeatures.append(float(i))
#                 if line.startswith(self.__pattern):
#                     line = line.lstrip(self.__pattern).strip(": ")
#                     optfeatures = map(float,line.split(","))
#                     Printer.print_c("Features: "+",".join(map(str,optfeatures)))
#                     break
            ft_file.close()
        
        if outp:
            trans_file.seek(0)
            instance.close()
            instance = trans_file
        else:
            stdout_file.seek(0)
            instance.close()
            instance = stdout_file
        
        try:
            t.cancel()
        except:
            pass
        
        return instance, optfeatures
    
    
    