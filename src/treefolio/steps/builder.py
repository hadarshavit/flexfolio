'''
Created on July 2, 2013

@author: CVSH
'''
from execute_step import Executionstep
from step import Decisionstep
from misc.printer import Printer
import json

class StepBuilder(object):
    '''
        This Class builds all steps as a tree out of a json configuration file.
        
        Structure = { 
            "steptype" = "decision"
            "next" = [Structure, ...] (can be empty)
            "names" = [String, ...] (should be same size as "next")
            "default" = Number (should be smaller than len("next"))
            "modelfile" = String (path to model-configuration file)
        }
        
        Structure = { 
            "steptype" = "execution" (everything different than "decision" is counted as execution) 
            "configfile" = String (path to execution-configuration file)
        }
        
    '''
    
    def __init__(self):
        
        # Class variables
        self.first = None
        self.ranges = []
        
    def read_and_build_from_data(self, args_, configfile):
        '''
            Build Tree from Configfile.
            This is used for Training, not for execution
            Parameter:
                args_: Parsed Arguments
                configfile: Path to configuration file
        '''
        self.first = None
        filepointer = open(configfile)
        steps = json.load(filepointer)
        filepointer.close()
        my_steps = []
        steps_by_name = {}
        counter = 0
        
        # Building non-Execution Steps
        if steps.has_key("Steps"):
            for st in steps["Steps"]:
                counter += 1
                this_step = Decisionstep(args_.treefolio_oracle, args_.treefolio_crossfold)
                this_step.name = st["Name"]
                this_step.command = st["Command"]
                this_step.feature_extractor_command = st["FeatureCommand"]
                this_step.activate_features = st["RelatedFeatures"]
                this_step.depends_on = st["DependsOn"]
                if st.has_key("Approach"):
                    this_step.approach = st["Approach"] #args_.approach # TODO: maybo not 1 approach for all...
                else:
                    this_step.approach = args_.approach
                torem = st["RemoveFeatures"]
                if torem and isinstance(torem,(list,tuple)):
                    my_list = []
                    for element in torem:
                        if isinstance(element,(list,tuple)):
                            if len(element) == 2:
                                my_list.extend(range(element[0],element[1]+1))
                        else:
                            my_list.append(element)
                    this_step.remove_features = sorted(list(my_list), reverse=True)
                    last = -1
                    for ind in this_step.remove_features:
                        if ind >= args_.n_feats or ind == last or ind < 0:
                            this_step.remove_features.remove(ind)
                
                if this_step.activate_features and isinstance(this_step.activate_features,(list,tuple)) and len(this_step.activate_features)==2:
                    this_step.activate_features[1] += 1
                self.ranges.append(this_step.activate_features)
                this_step.step_id = counter
                steps_by_name[this_step.name] = this_step
                if len(this_step.depends_on) <= 0:
                    self.first = this_step
                else:
                    my_steps.append(this_step)
            if not self.first:
                raise Exception("All Steps are dependent!")
            
            last_decided = [self.first]
            has_node = [self.first.name]
            
            # Construct Tree, this should be finished after one Iteration,
            # otherwise reconstruct your config
            # Dependencys should be simple... don't do any funny stuff
            
            ##while len(has_node) < len(steps): #repeat until all nodes are used
            if len(has_node) < len(steps):
                
                while len(last_decided)>0:
                    next_up = []
                    for las in last_decided:
                        for st in my_steps:
                            if las.name in st.depends_on:
                                if len(st.depends_on) <= len(las.fullfilled_dependencys)+1:
                                    dep = st.depends_on[:]
                                    dep.remove(las.name)
                                    for a in las.fullfilled_dependencys:
                                        if a in dep:
                                            dep.remove(a)
                                    if not dep:
                                        #print "%s: Dependencys %s covered by %s with %s"%(st.name,str(st.depends_on),las.name,str(las.fullfilled_dependencys))
                                        st.depends_on = []
                                        st.step_depth = las.step_depth +1
                                        st.fullfilled_dependencys.append(las.name)
                                        st.fullfilled_dependencys.extend(las.fullfilled_dependencys)
                                        las.nextstep.append(st)
                                        las.decision_name.append(st.name)
                                        has_node.append(st.name)
                                        next_up.append(st)
                                    else:
                                        print "%s: Missing in %s: %s"%(st.name,las.name,str(dep))
                                #else:
                                #    print "%s: Depends on %s, got %s"%(st.name,str(st.depends_on),str(las.fullfilled_dependencys))
                                
                                #if len(st.depends_on) <= 0:
                                #    st.step_depth = las.step_depth +1
                                #    las.nextstep.append(st)
                                #    las.decision_name.append(st.name)
                                #    next_up.append(st)
                    last_decided = next_up
                
                # This commented Part is for repeated looping dependencys
                '''
                for st in my_steps:
                    if len(st.depends_on) > 0:
                        print "Warning: Step %s still dependent on %s"%(st.name,str(st.depends_on))
                        for b in st.depends_on:
                            if len(steps_by_name[b].depends_on) <= 0:
                                
                                for las in my_steps:
                                    if len(las.depends_on) <= 0:
                                        dep = st.depends_on[:]
                                        dep.remove(las.name)
                                        for a in las.fullfilled_dependencys:
                                            dep.remove(a)
                                        if not dep:
                                                print "%s: Dependencys %s covered by %s with %s"%(st.name,str(st.depends_on),las.name,str(las.fullfilled_dependencys))
                                                st.depends_on = []
                                                st.step_depth = las.step_depth +1
                                                st.fullfilled_dependencys.append(las.name)
                                                st.fullfilled_dependencys.extend(las.fullfilled_dependencys)
                                                las.nextstep.append(st)
                                                las.decision_name.append(st.name)
                                                has_node.append(st.name)
                                                next_up.append(st)
                                        else: # copy some step!
                                            current = las
                                            for b in dep:
                                                #add step to current
                                                counter += 1
                                                this_step = Decisionstep()
                                                this_step.name = st["Name"]
                                                this_step.command = st["Command"]
                                                this_step.activate_features = st["RelatedFeatures"]
                                                this_step.depends_on = st["DependsOn"]
                                                this_step.step_id = counter
                                                #steps_by_name[this_step.name] = this_step
                                                my_steps.append(this_step)
                            else:
                                print "%s: Dependencys %s are not coverable."%(st.name,str(st.depends_on))
                '''
                    
                # Error if Dependencies are not Resolved
                for st in my_steps:
                    if len(st.depends_on) > 0:
                        raise Exception("Error: Step %s still dependent on %s! Edit config file and add additional dependencies!"%(st.name,str(st.depends_on)))
        else:
            raise Exception("No Attribute Steps!")
        
        # Build Execution Steps!
        if steps.has_key("Configurations"):
            for conf in steps["Configurations"]:
                exs = Executionstep()
                exs.execcommand = conf["Command"]
                exs.name = conf["Name"]
                highest = None
                val = -1
                for dependency in conf["DependsOn"]:
                    if steps_by_name.has_key(dependency):
                        if (steps_by_name[dependency].step_depth>val):
                            val = steps_by_name[dependency].step_depth
                            highest = steps_by_name[dependency]
                    else:
                        raise Exception("Dependency %s not Found!"%dependency)
                if (highest):
                    highest.nextstep.append(exs)
                    highest.decision_name.append(exs.name)
                else:
                    raise Exception("Best dependency not Found!")
        else:
            raise Exception("No Attribute Configurations!")
        
        for st in my_steps:
            if not st.nextstep:
                Printer.print_e("Step %s has no next steps! Cleaning up!"%st.name,100)
                my_steps.remove(st)
                for ste in my_steps:
                    if st in ste.nextstep:
                        ind = ste.nextstep.index(st)
                        ste.nextstep.remove(st)
                        del ste.decision_name[ind]
                del st
        
        #Done building.
    
    def test_print(self):
        '''
            Print Tree.
            Tree begins with self.first,
            needs to be build by read_and_build_from_data() OR build()
        '''
        if (self.first):
            steps = [self.first]
            Printer.print_c("Start Decision = %s"%self.first.name)
            while len(steps)>0:
                step = steps.pop(0)
                if hasattr(step,"nextstep"):
                    Printer.print_c("Decision %s"%step.name)
                    for s in step.nextstep:
                        steps.append(s)
                        if hasattr(s,"nextstep"):
                            Printer.print_c("-> Decision %s"%s.name)
                        else:
                            Printer.print_c("-> Execution %s"%s.name)
        else:
            raise Exception("No First!")
    
    def save_tree(self,args_,filename):
        '''
            Save the Tree to a file for main.
            (Should be done after training)
            Parameter:
                args_: Parsed Parameter
                filename: File to write
        '''
        if self.first:
            jstr = {
                        "args" : str(args_),
                        "tree" : self.tree_to_json(self.first)
                    }
            print jstr
            fi = open(filename, 'w')
            json.dump(jstr,fi)
            fi.close()
            Printer.print_c("Config written to file "+filename)
        else:
            Printer.print_w("Could not save Tree Structure!")
    
    def tree_to_json(self, current):
        '''
            return structure as String.
            Parameter:
                current: Tree point
        '''
        if (current):
            if isinstance(current, Decisionstep):
                jstr = {
                    "name" : current.name,
                    "steptype" : "decision",
                    "oracle" : current.oracle,
                    "folds" : current.crossfold,
                    "selection" : current.selection,
                    "default" : current.default_next_number,
                    "feature_count" : current.feature_count,
                    "names" : current.decision_name,
                    "modelpath" : current.modelpath,
                    "activate_features" : current.activate_features,
                    "approach" : current.approach,
                    #"feature_extractor" : current.feature_extractor,
                    #"translator" : current.translator,
                    "command" :current.command,
                    "feature_extractor_command" : current.feature_extractor_command,
                    #"translator_command" : current.translator_command,
                    "next" : []
                }
                for nextst in current.nextstep:
                    jstr["next"].append(self.tree_to_json(nextst))
            else:
                jstr = {
                    "name" : current.name,
                    "steptype" : "execution",
                    "configuration" : current.configuration,
                    "configuration_file" : current.configuration_file,
                    "execcommand": current.execcommand
                }
            return jstr
        else:
            return {}
    
    def build(self, filename, args_):
        '''
            Build Tree from File,
            made for treefolio_main (the file should be generated from Training)
            Parameter:
                filename: Path to File
                args_: Parsed Arguments
        '''
        
        filepointer = open(filename)
        structure = json.load(filepointer)
        filepointer.close()
        
        if (not isinstance(structure,dict) or not structure.has_key("tree")):
            raise Exception("Building failed: Structure File missing \"tree\" - key.")
        
        steps = structure["tree"]
        
        Printer.print_verbose("Building Configuration ...")
        
        if (isinstance(steps,dict)):
            if (steps.has_key("steptype")):
                if (steps["steptype"] == "decision"):
                    self.first = Decisionstep(steps["oracle"], steps["folds"])
                    self.first.init_from_structure_no_recursion(steps)
                    #self.first.name = steps["name"]
                    self.build_step(self.first,steps)
                else:
                    self.first = Executionstep()
                    self.first.name = steps["name"]
                    self.build_execution_step(self.first,steps)
            else:
                raise Exception("Building failed: Structure File missing \"steptype\" - key.")
        else:
            raise Exception("Building failed: Structure File is no Dictionary.")
        
        Printer.print_c("Configuration ready!")
        
        return True
        
    
    def build_step(self, currentstep, structure):
        
        if (isinstance(structure,dict) and hasattr(currentstep,"nextstep")):
            
            length = 0;
            if (structure.has_key("next") and isinstance(structure["next"],(list, tuple))):
                length = len(structure["next"])
                for nextsteps in structure["next"]:
                    
                    if (isinstance(nextsteps,dict)):
                        if (nextsteps.has_key("steptype")):
                            if (nextsteps["steptype"] == "decision"):
                                step = Decisionstep(nextsteps["oracle"], nextsteps["folds"])
                                step.init_from_structure_no_recursion(nextsteps)
                                #step.name = nextsteps["name"]
                                self.build_step(step,nextsteps)
                                currentstep.nextstep.append(step)
                            else:
                                step = Executionstep()
                                step.init_from_structure_no_recursion(nextsteps)
                                #step.name = nextsteps["name"]
                                self.build_execution_step(step,nextsteps)
                                currentstep.nextstep.append(step)
                                
                        else:
                            raise Exception("Building failed: Structure File missing steptype - key.")
                    else:
                        raise Exception("Building failed: Structure File is no Dictionary.")

            else:
                raise Exception("Building failed: Structure File missing \"next\" - key.")
            
            if (structure.has_key("names") and isinstance(structure["names"],(list, tuple))):
                if (length != len(structure["names"])):
                    raise Exception("Building failed: Length of \"next\" and \"names\" are different.")
                currentstep.decision_name = structure["names"]
            else:
                raise Exception("Building failed: Structure File missing \"names\" - key.")
            
            
            if structure.has_key("default"):
                currentstep.default_next_number = structure["default"]
                if length > currentstep.default_next_number:
                    currentstep.default_decision_name = currentstep.decision_name[currentstep.default_next_number]
                else:
                    #currentstep.default_decision_name = str(currentstep.default_next_number)
                    raise Exception("Building failed: Length of \"next\" is not greater than \"default\".")
            else:
                raise Exception("Building failed: Structure File missing \"default\" - key.")
            
            
            if structure.has_key("modelfile"):
                currentstep.modelfile = structure["modelfile"]
            
        else:
            raise Exception("Building failed: Structure is no Dictionary.")
        
    
    
    def build_execution_step(self, currentstep, structure):
        
        if (isinstance(structure,dict) and structure.has_key("steptype") and structure["steptype"] != "decision"):
            
            if structure.has_key("configuration_file"):
                currentstep.configuration_file = structure["configuration_file"]
                currentstep.load_executor(structure["execcommand"],structure["name"],0) #TODO!
            else:
                raise Exception("Building failed: Execution Step without \"configuration_file\" - key.")
            
        else:
            raise Exception("Building failed: Structure File is no Dictionary.")
        
        
        
        
        