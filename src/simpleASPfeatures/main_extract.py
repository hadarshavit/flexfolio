'''
Created on Oct 29, 2013

@author: Marius Lindauer
'''
import sys
import math
import json

class ASPSimpleFeatures(object):
    '''
        some simple stats for ASP instances in lparse format;
        motivated by ME-ASP feature extractor
    '''
    
    def __init__(self):
        '''
            Constructor
        '''
        self.HUMAN = True
        self.EXTENDED = True
        
        self.rules = 0
        self.positive_atoms = set()
        self.negative_atoms = set()
        self.facts = 0
        self.disjunctive_facts = 0 # empty body
        self.disjunctive_rules = 0 # non empty body; type 8
        self.unary_rules = 0 # size regarding head and body
        self.binary_rules = 0 # size regarding head and body
        self.ternary_rules = 0 # size regarding head and body
        self.horn_rules = 0 # rules with empty negative body
        self.disg = 0 # disjunctive rules
        self.normal_rules = 0 # basic rules; type 1
        self.constr = 0 # constraints; empty head
        self.constraint_rules = 0 # constraint rules; type 2
        self.choice_rules = 0 # choice rules; type 3
        self.weight_rulres = 0 # weight_rules; type 5
        self.minimize_rules = 0 # minimize_rules; type 6
    
    def extract(self, fp):
        '''
            iterate over lines and deciced the rule type 
        '''
        
        
        for line in fp:
            elements = map(int,line.split(" "))
            head = elements[0]
            if head == 0: # end of program
                break
            elif head == 1:
                self.__evaluate_normal_rule(elements)
            elif head == 2:
                self.__evaluate_contraint_rule(elements)
            elif head == 3:
                self.__evaluate_choice_rule(elements)                
            elif head == 5:
                self.__evaluate_weight_rule(elements)
            elif head == 6:
                self.__evaluate_minimize_rule(elements)
            elif head == 8:
                self.__evaluate_disjunctive_rule(elements)
            else:    
                sys.stderr.write("Unknown rule type: %s\n" %(line))
            self.rules += 1
        
        self.print_features()
            
    def __evaluate_normal_rule(self, rule):
        ''' 
            count statistics for normal rules
            1 head #literals #negative negative positive
        ''' 
        head = rule[1]

        if head > 1:
            self.normal_rules += 1
        
        n_literals = rule[2]
        n_negative = rule[3]
        #n_positive = n_literals - n_negative
        body = rule[4:] # negative and postive
        
        if head > 1:
            n_head = 1
        else:
            n_head = 0
        
        self.__basic_counts (n_literals, n_head, head, body, n_negative)
            
    def __evaluate_contraint_rule(self, rule):
        ''' 
            count statistics for constraint rules
            2 head #literals #negative bound negative positive
        ''' 
        self.constraint_rules += 1
        head = rule[1]
        n_literals = rule[2]
        n_negative = rule[3]
        #bound = rule[4]
        body = rule[5:] # negative and postive
        
        if head > 1:
            n_head = 1
        else:
            n_head = 0
        
        self.__basic_counts (n_literals, n_head, head, body, n_negative)
            
    def __evaluate_choice_rule(self, rule):
        ''' 
            count statistics for choice rules
            3 #heads heads #literals #negative negative positive
        '''
        self.choice_rules += 1
        n_heads = rule[1]
        heads = rule[ 2 : 2+n_heads ]
        n_literals = rule[ 2 + n_heads]
        n_negative = rule[ 3 + n_heads]
        body = rule[ 4 + n_heads : ]
        
        self.__basic_counts (n_literals, n_heads, 0, body, n_negative)
        
        for atom in heads:
            self.positive_atoms.add(atom)
    
    def __evaluate_weight_rule(self, rule):
        ''' 
            count statistics for weight rules
            5 head bound #lits #negative negative positive weights
        ''' 
        self.weight_rulres += 1
        head = rule[1]
        #bound = rule[2]
        n_literals = rule[3]
        n_negative = rule[4]
        body = rule[ 5 : 5+n_literals ] # rest weights
        
        if head > 1:
            n_head = 1
        else:
            n_head = 0
        
        self.__basic_counts (n_literals, n_head, head, body, n_negative)
        
    def __evaluate_minimize_rule(self, rule):
        ''' 
            count statistics for minimize rules
            6 0 #lits #negative negative positive weights
        '''
        self.minimize_rules += 1
        n_literals = rule[2]
        n_negative = rule[3]
        body = rule[ 4 : 4+n_literals] # rest weights
        
        self.__basic_counts(n_literals, 0, 0, body, n_negative)
                 
    def __evaluate_disjunctive_rule(self, rule):
        ''' 
            count statistics for normal rules
        ''' 
        self.disjunctive_rules += 1
        n_heads = rule[1]
        heads = rule[ 2 : 2+n_heads ]
        n_literals = rule[ 2 + n_heads]
        n_negative = rule[ 3 + n_heads]
        body = rule[ 4 + n_heads : ]
        
        if n_literals == 0:
            self.disjunctive_facts += 1 
        
        self.__basic_counts(n_literals, n_heads, 0, body, n_negative)
        
        for atom in heads:
            self.positive_atoms.add(atom)

    def __basic_counts (self, n_literals, n_heads, head, body, n_negative):
        '''
            some counters which apply to almost all rules
        '''

        if head > 1 and len(body) == 0:
            self.facts += 1
        
        if head > 1: # atom 0: true, atom 1: false
            self.positive_atoms.add(head)
        if head == 1:
            self.constr += 1
        
        n_positive = n_literals - n_negative
        for atom in body[0:n_positive]:
            self.positive_atoms.add(atom)
        for atom in body[n_positive:n_negative]:
            self.negative_atoms.add(atom)
        
        if n_literals + n_heads == 1:
            self.unary_rules += 1 # only head
        elif n_literals + n_heads == 2:
            self.binary_rules += 1 # head and one body literal
        elif n_literals + n_heads == 3:
            self.ternary_rules += 1 # head and two body literal

        if n_heads == 1 and n_negative == 0: # at most one positive literal
            self.horn_rules += 1
        if n_heads == 0 and n_negative == 1: 
            self.horn_rules += 1

    def print_features(self):
        r = self.rules
        a = len(set.union(self.positive_atoms,self.negative_atoms))
        a_r = float(a)/float(r)
        r_a = float(r)/float(a)
        out_dict ={
             "Rules"    : r,
             "Atoms"    : a,
             "A/R"      : a_r,
             "(A/R)^2"  : math.pow(a_r,2),
             "(A/R)^3"  : math.pow(a_r,3),
             "R/A"      : a_r,
             "(R/A)^2"  : math.pow(r_a,2),
             "(R/A)^3"  : math.pow(r_a,3),
             "true facts"    : self.facts,
             "disj facts/R": self.disjunctive_facts / float(r),
             "unary/R" : self.unary_rules / float(r),
             "binary/R" : self.binary_rules / float(r),
             "ternary/R": self.ternary_rules / float(r),
             "horn-facts/R": (self.horn_rules - (self.facts +self.disjunctive_facts)) / float(r),
             "horn/R"   : self.horn_rules / float(r),
             "disg/R"   : self.disjunctive_rules / float(r),
             "norm/R"   : self.normal_rules / float(r),
             "norm-facts/R" : (self.normal_rules - (self.facts +self.disjunctive_facts)) / float(r),
             "costr/R"  : self.constr / float(r)
         }
        
        if self.EXTENDED: 
            out_dict.update({
                             "constraint_rules/R"   : self.constraint_rules / float(r),
                             "choice_rules/R"       : self.choice_rules / float(r),
                             "weight_rules/R"       : self.weight_rulres / float(r),
                             "minimize_rules/R"     : self.minimize_rules / float(r)           
              })
        
        if self.HUMAN:
            print(json.dumps(out_dict,indent=2,sort_keys=True))
            #===================================================================
            # print("%s: %f" %("Rules", out_dict["Rules"]))
            # print("%s: %f" %("Atoms", out_dict["Atoms"]))
            # print("%s: %f" %("A/R", out_dict["A/R"]))
            # print("%s: %f" %("(A/R)^2", out_dict["(A/R)^2"]))
            # print("%s: %f" %("(A/R)^3", out_dict["(A/R)^3"]))
            # print("%s: %f" %("R/A", out_dict["R/A"]))
            # print("%s: %f" %("(R/A)^2", out_dict["(R/A)^2"]))
            # print("%s: %f" %("(R/A)^3", out_dict["(R/A)^3"]))
            # print("%s: %f" %("true facts", out_dict["true facts"]))
            # print("%s: %f" %("disj facts/R", out_dict["disj facts/R"]))
            # print("%s: %f" %("unary/R", out_dict["unary/R"]))
            # print("%s: %f" %("binary/R", out_dict["binary/R"]))
            # print("%s: %f" %("ternary/R", out_dict["ternary/R"]))
            # print("%s: %f" %("horn-facts/R", out_dict["horn-facts/R"]))
            # print("%s: %f" %("horn/R", out_dict["horn/R"]))
            # print("%s: %f" %("disg/R", out_dict["disg/R"]))
            # print("%s: %f" %("norm/R", out_dict["norm/R"]))
            # print("%s: %f" %("norm-facts/R", out_dict["norm-facts/R"]))
            # print("%s: %f" %("costr/R", out_dict["costr/R"]))
            #===================================================================
        else:
            sorted_keys = sorted(out_dict.keys())
            values = [out_dict[k] for k in sorted_keys]
            print("Features: %s\n" %(",".join(map(str,values))))

if __name__ == '__main__':
    if len(sys.argv) == 2:
        fp = open (sys.argv[1], "r")
    else:
        fp = sys.stdin
        
    extractor = ASPSimpleFeatures()
    extractor.extract(fp)
    
    try:
        fp.close()
    except:
        pass
    