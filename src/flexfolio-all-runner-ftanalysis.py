import sys
import os
import datetime
import ntpath


from src.ainovelty.data_io import get_dataset_folder_list
from src import flexfolio_train
from src.flexfolio_train import Trainer



def process_out_file(res_file_path):
    
    datasets = []
    par10_scores = []
    ft_imp = []
    imp_add_cnt = 0
    cost_add_cnt = 0
    ft_imp_val = 0
    ft_cost_val = 0
    ft_cost = []
    
    #http://stackoverflow.com/questions/14245227/python-reset-stdout-to-normal-after-previously-redirecting-it-to-a-file
    sys.stdout = sys.__stdout__ ## reset to its default value
        
    f = open(res_file_path,'r')
    
    print "@process_out_file: - res_file_path = ", res_file_path 

    for str_line in f.readlines():
        ###print(str_line)
        if "Started solving" in str_line:  
            ##datasets.append(str_line[15:len(str_line)].strip())
            ##http://stackoverflow.com/questions/8384737/python-extract-file-name-from-path-no-matter-what-the-os-path-format
            datasets.append(ntpath.basename(str_line[15:len(str_line)].strip()))
        elif "% PAR10: {1: " in str_line:
            par10_scores.append(str_line[13:str_line.index(',')].strip())
        elif "@@@@@@@@@@@" in str_line:
            ft_imp_val += float(str_line[12:].strip())
            imp_add_cnt += 1
        elif "ZZZZZZZZZZZ" in str_line:
            ft_cost_val += float(str_line[12:].strip())
            cost_add_cnt += 1
            
        if imp_add_cnt == 10 and cost_add_cnt == 10:    
            ft_imp.append((ft_imp_val/10.0))
            ft_cost.append((ft_cost_val/10.0))
            imp_add_cnt = 0
            cost_add_cnt = 0
            ft_imp_val = 0
            ft_cost_val = 0
            
            
    print "\nPAR10 \n---------------"
    for inx in range(len(par10_scores)):
        print par10_scores[inx], "\t", ft_imp[inx], "\t", ft_cost[inx]
        
    f.close()
    

bench_root_folder = "/home/misir/Desktop/aslib_data-aslib-v1.1"    
bench_folder_list = get_dataset_folder_list(bench_root_folder)

### console output to a file: http://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python
res_file = "res-"+datetime.datetime.now().strftime("%d%m%Y%H%M%S%f")+".txt"
res_file_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__)), "output"), res_file)

f = open(res_file_path, 'w')
sys.stdout = f

for bench_folder in bench_folder_list:
       
#     if not ("CSP" in bench_folder or "PREMARSHALLING" in bench_folder or "QBF" in bench_folder):
#         continue
    if "SAT11-HAND" not in bench_folder:
        continue
       
    print "Started solving ", bench_folder
       
    args = ['--aslib-dir', bench_folder, '--model-dir', 'MODEL_DIR']
       
    for ain_num_ft_to_remove in range(1,50):
           
        print "ain_num_ft_to_remove: ", ain_num_ft_to_remove 
        trainer = Trainer(ain_num_ft_to_remove)
        trainer.main(args)
       
        #http://stackoverflow.com/questions/3167494/how-often-does-python-flush-to-a-file
        f.flush()
       
    ##break ## just for test
       
   
f.close()

# ASP: res-18062015121454514532.txt
# res-17062015145135573801.txt
# res-17062015150844126646.txt
#res_file_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__)), "output"), "res-18062015121454514532.txt")
process_out_file(res_file_path)
