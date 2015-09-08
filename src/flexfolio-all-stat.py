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
            
    print "\ndataset \tPAR10 \n---------------"
    for inx in range(len(datasets)):
        print datasets[inx], "\t", par10_scores[inx]
        
    f.close()
    

bench_root_folder = "/home/misir/Desktop/aslib_data-aslib-v1.1"    
bench_folder_list = get_dataset_folder_list(bench_root_folder)
res_folder_name = datetime.datetime.now().strftime("%d%m%Y%H%M%S%f")

### console output to a file: http://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python
res_file = "log-out.txt"
res_folder_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__)), "output"), res_folder_name)
res_file_path = os.path.join(res_folder_path, res_file)

if not os.path.exists(res_folder_path):
    os.makedirs(res_folder_path)

f = open(res_file_path, 'w')
sys.stdout = f

for bench_folder in bench_folder_list:
    
#     if not ("CSP" in bench_folder or "PREMARSHALLING" in bench_folder or "QBF" in bench_folder):
#         continue
#     if "SAT11-RAND" in bench_folder or "SAT11-HAND" in bench_folder:
#         continue
#     if "SAT11-HAND" not in bench_folder:
#         continue
    
    print "Started solving ", bench_folder
    
    flexfolio_csv_file = os.path.join(res_folder_path, os.path.basename(os.path.normpath(bench_folder)) + ".csv")
    
    args = ['--aslib-dir', bench_folder, '--model-dir', 'MODEL_DIR', '--print-times', flexfolio_csv_file]
    
    trainer = Trainer()
    trainer.main(args)
    
    #http://stackoverflow.com/questions/3167494/how-often-does-python-flush-to-a-file
    f.flush()
    
    ##break ## just for test
    

f.close()

process_out_file(res_file_path)
