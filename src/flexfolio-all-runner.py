import sys
import os
import datetime
import ntpath
import time


from src.ainovelty.data_io import get_dataset_folder_list
from src import flexfolio_train
from src.flexfolio_train import Trainer
from src.ainovelty.experiment import Experiment
from src.ainovelty import settings



def process_out_file(res_file_path):
    
    kfold_cv = 10
    
    datasets = []
    par10_scores = []
    algs_selected = []
    insts_selected = []
    inst_fts_selected = []
    
    #http://stackoverflow.com/questions/14245227/python-reset-stdout-to-normal-after-previously-redirecting-it-to-a-file
    sys.stdout = sys.__stdout__ ## reset to its default value
        
    f = open(res_file_path,'r')
    
    print "@process_out_file: - res_file_path = ", res_file_path 

    dataset_inx = -1
    current_fold_inx = -1
    
    for str_line in f.readlines():
        ###print(str_line)
        if "Started solving" in str_line:
            
            dataset_inx += 1
            current_fold_inx = 0
            algs_selected.append("")
            inst_fts_selected.append("")
            
            
            ##datasets.append(str_line[15:len(str_line)].strip())
            ##http://stackoverflow.com/questions/8384737/python-extract-file-name-from-path-no-matter-what-the-os-path-format
            datasets.append(ntpath.basename(str_line[15:len(str_line)].strip()))
            
        elif "% PAR10: {1: " in str_line:
            par10_scores.append(str_line[13:str_line.index(',')].strip())

        elif "Features to keep are determined:" in str_line:
            num_fts_info = str_line[str_line.index("("):str_line.index(")")+1].strip().replace("out of", "/")
            
            inst_fts_selected[dataset_inx] += num_fts_info + " " + str_line[str_line.index(":")+1:].strip()
            if current_fold_inx < kfold_cv-1:
                inst_fts_selected[dataset_inx] += " : "

        elif "Algorithms/solvers to keep are determined" in str_line:
            num_algs_info = str_line[str_line.index("("):str_line.index(")")+1].strip().replace("out of", "/")
            
            algs_selected[dataset_inx] += num_algs_info + " " + str_line[str_line.index(":")+1:].strip()
            if current_fold_inx < kfold_cv-1:
                algs_selected[dataset_inx] += " : "
                
            current_fold_inx += 1
            

            
    print "\ndataset \tPAR10 \tAlgs_selected \tInst_fts_selected \n------------------------------------------------------------"
    for inx in range(len(datasets)):
        print datasets[inx], "\t###\t", par10_scores[inx], "\t###\t", algs_selected[inx], "\t###\t", inst_fts_selected[inx]
        
    f.close()
 
 

# def print_exp_settings(res_folder_path):
def print_exp_settings():
    
    exp = Experiment()
    
    print "inst_clst_ft_type\t:", exp.inst_clst_ft_type
    print "svd_type\t:", exp.svd_type
    print "dim_rd_type\t:", exp.dim_rd_type
    print "clst_method\t:", exp.clst_method
    print "ft_selection_method\t:", exp.ft_selection_method
    print "svd_dim\t:", exp.svd_dim
    print "svd_outlier_threshold\t:", exp.svd_outlier_threshold
    print "ft_outlier_threshold\t:", exp.ft_outlier_threshold
    print "to_report\t:", exp.to_report
    print "to_plot\t:", exp.to_plot
    
    print "alg_subset_selection\t:", exp.alg_subset_selection
    print "inst_subset_selection\t:", exp.inst_subset_selection
    print "ft_subset_selection\t:", exp.ft_subset_selection
    
    print "ft_postprocessing\t:", exp.ft_postprocessing
    
#     exp.output_folder_name = res_folder_path
#     print "output_folder_name\t:", exp.output_folder_name
    
    

bench_root_folder = "/home/misir/Desktop/aslib_data-aslib-v1.1"    
bench_folder_list = get_dataset_folder_list(bench_root_folder)
res_folder_name = datetime.datetime.now().strftime("%d%m%Y%H%M%S%f")

### console output to a file: http://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python
res_file = "log-out.txt"
res_folder_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__)), "output"), res_folder_name)
res_file_path = os.path.join(res_folder_path, res_file)

settings.___output_folder___ = res_folder_path

if not os.path.exists(res_folder_path):
    os.makedirs(res_folder_path)
 
f = open(res_file_path, 'w')
sys.stdout = f ## to write prints to res_file_path 
 

# print_exp_settings(res_folder_path)
print_exp_settings()
 
 
for bench_folder in bench_folder_list:
     
#     if not ("CSP" in bench_folder or "PREMARSHALLING" in bench_folder or "QBF" in bench_folder):
#         continue
#     if "SAT11-RAND" in bench_folder or "SAT11-HAND" in bench_folder:
#         continue
#     if "SAT11-INDU" not in bench_folder:
#         continue
       
    print "Started solving ", bench_folder
     
    flexfolio_csv_file = os.path.join(res_folder_path, os.path.basename(os.path.normpath(bench_folder)) + ".csv")
     
    args = ['--aslib-dir', bench_folder, '--model-dir', 'MODEL_DIR', '--print-times', flexfolio_csv_file]
     
     
    ## http://stackoverflow.com/questions/5998245/get-current-time-in-milliseconds-in-python
    start_millis = int(round(time.time() * 1000))
     
    trainer = Trainer()
    trainer.main(args)
     
    end_millis = int(round(time.time() * 1000))
     
    print "Total time passed: ", ((end_millis-start_millis) / 1000.0), " seconds"
     
    #http://stackoverflow.com/questions/3167494/how-often-does-python-flush-to-a-file
    f.flush()
     
    ##break ## just for test
     
 
f.close()

#####res_file_path = "/home/misir/Desktop/liclipse/Workspace/flexfolio-data_filtering-v3/output/03072015111256145591/log-out.txt"
process_out_file(res_file_path)
