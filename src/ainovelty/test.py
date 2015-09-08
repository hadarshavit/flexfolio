from data_io import *
from problem import *

'''
    ** fix when \a and \b are used -> /a /b: http://stackoverflow.com/questions/22961145/the-reason-python-string-assignments-accidentally-change-b-into-x08-and
    **
'''

# csv_file = "E:/_DELLNtb-Offce/_Bitbucket SRC/ainovelty/data/openml-accuracy.csv"
# 
# dio = DataIO()
# # dio.read_csv_perf_file(csv_file, ignore_first_row = True, ignore_first_col = True)
# 
# p_type = ProblemType.Optimization
# 
# if p_type == ProblemType.Optimization:
#     print("Optimization")
# elif p_type == ProblemType.Decision:
#     print("Decision")
# else:
#     raise Exception("Given problem type is undefined: "+str(p_type))


dict = {"a": {"bir": 1},
        "b": 2} 

print dict["b"]
