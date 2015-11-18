import os, shutil

PATH = "/home/lindauer/git/aslib-spec/scripts/results/"

for d in os.listdir(PATH):
    if not os.path.isdir(os.path.join(PATH,d)):
        continue
    shutil.copy(os.path.join(PATH,d, "sunny.csv"), "%s_sunny-ori.csv" %(d))