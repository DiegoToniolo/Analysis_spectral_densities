import sys
import subprocess
import numpy as np
import re

#Class definitions
#Class of the imput file
class Input_file:
    def __init__(self, path: str):
        self.file_path = path
        self.corr_runs_path = self.read_setting("corr_runs_path")[0]
        self.weight_runs_path = self.read_setting("weight_runs_path")[0]
        self.corr_runs_v1 = self.read_setting("corr_runs_v1")
        self.corr_runs_v2 = self.read_setting("corr_runs_v2")
        self.weight_runs = self.read_setting("weight_runs")
    
    def read_setting(self, opt_name: str, dtype='str'):
        self.in_file = open(self.file_path, "r")
        lines = self.in_file.readlines()
        flag = 0
        for l in lines:
            tokens = l.split()
            if tokens[0] == opt_name:
                del tokens[0]
                tokens = np.array(tokens, dtype)
                flag = 1
                break
        
        self.in_file.close()

        if flag == 0:
            print("Option " + opt_name + " not found")
            exit(1)
        if len(tokens) == 1 and dtype != 'str':
            return tokens[0]
        else:
            return tokens

class Read_connected:
    def __init__(self,  in_f: Input_file):
        for i in range(len(in_f.corr_runs_v1)):
            self.l0_v1 = self.level0_to_read(in_f.corr_runs_path + in_f.corr_runs_v1[i] + "/dat/")
            for f_to_read in self.l0_v1:
                self.read_level1_config(in_f.corr_runs_path + in_f.corr_runs_v1[i] + "/dat/" + f_to_read)

        #TO BE IMPLEMENTED
        #for i in range(len(in_f.corr_runs_v2)):
        #    self.l0_v2 = self.level0_to_read(in_f.corr_runs_path + in_f.corr_runs_v2[i] + "/dat/")
        
    def level0_to_read(self, path: str):
        cmd_cd = ['ls', path]
        proc = subprocess.Popen(cmd_cd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        o, e = proc.communicate()
        err = e.decode()
        if err != '':
            print("Error in changing directory")
            print(err)
            exit(1)
        
        runs_to_read = []
        for string in o.decode().split():
            if len(re.findall("dat$", string)) != 0:
                runs_to_read.append(string)

        return runs_to_read
        
    def read_level1_config(self, file_path: str):
        f = open(file_path, "rb")
        f.close()
        
        

#Exectution
if(len(sys.argv) < 2):
    print("Usage: python3 " + sys.argv[0] + " input_file.in")
    exit(1)

in_f = Input_file(sys.argv[1])
r = Read_connected(in_f)

#print(in_f.corr_runs_path)
#print(in_f.weight_runs_path)
#print(in_f.corr_runs)
#print(in_f.weight_runs)
