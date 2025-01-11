import sys
import numpy as np

#Class definitions
#Class of the imput file
class Input_file:
    def __init__(self, path):
        self.file_path = path
        self.corr_runs_path = self.read_setting("corr_runs_path")
        self.weight_runs_path = self.read_setting("weight_runs_path")
        self.corr_runs = self.read_setting("corr_runs")
        self.weight_runs = self.read_setting("weight_runs")
    
    def read_setting(self, opt_name, dtype='str'):
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
        if len(tokens) == 1:
            return tokens[0]
        else:
            return tokens
 
#Exectution
if(len(sys.argv) < 2):
    print("Usage: python3 " + sys.argv[0] + " input_file.in")
    exit(1)

in_f = Input_file(sys.argv[1])

print(in_f.corr_runs_path)
print(in_f.weight_runs_path)
print(in_f.corr_runs)
print(in_f.weight_runs)
