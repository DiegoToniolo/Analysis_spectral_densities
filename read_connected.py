import sys
import numpy as np

#Class definitions
#Class of the imput file
class Input_file:
    def __init__(self, path):
        self.file_path = path
        self.in_file = open(self.file_path, "r")
        self.runs = self.read_setting("runs", "float64")
    
    def read_setting(self, opt_name, dtype):
        lines = self.in_file.readlines()
        
        flag = 0
        for l in lines:
            tokens = l.split(" ")
            if tokens[0] == opt_name:
                del tokens[0]
                tokens = np.array(tokens, dtype)
                flag = 1
            break

        if flag == 0:
            print("Option" + opt_name + "not found")
            exit(1)
        
        return tokens
 
#Exectution
if(len(sys.argv) < 2):
    print("Usage: python3 " + sys.argv[0] + " input_file.in")
    exit(1)

in_f = Input_file(sys.argv[1])

print(in_f.file_path)
print(in_f.runs)