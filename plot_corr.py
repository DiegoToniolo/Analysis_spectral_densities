import numpy as np
import sys
import os
import subprocess
import re
from matplotlib import pyplot as plt

class Input_file:
    def __init__(self, path: str):
        if path != "":
            self.file_path = path
            self.in_dir = self.read_setting("in_dir")[0]
    
    def read_setting(self, opt_name: str, dtype='str'):
        self.in_file = open(self.file_path, "r")
        lines = self.in_file.readlines()
        flag = 0
        for l in lines:
            tokens = l.split()
            if not l.isspace():
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

#Execution
class main:
    def __init__(self):
        if(len(sys.argv) < 2):
            print("Usage: python3 " + sys.argv[0] + " input_file.in")
            exit(1)
        in_f = Input_file(sys.argv[1])
        dat = self.data_to_read(in_f)
    
    def data_to_read(self, in_f:Input_file = Input_file("")) -> list:
        cmd_cd = ['ls', in_f.in_dir]
        proc = subprocess.Popen(cmd_cd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        o, e = proc.communicate()
        err = e.decode()
        if err != '':
            print("Error in changing directory")
            print(err)
            exit(1)
        
        data = []
        for string in o.decode().split():
            if len(re.findall("txt$", string)) != 0:
                data.append(string)

        return data


main()