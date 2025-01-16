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
        self.operators = self.read_setting("op_to_average", dtype='int')
        self.n_config_l1 = self.read_setting("l1_configurations_per_slice", dtype='int')
        self.to_merge = self.read_setting("to_merge")
        for s in self.to_merge:
            v1_v2_to_merge = s.split('-')
            flag = 0
            for v1 in self.corr_runs_v1:
                if v1 == v1_v2_to_merge[0]:
                    flag = 1
                    break
            if flag == 0:
                print("First item in merge tuple " + s + " not in version 1 list of runs")
                exit(1)
            flag = 0
            for v2 in self.corr_runs_v2:
                if v2 == v1_v2_to_merge[1]:
                    flag = 1
                    break
            if flag == 0:
                print("Second item in merge tuple " + s + " not in version 2 list of runs")
                exit()


    
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

class Data_conn:
    def __init__(self, n_c:int = 0, n_src:int = 0, n_op:int = 0, n_t:int = 0, y0:int = 0):
        self.set_dims(n_c, n_src, n_op, n_t)
        self.set_y0(y0)

    def set_dims(self, n_c:int = 0, n_src:int = 0, n_op:int = 0, n_t:int = 0):
        self.configuration = np.zeros((n_c, n_src, n_op, n_t), dtype='f8')

    def set_y0(self, y0:int=0):
        self.y0 = y0
        
class Read_connected:
    def __init__(self,  in_f: Input_file):
        self.op = in_f.operators
        self.n_l1 = in_f.n_config_l1

        self.v1_to_merge = []
        for s in in_f.to_merge:
            self.v1_to_merge.append(s.split('-'))

        for i in range(len(in_f.corr_runs_v1)):
            print("Reading run number " + in_f.corr_runs_v1[i] + ", version 1")
            self.l0_v1 = self.level0_to_read(in_f.corr_runs_path + in_f.corr_runs_v1[i] + "/dat/")
            
            count = 1
            self.l1_config = []
            for f_to_read in self.l0_v1:
                print("Reading level 0 configuration number {}".format(count))
                count += 1
                self.l1_config.append(self.read_level1_config(in_f.corr_runs_path + in_f.corr_runs_v1[i] + "/dat/" + f_to_read, endian='little', version='V1'))
            
            for m in self.v1_to_merge:
                if m[0] == in_f.corr_runs_v1[i]:
                    print("Merging to " + m[1])
                    self.l0_v2 = self.level0_to_read(in_f.corr_runs_path + m[1] + "/dat/")
                    
                    for f_to_read in self.l0_v2:
                        print("Reading level 0 configuration number {}".format(count))
                        self.l1_config.append(self.read_level1_config(in_f.corr_runs_path + m[1] + "/dat/" + f_to_read, endian='little', version='V2'))
                        count += 1
                        print(self.l1_config[0].configuration[99][0][0], self.l1_config[0].y0)
                    in_f.corr_runs_v2 = np.delete(in_f.corr_runs_v2, np.where(in_f.corr_runs_v2 == m[1]))
                    break
        
        for i in range(len(in_f.corr_runs_v2)):
            print("Reading run number " + in_f.corr_runs_v1[i] + ", version 2")
            self.l0_v2 = self.level0_to_read(in_f.corr_runs_path + in_f.corr_runs_v2[i] + "/dat/")

            count = 1
            self.l1_config = []
            for f_to_read in self.l0_v2:
                print("Reading level 0 configuration number {}".format(count))
                count += 1
                self.l1_config.append(self.read_level1_config(in_f.corr_runs_path + in_f.corr_runs_v1[i] + "/dat/" + f_to_read, endian='little', version='V2'))
            
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
        
    def read_level1_config(self, file_path: str, endian:str='little', version:str='V1'):
        if endian == 'little':
            end = '<'
        elif endian == 'big':
            end = '>'
        else:
            print("Wrong specification of endianness")
            exit(1)
        
        f = open(file_path, "rb")
        data = Data_conn()
        if version == 'V1':
            #Reading header
            tmp = np.fromfile(f, dtype=end + 'i4', count = 5)
            header = {'m0': tmp[0], 'n_source': tmp[1], 'n_op': tmp[2], 't_max': tmp[3], 'n_slice': 2}
        
            if len(self.n_l1) != header['n_slice']:
                print("Number of level 1 configurations not correct")
                exit(1)
        
            tot_conf = np.prod(self.n_l1)
            #Reading configurations
            data.set_dims(tot_conf, header['n_source'], len(self.op), header['t_max'])
            op_to_read = np.zeros(header['n_op'])
            for i in self.op:
                op_to_read[i] = 1

            for c in range(tot_conf):
                print("\tReading level 1 config number {}".format(c + 1))
                np.fromfile(f, dtype=end + 'i4', count = header['n_slice'])
                src_pos = np.zeros((header['n_source'], 4))
        
                for src in range(header['n_source']):
                    src_pos[src] = np.fromfile(f, dtype=end + 'i4', count = 4)

                for m in range(header['m0']):
                    for s in range(header['n_source']):
                        counter = 0
                        for op in range(header['n_op']):
                            if op_to_read[op] == 1:
                                for t in range(header['t_max']):
                                    data.configuration[c][s][counter][t] = np.fromfile(f, dtype=end + 'f8', count = 2)[0]
                                counter += 1
                            else:
                                for t in range(header['t_max']):
                                    np.fromfile(f, dtype=end + 'f8', count = 2)
            
                for m in range(header['m0']):
                    for s in range(header['n_source']):
                        for op in range(4):
                            for t in range(header['t_max']):
                                np.fromfile(f, dtype=end + 'f8', count = 2)
            data.set_y0(int(src_pos[0][0]))
        elif version == 'V2':
            #Reading header
            tmp = np.fromfile(f, dtype=end + 'i4', count = 5)
            header = {'m0': tmp[0], 'n_source': tmp[1], 'n_op': tmp[2], 't_max': tmp[3], 'n_slice': 2}
        
            if len(self.n_l1) != header['n_slice']:
                print("Number of level 1 configurations not correct")
                exit(1)

            tot_conf = np.prod(self.n_l1)

            #Reading configurations
            data.set_dims(tot_conf, header['n_source'], len(self.op), header['t_max'])
            
            op_to_read = np.zeros(header['n_op'] + 4)
            for i in self.op:
                op_to_read[i] = 1

            for c in range(tot_conf):
                print("\tReading level 1 config number {}".format(c + 1))
                np.fromfile(f, dtype=end + 'i4', count = header['n_slice'])
                src_pos = np.zeros((header['n_source'], 4))
        
                for src in range(header['n_source']):
                    src_pos[src] = np.fromfile(f, dtype=end + 'i4', count = 4)

                for m in range(header['m0']):
                    for s in range(header['n_source']):
                        counter = 0
                        for op in range(header['n_op'] + 4):
                            if op_to_read[op] == 1:
                                for t in range(header['t_max']):
                                    data.configuration[c][s][counter][t] = np.fromfile(f, dtype=end + 'f8', count = 2)[0]
                                counter += 1
                            else:
                                for t in range(header['t_max']):
                                    np.fromfile(f, dtype=end + 'f8', count = 2)
            data.set_y0(int(src_pos[0][0]))
        else:
            print("Wrong version number")
            exit(1)
        return data
        f.close()
        
#Execution
if(len(sys.argv) < 2):
    print("Usage: python3 " + sys.argv[0] + " input_file.in")
    exit(1)

in_f = Input_file(sys.argv[1])
r = Read_connected(in_f)