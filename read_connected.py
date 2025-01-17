import sys
import subprocess
import numpy as np
import re

stdoustream = sys.stdout

#Function definitions
def jacknife(a):
    mean = np.mean(a, dtype='f8')
    jack = np.zeros(len(a), dtype='f8')
    for i in range(len(a)):
        jack[i] = mean - (a[i] - mean)/(len(a) - 1)
    
    return mean, jack

def var_jack(jack):
    return (len(jack) - 1) * np.var(jack)

def cov_jack(mean1, mean2, jack1, jack2):
    return (len(jack1) - 1) * np.sum((jack1 - mean1)*(jack2 - mean2))

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
        self.log_file = self.read_setting("log_file")[0]
        self.out_path = self.read_setting("out_path")[0]
    
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

#Class for the structure of the unweighted correlators
class Data_conn:
    def __init__(self, n_c:int = 0, n_src:int = 0, n_op:int = 0, n_t:int = 0, y0:int = 0):
        self.set_dims(n_c, n_src, n_op, n_t)
        self.set_y0(y0)

    def set_dims(self, n_c:int = 0, n_src:int = 0, n_op:int = 0, n_t:int = 0):
        self.configuration = np.zeros((n_c, n_src, n_op, n_t), dtype='f8')

    def set_y0(self, y0:int=0):
        self.y0 = y0

#Class for the structure of the reweighting    
class Data_weight:
    def __init__(self, n_rw:int = 0, n_c:int = 0):
        self.set_dims(n_rw, n_c)
    
    def set_dims(self, n_rw:int, n_c:int):
        self.configuration = np.zeros((n_rw, n_c))

#Class to read all the data and to perform the needed averages.       
class Read_connected:
    def __init__(self,  in_f: Input_file):
        self.op = in_f.operators
        self.n_l1 = in_f.n_config_l1
        self.log = open(in_f.log_file, 'w')
        
        self.v1_to_merge = []
        for s in in_f.to_merge:
            self.v1_to_merge.append(s.split('-'))
        
        #Reading weights
        self.l1_w_config = []
        for i in range(len(in_f.weight_runs)):
            self.write_log("Reading weight run number " + in_f.weight_runs[i], self.log)
            self.l0_w = self.level0_to_read(in_f.weight_runs_path + in_f.weight_runs[i] + "/dat/")
            count = 1
            for f_to_read in self.l0_w:
                self.write_log("\tReading level 1 configuration number {}".format(count), self.log)
                count += 1
                self.l1_w_config.append(self.read_level1_weight(in_f.weight_runs_path + in_f.weight_runs[i] + "/dat/" + f_to_read, endian='little'))

        self.averages = []
        self.jack = []    
        #Reading correlators version 1 
        for i in range(len(in_f.corr_runs_v1)):
            self.write_log("Reading run number " + in_f.corr_runs_v1[i] + ", version 1", self.log)
            self.l0_v1 = self.level0_to_read(in_f.corr_runs_path + in_f.corr_runs_v1[i] + "/dat/")
            
            self.l1_config = []
            count = 0 
            for f_to_read in self.l0_v1:
                self.write_log("Reading level 0 configuration number {}".format(count + 1), self.log)
                d = self.read_level1_config(in_f.corr_runs_path + in_f.corr_runs_v1[i] + "/dat/" + f_to_read, endian='little', version='V1')
                l1_av = self.compute_l1_averages(d, self.l1_w_config[count])
                l1_av = np.reshape(l1_av, (1, len(l1_av)))
                if len(self.l1_config) == 0:
                    self.l1_config = l1_av
                else:
                    self.l1_config = np.append(self.l1_config, l1_av, axis=0)
                count += 1
            
            for m in self.v1_to_merge:
                if m[0] == in_f.corr_runs_v1[i]:
                    self.write_log("Merging to " + m[1], self.log)
                    self.l0_v2 = self.level0_to_read(in_f.corr_runs_path + m[1] + "/dat/")
                    
                    for f_to_read in self.l0_v2:
                        self.write_log("Reading level 0 configuration number {}".format(count + 1), self.log)
                        d = self.read_level1_config(in_f.corr_runs_path + m[1] + "/dat/" + f_to_read, endian='little', version='V2')
                        l1_av = self.compute_l1_averages(d, self.l1_w_config[count])
                        l1_av = np.reshape(l1_av, (1, len(l1_av)))
                        if len(self.l1_config) == 0:
                            self.l1_config = l1_av
                        else:
                            self.l1_config = np.append(self.l1_config, l1_av, axis=0)
                        count += 1
                    in_f.corr_runs_v2 = np.delete(in_f.corr_runs_v2, np.where(in_f.corr_runs_v2 == m[1]))
                    break
            
            av_run, jack_run = self.compute_l0_averages(self.l1_config)
            var_run = np.zeros(0, dtype='f8')
            for t in range(len(jack_run[:, 0])):
                var_run = np.append(var_run, var_jack(jack_run[t]))
            self.print_run(av_run, np.sqrt(var_run), in_f.out_path + in_f.corr_runs_v1[i] + ".txt")

            self.averages.append(av_run)
            self.jack.append(jack_run)
        #Reading correlators version 2  
        for i in range(len(in_f.corr_runs_v2)):
            self.write_log("Reading run number " + in_f.corr_runs_v2[i] + ", version 2", self.log)
            self.l0_v2 = self.level0_to_read(in_f.corr_runs_path + in_f.corr_runs_v2[i] + "/dat/")

            count = 0
            self.l1_config = []
            for f_to_read in self.l0_v2:
                self.write_log("Reading level 0 configuration number {}".format(count + 1), self.log)
                d = self.read_level1_config(in_f.corr_runs_path + in_f.corr_runs_v2[i] + "/dat/" + f_to_read, endian='little', version='V1')
                l1_av = self.compute_l1_averages(d, self.l1_w_config[count])
                l1_av = np.reshape(l1_av, (1, len(l1_av)))
                if len(self.l1_config) == 0:
                    self.l1_config = l1_av
                else:
                    self.l1_config = np.append(self.l1_config, l1_av, axis=0)
                count += 1
            
            av_run, jack_run = self.compute_l0_averages(self.l1_config)
            var_run = np.zeros(0, dtype='f8')
            for t in len(jack_run[:, 0]):
                var_run = np.append(var_run, var_jack(jack_run[t]))
            self.print_run(av_run, np.sqrt(var_run), in_f.out_path + in_f.corr_runs_v2[i] + ".txt")

            self.averages.append(av_run)
            self.jack.append(jack_run)
        
        all_run_av, all_run_var = self.compute_run_averages(np.array(self.averages), np.array(self.jack))
        self.print_run(all_run_av, np.sqrt(all_run_var), in_f.out_path + "_total.txt")
        self.log.close()

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
        
    def read_level1_config(self, file_path: str, endian:str='little', version:str='V1') -> Data_conn:
        if endian == 'little':
            end = '<'
        elif endian == 'big':
            end = '>'
        else:
            sys.stdout = self.outstream
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
                if c % 10 == 0:
                    self.write_log("\tReading level 1 config numbers {}-{}".format(c, c + 10), self.log)
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
                if c % 10 == 0:
                    self.write_log("\tReading level 1 config numbers {}-{}".format(c, c + 10), self.log)
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

    def read_level1_weight(self, file_path: str, endian:str='little') -> Data_weight:
        if endian == 'little':
            end = '<'
        elif endian == 'big':
            end = '>'
        else:
            sys.stdout = self.outstream
            print("Wrong specification of endianness")
            exit(1)
        
        f = open(file_path, 'rb')

        tmp = np.fromfile(f, dtype=end + 'i4', count = 2)
        n_rw, n_slice = tmp[0], tmp[1]
        n_fct, n_src = np.fromfile(f, dtype=end + 'i4', count = n_rw), np.fromfile(f, dtype=end + 'i4', count = n_rw)

        if len(self.n_l1) != n_slice:
            print("Number of level 1 configurations not correct")
            exit(1)
        
        tot_conf = np.prod(self.n_l1)
        data = Data_weight(n_rw, tot_conf)
        for c in range(tot_conf):
            np.fromfile(f, dtype=end + 'i4', count = n_slice)
            for rw in range(n_rw):
                for fct in range(n_fct[rw]):
                    for src in range(n_src[rw]):
                        np.fromfile(f, dtype=end + 'f8', count = 1)
            for rw in range(n_rw):
                tmp1 = np.zeros(n_fct[rw])
                for fct in range(n_fct[rw]):
                    tmp2 = np.zeros(n_src[rw])
                    for src in range(n_src[rw]):
                        tmp2[src] = np.fromfile(f, dtype=end + 'f8', count = 1)[0] + 437.0
                    #print(tmp2)
                    tmp1[fct] = np.mean(np.exp(-tmp2))
                data.configuration[rw][c] = np.prod(tmp1)
        return data
        f.close()

    def write_log(self, string:'str', f):
        sys.stdout = f
        print(string, flush=True)
        sys.stdout = stdoustream

    def compute_l1_averages(self, d:Data_conn, w:Data_weight):
        av_op = np.mean(d.configuration, axis=2, dtype='f8')
        av_src = np.mean(av_op, axis=1, dtype='f8')
        dims = av_src.shape
        
        av_w = np.mean(w.configuration[0], dtype='f8')
        
        corr_reweighted = np.zeros(dims[1])
        for t in range(dims[1]):
            av_prod = np.mean(-1.0 * w.configuration[0] * av_src[:, t])
            corr_reweighted[t] = av_prod/av_w
        
        return np.roll(corr_reweighted, -d.y0)

    def compute_l0_averages(self, l1:np.ndarray):
        #av_l0 = np.mean(l1, axis=0, dtype = 'f8')
        av_t = np.zeros(0, dtype='f8')
        n0 = len(l1[:, 0])
        T = len(l1[0, :])
        jack_t = np.zeros((0, n0), dtype='f8')
        for t in range(int(T/2)):
            mean, jack = jacknife(l1[:, t])
            if t == 0:
                av_t = np.append(av_t, mean)
                jack_t = np.append(jack_t, np.reshape(jack, (1, n0)), axis = 0)
            else:
                mean_t_refl, jack_t_refl = jacknife(l1[:, len(l1[0, :]) - t])
                av_t = np.append(av_t, 0.5*(mean + mean_t_refl))
                jack_t = np.append(jack_t, np.reshape(0.5*(jack + jack_t_refl), (1, n0)), axis = 0)
        return av_t, jack_t

    def print_run(self, corr, std_dev_corr, path):
        f = open(path, "w")
        
        sys.stdout = f
        for i in range(len(corr)):
            print("{} {}".format(corr[i], std_dev_corr[i]), flush=True)
        sys.stdout = stdoustream
        
        f.close()
    
    def compute_run_averages(self, av_l0:np.ndarray, jack_l0:np.ndarray):
        weights = np.zeros(av_l0.shape, dtype = 'f8')
        for run in range(len(av_l0[:, 0])):
            for t in range(len(av_l0[0, :])):
                weights[run, t] = 1.0/var_jack(jack_l0[run, t])

        av = np.average(av_l0, axis = 0, weights=weights)
        jack_av = np.zeros((len(av), 0), dtype='f8')
        for j in range(len(jack_l0[0, 0, :])):
            jack_av = np.append(jack_av, np.reshape(np.average(jack_l0[:, :, j], axis = 0, weights=weights), (len(av), 1)), axis=1)
        
        var = np.zeros(0, dtype='f8')
        for t in range(len(av)):
            var = np.append(var, var_jack(jack_av[t]))
        
        return av, var

#Execution
if(len(sys.argv) < 2):
    print("Usage: python3 " + sys.argv[0] + " input_file.in")
    exit(1)

in_f = Input_file(sys.argv[1])
r = Read_connected(in_f)