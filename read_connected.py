import sys
import subprocess
import numpy as np
import re

stdoustream = sys.stdout

#Class definitions
#Class of the imput file
class Input_file:
    def __init__(self, path: str):
        if path != "":
            self.file_path = path
            self.corr_runs_path = self.read_setting("corr_runs_path")[0]
            self.weight_runs_path = self.read_setting("weight_runs_path")[0]
            self.corr_runs_v1 = self.read_setting("corr_runs_v1")
            self.corr_runs_v2 = self.read_setting("corr_runs_v2")
            self.weight_runs = self.read_setting("weight_runs")
            self.operators = self.read_setting("op_to_average", dtype='int')
            self.n_config_l1 = self.read_setting("l1_configurations_per_slice", dtype='int')
            to_merge = self.read_setting("to_merge")
            self.v1_v2_to_merge = []
            for s in to_merge:
                spl = s.split('-')
                flag = 0
                for v1 in self.corr_runs_v1:
                    if v1 == spl[0]:
                        flag = 1
                        break
                if flag == 0:
                    print("First item in merge tuple " + s + " not in version 1 list of runs")
                    exit(1)
                flag = 0
                for v2 in self.corr_runs_v2:
                    if v2 == spl[1]:
                        flag = 1
                        break
                if flag == 0:
                    print("Second item in merge tuple " + s + " not in version 2 list of runs")
                    exit()
                self.v1_v2_to_merge.append(s.split('-'))
            self.log_file = self.read_setting("log_file")[0]
            self.out_path = self.read_setting("out_path")[0]
            self.std_av = []
            for s in self.read_setting("std_av"):
                self.std_av.append(s.split('-'))
            self.ml_av = []
            for s in self.read_setting("ml_av"):
                self.ml_av.append(s.split('-'))

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

#Class of jacknife variables
class Jacknife:
    def __init__(self, data:np.ndarray=np.array([0, 0])):
        self.mean, self.jack = self.jacknife(data)
    
    def jacknife(self, a):
        mean = np.mean(a, dtype='f8')
        jack = np.zeros(len(a), dtype='f8')
        for i in range(len(a)):
            jack[i] = mean - (a[i] - mean)/(len(a) - 1)
    
        return mean, jack
    
    def variance(self):
        return (len(self.jack) - 1) * np.var(self.jack)
    
    def iscompatible(self, var):
        if not isinstance(var, Jacknife) and not isinstance(var, float):
            print("Non compatible types: Jacknife + " + str(type(var)))
            return False
        elif isinstance(var, Jacknife) and len(self.jack) != len(var.jack):
            print("Jacknife vectors have not equal length: {} and {}".format(len(self.jack), len(var.jack)))
            return False
        else:
            return True

    def __add__(self, var):
        if not self.iscompatible(var):
            exit(1)
        else:
            sum = Jacknife()
            sum.mean = self.mean + var.mean
            sum.jack = self.jack + var.jack
            return sum
    
    def __sub__(self, var):
        if not self.iscompatible(var):
            exit(1)
        else:
            diff = Jacknife()
            diff.mean = self.mean - var.mean
            diff.jack = self.jack - var.jack
            return diff
    
    def __mul__(self, var):
        if not self.iscompatible(var):
            exit(1)
        if isinstance(var, Jacknife):
            prod = Jacknife()
            prod.mean = self.mean * var.mean
            prod.jack = self.jack * var.jack
            return prod
        else:
            prod = Jacknife()
            prod.mean = self.mean * var
            prod.jack = self.jack * var
            return prod
        
    def __truediv__(self, var):
        if not self.iscompatible(var):
            exit(1)
        if isinstance(var, Jacknife):
            quo = Jacknife()
            quo.mean = self.mean / var.mean
            quo.jack = self.jack / var.jack
            return quo
        else:
            quo = Jacknife()
            quo.mean = self.mean / var
            quo.jack = self.jack / var
            return quo

#Class for the structure of the unweighted correlators
class Data_conn:
    def __init__(self, n_src:int = 0, n_op:int = 0, n_t:int = 0, sl:tuple = (0), y0:int = 0):
        self.set_dims(n_src, n_op, n_t, sl)
        self.set_y0(y0)

    def set_dims(self, n_src:int, n_op:int, n_t:int, sl:tuple):
        self.configuration = np.zeros(sl + tuple([n_src, n_op, n_t]), dtype='f8')

    def set_y0(self, y0:int=0):
        self.y0 = y0

#Class for the structure of the reweighting    
class Data_weight:
    def __init__(self, n_rw:int = 0, sl:tuple = (0)):
        self.set_dims(n_rw, sl)
    
    def set_dims(self, n_rw:int, sl:np.ndarray) -> None:
        self.configuration = np.zeros(tuple([n_rw]) + sl)

#Class to read all the data and to perform the needed averages.       
class Read_connected:
    def __init__(self,  in_f: Input_file):
        log = open(in_f.log_file, 'w')
        
        #Reading weights
        l1_w_config = []
        for i in range(len(in_f.weight_runs)):
            self.write_log("Reading weight run number " + in_f.weight_runs[i], log)
            
            l0_files = self.level0_to_read(in_f.weight_runs_path + in_f.weight_runs[i] + "/dat/")
            
            for f_to_read in l0_files:
                l1_w_config.append(self.read_level1_weight(in_f.weight_runs_path + in_f.weight_runs[i] + "/dat/" + f_to_read, endian='little', in_f=in_f))

        averages = []   
        #Reading correlators version 1 
        for i in range(len(in_f.corr_runs_v1)):
            self.write_log("Reading run number " + in_f.corr_runs_v1[i] + ", version 1", log)
            
            l0_files = self.level0_to_read(in_f.corr_runs_path + in_f.corr_runs_v1[i] + "/dat/")
            
            l1_prod = []
            l1_w_av = []
            
            count = 0 
            for f_to_read in l0_files:
                self.write_log("\tReading level 0 configuration number {}".format(count + 1), log)

                d = self.read_level1_config(in_f.corr_runs_path + in_f.corr_runs_v1[i] + "/dat/" + f_to_read, endian='little', version='V1', in_f=in_f)
                p, w = self.l1_averages(d, l1_w_config[count], in_f)
                if count == 0:
                    l1_prod = p[np.newaxis, ...]
                    l1_w_av = w[np.newaxis, ...]
                else:
                    l1_prod = np.append(l1_prod, p[np.newaxis, ...], axis=0)
                    l1_w_av = np.append(l1_w_av, w[np.newaxis, ...], axis=0)
                count += 1
            for m in in_f.v1_v2_to_merge:
                if m[0] == in_f.corr_runs_v1[i]:
                    self.write_log("Merging to " + m[1], log)
                    l0_files = self.level0_to_read(in_f.corr_runs_path + m[1] + "/dat/")
                    
                    for f_to_read in l0_files:
                        self.write_log("\tReading level 0 configuration number {}".format(count + 1), log)
                        
                        d = self.read_level1_config(in_f.corr_runs_path + m[1] + "/dat/" + f_to_read, endian='little', version='V2', in_f=in_f)
                        p, w = self.l1_averages(d, l1_w_config[count], in_f)
                        if count == 0:
                            l1_prod = p[np.newaxis, ...]
                            l1_w_av = w[np.newaxis, ...]
                        else:
                            l1_prod = np.append(l1_prod, p[np.newaxis, ...], axis=0)
                            l1_w_av = np.append(l1_w_av, w[np.newaxis, ...], axis=0)
                        count += 1
                    break
            
            c_run = self.l0_averages(np.array(l1_prod), np.array(l1_w_av))
            self.print_run(c_run, in_f.out_path + in_f.corr_runs_v1[i])
            averages.append(c_run)

        w_av = self.run_averages(averages)
        self.print_run(w_av, in_f.out_path + "_total")
        self.print_run(self.folding(w_av), in_f.out_path + "_folded")
        log.close()

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
        
    def read_level1_config(self, file_path: str, endian:str='little', version:str='V1', in_f:Input_file=Input_file("")) -> Data_conn:
        if endian == 'little':
            end = '<'
        elif endian == 'big':
            end = '>'
        else:
            print("Wrong specification of endianness")
            exit(1)
        
        f = open(file_path, "rb")

        #Reading header
        tmp = np.fromfile(f, dtype=end + 'i4', count = 5)
        header = {'m0': tmp[0], 'n_source': tmp[1], 'n_op': tmp[2], 't_max': tmp[3], 'n_slice': tmp[4]}
        
        if len(in_f.n_config_l1) != header['n_slice']:
            print("Number of level 1 configurations not correct")
            exit(1)
        
        tot_conf = np.prod(in_f.n_config_l1)
        #Reading configurations
        data = Data_conn(header['n_source'], len(in_f.operators), header['t_max'], tuple(in_f.n_config_l1))
        if version == 'V1':
            op_to_read = np.zeros(header['n_op'])
            for i in in_f.operators:
                op_to_read[i] = 1

            for c in range(tot_conf):
                id_conf = np.array(np.fromfile(f, dtype=end + 'i4', count = header['n_slice'])/int(4) - 1, dtype=int)

                src_pos = np.zeros((header['n_source'], 4))
                for src in range(header['n_source']):
                    src_pos[src] = np.fromfile(f, dtype=end + 'i4', count = 4)
                
                for m in range(header['m0']):
                    for s in range(header['n_source']):
                        counter = 0
                        for op in range(header['n_op']):
                            if op_to_read[op] == 1:
                                for t in range(header['t_max']):
                                    index = tuple(id_conf) +  tuple([s, counter, t])
                                    data.configuration[index] = np.fromfile(f, dtype=end + 'f8', count = 2)[0]
                                counter += 1
                            else:
                                for t in range(header['t_max']):
                                    np.fromfile(f, dtype=end + 'f8', count = 2)
            
                for m in range(header['m0']):
                    for s in range(header['n_source']):
                        for op in range(4):
                            for t in range(header['t_max']):
                                np.fromfile(f, dtype=end + 'f8', count = 2)
        
        elif version == 'V2':
            op_to_read = np.zeros(header['n_op'] + 4)
            for i in in_f.operators:
                op_to_read[i] = 1

            for c in range(tot_conf):
                id_conf = np.array(np.fromfile(f, dtype=end + 'i4', count = header['n_slice'])/int(4) - 1, dtype=int)

                src_pos = np.zeros((header['n_source'], 4))
                for src in range(header['n_source']):
                    src_pos[src] = np.fromfile(f, dtype=end + 'i4', count = 4)

                for m in range(header['m0']):
                    for s in range(header['n_source']):
                        counter = 0
                        for op in range(header['n_op'] + 4):
                            if op_to_read[op] == 1:
                                for t in range(header['t_max']):
                                    index = tuple(id_conf) +  tuple([s, counter, t])
                                    data.configuration[index] = np.fromfile(f, dtype=end + 'f8', count = 2)[0]
                                counter += 1
                            else:
                                for t in range(header['t_max']):
                                    np.fromfile(f, dtype=end + 'f8', count = 2)
        else:
            print("Wrong version number")
            exit(1)
        data.set_y0(int(src_pos[0][0]))
        f.close()
        return data

    def read_level1_weight(self, file_path: str, endian:str='little', in_f:Input_file=Input_file("")) -> Data_weight:
        if endian == 'little':
            end = '<'
        elif endian == 'big':
            end = '>'
        else:
            print("Wrong specification of endianness")
            exit(1)
        
        f = open(file_path, 'rb')

        tmp = np.fromfile(f, dtype=end + 'i4', count = 2)
        n_rw, n_slice = tmp[0], tmp[1]
        n_fct, n_src = np.fromfile(f, dtype=end + 'i4', count = n_rw), np.fromfile(f, dtype=end + 'i4', count = n_rw)

        if len(in_f.n_config_l1) != n_slice:
            print("Number of level 1 configurations not correct")
            exit(1)
        
        tot_conf = np.prod(in_f.n_config_l1)
        data = Data_weight(n_rw, tuple(in_f.n_config_l1))
        for c in range(tot_conf):
            id_conf = np.array(np.fromfile(f, dtype=end + 'i4', count = n_slice)/int(4) - 1, dtype=int)
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
                    tmp1[fct] = np.mean(np.exp(-tmp2))
                index = tuple([rw]) + tuple(id_conf)
                data.configuration[index] = np.prod(tmp1)
        f.close()
        return data
        
    def write_log(self, string:str, f):
        sys.stdout = f
        print(string, flush=True)
        sys.stdout = stdoustream

    def l1_averages(self, d:Data_conn, w:Data_weight, f:Input_file):
        av_op = np.mean(d.configuration, axis=len(f.n_config_l1) + 1, dtype='f8')
        av_src = np.mean(av_op, axis=len(f.n_config_l1), dtype='f8')
        dims = av_src.shape

        #Standard average
        mean = np.zeros((2, len(in_f.std_av), dims[-1]))
        weight = np.zeros((2, len(in_f.std_av)))
        for i in range(len(in_f.std_av)):
            for j in range(len(in_f.std_av[i])):
                for t in range(dims[-1]):
                    mean[0][i][t] += -av_src[int(in_f.std_av[i][j]), int(in_f.std_av[i][j])][t] * w.configuration[0, int(in_f.std_av[i][j]), int(in_f.std_av[i][j])] / len(in_f.std_av[i])
                weight[0][i] += w.configuration[0, int(in_f.std_av[i][j]), int(in_f.std_av[i][j])] / len(in_f.std_av[i])

        for i in range(len(in_f.ml_av)):
            for j in range(len(in_f.ml_av[i])):
                for k in range(len(in_f.ml_av[i])):
                    for t in range(dims[-1]):
                        mean[1][i][t] += -av_src[int(in_f.ml_av[i][j]), int(in_f.ml_av[i][k])][t] * w.configuration[0, int(in_f.ml_av[i][j]), int(in_f.ml_av[i][k])] / (len(in_f.ml_av[i])**2.0)
                    weight[1][i] += w.configuration[0, int(in_f.ml_av[i][j]), int(in_f.ml_av[i][k])] / (len(in_f.ml_av[i])**2.0)
        
        mean = np.roll(mean, -d.y0, axis=2)
        return mean, weight

    def l0_averages(self, prod:np.ndarray, weight:np.ndarray):
        av_type = prod.shape[1]
        l1_set = prod.shape[2]
        times = prod.shape[3]
        
        cav = []
        for av in range(av_type):
            cs = []
            for s in range(l1_set):
                ct = []
                for t in range(times):
                    p = Jacknife(prod[:, av, s, t])
                    w = Jacknife(weight[:, av, s])
                    ct.append(p/w)
                cs.append(ct)
            cav.append(cs)
        
        return cav

    def print_run(self, corr, path):
        n_sets = len(corr[0])
        n_times = len(corr[0][0])
        for s in range(n_sets):
            f = open(path + "_std_{}.txt".format(s + 1), "w")
            sys.stdout = f
            for t in range(n_times):
                print("{} {}".format(corr[0][s][t].mean, np.sqrt(corr[0][s][t].variance())))
            sys.stdout = stdoustream
        f.close()

        for s in range(n_sets):
            f = open(path + "_ml_{}.txt".format(s + 1), "w")
            sys.stdout = f
            for t in range(n_times):
                print("{} {}".format(corr[1][s][t].mean, np.sqrt(corr[1][s][t].variance())))
            sys.stdout = stdoustream
        f.close()
    
    def run_averages(self, runs):
        n_runs = len(runs)
        n_av_type = len(runs[0])
        n_sets = len(runs[0][0])
        n_times = len(runs[0][0][0])
        
        corr = []
        for at in range(n_av_type):
            corr_s = []
            for s in range(n_sets):
                corr_t = []
                for t in range(n_times):
                    sum = Jacknife(np.zeros(runs[0][0][0][0].jack.shape))
                    sum_weight = 0.0
                    for i in range(n_runs):
                        sum = sum + (runs[i][at][s][t] / runs[i][at][s][t].variance())
                        sum_weight += 1/runs[i][at][s][t].variance()
                    corr_t.append(sum/sum_weight)
                corr_s.append(corr_t)
            corr.append(corr_s)

        return corr

    def folding(self, av):
        n_av_type = len(av)
        n_sets = len(av[0])
        n_times = len(av[0][0])

        corr = []
        for at in range(n_av_type):
            corr_s = []
            for s in range(n_sets):
                corr_t = []
                for t in range(int(n_times / 2)):
                    if t == 0:
                        corr_t.append(av[at][s][t])
                    else:
                        corr_t.append((av[at][s][t] + av[at][s][n_times - t]) / 2.0)
                corr_s.append(corr_t)
            corr.append(corr_s)
        
        return corr
        
#Execution
if(len(sys.argv) < 2):
    print("Usage: python3 " + sys.argv[0] + " input_file.in")
    exit(1)

in_f = Input_file(sys.argv[1])
r = Read_connected(in_f)