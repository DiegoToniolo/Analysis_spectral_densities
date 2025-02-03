import numpy as np
import sys
import subprocess
import re
from matplotlib import pyplot as plt

class Input_file:
    def __init__(self, path: str):
        if path != "":
            self.file_path = path
            self.in_dir = self.read_setting("in_dir")[0]
            self.out_dir = self.read_setting("out_dir")[0]
            s = self.read_setting("err_ratios")
            self.err_ratios = []
            for i in range(len(s)):
                self.err_ratios.append(s[i].split("-"))
            self.err_ratios = np.array(self.err_ratios)
            self.exp_perf = self.read_setting("exp_perf", 'float')
            if len(self.exp_perf) != len(self.err_ratios[:, 0]):
                print("Number of expected performance different from number of ratios")
                exit(1)
            self.op = self.read_setting("op")[0]
            self.reference = self.read_setting("reference")[0]
            self.compared = self.read_setting("compared")
            self.names = self.read_setting("names")
            if len(self.names) != len(self.compared):
                print("Number of compared and names are different")
                exit(1)
            self.perf_ml = self.read_setting("perf_ml", 'float')
            t = self.read_setting("title")
            self.title = t[0]
            for i in range(1, len(t)):
                self.title += " " + t[i]
            self.plot_name = self.read_setting("plot_name")[0]
            self.y_range = self.read_setting("y_range", 'float')
    
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
        
        return tokens

#Execution
class main:
    def __init__(self):
        if(len(sys.argv) < 2):
            print("Usage: python3 " + sys.argv[0] + " input_file.in")
            exit(1)
        in_f = Input_file(sys.argv[1])
        dat_files = self.data_to_read(in_f)
        for df in dat_files:
            av = np.zeros(0)
            err = np.zeros(0)

            f = open(in_f.in_dir + df, "r")
            lines = f.readlines()
            for l in lines:
                av = np.append(av, float(l.split()[0]))
                err = np.append(err, float(l.split()[1]))
            plt.errorbar(np.array(range(len(av))), av, err, fmt="o", markersize=4)
            plt.grid(True)
            plt.xlabel("t/a")
            plt.ylabel(r"$C(t)$")
            plt.title("Correlator of the " + in_f.op + " current")
            plt.semilogy()
            plt.savefig(in_f.out_dir + df.split(".")[0] + ".png")
            plt.close()
            f.close()

        for i in range(len(in_f.err_ratios)):
            err = np.zeros(0)
            f1 = open(in_f.in_dir + in_f.err_ratios[i, 0])
            f2 = open(in_f.in_dir + in_f.err_ratios[i, 1])
            l1 = f1.readlines()
            l2 = f2.readlines()
            if len(l1) != len(l2):
                print("Data of different length in err_ratios block {}".format(i+1))
                exit(1)
            for l in range(len(l1)):
                err = np.append(err, float((l1[l].split())[1]) ** 2.0/float((l2[l].split())[1]) ** 2.0)
            plt.plot(np.array(range(len(err))), err, "o", markersize=4, label="Error ratio")
            x_grid = np.linspace(0, len(err), 1000)
            y_grid = np.zeros(len(x_grid))
            y_grid.fill(in_f.exp_perf[i])
            plt.grid()
            plt.xlabel("t/a")
            plt.ylabel(r"$\sigma_{\mathrm{std}}^2/\sigma_{\mathrm{ml}}^2$ of the " + in_f.op + " current")
            plt.plot(x_grid, y_grid, "black", linestyle="--", linewidth=3, label="Expected performance")
            plt.legend()
            plt.title("Variance ratio between Standard and Multilevel averages")
            plt.savefig(in_f.out_dir + "err_ratio_" + in_f.err_ratios[i, 0].split(".")[0] + "_" + in_f.err_ratios[i, 1].split(".")[0])
            plt.close()
            f1.close()
            f2.close()
        
        for i in range(len(in_f.compared)):
            x_grid = np.linspace(0, len(err), 1000)
            y_grid = np.zeros(len(x_grid))
            y_grid.fill(1.0/in_f.perf_ml[i])
            plt.plot(x_grid, y_grid, "black", linestyle="--", linewidth=2)
        for i in range(len(in_f.compared)):
            err = np.zeros(0)
            f_ref = open(in_f.in_dir + in_f.reference)
            f_comp = open(in_f.in_dir + in_f.compared[i])
            l_ref = f_ref.readlines()
            l_comp = f_comp.readlines()
            if len(l_ref) != len(l_comp):
                print("Plot of ML performance: reference and compared have different number of times")
                print(len(l_ref), len(l_comp))
                exit(1)
            for l in range(len(l_ref)):
                err = np.append(err, float(l_comp[l].split()[1]) ** 2.0 / float(l_ref[l].split()[1]) ** 2.0)
            plt.plot(np.array(range(len(l_ref))), err, "o", markersize=4, label=r"$n_1 = $" + in_f.names[i])
        plt.grid()
        plt.semilogy()
        plt.title(in_f.title)
        plt.xlabel("t/a")
        plt.ylabel("Ratio of variances")
        plt.legend(loc="upper left")
        plt.ylim(in_f.y_range[0], in_f.y_range[1])
        plt.savefig(in_f.out_dir + in_f.plot_name)
        plt.close()
        
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