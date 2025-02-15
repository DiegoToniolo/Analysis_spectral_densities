import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import sys
from jackknife import Jackknife

class main:
    def __init__(self, pd:str, pj:str):
        c = self.read_jack(pd, pj)

        em = np.zeros(0)
        err_em = np.zeros(0)
        for t in range(2, len(c) - 1):
            res = ((c[t + 1] + c[t - 1])/(c[t] * 2.0)).der_function(np.cosh)
            em = np.append(em, res.mean)
            err_em = np.append(err_em, np.sqrt(res.variance()))
        
        plt.errorbar(np.array(range(2, len(c) - 1)), em, err_em, fmt = "o", markersize=4)
        plt.xlabel("t/a")
        plt.ylabel("Effective mass")
        plt.savefig("plot_em.png")
        plt.close()

    def read_jack(self, path_mean, path_jack):
        df = open(path_mean, "r")
        jf = open(path_jack, "r")

        d = df.readlines()

        data = []
        for line in d:
            if float(line.split()[1])/float(line.split()[0]) > 0.10:
                break
            jack = np.zeros(25)
            for i in range(25):
                jack[i]  = float(jf.readline())
            data.append(Jackknife())
            data[-1].mean = float((line.split())[0])
            data[-1].jack = jack
        
        df.close()
        jf.close()
        return data


if len(sys.argv) < 2:
    print("Usage " + sys.argv[0] + " mean_file jack_file")
    exit(0)

main(sys.argv[1], sys.argv[2])