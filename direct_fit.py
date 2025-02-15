import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import sys

class main:
    def __init__(self, path:str) -> None:
        df = open(sys.argv[1], "r")
        
        corr, err_c = self.read_points(df)
        t = np.array(range(len(corr)))
        par, cov = curve_fit(self.expr_corr_cond, t[3:], corr[3:], p0 = [ 0.02, 0.75], sigma=err_c[3:])
        err_par = np.zeros(0)
        for i in range(len(par)):
            err_par = np.append(err_par, np.sqrt(cov[i, i]))
        print(par)
        print(err_par)
        
        xgrid = np.linspace(0, t[-1], 1000)
        plt.errorbar(t, corr, err_c, fmt="o", ecolor='black', elinewidth=2, markersize = 4)
        plt.plot(xgrid, self.expr_corr_cond(xgrid, par[0], par[1]))
        plt.semilogy()
        plt.xlabel(r"$t/a$")
        plt.ylabel(r"Correlator")
        plt.grid(True)
        plt.savefig("plot.png")
        plt.close()
        df.close()

    def read_points(self, data_file:__file__):
        data = data_file.readlines()
        c = np.zeros(0)
        err = np.zeros(0)
        for d in data:
            p = d.split()
            if float(p[1]) / float(p[0]) < 0.10: 
                c = np.append(c, float(p[0]))
                err = np.append(err, float(p[1]))
            else:
                break
        
        return c, err
    
    def single_exp(self, x, C1, m1):
        return C1 * np.exp(-m1 * x)
    
    def expr_corr(self, x, C1, m1, C2, m2):
        return C1 * np.exp(-m1 * x) + C2 * np.exp(-m2 * x)
    
    def expr_corr_cond(self, x, C2, m2):
        return self.expr_corr(x, 0.001539, 0.28189, C2, m2)
        
if len(sys.argv) < 2:
    print("Usage: " + sys.argv[0] + " data_file")
    exit(1)

main(sys.argv[1])