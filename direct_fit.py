import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import sys
from corr_fits import Corr_fits
from jackknife import Jackknife

class Single_exp:
    def fit_f(self, x, C1, m1):
        return C1 * np.exp(-m1 * x)
    
    def f(self, x, par):
        return self.fit_f(x, par[0], par[1])
    
    def der0(self, x, par):
        return np.exp(- par[1] * x)
    
    def der1(self, x, par):
        return - par[0] * par[1] * x * np.exp(- par[1] * x)
    
    def der_list(self):
        return [self.der0, self.der1]

s_exp = Single_exp()

class main:
    def __init__(self, path_data:str, path_jack:str) -> None:
        jack = self.read_jack(path_data, path_jack)
        
        cov_data = np.zeros((len(jack), len(jack)))
        for t_a in range(len(jack)):
            for t_b in range(len(jack)):
                cov_data[t_a, t_b] = jack[t_a].covariance(jack[t_b])

        corr = np.zeros(len(jack))
        err_c = np.zeros(len(jack))
        for t in range(len(jack)):
            corr[t], err_c[t] = jack[t].mean, np.sqrt(cov_data[t, t])

        t = np.array(range(len(corr)))
        for t0 in range(10, 26):
            par, _ = curve_fit(s_exp.fit_f, t[t0:], corr[t0:], p0 = [0.02, 0.2], sigma=err_c[t0:])
            c_f = Corr_fits(s_exp.f, s_exp.der_list(), par, t[t0:], corr[t0:], cov_data[t0:, t0:])
            print(t0, c_f.chi2() / c_f.exp_chi2(), c_f.p_val(10000) * 100)
            
        xgrid = np.linspace(0, t[-1], 1000)
        plt.errorbar(t, corr, err_c, fmt="o", ecolor='black', elinewidth=2, markersize = 4)
        plt.plot(xgrid, s_exp.f(xgrid, par))
        plt.semilogy()
        plt.xlabel(r"$t/a$")
        plt.ylabel(r"Correlator")
        plt.grid(True)
        plt.savefig("plot.png")
        plt.close()

    def read_points(self, path:str):
        df = open(path, "r")
        data = df.readlines()
        c = np.zeros(0)
        err = np.zeros(0)
        for d in data:
            p = d.split()
            if float(p[1]) / float(p[0]) < 0.05: 
                c = np.append(c, float(p[0]))
                err = np.append(err, float(p[1]))
            else:
                break
        
        df.close()
        return c, err
    
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
        
if len(sys.argv) < 3:
    print("Usage: " + sys.argv[0] + " data_file jack_file")
    exit(1)

main(sys.argv[1], sys.argv[2])