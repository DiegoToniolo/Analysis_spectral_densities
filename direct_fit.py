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

class Double_exp:
    def fit_f(self, x, C1, m1, C2, m2):
        return C1 * np.exp(-m1 * x) + C2 * np.exp(-m2 * x)
    
    def f(self, x, par):
        return self.fit_f(x, par[0], par[1], par[2], par[3])
    
    def der0(self, x, par):
        return np.exp(- par[1] * x)
    
    def der1(self, x, par):
        return - par[0] * x * np.exp(- par[1] * x)
    
    def der2(self, x, par):
        return np.exp(- par[3] * x)
    
    def der3(self, x, par):
        return - par[2] * x * np.exp(- par[3] * x)
    
    def der_list(self):
        return [self.der0, self.der1, self.der2, self.der3]

d_exp = Double_exp()

class main:
    def __init__(self, path_data:str, path_jack:str, path_out1:str, path_out2:str) -> None:
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
        out = open(path_out1, "w")
        for t0 in range(10, 26):
            print("T0 = {}".format(t0))
            par, _ = curve_fit(s_exp.fit_f, t[t0:], corr[t0:], p0 = [0.02, 0.2], sigma=err_c[t0:])
            c_f = Corr_fits(s_exp.f, s_exp.der_list(), par, t[t0:], corr[t0:], cov_data[t0:, t0:])
            cov_par = c_f.cov_par()
            print(t0, par[0], np.sqrt(cov_par[0, 0]), par[1], np.sqrt(cov_par[1, 1]), c_f.p_val(10000) * 100.0, file=out)
            xgrid = np.linspace(0, t[-1], 1000)
            plt.errorbar(t, corr, err_c, fmt="o", ecolor='black', elinewidth=2, markersize = 4)
            plt.plot(xgrid, s_exp.f(xgrid, par))
            plt.semilogy()
            plt.xlabel(r"$t/a$")
            plt.ylabel(r"Correlator")
            plt.grid(True)
            plt.title(r"Fit of the correlator, $C(t) = C_1 e^{-m_1\,t}$")
            plt.savefig("out/vector/plots/fits/single_t0_{}.png".format(t0))
            plt.close()
        out.close()

        out = open(path_out2, "w")
        for t0 in range(0, 21):
            print("T0 = {}".format(t0))
            par, _ = curve_fit(d_exp.fit_f, t[t0:], corr[t0:], p0 = [0.02, 0.2, 0.02, 0.5], sigma=err_c[t0:])
            c_f = Corr_fits(d_exp.f, d_exp.der_list(), par, t[t0:], corr[t0:], cov_data[t0:, t0:])
            cov_par = c_f.cov_par()
            if t0 == 8:
                f = open("out/vector/data/fits/cov_b.txt", "w")
                for i in range(len(cov_par[:, 0])):
                    for j in range(len(cov_par[:, 0])):
                        print(cov_par[i, j], file=f)
                f.close()
                        
            print(t0, par[0], np.sqrt(cov_par[0, 0]), par[1], np.sqrt(cov_par[1, 1]), end=" ", file=out)
            print(par[2], np.sqrt(cov_par[2, 2]), par[3], np.sqrt(cov_par[3, 3]), c_f.p_val(10000) * 100.0, file=out)
            plt.errorbar(t, corr, err_c, fmt="o", ecolor='black', elinewidth=2, markersize = 4)
            plt.plot(xgrid, d_exp.f(xgrid, par))
            plt.semilogy()
            plt.xlabel(r"$t/a$")
            plt.ylabel(r"Correlator")
            plt.grid(True)
            plt.title(r"Fit of the correlator, $C(t) = C_1 e^{-m_1\,t} + C_2 e^{-m_2\,t}$")
            plt.savefig("out/vector/plots/fits/double_t0_{}.png".format(t0))
            plt.close()
        out.close()

    def read_points(self, path:str):
        df = open(path, "r")
        data = df.readlines()
        c = np.zeros(0)
        err = np.zeros(0)
        for d in data:
            p = d.split()
            if float(p[1]) / float(p[0]) < 0.10: 
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
        t_max = 0
        for line in d:
            if float(line.split()[1])/float(line.split()[0]) > 0.30:
                print("t_max = {}".format(t_max-1))
                break
            t_max += 1 
            jack = np.zeros(25)
            for i in range(25):
                jack[i]  = float(jf.readline())
            data.append(Jackknife())
            data[-1].mean = float((line.split())[0])
            data[-1].jack = jack
        
        df.close()
        jf.close()
        return data
        
if len(sys.argv) < 5:
    print("Usage: " + sys.argv[0] + " data_file jack_file out_file_single out_file_double")
    exit(1)

main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])