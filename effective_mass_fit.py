import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import sys
from jackknife import Jackknife
from corr_fits import Corr_fits

class Constant:
    def fit_f(self, x, m1):
        return np.full(x.shape, m1)
    
    def f(self, x, par):
        return self.fit_f(x, par[0])
    
    def der0(self, x, par):
        return 1.0
    
    def der_list(self):
        return [self.der0]

const = Constant()

class Exponential:
    def fit_f(self, x, m1, A, B):
        return m1 + A * np.exp(-B * x)
    
    def f(self, x, par):
        return self.fit_f(x, par[0], par[1], par[2])
    
    def der0(self, x, par):
        return 1.0
    
    def der1(self, x, par):
        return np.exp(-par[2] * x)
    
    def der2(self, x, par):
        return - par[1] * par[2] * x * np.exp(-par[2] * x)
    
    def der_list(self):
        return [self.der0, self.der1, self.der2]
    
exp = Exponential()

class main:
    def __init__(self, pd:str, pj:str, po1:str, po2:str):
        c = self.read_jack(pd, pj)
        em = np.zeros(0)
        err_em = np.zeros(0)
        em_jack = []
        for t in range(2, len(c) - 1):
            em_jack.append(((c[t + 1] + c[t - 1])/(c[t] * 2.0)).der_function(np.acosh))
            em = np.append(em, em_jack[-1].mean)
            err_em = np.append(err_em, np.sqrt(em_jack[-1].variance()))
        
        cov_data = np.zeros((len(em_jack), len(em_jack)))
        for t_a in range(len(em_jack)):
            for t_b in range(len(em_jack)):
                cov_data[t_a, t_b] = em_jack[t_a].covariance(em_jack[t_b])

        out = open(po1, "w")
        t = np.array(range(2, len(c) - 1))
        for t0 in range(10, 26):
            print("T0 = {}".format(t0))
            par, _ = curve_fit(const.fit_f, t[t0:], em[t0:], p0 = [0.2], sigma=err_em[t0:])
            c_f = Corr_fits(const.f, const.der_list(), par, t[t0:], em[t0:], cov_data[t0:, t0:])
            cov_par = c_f.cov_par()
            print(t0, par[0], np.sqrt(cov_par[0, 0]), c_f.p_val(10000) * 100.0, file=out)
            xgrid = np.linspace(0, t[-1], 1000)
            plt.errorbar(np.array(range(2, len(c) - 1)), em, err_em, fmt = "o", markersize=4)
            plt.plot(xgrid, const.f(xgrid, par))
            plt.xlabel("t/a")
            plt.ylabel("Effective mass")
            plt.grid()
            plt.semilogy()
            plt.title(r"Fit of the effective mass, $M(t) = m_1$")
            plt.savefig("out/vector/plots/fits/em_const_t0_{}.png".format(t0))
            plt.close()
        out.close()

        out = open(po2, "w")
        t = np.array(range(2, len(c) - 1))
        for t0 in range(0, 21):
            print("T0 = {}".format(t0))
            par, _ = curve_fit(exp.fit_f, t[t0:], em[t0:], p0 = [0.2, 2.0, 0.3], sigma=err_em[t0:])
            c_f = Corr_fits(exp.f, exp.der_list(), par, t[t0:], em[t0:], cov_data[t0:, t0:])
            cov_par = c_f.cov_par()
            print(t0, par[0], np.sqrt(cov_par[0, 0]), end=" ", file=out)
            print(par[0] + par[2], np.sqrt(cov_par[0, 0] + cov_par[2, 2] + 2.0 * cov_par[0, 2]), c_f.p_val(10000) * 100.0, file=out)
            xgrid = np.linspace(0, t[-1], 1000)
            plt.errorbar(np.array(range(2, len(c) - 1)), em, err_em, fmt = "o", markersize=4)
            plt.plot(xgrid, exp.f(xgrid, par))
            plt.xlabel("t/a")
            plt.ylabel("Effective mass")
            plt.grid()
            plt.semilogy()
            plt.title(r"Fit of the effective mass, $M(t) = m_1 + A e^{-\Delta m\,t}$")
            plt.savefig("out/vector/plots/fits/em_exp_t0_{}.png".format(t0))
            plt.close()
        out.close()

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


if len(sys.argv) < 5:
    print("Usage " + sys.argv[0] + " mean_file jack_file out_file_const out_file_exp")
    exit(0)

main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])