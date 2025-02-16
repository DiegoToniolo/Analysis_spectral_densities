import numpy as np

class Corr_fits:
    def exp_chi2(self, der:list, par:np.ndarray, x:np.ndarray, cov:np.ndarray):
        err = np.zeros(len(x))
        for i in range(len(err)):
            err[i] = np.sqrt(cov[i][i])
        
        H = np.zeros((len(par), len(par)))
        for p1 in range(len(par)):
            for p2 in range(len(par)):
                for i in range(len(x)):
                    H[p1][p2] += der[p1](x[i], par) * der[p2](x[i], par) / (err[i] ** 2.0)
        
        H_inv = np.linalg.inv(H)

        P = np.zeros((len(x), len(x)))

        for i in range(len(x)):
            for j in range(len(x)):
                for alpha in range(len(par)):
                    for beta in range(len(par)):
                        P[i][j] += (der[alpha](x[i], par) * H_inv[alpha, beta] * der[beta](x[j], par)) / (err[i] * err[j])

        sum = 0
        for i in range(len(x)):
            for j in range(len(x)):
                sum += cov[i, j] * P[j, i] / (err[i] * err[j])
        
        return len(x) - sum
    
    def chi2(self, model, par, x, y, err_y):
        return np.sum((y - model(x, par)) ** 2.0/ err_y ** 2.0)