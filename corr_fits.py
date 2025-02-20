import numpy as np
import scipy.stats as st

class Corr_fits:
    def __init__(self, model:callable, der_list:list, parameters:np.ndarray, x_points:np.ndarray, y_points:np.ndarray, covariance:np.ndarray):
        self.f = model
        self.dl = der_list
        self.par = parameters
        self.x = x_points
        self.y = y_points
        self.cov = covariance
        val, _ = np.linalg.eigh(self.cov)
        if np.any(val < 0):
            print("Warning: covariance not positive definite")

    def error(self):
        err = np.zeros(len(self.x))
        for i in range(len(err)):
            err[i] = np.sqrt(self.cov[i][i])
        
        return err
    
    def exp_chi2(self):
        err = self.error()

        P = self.P_matrix()
        
        sum = 0
        for i in range(len(self.x)):
            for j in range(len(self.x)):
                sum += self.cov[i, j] * P[j, i] / (err[i] * err[j])
        
        return len(self.x) - sum
    
    def H_matrix(self):
        err = self.error()

        H = np.zeros((len(self.par), len(self.par)))
        for p1 in range(len(self.par)):
            for p2 in range(len(self.par)):
                for i in range(len(self.x)):
                    H[p1][p2] += self.dl[p1](self.x[i], self.par) * self.dl[p2](self.x[i], self.par) / (err[i] ** 2.0)

        return H
    
    def P_matrix(self):
        err = self.error()

        H = self.H_matrix()
        
        H_inv = np.linalg.inv(H)

        P = np.zeros((len(self.x), len(self.x)))
        for i in range(len(self.x)):
            for j in range(len(self.x)):
                for alpha in range(len(self.par)):
                    for beta in range(len(self.par)):
                        P[i][j] += (self.dl[alpha](self.x[i], self.par) * H_inv[alpha, beta] * self.dl[beta](self.x[j], self.par)) / (err[i] * err[j])

        return P

    def chi2(self):
        return np.sum((self.y - self.f(self.x, self.par)) ** 2.0/ self.error() ** 2.0)
    
    def p_val(self, n_sample = 1000):
        err = self.error()
        W = np.diag(1/err)
        Id = np.diag(np.full(len(self.cov[:, 0]), 1.0))
        [val, vect] = np.linalg.eigh(self.cov)
        for i in range(len(val)):
            if val[i] < 0.0:
                val[i] = 0.0
        C_sqrt = vect @ np.diag(np.sqrt(val)) @ vect.T

        nu = C_sqrt @ W @ (Id - self.P_matrix()) @ W @ C_sqrt

        nu_val, _ = np.linalg.eigh(nu)
        for i in range(len(nu_val)):
            if nu_val[i] < 1.0e-14:
                nu_val[i] = 0.0
        
        sum = 0
        c = self.chi2()
        for i in range(n_sample):
            z = np.random.normal(0.0, 1.0, len(nu_val))
            
            if z.T @ (nu_val * z) - c > 0:
                sum += 1
        
        return sum / n_sample
    
    def cov_par(self):
        H = self.H_matrix()
        H_inv = np.linalg.inv(H)
        err = self.error()

        prod = np.zeros((len(self.par), len(self.par)))
        for alpha in range(len(self.par)):
            for beta in range(len(self.par)):
                for i in range(len(self.x)):
                    for j in range(len(self.x)):
                        prod[alpha][beta] += self.dl[alpha](self.x[i], self.par) * self.cov[i, j] * self.dl[beta](self.x[j], self.par) / (err[i] ** 2.0 * err[j] ** 2.0)
        
        return H_inv @ prod @ H_inv