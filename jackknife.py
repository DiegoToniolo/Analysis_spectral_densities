import numpy as np
from scipy.optimize import curve_fit

class Jackknife:
    def __init__(self, data:np.ndarray=None):
        if data != None:
            self.mean, self.jack = self.jacknife(data)
        else:
            self.mean = 0
            self.jack = np.zeros(0)
    
    def jacknife(self, a) -> list[float, np.ndarray]:
        mean = np.mean(a, dtype='f8')
        jack = np.zeros(len(a), dtype='f8')
        for i in range(len(a)):
            jack[i] = mean - (a[i] - mean)/(len(a) - 1)
    
        return mean, jack
    
    def variance(self) -> float:
        return (len(self.jack) - 1) * np.var(self.jack)
    
    def covariance(self, var) -> float:
        if not self.iscompatible(var):
            exit(1)
        elif isinstance(var, float):
            exit(1)
        else:
            return (len(self.jack) - 1) * np.sum((self.jack - np.mean(self.jack)) * (var.jack - np.mean(var.jack)), dtype='f8') /len(self.jack)
    
    def iscompatible(self, var) -> bool:
        if not isinstance(var, Jackknife) and not isinstance(var, float):
            print("Non compatible types: Jacknife + " + str(type(var)))
            return False
        elif isinstance(var, Jackknife) and len(self.jack) != len(var.jack):
            print("Jacknife vectors have not equal length: {} and {}".format(len(self.jack), len(var.jack)))
            return False
        else:
            return True

    def __add__(self, var):
        if not self.iscompatible(var):
            exit(1)
        else:
            sum = Jackknife()
            sum.mean = self.mean + var.mean
            sum.jack = self.jack + var.jack
            return sum
    
    def __sub__(self, var):
        if not self.iscompatible(var):
            exit(1)
        else:
            diff = Jackknife()
            diff.mean = self.mean - var.mean
            diff.jack = self.jack - var.jack
            return diff
    
    def __mul__(self, var):
        if not self.iscompatible(var):
            exit(1)
        if isinstance(var, Jackknife):
            prod = Jackknife()
            prod.mean = self.mean * var.mean
            prod.jack = self.jack * var.jack
            return prod
        else:
            prod = Jackknife()
            prod.mean = self.mean * var
            prod.jack = self.jack * var
            return prod
        
    def __truediv__(self, var):
        if not self.iscompatible(var):
            exit(1)
        if isinstance(var, Jackknife):
            quo = Jackknife()
            quo.mean = self.mean / var.mean
            quo.jack = self.jack / var.jack
            return quo
        else:
            quo = Jackknife()
            quo.mean = self.mean / var
            quo.jack = self.jack / var
            return quo

    def der_function(f:callable, list_jack:list):
        res = Jackknife()
        av = np.zeros(0)
        for i in range(len(list_jack)):
            av = np.append(av, list_jack[i].mean)

        res.mean = f(av)
        res.jack = np.zeros(0)
        for i in range(len(list_jack[0].jack)):
            v = np.zeros(0)
            for j in range(len(list_jack)):
                v = np.append(v, list_jack[j].jack[i])
            res.jack = np.append(res.jack, f(v))
        return res
    
    def to_lists(list_jack) -> list[np.ndarray, np.ndarray, np.ndarray]:
        av = np.zeros(0)
        cov = np.zeros((len(list_jack), len(list_jack)))

        for i in range(len(list_jack)):
            av = np.append(av, list_jack[i].mean)
            for j in range(len(list_jack)):
                cov[i, j] = list_jack[i].covariance(list_jack[j])
        
        return av, np.sqrt(np.diagonal(cov)), cov
    
    def fit(f:callable, x:np.ndarray, y_jack:list, p_init:np.ndarray) -> list:
        y_m, y_err, _ = Jackknife.to_lists(y_jack)
        par_m, _ = curve_fit(f, x, y_m, p0 = p_init, sigma = y_err)

        par_jack = []
        for i in range(len(par_m)):
            par_jack.append(Jackknife())
            par_jack[-1].mean = par_m[i]
            par_jack[-1].jack = np.zeros(len(y_jack[0].jack))
        
        for i in range(len(y_jack[0].jack)):
            y = []
            for j in range(len(y_m)):
                y.append(y_jack[j].jack[i])
            pj, _ = curve_fit(f, x, y, p0 = par_m, sigma = y_err)
            for j in range(len(par_jack)):
                par_jack[j].jack[i] = pj[j]
        
        return par_jack
        
