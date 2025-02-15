import numpy as np

class Jackknife:
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

