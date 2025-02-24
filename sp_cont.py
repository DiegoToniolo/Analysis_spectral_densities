import numpy as np
from scipy.special import gamma
from scipy.integrate import quad
from matplotlib import pyplot as plt

def g_integrand(x:complex, t:float, omega:float , alpha:float = 1.0e-15) -> float:
    return (np.exp(-1.0j * x * np.log(t*omega)) * gamma(0.5 + x * 1.0j)/(alpha + np.pi/np.cosh(np.pi * x)) / (2.0 * np.pi * np.sqrt(t*omega))).real

def g(t:float, omega:float , alpha:float = 1.0e-15):
    i = 0.0
    while np.abs(g_integrand(i, t, omega, alpha)) > 1.0e-7:
        i += 0.5
    
    return quad(lambda x: 2.0 * np.sqrt(alpha) * g_integrand(x, t, omega, alpha), 0.0, i)

#xgrid = np.logspace(-1, 2, 1000)
#ygrid = []
#for x in xgrid:
#    ygrid.append(g(x, 0.5)[0])
#plt.plot(xgrid, ygrid)
#plt.semilogx()
#plt.savefig("plot.png")