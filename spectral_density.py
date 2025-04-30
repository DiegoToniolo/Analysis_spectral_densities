import numpy as np
import mpmath as mpmath
from scipy.integrate import quad
from scipy.stats import norm
from scipy.special import gamma
from scipy.special import erfc

#Class for the Backus-Gilbert spectral density
class SDnum:
    #Global variables
    t_max = mpmath.mpf(1)
    w0 = mpmath.mpf(0)
    alpha = mpmath.mpf(0)
    w1 = mpmath.mpf(0)
    sigma = mpmath.mpf(0)
    T = mpmath.mpf(0)
    computed = False
    periodic = False
    
    A_inv = mpmath.matrix(int(t_max), int(t_max))
    f = mpmath.matrix(np.zeros(int(t_max)))

    def __init__(self, t_max_new:int = 1, w0_new:float = 0, alpha_new:float = 0, w1_new:float = 0, sigma_new:float = 0, periodic:bool = False, T_new = 0):
        #MP conversion
        t_max_new = mpmath.mpf(t_max_new)
        w0_new = mpmath.mpf(str(w0_new))
        alpha_new = mpmath.mpf(str(alpha_new))
        w1_new = mpmath.mpf(str(w1_new))
        sigma_new = mpmath.mpf(str(sigma_new))
        T_new = mpmath.mpf(T_new)

        SDnum.periodic = periodic
        if not SDnum.computed:
            SDnum.t_max = t_max_new
            SDnum.w0 = w0_new
            SDnum.alpha = alpha_new
            SDnum.w1 = w1_new
            SDnum.sigma = sigma_new
            SDnum.T = T_new

            SDnum.A_inv = self.A_alpha() ** (-1)
            SDnum.f = self.f_vect()
            SDnum.computed = True
        
        else:
            if (SDnum.w0 != w0_new or SDnum.t_max != t_max_new or SDnum.T != T_new):
                SDnum.computed = False
                SDnum.t_max = t_max_new
                SDnum.w0 = w0_new
                SDnum.alpha = alpha_new
                SDnum.w1 = w1_new
                SDnum.sigma = sigma_new
                SDnum.T = T_new

                SDnum.A_inv = self.A_alpha() ** (-1)
                SDnum.f = self.f_vect()
                SDnum.computed = True

            else:
                if (SDnum.alpha != alpha_new):
                    SDnum.computed = False
                    SDnum.t_max = t_max_new
                    SDnum.w0 = w0_new
                    SDnum.alpha = alpha_new

                    SDnum.A_inv = self.A_alpha() ** -1
                    SDnum.computed = True
            
                if (SDnum.w1 != w1_new or SDnum.sigma != sigma_new):
                    SDnum.computed = False
                    SDnum.w1 = w1_new
                    SDnum.sigma = sigma_new

                    SDnum.f = self.f_vect()
                    SDnum.computed = True

    def int_basis(self, a):
        w0 = SDnum.w0
        return (w0**mpmath.mpf('4.0')/a + mpmath.mpf('4.0') * w0**mpmath.mpf('3.0')/a**mpmath.mpf('2.0') + \
                mpmath.mpf('12.0') * w0** mpmath.mpf('2.0')/a**mpmath.mpf('3.0') +\
                mpmath.mpf('24.0') * w0/a**mpmath.mpf('4.0') + mpmath.mpf('24.0')/a**mpmath.mpf('5')) * mpmath.exp(-a*w0)

    def A_func(self, t, r, T):
        if SDnum.periodic:
            return self.int_basis(t+r+2) + self.int_basis(T+t-r) + self.int_basis(T-t+r) + self.int_basis(2*(T-1)-t-r)
        return self.int_basis(t+r+2)

    def A_alpha(self):
        t_max = SDnum.t_max
        alpha = SDnum.alpha

        if SDnum.T == mpmath.mpf(0):
            T = mpmath.mpf(2) * t_max
        else:
            T = SDnum.T

        A = mpmath.matrix(int(t_max), int(t_max))
        
        for i in range(int(t_max)):
            for j in range(int(t_max)):
                A[i, j] = self.A_func(mpmath.mpf(i), mpmath.mpf(j), T)
        
        return A + alpha * mpmath.eye(int(t_max))

    def f_integral(self, w1, a, sigma):
        w0 = SDnum.w0

        res = mpmath.exp(-a*w0 - (w0-w1)**mpmath.mpf(2)/(mpmath.mpf(2)*sigma**mpmath.mpf(2)))/(mpmath.mpf(2)*mpmath.sqrt(mpmath.pi))*\
            (mpmath.sqrt(mpmath.mpf(2))*sigma*(w0+w1-a*sigma**mpmath.mpf(2)) + \
            mpmath.sqrt(mpmath.pi) * (sigma**mpmath.mpf(2) + (w1-a*sigma**mpmath.mpf(2))**mpmath.mpf(2))*\
            mpmath.exp((w0-w1+a*sigma**mpmath.mpf(2))**mpmath.mpf(2)/(mpmath.mpf(2)*sigma**mpmath.mpf(2)))
            *mpmath.erfc((w0-w1+a*sigma**mpmath.mpf(2))/(mpmath.sqrt(mpmath.mpf(2))*sigma)))
        
        return res

    def f_func(self, w1, t, T, sigma):
        if SDnum.periodic:
            return self.f_integral(w1, t+mpmath.mpf(1), sigma) + self.f_integral(w1, T-t-mpmath.mpf(1), sigma)
        return self.f_integral(w1, t+mpmath.mpf(1), sigma)
    
    def f_vect(self):
        t_max = SDnum.t_max
        w1 = SDnum.w1
        sigma = SDnum.sigma
        
        if SDnum.T == mpmath.mpf(0):
            T = mpmath.mpf(2) * t_max
        else:
            T = SDnum.T

        f = mpmath.matrix(np.zeros(int(t_max)))
        for i in range(len(f)):
            f[i] = self.f_func(w1, mpmath.mpf(i), T, sigma)

        return f
    
    def g(self):
        return SDnum.A_inv *  SDnum.f
    
    def rho(self, corr):
        g = self.g()
        if g.rows < len(corr):
            print("len g less than len corr")
            return 0
        g.rows = len(corr)
        corr = mpmath.matrix(corr)
        return (g.T * corr)[0, 0]
    
    def err_rho(self, cov):
        J = self.g()
        if J.rows < len(cov[0, :]):
            print("len g less than len corr")
            return 0
        J.rows = len(cov[0, :])
        cov = mpmath.matrix(cov)
        return mpmath.sqrt((J.T * cov * J)[0, 0])
    
    def basis_T(self, t, w):
        T = SDnum.T
        if SDnum.periodic:
            return w ** 2.0 * (mpmath.exp(-t*w) + mpmath.exp(-(T-t)*w))
        else:
            return w ** 2.0 * mpmath.exp(-t*w)
    
    def delta(self, w, lenght, trunc = 0):
        A_inv  = SDnum.A_inv

        if trunc != 0:
            t1 = trunc
        else:
            t1 = int(SDnum.t_max)
        sum = mpmath.mpf(0)
        for i in range(t1):
            for j in range(lenght):
                sum += self.basis_T(mpmath.mpf(i+1), w) * self.A_inv[i, j] * self.basis_T(mpmath.mpf(j+1), SDnum.w1)
        return sum
        

#Class for the spectral density
class SDnum_np:
    #Global variables
    t_max = 1
    w0 = 0
    alpha = 0
    w1 = 0
    sigma = 0 
    T = 0
    computed = False
    periodic = False
    
    A_inv = np.zeros((int(t_max), int(t_max)))
    f = np.zeros(int(t_max))

    def __init__(self, t_max_new:int = 1, w0_new:float = 0, alpha_new:float = 0, w1_new:float = 0, sigma_new:float = 0, periodic:bool = False, T_new = 0):
        SDnum_np.periodic = periodic
        if not SDnum_np.computed:
            SDnum_np.t_max = t_max_new
            SDnum_np.w0 = w0_new
            SDnum_np.alpha = alpha_new
            SDnum_np.w1 = w1_new
            SDnum_np.sigma = sigma_new
            SDnum_np.T = T_new

            SDnum_np.A_inv = np.linalg.inv(self.A_alpha())
            SDnum_np.f = self.f_vect()
            SDnum_np.computed = True
        
        else:
            if (SDnum_np.w0 != w0_new or SDnum_np.t_max != t_max_new or SDnum_np.T != T_new):
                SDnum_np.computed = False
                SDnum_np.t_max = t_max_new
                SDnum_np.w0 = w0_new
                SDnum_np.alpha = alpha_new
                SDnum_np.w1 = w1_new
                SDnum_np.sigma = sigma_new
                SDnum_np.T = T_new

                SDnum_np.A_inv = np.linalg.inv(self.A_alpha())
                SDnum_np.f = self.f_vect()
                SDnum_np.computed = True

            else:
                if (SDnum_np.alpha != alpha_new):
                    SDnum_np.computed = False
                    SDnum_np.t_max = t_max_new
                    SDnum_np.w0 = w0_new
                    SDnum_np.alpha = alpha_new

                    SDnum_np.A_inv = np.linalg.inv(self.A_alpha())
                    SDnum_np.computed = True
            
                if (SDnum_np.w1 != w1_new or SDnum_np.sigma != sigma_new):
                    SDnum_np.computed = False
                    SDnum_np.w1 = w1_new
                    SDnum_np.sigma = sigma_new

                    SDnum_np.f = self.f_vect()
                    SDnum_np.computed = True

    def int_basis(self, a):
        w0 = SDnum_np.w0
        return (w0**4./a + 4. * w0**3./a**2. + 12. * w0** 2./a**3. + 24. * w0/a**4. + 24./a**5.) * np.exp(-a*w0)

    def A_func(self, t, r, T):
        if SDnum_np.periodic:
            return self.int_basis(t+r+2) + self.int_basis(T+t-r) + self.int_basis(T-t+r) + self.int_basis(2*(T-1)-t-r)
        return self.int_basis(t+r+2)

    def A_alpha(self):
        t_max = SDnum_np.t_max
        alpha = SDnum_np.alpha

        if SDnum_np.T == 0.:
            T = 2. * t_max
        else:
            T = SDnum_np.T

        A = np.zeros((int(t_max), int(t_max)))
        
        for i in range(int(t_max)):
            for j in range(int(t_max)):
                A[i, j] = self.A_func(i, j, T)
        
        return A + alpha * np.identity(int(t_max))

    def f_integral(self, w1, a, sigma):
        w0 = SDnum_np.w0

        res = float(mpmath.exp(-a*w0 - (w0-w1)**2./(2.*sigma**2.))/(2.*mpmath.sqrt(np.pi))*(mpmath.sqrt(2.)*sigma*(w0+w1-a*sigma**2.) + \
            mpmath.sqrt(mpmath.pi) * (sigma**2. + (w1-a*sigma**2.)**2.)*mpmath.exp((w0-w1+a*sigma**2.)**2./(2.*sigma**2.))
            *mpmath.erfc((w0-w1+a*sigma**2.)/(np.sqrt(2.)*sigma))))
        
        return res

    def f_func(self, w1, t, T, sigma):
        if SDnum_np.periodic:
            return self.f_integral(w1, t+1, sigma) + self.f_integral(w1, T-t-1, sigma)
        return self.f_integral(w1, t+1, sigma)
    
    def f_vect(self):
        t_max = SDnum_np.t_max
        w1 = SDnum_np.w1
        sigma = SDnum_np.sigma
        
        if SDnum_np.T == 1:
            T = 2. * t_max
        else:
            T = SDnum_np.T

        f = np.zeros(int(t_max))
        for i in range(len(f)):
            f[i] = self.f_func(w1, i, T, sigma)

        return f
    
    def g(self):
        return SDnum_np.A_inv @ SDnum_np.f
    
    def rho(self, corr):
        return (self.g()[:len(corr)].T @ corr)
    
    def err_rho(self, cov):
        J = self.g()
        return np.sqrt((J[:len(cov[:, 0])].T @ cov @ J[:len(cov[:, 0])]))
    
    def basis_T(self, t, w):
        T = SDnum_np.T
        if SDnum_np.periodic:
            return w ** 2.0 * (np.exp(-t*w) + np.exp(-(T-t)*w))
        else:
            return w ** 2.0 * np.exp(-t*w)
    
    def delta(self, w, lenght, trunc = 0):
        A_inv  = SDnum_np.A_inv

        if trunc != 0:
            t1 = trunc
        else:
            t1 = int(SDnum_np.t_max)
        sum = 0
        for i in range(t1):
            for j in range(lenght):
                sum += self.basis_T(i+1, w) * self.A_inv[i, j] * self.basis_T(j+1, SDnum_np.w1)
        return sum 

#Class for the analytic spectral density
class SDan:
    def u_s(self, s, w):
        return (w**(-0.5 + s*1.0j)/np.sqrt(2*np.pi))

    def H_a(self, s, a):
        return np.pi/(np.pi + a*np.cosh(np.pi * s))

    def delta(self, w1, w2, a):
        f = lambda x: (2.0 * self.u_s(x, w1).conjugate() * self.H_a(x, a) * self.u_s(x, w2)).real
        return quad(f, 0.0, +np.inf)[0]

    def rho(self, w, par, a):
        return par[0] * self.delta(w, par[1], a) / par[1]**2.0 + par[2] * \
            self.delta(w, par[3], a) / par[3]**2.0

    def r_a_d_C0(self, w, par, a):
        return self.delta(w, par[1], a)/(par[1]**2.0)

    def r_a_d_C1(self, w, par, a):
        return self.delta(w, par[3], a)/(par[3]**2.0)

    def delta_d_m(self, w, m, a):
        f = lambda x: ((-1.0 + x *2.0j) * self.u_s(x, w).conjugate() * self.H_a(x, a) * self.u_s(x, m) / m).real
        return quad(f, 0.0, +np.inf)[0]

    def r_a_d_m0(self, w, par, a):
        return par[0] * (-2.0/(par[1]**3.0) * self.delta(w, par[1], a) + \
                         self.delta_d_m(w, par[1], a) / par[1]**2.0)

    def r_a_d_m1(self, w, par, a):
        return par[2] * (-2.0/(par[3]**3.0) * self.delta(w, par[3], a) + self.delta_d_m(w, par[3], a) / par[3]**2.0)

    def jac_r_a(self, w, par, a):
        return np.array([self.r_a_d_C0(w, par, a), self.r_a_d_m0(w, par, a), self.r_a_d_C1(w, par, a), self.r_a_d_m1(w, par, a)])

    def err_rho(self, w, par, a, cov):
        return np.sqrt(self.jac_r_a(w, par, a).T @ cov @ self.jac_r_a(w, par, a))

#Class for the analytic smeared spectral density
class SDan_sm:
    #Alpha == 0
    def rho_na(self, w, par, sigma):
        k = lambda x: norm(w, sigma).pdf(x)
        return par[0]/(par[1]**2.0)*k(par[1]) + par[2]/(par[3]**2.0)*k(par[3])

    def der_C0(self, omega, par, sigma):
        return norm(par[1], sigma).pdf(omega) * 1/(par[1]**2.0)

    def der_m0(self, omega, par, sigma):
        return par[0] * (-2.0 / (par[1]**3.0) + (omega - par[1])/(sigma**2.0 * par[1]**2.0)) * norm(omega, sigma).pdf(par[1])

    def der_C1(self, omega, par, sigma):
        return norm(par[3], sigma).pdf(omega)/(par[3]**2.0)

    def der_m1(self, omega, par, sigma):
        return par[2] * (-2.0 / (par[3]**3.0) + (omega - par[3])/(sigma**2.0 * par[3]**2.0)) * norm(omega, sigma).pdf(par[3])

    def jacobian(self, omega, par, sigma):
        return np.array([self.der_C0(omega, par, sigma), self.der_m0(omega, par, sigma), self.der_C1(omega, par, sigma), self.der_m1(omega, par, sigma)])

    def err_rho_na(self, omega, par, sigma, cov):
        return np.sqrt(self.jacobian(omega, par, sigma).T @ cov @ self.jacobian(omega, par, sigma))
    #Alpha != 0
    def u_s(self, s, w):
        return (w**(-0.5 + s*1.0j)/np.sqrt(2*np.pi))

    def H_a(self, s, a):
        return np.pi/(np.pi + a*np.cosh(np.pi * s))
    def KI(self, s, w, sigma):
        h1 = complex(mpmath.hyp1f1(0.25 - s * 0.5j, 0.5, -w**2.0/(2*sigma**2.0)))
        h2 = complex(mpmath.hyp1f1(0.75 - s * 0.5j, 1.5, -w**2.0/(2*sigma**2.0)))
        if abs(h1) == float('inf') or abs(h2) == float('inf'):
            return 0.0+0.0j
        return (sigma * gamma(0.25 + s * 0.5j) * h1 + np.sqrt(2) * w * gamma(0.75 + s * 0.5j) * h2) *\
           (2.0**(-1.75 + s * 0.5j)) * (sigma ** (-1.5 + s * 1.0j)) / np.pi

    def kernel(self, w1, w2, a, s):
        f = lambda x: (2.0 * self.u_s(x, w1).conjugate() * self.H_a(x, a) * self.KI(x, w2, s)).real
        return quad(f, 0.0, +np.inf)[0]

    def rho_k_a(self, w, par, a, s):
        return (par[2] * self.kernel(par[3], w, a, s)/(par[3]**2.0) + par[0] * \
            self.kernel(par[1], w, a, s)/(par[1]**2.0))

    def kernel_d_m(self, m, w, a, s):
        f = lambda x: (2.0 * (-0.5 - x * 1.0j) * self.u_s(x, m).conjugate() * self.H_a(x, a) *\
                       self.KI(x, w, s)/m).real
        return quad(f, 0.0, +np.inf)[0]

    def r_k_d_C0(self, w, par, a, s):
        return self.kernel(par[1], w, a, s)/par[1]**2.0

    def r_k_d_C1(self, w, par, a, s):
        return self.kernel(par[3], w, a, s)/par[3]**2.0

    def r_k_d_m0(self, w, par, a, s):
        return par[0] * (-2.0/(par[1]**3.0) * self.kernel(par[1], w, a, s) + \
                         self.kernel_d_m(par[1], w, a, s) /(par[1] ** 2.0))

    def r_k_d_m1(self, w, par, a, s):
        return par[2] * (-2.0/(par[3]**3.0) * self.kernel(par[3], w, a, s) + \
                         self.kernel_d_m(par[3], w, a, s) /(par[3] ** 2.0))

    def jac_r_k(self, w, par, a, s):
        return np.array([self.r_k_d_C0(w, par, a, s), self.r_k_d_m0(w, par, a, s), self.r_k_d_C1(w, par, a, s), self.r_k_d_m1(w, par, a, s)])

    def err_r_k_a(self, w, par, a, s, cov):
        return np.sqrt((self.jac_r_k(w, par, a, s).T @ cov @ self.jac_r_k(w, par, a, s)))
    
    def rho(self, w, par, a, sigma):
        if a == 0:
            return self.rho_na(w, par, sigma)
        else:
            return self.rho_k_a(w, par, a, sigma)
    
    def err_rho(self, w, par, a, sigma, cov):
        if a == 0:
            return self.err_rho_na(w, par, sigma, cov)
        else:
            return self.err_r_k_a(w, par, a, sigma, cov)

#Class of the correlator
class Double_exp:

    def fit_f(self, x, C1, m1, C2, m2):
        return C1 * mpmath.exp(-m1 * x) + C2 * mpmath.exp(-m2 * x)
    
    def f(self, x, par):
        x = mpmath.mpf(float(x))
        par = mpmath.matrix(par)
        return self.fit_f(x, par[0], par[1], par[2], par[3])
    
    def der0(self, x, par):
        return mpmath.exp(- par[1] * x)
    
    def der1(self, x, par):
        return - par[0] * x * mpmath.exp(- par[1] * x)
    
    def der2(self, x, par):
        return mpmath.exp(- par[3] * x)
    
    def der3(self, x, par):
        return - par[2] * x * mpmath.exp(- par[3] * x)
    
    def der_list(self):
        return [self.der0, self.der1, self.der2, self.der3]
    
    def jac(self, x, par):
        return mpmath.matrix([self.der0(x, par), self.der1(x, par), self.der2(x, par), self.der3(x, par)])
    
    def cov_matrix(self, x1, x2, par, cov_par):
        x1 = mpmath.mpf(float(x1))
        x2 = mpmath.mpf(float(x2))
        par = mpmath.matrix(par)
        cov_par = mpmath.matrix(cov_par)
        return (self.jac(x1, par).T * cov_par * self.jac(x2, par))[0, 0]
    
class Double_exp_np:
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
    
    def jac(self, x, par):
        return np.array([self.der0(x, par), self.der1(x, par), self.der2(x, par), self.der3(x, par)])
    
    def cov_matrix(self, x1, x2, par, cov_par):
        return (self.jac(x1, par).T @ cov_par @ self.jac(x2, par))