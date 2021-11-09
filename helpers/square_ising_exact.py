# https://gandhiviswanathan.wordpress.com/2015/01/09/onsagers-solution-of-the-2-d-ising-model-the-combinatorial-method/

import numpy as np 
from scipy.integrate import dblquad

def onsager_sol(beta):
    k = 0.5 * np.sinh(2*beta) / np.cosh(2*beta)**2
    f = lambda kx, ky: np.log(1-2*k*(np.cos(kx)+np.cos(ky)))
    return np.log(2*np.cosh(2*beta)) + (1/2/np.pi**2) * dblquad(f, 0, np.pi, lambda kx: 0, lambda kx: np.pi)[0]
print(onsager_sol(1.))