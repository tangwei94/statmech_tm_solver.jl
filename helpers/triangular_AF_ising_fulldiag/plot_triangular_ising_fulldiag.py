import numpy as np 
import matplotlib 
import matplotlib.pyplot as plt 
import io 

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 15

f = io.open("result_triangular_ising_fulldiag.txt", "r")
data = np.loadtxt(f)
f.close()

fig, axes = plt.subplots(4, 2, figsize=(8, 16))
axes = axes.flatten()
thetas = np.linspace(0, 2*np.pi, 1001)
polar_thetas = np.array([0, 2*np.pi/3, 4*np.pi/3, 2*np.pi])

for ix, n in enumerate([5, 6, 7, 8, 9, 10, 11, 12]):
    msk = (np.isclose(data[:, 0], n))
    w_reals = data[msk, 1]
    w_imags = data[msk, 2]
    w_max = max(w_reals)
    
    axes[ix].set(xlabel='real part', ylabel='imag part', xlim=(-w_max, w_max), ylim=(-w_max, w_max))
    axes[ix].set_aspect('equal')

    axes[ix].plot(w_max*np.cos(thetas), w_max*np.sin(thetas), '-', color='tab:blue', alpha=0.5)
    axes[ix].plot(w_max*np.cos(polar_thetas), w_max*np.sin(polar_thetas), '--', color='tab:gray', alpha=0.5)
    axes[ix].plot(w_reals, w_imags, '.', color='tab:red', alpha=0.5)

    axes[ix].text(0.1, 0.9, "L={:d}".format(n), horizontalalignment='center', transform=axes[ix].transAxes, fontsize='small')

fig.tight_layout()
plt.savefig("result_triangular_ising_fulldiag.pdf", bbox_inches='tight')
plt.close(fig)

