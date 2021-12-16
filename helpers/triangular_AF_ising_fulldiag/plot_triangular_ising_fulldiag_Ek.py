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

for ix, n in enumerate([5, 6, 7, 8, 9, 10, 11, 12]):
    msk = (np.isclose(data[:, 0], n))
    w_reals = data[msk, 1]
    w_imags = data[msk, 2]

    w_norms = np.sqrt(w_reals**2 + w_imags**2)
    w_norms /= max(w_norms)
    w_angles = np.angle(w_reals + 1j*w_imags)

    axes[ix].set(xlabel=r'$\mathrm{arg} w / \pi$', ylabel='-log |w|', xlim=(-1, 1), ylim=(0, 3))

    axes[ix].plot(w_angles / np.pi, -np.log(w_norms), 'o', color='tab:red', alpha=0.5)

    axes[ix].text(0.1, 0.9, "L={:d}".format(n), horizontalalignment='center', transform=axes[ix].transAxes, fontsize='small')

fig.tight_layout()
plt.savefig("result_triangular_ising_fulldiag_Ek.pdf", bbox_inches='tight')
plt.close(fig)

