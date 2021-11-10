import numpy as np 
import matplotlib
from matplotlib import pyplot as plt 
import h5py
import glob, io

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 15

data_names = ['result_iTEBD.txt', 'result_iTEBD_alternative.txt']
Fexacts = [0.3230659669, 0.3230659669,]
fig, axes = plt.subplots(2, 1, figsize=(6,8))

for ax, data_name, Fexact in zip(axes, data_names, Fexacts):
    f = io.open(data_name, 'r')
    data = np.loadtxt(f)

    ax.set(xlabel='iTEBD step')
    ax.set(yscale='log')
    ax.plot(np.abs(data[:, 1]-Fexact)/Fexact, '-', label='free energy err')
    ax.plot(data[:, 2], '-', label='nonherm cost func')
    ax.plot(-data[:, 3] + 1e-16, '--', label='-ln fidelity')
    ax.text(0.25, 0.95, data_name, horizontalalignment='center', transform=ax.transAxes, fontsize='small')

for ax in axes:
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize = 'small') #

fig.tight_layout()
plt.savefig("result_iTEBD.pdf", bbox_inches='tight')
plt.close(fig)