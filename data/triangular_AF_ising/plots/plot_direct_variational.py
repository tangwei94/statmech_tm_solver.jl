import numpy as np 
import matplotlib
from matplotlib import pyplot as plt 
import h5py
import glob, io

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 15

data_name = 'result_direct_variational.txt'
Fexact = 0.3230659669
fig, axes = plt.subplots(2, 1, figsize=(6,8))

f = io.open(data_name, 'r')
data = np.loadtxt(f)

axes[0].set(xlabel='bond dimension', xscale='log', yscale='log')
axes[0].plot(data[:, 0], np.abs(data[:, 1]-Fexact)/Fexact, 'o-', alpha=0.5, label='free energy err')
axes[0].plot(data[:, 0], data[:, 2], 's-', alpha=0.5, label='non herm cost func')
axes[0].text(0.25, 0.95, data_name, horizontalalignment='center', transform=axes[0].transAxes, fontsize='small')

axes[1].set(xlabel='bond dimension', xscale='log')
axes[1].plot(data[:, 0], data[:, 3], 'o-', alpha=0.5, label='free energy measured in the other MPO')

for ax in axes:
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize = 'small') #

fig.tight_layout()
plt.savefig("result_direct_variational.pdf", bbox_inches='tight')
plt.close(fig)