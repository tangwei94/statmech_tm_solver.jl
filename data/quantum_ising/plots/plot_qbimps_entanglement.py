import numpy as np 
import matplotlib
from matplotlib import pyplot as plt 
import h5py
import glob, io

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 15

chis_full = [2, 4, 8, 16, 32, 64, 128]
Gammas = [0.75, 0.95, 1.00, 1.05, 1.25]
fitting_ranges = [(0, None), (1, None), (0, -1), (1, None), (1, None)]
for Gamma, fitting_range in zip(Gammas, fitting_ranges):
    num_chi = len(glob.glob("result_qbimps_chi*_Gamma{:.2f}.txt".format(Gamma)))
    chis = np.array(chis_full[:num_chi])

    data_names = ["result_qbimps_chi{:d}_Gamma{:.2f}.txt".format(chi, Gamma) for chi in chis]
    fig, axes = plt.subplots(3, 1, figsize=(6,12))

    axes[0].set(xlabel=r'$i$', ylabel=r'$s_i$')
    axes[0].set(yscale='log')
    axes[0].text(0.5, 0.9, 'Gamma = {:.2f}'.format(Gamma), horizontalalignment='center', transform=axes[0].transAxes, fontsize='small')

    axes[1].set(xlabel=r'$i$', ylabel=r'$s_{\mathrm{cmps}} / s_{\mathrm{mps}}$')

    axes[2].set(xlabel=r'$\log \chi$', ylabel=r'$\log (s_{\mathrm{cmps}} / s_{\mathrm{mps}})$')

    ratios = np.array([])
    for data_name, chi in zip(data_names, chis):
        f = io.open(data_name, 'r')
        data = np.loadtxt(f)

        axes[0].plot(data[0, :], 'o-', color='tab:blue', label=r'MPS' if chi==2 else None, alpha=0.5)
        axes[0].plot(data[1, :], 'o-', color='tab:red', label=r'cMPS' if chi==2 else None, alpha=0.5)

        axes[1].plot(data[1, :]/data[0, :], 'o-', label=r'$\chi$='+'{:d}'.format(chi), alpha=0.5)

        ratios = np.append(ratios, np.average(data[1, :chi//2]/data[0, :chi//2]))

    X, Y = np.log(chis), np.log(ratios)
    axes[2].plot(X, Y, 'o-', alpha=0.5, label='numerical data')
    k, c = np.polyfit(X[fitting_range[0]:fitting_range[1]], Y[fitting_range[0]:fitting_range[1]], 1)
    axes[2].plot(X, k*X+c, '-', label='linear fit: Y = {:.4f} X + {:.4f}'.format(k,c))
    axes[2].text(0.9, 0.7, 'ratio = {:.4f} * chi^({:.4f})'.format(np.exp(c), k), horizontalalignment='right', transform=axes[2].transAxes, fontsize='small')

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize = 'small') 

    fig.tight_layout()
    plt.savefig("result_qbimps_entanglement_Gamma{:.2f}.pdf".format(Gamma), bbox_inches='tight')
    plt.close(fig)

    print('plot Gamma={:.2f} done'.format(Gamma))
