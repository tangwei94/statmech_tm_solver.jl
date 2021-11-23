import numpy as np 
import matplotlib
from matplotlib import pyplot as plt 
import h5py
import glob, io

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 15

chis_full = np.arange(8, 81, 8)
Deltas = [-0.5, 0.0, 0.5]
fitting_ranges = [(3, None), (3, None), (3, None)]
for Delta, fitting_range in zip(Deltas, fitting_ranges):
    num_chi = len(glob.glob("rawdata/result_qbimps_chi*_xxz_Delta{:.2f}.txt".format(Delta)))
    chis = np.array(chis_full[:num_chi])

    data_names = ["rawdata/result_qbimps_chi{:d}_xxz_Delta{:.2f}.txt".format(chi, Delta) for chi in chis]
    fig, axes = plt.subplots(3, 1, figsize=(6,18))
    #axes = axes.flatten()

    axes[0].set(xlabel=r'$i$', ylabel=r'$\ln s_i$')
    axes[0].text(0.5, 0.9, 'Delta = {:.2f}'.format(Delta), horizontalalignment='center', transform=axes[0].transAxes, fontsize='small')

    axes[1].set(xlabel=r'$i$', ylabel=r'$\ln s_i^{\mathrm{cmps}} - \ln s_i^{\mathrm{mps}}$')

    axes[2].set(xlabel=r'$\log \chi$', ylabel='entanglement entropy')

    ratios = np.array([])
    EEs_MPS, EEs_cMPS = np.array([]), np.array([])
    get_EE = lambda arr: -np.sum(arr * np.log(arr))
    for data_name, chi in zip(data_names, chis):
        f = io.open(data_name, 'r')
        data = np.loadtxt(f)

        axes[0].plot(np.log(data[0, :]), 'o-', color='tab:blue', label=r'MPS' if chi==8 else None, alpha=0.5)
        axes[0].plot(np.log(data[1, :]), 'o-', color='tab:red', label=r'cMPS' if chi==8 else None, alpha=0.5)

        axes[1].plot(np.log(data[1, :]/data[0, :]), 'o-', label=r'$\chi$='+'{:d}'.format(chi), alpha=0.5)

        ratios = np.append(ratios, np.average(data[1, :chi//2]/data[0, :chi//2]))
        EEs_MPS = np.append(EEs_MPS, get_EE(data[0, :]))
        EEs_cMPS = np.append(EEs_cMPS, get_EE(data[1, :]))

    X, Y = np.log(chis), EEs_MPS
    axes[2].plot(X, Y, 'o-', color='tab:blue', alpha=0.5, label='iMPS')
    k, c = np.polyfit(X[fitting_range[0]:fitting_range[1]], Y[fitting_range[0]:fitting_range[1]], 1)
    axes[2].plot(X, k*X+c, '--', color='tab:blue', label='iMPS fit, S={:.3f}*log(chi){:+.3f}'.format(k, c))

    X, Y = np.log(chis), EEs_cMPS
    axes[2].plot(X, Y, 'o-', color='tab:red', alpha=0.5, label='cMPS')
    k, c = np.polyfit(X[fitting_range[0]:fitting_range[1]], Y[fitting_range[0]:fitting_range[1]], 1)
    axes[2].plot(X, k*X+c, '--', color='tab:red', label='cMPS fit, S={:.3f}*log(chi){:+.3f}'.format(k, c))

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize = 'small') 

    fig.tight_layout()
    plt.savefig("result_qbimps_entanglement_Delta{:.2f}.pdf".format(Delta), bbox_inches='tight')
    plt.close(fig)

    print('plot Delta={:.2f} done'.format(Delta))
