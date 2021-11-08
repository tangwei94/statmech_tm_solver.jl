import numpy as np  
import matplotlib
import matplotlib.pyplot as plt 
import io

datafile_name = 'result_iTEBD.txt'

f = io.open(datafile_name, 'r')
data = np.loadtxt(f)

fig, ax = plt.subplots()

ax.set(xlabel='iTEBD step')

ax.set(yscale='log')
ax.plot(data[:, 2], '-', label='nonherm cost func')

Fexact = 0.3230659669 
ax.plot(np.abs(data[:, 1]-Fexact)/Fexact, '-', label='free energy err')
ax.legend()
plt.savefig("result_iTEBD.pdf")