import numpy as np
import os
import matplotlib.pyplot as plt

digits = 7

snr_dBs = np.linspace(-2,10,20)
snrs = 10 ** (snr_dBs/10)


if True:
    if os.path.isfile('results.csv'):
        os.remove('results.csv')

    for snr in snrs:
        hyper_lst = [500,2,2,0,snr,digits,1]
        hyper_str = '--n_epochs {0} --M {1} --n {2} --verbose {3} --SNR {4} --e_prec {5} --write_out {6}'.format(*hyper_lst)
        os.system('python3 autoencoder.py '+hyper_str)


data = np.loadtxt(open("results.csv", "rb"), delimiter=",", skiprows=0)
x,y = data.T

xx = 10 * np.log10(x)
yy = y + 10**(-digits) # to soften the graph

plt.figure(dpi=300)
plt.semilogy(xx,yy)
plt.title('Error profile of autoencoder')
plt.xlabel('$E_b/N_0$ [dB]')
plt.ylabel('Block Error Rate')
plt.show()
