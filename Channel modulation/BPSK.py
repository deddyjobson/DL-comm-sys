'''
"borrowed" from link:http://www.raymaps.com/index.php/bpsk-bit-error-rate-calculation-using-python/
'''
from numpy import sqrt
from numpy.random import rand, randn
import matplotlib.pyplot as plt
import pickle
import numpy as np

N = 5000000
EbNodB_range = range(0,11)
itr = len(EbNodB_range)
ber = [None]*itr

for n in range (0, itr):

    EbNodB = EbNodB_range[n]
    EbNo=10.0**(EbNodB/10.0)
    x = 2 * (rand(N) >= 0.5) - 1
    noise_std = 1/sqrt(2*EbNo)
    y = x + noise_std * randn(N)
    y_d = 2 * (y >= 0) - 1
    errors = (x != y_d).sum()
    ber[n] = 1.0 * errors / N

    # print("EbNodB:" , EbNodB)
    # print("Error bits:" , errors)
    # print("Error probability:" , ber[n])

accs = pickle.load(open('error_rates_({0},{1}).pkl'.format(2,2),'rb'))
xx = np.linspace(-2,10,20)
yy = np.min(accs,axis=0) + 1e-10

plt.figure(dpi=300)
plt.plot(EbNodB_range, ber, 'bo', EbNodB_range, ber, 'k', label='BPSK')
plt.plot(xx,yy,label='AE (2,2)')
plt.axis([0, 10, 1e-6, 0.1])
plt.xscale('linear')
plt.yscale('log')
plt.xlabel('EbNo(dB)')
plt.ylabel('BER')
plt.legend()
plt.grid(True)
plt.title('BPSK Modulation')
plt.show()
