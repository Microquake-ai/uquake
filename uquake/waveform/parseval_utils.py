import numpy as np
from scipy.fftpack import fft, fftfreq, rfft, rfftfreq

import matplotlib.pyplot as plt

"""
    mag_utils - a collection of routines to assist in the moment magnitude calculation
"""


def parsevals(data, dt, nfft):
    """
    Proper scaling to satisfy Parsevals:

        Scale a 2-sided fft by dt
        Scale a 1-sided rft by dt x sqrt(2) 

    The 2-sided nfft-point fft returns nfft Complex values (+ve and -ve freqs)
    The 1-sided nfft-point fft returns nfft/2 + 1 Complex values (+ve freqs + DC + Nyq)

    >>>parsevals(tr.data, tr.stats.sampling_rate, npow2(tr.data.size))
       Parseval's: [time] ndata=11750 dt=6000.000000 sum=0.0001237450569 [time]
       Parseval's: [freq]  nfft=65536 df=2.54313e-09 sum=0.0001237450441 [2-sided]
       Parseval's: [freq]  nfft=65536 df=2.54313e-09 sum=0.0001237450441 [1-sided]
    """

    tsum = np.sum(np.square(data)) * dt

    print("Parseval's: [time] ndata=%7d dt=%12.6f sum=%12.10g [time]" % (
    data.size, dt, tsum))

    # 2-sided (+ve & -ve freqs) FFT:
    X = fft(data, nfft) * dt

    df = 1. / (dt * float(X.size))  # X.size = nfft
    # fsum = np.sum(X*np.conj(X))*df
    # Do it this way so it doesn't spit a ComplexWarning about throwing away imag part
    fsum = np.sum(np.abs(X) * np.abs(X)) * df
    print("Parseval's: [freq]  nfft=%7d df=%12.6g sum=%12.10g [2-sided]" % (
    X.size, df, fsum))

    # 1-sided: N/2 -1 +ve freqs + [DC + Nyq] = N/2 + 1 values:
    df = 1. / (dt * float(nfft))  # df is same as for 2-sided case
    Y, freqs = unpack_rfft(rfft(data, n=nfft), df)
    Y *= dt
    '''
        Note: We only have the +ve freq half, so we need to double all the power
              at each frequency *except* for DC & Nyquist,
              which are purely real and don't have a -ve freq.
              So either scale the fsum by 2, or scale Y (minus DC/Nyq) by sqrt(2) here
    '''
    Y[1:-1] *= np.sqrt(2.)
    fsum = np.sum(np.abs(Y) * np.abs(Y)) * df
    print("Parseval's: [freq]  nfft=%7d df=%12.6g sum=%12.10g [1-sided]" % (
    nfft, df, fsum))

    return


def unpack_rfft(rfft, df):
    n = rfft.size
    if n % 2 == 0:
        n2 = int(n / 2)
    else:
        print("n is odd!!")
        exit()
    # print("n2=%d" % n2)

    c_arr = np.array(np.zeros(n2 + 1, ), dtype=np.complex_)
    freqs = np.array(np.zeros(n2 + 1, ), dtype=np.float_)

    c_arr[0] = rfft[0]
    c_arr[n2] = rfft[n - 1]
    freqs[0] = 0.
    freqs[n2] = float(n2) * df

    for i in range(1, n2):
        freqs[i] = float(i) * df
        c_arr[i] = np.complex(rfft[2 * i - 1], rfft[2 * i])

    return c_arr, freqs


def npow2(n: int) -> int:
    """ return power of 2 >= n
    """
    if n <= 0:
        return 0

    nfft = 2
    while (nfft < n):
        nfft = (nfft << 1)
    return nfft


if __name__ == '__main__':
    main()
