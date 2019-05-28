from os import path
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from AsciiDataFile.Readers import GenericDataReader as Reader
from XRDTools.Diffractometer import Diffractometer
from XRDTools.Geometry import two_theta_2_wave_vector_length
from XRDTools.Spectrum import centroid, calculateA, fitAGFct, AGFct

data_path = 'Data/'

sample_tuples = [
    ('thickness_100.dat',(35.25,36)),
    ('thickness_111.dat',(30.72,31.8)),
    ('thickness_PCCO.dat',(28,30.5))
    ]

dm = Diffractometer()

fig = plt.figure()
AX1 = fig.add_subplot(1,2,1)
AX2 = fig.add_subplot(2,2,2)
AX3 = fig.add_subplot(2,2,4)

AX1.set_yscale('log')
# AX2.set_yscale('log')

for sample_tuple in sample_tuples:
    file_name = sample_tuple[0]
    two_theta_lim = sample_tuple[1]
    reader = Reader(' ',['two theta','signal'])
    CS = reader.read(path.join(data_path,file_name))

    two_theta = CS.get_column_by_name('two theta')
    signal = CS.get_column_by_name('signal')
    subInd = (two_theta > two_theta_lim[0]) * (two_theta < two_theta_lim[1])
    # print(subInd)
    
    two_theta = two_theta[subInd]
    signal = signal[subInd]
    signal_error = 1*np.ones(signal.shape)

    K = two_theta_2_wave_vector_length(dm.wave_length,two_theta)
    G = centroid(K,signal)

    X, complexA, ampA_error, argA_error = calculateA(K-G,signal,signal_error,stepX = 2,maxX = 120)
    A = np.abs(complexA)

    p0 = [100,1e-3,A[0],0]
    popt, perr = fitAGFct(X,A,ampA_error,G,p0 = p0)

    AX1.plot(two_theta,signal)
    AX2.plot(X,A)
    AX2.plot(X,AGFct(X,G,*popt),'--k')
    AX3.plot(X,np.angle(complexA))

    print(popt,perr)

plt.show()
    




