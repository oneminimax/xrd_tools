from os import path
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from AsciiDataFile.Readers import GenericDataReader as Reader
from XRDTools.Diffractometer import Diffractometer
from XRDTools.Spectrum import centroid, calculateA, fitAGFct, AGFct

dataPath = 'Data/'

sampleList = [
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

for sample in sampleList:
    fileName = sample[0]
    thetaLim = sample[1]
    reader = Reader(' ',['angle','signal'])
    CS = reader.read(path.join(dataPath,fileName))

    angle = CS.getFieldByName('angle')
    signal = CS.getFieldByName('signal')
    subInd = (angle > thetaLim[0]) * (angle < thetaLim[1])
    # print(subInd)
    
    angle = angle[subInd]
    signal = signal[subInd]
    signal_error = 1*np.ones(signal.shape)

    K = dm.TwoTheta2GLength(angle)
    G = centroid(K,signal)

    X, complexA, ampA_error, argA_error = calculateA(K-G,signal,signal_error,stepX = 2,maxX = 120)
    A = np.abs(complexA)

    p0 = [100,1e-3,A[0],0]
    popt, perr = fitAGFct(X,A,ampA_error,G,p0 = p0)

    AX1.plot(angle,signal)
    AX2.plot(X,A)
    AX2.plot(X,AGFct(X,G,*popt),'--k')
    AX3.plot(X,np.angle(complexA))

    print(popt,perr)

plt.show()
    




