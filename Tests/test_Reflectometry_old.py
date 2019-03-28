from XRDTools.Reflectometry import *
import numpy as np
from scipy.special import erf, erfc

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import time

lamb = 0.15406
TwoThetaArray = np.linspace(0.1,8,400)
# thetaArray = np.array([1,])
substrateSi = Substrate(2.3291,roughness = 0,atomicRatio = 14/28.0855)
layerBore1 = Layer(2.37,4,roughness = 0,atomicRatio = 5/10.811)
layerZircon = Layer(6.52,4,roughness = 0,atomicRatio = 40/91.224)
layerBore2 = Layer(2.37,4,roughness = 0,atomicRatio = 5/10.811)

sample = Sample(substrate = substrateSi,layerList = [layerBore1,layerZircon,layerBore2],size = 20)

rD = 300
slit = 0.2
sampleSize = 20
effectiveSection = sampleSize*np.sin(np.deg2rad(TwoThetaArray/2))
effectiveSection[effectiveSection>slit] = slit
effectiveSection = effectiveSection/effectiveSection.max()
# effectiveSection = np.where()

# X,n = sample.getRefractionIndexProfils(lamb)

# TAR, _ = sample.getInterfaceTransferMatrix(lamb,np.deg2rad(thetaArray))
# print(TAR[:,:,0,0])

tic = time.time()
_,PI = sample.getInterfaceTransferMatrix(lamb,np.deg2rad(TwoThetaArray/2))
RArray, TArray, LArray = sample.getRandT(lamb,np.deg2rad(TwoThetaArray/2))


print(LArray[:,:,200])
toc = time.time()

print(toc-tic,'sec')



fig1 = plt.figure()

AX1 = fig1.add_subplot(2,1,1)
AX1.plot(TwoThetaArray,np.abs(RArray)**2,'.',label = 'left')
AX1.plot(TwoThetaArray,np.abs(effectiveSection*RArray)**2,'.',label = 'with sec')
AX1.legend()
AX1.set_yscale('log')

AX2 = fig1.add_subplot(2,1,2)
AX2.plot(TwoThetaArray,PI,'.')

plt.show()