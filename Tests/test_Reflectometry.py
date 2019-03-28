from XRDTools.ReflectometryP import *
import numpy as np
from scipy.special import erf, erfc

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import time

lamb = 0.15406
TwoThetaArray = np.linspace(0.1,8,200)
# thetaArray = np.array([1,])
substrateSi = Substrate(2.3291,atomicNumber = 14,atomicMass = 28.0855)
layerSiO2 = Substrate(2.249,atomicNumber = 14 + 2*8,atomicMass = 28.0855 + 2*15.9994)
layerBore1 = Layer(3.34,4,atomicNumber = 5, atomicMass = 10.811)
layerZircon = Layer(6.52,4,atomicNumber = 40, atomicMass = 91.224)
layerBore2 = Layer(2.37,4,atomicNumber = 5, atomicMass = 10.811)
# layerB2O3 = Substrate(2,46,atomicNumber = 2*5 + 3*8,atomicMass = 2*10.811 + 3*15.9994)

# print('Si  ',substrateSi.getAtomicDensity(),(1-substrateSi.getn(lamb))*1e6)
# print('SiO2',layerSiO2.getAtomicDensity(),(1-layerSiO2.getn(lamb))*1e6)
# print('B   ',layerBore1.getAtomicDensity(),(1-layerBore1.getn(lamb))*1e6)
# print('Bam ',layerBore2.getAtomicDensity(),layerBore1.getn(lamb))
# print('Zr  ',layerZircon.getAtomicDensity(),(1-layerZircon.getn(lamb))*1e6)
# print('B2O3',layerB2O3.getAtomicDensity(),layerB2O3.getn(lamb),1-layerB2O3.getn(lamb))

# layerBore1.

# layerSTO = Layer(5.12,4,roughness = 0,atomicNumber = 38, atomicMass = 87.62 + 47.87 + 3*16)
# print(layerSiO2.getDensityGenX())

sample = Sample(substrate = substrateSi,layerList = [layerBore1,layerZircon,layerBore2])
R = sample.getRCoef(lamb,np.deg2rad(TwoThetaArray/2))
# sample = Sample(substrate = substrateSi,layerList = [layerBore1])
# sample = Sample(substrate = substrateSi,layerList = [])
# XLayer = sample.getXLayer()
# XInterface = sample.getXInterface()
# kzLayer = sample.getkzLayer(lamb,np.deg2rad(TwoThetaArray/2))
# interfaceMatrix = sample.getinterfaceMatrix(XInterface,kzLayer)
# interfaceDeterminant = sample.getinterfaceDeterminant(interfaceMatrix)
# totDeterminant = np.prod(interfaceDeterminant,axis = 0)
# LMatrix = sample.getLArray(kzLayer,interfaceMatrix)


# fig1 = plt.figure()

# AX1 = fig1.add_subplot(3,1,1)
# AX2 = fig1.add_subplot(3,1,2, sharex=AX1)
# AX3 = fig1.add_subplot(3,1,3, sharex=AX1)
# # 
# # AX3 = fig1.add_subplot(2,2,3)
# # AX4 = fig1.add_subplot(2,2,4)
# layerName = ['Air','B1','Zr','B2','Sub']
# AX1.set_xlabel('2T')
# AX1.set_ylabel('Re[k_z]')
# AX2.set_xlabel('2T')
# AX2.set_ylabel('Im[k_z]')
# AX3.set_xlabel('2T')
# AX3.set_ylabel('Abs(Det(int))')
# AX3.set_ylim((-0.1,0.1))
# for i in range(1,kzLayer.shape[0]):
#     AX1.plot(TwoThetaArray,np.real(kzLayer[i,:]),'.-',label = layerName[i])
#     AX2.plot(TwoThetaArray,np.imag(kzLayer[i,:]),'.-')
#     AX3.plot(TwoThetaArray,np.abs(interfaceDeterminant[i-1,:]),'.-')
# # AX3.plot(TwoThetaArray,np.abs(totDeterminant),'.-')

# # AX3.plot(TwoThetaArray,np.real(np.prod(interfaceDeterminant,axis = 0)),'.-')
# # AX4.plot(TwoThetaArray,np.imag(np.prod(interfaceDeterminant,axis = 0)),'.-')

# # for i in range(kzLayer.shape[0]-1):
# #     AX3.plot(TwoThetaArray,np.real(interfaceMatrix[0,0,i,:]),'.-')
# #     AX4.plot(TwoThetaArray,np.imag(interfaceMatrix[0,0,i,:]),'.-')
# AX1.legend()

fig2 = plt.figure()

AX1 = fig2.add_subplot(1,1,1)
AX1.set_yscale('log')

AX1.plot(TwoThetaArray,np.abs(R)**1,'.')



plt.show()


# pi = sample.getPenetrationIndex(lamb,np.deg2rad(TwoThetaArray/2))
# # print(sample.getLArray(lamb,np.deg2rad(TwoThetaArray/2)))

# # print(np.abs(LArray[1,0]/LArray[1,1])**2)

# # out1 = sample.getrArray(lamb,np.deg2rad(TwoThetaArray/2))



# # LArray = sample.getLArray(lamb,np.deg2rad(TwoThetaArray/2))
# # AX1.plot(TwoThetaArray,np.real(sample.getRCoef(lamb,np.deg2rad(TwoThetaArray/2))),'.')
# # AX1.plot(TwoThetaArray,np.imag(sample.getRCoef(lamb,np.deg2rad(TwoThetaArray/2))),'.')
# AX1.plot(TwoThetaArray,np.abs(sample.getRCoef(lamb,np.deg2rad(TwoThetaArray/2)))**2,'.')
# AX1.plot(TwoThetaArray,np.abs(sample.getRCoefApprox(lamb,np.deg2rad(TwoThetaArray/2)))**2,'.')
# AX1.plot(TwoThetaArray,pi+0.5,'.')

# # # layerBore2.setRoughness(0.4)
# # # LArray = sample.getLArray(lamb,np.deg2rad(TwoThetaArray/2))
# # # AX1.plot(TwoThetaArray,np.abs(LArray[1,0]/LArray[1,1])**2,'.')
# # # AX1.plot(sample.getXLayer(),sample.getnLayer(lamb),'.')
# # # AX1.legend()
# AX1.set_yscale('log')

# # # AX2 = fig1.add_subplot(2,1,2)
# # # AX2.plot(XMiddle,dn,'.',label = 'n')
# # # AX2.plot(TwoThetaArray,PI,'.')

# plt.show()