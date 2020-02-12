from XRDTools.Reflectometry import *
import numpy as np
from scipy.special import erf, erfc

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import time

lamb = 0.15406
two_thetas = np.linspace(0.1,8,200)
# thetaArray = np.array([1,])
substrateSi = Substrate('Si',2.3291,atomic_number = 14,atomic_mass = 28.0855)
substrateSiO2 = Substrate('SiO2',2.249,atomic_number = 14 + 2*8,atomic_mass = 28.0855 + 2*15.9994)
# layerBore = Layer('B1',3.34,4,atomic_number = 5, atomic_mass = 10.811)
# layerZircon = Layer('Zr',6.52,4,atomic_number = 40, atomic_mass = 91.224)
layerMoSi2 = Layer('MoSi2',6.31,20,atomic_number = 42+2*14, atomic_mass = 95.95+2*28.0855)
# layerB2O3 = Substrate(2,46,atomic_number = 2*5 + 3*8,atomic_mass = 2*10.811 + 3*15.9994)

n_layers = 10
layers = list()
for n in range(n_layers):
    layers.append(Layer('MoSi2',6.31,20/n_layers,atomic_number = 42+2*14, atomic_mass = 95.95+2*28.0855))


sample = Sample(substrate = substrateSiO2,layers = layers)
Rp = sample.get_reflect_coef(lamb,np.deg2rad(two_thetas/2),polarisation = 'p')
Rs = sample.get_reflect_coef(lamb,np.deg2rad(two_thetas/2),polarisation = 's')
# sample = Sample(substrate = substrateSi,layerList = [layerBore1])
# sample = Sample(substrate = substrateSi,layerList = [])
# XLayer = sample.getXLayer()
# XInterface = sample.getXInterface()
# kzLayer = sample.getkzLayer(lamb,np.deg2rad(two_thetas/2))
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
#     AX1.plot(two_thetas,np.real(kzLayer[i,:]),'.-',label = layerName[i])
#     AX2.plot(two_thetas,np.imag(kzLayer[i,:]),'.-')
#     AX3.plot(two_thetas,np.abs(interfaceDeterminant[i-1,:]),'.-')
# # AX3.plot(two_thetas,np.abs(totDeterminant),'.-')

# # AX3.plot(two_thetas,np.real(np.prod(interfaceDeterminant,axis = 0)),'.-')
# # AX4.plot(two_thetas,np.imag(np.prod(interfaceDeterminant,axis = 0)),'.-')

# # for i in range(kzLayer.shape[0]-1):
# #     AX3.plot(two_thetas,np.real(interfaceMatrix[0,0,i,:]),'.-')
# #     AX4.plot(two_thetas,np.imag(interfaceMatrix[0,0,i,:]),'.-')
# AX1.legend()

fig2 = plt.figure()

AX1 = fig2.add_subplot(1,1,1)
AX1.set_yscale('log')

AX1.plot(two_thetas,np.abs(Rs)**2,'.')
AX1.plot(two_thetas,np.abs(Rp)**2,'.')



plt.show()


# pi = sample.getPenetrationIndex(lamb,np.deg2rad(two_thetas/2))
# # print(sample.getLArray(lamb,np.deg2rad(two_thetas/2)))

# # print(np.abs(LArray[1,0]/LArray[1,1])**2)

# # out1 = sample.getrArray(lamb,np.deg2rad(two_thetas/2))



# # LArray = sample.getLArray(lamb,np.deg2rad(two_thetas/2))
# # AX1.plot(two_thetas,np.real(sample.getRCoef(lamb,np.deg2rad(two_thetas/2))),'.')
# # AX1.plot(two_thetas,np.imag(sample.getRCoef(lamb,np.deg2rad(two_thetas/2))),'.')
# AX1.plot(two_thetas,np.abs(sample.getRCoef(lamb,np.deg2rad(two_thetas/2)))**2,'.')
# AX1.plot(two_thetas,np.abs(sample.getRCoefApprox(lamb,np.deg2rad(two_thetas/2)))**2,'.')
# AX1.plot(two_thetas,pi+0.5,'.')

# # # layerBore2.setRoughness(0.4)
# # # LArray = sample.getLArray(lamb,np.deg2rad(two_thetas/2))
# # # AX1.plot(two_thetas,np.abs(LArray[1,0]/LArray[1,1])**2,'.')
# # # AX1.plot(sample.getXLayer(),sample.getnLayer(lamb),'.')
# # # AX1.legend()
# AX1.set_yscale('log')

# # # AX2 = fig1.add_subplot(2,1,2)
# # # AX2.plot(XMiddle,dn,'.',label = 'n')
# # # AX2.plot(two_thetas,PI,'.')

# plt.show()