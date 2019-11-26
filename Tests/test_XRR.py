import numpy as np
from numpy.linalg import inv
from numpy import matmul

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import pint
ureg = pint.UnitRegistry()

relec = 2.818e-15 * ureg.meter
lamb = 0.15406 * ureg.nanometer

BoreDensity = 2.37 * ureg.gram/ureg.cm**3
BoreAtomicMass = 10.811 * ureg.atomic_mass_unit
BoreAtomicNumber = 5
BoreDelta = 1/(2*np.pi)*relec*BoreAtomicNumber/BoreAtomicMass*BoreDensity*lamb**2
nBore = 1 - BoreDelta

BoreDensity = 2.37 * ureg.gram/ureg.cm**3
BoreAtomicMass = 10.811 * ureg.atomic_mass_unit
BoreAtomicNumber = 5
BoreDelta = 1/(2*np.pi)*relec*BoreAtomicNumber/BoreAtomicMass*BoreDensity*lamb**2
nBore = 1 - BoreDelta

SiDensity = 2.3291 * ureg.gram/ureg.cm**3
SiAtomicMass = 28.0855 * ureg.atomic_mass_unit
SiAtomicNumber = 14
SiDelta = 1/(2*np.pi)*relec*SiAtomicNumber/SiAtomicMass*SiDensity*lamb**2
SiDelta2 = 1/(2*np.pi)*(2.818e-13)*(14)/(28.0855)*(6.02214076e23)*2.3291*0.15406**2*1e-14
nSi = 1 - SiDelta

print(BoreDelta.to_base_units(),SiDelta.to_base_units())
print(1 - SiDelta,1-SiDelta2)

quit()

def SMatrixType0(n,theta0,k0,z):

    alpha = np.sqrt(1 - (np.cos(theta0)/n)**2)
    kz = k0*n*alpha

    return np.array([[np.exp(1j * kz * z),np.exp(-1j * kz * z)],[1j*kz*np.exp(1j * kz * z),-1j*kz*np.exp(-1j * kz * z)]])

def SMatrixType1(n,theta0,k0,z):

    alpha = np.sqrt(1 - (np.cos(theta0)/n)**2)
    beta = n**2 * np.cos(theta0)
    kz = k0*n*alpha

    return np.array([[alpha*np.exp(1j * kz * z),-alpha*np.exp(-1j * kz * z)],[beta*np.exp(1j * kz * z),beta*np.exp(-1j * kz * z)]])

def SMatrixType2(n,theta0,k0,z):

    alpha = np.sqrt(1 - (np.cos(theta0)/n)**2)
    kz = k0*n*alpha

    return np.array([[np.exp(1j * kz * z),np.exp(-1j * kz * z)],[kz*np.exp(1j * kz * z),-kz*np.exp(-1j * kz * z)]])

z0 = 0
z1 = 4
n0 = 1
n1 = nBore
n2 = nSi
k0 = 2*np.pi/0.15

thetaArray = np.linspace(1,8,20)
thetaArray = np.array([1,])
r0Array = np.zeros(thetaArray.shape)
t0Array = np.zeros(thetaArray.shape)
r1Array = np.zeros(thetaArray.shape)
t1Array = np.zeros(thetaArray.shape)
r2Array = np.zeros(thetaArray.shape)
t2Array = np.zeros(thetaArray.shape)

for i,theta0 in enumerate(np.deg2rad(thetaArray)):

    print(n0,n1,n2)
    print([ 0.99997005,0.99996715,1.        ])

    print(matmul(inv(SMatrixType0(n1,theta0,k0,z0)),SMatrixType0(n0,theta0,k0,z0)))
    print(matmul(inv(SMatrixType0(n2,theta0,k0,z1)),SMatrixType0(n1,theta0,k0,z1)))

    L0 = matmul(matmul(inv(SMatrixType0(n2,theta0,k0,z1)),SMatrixType0(n1,theta0,k0,z1)),matmul(inv(SMatrixType0(n1,theta0,k0,z0)),SMatrixType0(n0,theta0,k0,z0)))
    L1 = matmul(matmul(inv(SMatrixType1(n2,theta0,k0,z1)),SMatrixType1(n1,theta0,k0,z1)),matmul(inv(SMatrixType1(n1,theta0,k0,z0)),SMatrixType1(n0,theta0,k0,z0)))
    L2 = matmul(matmul(inv(SMatrixType2(n2,theta0,k0,z1)),SMatrixType2(n1,theta0,k0,z1)),matmul(inv(SMatrixType2(n1,theta0,k0,z0)),SMatrixType2(n0,theta0,k0,z0)))

    r0Array[i] = np.abs(- L0[1,0]/L0[1,1])
    t0Array[i] = np.abs(L0[0,0] - L0[1,0]*L0[0,1]/L0[1,1])

    r1Array[i] = np.abs(- L1[1,0]/L1[1,1])
    t1Array[i] = np.abs(L1[0,0] - L1[1,0]*L1[0,1]/L1[1,1])

    r2Array[i] = np.abs(- L2[1,0]/L2[1,1])
    t2Array[i] = np.abs(L2[0,0] - L2[1,0]*L2[0,1]/L2[1,1])

# print(r1,t1,np.abs(r1)**2+np.abs(t1)**2)
# print(r2,t2,np.abs(r2)**2+np.abs(t2)**2)

fig1 = plt.figure()

AX1 = fig1.add_subplot(1,1,1)
AX1.set_yscale('log')
# AX2 = fig1.add_subplot(1,2,2)

# AX1.plot(thetaArray,r0Array**2,label = 'r type 0')
AX1.plot(thetaArray,r1Array,label = 'r type 1')
AX1.plot(thetaArray,r2Array,label = 'r type 2')

# AX1.plot(thetaArray,t1Array**2,label = 't type 1')
# AX1.plot(thetaArray,t2Array**2,label = 't type 2')
AX1.legend()
# AX2.legend()

plt.show()
