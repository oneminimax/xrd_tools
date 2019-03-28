import numpy as np
from scipy.special import erfc

from numpy import matmul
from numpy.linalg import inv

deltaFactor = (2.818e-13)*(6.02214076e+23)*1e-14/(2*np.pi) #atomicRatio, density g/cm^3, (lambda nm)^2
nSigmaRoughness = 4
roughnessDivision = 20

class Sample(object):

    def __init__(self,substrate = None,layerList = list()):
        
        if substrate:
            self.setSubstrate(substrate)
        self.layerList = layerList
        self.vacuum = Vacuum()

    def setSubstrate(self,substrate):
        self.substrate = substrate

    def addLayer(self,layer):
        self.layerList.append(layer)

    def getLastLayer(self):
        if len(self.layerList) > 0:
            return self.layerList[-1]
        else:
            return self.substrate

    def getPreviousLayer(self,layer):

        index = self.layerList.index(layer)
        if index == 0:
            return self.substrate
        else:
            return self.layerList[index-1]

    def getTotalThickness(self):

        thickness = 0
        for layer in self.layerList:
            thickness += layer.thickness

        return thickness

    def getXInterface(self):

        self.X0 = 0
        XInterface = self.substrate.getXInterface(self)
        for layer in self.layerList:
            XInterface = np.append(XInterface,layer.getXInterface(self))
        XInterface = np.append(XInterface,self.vacuum.getXInterface(self))

        return XInterface

    def getXMiddle(self):

        XInterface = self.getXInterface()
        XMiddle = (XInterface[:-1] + XInterface[1:])/2

        return XMiddle

    def getDensityProfil(self,X):

        densityProfil = np.zeros(XLeftRight.shape)
        
        X0 = 0
        densityProfil += self.substrate.getDensityProfil(self,X0,X)
        
        for layer in self.layerList:
            densityProfil += layer.getDensityProfil(self,X0,X)
            X0 += layer.thickness

        return  X, densityProfil

    def getnProfil(self,lamb,X):

        deltaProfil = np.zeros(X.shape)
        
        X0 = 0
        deltaProfil += self.substrate.getDeltaProfil(self,X0,X,lamb)
        
        for layer in self.layerList:
            deltaProfil += layer.getDeltaProfil(self,X0,X,lamb)
            X0 += layer.thickness

        nProfil = 1 - deltaProfil

        return nProfil

    def getdn(self,lamb):

        XInterface = self.getXInterface()
        nInterface = self.getnProfil(lamb,XInterface)

        dn = np.diff(nInterface)/np.diff(XInterface)

        return dn

    def getMaxXDepthIndex(self,nArray,thetaArray):

        dn = nArray[:,None] - np.cos(thetaArray[None,:])
        penetrationBool = (dn >= 0)
        penetrationIndex = np.zeros(dn.shape[1],dtype = int)

        for iTheta in range(dn.shape[1]):
            index = np.argwhere(np.flip(penetrationBool[:,iTheta],axis = 0) == False)
            if len(index)>0:
                penetrationIndex[iTheta] = index[0,0]
            else:
                penetrationIndex[iTheta] = -1

        return penetrationIndex

    def kzIntegral(self,penetrationIndex,kzArray):

        XInterface = self.getXInterface()
        DX = np.diff(XInterface)
        
        kzIntegral = np.zeros(penetrationIndex.shape)

        for iTheta in range(penetrationIndex.size):
            if penetrationIndex[iTheta] == -1:
                kzIntegral[iTheta] = np.sum(kzArray[:,iTheta]**2 * DX)
            else:
                kzIntegral[iTheta] = np.sum(kzArray[-(penetrationIndex[iTheta]-1):,iTheta]**2 * DX[-(penetrationIndex[iTheta]-1):])

        return kzIntegral

    def integral1(self,penetrationIndex,kzArray,kzPrimeArray):

        XInterface = self.getXInterface()
        XMiddle = self.getXMiddle()
        DX = np.diff(XInterface)
        
        integral1 = np.zeros(penetrationIndex.shape,dtype = complex)

        for iTheta in range(penetrationIndex.size):
            if penetrationIndex[iTheta] == -1:
                integral1[iTheta] = 1j*np.sum(XMiddle[:,None]*kzPrimeArray[:,iTheta] * DX)
            else:
                integral1[iTheta] = 1j*np.sum(XMiddle[-(penetrationIndex[iTheta]-1):,None]*kzPrimeArray[-(penetrationIndex[iTheta]-1):,iTheta] * DX[-(penetrationIndex[iTheta]-1):])
                # np.sum(kzArray[-(penetrationIndex[iTheta]-1):,iTheta]**2 * DX[-(penetrationIndex[iTheta]-1):])

        return integral1

    def integral2(self,penetrationIndex,kzArray,kzPrimeArray):

        XInterface = self.getXInterface()
        XMiddle = self.getXMiddle()
        DX = np.diff(XInterface)
        
        integral2 = np.zeros(penetrationIndex.shape,dtype = complex)

        for iTheta in range(penetrationIndex.size):
            if penetrationIndex[iTheta] == -1:
                integral2[iTheta] = np.sum(np.exp(2j*kzArray[:,iTheta]*XMiddle[:,None])*kzPrimeArray[:,iTheta] * DX/(2*kzArray[:,iTheta]))
            else:
                integral2[iTheta] = np.sum(np.exp(2j*kzArray[-(penetrationIndex[iTheta]-1):,iTheta]*XMiddle[-(penetrationIndex[iTheta]-1):,None])*kzPrimeArray[-(penetrationIndex[iTheta]-1):,iTheta] * DX[-(penetrationIndex[iTheta]-1):]/(2*kzArray[-(penetrationIndex[iTheta]-1):,iTheta]))
                # np.sum(kzArray[-(penetrationIndex[iTheta]-1):,iTheta]**2 * DX[-(penetrationIndex[iTheta]-1):])

        return integral2

    def getkzArray(self,lamb,nArray,thetaArray):

        return (2*np.pi/lamb) * np.sqrt(np.abs(nArray[:,None]**2 - np.cos(thetaArray[None,:]))**2)

    def getkzPrimeArray(self,lamb,nArray,thetaArray):

        kzArray = self.getkzArray(lamb,nArray,thetaArray)
        dnArray = self.getdn(lamb)

        return (2*np.pi/lamb)**2 * nArray[:,None]/kzArray * dnArray[:,None]

    def getrArray(self,lamb,thetaArray):

        XInterface = self.getXInterface()
        XMiddle = self.getXMiddle()
        nArray = self.getnProfil(lamb,XMiddle)
        penetrationIndex = self.getMaxXDepthIndex(nArray,thetaArray)

        kzArray = self.getkzArray(lamb,nArray,thetaArray)
        kzPrimeArray = self.getkzPrimeArray(lamb,nArray,thetaArray)
        # kzIntegral = self.kzIntegral(penetrationIndex,kzArray)
        integral1 = self.integral1(penetrationIndex,kzArray,kzPrimeArray)
        integral2 = self.integral2(penetrationIndex,kzArray,kzPrimeArray)

        rArray = np.zeros(thetaArray.shape,dtype = complex)

        for iTheta in range(thetaArray.size):
            
            if penetrationIndex[iTheta] == 0:
                rArray[iTheta] = 1
            else:
                # na = self.vacuum.getn(lamb)
                # kza = (2*np.pi/lamb) * np.sqrt(np.abs(na**2 - np.cos(thetaArray[iTheta])**2))
                # if penetrationIndex[iTheta] == -1:
                #     nb = self.substrate.getn(lamb)
                #     kzb = (2*np.pi/lamb) * np.sqrt(np.abs(nb**2 - np.cos(thetaArray[iTheta])**2))
                #     d = XInterface[-1] - XInterface[0]
                # else:
                #     kzb = 0
                #     d = XInterface[-1] - XInterface[-(penetrationIndex[iTheta]-1)]

                # rArray[iTheta] = (kzb - kza + d*kza*kzb - kzIntegral[iTheta])/ (kzb + kza - d*kza*kzb - kzIntegral[iTheta])
                rArray[iTheta] = integral2[iTheta]/(1+integral1[iTheta])

        return rArray



class Layer(object):
    def __init__(self,density,thickness,roughness = 0,atomicRatio = 0.5):
        self.thickness = thickness # nm
        self.density = density # g/cm^3
        self.roughness = roughness # nm
        self.atomicRatio = atomicRatio # nb proton/atomic mass (u)

    def setThickness(self,thickness):

        self.thickness = thickness

    def setDensity(self,density):

        self.density = density

    def setRoughness(self,roughness):

        self.roughness = roughness

    def getn(self,lamb):

        return 1 - deltaFactor*self.density*self.atomicRatio*lamb**2

    def getXInterface(self,sample):

        thickness = self.thickness
        roughness1 = sample.getPreviousLayer(self).roughness
        roughness2 = self.roughness
        if self.thickness < nSigmaRoughness*(roughness1 + roughness2):
            DX = np.linspace(0,thickness,2*roughnessDivision)
        else:
            if roughness1 == 0:
                DX1 = np.array([0,])
            else:
                DX1 = np.linspace(0,nSigmaRoughness*roughness1,roughnessDivision)
            if roughness2 == 0:
                DX2 = np.array([thickness,])
            else:
                DX2 = thickness + np.linspace(-nSigmaRoughness*roughness2,0,roughnessDivision)
            
            DX = np.append(DX1,DX2)
            
        X = sample.X0 + DX[1:]
        sample.X0 += self.thickness

        return X

    def getDensityProfil(self,sample,X0,X):

        previousRoughness = sample.getPreviousLayer(self).roughness
        actualRoughness = self.roughness

        densityProfil = np.ones(X.shape) * self.density

        if previousRoughness == 0:
            densityProfil[X < X0] *= 0
        else:
            densityProfil *= erfc(-(X-X0)/previousRoughness)/2

        if actualRoughness == 0:
            densityProfil[X > X0 + self.thickness] *= 0
        else:
            densityProfil *= erfc((X-(X0+self.thickness))/actualRoughness)/2
        
        return densityProfil

    def getDeltaProfil(self,sample,X0,X,lamb):

        densityProfil = self.getDensityProfil(sample,X0,X)

        deltaProfil = deltaFactor*densityProfil*self.atomicRatio*lamb**2
        
        return deltaProfil

class Substrate(Layer):
    def __init__(self,density,roughness = 0,atomicRatio = 0.5):
        Layer.__init__(self,density,0,roughness,atomicRatio)

    def getXInterface(self,sample):

        if self.roughness == 0:
            X = np.array([0])
        else:
            X = sample.X0 + np.linspace(-nSigmaRoughness*self.roughness,0,roughnessDivision)
        return X

    def getDensityProfil(self,sample,X0,X):

        actualRoughness = self.roughness

        densityProfil = np.ones(X.shape) * self.density

        if actualRoughness == 0:
            densityProfil[X > (X0 + self.thickness)] *= 0
        else:
            densityProfil *= erfc((X-X0)/actualRoughness)/2
        
        return densityProfil

class Vacuum(Layer):
    def __init__(self):
        Layer.__init__(self,0,0,0)

    def getXInterface(self,sample):
        lastRoughness = sample.getLastLayer().roughness
        if lastRoughness == 0:
            X = np.array([])
        else:
            DX = np.linspace(0,nSigmaRoughness*lastRoughness,roughnessDivision)
            X = sample.X0 + DX[1:]

        return X