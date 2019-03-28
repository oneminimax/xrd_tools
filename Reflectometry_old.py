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

    def getXLeftRight(self):

        XInterface = self.getXInterface()
        XLeftRight = np.zeros((len(XInterface)+1))

        XMiddle = (XInterface[:-1] + XInterface[1:])/2

        XLeftRight[0] = XMiddle[0] - (XInterface[1] - XInterface[0])
        XLeftRight[1:-1] = XMiddle
        XLeftRight[-1] = XMiddle[-1] + (XInterface[-1] - XInterface[-2])
        
        return XLeftRight

    def getDensityProfil(self,X):

        densityProfil = np.zeros(XLeftRight.shape)
        
        X0 = 0
        densityProfil += self.substrate.getDensityProfil(self,X0,X)
        
        for layer in self.layerList:
            densityProfil += layer.getDensityProfil(self,X0,X)
            X0 += layer.thickness

        return  densityProfil

    def getRefractionIndexProfil(self,lamb,X):

        XInterface = self.getXInterface()
        XLeftRight = self.getXLeftRight()

        deltaProfil = np.zeros(XLeftRight.shape)
        
        X0 = 0
        deltaProfil += self.substrate.getDeltaProfil(self,X0,X,lamb)
        
        for layer in self.layerList:
            deltaProfil += layer.getDeltaProfil(self,X0,X,lamb)
            X0 += layer.thickness

        nProfil = 1 - deltaProfil

        return nProfil

    def getInterfaceTransferMatrix(self,lamb,theta):

        XInterface = self.getXInterface()
        ZInterface = self.getTotalThickness() - XInterface
        nProfil = self.getRefractionIndexProfil(lamb)

        SArrayTop = SMatrixType0(nProfil[:-1],theta,lamb,ZInterface)
        SArrayBot = SMatrixType0(nProfil[1:],theta,lamb,ZInterface)
        invSArrayBot = 1/(SArrayBot[0,0,:,:]*SArrayBot[1,1,:,:]-SArrayBot[0,1,:,:]*SArrayBot[1,0,:,:])*np.array([[SArrayBot[1,1,:,:],-SArrayBot[0,1,:,:]],[-SArrayBot[1,0,:,:],SArrayBot[0,0,:,:]]])
        
        # TArray = np.einsum('abnt,bcnt->acnt',invSArrayBot,SArrayTop)
        TArray = np.zeros(invSArrayBot.shape,dtype = complex)
        for iTheta in range(TArray.shape[3]):
            for iX in range(TArray.shape[2]):
                TArray[:,:,iX,iTheta] = matmul(invSArrayBot[:,:,iX,iTheta],SArrayTop[:,:,iX,iTheta])

        penetrationIndex = PenetrationIndexFor(nProfil[1:],theta)

        return TArray, penetrationIndex

    def getRandT(self,lamb,theta):

        TransferArray, penetrationIndex = self.getInterfaceTransferMatrix(lamb,theta)
        penetrationBool = (penetrationIndex == -1)

        LArray = np.zeros((2,2,len(theta)),dtype = complex)
        TArray = np.zeros(theta.shape,dtype = complex)
        RArray = np.zeros(theta.shape,dtype = complex)

        TArray[penetrationIndex == 0] = 0
        RArray[penetrationIndex == 0] = 1

        for iTheta in range(len(theta)):
            if penetrationIndex[iTheta] == 1:
                TArray[iTheta] = 0
                RArray[iTheta] = 1
            else:
                if penetrationIndex[iTheta] == -1:
                    L = sequenceProductThetaFor(TransferArray[:,:,:,iTheta])
                    TArray[iTheta] = L[0,0] - L[0,1]*L[1,0]/L[1,1]
                    RArray[iTheta] = -L[1,0]/L[1,1]
                    LArray[:,:,iTheta] = L
                else:
                    L = sequenceProductThetaFor(TransferArray[:,:,-(penetrationIndex[iTheta]-1):,iTheta])
                    TArray[iTheta] = 0
                    RArray[iTheta] = -L[1,0]/L[1,1]
                    LArray[:,:,iTheta] = L
            
        return RArray, TArray, LArray

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


def SMatrixType0(nArray,thetaArray,lamb,z):

    kc = np.cos(thetaArray[None,:])/nArray[:,None]
    alpha = np.sqrt(1 - (kc)**2)
    kz = 2*np.pi/lamb*nArray[:,None]*alpha

    return np.array([[np.exp(1j * kz * z[:,None]),np.exp(-1j * kz * z[:,None])],[1j*kz*np.exp(1j * kz * z[:,None]),-1j*kz*np.exp(-1j * kz * z[:,None])]])

def PenetrationIndexFor(nArray,thetaArray):

    penetrationIndex = np.zeros(thetaArray.shape,dtype = int)
    kc = np.cos(thetaArray[None,:])/nArray[:,None]
    for iTheta in range(len(thetaArray)):
        penetrationBool = kc[:,iTheta] <= 1

        index = np.argwhere(np.flip(penetrationBool,axis = 0) == False)
        # print(np.flip(penetrationBool,axis = 0),index[0,0])
        if len(index)>0:
            penetrationIndex[iTheta] = index[0,0]
        else:
            penetrationIndex[iTheta] = -1

    return penetrationIndex

def PenetrationBool(nArray,thetaArray):

    kc = np.cos(thetaArray[None,:])/nArray[:,None]
    penetrationBool = np.all(kc < 1,axis = 0)
    # print(np.sum(np.all(kc < 1,axis = 0)) + np.sum(np.all(kc > 1,axis = 0)),thetaArray.size)

    return penetrationBool

def sequenceProductFor(matrixArray):

    LArray = np.zeros((2,2,1,matrixArray.shape[3]),dtype = complex)

    for iTheta in range(matrixArray.shape[3]):
        TotMatrix = np.eye(2)
        for iX in range(matrixArray.shape[2]-1,-1,-1):
            TotMatrix = matmul(matrixArray[:,:,iX,iTheta],TotMatrix)

        LArray[:,:,0,iTheta] = TotMatrix
            
    return LArray

def sequenceProductThetaFor(matrixArray):

    TotMatrix = np.eye(2)
    for iX in range(matrixArray.shape[2]-1,-1,-1):
        TotMatrix = matmul(matrixArray[:,:,iX],TotMatrix)
            
    return TotMatrix

def sequenceProduct(matrixArray):

    powerGroupList = sumPower2(matrixArray.shape[2])
    
    foldedMatrixArray = np.zeros((2,2,0,matrixArray.shape[3]),dtype = complex)
    lastChunkPosition = 0

    for iPower in range(len(powerGroupList)-1):
        foldedMatrixArray = np.concatenate((foldedMatrixArray,matrixArray[:,:,lastChunkPosition:lastChunkPosition + 2**powerGroupList[iPower],:]),axis = 2)
        lastChunkPosition += powerGroupList[iPower]
        for iFold in range(int(np.log2(foldedMatrixArray.shape[2]))-powerGroupList[iPower+1]):
            foldedMatrixArray = twinProduct(foldedMatrixArray)
            
    return foldedMatrixArray

def twinProduct(matrixArray):

    foldedMatrixArray = np.einsum('abnt,bcnt->acnt',matrixArray[:,:,0::2,:],matrixArray[:,:,1::2,:])

    return foldedMatrixArray


def sumPower2(number):

    powerList = list()
    for i in range (10):
        newPower = int(np.log2(number))
        powerList.append(newPower)
        number += -2**newPower
        if number == 0:
            powerList.append(0)
            break

    return np.array(powerList)

    



    


