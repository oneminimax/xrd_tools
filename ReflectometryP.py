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

    def getTotalThickness(self):

        thickness = 0
        for layer in self.layerList:
            thickness += layer.thickness

        return thickness

    def getXInterface(self):

        XInterface = np.zeros((len(self.layerList)+1,))
        for i, layer in enumerate(reversed(self.layerList)):
            XInterface[i+1] = XInterface[i] + layer.thickness

        return XInterface

    def getThicknessLayer(self):

        ThicknessLayer = np.zeros((len(self.layerList),))
        for i, layer in enumerate(reversed(self.layerList)):
            ThicknessLayer[i] = layer.thickness

        return ThicknessLayer

    def getXLayer(self):

        XInterface = self.getXInterface()
        XLayer = (XInterface[:-1] + XInterface[1:])/2

        return XLayer

    def getnLayer(self,lamb):

        nLayer = np.zeros((len(self.layerList)+2,))
        nLayer[0] = self.vacuum.getn(lamb)
        for i, layer in enumerate(reversed(self.layerList)):
            nLayer[i+1] = layer.getn(lamb)
        nLayer[-1] = self.substrate.getn(lamb)

        return nLayer

    def getRoughnessInterface(self):
        
        RInterface = np.zeros((len(self.layerList)+1,))
        for i, layer in enumerate(reversed(self.layerList)):
            RInterface[i] = layer.roughness
        RInterface[-1] = self.substrate.roughness

        return RInterface

    def getkzLayer(self,lamb,thetaArray):

        nLayer = self.getnLayer(lamb)
        kzLayer = (2*np.pi/lamb) * np.sqrt(nLayer[:,None]**2 - np.cos(thetaArray[None,:])**2 + 0j)

        return kzLayer

    def getinterfaceMatrix(self,XLayer,kzLayer):

        R = self.getRoughnessInterface()
        
        kzp = kzLayer[1:,:] + kzLayer[:-1,:] 
        kzm = kzLayer[1:,:] - kzLayer[:-1,:]
        
        interfaceMatrix = np.zeros([2,2,XLayer.size,kzLayer.shape[1]],dtype = complex)
        interfaceMatrix[0,0,:,:] = kzp*np.exp(-1j*kzm*XLayer[:,None])*np.exp(-1/2*(kzm*R[:,None])**2)
        interfaceMatrix[0,1,:,:] = kzm*np.exp(-1j*kzp*XLayer[:,None])*np.exp(-1/2*(kzp*R[:,None])**2)
        interfaceMatrix[1,0,:,:] = kzm*np.exp(1j*kzp*XLayer[:,None])*np.exp(-1/2*(kzp*R[:,None])**2)
        interfaceMatrix[1,1,:,:] = kzp*np.exp(1j*kzm*XLayer[:,None])*np.exp(-1/2*(kzm*R[:,None])**2)

        return interfaceMatrix

    def getLArray(self,kzLayer,interfaceMatrix):

        LArray = np.zeros((2,2,kzLayer.shape[1]),dtype = complex)
        for iTheta in range(interfaceMatrix.shape[3]):
            LMatrix = np.eye(2)
            for iX in range(interfaceMatrix.shape[2]):
                LMatrix = matmul(interfaceMatrix[:,:,iX,iTheta],LMatrix)#/kzLayer[iX,iTheta]
            LArray[:,:,iTheta] = LMatrix

        return LArray

    def getRCoef(self,lamb,thetaArray):

        X = self.getXInterface()
        kzLayer = self.getkzLayer(lamb,thetaArray)
        
        interfaceMatrix = self.getinterfaceMatrix(X,kzLayer)
        LArray = self.getLArray(kzLayer, interfaceMatrix)
        
        RCoef = LArray[1,0,:]/LArray[1,1,:]

        return RCoef

    def getRCoefApprox(self,lamb,thetaArray):
        X = self.getXInterface()
        DX = np.diff(X)
        kzLayer = self.getkzLayer(lamb,thetaArray)
        # penetrationIndex = self.getPenetrationIndex(lamb,thetaArray)

        RCoef = np.zeros(thetaArray.shape)
        kzReal = np.real(kzLayer)
        for iTheta in range(thetaArray.size):
            # print(penetrationIndex[iTheta],kzReal[1:penetrationIndex[iTheta],iTheta]**2,DX[:penetrationIndex[iTheta]-1])
            ka = -kzReal[0,iTheta]
            kb = -kzReal[-1,iTheta]
            # if penetrationIndex[iTheta]<kzReal.shape[0]-1:
            #     kb = 0
            # else:
            #     kb = -kzReal[kzReal.shape[0]-1,iTheta]
            
            
            theta = thetaArray[iTheta]
            # kzlin = kzReal[1:penetrationIndex[iTheta],iTheta]
            # dxlin = DX[:penetrationIndex[iTheta]-1]
            kzlin = -kzReal[1:-1,iTheta]
            dxlin = DX
            d = np.sum(dxlin)
            # print(penetrationIndex[iTheta],kzReal[:,iTheta],kb)
            inte = np.sum(kzlin**2*dxlin)
            # print(penetrationIndex[iTheta],ka,kb,d,inte,kzlin,dxlin)
            RCoef[iTheta] = (kb-ka+d*kb*ka-inte)/(kb+ka-d*kb*ka-inte)

        return RCoef

class Material(object):
    def __init__(self,atomicMass,atomicNumber,unitCellVolume):

        self.atomicMass = atomicMass
        self.atomicNumber = atomicNumber
        self.unitCellVolume = unitCellVolume

class Layer(object):
    def __init__(self,density,thickness,roughness = 0,atomicMass = 2,atomicNumber = 1):
        self.thickness = thickness # nm
        self.density = density # g/cm^3
        self.roughness = roughness # nm
    
        self.atomicMass = atomicMass
        self.atomicNumber = atomicNumber

        self.atomicRatio = self.atomicNumber/self.atomicMass # nb proton/atomic mass (u)

    def setThickness(self,thickness):

        self.thickness = thickness

    def setDensity(self,density):

        self.density = density

    def getAtomicDensity(self):

        return self.density * 6.02214086e+24/10e+24/self.atomicMass

    def setRoughness(self,roughness):

        self.roughness = roughness

    def getn(self,lamb):

        self.atomicRatio = self.atomicNumber/self.atomicMass

        return 1 - deltaFactor*self.density*self.atomicRatio*lamb**2

class Substrate(Layer):
    def __init__(self,density,roughness = 0,atomicMass = 2,atomicNumber = 1):
        Layer.__init__(self,density,0,roughness,atomicMass,atomicNumber)

class Vacuum(Layer):
    def __init__(self):
        Layer.__init__(self,0,0,0,1)