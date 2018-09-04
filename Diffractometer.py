import numpy as np

class Diffractometer(object):
    def __init__(self,waveLength = 0.15406,cristalStructure = None,HKL_z = None,PQR_x = None):
        
        self.waveLength = waveLength
        self.boolCS = False
        self.boolHKL_z = False
        self.boolPQR_x = False
        if cristalStructure:
            self.setSample(cristalStructure)
        if HKL_z:
            self.setHKL_z(HKL_z)
        if PQR_x:
            self.setPQR_x(PQR_x)

    # Get/Set Sample definition, orientation

    def setSample(self,cristalStructure):

        self.CS = cristalStructure
        self.boolCS = True

    def getSample(self):

        if self.boolCS:
            return self.CS
        else:
            raise SampleError('Sample is not defined')

    def setHKL_z(self,HKL_z):

        if self.boolPQR_x:
            CS = self.getSample()
            if CS.GRdot(HKL_z,self.getPQR_x()) == 0:
                self.HKL_z = HKL_z
                self.boolHKL_z = True
            else:
                raise ValueError('HKL_z must be orthogonal to PQR_x')
        else:
            self.HKL_z = HKL_z
            self.boolHKL_z = True

    def getHKL_z(self):
        if self.boolHKL_z:
            return self.HKL_z
        else:
            raise SampleError('Z Sample axis is not defined')

    def setPQR_x(self,PQR_x):

        if self.boolHKL_z:
            CS = self.getSample()
            if CS.GRdot(self.getHKL_z(),PQR_x) == 0:
                self.PQR_x = PQR_x
                self.boolPQR_x = True
            else:
                raise ValueError('PQR_x must be orthogonal to HKL_z')
        else:
            self.PQR_x = PQR_x
            self.boolPQR_x = True
    
    def getPQR_x(self):
        if self.boolPQR_x:
            return self.PQR_x
        else:
            raise SampleError('X Sample axis is not defined')

    # Geometry

    def TwoTheta2GLength(self,twoTheta):

        return 4*np.pi*np.sin(np.deg2rad(twoTheta)/2)/self.waveLength

    def GLength2TwoTheta(self,GLength):

        return 2*np.rad2deg(np.arcsin(self.waveLength/(4*np.pi)*GLength))

    # Peak to angles

    def HKL2TwoTheta(self,hkl):

        CS = self.getSample()

        return self.GLength2TwoTheta(CS.GLength(hkl))
        
    def HKL2ThetaSourceThetaDetector(self,hkl):

        TT = self.HKL2TwoTheta(hkl)
        CS = self.getSample()
        HKL_z = self.getHKL_z()
        alpha = CS.GGAngle(hkl,HKL_z)

        thetaSource = TT/2 - alpha
        thetaDetector = TT/2 + alpha

        return thetaSource, thetaDetector

    def HKL2ThetaSourceThetaDetectorPhi(self,hkl):

        thetaSource, thetaDetector = self.HKL2ThetaSourceThetaDetector(hkl)

        CS = self.getSample()
        HKL_z = self.getHKL_z()
        PQR_x = self.getPQR_x()

        if CS.GGAngle(hkl,HKL_z) % 180 < 1e-2:
            phi = 0
            Warning('G Vector is along z axis')
        else:
            phi = CS.GRAngle_onplaneHKL(hkl,PQR_x,HKL_z)

        return thetaSource, thetaDetector, phi

    # Absolute coordinate

    def getAbsoluteCoordHKL(self,hkl):

        hkl = np.array(hkl)

        CS = self.getSample()
        HKL_z = self.getHKL_z()
        PQR_x = self.getPQR_x()

        Z = CS.GGdot(hkl,HKL_z) / CS.GLength(HKL_z)
        X = CS.GRdot(hkl,PQR_x) / CS.RLength(PQR_x)
        Y = CS.RRdot(CS.GGcross(hkl,HKL_z),PQR_x) / (CS.GLength(HKL_z) * CS.RLength(PQR_x))

        return np.array([X,Y,Z])

    def getAbsoluteCoordPQR(self,pqr):

        pqr = np.array(pqr)

        CS = self.getSample()
        HKL_z = self.getHKL_z()
        PQR_x = self.getPQR_x()

        Z = CS.GRdot(HKL_z,pqr)/CS.GLength(HKL_z)
        X = CS.RRdot(PQR_x,pqr)/CS.RLength(PQR_x)
        Y = CS.GGdot(HKL_z,CS.RRcross(PQR_x,pqr)) / (CS.GLength(HKL_z) * CS.RLength(PQR_x))

        return np.array([X,Y,Z])


class SampleError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)



