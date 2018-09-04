import numpy as np

class GeneralStructure(object):
    def __init__(self,aLengthArray,aAngleArray):

        '''
        aLengthArray : lattice parameter in nm
        aAngleArray : lattice angles in deg (converted in rad) (23),(13),(12)

        '''

        self.aLengthArray = np.array(aLengthArray)
        self.aAngleArray = np.deg2rad(np.array(aAngleArray))
        self.angleMat = np.array([[0,self.aAngleArray[2],self.aAngleArray[1]],[self.aAngleArray[2],0,self.aAngleArray[0]],[self.aAngleArray[1],self.aAngleArray[0],0]])

        self._precalc_aa_dot_mat()
        self._precalc_bb_dot_mat()      

        self.atomList = list()

    def getAtomList(self):

        return self.atomList

    def addAtom(self,position,formFactor):

        self.atomList.append(Atom(position,formFactor))

    def structureFactor(self,hkl):

        hkl = np.array(hkl)

        factor = 0 + 0j

        for atom in self.getAtomList():
            factor += np.exp(1j*2*np.pi*np.dot(hkl,atom.position))

        return factor

    def aLength(self,i):

        '''
        Get lattice parameter, cyclic function
        '''

        i = i%3
        return self.aLengthArray[i]

    def aaAngle(self,i,j):

        '''
        Get angle between lattice vectors
        '''

        i = i%3
        j = j%3
        return self.angleMat[i,j]

    def aCellVolume(self):

        '''
        Unit cell volume
        '''

        return np.prod(self.aLengthArray)*np.sqrt(1 + 2*np.prod(np.cos(self.aAngleArray)) - np.sum(np.cos(self.aAngleArray)**2))

    def RLength(self,pqr):

        '''
        Length of a direct vector in PQR coordinates
        '''

        return np.sqrt(self.RRdot(pqr,pqr))

    def RRAngle(self,pqr,pqrp):

        '''
        Get angle between two direct vectors
        '''

        return np.rad2deg(np.arccos(self.RRdot(pqr,pqrp) / np.sqrt( self.RRdot(pqr,pqr) * self.RRdot(pqrp,pqrp) )))

    def bLength(self,i):


        '''
        Get reciprocal lattice parameter, cyclic function
        '''

        i = i%3
        return np.sqrt(self.bb_dot_mat[i,i])

    def bbAngle(self,i,j):

        '''
        Get angle between reciprocal lattice vectors
        '''

        return np.rad2deg(np.arccos(self.bb_dot_mat[i,j] * np.sqrt(1/(self.bb_dot_mat[i,i]*self.bb_dot_mat[j,j]))))

    def GLength(self,hkl):

        '''
        Get length of reciprocal vector
        '''

        return np.sqrt(self.GGdot(hkl,hkl))

    def GGAngle(self,hkl,hklp):

        '''
        Get angle between two reciprocal vectors
        '''

        return np.rad2deg(np.arccos(self.GGdot(hkl,hklp) / np.sqrt( self.GGdot(hkl,hkl) * self.GGdot(hklp,hklp) )))

    def RRAngle(self,pqr,pqrp):

        '''
        Get angle between two direct vectors
        '''

        return np.rad2deg(np.arccos(self.RRdot(pqr,pqrp) / np.sqrt( self.RRdot(pqr,pqr) * self.RRdot(pqrp,pqrp) )))

    def GRAngle(self,hkl,pqr):

        '''
        Get angle between two reciprocal/direct vectors
        '''

        return np.rad2deg(np.arccos(self.GRdot(hkl,pqr) / np.sqrt( self.GGdot(hkl,hkl) * self.RRdot(pqr,pqr) )))

    def dSpacingLength(self,hkl):

        '''
        Spacing between plane of familly hkl
        '''

        return 2*np.pi/self.GLength(hkl)

    def _precalc_aa_dot_mat(self):

        self.aa_dot_mat = self.aLengthArray[:,None] * self.aLengthArray[None,:] * np.cos(self.angleMat)

    def _precalc_bb_dot_mat(self):

        aa_dot_mat = self.aa_dot_mat

        aaaa_cdc = aa_dot_mat[:,:,None,None] * aa_dot_mat[None,None,:,:] - aa_dot_mat[:,None,None,:] * aa_dot_mat[None,:,:,None]

        eijk = np.zeros((3, 3, 3))
        eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
        eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1

        self.bb_dot_mat = (np.pi/self.aCellVolume())**2 * np.einsum('imn,jmn',eijk,np.einsum('jpq,mpnq',eijk,aaaa_cdc))

    def GGdot(self,HKL,hkl):

        HKL = np.array(HKL)
        hkl = np.array(hkl)

        prod = np.sum(HKL[:,None]*hkl[None,:]*self.bb_dot_mat)

        return prod

    def RRdot(self,PQR,pqr):

        PQR = np.array(PQR)
        pqr = np.array(pqr)

        prod = np.sum(PQR[:,None]*pqr[None,:]*self.aa_dot_mat)

        return prod

    def GRdot(self,hkl,pqr):

        hkl = np.array(hkl)
        pqr = np.array(pqr)

        prod = 2 * np.pi * np.dot(hkl,pqr)

        return prod

    def GGcross(self,HKL,hkl):

        HKL = np.array(HKL)
        hkl = np.array(hkl)

        eijk = np.zeros((3, 3, 3))
        eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
        eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1

        hh = HKL[:,None] * hkl[None,:]

        PQR = (2 * np.pi)**2 / self.aCellVolume() * np.einsum('ijk,ij',eijk,hh)

        return PQR

    def RRcross(self,PQR,pqr):

        PQR = np.array(PQR)
        pqr = np.array(pqr)

        eijk = np.zeros((3, 3, 3))
        eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
        eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1

        pp = PQR[:,None] * pqr[None,:]

        HKL = self.aCellVolume()/(2*np.pi) * np.einsum('ijk,ij',eijk,pp)

        return HKL

    def projGonG(self,HKL,hkl):

        HKL = np.array(HKL)
        hkl = np.array(hkl)

        return self.GGdot(HKL,hkl)/self.GGdot(hkl,hkl) * hkl

    def projRonR(self,PQR,pqr):

        PQR = np.array(PQR)
        pqr = np.array(pqr)

        return self.RRdot(PQR,pqr)/self.RRdot(pqr,pqr) * pqr

    def projGonR(self,hkl,pqr):

        hkl = np.array(hkl)
        pqr = np.array(pqr)

        return self.GRdot(hkl,pqr)/self.RRdot(pqr,pqr) * pqr

    def projRonG(self,pqr,hkl):

        pqr = np.array(pqr)
        hkl = np.array(hkl)

        return self.GRdot(hkl,pqr)/self.GGdot(hkl,hkl) * hkl

    def RtoG(self,pqr):

        pqr = np.array(pqr)

        return 1/(2*np.pi)*np.dot(self.aa_dot_mat,pqr)

    def GtoR(self,hkl):

        hkl = np.array(hkl)

        return 1/(2*np.pi)*np.dot(self.bb_dot_mat,hkl)

    def GGAngle_onplaneHKL(self,hkl,hklp,HKL):

        '''
        Get angle between two reciprocal vectors projected on a plane perpendicular to vector HKL
        '''

        HKL = np.array(HKL)
        hkl = np.array(hkl)
        hklp = np.array(hklp)

        hkl_onplane = hkl - self.projGonG(hkl,HKL)
        hklp_onplane = hklp - self.projGonG(hklp,HKL)

        angle = self.GGAngle(hkl_onplane,hklp_onplane)
        direction = np.sign(self.GGdot(self.GGcross(hkl,hklp),HKL))

        return angle * direction

    def GRAngle_onplaneHKL(self,hkl,pqr,HKL):

        '''
        Get angle between two vectors (reciprocal and direct) projected on a plane perpendicular to vector HKL
        '''

        HKL = np.array(HKL)
        hkl = np.array(hkl)
        pqr = np.array(pqr)

        hkl_onplane = hkl - self.projGonG(hkl,HKL)
        pqr_onplane = pqr - self.GtoR(self.projRonG(pqr,HKL))

        angle = self.GRAngle(hkl_onplane,pqr_onplane)
        direction = np.sign(self.GGdot(self.GGcross(hkl,self.RtoG(pqr)),HKL))

        return angle * direction

class Cubic(GeneralStructure):
    def __init__(self,aLength):

        super().__init__((aLength,aLength,aLength),(90,90,90))

class Diamond(Cubic):

    def __init__(self,aLength):

        super().__init__(aLength)

    def getAtomList(self):

        atomList = list()
        for atom in self.atomList:
            for v1 in [np.array((0,0,0)),np.array((0.5,0.5,0)),np.array((0,0.5,0.5)),np.array((0.5,0,0.5))]:
                for v2 in [np.array((0,0,0)),np.array((0.25,0.25,0.25))]:
                    atomList.append(Atom(v1 + v2 + atom.position, atom.formFactor))

        return atomList


class Atom(object):

    def __init__(self,position,formFactor):

        self.position = np.array(position)
        self.formFactor = formFactor
