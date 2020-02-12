import XRDTools.CristalStructure as CS
from XRDTools.Diffractometer import Diffractometer
import numpy as np

# Herbet = CS.GeneralStructure([0.68263,0.68263, 1.4066],[90,90,120])
# TaAs = CS.GeneralStructure([0.34348,0.34348,1.1641],[90,90,90])
# Str = CS.GeneralStructure([(2*np.pi),(2*np.pi),(2*np.pi)],[90,90,90])

# hkl_z = (0,0,1)
# pqr_x = (1,0,0)

# dm = Diffractometer(cristalStructure = Str, HKL_z = hkl_z, PQR_x = pqr_x)

# HKL1 = (0,0,1)
# HKL2 = (1,0,0)

# cHKL1 = dm.getAbsoluteCoordHKL(HKL1)
# cHKL2 = dm.getAbsoluteCoordHKL(HKL2)

# print(cHKL1,Str.GLength(HKL1),np.sqrt(np.sum(cHKL1**2)))
# print(cHKL2,Str.GLength(HKL2),np.sqrt(np.sum(cHKL2**2)))

# print(TaAs.aaprod)
# print(TaAs.bbprod)

# print(TaAs.RRdot((1,0,0),(1,0,0)))
# print(TaAs.RLength((0,1,0)))
# print(TaAs.GGcross((1,0,0),(0,1,0)))

# print(Str.GGcross((1,0,0),(0,1,0)))
# print(TaAs._calc_bbprod(2,2))


def testCristalStructure():
    Factor = 1
    cs = CS.GeneralStructure([0.7298*Factor,0.851*Factor,0.5653],[90,90,100])
    
    hkl = (1,1,0)
    hklp = (0,1,0)
    HKL = (0,0,1)

    print(cs.GRAngle((1,0,0),(1,0,0)))

    print(cs.GGAngle(hkl,hklp))
    print(cs.projGonG(hkl,HKL))
    print(cs.GGAngle_onplaneHKL(hkl,hklp,HKL))


def testCristalStructureSTO():
    Factor = 1
    cs = CS.Cubic(0.5568)
    
    cs.addAtom((0.5,0.5,0.5),'Sr')
    cs.addAtom((0,0,0),'Ti')
    cs.addAtom((0.5,0,0),'O')
    cs.addAtom((0,0.5,0),'O')
    cs.addAtom((0,0,0.5),'O')
    for atom in cs.getAtomList():
        print(atom)

    print(np.abs(cs.structureFactor((1,0,0)))**2)
    print(np.abs(cs.structureFactor((1,1,2)))**2)
    print(np.abs(cs.structureFactor((2,0,0)))**2)
    print(np.abs(cs.structureFactor((1,0,0)))**2/np.abs(cs.structureFactor((1,1,2)))**2)

def testCristalStructureSilcium():
    
    cs = CS.Diamond(0.543095)
    
    cs.addAtom((0,0,0),'Siv')
    for atom in cs.getAtomList():
        print(atom)

    print(np.abs(cs.structureFactor((1,1,1))))
    print(np.abs(cs.structureFactor((2,2,0))))

def testCristalStructureNiobium():
    
    cs = CS.CubicBodyCentered(0.33004)
    
    cs.addAtom((0,0,0),'Nb')
    for atom in cs.getAtomList():
        print(atom)


def testCristalStructureNiobiumNitride():
    
    cs = CS.CubicFaceCentered(0.33004)
    
    cs.addAtom((0,0,0),'Nb')
    cs.addAtom((0.5,0,0),'N')
    # for atom in cs.getAtomList():
    #     print(atom)

    # print()
    print(np.sum(cs.structureFactor((2,0,0))))
    print(np.sum(cs.structureFactor((1,1,1))))
    # print((np.abs(/cs.structureFactor((1,1,1))))**2)


def test_wickoff():

    cs = CS.No139(1,2)

    cs.add_atom_wickoff('a',1,label = 'a1')
    cs.add_atom_wickoff('e',2-1j,label = 'a2',z = 0.1)

    cs.show_atoms()


if __name__ == '__main__':
    # testCristalStructure()
    # testCristalStructureSTO()
    # testCristalStructureSilcium()
    # testCristalStructureNiobiumNitride()
    # calc()
    test_wickoff()
