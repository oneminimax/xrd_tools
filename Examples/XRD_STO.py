from XRDTools.CristalStructure import Cubic
from XRDTools.Diffractometer import Diffractometer

import numpy as np

# Build unit cell. Using atomic number as atomic factor
STO = Cubic(0.3905)
STO.add_atom((0,0,0),38,label = 'Sr')
STO.add_atom((0.5,0.5,0.5),22,label = 'Ti')
STO.add_atom((0.5,0.5,0),8,label = 'O1')
STO.add_atom((0.5,0,0.5),8,label = 'O2')
STO.add_atom((0,0.5,0.5),8,label = 'O3')

# Orientation of normal and in plane of the cristal
hkl_z = (0,0,1)
pqr_x = (1,0,0)

# Put the sample in the diffractometer with the right orientation
dm = Diffractometer(cristal_structure = STO, surface_normal_hkl = hkl_z,azimuth_pqr = pqr_x)

# Show some peaks values
HKL = (0,0,2)
print('hkl : {0:d}{1:d}{2:d}, 2T : {3:.4f}, Source : {4:.4f}, Detector : {5:.4f}, Phi : {6:.4f}, Factor : {7:.3f}'.format(*HKL,dm.hkl_2_two_theta(HKL),*dm.hkl_2_theta_source_theta_detector_phi(HKL),STO.structure_factor(HKL)))

HKL = (1,0,3)
print('hkl : {0:d}{1:d}{2:d}, 2T : {3:.4f}, Source : {4:.4f}, Detector : {5:.4f}, Phi : {6:.4f}, Factor : {7:.3f}'.format(*HKL,dm.hkl_2_two_theta(HKL),*dm.hkl_2_theta_source_theta_detector_phi(HKL),STO.structure_factor(HKL)))

HKL = (0,1,3)
print('hkl : {0:d}{1:d}{2:d}, 2T : {3:.4f}, Source : {4:.4f}, Detector : {5:.4f}, Phi : {6:.4f}, Factor : {7:.3f}'.format(*HKL,dm.hkl_2_two_theta(HKL),*dm.hkl_2_theta_source_theta_detector_phi(HKL),STO.structure_factor(HKL)))
