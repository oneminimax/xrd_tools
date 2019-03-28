from XRDTools.CristalStructure import Cubic
from XRDTools.Diffractometer import Diffractometer

import numpy as np

STO = Cubic(0.3905)

hkl_z = (0,0,1)

dm = Diffractometer(cristal_structure = STO, surface_normal_hkl = hkl_z)

HKL = (0,0,2)
print('hkl : {0:d}{1:d}{2:d}, 2T : {3:.4f}, Source : {4:.4f}, Detector : {5:.4f}'.format(*HKL,dm.hkl_2_two_theta(HKL),*dm.hkl_2_theta_source_theta_detector(HKL)))


