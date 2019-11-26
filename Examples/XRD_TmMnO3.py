from XRDTools.CristalStructure import Hexagonal
from XRDTools.Diffractometer import Diffractometer
from XRDTools.Geometry import two_theta_2_wave_vector_length
from XRDTools import wave_length

import numpy as np

TMO = Hexagonal(0.608,1.137)
# TMO = Hexagonal(0.6144,1.1490)
TMO.add_atom((0.666667, 0.333333, 0.728014),69,'Tm')
TMO.add_atom((0.000000, 0.000000, 0.771913),69,'Tm')
TMO.add_atom((0.333333, 0.666667, 0.228014),69,'Tm')
TMO.add_atom((0.666667, 0.333333, 0.228014),69,'Tm')
TMO.add_atom((0.000000, 0.000000, 0.271913),69,'Tm')
TMO.add_atom((0.333333, 0.666667, 0.728014),69,'Tm')
TMO.add_atom((0.666622, 0.666622, 0.997194),25,'MN')
TMO.add_atom((0.000000, 0.333378, 0.997194),25,'MN')
TMO.add_atom((0.000000, 0.666622, 0.497194),25,'MN')
TMO.add_atom((0.333378, 0.000000, 0.997194),25,'MN')
TMO.add_atom((0.666622, 0.000000, 0.497194),25,'MN')
TMO.add_atom((0.333378, 0.333378, 0.497194),25,'MN')
TMO.add_atom((0.666667, 0.333333, 0.519377),8,'O')
TMO.add_atom((0.640025, 0.640025, 0.831484),8,'O')
TMO.add_atom((0.359975, 0.000000, 0.831484),8,'O')
TMO.add_atom((0.693602, 0.000000, 0.663081),8,'O')
TMO.add_atom((0.000000, 0.359975, 0.831484),8,'O')
TMO.add_atom((0.333333, 0.666667, 0.519377),8,'O')
TMO.add_atom((0.000000, 0.306398, 0.163081),8,'O')
TMO.add_atom((0.359975, 0.359975, 0.331484),8,'O')
TMO.add_atom((0.000000, 0.640025, 0.331484),8,'O')
TMO.add_atom((0.306398, 0.306398, 0.663081),8,'O')
TMO.add_atom((0.693602, 0.693602, 0.163081),8,'O')
TMO.add_atom((0.000000, 0.000000, 0.971025),8,'O')
TMO.add_atom((0.306398, 0.000000, 0.163081),8,'O')
TMO.add_atom((0.333333, 0.666667, 0.019377),8,'O')
TMO.add_atom((0.000000, 0.693602, 0.663081),8,'O')
TMO.add_atom((0.640025, 0.000000, 0.331484),8,'O')
TMO.add_atom((0.000000, 0.000000, 0.471025),8,'O')
TMO.add_atom((0.666667, 0.333333, 0.019377),8,'O')

hkl_z = (0,0,1)
dm = Diffractometer(cristal_structure = TMO, surface_normal_hkl = hkl_z)

# HKL = (0,0,2)
# print('hkl : {0:d}{1:d}{2:d}, 2T : {3:.4f}, Source : {4:.4f}, Detector : {5:.4f}, Phi : {6:.4f}, Factor : {7:.3f}'.format(*HKL,dm.hkl_2_two_theta(HKL),*dm.hkl_2_theta_source_theta_detector(HKL),0,np.abs(TMO.structure_factor(HKL))**2))
HKL = (0,0,4)
print('hkl : {0:d}{1:d}{2:d}, 2T : {3:.4f}, Source : {4:.4f}, Detector : {5:.4f}, Phi : {6:.4f}, Factor : {7:.3f}'.format(*HKL,dm.hkl_2_two_theta(HKL),*dm.hkl_2_theta_source_theta_detector(HKL),0,np.abs(TMO.structure_factor(HKL))**2))
HKL = (1,1,8)
print('hkl : {0:d}{1:d}{2:d}, 2T : {3:.4f}, Source : {4:.4f}, Detector : {5:.4f}, Phi : {6:.4f}, Factor : {7:.3f}'.format(*HKL,dm.hkl_2_two_theta(HKL),*dm.hkl_2_theta_source_theta_detector(HKL),0,np.abs(TMO.structure_factor(HKL))**2))
HKL = (-1,1,8)
print('hkl : {0:d}{1:d}{2:d}, 2T : {3:.4f}, Source : {4:.4f}, Detector : {5:.4f}, Phi : {6:.4f}, Factor : {7:.3f}'.format(*HKL,dm.hkl_2_two_theta(HKL),*dm.hkl_2_theta_source_theta_detector(HKL),0,np.abs(TMO.structure_factor(HKL))**2))
HKL = (0,1,8)
print('hkl : {0:d}{1:d}{2:d}, 2T : {3:.4f}, Source : {4:.4f}, Detector : {5:.4f}, Phi : {6:.4f}, Factor : {7:.3f}'.format(*HKL,dm.hkl_2_two_theta(HKL),*dm.hkl_2_theta_source_theta_detector(HKL),0,np.abs(TMO.structure_factor(HKL))**2))


# HKL = (0,0,8)
# print('hkl : {0:d}{1:d}{2:d}, 2T : {3:.4f}, Source : {4:.4f}, Detector : {5:.4f}, Phi : {6:.4f}, Factor : {7:.3f}'.format(*HKL,dm.hkl_2_two_theta(HKL),*dm.hkl_2_theta_source_theta_detector(HKL),0,np.abs(TMO.structure_factor(HKL))**2))

print(4*2*np.pi/two_theta_2_wave_vector_length(wave_length,31.415))
