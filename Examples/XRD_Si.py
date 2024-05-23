# from XRDTools.CristalStructure import Diamond
# from XRDTools.Diffractometer import Diffractometer

# import numpy as np

# # Build unit cell. Using atomic number as atomic factor
# Si = Diamond(0.5431)
# Si.add_atom((0,0,0),1,label = 'Si')

# # Orientation of normal and in plane of the cristal
# hkl_z = (0,0,1)
# pqr_x = (1,0,0)

# # Put the sample in the diffractometer with the right orientation
# dm1 = Diffractometer(cristal_structure = Si, surface_normal_hkl = hkl_z,azimuth_pqr = pqr_x)
# dm2 = Diffractometer(wave_length = ,cristal_structure = Si, surface_normal_hkl = hkl_z,azimuth_pqr = pqr_x)

# # Show some peaks values
# HKL = (0,0,2)
# print('hkl : {0:d}{1:d}{2:d}, 2T : {3:.4f}, Source : {4:.4f}, Detector : {5:.4f}, Phi : {6:.4f}, Factor : {7:.3f}'.format(*HKL,dm1.hkl_2_two_theta(HKL),*dm1.hkl_2_theta_source_theta_detector_phi(HKL),Si.structure_factor(HKL)))

# HKL = (1,0,3)
# print('hkl : {0:d}{1:d}{2:d}, 2T : {3:.4f}, Source : {4:.4f}, Detector : {5:.4f}, Phi : {6:.4f}, Factor : {7:.3f}'.format(*HKL,dm1.hkl_2_two_theta(HKL),*dm1.hkl_2_theta_source_theta_detector_phi(HKL),Si.structure_factor(HKL)))

# HKL = (0,1,3)
# print('hkl : {0:d}{1:d}{2:d}, 2T : {3:.4f}, Source : {4:.4f}, Detector : {5:.4f}, Phi : {6:.4f}, Factor : {7:.3f}'.format(*HKL,dm1.hkl_2_two_theta(HKL),*dm1.hkl_2_theta_source_theta_detector_phi(HKL),Si.structure_factor(HKL)))
