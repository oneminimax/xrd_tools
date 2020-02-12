import numpy as np
from . import Geometry as Geo

class Space(object):
    def __init__(self,cristal_structure,z_hkl,x_pqr):
        self.cs = cristal_structure
        self.z_hkl = np.array(z_hkl)/self.cs.g_length(np.array(z_hkl))
        self.x_pqr = np.array(x_pqr)/self.cs.r_length(np.array(x_pqr))

        self.hkl_scale = 1
        self.pqr_scale = 1

    def hkl2xyz(self,hkl):

        x = self.cs.g_r_dot(hkl,self.x_pqr)
        y = self.cs.r_r_dot(self.cs.g_g_cross(hkl,self.z_hkl),self.x_pqr)
        z = self.cs.g_g_dot(hkl,self.z_hkl)

        return self.hkl_scale*np.array((x,y,z))

    def pqr2xyz(self,pqr):

        x = self.cs.r_r_dot(pqr,self.x_pqr)
        y = self.cs.g_r_dot(self.cs.r_r_cross(self.x_pqr,pqr),self.z_hkl)
        z = self.cs.g_r_dot(self.z_hkl,pqr)

        return self.pqr_scale*np.array((x,y,z))

    def angle2xyz(self,wave_length,two_theta,theta_source,phi):

        theta_detector = two_theta - theta_source
        return self.hkl_scale*np.array(Geo.theta_source_theta_detector_phi_2_wave_vector_x_y_z(wave_length,theta_source,theta_detector,phi))

        

