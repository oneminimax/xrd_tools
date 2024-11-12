import numpy as np

from xrd_tools import geometry as geo


class CirstalInSpace(object):
    def __init__(self, cristal_structure, z_miller_index_hkl, x_position_index_pqr):
        self.cs = cristal_structure
        self.z_miller_index_hkl = np.array(z_miller_index_hkl)
        self.x_position_index_pqr = np.array(x_position_index_pqr)

    def hkl2xyz(self, hkl):
        x = self.cs.reciprocal_direct_dot(hkl, self.x_position_index_pqr) / self.cs.direct_length(
            np.array(self.x_position_index_pqr)
        )
        y = (
            self.cs.direct_dot(self.cs.reciprocal_cross(hkl, self.z_miller_index_hkl), self.x_position_index_pqr)
            / self.cs.direct_length(np.array(self.x_position_index_pqr))
            / self.cs.reciprocal_length(np.array(self.z_miller_index_hkl))
        )
        z = self.cs.reciprocal_dot(hkl, self.z_miller_index_hkl) / self.cs.reciprocal_length(
            np.array(self.z_miller_index_hkl)
        )

        return np.array((x, y, z))

    def pqr2xyz(self, pqr):
        x = self.cs.direct_dot(pqr, self.x_position_index_pqr) / self.cs.direct_length(
            np.array(self.x_position_index_pqr)
        )
        y = (
            self.cs.reciprocal_dot(self.z_miller_index_hkl, self.cs.direct_cross(self.x_position_index_pqr, pqr))
            / self.cs.direct_length(np.array(self.x_position_index_pqr))
            / self.cs.reciprocal_length(np.array(self.z_miller_index_hkl))
        )
        z = self.cs.reciprocal_direct_dot(self.z_miller_index_hkl, pqr) / self.cs.reciprocal_length(
            np.array(self.z_miller_index_hkl)
        )

        return np.array((x, y, z))

    def angle2xyz(self, wave_length, two_theta, theta_source, phi):
        theta_detector = two_theta - theta_source
        return np.array(
            geo.theta_source_theta_detector_phi_2_wave_vector_x_y_z(wave_length, theta_source, theta_detector, phi)
        )
