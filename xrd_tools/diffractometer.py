import numpy as np

from xrd_tools import geometry


class Diffractometer(object):
    def __init__(self, wave_length=0.15406, cristal_structure=None, surface_normal_hkl=None, azimuth_pqr=None):
        """Construct a diffractometer object

        wave_length :           The wave lenght (in nm) of the x-ray source
        cristal_structure :     Cristal sructure of the sample
        surface_normal_hkl :    Miller index of the surface of the sample (along the z direction)
        azimuth_pqr :           Direct latice vector that is in the sample-source-detector plane.
                                Must be orthogonal to surface
        """
        self.wave_length = wave_length  # nanometer

        self.surface_normal_hkl = None
        self.azimuth_pqr = None

        if cristal_structure:
            self.set_cristal_structure(cristal_structure)
        if surface_normal_hkl:
            self.set_surface_normal_hkl(surface_normal_hkl)
        if azimuth_pqr:
            self.set_azimuth_pqr(azimuth_pqr)

    def set_cristal_structure(self, cristal_structure):
        self.cristal_structure = cristal_structure

    def get_cristal_structure(self):
        if self.cristal_structure is None:
            raise SampleError("Sample is not defined")
        else:
            return self.cristal_structure

    def set_surface_normal_hkl(self, surface_normal_hkl):
        if self.azimuth_pqr is None:
            self.surface_normal_hkl = surface_normal_hkl
        else:
            cristal_structure = self.get_cristal_structure()
            if cristal_structure.reciprocal_direct_dot(surface_normal_hkl, self.get_azimuth_pqr()) == 0:
                self.surface_normal_hkl = surface_normal_hkl
            else:
                raise ValueError("surface_normal_hkl must be orthogonal to azimuth_pqr")

    def get_surface_normal_hkl(self):
        if self.surface_normal_hkl is None:
            raise SampleError("Z Sample axis is not defined")
        else:
            return self.surface_normal_hkl

    def set_azimuth_pqr(self, azimuth_pqr):
        if self.surface_normal_hkl is None:
            self.azimuth_pqr = azimuth_pqr
        else:
            cristal_structure = self.get_cristal_structure()
            if cristal_structure.reciprocal_direct_dot(self.get_surface_normal_hkl(), azimuth_pqr) == 0:
                self.azimuth_pqr = azimuth_pqr
            else:
                raise ValueError("azimuth_pqr must be orthogonal to surface_normal_hkl")

    def get_azimuth_pqr(self):
        if self.azimuth_pqr is None:
            raise SampleError("X Sample axis is not defined")
        else:
            return self.azimuth_pqr

    def hkl_2_two_theta(self, hkl):
        cristal_structure = self.get_cristal_structure()

        return geometry.wave_vector_length_2_two_theta(self.wave_length, cristal_structure.reciprocal_length(hkl))

    def hkl_2_theta_source_theta_detector(self, hkl):
        two_theta = self.hkl_2_two_theta(hkl)
        cristal_structure = self.get_cristal_structure()
        surface_normal_hkl = self.get_surface_normal_hkl()
        alpha = cristal_structure.reciprocal_angle(hkl, surface_normal_hkl)

        theta_source = two_theta / 2 - alpha
        theta_detector = two_theta / 2 + alpha

        return theta_source, theta_detector

    def hkl_2_two_theta_omega_offset(self, hkl):
        two_theta = self.hkl_2_two_theta(hkl)
        cristal_structure = self.get_cristal_structure()
        surface_normal_hkl = self.get_surface_normal_hkl()
        alpha = cristal_structure.reciprocal_angle(hkl, surface_normal_hkl)

        return two_theta, alpha

    def hkl_2_theta_source_theta_detector_phi(self, hkl):
        theta_source, theta_detector = self.hkl_2_theta_source_theta_detector(hkl)

        cristal_structure = self.get_cristal_structure()
        surface_normal_hkl = self.get_surface_normal_hkl()
        azimuth_pqr = self.get_azimuth_pqr()

        if cristal_structure.reciprocal_angle(hkl, surface_normal_hkl) % 180 < 1e-2:
            phi = 0
            Warning("G Vector is along z axis")
        else:
            phi = cristal_structure.g_r_angle_on_plane_hkl(hkl, azimuth_pqr, surface_normal_hkl)

        return theta_source, theta_detector, phi

    def structure_factor(self, hkl):
        return self.get_cristal_structure().structure_factor(hkl)

    def compton_factor(self, hkl):
        two_theta = self.hkl_2_two_theta(hkl)
        return (1 + np.cos(np.deg2rad(two_theta)) ** 2) / 2

    def lorentz_factor(self, hkl):
        two_theta = self.hkl_2_two_theta(hkl)
        return 1 / np.sin(np.deg2rad(two_theta))

    def intensity(self, hkl):
        two_theta = self.hkl_2_two_theta(hkl)
        structure_factor = self.structure_factor(hkl)
        thompson_factor = (1 + np.cos(np.deg2rad(two_theta)) ** 2) / 2
        reciprocal_density_factor = 1 / np.sin(np.deg2rad(two_theta))

        return np.abs(structure_factor) ** 2 * thompson_factor * reciprocal_density_factor

    def get_absolute_coord_hkl(self, hkl):
        hkl = np.array(hkl)

        cristal_structure = self.get_cristal_structure()
        surface_normal_hkl = self.get_surface_normal_hkl()
        azimuth_pqr = self.get_azimuth_pqr()

        Z = cristal_structure.reciprocal_dot(hkl, surface_normal_hkl) / cristal_structure.GLength(surface_normal_hkl)
        X = cristal_structure.reciprocal_direct_dot(hkl, azimuth_pqr) / cristal_structure.RLength(azimuth_pqr)
        Y = cristal_structure.direct_dot(cristal_structure.GGcross(hkl, surface_normal_hkl), azimuth_pqr) / (
            cristal_structure.GLength(surface_normal_hkl) * cristal_structure.RLength(azimuth_pqr)
        )

        return np.array([X, Y, Z])

    def get_absolute_coord_pqr(self, pqr):
        pqr = np.array(pqr)

        cristal_structure = self.get_cristal_structure()
        surface_normal_hkl = self.get_surface_normal_hkl()
        azimuth_pqr = self.get_azimuth_pqr()

        Z = cristal_structure.reciprocal_direct_dot(surface_normal_hkl, pqr) / cristal_structure.GLength(
            surface_normal_hkl
        )
        X = cristal_structure.direct_dot(azimuth_pqr, pqr) / cristal_structure.RLength(azimuth_pqr)
        Y = cristal_structure.reciprocal_dot(surface_normal_hkl, cristal_structure.RRcross(azimuth_pqr, pqr)) / (
            cristal_structure.GLength(surface_normal_hkl) * cristal_structure.RLength(azimuth_pqr)
        )

        return np.array([X, Y, Z])


class SampleError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
