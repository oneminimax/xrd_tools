from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from xrd_tools.cristal_structure.form_factor import ITCFct
from xrd_tools.cristal_structure.wyckoff import Position

LEVI_CIVITA = np.zeros((3, 3, 3))
LEVI_CIVITA[0, 1, 2] = LEVI_CIVITA[1, 2, 0] = LEVI_CIVITA[2, 0, 1] = 1
LEVI_CIVITA[0, 2, 1] = LEVI_CIVITA[2, 1, 0] = LEVI_CIVITA[1, 0, 2] = -1


class GeneralStructure(object):

    """General notation :

    a     : lattice parameters (ex : a1, a2, etc)
    angle : angles between lattice base vector (ex : angle3 is angle between a1 and a2)
    b     : reciprocal lattice parameters (ex : b1, b2, etc)
    r     : direct space vector
    position_index_pqr   : components of a direct space vector in the lattice base
    g     : reciprocal space vector
    miller_index_hkl   : components of a reciprocal space vector in the reciprocal lattice base
    """

    base_wyckoff_positions = "(0,0,0)"
    wyckoff_positions = {"a": "(x,y,z)"}

    def __init__(self, lattice_distances: Tuple[float, float, float], lattice_angles: Tuple[float, float, float]):
        """
        lattice_distances : lattice parameter (a,b,c) in nm
        lattice_angles : lattice angles (alpha (b-c), beta (c-a), gamma (a-b)) in deg (converted in rad) (23),(13),(12)
        """

        assert len(lattice_angles) == 3
        assert len(lattice_angles) == 3

        self.lattice_distances = np.array(lattice_distances)
        self.lattice_angles = np.deg2rad(np.array(lattice_angles))
        self.angles_matrix = np.array(
            [
                [0, self.lattice_angles[2], self.lattice_angles[1]],
                [self.lattice_angles[2], 0, self.lattice_angles[0]],
                [self.lattice_angles[1], self.lattice_angles[0], 0],
            ]
        )

        self._precalc_direct_dot_matrix()
        self._precalc_reciprocal_dot_matrix()

        self.atoms = list()

    def _precalc_direct_dot_matrix(self):
        self.direct_dot_matrix = (
            self.lattice_distances[:, None] * self.lattice_distances[None, :] * np.cos(self.angles_matrix)
        )

    def _precalc_reciprocal_dot_matrix(self):
        aa_dot_mat = self.direct_dot_matrix

        aaaa_cdc = (
            aa_dot_mat[:, :, None, None] * aa_dot_mat[None, None, :, :]
            - aa_dot_mat[:, None, None, :] * aa_dot_mat[None, :, :, None]
        )

        self.reciprocal_dot_matrix = (np.pi / self.direct_unit_cell_volume()) ** 2 * np.einsum(
            "imn,jmn", LEVI_CIVITA, np.einsum("jpq,mpnq", LEVI_CIVITA, aaaa_cdc)
        )

    def __str__(self):
        return "General structure : \n a = {0:.4f}, b = {1:.4f}, c = {2:.4f}\n alpha = {3:.4f}, beta = {4:.4f}, gamma = {5:.4f}".format(
            *self.lattice_distances, *np.rad2deg(self.lattice_angles)
        )

    def list_atoms(self):
        for atom in self.atoms:
            print(atom)

    def get_atoms(self):
        return self.atoms

    def add_atom_wickoff(self, letter: str, form_factor: float, label: str = "", **kwargs):
        wyckoff_position = Position.generate_from_string(self.wyckoff_positions[letter], self.base_wyckoff_positions)
        coordinates = wyckoff_position(**kwargs)
        for coord in coordinates:
            self._add_atom(coord, form_factor, label)

    def add_atom(self, coordinates: Tuple[float, float, float], form_factor: float, label: str = ""):
        self._add_atom(coordinates, form_factor, label)

    def _add_atom(self, coordinates: Tuple[float, float, float], form_factor: float, label: str = ""):
        self.atoms.append(Atom(coordinates, label=label, form_factor=form_factor))

    def structure_factor(self, miller_index_hkl: Tuple[int, int, int]):
        miller_index_hkl = np.array(miller_index_hkl)
        hkl_reciprocal_length = self.reciprocal_length(miller_index_hkl)

        atoms = self.get_atoms()

        form_factors = np.zeros((len(atoms),))
        positions = np.zeros((3, len(atoms)))

        for i, atom in enumerate(atoms):
            form_factors[i] = atom.get_form_factor(hkl_reciprocal_length)
            positions[:, i] = atom.coordinates

        factor = np.sum(form_factors * np.exp(1j * 2 * np.pi * np.dot(miller_index_hkl, positions)))

        return factor

    def debye_waller_factor(self, miller_index_hkl: Tuple[int, int, int]):
        miller_index_hkl = np.array(miller_index_hkl)
        hkl_reciprocal_length = self.reciprocal_length(miller_index_hkl)

    def direct_lattice_length(self, i: int):
        """
        Get lattice parameter, cyclic function
        """

        return self.lattice_distances[i % 3]

    def direct_lattice_angle(self, i: int, j: int):
        """
        Get angle between lattice vectors
        """

        return self.angles_matrix[i % 3, j % 3]

    def direct_unit_cell_volume(self):
        """
        Unit cell volume
        """

        return np.prod(self.lattice_distances) * np.sqrt(
            1 + 2 * np.prod(np.cos(self.lattice_angles)) - np.sum(np.cos(self.lattice_angles) ** 2)
        )

    def direct_length(self, position_index_pqr: Tuple[int, int, int]):
        """
        Length of a direct vector in PQR coordinates
        """

        return np.sqrt(self.direct_dot(position_index_pqr, position_index_pqr))

    def direct_angle(self, position_index_pqr1: Tuple[int, int, int], position_index_pqr2: Tuple[int, int, int]):
        """
        Get angle between two direct vectors
        """

        return np.rad2deg(
            np.arccos(
                self.direct_dot(position_index_pqr1, position_index_pqr2)
                / np.sqrt(
                    self.direct_dot(position_index_pqr1, position_index_pqr1)
                    * self.direct_dot(position_index_pqr2, position_index_pqr2)
                )
            )
        )

    def reciprocal_lattice_length(self, i: int):
        """
        Get reciprocal lattice parameter, cyclic function
        """

        i = i % 3
        return np.sqrt(self.reciprocal_dot_matrix[i, i])

    def reciprocal_lattice_angle(self, i: int, j: int):
        """
        Get angle between reciprocal lattice vectors
        """

        return np.rad2deg(
            np.arccos(
                self.reciprocal_dot_matrix[i, j]
                * np.sqrt(1 / (self.reciprocal_dot_matrix[i, i] * self.reciprocal_dot_matrix[j, j]))
            )
        )

    def reciprocal_length(self, miller_index_hkl: Tuple[int, int, int]):
        """
        Get length of reciprocal vector
        """

        return np.sqrt(self.reciprocal_dot(miller_index_hkl, miller_index_hkl))

    def reciprocal_angle(self, miller_index_hkl1: Tuple[int, int, int], miller_index_hkl2: Tuple[int, int, int]):
        """
        Get angle between two reciprocal vectors
        """

        return np.rad2deg(
            np.arccos(
                self.reciprocal_dot(miller_index_hkl1, miller_index_hkl2)
                / np.sqrt(
                    self.reciprocal_dot(miller_index_hkl1, miller_index_hkl1)
                    * self.reciprocal_dot(miller_index_hkl2, miller_index_hkl2)
                )
            )
        )

    def reciprocal_direct_vectors_angle(
        self, miller_index_hkl: Tuple[int, int, int], position_index_pqr: Tuple[int, int, int]
    ):
        """
        Get angle between two reciprocal/direct vectors
        """

        return np.rad2deg(
            np.arccos(
                self.reciprocal_direct_dot(miller_index_hkl, position_index_pqr)
                / np.sqrt(
                    self.reciprocal_dot(miller_index_hkl, miller_index_hkl)
                    * self.direct_dot(position_index_pqr, position_index_pqr)
                )
            )
        )

    def plane_family_spacing(self, miller_index_hkl: Tuple[int, int, int]):
        """
        Spacing between plane of familly miller_index_hkl
        """

        return 2 * np.pi / self.reciprocal_length(miller_index_hkl)

    def reciprocal_dot(self, miller_index_hkl1: Tuple[int, int, int], miller_index_hkl2: Tuple[int, int, int]):
        miller_index_hkl1 = np.array(miller_index_hkl1)
        miller_index_hkl2 = np.array(miller_index_hkl2)

        prod = np.sum(miller_index_hkl1[:, None] * miller_index_hkl2[None, :] * self.reciprocal_dot_matrix)

        return prod

    def direct_dot(self, position_index_pqr1: Tuple[int, int, int], position_index_pqr2: Tuple[int, int, int]):
        position_index_pqr1 = np.array(position_index_pqr1)
        position_index_pqr2 = np.array(position_index_pqr2)

        prod = np.sum(position_index_pqr1[:, None] * position_index_pqr2[None, :] * self.direct_dot_matrix)

        return prod

    def reciprocal_direct_dot(self, miller_index_hkl: Tuple[int, int, int], position_index_pqr: Tuple[int, int, int]):
        miller_index_hkl = np.array(miller_index_hkl)
        position_index_pqr = np.array(position_index_pqr)

        prod = 2 * np.pi * np.dot(miller_index_hkl, position_index_pqr)

        return prod

    def reciprocal_cross(self, miller_index_hkl1: Tuple[int, int, int], miller_index_hkl2: Tuple[int, int, int]):
        miller_index_hkl1 = np.array(miller_index_hkl1)
        miller_index_hkl2 = np.array(miller_index_hkl2)

        hh = miller_index_hkl1[:, None] * miller_index_hkl2[None, :]

        position_index_pqr = (2 * np.pi) ** 2 / self.direct_unit_cell_volume() * np.einsum("ijk,ij", LEVI_CIVITA, hh)

        return position_index_pqr

    def direct_cross(self, position_index_pqr1: Tuple[int, int, int], position_index_pqr2: Tuple[int, int, int]):
        position_index_pqr1 = np.array(position_index_pqr1)
        position_index_pqr2 = np.array(position_index_pqr2)

        pp = position_index_pqr1[:, None] * position_index_pqr2[None, :]

        miller_index_hkl = self.direct_unit_cell_volume() / (2 * np.pi) * np.einsum("ijk,ij", LEVI_CIVITA, pp)

        return miller_index_hkl

    def projection_reciprocal_on_reciprocal(
        self, miller_index_hkl1: Tuple[int, int, int], miller_index_hkl2: Tuple[int, int, int]
    ):
        miller_index_hkl1 = np.array(miller_index_hkl1)
        miller_index_hkl2 = np.array(miller_index_hkl2)

        return (
            self.reciprocal_dot(miller_index_hkl1, miller_index_hkl2)
            / self.reciprocal_dot(miller_index_hkl2, miller_index_hkl2)
            * miller_index_hkl2
        )

    def projection_direct_on_direct(
        self, position_index_pqr1: Tuple[int, int, int], position_index_pqr2: Tuple[int, int, int]
    ):
        position_index_pqr1 = np.array(position_index_pqr1)
        position_index_pqr2 = np.array(position_index_pqr2)

        return (
            self.direct_dot(position_index_pqr1, position_index_pqr2)
            / self.direct_dot(position_index_pqr2, position_index_pqr2)
            * position_index_pqr2
        )

    def projection_reciprocal_on_direct(
        self, miller_index_hkl: Tuple[int, int, int], position_index_pqr: Tuple[int, int, int]
    ):
        miller_index_hkl = np.array(miller_index_hkl)
        position_index_pqr = np.array(position_index_pqr)

        return (
            self.reciprocal_direct_dot(miller_index_hkl, position_index_pqr)
            / self.direct_dot(position_index_pqr, position_index_pqr)
            * position_index_pqr
        )

    def reciprocal_direct_on_reciprocal(
        self, position_index_pqr: Tuple[int, int, int], miller_index_hkl: Tuple[int, int, int]
    ):
        position_index_pqr = np.array(position_index_pqr)
        miller_index_hkl = np.array(miller_index_hkl)

        return (
            self.reciprocal_direct_dot(miller_index_hkl, position_index_pqr)
            / self.reciprocal_dot(miller_index_hkl, miller_index_hkl)
            * miller_index_hkl
        )

    def convert_direct_to_reciprocal(self, position_index_pqr: Tuple[int, int, int]):
        position_index_pqr = np.array(position_index_pqr)

        return 1 / (2 * np.pi) * np.dot(self.direct_dot_matrix, position_index_pqr)

    def convert_reciprocal_to_direct(self, miller_index_hkl: Tuple[int, int, int]):
        miller_index_hkl = np.array(miller_index_hkl)

        return 1 / (2 * np.pi) * np.dot(self.reciprocal_dot_matrix, miller_index_hkl)

    def reciprocal_angle_on_plane_hkl(
        self,
        miller_index_hkl1: Tuple[int, int, int],
        miller_index_hkl2: Tuple[int, int, int],
        hkl_plane: Tuple[int, int, int],
    ):
        """
        Get angle between two reciprocal vectors projected on a plane perpendicular to vector hkl_plane
        """

        hkl_plane = np.array(hkl_plane)
        miller_index_hkl1 = np.array(miller_index_hkl1)
        miller_index_hkl2 = np.array(miller_index_hkl2)

        hkl1_onplane = miller_index_hkl1 - self.projection_reciprocal_on_reciprocal(miller_index_hkl1, hkl_plane)
        hkl2_onplane = miller_index_hkl2 - self.projection_reciprocal_on_reciprocal(miller_index_hkl2, hkl_plane)

        angle = self.reciprocal_angle(hkl1_onplane, hkl2_onplane)
        direction = np.sign(self.reciprocal_dot(self.reciprocal_cross(miller_index_hkl1, miller_index_hkl2), hkl_plane))

        return angle * direction

    def g_r_angle_on_plane_hkl(
        self,
        miller_index_hkl: Tuple[int, int, int],
        position_index_pqr: Tuple[int, int, int],
        hkl_plane: Tuple[int, int, int],
    ):
        """
        Get angle between two vectors (reciprocal and direct) projected on a plane perpendicular to vector hkl_plane
        """

        hkl_plane = np.array(hkl_plane)
        miller_index_hkl = np.array(miller_index_hkl)
        position_index_pqr = np.array(position_index_pqr)

        hkl_onplane = miller_index_hkl - self.projection_reciprocal_on_reciprocal(miller_index_hkl, hkl_plane)
        pqr_onplane = position_index_pqr - self.convert_reciprocal_to_direct(
            self.reciprocal_direct_on_reciprocal(position_index_pqr, hkl_plane)
        )

        angle = self.reciprocal_direct_vectors_angle(hkl_onplane, pqr_onplane)
        direction = np.sign(
            self.reciprocal_dot(
                self.reciprocal_cross(miller_index_hkl, self.convert_direct_to_reciprocal(position_index_pqr)),
                hkl_plane,
            )
        )

        return angle * direction


class Rhombohedral(GeneralStructure):
    def __init__(self, a_length, angle):
        super().__init__((a_length, a_length, a_length), (angle, angle, angle))


class Hexagonal(GeneralStructure):
    def __init__(self, a12_length, a3_length):
        super().__init__((a12_length, a12_length, a3_length), (90, 90, 120))


class Monoclinic(GeneralStructure):
    def __init__(self, a1_length, a2_length, a3_length, angle2):
        super().__init__((a1_length, a2_length, a3_length), (90, angle2, 90))


class Tetragonal(GeneralStructure):
    def __init__(self, a12_length, a3_length):
        super().__init__((a12_length, a12_length, a3_length), (90, 90, 90))


class Orthorhombic(GeneralStructure):
    def __init__(self, a1_length, a2_length, a3_length):
        super().__init__((a1_length, a2_length, a3_length), (90, 90, 90))


class Cubic(GeneralStructure):
    def __init__(self, a_length):
        super().__init__((a_length, a_length, a_length), (90, 90, 90))


class CubicFaceCentered(Cubic):
    def add_atom(self, coordinates, form_factor, label=""):
        for v1 in [np.array((0, 0, 0)), np.array((0.5, 0.5, 0)), np.array((0, 0.5, 0.5)), np.array((0.5, 0, 0.5))]:
            self._add_atom(coordinates + v1, form_factor, label)


class CubicBodyCentered(Cubic):
    def add_atom(self, coordinates, form_factor, label=""):
        for v1 in [np.array((0, 0, 0)), np.array((0.5, 0.5, 0.5))]:
            self._add_atom(coordinates + v1, form_factor, label)


class Diamond(Cubic):
    def add_atom(self, coordinates, form_factor, label=""):
        for v1 in [np.array((0, 0, 0)), np.array((0.5, 0.5, 0)), np.array((0, 0.5, 0.5)), np.array((0.5, 0, 0.5))]:
            for v2 in [np.array((0, 0, 0)), np.array((0.25, 0.25, 0.25))]:
                self._add_atom(coordinates + v1 + v2, form_factor, label)


class Atom(object):
    def __init__(self, coordinates, label="", form_factor="ITC"):
        self.coordinates = np.mod(np.array(coordinates), 1)
        self.form_factor = form_factor
        if form_factor == "ITC":
            self.formFactorFct = ITCFct(label)
        self.label = label

    def __str__(self):
        return "{0:s} : ({1:.3f},{2:.3f},{3:.3f}) : f = {4:.4f}".format(self.label, *self.coordinates, self.form_factor)

    def get_form_factor(self, q):
        if self.form_factor == "ITC":
            return self.formFactorFct(q)
        else:
            return self.form_factor


# def GA_GB_GC_to_cristalStructure(GA, GB, GC):
#     VG = np.abs(np.dot(GA, np.cross(GB, GC)))

#     Ra = 2 * np.pi / VG * np.cross(GB, GC)
#     Rb = 2 * np.pi / VG * np.cross(GC, GA)
#     Rc = 2 * np.pi / VG * np.cross(GA, GB)

#     a_length, b_length, a3_length = np.sqrt(np.sum(Ra**2)), np.sqrt(np.sum(Rb**2)), np.sqrt(np.sum(Rc**2))
#     alpha = np.rad2deg(np.arccos(np.dot(Rb, Rc) / (b_length * a3_length)))
#     beta = np.rad2deg(np.arccos(np.dot(Ra, Rc) / (a_length * a3_length)))
#     gamma = np.rad2deg(np.arccos(np.dot(Ra, Rb) / (a_length * b_length)))

#     return GeneralStructure([a_length, b_length, a3_length], [alpha, beta, gamma])
