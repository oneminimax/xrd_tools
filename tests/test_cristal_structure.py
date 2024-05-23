import unittest

from xrd_tools import cristal_structures
from xrd_tools.cristal_structures import numbered_cristal_structures


class TestCristalStructures(unittest.TestCase):
    def test_general_structure(self):

        factor = 1
        cs = cristal_structures.GeneralStructure([0.7298 * factor, 0.851 * factor, 0.5653], [90, 90, 100])

    def test_sto(self):

        cs = cristal_structures.Cubic(0.5568)

        cs.add_atom((0.5, 0.5, 0.5), "Sr")
        cs.add_atom((0, 0, 0), "Ti")
        cs.add_atom((0.5, 0, 0), "O")
        cs.add_atom((0, 0.5, 0), "O")
        cs.add_atom((0, 0, 0.5), "O")

    def test_silicium(self):

        cs = cristal_structures.Diamond(0.543095)

        cs.add_atom((0, 0, 0), "Si")

    def test_niobium(self):

        cs = cristal_structures.CubicBodyCentered(0.33004)

        cs.add_atom((0, 0, 0), "Nb")

    def test_niobium_nitride(self):

        cs = cristal_structures.CubicFaceCentered(0.33004)

        cs.add_atom((0, 0, 0), "Nb")
        cs.add_atom((0.5, 0, 0), "N")

    def test_wikcoff(self):

        cs = numbered_cristal_structures.No139(1, 2)

        cs.add_atom_wickoff("a", 1, label="a1")
        cs.add_atom_wickoff("e", 2 - 1j, label="a2", z=0.1)
