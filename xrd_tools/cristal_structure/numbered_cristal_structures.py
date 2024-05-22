from xrd_tools.cristal_structure.general_cristal_structures import Cubic, Hexagonal, Orthorhombic, Tetragonal


class No59(Orthorhombic):  # Pmmn
    base_wyckoff_positions = "(0,0,0)"
    wyckoff_positions = {
        "a": "(1/4,1/4,z) (3/4,3/4,-z)",
        "b": "(1/4,3/4,z) (3/4,1/4,-z)",
        "c": "(0,0,0) (1/2,1/2,0) (0,1/2,0) (1/2,0,0)",
        "d": "(0,0,1/2) (1/2,1/2,1/2) (0,1/2,1/2) (1/2,0,1/2)",
        "e": "(1/4,y,z) (1/4,-y+1/2,z) (3/4,y+1/2,-z) (3/4,-y,-z)",
        "f": "(x,1/4,z) (-x+1/2,1/4,z) (-x,3/4,-z) (x+1/2,3/4,-z)",
        "g": "(x,y,z) (-x+1/2,-y+1/2,z) (-x,y+1/2,-z) (x+1/2,-y,-z) (-x,-y,-z) (x+1/2,y+1/2,-z) (x,-y+1/2,z) (-x+1/2,y,z)",
    }


class No59_2(Orthorhombic):  # Pmmn
    base_wyckoff_positions = "(0,0,0)"
    wyckoff_positions = {
        "a": "(1/4,1/4,z) (3/4,3/4,-z)",
        "b": "(1/4,3/4,z) (3/4,1/4,-z)",
        "c": "(0,0,0) (1/2,1/2,0) (0,1/2,0) (1/2,0,0)",
        "d": "(0,0,1/2) (1/2,1/2,1/2) (0,1/2,1/2) (1/2,0,1/2)",
        "e": "(1/4,y,z) (1/4,-y+1/2,z) (3/4,y+1/2,-z) (3/4,-y,-z)",
        "f": "(x,1/4,z) (-x+1/2,1/4,z) (-x,3/4,-z) (x+1/2,3/4,-z)",
        "g": "(x,y,z) (-x+1/2,-y+1/2,z) (-x,y+1/2,-z) (x+1/2,-y,-z) (-x,-y,-z) (x+1/2,y+1/2,-z) (x,-y+1/2,z) (-x+1/2,y,z)",
    }


class No113(Tetragonal):  # P-42_1 m
    base_wyckoff_positions = "(0,0,0)"
    wyckoff_positions = {
        "a": "(0,0,0) (1/2,1/2,0)",
        "b": "(0,0,1/2) (1/2,1/2,1/2)",
        "c": "(0,1/2,z) (1/2,0,-z)",
        "d": "(0,0,z) (0,0,-z) (1/2,1/2,-z) (1/2,1/2,z)",
        "e": "(x,x+1/2,z) (-x,-x+1/2,z) (x+1/2,-x,-z) (-x+1/2,x,-z)",
        "f": "(x,y,z) (-x,-y,z) (y,-x,-z) (-y,x,-z) (-x+1/2,y+1/2,-z)",
    }


class No136(Tetragonal):  # P4_2/mnm
    base_wyckoff_positions = "(0,0,0)"
    wyckoff_positions = {
        "a": "(0,0,0) (1/2,1/2,1/2)",
        "b": "(0,0,1/2) (1/2,1/2,0)",
        "c": "(0,1/2,0) (0,1/2,1/2) (1/2,0,1/2) (1/2,0,0)",
        "d": "(0,1/2,1/4) (0,1/2,3/4) (1/2,0,1/4) (1/2,0,3/4)",
        "e": "(0,0,z) (1/2,1/2,z+1/2) (1/2,1/2,-z+1/2) (0,0,-z)",
        "f": "(x,x,0) (-x,-x,0) (-x+1/2,x+1/2,1/2) (x+1/2,-x+1/2,1/2)",
        "g": "(x,-x,0) (-x,x,0) (x+1/2,x+1/2,1/2) (-x+1/2,-x+1/2,1/2)",
        "h": "(0,1/2,z) (0,1/2,z+1/2) (1/2,0,-z+1/2) (1/2,0,-z) (0,1/2,-z) (0,1/2,-z+1/2) (1/2,0,z+1/2) (1/2,0,z)",
        "i": "(x,y,0) (-x,-y,0) (-y+1/2,x+1/2,1/2) (y+1/2,-x+1/2,1/2) (-x+1/2,y+1/2,1/2) (x+1/2,-y+1/2,1/2) (y,x,0) (-y,-x,0)",
        "j": "(x,x,z) (-x,-x,z) (-x+1/2,x+1/2,z+1/2) (x+1/2,-x+1/2,z+1/2) (-x+1/2,x+1/2,-z+1/2) (x+1/2,-x+1/2,-z+1/2) (x,x,-z) (-x,-x,-z)",
        "k": "(x,y,z) (-x,-y,z) (-y+1/2,x+1/2,z+1/2) (y+1/2,-x+1/2,z+1/2) (-x+1/2,y+1/2,-z+1/2) (x+1/2,-y+1/2,-z+1/2) (y,x,-z) (-y,-x,-z) (-x,-y,-z) (x,y,-z) (y+1/2,-x+1/2,-z+1/2) (-y+1/2,x+1/2,-z+1/2) (x+1/2,-y+1/2,z+1/2) (-x+1/2,y+1/2,z+1/2) (-y,-x,z) (y,x,z)",
    }


class No139(Tetragonal):  # I 4 / m m m
    base_wyckoff_positions = "(0,0,0) (1/2,1/2,1/2)"
    wyckoff_positions = {
        "a": "(0,0,0)",
        "b": "(0,0,1/2)",
        "c": "(0,1/2,0) (1/2,0,0)",
        "d": "(0,1/2,1/4) (1/2,0,1/4)",
        "e": "(0,0,z) (0,0,-z)",
        "f": "(1/4,1/4,1/4) (3/4,3/4,1/4) (3/4,1/4,1/4) (1/4,3/4,1/4)",
        "g": "(0,1/2,z) (1/2,0,z) (0,1/2,-z) (1/2,0,-z)",
        "h": "(x,x,0) (-x,-x,0) (-x,x,0) (x,-x,0)",
        "i": "(x,0,0) (-x,0,0) (0,x,0) (0,-x,0)",
        "j": "(x,1/2,0) (-x,1/2,0) (1/2,x,0) (1/2,-x,0)",
        "k": "(x,x+1/2,1/4) (-x,-x+1/2,1/4) (-x+1/2,x,1/4) (x+1/2,-x,1/4) (-x,-x+1/2,3/4) (x,x+1/2,3/4) (x+1/2,-x,3/4) (-x+1/2,x,3/4)",
        "l": "(x,y,0) (-x,-y,0) (-y,x,0) (y,-x,0) (-x,y,0) (x,-y,0) (y,x,0) (-y,-x,0)",
        "m": "(x,x,z) (-x,-x,z) (-x,x,z) (x,-x,z) (-x,x,-z) (x,-x,-z) (x,x,-z) (-x,-x,-z)",
        "n": "(0,y,z) (0,-y,z) (-y,0,z) (y,0,z) (0,y,-z) (0,-y,-z) (y,0,-z) (-y,0,-z)",
        "o": "(x,y,z) (-x,-y,z) (-y,x,z) (y,-x,z) (-x,y,-z) (x,-y,-z) (y,x,-z) (-y,-x,-z) (-x,-y,-z) (x,y,-z) (y,-x,-z) (-y,x,-z) (x,-y,z) (-x,y,z) (-y,-x,z) (y,x,z)",
    }


class No167(Hexagonal):  # R-3c
    base_wyckoff_positions = "(0,0,0) (2/3,1/3,1/3) (1/3,2/3,2/3)"
    wyckoff_positions = {
        "a": "(0,0,1/4) (0,0,3/4)",
        "b": "(0,0,0) (0,0,1/2)",
        "c": "(0,0,z) (0,0,-z+1/2) (0,0,-z) (0,0,z+1/2)",
        "d": "(1/2,0,0) (0,1/2,0) (1/2,1/2,0) (0,1/2,1/2) (1/2,0,1/2) (1/2,1/2,1/2)",
        "e": "(x,0,1/4) (0,x,1/4) (-x,-x,1/4) (-x,0,3/4) (0,-x,3/4)",
        "f": "(x,y,z) (-y,x-y,z) (-x+y,-x,z) (y,x,-z+1/2) (x-y,-y,-z+1/2) (-x,-x+y,-z+1/2) (-x,-y,-z) (y,-x+y,-z) (x-y,x,-z) (-y,-x,z+1/2) (-x+y,y,z+1/2) (x,x-y,z+1/2)",
    }


class No206(Cubic):  # I a -3
    base_wyckoff_positions = "(0,0,0) (1/2,1/2,1/2)"
    wyckoff_positions = {
        "a": "(0,0,0) (1/2,0,1/2) (0,1/2,1/2) (1/2,1/2,0)",
        "b": "(1/4,1/4,1/4) (1/4,3/4,3/4) (3/4,3/4,1/4) (3/4,1/4,3/4)",
        "c": "(x,x,x) (-x+1/2,-x,x+1/2) (-x,x+1/2,-x+1/2) (x+1/2,-x+1/2,-x) (-x,-x,-x) (x+1/2,x,-x+1/2) (x,-x+1/2,x+1/2) (-x+1/2,x+1/2,x)",
        "d": "(x,0,1/4) (-x+1/2,0,3/4) (1/4,x,0) (3/4,-x+1/2,0) (0,1/4,x) (0,3/4,-x+1/2) (-x,0,3/4) (x+1/2,0,1/4) (3/4,-x,0) (1/4,x+1/2,0) (0,3/4,-x) (0,1/4,x+1/2)",
        "e": "(x,y,z) (-x+1/2,-y,z+1/2) (-x,y+1/2,-z+1/2) (x+1/2,-y+1/2,-z) (z,x,y) (z+1/2,-x+1/2,-y) (-z+1/2,-x,y+1/2) (-z,x+1/2,-y+1/2) (y,z,x) (-y,z+1/2,-x+1/2) (y+1/2,-z+1/2,-x) (-y+1/2,-z,x+1/2) (-x,-y,-z) (x+1/2,y,-z+1/2) (x,-y+1/2,z+1/2) (-x+1/2,y+1/2,z) (-z,-x,-y) (-z+1/2,x+1/2,y) (z+1/2,x,-y+1/2) (z,-x+1/2,y+1/2) (-y,-z,-x) (y,-z+1/2,x+1/2) (-y+1/2,z+1/2,x) (y+1/2,z,-x+1/2)",
    }


class No221(Cubic):  # P m -3 m
    base_wyckoff_positions = "(0,0,0)"
    wyckoff_positions = {
        "a": "(0,0,0)",
        "b": "(1/2,1/2,1/2)",
        "c": "(0,1/2,1/2) (1/2,0,1/2) (1/2,1/2,0)",
        "d": "(1/2,0,0) (0,1/2,0) (0,0,1/2)",
        "e": "(x,0,0) (-x,0,0) (0,x,0) (0,-x,0) (0,0,x) (0,0,-x)",
        "f": "(x,1/2,1/2) (-x,1/2,1/2) (1/2,x,1/2) (1/2,-x,1/2) (1/2,1/2,x) (1/2,1/2,-x)",
        "g": "(x,x,x) (-x,-x,x) (-x,x,-x) (x,-x,-x) (x,x,-x) (-x,-x,-x) (x,-x,x) (-x,x,x)",
        "h": "(x,1/2,0) (-x,1/2,0) (0,x,1/2) (0,-x,1/2) (1/2,0,x) (1/2,0,-x) (1/2,x,0) (1/2,-x,0) (x,0,1/2) (-x,0,1/2) (0,1/2,-x) (0,1/2,x)",
        "i": "(0,y,y) (0,-y,y) (0,y,-y) (0,-y,-y) (y,0,y) (y,0,-y) (-y,0,y) (-y,0,-y) (y,y,0) (-y,y,0) (y,-y,0) (-y,-y,0)",
        "j": "(1/2,y,y) (1/2,-y,y) (1/2,y,-y) (1/2,-y,-y) (y,1/2,y) (y,1/2,-y) (-y,1/2,y) (-y,1/2,-y) (y,y,1/2) (-y,y,1/2) (y,-y,1/2) (-y,-y,1/2)",
        "k": "(0,y,z) (0,-y,z) (0,y,-z) (0,-y,-z) (z,0,y) (z,0,-y) (-z,0,y) (-z,0,-y) (y,z,0) (-y,z,0) (y,-z,0) (-y,-z,0) (y,0,-z) (-y,0,-z) (y,0,z) (-y,0,z) (0,z,-y) (0,z,y) (0,-z,-y) (0,-z,y) (z,y,0) (z,-y,0) (-z,y,0) (-z,-y,0)",
        "l": "(1/2,y,z) (1/2,-y,z) (1/2,y,-z) (1/2,-y,-z) (z,1/2,y) (z,1/2,-y) (-z,1/2,y) (-z,1/2,-y) (y,z,1/2) (-y,z,1/2) (y,-z,1/2) (-y,-z,1/2) (y,1/2,-z) (-y,1/2,-z) (y,1/2,z) (-y,1/2,z) (1/2,z,-y) (1/2,z,y) (1/2,-z,-y) (1/2,-z,y) (z,y,1/2) (z,-y,1/2) (-z,y,1/2) (-z,-y,1/2)",
        "m": "(x,x,z) (-x,-x,z) (-x,x,-z) (x,-x,-z) (z,x,x) (z,-x,-x) (-z,-x,x) (-z,x,-x) (x,z,x) (-x,z,-x) (x,-z,-x) (-x,-z,x) (x,x,-z) (-x,-x,-z) (x,-x,z) (-x,x,z) (x,z,-x) (-x,z,x) (-x,-z,-x) (x,-z,x) (z,x,-x) (z,-x,x) (-z,x,x) (-z,-x,-x)",
        "n": "(x,y,z) (-x,-y,z) (-x,y,-z) (x,-y,-z) (z,x,y) (z,-x,-y) (-z,-x,y) (-z,x,-y) (y,z,x) (-y,z,-x) (y,-z,-x) (-y,-z,x) (y,x,-z) (-y,-x,-z) (y,-x,z) (-y,x,z) (x,z,-y) (-x,z,y) (-x,-z,-y) (x,-z,y) (z,y,-x) (z,-y,x) (-z,y,x) (-z,-y,-x) (-x,-y,-z) (x,y,-z) (x,-y,z) (-x,y,z) (-z,-x,-y) (-z,x,y) (z,x,-y) (z,-x,y) (-y,-z,-x) (y,-z,x) (-y,z,x) (y,z,-x) (-y,-x,z) (y,x,z) (-y,x,-z) (y,-x,-z) (-x,-z,y) (x,-z,-y) (x,z,y) (-x,z,-y) (-z,-y,x) (-z,y,-x) (z,-y,-x) (z,y,x)",
    }


class No225(Cubic):  # F m -3 m
    base_wyckoff_positions = "(0,0,0) (0,1/2,1/2) (1/2,0,1/2) (1/2,1/2,0)"
    wyckoff_positions = {
        "a": "(0,0,0)",
        "b": "(1/2,1/2,1/2)",
        "c": "(1/4,1/4,1/4) (1/4,1/4,3/4)",
        "d": "(0,1/4,1/4) (0,3/4,1/4) (1/4,0,1/4) (1/4,0,3/4) (1/4,1/4,0) (3/4,1/4,0)",
        "e": "(0,1/4,1/4) (0,3/4,1/4) (1/4,0,1/4) (1/4,0,3/4) (1/4,1/4,0) (3/4,1/4,0)",
        "f": "(x,x,x) (-x,-x,x) (-x,x,-x) (x,-x,-x) (x,x,-x) (-x,-x,-x) (x,-x,x) (-x,x,x)",
        "g": "(x,1/4,1/4) (-x,3/4,1/4) (1/4,x,1/4) (1/4,-x,3/4) (1/4,1/4,x) (3/4,1/4,-x) (1/4,x,3/4) (3/4,-x,3/4) (x,1/4,3/4) (-x,1/4,1/4) (1/4,1/4,-x) (1/4,3/4,x) ",
        "h": "(0,y,y) (0,-y,y) (0,y,-y) (0,-y,-y) (y,0,y) (y,0,-y) (-y,0,y) (-y,0,-y) (y,y,0) (-y,y,0) (y,-y,0) (-y,-y,0)",
        "i": "(1/2,y,y) (1/2,-y,y) (1/2,y,-y) (1/2,-y,-y) (y,1/2,y) (y,1/2,-y) (-y,1/2,y) (-y,1/2,-y) (y,y,1/2) (-y,y,1/2) (y,-y,1/2) (-y,-y,1/2)",
        "j": "(0,y,z) (0,-y,z) (0,y,-z) (0,-y,-z) (z,0,y) (z,0,-y) (-z,0,y) (-z,0,-y) (y,z,0) (-y,z,0) (y,-z,0) (-y,-z,0) (y,0,-z) (-y,0,-z) (y,0,z) (-y,0,z) (0,z,-y) (0,z,y) (0,-z,-y) (0,-z,y) (z,y,0) (z,-y,0) (-z,y,0) (-z,-y,0)",
        "k": "(x,x,z) (-x,-x,z) (-x,x,-z) (x,-x,-z) (z,x,x) (z,-x,-x) (-z,-x,x) (-z,x,-x) (x,z,x) (-x,z,-x) (x,-z,-x) (-x,-z,x) (x,x,-z) (-x,-x,-z) (x,-x,z) (-x,x,z) (x,z,-x) (-x,z,x) (-x,-z,-x) (x,-z,x) (z,x,-x) (z,-x,x) (-z,x,x) (-z,-x,-x)",
        "l": "(x,y,z) (-x,-y,z) (-x,y,-z) (x,-y,-z) (z,x,y) (z,-x,-y) (-z,-x,y) (-z,x,-y) (y,z,x) (-y,z,-x) (y,-z,-x) (-y,-z,x) (y,x,-z) (-y,-x,-z) (y,-x,z) (-y,x,z) (x,z,-y) (-x,z,y) (-x,-z,-y) (x,-z,y) (z,y,-x) (z,-y,x) (-z,y,x) (-z,-y,-x) (-x,-y,-z) (x,y,-z) (x,-y,z) (-x,y,z) (-z,-x,-y) (-z,x,y) (z,x,-y) (z,-x,y) (-y,-z,-x) (y,-z,x) (-y,z,x) (y,z,-x) (-y,-x,z) (y,x,z) (-y,x,-z) (y,-x,-z) (-x,-z,y) (x,-z,-y) (x,z,y) (-x,z,-y) (-z,-y,x) (-z,y,-x) (z,-y,-x) (z,y,x)",
    }


class No229(Cubic):  # Im-3m
    base_wyckoff_positions = "(0,0,0) (1/2,1/2,1/2)"
    wyckoff_positions = {
        "a": "(0,0,0)",
        "b": "(0,1/2,1/2) (1/2,0,1/2) (1/2,1/2,0)",
        "c": "(1/4,1/4,1/4) (3/4,3/4,1/4) (3/4,1/4,3/4) (1/4,3/4,3/4)",
        "d": "(1/4,0,1/2) (3/4,0,1/2) (1/2,1/4,0) (1/2,3/4,0) (0,1/2,1/4) (0,1/2,3/4)",
        "e": "(x,0,0) (-x,0,0) (0,x,0) (0,-x,0) (0,0,x) (0,0,-x)",
        "f": "(x,x,x) (-x,-x,x) (-x,x,-x) (x,-x,-x) (x,x,-x) (-x,-x,-x) (x,-x,x) (-x,x,x)",
        "g": "(x,0,1/2) (-x,0,1/2) (1/2,x,0) (1/2,-x,0) (0,1/2,x) (0,1/2,-x) (0,x,1/2) (0,-x,1/2) (x,1/2,0) (-x,1/2,0) (1/2,0,-x) (1/2,0,x)",
        "h": "(0,y,y) (0,-y,y) (0,y,-y) (0,-y,-y) (y,0,y) (y,0,-y) (-y,0,y) (-y,0,-y) (y,y,0) (-y,y,0) (y,-y,0) (-y,-y,0)",
        "i": "(1/4,y,-y+1/2) (3/4,-y,-y+1/2) (3/4,y,y+1/2) (1/4,-y,y+1/2) (-y+1/2,1/4,y) (-y+1/2,3/4,-y) (y+1/2,3/4,y) (y+1/2,1/4,-y) (y,-y+1/2,1/4) (-y,-y+1/2,3/4) (y,y+1/2,3/4) (-y,y+1/2,1/4) (3/4,-y,y+1/2) (1/4,y,y+1/2) (1/4,-y,-y+1/2) (3/4,y,-y+1/2) (y+1/2,3/4,-y) (y+1/2,1/4,y) (-y+1/2,1/4,-y) (-y+1/2,3/4,y) (-y,y+1/2,3/4) (y,y+1/2,1/4) (-y,-y+1/2,1/4) (y,-y+1/2,3/4)",
        "j": "(0,y,z) (0,-y,z) (0,y,-z) (0,-y,-z) (z,0,y) (z,0,-y) (-z,0,y) (-z,0,-y) (y,z,0) (-y,z,0) (y,-z,0) (-y,-z,0) (y,0,-z) (-y,0,-z) (y,0,z) (-y,0,z) (0,z,-y) (0,z,y) (0,-z,-y) (0,-z,y) (z,y,0) (z,-y,0) (-z,y,0) (-z,-y,0)",
        "k": "(x,x,z) (-x,-x,z) (-x,x,-z) (x,-x,-z) (z,x,x) (z,-x,-x) (-z,-x,x) (-z,x,-x) (x,z,x) (-x,z,-x) (x,-z,-x) (-x,-z,x) (x,x,-z) (-x,-x,-z) (x,-x,z) (-x,x,z) (x,z,-x) (-x,z,x) (-x,-z,-x) (x,-z,x) (z,x,-x) (z,-x,x) (-z,x,x) (-z,-x,-x)",
        "l": "(x,y,z) (-x,-y,z) (-x,y,-z) (x,-y,-z) (z,x,y) (z,-x,-y) (-z,-x,y) (-z,x,-y) (y,z,x) (-y,z,-x) (y,-z,-x) (-y,-z,x) (y,x,-z) (-y,-x,-z) (y,-x,z) (-y,x,z) (x,z,-y) (-x,z,y) (-x,-z,-y) (x,-z,y) (z,y,-x) (z,-y,x) (-z,y,x) (-z,-y,-x) (-x,-y,-z) (x,y,-z) (x,-y,z) (-x,y,z) (-z,-x,-y) (-z,x,y) (z,x,-y) (z,-x,y) (-y,-z,-x) (y,-z,x) (-y,z,x) (y,z,-x) (-y,-x,z) (y,x,z) (-y,x,-z) (y,-x,-z) (-x,-z,y) (x,-z,-y) (x,z,y) (-x,z,-y) (-z,-y,x) (-z,y,-x) (z,-y,-x) (z,y,x)",
    }