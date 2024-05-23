# %%

from xrd_tools.cristal_structures import Cubic
from xrd_tools.diffractometer import Diffractometer

# Build unit cell. Using atomic number as atomic factor
sto_cs = Cubic(0.3905)
sto_cs.add_atom((0, 0, 0), 38, label="Sr")
sto_cs.add_atom((0.5, 0.5, 0.5), 22, label="Ti")
sto_cs.add_atom((0.5, 0.5, 0), 8, label="O1")
sto_cs.add_atom((0.5, 0, 0.5), 8, label="O2")
sto_cs.add_atom((0, 0.5, 0.5), 8, label="O3")

# Orientation of normal and in plane of the cristal
hkl_z = (0, 0, 1)
pqr_x = (1, 0, 0)

# Put the sample in the diffractometer with the right orientation
dm = Diffractometer(cristal_structure=sto_cs, surface_normal_hkl=hkl_z, azimuth_pqr=pqr_x)

# Show some peaks values
peaks_hkl = [(0, 0, 2), (1, 0, 3), (0, 1, 3)]
for peak_hkl in peaks_hkl:
    print(
        "hkl : {0:d}{1:d}{2:d}, 2T : {3:.4f}, Source : {4:.4f}, Detector : {5:.4f}, Phi : {6:.4f}, Factor : {7:.3f}".format(
            *peak_hkl,
            dm.hkl_2_two_theta(peak_hkl),
            *dm.hkl_2_theta_source_theta_detector_phi(peak_hkl),
            sto_cs.structure_factor(peak_hkl)
        )
    )

# %%
