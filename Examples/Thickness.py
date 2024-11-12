# %%

from os import path

import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model

from xrd_tools.diffractometer import Diffractometer
from xrd_tools.functions import ag_fct
from xrd_tools.geometry import two_theta_2_wave_vector_length
from xrd_tools.spectrum import calculate_ag, centroid, estimate_noise_level

data_path = "Data/"

sample_tuples = [
    # ('T2T_22_24_STO.dat',(22,24)),
    # ('T2T_22_23p5_LSAT.dat',(22,23.5)),
    ("thickness_PCCO.dat", (28, 30.5))
]

dm = Diffractometer()
wave_length = dm.wave_length

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 4)

ax1.set_yscale("log")
ax3.set_yscale("log")

ax1.set_xlabel("Angle")
ax1.set_ylabel("Intensity")

ax3.set_xlabel("Distance (nm)")
ax2.set_ylabel("AG(x)")

fig.tight_layout()

ag_model = Model(ag_fct)
params = ag_model.make_params()
params["wave_vector_g"].vary = False
params["thickness"].value = 60
params["sigma_eps"].value = 1e-5
params["scale"].value = 1
params["noise"].value = 1e-6

for sample_tuple in sample_tuples:
    file_name = sample_tuple[0]
    two_theta_lim = sample_tuple[1]

    data = np.loadtxt(path.join(data_path, file_name), skiprows=0)

    two_thetas = data[:, 0]
    intensities = data[:, 1]
    noises = estimate_noise_level(two_thetas, intensities)
    signal_error = 1 * np.ones(intensities.shape)

    wave_vector = two_theta_2_wave_vector_length(wave_length, two_thetas)
    wave_vector_g = centroid(wave_vector, intensities)
    params["wave_vector_g"].value = wave_vector_g

    x, ag, abs_ag_error, arg_ag_error = calculate_ag(
        wave_vector - wave_vector_g, intensities - noises, signal_error, x_max=120, x_step=2
    )
    abs_ag = np.abs(ag)

    result = ag_model.fit(abs_ag, params, x=x, method="least_squares")
    dely = result.eval_uncertainty(sigma=3)

    print(result.fit_report())

    ax1.plot(two_thetas, intensities)
    ax1.plot(two_thetas, noises, "--k")

    ax2.plot(x, abs_ag)
    ax2.fill_between(x, result.best_fit - dely, result.best_fit + dely, color="#ABABAB")
    ax2.plot(x, result.best_fit)

    ax3.plot(x, abs_ag)
    ax3.fill_between(x, result.best_fit - dely, result.best_fit + dely, color="#ABABAB")
    ax3.plot(x, result.best_fit)

# %%
