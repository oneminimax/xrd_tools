from os import path

import matplotlib
import numpy as np

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from AsciiDataFile.Readers import GenericDataReader as Reader
from lmfit import Model
from XRDTools.Diffractometer import Diffractometer
from XRDTools.Functions import ag_fct
from XRDTools.Geometry import two_theta_2_wave_vector_length
from XRDTools.Spectrum import calculate_ag, centroid, estimate_noise_level

data_path = "Data/"

sample_tuples = [
    # ('T2T_22_24_STO.dat',(22,24)),
    # ('T2T_22_23p5_LSAT.dat',(22,23.5)),
    ("thickness_PCCO.dat", (28, 30.5))
]

dm = Diffractometer()
wave_length = dm.wave_length

fig = plt.figure()
AX1 = fig.add_subplot(1, 2, 1)
AX2 = fig.add_subplot(2, 2, 2)
AX3 = fig.add_subplot(2, 2, 4)

AX1.set_yscale("log")
AX3.set_yscale("log")

# ag_fct(x,wave_vector_g,thickness,sigma_eps,scale,noise)
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
    reader = Reader(" ", ["two theta", "signal"])
    CS = reader.read(path.join(data_path, file_name))

    CS.add_column("noise", "count", estimate_noise_level(CS.get_column("two theta"), CS.get_column("signal")))

    subCS = CS.select_range("two theta", two_theta_lim, new_curve=True)

    two_theta = subCS.get_column("two theta")
    signal = subCS.get_column("signal")
    noise = subCS.get_column("noise")
    signal_error = 1 * np.ones(signal.shape)

    wave_vector = two_theta_2_wave_vector_length(wave_length, two_theta)
    wave_vector_g = centroid(wave_vector, signal)
    params["wave_vector_g"].value = wave_vector_g

    x, ag, abs_ag_error, arg_ag_error = calculate_ag(
        wave_vector - wave_vector_g, signal - noise, signal_error, x_max=120, x_step=2
    )
    abs_ag = np.abs(ag)

    result = ag_model.fit(abs_ag, params, x=x, method="least_squares")
    dely = result.eval_uncertainty(sigma=3)

    print(result.fit_report())

    AX1.plot(two_theta, signal)
    AX1.plot(two_theta, noise, "--k")

    AX2.plot(x, abs_ag)
    AX2.fill_between(x, result.best_fit - dely, result.best_fit + dely, color="#ABABAB")
    AX2.plot(x, result.best_fit)

    AX3.plot(x, abs_ag)
    AX3.fill_between(x, result.best_fit - dely, result.best_fit + dely, color="#ABABAB")
    AX3.plot(x, result.best_fit)

plt.show()
