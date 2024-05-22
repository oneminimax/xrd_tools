from os import path

import matplotlib
import numpy as np

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from AsciiDataFile.DataContainer import ureg
from AsciiDataFile.Readers import GenericDataReader as Reader
from XRDTools.Diffractometer import Diffractometer
from XRDTools.Geometry import two_theta_2_wave_vector_length
from XRDTools.Spectrum import AGFct, calculate_A, centroid, estimate_noise_level, fitAGFct

data_path = "Data/"

sample_tuples = [
    ("T2T_22_24_STO.dat", (22, 24), (22.6, 22.96)),
    ("T2T_22_23p5_LSAT.dat", (22, 24), (22.93, 23.1)),
    # ('thickness_PCCO.dat',(28,30.5))
]

dm = Diffractometer()

fig = plt.figure()
AX1 = fig.add_subplot(1, 2, 1)
AX2 = fig.add_subplot(2, 2, 2)
AX3 = fig.add_subplot(2, 2, 4)

AX1.set_yscale("log")
AX2.set_yscale("log")

for sample_tuple in sample_tuples:
    file_name = sample_tuple[0]
    two_theta_lim = sample_tuple[1]
    two_theta_sub = sample_tuple[2]
    reader = Reader(" ", ["two theta", "signal"], ["deg", "count"])
    data_curve = reader.read(path.join(data_path, file_name))

    two_theta = data_curve.get_column("two theta").magnitude

    # print(two_theta)

    sub_ind = (
        (two_theta > two_theta_lim[0])
        * (two_theta < two_theta_lim[1])
        * np.logical_not((two_theta > two_theta_sub[0]) * (two_theta < two_theta_sub[1]))
    )
    data_curve.filter(sub_ind)

    two_theta = data_curve.get_column("two theta").magnitude
    signal = data_curve.get_column("signal").magnitude

    KZ = two_theta_2_wave_vector_length(dm.wave_length, two_theta)
    noise = estimate_noise_level(KZ, signal, noisePolyDeg=1)
    error = np.ones(signal.shape)

    G = centroid(KZ, signal - noise)

    X, complexA, ampA_error, argA_error = calculate_A(KZ, signal, error, maxX=300, stepX=1)
    ampA = np.abs(complexA)

    # thick_index = np.argmax(ampA < 0.02)
    # start_thick = X[thick_index]
    # fitAGFct(X,ampA,ampA_error,G, p0 = [start_thick,0,1,0])

    #     K = two_theta_2_wave_vector_length(dm.wave_length*ureg.nanometer,two_theta)
    #     G = centroid(K,signal)

    #     X, complexA, ampA_error, argA_error = calculateA(K-G,signal,signal_error,stepX = 2,maxX = 120)
    #     A = np.abs(complexA)

    #     # p0 = [100,1e-3,A[0],0]
    #     # popt, perr = fitAGFct(X,A,ampA_error,G,p0 = p0)

    AX1.plot(two_theta, signal, ".")
    AX2.plot(X, ampA)
    AX3.plot(X, ampA)
#     # AX2.plot(X,AGFct(X,G,*popt),'--k')
#     # AX3.plot(X,np.angle(complexA))

#     # print(popt,perr)

plt.show()

# def xrd_analysis(dc_pcco,g_pcco,windowPCCO,dc_po,GPO,windowPO):

#     X,ampA,ampA_error = FT_peak(dc_pcco)

#     minX = 2*np.pi/windowPCCO
#     L, std_eps, scale, noise = extract_thickness_from_FT(dc_pcco,g_pcco,X,minX,ampA,ampA_error)
#     # L, std_eps, scale, noise = (0,0),(0,0),(0,0),(0,0)

#     return X,ampA,ampA_error,L,std_eps, scale, noise
