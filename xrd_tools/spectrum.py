import numpy as np

from xrd_tools.math import fourier_transform

# import scipy.odr as odr
# from scipy.optimize import leastsq


def centroid(wave_vector, amplitude, expo=1):
    return np.dot(wave_vector, amplitude**expo) / np.sum(amplitude**expo)


def width(wave_vector, amplitude):
    return np.sqrt(np.dot(wave_vector**2, amplitude) / np.sum(amplitude))


def signal_area(wave_vector, amplitude):
    delta_wave_vector = np.gradient(wave_vector)

    return np.sum(amplitude * delta_wave_vector)


def estimate_noise_level(wave_vector, amplitude, noise_poly_order=0):
    Y = np.log(amplitude + 1)

    histo, bin_edges = np.histogram(Y, 100)

    max_histo_index = histo.argmax()
    noise_bin_bool = histo > histo[max_histo_index] * 0.95

    noise_index = np.zeros(wave_vector.shape, dtype=bool)

    for ibin in range(len(histo)):
        if ibin < max_histo_index or noise_bin_bool[ibin]:
            bin_index = (bin_edges[ibin] <= Y) * (Y < bin_edges[ibin + 1])
            noise_index[bin_index] = True

    noise_poly = np.polyfit(wave_vector[noise_index], amplitude[noise_index], noise_poly_order)
    noise_level = np.polyval(noise_poly, wave_vector)

    return noise_level


def calculate_ag(wave_vector, amplitude, amplitude_error, x_max=130, x_step=2):
    x = np.arange(0, x_max, x_step)
    ag, abs_ag_error, arg_ag_error = fourier_transform(
        wave_vector, amplitude, x, window_fct=np.hanning, direct_y_error=amplitude_error, error_output=True
    )

    ag = ag / ag[0]
    abs_ag_error = abs_ag_error / np.abs(ag[0])

    return x, ag, abs_ag_error, arg_ag_error


"""
"""

# def fitAGFct(X,ampA,ampA_Error,G,p0 = None,mode = 'lin'):

#     if mode == 'lin':
#         resFct = lambda p : np.abs(ampA - AGFct(X,G,*p))/ampA_Error
#     elif mode == 'log':
#         resFct = lambda p : np.abs(np.log(ampA) - np.log(AGFct(X,G,*p)))/np.log(ampA_Error)

#     lsq = leastsq(resFct,p0,full_output=1)

#     popt = lsq[0]
#     pcov = lsq[1]

#     sdcv = np.sqrt(np.diag(pcov))
#     chi2 = np.sum(resFct(popt)**2)
#     chi2r = chi2/(ampA.size - len(popt))
#     perr = sdcv * np.sqrt(chi2r)

#     return popt, perr

# def fitAGLambdaFct(X,complexA,ampA_Error,argA_Error,XLim,p0 = None):

#     sub_ind = (X < XLim[1]) * (X > XLim[0])

#     resFct = lambda p : np.abs(np.abs(complexA) - AGLambdaFct_abs(X,*p))/ampA_Error + np.abs(np.unwrap(np.angle(complexA)) - AGLambdaFct_angle(X,*p))/argA_Error

#     lsq = leastsq(resFct,p0,full_output=1)

#     popt = lsq[0]
#     pcov = lsq[1]

#     sdcv = np.sqrt(np.diag(pcov))
#     chi2 = np.sum(resFct(popt)**2)
#     chi2r = chi2/(ampA.size - len(popt))
#     perr = sdcv * np.sqrt(chi2r)

#     return popt, perr

# def fitAGLambdaFct_angle(X,A_angle,ampA_Error,argA_Error,G,thickness,XLim,p0 = None):

#     sub_ind = (X < XLim[1]) * (X > XLim[0])


#     resFct = lambda p : np.abs(A_angle[sub_ind] - AGLambdaFct_angle(X[sub_ind],G,thickness,*p))/argA_Error[sub_ind]

#     lsq = leastsq(resFct,p0,full_output=1)

#     print(lsq[0])

#     popt = lsq[0]
#     pcov = lsq[1]

#     try:
#         sdcv = np.sqrt(np.diag(pcov))
#         chi2 = np.sum(resFct(popt)**2)
#         chi2r = chi2/(ampA.size - len(popt))
#         perr = sdcv * np.sqrt(chi2r)
#     except:
#         perr = np.zeros(popt.shape)

#     return popt, perr


# def extractThickness(K,amplitude,amplitude_error,maxX = 130,stepX = 2,p0 = None,allOutput = False):

#     X, complexA, ampA_Error, argA_Error = calculate_A(K,amplitude,amplitude_error,maxX = maxX,stepX = stepX)
#     A = np.abs(complexA)
#     G = centroid(K,amplitude)

#     KWindow = K.max() - K.min()
#     minX = 2*np.pi/KWindow

#     subFitInt = X > minX

#     popt, perr = fitAGFct(X[subFitInt],A[subFitInt],ampA_Error[subFitInt],G, p0 = p0)

#     L = (popt[0], perr[0])
#     sigma_eps_homo = (popt[1], perr[1])
#     scale = (popt[2], perr[2])
#     noise = (popt[3], perr[3])

#     if allOutput:
#         return L, sigma_eps_homo, scale, noise, X, G
#     else:
#         return L
