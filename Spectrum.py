import numpy as np
from XRDTools.Math import fourierTransform as FT
# import scipy.odr as odr
from scipy.optimize import leastsq

def centroid(K,amplitude):

    return np.dot(K,amplitude)/np.sum(amplitude)

def width(K,amplitude):

    return np.sqrt(np.dot(K**2,amplitude)/np.sum(amplitude))

def signalArea(K,amplitude):

    deltaK = np.gradient(K)

    return np.sum(amplitude*deltaK) 

def LambdaFct(X,L):

    Y = L-X
    Y[Y<0] = 1e-9

    return Y

def AGFct(X,G,L,sigma_eps_homo,scale,noise):

    with np.errstate(divide='ignore', invalid='ignore'):
        A = np.exp(np.log(scale*LambdaFct(X,L)/L) - ((X*G*sigma_eps_homo)**2)/2 ) + noise

    # A[A<noise] = noise

    return A

def AGLambdaFct_abs(X,G,L,sigma_eps_homo,scale,noise,ave_eps_interface,sigma_eps_interface,la):

    # sigma_eps_interface = 0

    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_eps_X2_Z = 2 * la**2 * sigma_eps_interface**2 * np.exp(-L/la) * np.sinh(LambdaFct(X,L)/la)/(LambdaFct(X,L)/la)
        A_abs = np.exp(
                np.log(scale*LambdaFct(X,L)/L)
                - (G**2 * (sigma_eps_X2_Z + sigma_eps_homo**2 * X**2))/2 
            ) + noise

    # A[A<noise] = noise

    return A_abs

def AGLambdaFct_angle(X,G,L,ave_eps_homo,ave_eps_interface,la):

    # sigma_eps_interface = 0

    with np.errstate(divide='ignore', invalid='ignore'):
        ave_eps_Z = ave_eps_interface * np.exp(-L/(2*la)) * np.sinh(X/(2*la))/(X/(2*la)) * np.sinh(LambdaFct(X,L)/(2*la))/(LambdaFct(X,L)/(2*la))
        A_angle = G * X * (ave_eps_homo + ave_eps_Z)

    # A[A<noise] = noise

    A_angle[X == 0] = 0

    return A_angle

def fitAGFct(X,ampA,ampA_Error,G,p0 = None,mode = 'lin'):

    if mode == 'lin':
        resFct = lambda p : np.abs(ampA - AGFct(X,G,*p))/ampA_Error
    elif mode == 'log':
        resFct = lambda p : np.abs(np.log(ampA) - np.log(AGFct(X,G,*p)))/np.log(ampA_Error)

    lsq = leastsq(resFct,p0,full_output=1)

    popt = lsq[0]
    pcov = lsq[1]

    sdcv = np.sqrt(np.diag(pcov))
    chi2 = np.sum(resFct(popt)**2)
    chi2r = chi2/(ampA.size - len(popt))
    perr = sdcv * np.sqrt(chi2r)

    return popt, perr

def fitAGLambdaFct(X,complexA,ampA_Error,argA_Error,XLim,p0 = None):

    sub_ind = (X < XLim[1]) * (X > XLim[0]) 

    resFct = lambda p : np.abs(np.abs(complexA) - AGLambdaFct_abs(X,*p))/ampA_Error + np.abs(np.unwrap(np.angle(complexA)) - AGLambdaFct_angle(X,*p))/argA_Error

    lsq = leastsq(resFct,p0,full_output=1)

    print(lsq[0])

    popt = lsq[0]
    pcov = lsq[1]

    sdcv = np.sqrt(np.diag(pcov))
    chi2 = np.sum(resFct(popt)**2)
    chi2r = chi2/(ampA.size - len(popt))
    perr = sdcv * np.sqrt(chi2r)

    return popt, perr

def fitAGLambdaFct_angle(X,A_angle,ampA_Error,argA_Error,G,thickness,XLim,p0 = None):

    sub_ind = (X < XLim[1]) * (X > XLim[0]) 


    resFct = lambda p : np.abs(A_angle[sub_ind] - AGLambdaFct_angle(X[sub_ind],G,thickness,*p))/argA_Error[sub_ind]

    lsq = leastsq(resFct,p0,full_output=1)

    print(lsq[0])

    popt = lsq[0]
    pcov = lsq[1]

    try:
        sdcv = np.sqrt(np.diag(pcov))
        chi2 = np.sum(resFct(popt)**2)
        chi2r = chi2/(ampA.size - len(popt))
        perr = sdcv * np.sqrt(chi2r)
    except:
        perr = np.zeros(popt.shape)

    return popt, perr



def calculateA(K,amplitude,amplitude_error,maxX = 130,stepX = 2):

    X = np.arange(0,maxX,stepX)
    complexA, ampA_Error, argA_Error = FT(K,amplitude,X,windowFct = np.hanning, YError = amplitude_error)

    complexA = complexA/complexA[0]
    ampA_Error = ampA_Error/np.abs(complexA[0])

    return X, complexA, ampA_Error, argA_Error

def extractThickness(K,amplitude,amplitude_error,maxX = 130,stepX = 2,p0 = None,allOutput = False):

    X, complexA, ampA_Error, argA_Error = calculateA(K,amplitude,amplitude_error,maxX = maxX,stepX = stepX)
    A = np.abs(complexA)
    G = centroid(K,amplitude)

    KWindow = K.max() - K.min()
    minX = 2*np.pi/KWindow

    subFitInt = X > minX

    popt, perr = fitAGFct(X[subFitInt],A[subFitInt],ampA_Error[subFitInt],G, p0 = p0)

    L = (popt[0], perr[0])
    sigma_eps_homo = (popt[1], perr[1])
    scale = (popt[2], perr[2])
    noise = (popt[3], perr[3])

    if allOutput:
        return L, sigma_eps_homo, scale, noise, X, G
    else:
        return L

def estimateNoiseLevel(K,amplitude,noisePolyDeg = 0):

    Y = np.log(amplitude)

    histo, bin_edges = np.histogram(Y,100)
    
    # last_noise_bin_index = histo.argmax()
    # noise_index = np.zeros(K.shape,dtype = bool)
    # for ibin in range(last_noise_bin_index+1):
    #     bin_index = (bin_edges[ibin] <= Y) * (Y < bin_edges[ibin+1])
    #     noise_index[bin_index] = True


    maxhistoindex = histo.argmax()
    noise_bin_bool = histo > histo[maxhistoindex]*0.95

    noise_index = np.zeros(K.shape,dtype = bool)

    for ibin in range(len(histo)):
        if ibin < maxhistoindex or noise_bin_bool[ibin]:
            bin_index = (bin_edges[ibin] <= Y) * (Y < bin_edges[ibin+1])
            noise_index[bin_index] = True

    noisePoly = np.polyfit(K[noise_index],amplitude[noise_index],noisePolyDeg)
    noiseLevel = np.polyval(noisePoly,K)

    return noiseLevel

















