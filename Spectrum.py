import numpy as np
from XRDTools.Math import fourierTransform as FT
# import scipy.odr as odr
from scipy.optimize import leastsq

def centroid(K,amplitude):

    return np.dot(K,amplitude)/np.sum(amplitude)

def width(K,amplitude):

    return np.sqrt(np.dot(K**2,amplitude)/np.sum(amplitude))

def LambdaFct(X,L):

    Y = L-X
    Y[Y<0] = 0

    return Y

def AGFct(X,G,L,stdEps,scale,noise):

    with np.errstate(divide='ignore', invalid='ignore'):
        A = np.exp(np.log(scale*LambdaFct(X,L)/L) - ((X*G*stdEps)**2)/2 )

    A[A<noise] = noise

    return A

def fitAGFct(X,ampA,ampA_Error,G,p0 = None):

    resFct = lambda p : np.abs(ampA - AGFct(X,G,*p))/ampA_Error

    lsq = leastsq(resFct,p0,full_output=1)

    popt = lsq[0]
    pcov = lsq[1]

    sdcv = np.sqrt(np.diag(pcov))
    chi2 = np.sum(resFct(popt)**2)
    chi2r = chi2/(ampA.size - len(popt))
    perr = sdcv * np.sqrt(chi2r)

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
    stdEps = (popt[1], perr[1])
    scale = (popt[2], perr[2])
    noise = (popt[3], perr[3])

    if allOutput:
        return L, stdEps, scale, noise, X, G
    else:
        return L















