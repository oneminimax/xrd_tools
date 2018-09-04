import numpy as np

def fourierTransform(XData,YData,KData,windowFct = None,YError = None, NoError = False):

    M = _nonLinearPhaseMatrix(XData,KData)

    if isinstance(YError,np.ndarray):
        pass
    else:
        YError = np.zeros(XData.shape)

    if callable(windowFct):
        window = windowFct(YData.size)
        YData = YData*window
        YError = YError*window
        
    TF = 1/(2*np.pi) * np.dot(M,YData)

    if NoError:
        return TF

    else:
        # Error calculation
        DReTF = 1/(2*np.pi) * np.sqrt(np.dot(np.real(M)**2,YError**2))
        DImTF = 1/(2*np.pi) * np.sqrt(np.dot(np.imag(M)**2,YError**2))

        DAbsTF = 1/np.abs(TF)    * np.sqrt((np.real(TF)**2 * DReTF**2 + np.imag(TF)**2 * DImTF**2))
        DArgTF = 1/np.abs(TF)**2 * np.sqrt((np.real(TF)**2 * DImTF**2 + np.imag(TF)**2 * DReTF**2))

        return TF, DAbsTF, DArgTF

def _nonLinearPhaseMatrix(XData,KData):

    StepX = np.gradient(XData)

    A = np.outer(KData,XData)
    M = np.exp(1j * A) * StepX[None,:]

    return M