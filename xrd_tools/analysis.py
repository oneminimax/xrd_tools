# from lmfit import Model
import numpy as np

from lmfit import Model

# def fit_ag_fct(x,amplitude,amplitude_error,wave_vector_g,ag_fct = None,initial_params = None,mode = 'lin'):

#     if ag_fct is None:
#         from XRDTools.Functions import ag_fct # ag_fct(x,wave_vector_g,thickness,sigma_eps,scale,noise)
    
#     ag_model = Model(ag_fct)
#     print(ag_model.param_names)
#     result = ag_model.fit(amplitude,x=x,wave_vector_g = wave_vector_g,thickness = 100, sigma_eps = 0, scale = 1, noise = 0)

#     return result
#     # ag_params = ag_model.make_params()

#     # if mode == 'lin':
#     #     residual = lambda p, x : np.abs(amplitude - ag_fct(x,wave_vector_g,*p))/amplitude_error
#     # elif mode == 'log':
#     #     residual = lambda p, x : np.abs(np.log(amplitude) - np.log(ag_fct(x,wave_vector_g,*p)))/np.log(amplitude_error)

#     # lsq = leastsq(residual,initial_params,full_output=1)

#     # popt = lsq[0]
#     # pcov = lsq[1]

#     # sdcv = np.sqrt(np.diag(pcov))
#     # chi2 = np.sum(residual(popt)**2)
#     # chi2r = chi2/(amplitude.size - len(popt))
#     # perr = sdcv * np.sqrt(chi2r)

#     # return popt, perr