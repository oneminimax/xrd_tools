import numpy as np

def lambda_fct(X,L):

    Y = L-X
    Y[Y<0] = 1e-9

    return Y

def ag_fct(x,wave_vector_g,thickness,sigma_eps,scale,noise):

    with np.errstate(divide='ignore', invalid='ignore'):
        A = np.exp(np.log(scale*lambda_fct(x,thickness)/thickness) - ((x*wave_vector_g*sigma_eps)**2)/2 ) + noise

    return A

# def AGLambdaFct_abs(X,G,L,sigma_eps,scale,noise,ave_eps_interface,sigma_eps_interface,la):

#     with np.errstate(divide='ignore', invalid='ignore'):
#         sigma_eps_X2_Z = 2 * la**2 * sigma_eps_interface**2 * np.exp(-L/la) * np.sinh(lambda_fct(X,L)/la)/(lambda_fct(X,L)/la)
#         A_abs = np.exp(
#                 np.log(scale*lambda_fct(X,L)/L)
#                 - (G**2 * (sigma_eps_X2_Z + sigma_eps**2 * X**2))/2 
#             ) + noise

#     return A_abs

# def AGLambdaFct_angle(X,G,L,ave_eps_homo,ave_eps_interface,la):

#     with np.errstate(divide='ignore', invalid='ignore'):
#         ave_eps_Z = ave_eps_interface * np.exp(-L/(2*la)) * np.sinh(X/(2*la))/(X/(2*la)) * np.sinh(lambda_fct(X,L)/(2*la))/(lambda_fct(X,L)/(2*la))
#         A_angle = G * X * (ave_eps_homo + ave_eps_Z)

#     A_angle[X == 0] = 0

#     return A_angle