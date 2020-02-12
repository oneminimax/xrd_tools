import numpy as np

def fourier_transform(x_data,y_data,k_transform,window_fct = None,y_error = None, no_error_output = False):

    phase_matrix = _non_linear_phase_matrix(x_data,k_transform)

    if isinstance(y_error,np.ndarray):
        pass
    else:
        y_error = np.zeros(x_data.shape)

    if callable(window_fct):
        window = window_fct(y_data.size)
        y_data = y_data*window
        y_error = y_error*window
        
    tf = 1/(2*np.pi) * np.dot(phase_matrix,y_data)

    if no_error_output:
        return tf

    else:
        # Error calculation
        re_y_tf_error = 1/(2*np.pi) * np.sqrt(np.dot(np.real(phase_matrix)**2,y_error**2))
        im_y_tf_error = 1/(2*np.pi) * np.sqrt(np.dot(np.imag(phase_matrix)**2,y_error**2))

        abs_tf_error = 1/np.abs(tf)    * np.sqrt((np.real(tf)**2 * re_y_tf_error**2 + np.imag(tf)**2 * im_y_tf_error**2))
        arg_tc_error = 1/np.abs(tf)**2 * np.sqrt((np.real(tf)**2 * im_y_tf_error**2 + np.imag(tf)**2 * re_y_tf_error**2))

        return tf, abs_tf_error, arg_tc_error

def _non_linear_phase_matrix(x_data,k_transform):

    x_step = np.gradient(x_data)

    phase = np.outer(k_transform,x_data)
    phase_matrix = np.exp(1j * phase) * x_step[None,:]

    return phase_matrix