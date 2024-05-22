import numpy as np


def fourier_transform(direct_x, direct_y, reciprocal_x, window_fct=None, direct_y_error=None, error_output=False):
    phase_factor_matrix = _non_linear_phase_matrix(direct_x, reciprocal_x)

    if isinstance(direct_y_error, np.ndarray):
        pass
    else:
        direct_y_error = np.zeros(direct_x.shape)

    if callable(window_fct):
        window = window_fct(direct_y.size)
        direct_y = direct_y * window
        direct_y_error = direct_y_error * window

    reciprocal_y = 1 / (2 * np.pi) * np.dot(phase_factor_matrix, direct_y)

    if not error_output:
        return reciprocal_y

    # Error calculation
    re_y_tf_error = 1 / (2 * np.pi) * np.sqrt(np.dot(np.real(phase_factor_matrix) ** 2, direct_y_error**2))
    im_y_tf_error = 1 / (2 * np.pi) * np.sqrt(np.dot(np.imag(phase_factor_matrix) ** 2, direct_y_error**2))

    abs_tf_error = (
        1
        / np.abs(reciprocal_y)
        * np.sqrt((np.real(reciprocal_y) ** 2 * re_y_tf_error**2 + np.imag(reciprocal_y) ** 2 * im_y_tf_error**2))
    )
    arg_tc_error = (
        1
        / np.abs(reciprocal_y) ** 2
        * np.sqrt((np.real(reciprocal_y) ** 2 * im_y_tf_error**2 + np.imag(reciprocal_y) ** 2 * re_y_tf_error**2))
    )

    return reciprocal_y, abs_tf_error, arg_tc_error


def _non_linear_phase_matrix(direct_x, reciprocal_x):
    direct_x_step = np.gradient(direct_x)

    phase = np.outer(reciprocal_x, direct_x)
    phase_factor_matrix = np.exp(1j * phase) * direct_x_step[None, :]

    return phase_factor_matrix
