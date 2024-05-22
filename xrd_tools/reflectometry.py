import numpy as np
from numpy import matmul
from numpy.linalg import inv
from scipy.special import erfc

delta_factor = (2.818e-13) * (6.02214076e23) * 1e-14  # atomic_ratio, density g/cm^3, (lambda nm)^2
mass_unit = 1.660539e-27  # kg
re = 2.8179403e-15  # m


class Sample(object):
    def __init__(self, substrate=None, layers=list()):
        if substrate:
            self.set_substrate(substrate)
        self.layers = layers
        self.vacuum = Vacuum()

    def set_substrate(self, substrate):
        self.substrate = substrate

    def add_layer(self, layer):
        self.layers.append(layer)

    def get_total_thickness(self):
        thickness = 0
        for layer in self.layers:
            thickness += layer.thickness

        return thickness

    def get_layers_name(self):
        layers_name = list()

        for layer in reversed(self.layers):
            layers_name.append(layer.name)
        layers_name.append(self.substrate.name)

        return layers_name

    def get_layers_thickness(self):
        layer_thickness = np.zeros((len(self.layers),))
        for i, layer in enumerate(reversed(self.layers)):
            layer_thickness[i] = layer.thickness

        return layer_thickness

    def get_layers_x(self):
        interfaces_x = self.get_interfaces_x()
        layers_x = (interfaces_x[:-1] + interfaces_x[1:]) / 2

        return layers_x

    """ Interfaces gets """

    def get_interfaces_x(self):
        interfaces_x = np.zeros((len(self.layers) + 1,))
        for i, layer in enumerate(reversed(self.layers)):
            interfaces_x[i + 1] = interfaces_x[i] + layer.thickness

        return interfaces_x

    def get_interfaces_refraction_index(self, lamb):
        interfaces_refraction_index = np.zeros((len(self.layers) + 1, 2))
        interfaces_refraction_index[0, 0] = self.vacuum.get_refraction_index(lamb, "bottom")
        for i, layer in enumerate(reversed(self.layers)):
            interfaces_refraction_index[i, 1] = layer.get_refraction_index(lamb, "top")
            interfaces_refraction_index[i + 1, 0] = layer.get_refraction_index(lamb, "bottom")
        interfaces_refraction_index[-1, 1] = self.substrate.get_refraction_index(lamb, "top")

        return interfaces_refraction_index

    def get_interfaces_roughness(self):
        interface_roughness = np.zeros((len(self.layers) + 1,))
        for i, layer in enumerate(reversed(self.layers)):
            interface_roughness[i] = layer.roughness
        interface_roughness[-1] = self.substrate.roughness

        return interface_roughness

    def get_interfaces_kz(self, lamb, thetas):
        # dim 0, interface
        # dim 1 top bottom
        # dim 2 different theta value

        interfaces_refraction_index = self.get_interfaces_refraction_index(lamb)
        return (2 * np.pi / lamb) * np.sqrt(
            interfaces_refraction_index[:, :, None] ** 2 - np.cos(thetas[None, None, :]) ** 2 + 0j
        )

    def get_interfaces_matrix_p(self, interfaces_x, interfaces_kz):
        """
        Transfer matrix from one side of an interface to the other (same position)
        """

        interfaces_roughness = self.get_interfaces_roughness()

        # kzp = interfaces_kz[1:,:] + interfaces_kz[:-1,:]
        # kzm = interfaces_kz[1:,:] - interfaces_kz[:-1,:]

        kzp = np.sum(interfaces_kz, axis=1)
        kzm = np.squeeze(np.diff(interfaces_kz, axis=1))

        pfac = kzp / 2
        mfac = kzm / 2

        # dim 0, 1 2d matrix
        # dim 2 interface position
        # dim 3 value of k for different theta value

        interfaces_matrix = np.zeros([2, 2, interfaces_x.size, interfaces_kz.shape[2]], dtype=complex)
        interfaces_matrix[0, 0, :, :] = (
            pfac
            * np.exp(-1j * kzm * interfaces_x[:, None])
            * np.exp(-1 / 2 * (kzm * interfaces_roughness[:, None]) ** 2)
        )
        interfaces_matrix[0, 1, :, :] = (
            mfac
            * np.exp(-1j * kzp * interfaces_x[:, None])
            * np.exp(-1 / 2 * (kzp * interfaces_roughness[:, None]) ** 2)
        )
        interfaces_matrix[1, 0, :, :] = (
            mfac
            * np.exp(1j * kzp * interfaces_x[:, None])
            * np.exp(-1 / 2 * (kzp * interfaces_roughness[:, None]) ** 2)
        )
        interfaces_matrix[1, 1, :, :] = (
            pfac
            * np.exp(1j * kzm * interfaces_x[:, None])
            * np.exp(-1 / 2 * (kzm * interfaces_roughness[:, None]) ** 2)
        )

        return interfaces_matrix

    def get_interfaces_matrix_s(self, interfaces_x, interfaces_kz, interfaces_n):
        """
        Transfer matrix from one side of an interface to the other (same position)
        """

        interfaces_roughness = self.get_interfaces_roughness()

        # kzp = interfaces_kz[1:,:] + interfaces_kz[:-1,:]
        # kzm = interfaces_kz[1:,:] - interfaces_kz[:-1,:]

        kzp = np.sum(interfaces_kz, axis=1)
        kzm = np.squeeze(np.diff(interfaces_kz, axis=1))

        pfac = (
            interfaces_n[:, 0, None] ** 2 * interfaces_kz[:, 1, :]
            + interfaces_n[:, 1, None] ** 2 * interfaces_kz[:, 0, :]
        ) / (2 * interfaces_n[:, 0, None] * interfaces_n[:, 1, None])
        mfac = (
            interfaces_n[:, 0, None] ** 2 * interfaces_kz[:, 1, :]
            - interfaces_n[:, 1, None] ** 2 * interfaces_kz[:, 0, :]
        ) / (2 * interfaces_n[:, 0, None] * interfaces_n[:, 1, None])

        # dim 0, 1 2d matrix
        # dim 2 interface position
        # dim 3 value of k for different theta value

        interfaces_matrix = np.zeros([2, 2, interfaces_x.size, interfaces_kz.shape[2]], dtype=complex)
        interfaces_matrix[0, 0, :, :] = (
            pfac
            * np.exp(-1j * kzm * interfaces_x[:, None])
            * np.exp(-1 / 2 * (kzm * interfaces_roughness[:, None]) ** 2)
        )
        interfaces_matrix[0, 1, :, :] = (
            mfac
            * np.exp(-1j * kzp * interfaces_x[:, None])
            * np.exp(-1 / 2 * (kzp * interfaces_roughness[:, None]) ** 2)
        )
        interfaces_matrix[1, 0, :, :] = (
            mfac
            * np.exp(1j * kzp * interfaces_x[:, None])
            * np.exp(-1 / 2 * (kzp * interfaces_roughness[:, None]) ** 2)
        )
        interfaces_matrix[1, 1, :, :] = (
            pfac
            * np.exp(1j * kzm * interfaces_x[:, None])
            * np.exp(-1 / 2 * (kzm * interfaces_roughness[:, None]) ** 2)
        )

        return interfaces_matrix

    """ Layers get """

    def get_layers_matrix(self, lamb, thetas, interfaces_x, interfaces_kz):
        n = 10

        layers_matrix = np.zeros([2, 2, len(self.layers), len(thetas)], dtype=complex)
        # layers_matrix[0,0,:,:] = 1
        # layers_matrix[0,1,:,:] = 0
        # layers_matrix[1,0,:,:] = 0
        # layers_matrix[1,1,:,:] = 1

        for i, layer in enumerate(reversed(self.layers)):
            integral1 = 0
            integral2 = 0
            position_array = layer.get_position_array(n) + interfaces_x[i]

            ave_position = (position_array[1:, None] + position_array[:-1, None]) / 2

            dx = np.diff(position_array)
            refraction_index_array = layer.get_refraction_index_array(lamb, n)

            kz_array = (2 * np.pi / lamb) * np.sqrt(
                refraction_index_array[:, None] ** 2 - np.cos(thetas[None, :]) ** 2 + 0j
            )
            # kzm = np.diff(kz_array,axis = 0)
            kzm = kz_array[1:, :] - kz_array[:-1, :]
            kzp = kz_array[1:, :] + kz_array[:-1, :]

            kz_all_prod = np.prod(kzp, axis=0)
            kz_sub_prod = np.zeros((kzp.shape), dtype=complex)
            for j in range(kzp.shape[0]):
                # where = np.ones((kzp.shape),dtype = bool)
                # where[j,:] = False
                # kz_sub_prod[j,:] = np.prod(kzp,axis = 0,where = where)
                kz_sub_prod[j, :] = kz_all_prod / kzp[j]

            interal00 = np.sum(kzm * position_array[1:, None], axis=0)

            integral01 = np.sum(kzm * kz_sub_prod * np.exp(-1j * kzp * position_array[1:, None]), axis=0)
            integral10 = np.sum(kzm * kz_sub_prod * np.exp(1j * kzp * position_array[1:, None]), axis=0)

            layers_matrix[0, 0, i, :] = (1 - 1j * interal00) * kz_all_prod
            layers_matrix[0, 1, i, :] = integral01
            layers_matrix[1, 0, i, :] = integral10
            layers_matrix[1, 1, i, :] = (1 + 1j * interal00) * kz_all_prod

        return layers_matrix

    def get_transfert_matrix(self, interfaces_matrix, layers_matrix):
        """
        The whole transfert from vacuum to substrate
        """

        transfert_matrix = np.zeros((2, 2, interfaces_matrix.shape[3]), dtype=complex)
        for i_theta in range(interfaces_matrix.shape[3]):
            tmp_transfert_matrix = interfaces_matrix[:, :, 0, i_theta]
            # tmp_transfert_matrix = np.eye(2)
            for iX in range(len(self.layers)):
                tmp_transfert_matrix = matmul(layers_matrix[:, :, iX, i_theta], tmp_transfert_matrix)
                tmp_transfert_matrix = matmul(
                    interfaces_matrix[:, :, iX + 1, i_theta], tmp_transfert_matrix
                )  # /layers_kz[iX,i_theta]
            transfert_matrix[:, :, i_theta] = tmp_transfert_matrix

        return transfert_matrix

    def get_reflect_coef(self, lamb, thetas, polarisation="p"):
        interfaces_x = self.get_interfaces_x()
        interfaces_kz = self.get_interfaces_kz(lamb, thetas)
        interfaces_n = self.get_interfaces_refraction_index(lamb)

        layers_matrix = self.get_layers_matrix(lamb, thetas, interfaces_x, interfaces_kz)
        if polarisation == "p":
            interfaces_matrix_p = self.get_interfaces_matrix_p(interfaces_x, interfaces_kz)
            transfert_matrix_p = self.get_transfert_matrix(interfaces_matrix_p, layers_matrix)
            reflect_coef_p = transfert_matrix_p[1, 0, :] / transfert_matrix_p[1, 1, :]
            return reflect_coef_p
        elif polarisation == "s":
            interfaces_matrix_s = self.get_interfaces_matrix_s(interfaces_x, interfaces_kz, interfaces_n)
            transfert_matrix_s = self.get_transfert_matrix(interfaces_matrix_s, layers_matrix)
            reflect_coef_s = transfert_matrix_s[1, 0, :] / transfert_matrix_s[1, 1, :]
            return reflect_coef_s
        elif polarisation == "both":
            interfaces_matrix_p = self.get_interfaces_matrix_p(interfaces_x, interfaces_kz)
            interfaces_matrix_s = self.get_interfaces_matrix_s(interfaces_x, interfaces_kz, interfaces_n)
            transfert_matrix_p = self.get_transfert_matrix(interfaces_matrix_p, layers_matrix)
            transfert_matrix_s = self.get_transfert_matrix(interfaces_matrix_s, layers_matrix)
            reflect_coef_p = transfert_matrix_p[1, 0, :] / transfert_matrix_p[1, 1, :]
            reflect_coef_s = transfert_matrix_s[1, 0, :] / transfert_matrix_s[1, 1, :]
            return reflect_coef_p + reflect_coef_s

    def get_density_profil(self, n_point=100):
        n_roughness = 4

        # top to bottom
        layers_thickness = self.get_layers_thickness()
        interfaces_x = self.get_interfaces_x()  # from top
        interfaces_roughness = self.get_interfaces_roughness()
        total_thickness = self.get_total_thickness()

        X = np.linspace(
            -n_roughness * interfaces_roughness[-1], total_thickness + n_roughness * interfaces_roughness[0], n_point
        )

        density_profil = self.substrate.density * erfc((X) / interfaces_roughness[-1]) / 2
        for i, layer in enumerate(reversed(self.layers)):
            density_profil += (
                layer.density
                * erfc(-(X - (total_thickness - interfaces_x[i + 1])) / interfaces_roughness[i + 1])
                * erfc((X - (total_thickness - interfaces_x[i])) / interfaces_roughness[i])
                / 4
            )

        return X, density_profil

    def get_scaterinreciprocal_length_profil(self, n_point=100):
        n_roughness = 4

        # top to bottom
        layers_thickness = self.get_layers_thickness()
        interfaces_x = self.get_interfaces_x()  # from top
        interfaces_roughness = self.get_interfaces_roughness()
        total_thickness = self.get_total_thickness()

        X = np.linspace(
            -n_roughness * interfaces_roughness[-1], total_thickness + n_roughness * interfaces_roughness[0], n_point
        )

        scaterinreciprocal_length_profil = (
            self.substrate.get_scaterinreciprocal_length("top") * erfc((X) / interfaces_roughness[-1]) / 2
        )
        for i, layer in enumerate(reversed(self.layers)):
            layer_scaterinreciprocal_lengths = layer.get_scaterinreciprocal_length("bottom_top")
            first_interface_X = total_thickness - interfaces_x[i + 1]
            second_interface_X = total_thickness - interfaces_x[i]
            slope = (layer_scaterinreciprocal_lengths[1] - layer_scaterinreciprocal_lengths[0]) / (
                first_interface_X - second_interface_X
            )

            scaterinreciprocal_length_array = (X - first_interface_X) * slope + layer_scaterinreciprocal_lengths[1]
            scaterinreciprocal_length_profil += (
                scaterinreciprocal_length_array
                * erfc(-(X - (first_interface_X)) / interfaces_roughness[i + 1])
                * erfc((X - (second_interface_X)) / interfaces_roughness[i])
                / 4
            )

        return X, scaterinreciprocal_length_profil


class Material(object):
    def __init__(self, atomic_mass, atomic_number, unit_cell_volume):
        self.atomic_mass = atomic_mass
        self.atomic_number = atomic_number
        self.unit_cell_volume = unit_cell_volume


class Layer(object):
    def __init__(self, name, density, thickness, roughness=0, atomic_mass=2, atomic_number=1):
        self.name = name
        self.thickness = thickness  # nm
        self.density = density  # g/cm^3
        self.roughness = roughness  # nm

        self.atomic_mass = atomic_mass
        self.atomic_number = atomic_number

        self.atomic_ratio = self.atomic_number / self.atomic_mass  # nb proton/atomic mass (u)

    def set_thickness(self, thickness):
        self.thickness = thickness

    def set_density(self, density):
        self.density = density

    def get_density(self, where):
        if where in ["top", "bottom"]:
            return self.density
        elif where in ["top_bottom", "bottom_top"]:
            return np.ones((2,)) * self.density

    def get_atomic_density(self):
        return self.density * 6.02214086e24 / 10e24 / self.atomic_mass

    def set_roughness(self, roughness):
        self.roughness = roughness

    def get_refraction_index(self, lamb, where):
        return 1 - delta_factor * self.get_density(where) * self.atomic_ratio * lamb**2 / (2 * np.pi)

    def get_scaterinreciprocal_length(self, where):
        unit_cell_volume = self.atomic_mass * mass_unit / (self.get_density(where) * 1e3)

        return self.atomic_number / unit_cell_volume / 1e30

    def get_position_array(self, n=2):
        return np.linspace(0, self.thickness, n)

    def get_density_array(self, n=2):
        return np.ones((n,)) * self.density

    def get_refraction_index_array(self, lamb, n=2):
        return 1 - delta_factor * self.get_density_array(n) * self.atomic_ratio * lamb**2 / (2 * np.pi)


class LayerSlopeDensity(Layer):
    def __init__(self, name, density_top, density_bottom, thickness, roughness=0, atomic_mass=2, atomic_number=1):
        self.name = name
        self.thickness = thickness  # nm
        self.density_top = density_top  # g/cm^3
        self.density_bottom = density_bottom  # g/cm^3
        self.roughness = roughness  # nm

        self.atomic_mass = atomic_mass
        self.atomic_number = atomic_number

        self.atomic_ratio = self.atomic_number / self.atomic_mass  # nb proton/atomic mass (u)

    def set_density(self, density_top, density_bottom):
        self.density_top = density_top
        self.density_bottom = density_bottom

    def get_density(self, where):
        if where is "top":
            return self.density_top
        elif where is "bottom":
            return self.density_bottom
        elif where in "top_bottom":
            return np.array([self.density_top, self.density_bottom])
        elif where is "bottom_top":
            return np.array([self.density_bottom, self.density_top])

    def get_position_array(self, n=10):
        return np.linspace(0, self.thickness, n)

    def get_density_array(self, n=10):
        top_bottom_array = np.power(np.linspace(0, 1, n), 1)
        density_array = self.density_top + top_bottom_array * (self.density_bottom - self.density_top)

        return density_array


class Substrate(Layer):
    def __init__(self, name, density, roughness=0, atomic_mass=2, atomic_number=1):
        Layer.__init__(self, name, density, 0, roughness, atomic_mass, atomic_number)

    def get_density(self, where):
        if where in ["top"]:
            return self.density
        else:
            print("Substrate bottom is not accessible")


class Vacuum(Layer):
    def __init__(self):
        Layer.__init__(self, "Vacuum", 0, 0, 0, 1)

    def get_density(self, where):
        if where in ["bottom"]:
            return self.density
        else:
            print("Vaccum top is not accessible")
