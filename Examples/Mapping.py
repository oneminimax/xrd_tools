from os import path
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from AsciiDataFile.Readers import GenericDataReader as Reader
from XRDTools.Diffractometer import Diffractometer
from XRDTools.Geometry import alpha_beta_2_wave_vector_x_z

dm = Diffractometer()

# data_path = 'Data/'
data_path = '/Users/maximedion/Documents/Projets Physique/2019/Services/Alex Brice/Data/Ge'
file_tuple = ('Map_115_source_low_smaller_region_cs_0p3.dat',-15.7932,'115')
# file_tuple = ('Map_115_source_low.dat',-15.7932,'115')
# file_tuple = ('Map_113_source_low.dat',-25.24,'113')
file_name, omega_offset, peak_order = file_tuple

reader = Reader(' ',['rel omega offset','two theta','signal'])
CS = reader.read(path.join(data_path,file_name))

alpha = CS.get_column_by_name('rel omega offset') + omega_offset
two_theta = CS.get_column_by_name('two theta')
signal = CS.get_column_by_name('signal')

n_alpha = len(np.unique(alpha))
n_two_theta = len(np.unique(two_theta))

new_shape = (n_alpha,n_two_theta)
alpha.shape = new_shape
two_theta.shape = new_shape
signal.shape = new_shape

qx,qz = alpha_beta_2_wave_vector_x_z(dm.wave_length,alpha,two_theta/2)

fig, axs = plt.subplots(ncols=1, nrows=1,figsize = (6,6))
axs.set_ylabel(r'$q_z$ (rad/nm)')
axs.set_xlabel(r'$q_x$ (rad/nm)')
axs.set_title(peak_order)

axs.contourf(qx,qz,np.log(signal),levels = 30)
plt.show()
