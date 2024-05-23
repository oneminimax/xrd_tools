# from XRDTools.Spectrum import *
# from XRDTools.Geometry import TwoTheta_2_GLength
# from XRDTools.Diffractometer import Diffractometer
# from AsciiDataFile.Readers import GenericDataReader as Reader

# import numpy as np

# from os import path

# import matplotlib
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt

# dm = Diffractometer()
# waveLength = dm.waveLength

# dataPath = '../DataTests'
# reader = Reader(' ',['angle','signal'])

# def test_estimateNoiseLevel():

#     DC = reader.read(path.join(dataPath,'16112018a_t2t_03.dat'))

#     angle = DC.getFieldByName('angle')
#     K = TwoTheta_2_GLength(waveLength,angle)
#     amplitude = DC.getFieldByName('signal')

#     noiseLevel = estimateNoiseLevel(K,amplitude)

#     print(noiseLevel)


# test_estimateNoiseLevel()
