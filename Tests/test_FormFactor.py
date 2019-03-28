from XRDTools.FormFactor import ITCFct
import numpy as np

NbFct = ITCFct('Nb')
NFct = ITCFct('N')

lamb = 0.15406
q = 2*np.pi/lamb

print(q)

# print(NbFct(0),NbFct(q/10))
# print(NFct(0),NFct(q/10))

