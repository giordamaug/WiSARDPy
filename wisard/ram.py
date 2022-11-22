#
# WiSARD in python: 
# Classification and Regression
# by Maurizio Giordano (2022)
#

from scipy.sparse.csr import csr_matrix
import numpy as np

class Ram:
    def __init__(self):
        self.wentry = {}
    def __str__(self):
        return f"{self.wentry}"
    def getEntry(self, key):
        return self.wentry[key] if key in self.wentry.keys() else (0,0) 
    def updEntry(self, key, value):
        self.wentry[key] = (self.wentry[key][0] + 1, self.wentry[key][1] + value) if key in self.wentry.keys() else (1,value)

class WRam:                 # 3 times faster then SRam (with sparse matrix)
    def __init__(self):
        self.wentry = {}
    def __str__(self):
        return f"{self.wentry}"
    def getEntry(self, key):
        return self.wentry[key] if key in self.wentry.keys() else 0.0 
    def updEntry(self, key):
        self.wentry[key] = self.wentry[key] + 1.0 if key in self.wentry.keys() else 1.0


class SRam:
    def __init__(self, nlocs):
        self.wentry = csr_matrix((1,nlocs), dtype=float)    # sparse array
    def __str__(self):
        return f"{self.wentry.nonzero()}"
    def getEntry(self, key):
        return self.wentry[0,key] 
    def updEntry(self, key):
        self.wentry[0,key] += 1.0


class VRam:
    def __init__(self, nlocs):
        self.wentry = np.zeros(nlocs, dtype=float)    # sparse array
    def __str__(self):
        return f"{self.wentry.nonzero()}"
    def getEntry(self, key):
        return self.wentry[key] 
    def updEntry(self, key):
        self.wentry[key] += 1.0