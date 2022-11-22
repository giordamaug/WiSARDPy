#
# WiSARD in python: 
# Classification and Regression
# by Maurizio Giordano (2022)
#
import numpy as np
from .utilities import *
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import .ram
mypowers = 2**np.arange(65, dtype = np.uint64)[::]

class WiSARDEstimator():
    """WiSARD Encoder """

    # Binarize input (thermomer encoding) terand generates address tuple for Ram access
    def _mk_tuple(self, X):
        addresses = [0]*self._nrams
        for i in range(self._nrams):
            for j in range(self._nobits):
                x = self._mapping[((i * self._nobits) + j) % self._retina_size]
                index = x // self._notics
                value = int((X[index] - self._offsets[index]) * self._notics / self._ranges[index])
                if x % self._notics < value:
                    addresses[i] += mypowers[self._nobits -1 - j]
        return addresses

class WiSARDRegressor(BaseEstimator, RegressorMixin, WiSARDEstimator):
    """WiSARD Regressor """
    
    #def __init__(self,  nobits, size, map=-1, classes=[0,1], dblvl=0):
    def __init__(self,  n_bits=8, n_tics=256, random_state=0, mapping='random', code='t', scale=True, debug=False):
        if (not isinstance(n_bits, int) or n_bits<1 or n_bits>64):
            raise Exception('number of bits must be an integer between 1 and 64')
        if (not isinstance(n_tics, int) or n_tics<1):
            raise Exception('number of bits must be an integer greater than 1')
        if (not isinstance(debug, bool)):
            raise Exception('debug flag must be a boolean')
        if (not isinstance(mapping, str)) or (not (mapping=='random' or mapping=='linear')):
            raise Exception('mapping must either \"random\" or \"linear\"')
        if (not isinstance(code, str)) or (not (code=='g' or code=='t' or code=='c')):
            raise Exception('code must either \"t\" (termometer) or \"g\" (graycode) or \"c\" (cursor)')
        if (not isinstance(random_state, int)) or random_state<0:
            raise Exception('random state must be an integer greater than 0')
        self._nobits = n_bits
        self._notics = n_tics
        self._code = code
        self._scale = scale
        self._nrams = 0
        self._maptype = mapping
        self._seed = random_state
        if self._seed > -1: np.random.seed(self._seed) 
        self._debug = debug
        self._nloc = mypowers[self._nobits]
        
    def train(self, X, y):
        ''' Learning '''
        addresses = self._mk_tuple(X)
        for i in range(self._nrams):
            self._rams[i].updEntry(addresses[i], y)

    def test(self, X):
        ''' Testing '''
        addresses = self._mk_tuple(X)
        res = [sum(i) for i in zip(*[self._rams[i].getEntry(addresses[i]) for i in range(self._nrams)])]
        return float(res[1])/float(res[0]) if res[0] != 0 else 0.0
    
    def fit(self, X, y):
        self._retina_size = self._notics * len(X[0])   # set retin size (# feature x # of tics)
        self._nrams = int(self._retina_size/self._nobits) if self._retina_size % self._nobits == 0 else int(self._retina_size/self._nobits + 1)
        self._mapping = np.arange(self._retina_size, dtype=int)
        self._rams = [ram.Ram() for _ in range(self._nrams)] 
        if self._maptype=="random":                 # random mapping
            np.random.shuffle(self._mapping)
        self._ranges = X.max(axis=0)-X.min(axis=0)
        self._offsets = X.min(axis=0)
        self._ranges[self._ranges == 0] = 1
        if self._debug: timing_init()
        delta = 0                                   # inizialize error
        for i,sample in enumerate(X):
            if self._debug:  print("Target %d"%y[i], end='')
            self.train(sample, y[i])        
            res = self.test(sample)
            delta += abs(y[i] - res)
            if self._debug: timing_update(i,y[i]==res,title='train ',size=len(X),error=delta/float(i+1))
        if self._debug: print()
        return self

    def predict(self,X):
        if self._debug: timing_init()
        y_pred = np.array([])
        for i,sample in enumerate(X):
            y_pred = np.append(y_pred,[self.test(sample)])
            if self._debug: timing_update(i,True,title='test  ',clr=color.GREEN,size=len(X))
        if self._debug: print()
        return y_pred

    def __repr__(self): 
        return "WiSARDRegressor(n_tics: %d, n_bits:, %d, n_rams: %d, random_state: %d, n_locs: %r, mapping: %r)\n"%(self._notics, self._nobits, self._nrams, self._seed,self._nloc, self._maptype)

    def __str__(self):
        ''' Printing function'''
        return "WiSARDRegressor(n_tics: %d, n_bits:, %d, n_rams: %d)\n"%(self._notics, self._nobits, self._nrams)

    def printRams(self):
        rep = ""
        for j in range(self._nrams):
            ep += f'{self._wiznet[cl][j]}'
        return rep

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {"n_bits": self._nobits, "n_tics": self._notics, "mapping": self._maptype, "debug": self._debug, "code" : self._code, "random_state": self._seed
              }

    def getDataType(self):
        return self._datatype

    def getMapping(self):
        return self._mapping

    def getNoBits(self):
        return self._nobits

    def getNoTics(self):
        return self._notics

    def getNoRams(self):
        return self._nrams

class WiSARDClassifier(BaseEstimator, ClassifierMixin, WiSARDEstimator):
    """WiSARD Regressor """
    
    def __init__(self,  n_bits=8, n_tics=256, random_state=0, mapping='random', code='t', scale=True, debug=False):
        if (not isinstance(n_bits, int) or n_bits<1 or n_bits>64):
            raise Exception('number of bits must be an integer between 1 and 64')
        if (not isinstance(n_tics, int) or n_tics<1):
            raise Exception('number of bits must be an integer greater than 1')
        if (not isinstance(debug, bool)):
            raise Exception('debug flag must be a boolean')
        if (not isinstance(mapping, str)) or (not (mapping=='random' or mapping=='linear')):
            raise Exception('mapping must either \"random\" or \"linear\"')
        if (not isinstance(code, str)) or (not (code=='g' or code=='t' or code=='c')):
            raise Exception('code must either \"t\" (termometer) or \"g\" (graycode) or \"c\" (cursor)')
        if (not isinstance(random_state, int)) or random_state<0:
            raise Exception('random state must be an integer greater than 0')
        self._nobits = n_bits
        self._notics = n_tics
        self._code = code
        self._scale = scale
        self._nrams = 0
        self._nclasses = 0
        self._maptype = mapping
        self._seed = random_state
        if self._seed > -1: np.random.seed(self._seed) 
        self._debug = debug
        self._nloc = mypowers[self._nobits]
        self._wiznet = {}
        
    def train(self, X, y):
        ''' Learning '''
        addresses = self._mk_tuple(X)
        for i in range(self._nrams):
            self._wiznet[y][i].updEntry(addresses[i])

    def test(self, X):
        ''' Testing '''
        addresses = self._mk_tuple(X)
        res = [[1 if self._wiznet[y][i].getEntry(addresses[i]) > 0 else 0 for i in range(self._nrams)].count(1) for y in self._classes]
        return max(enumerate(res), key=(lambda x: x[1]))[0]
    
    def fit(self, X, y):
        self._retina_size = self._notics * len(X[0])   # set retins size (# feature x # of tics)
        self._nrams = int(self._retina_size/self._nobits) if self._retina_size % self._nobits == 0 else int(self._retina_size/self._nobits + 1)
        self._mapping = np.arange(self._retina_size, dtype=int)
        self._classes, y = np.unique(y, return_inverse=True)
        self._nclasses = len(self._classes)
        for cl in self._classes:
            #self._wiznet[cl] = [ram.VRam(self._nloc) for _ in range(self._nrams)] alternative Ram implementations
            #self._wiznet[cl] = [ram.SRam(self._nloc) for _ in range(self._nrams)] 
            self._wiznet[cl] = [ram.WRam() for _ in range(self._nrams)] 
        if self._maptype=="random":                 # random mapping
            np.random.shuffle(self._mapping)
        self._ranges = X.max(axis=0)-X.min(axis=0)
        self._offsets = X.min(axis=0)
        self._ranges[self._ranges == 0] = 1
        if self._debug: timing_init()
        delta = 0                                   # inizialize error
        for i,sample in enumerate(X):
            if self._debug:  print("Label %d"%y[i], end='')
            self.train(sample, y[i])        
            res = self.test(sample)
            delta += abs(y[i] - res)
            if self._debug: timing_update(i,y[i]==res,title='train ',size=len(X),error=delta/float(i+1))
        if self._debug: print()
        return self

    def predict(self,X):
        if self._debug: timing_init()
        y_pred = np.array([])
        for i,sample in enumerate(X):
            y_pred = np.append(y_pred,[self.test(sample)])
            if self._debug: timing_update(i,True,title='test  ',clr=color.GREEN,size=len(X))
        if self._debug: print()
        return y_pred

    def __repr__(self): 
        return "WiSARDClassifier(n_tics: %d, n_bits:, %d, n_rams: %d, random_state: %d, n_locs: %r, mapping: %r)\n"%(self._notics, self._nobits, self._nrams, self._seed,self._nloc, self._maptype)

    def __str__(self):
        ''' Printing function'''
        return "WiSARDClassifier(n_tics: %d, n_bits:, %d, n_rams: %d)\n"%(self._notics, self._nobits, self._nrams)

    def printWiznet(self):
        rep = ""
        for cl in range(self._nclasses):
            rep += f'[{cl} '
            for j in range(self._nrams):
                rep += f'{self._wiznet[cl][j]}'
            rep += '\n'
        return rep

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {"n_bits": self._nobits, "n_tics": self._notics, "mapping": self._maptype, "debug": self._debug, "code" : self._code, "random_state": self._seed
              }

    def getMapping(self):
        return self._mapping

    def getCode(self):
        return self._code

    def getNoBits(self):
        return self._nobits

    def getNoTics(self):
        return self._notics

    def getNoRams(self):
        return self._nrams

    def getClasses(self):
        return self._classes
