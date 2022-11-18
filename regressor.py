#
# WiSARD in python: 
# Classification and Regression
# by Maurizio Giordano (2022)
#
import numpy as np
from utilities import *
import ram
mypowers = 2**np.arange(32, dtype = np.uint32)[::]

class WiSARDRegressor:
    """WiSARD Regressor """
    def _mk_tuple(self, X):
        addresses = [0]*self._nrams
        for i in range(self._nrams):
            for j in range(self._nobits):
                addresses[i] += mypowers[self._nobits -1 - j] * X[self._mapping[((i * self._nobits) + j) % self._retina_size]]
        return addresses
    
    #def __init__(self,  nobits, size, map=-1, classes=[0,1], dblvl=0):
    def __init__(self,  nobits=4, size=128, seed=-1, dblvl=0):
        self._nobits = nobits
        self._datatype = 'binary'
        self._seed = seed
        self._dblvl = dblvl
        self._retina_size = size
        self._nloc = mypowers[self._nobits]
        #self._classes = classes 
        self._nrams = int(size/self._nobits) if size % self._nobits == 0 else int(size/self._nobits + 1)
        self._mapping = np.arange(self._retina_size, dtype=int)
        #self._rams = [np.full((self._nrams, self._nloc),0) for c in classes]
        self._rams = [ram.Ram() for i in range(self._nrams)] 
        if seed > -1: np.random.seed(self._seed); np.random.shuffle(self._mapping)
        
    def train(self, X, y):
        ''' Learning '''
        addresses = self._mk_tuple(X)
        for i in range(self._nrams):
            #self._rams[y][i][intuple[i]] = 1
            self._rams[i].updEntry(addresses[i], y)

    def test(self, X):
        ''' Testing '''
        addresses = self._mk_tuple(X)
        res = [sum(i) for i in zip(*[self._rams[i].getEntry(addresses[i]) for i in range(self._nrams)])]
        #print("res", [self._rams[i].getEntry(addresses[i]) for i in range(self._nrams)])
        return float(res[1])/float(res[0]) if res[0] != 0 else 0.0
        #a = [[self._rams[y][i][intuple[i]] for i in range(self._nrams)].count(1) for y in self._classes]
        #return max(enumerate(a), key=(lambda x: x[1]))[0]
    
    def fit(self, X, y):
        if self._dblvl > 0: timing_init()
        delta = 0
        for i,sample in enumerate(X):
            if self._dblvl > 1:  print("Label %d"%y[i])
            self.train(sample, y[i])        
            res = self.test(sample)
            delta += abs(y[i] - res)
            if self._dblvl > 0: timing_update(i,y[i]==res,title='train ',size=len(X),error=delta/float(i+1))
        if self._dblvl > 0: print()
        return self

    def predict(self,X):
        if self._dblvl > 0: timing_init()
        y_pred = np.array([])
        for i,sample in enumerate(X):
            y_pred = np.append(y_pred,[self.test(sample)])
            if self._dblvl > 0: timing_update(i,True,title='test  ',clr=color.GREEN,size=len(X))
        if self._dblvl > 0: print()
        return y_pred

    def __str__(self):
        ''' Printing function'''
        rep = "WiSARD (Size: %d, NoBits: %d, Seed: %d, RAMs: %r)\n"%(self._retina_size, self._nobits,self._seed,self._nrams)
        for i,l in enumerate(self._rams):  
            rep += "[%d] "%(i)
            c = 0
            for r in l:
                if c == 0: 
                    rep += ""
                else:
                    rep += "    "
                c += 1
                for e in r:
                    if e == 1:
                        rep += '\x1b[5;34;46m' + '%s'%(self._skip) + '\x1b[0m'   # light blue
                    else:
                        rep += '\x1b[2;35;40m' + '%s'%(self._skip) + '\x1b[0m'   # black
                rep += "\n"
            rep += "\n"
        return rep   

    def getDataType(self):
        return self._datatype

    def getMapping(self):
        return self._mapping

    def getNoBits(self):
        return self._nobits

    def getNoRams(self):
        return self._nrams
