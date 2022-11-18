#
# WiSARD in python: 
# Classification and Regression
# by Maurizio Giordano (2022)
#

class Ram:
    wentry = {}
    def __str__(self):
        return f"{self.wentry}"
    def getEntry(self, key):
        return self.wentry[key] if key in self.wentry.keys() else (0,0) 
    def updEntry(self, key, value):
        self.wentry[key] = (self.wentry[key][0] + 1, self.wentry[key][1] + value) if key in self.wentry.keys() else (1,value)