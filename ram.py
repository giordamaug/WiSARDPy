#
# WiSARD in python: 
# Classification and Regression
# by Maurizio Giordano (2022)
#

class Ram:
    int nlocs = 0
    Dict wentry = {}
    def __init__(self, nlocs):
         self.nlocs = nlocs
    def __str__(self):
        return f"{self.nlocs}({self.age})"
    def getEntry(self, key):
        return wentry[key] if key in wentry.keys() else None 
    def setEntry(self, key, value):
        return (wentry[key] := value)    
