from time import time
import numpy as np

from .utilities import Computations
from . import utilities as do
from .compat import Cluster

class SinglePhase:
    def __init__(self):
        pass

def initialize(self):
    computeSinglePhaseArea(self)
    self.gSP = computeSinglePhaseConductance(self)
    self.gwSPhase = self.gSP*self.mu/self.muw
    self.gnwSPhase = self.gSP*self.mu/self.munw
    
def computeSinglePhaseArea(self):
    self.areaSPhase = np.zeros(self.totElements, dtype=np.float32)
    self.areaSPhase[self.elementLists] = (
        (self.Rarray[1:-1])**2)/(4*self.Garray[1:-1])

def computeSinglePhaseConductance(self):
    gSP = np.zeros(self.totElements, dtype=np.float32)
    gSP[1:-1] = 1/self.mu*(
        (np.pi*self.Rarray[1:-1]**4/8)*(self.Garray[1:-1] > self.bndG2)
        + (self.Rarray[1:-1]**4*0.5623*(
            (self.Garray[1:-1] >= self.bndG1) & (self.Garray[1:-1] <= self.bndG2)))
        + (self.Rarray[1:-1]**4/(16*self.Garray[1:-1])*0.6*(self.Garray[1:-1] < self.bndG1))
    )
    return gSP    


def singlephase(self):
    '''determine the single phase parameters'''
    print('------------------------------------------------------------------')
    print('---------------------------Single Phase---------------------------')

    Computations(self)
    self.fluid = np.zeros(self.totElements, dtype=np.int32)
    self.fluid[-1] = 1   # already filled  
    self.cWP = Cluster(self, 0) # wetting phase cluster object
    hasFluid = self.cWP.hasFluid 

    self.cWP.doClustering(np.flatnonzero(hasFluid).astype(np.int32), 0.0, True, True, True)
    self.cWP.computeFlowrate(self.gSP)

    self.qSP = self.cWP.flowrate
    self.absPerm = self.mu*self.qSP*(self.xend - self.xstart)/self.Area_
    self.qwSPhase = self.qSP*self.mu/self.muw
    self.qnwSPhase = self.qSP*self.mu/self.munw

    self.gwLSP = self.cWP.gL*self.mu/self.muw
    self.gnwLSP = np.zeros(self.nThroats, dtype=np.float32)

    print("SPhase flowrate: w = {}, nw = {}".format(
        self.qwSPhase, self.qnwSPhase))
    print('Absolute permeability = ', self.absPerm)

    print("Time taken: {} s \n\n".format(round(time() - start, 3)))


start = time()
if __name__ == "__main__":
    SinglePhase().singlephase()
