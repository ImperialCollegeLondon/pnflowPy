from time import time
import numpy as np
from utilities import Computations
import utilities as do
from cluster import Cluster


class SinglePhase:
    def __init__(self):
        pass

def initialize(self):
    computeSinglePhaseArea(self)
    self.gSP = computeSinglePhaseConductance(self)
    self.gwSPhase = self.gSP*self.mu/self.muw
    self.gnwSPhase = self.gSP*self.mu/self.munw
    
def computeSinglePhaseArea(self):
    self.areaSPhase = np.zeros(self.totElements)
    self.areaSPhase[self.elementLists] = (
        (self.Rarray[1:-1])**2)/(4*self.Garray[1:-1])

def computeSinglePhaseConductance(self):
    gSP = np.zeros(self.totElements)
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
    gLSP = do.computegL(self, self.gSP)
    
    arrr = np.zeros(self.totElements, dtype='bool')    
    arrr[self.P1array[(gLSP > 0.0)]] = True
    arrr[self.P2array[(gLSP > 0.0)]] = True
    arrr[self.tList[(gLSP > 0.0)]] = True
    arrr = (arrr & self.connected)

    self.clusterW_ID = -5*np.ones(self.totElements, dtype=np.int32)
    self.trappedW = np.zeros(self.totElements, dtype='bool')
    self.fluid = np.zeros(self.totElements, dtype=np.int32)
    self.fluid[-1] = 1   # already filled
    self.hasWFluid = (self.fluid==0)|self.isPolygon
    self.hasWFluid[[-1,0]] = False
    self.connW = np.zeros(self.totElements, dtype='bool')
    
    self.clusterW = Cluster(self, 0)
    self.clusterW.doClustering(np.flatnonzero(arrr), arrr, 0, True, True, True)
    conn = self.clusterW.conn & self.isinsideBox
    AmatrixW, CmatrixW = do.__getValue__(self, conn, gLSP)
    presSP = np.zeros(self.nPores+2)
    presSP[conn[self.poreListS]] = do.matrixSolver(AmatrixW, CmatrixW)
    presSP[self.isOnInletBdr[self.poreListS]] = 1.0

    delSP = np.abs(presSP[self.P1array] - presSP[self.P2array])
    qp = gLSP*delSP
    
    try:
        conTToInletBdr = self._conTToInletBdr[conn[self.conTToInletBdr]]
        conTToOutletBdr = self._conTToOutletBdr[conn[self.conTToOutletBdr]]
        qinto = qp[conTToInletBdr-1].sum()
        qout = qp[conTToOutletBdr-1].sum()
        assert np.isclose(qinto, qout, atol=1e-30)
        qout = (qinto+qout)/2
    except AssertionError:
        pass

    self.absPerm = self.mu*qout*(self.xend - self.xstart)/self.Area_
    self.qSP = qout
    self.qwSPhase = self.qSP*self.mu/self.muw
    self.qnwSPhase = self.qSP*self.mu/self.munw

    self.gwLSP = gLSP*self.mu/self.muw
    self.gnwLSP = np.zeros(self.nThroats)

    print("SPhase flowrate: w = {}, nw = {}".format(
        self.qwSPhase, self.qnwSPhase))
    print('Absolute permeability = ', self.absPerm)
    print("Time taken: {} s \n\n".format(round(time() - start, 3)))

start = time()
if __name__ == "__main__":
    SinglePhase().singlephase()
