import numpy as np
from sortedcontainers import SortedList
from functools import partial

import utilities as do
import tPhaseD

class SecDrainage:
    def __init__(self, obj, writeData=False, writeTrappedData=True):
        obj.writeData = writeData
        obj.writeTrappedData = writeTrappedData
        tPhaseD.popUpdateOilInj = popUpdateOilInj

def initialize(self):
    self.fluid[[-1, 0]] = 1, 0
    self.capPresMax = self.capPresMin
    self.is_oil_inj = True
    self.contactAng, self.thetaRecAng, self.thetaAdvAng = self.prop_drainage.values()

    do.__initCornerApex__(self)
    self.Fd_Tr = do.__computeFd__(self, self.elemTriangle, self.halfAnglesTr)
    self.Fd_Sq = do.__computeFd__(self, self.elemSquare, self.halfAnglesSq)
    tPhaseD.__computePistonPc__(self)
    self.centreEPOilInj[self.elementLists] = 2*self.sigma*np.cos(
        self.thetaRecAng[self.elementLists])/self.Rarray[self.elementLists]
    self.PcD[:] = self.PistonPcRec
    self.PistonPcRec[self.fluid==1] = self.centreEPOilInj[self.fluid==1]
    self.ElemToFill = SortedList(key=partial(tPhaseD.LookupList, self))
    self.NinElemList[:] = True
    self.prevFilled = (self.fluid==1)
    populateToFill(self, self.conTToIn.copy())

    self._cornArea = self._areaWP.copy()
    self._centerArea = self._areaNWP.copy()
    self._cornCond = self._condWP.copy()
    self._centerCond = self._condNWP.copy()
    self.areaWPhase = self._cornArea.view()
    self.areaNWPhase = self._centerArea.view()
    self.gWPhase = self._cornCond.view()
    self.gNWPhase = self._centerCond.view()
    
    self.cycle += 1
    
    if self.writeData: tPhaseD.__fileName__(self)
    self.primary = False
    self.totNumFill = 0
    
def popUpdateOilInj(self):
    k = self.ElemToFill.pop(0)
    capPres = self.PcD[k]
    self.capPresMax = np.max([self.capPresMax, capPres])

    try:
        assert not self.trappedW[k]
        self.fluid[k] = 1
        self.hasNWFluid[k] = True
        self.connNW[k] = True
        self.clusterNW_ID[k] = 0
        self.clusterNW.members[0, k] = True
        self.PistonPcRec[k] = self.centreEPOilInj[k]
        arr = self.elem[k].neighbours[self.elem[k].neighbours>0]
        arr0 = arr[(self.fluid[arr]==0) & (~self.trappedW[arr])]
        arr1 = arr[self.hasNWFluid[arr]]
        try:
            assert self.isCircle[k]
            kk = self.clusterW_ID[k]
            self.clusterW_ID[k] = -5
            self.clusterW.members[kk,k] = False
            self.connW[k] = False
            self.hasWFluid[k] = False
            do.check_Trapping_Clustering(
                self, arr0.copy(), self.hasWFluid.copy(), 0, self.capPresMax, True)
        except AssertionError:
            pass
        try:
            assert arr1.size>0
            ids = self.clusterNW_ID[arr1]
            ids = ids[ids>0]
            assert ids.size>0
            ''' need to coalesce '''
            mem = self.elementListS[self.clusterNW.members[ids].any(axis=0)]
            self.connNW[mem] = self.clusterNW.connected[0]
            self.clusterNW_ID[mem] = 0
            self.clusterNW.members[ids] = False
            self.clusterNW.members[0, mem] = True
            self.clusterNW.availableID.update(ids)
            self.trappedNW[mem] = False
            populateToFill(self, mem)
        except AssertionError:
            tPhaseD.__update_PcD_ToFill__(self, arr0)
        
        self.cnt += 1
        self.invInsideBox += self.isinsideBox[k]
    except (AssertionError, IndexError):
        pass


def populateToFill(self, arr):
    done = np.zeros(self.totElements, dtype='bool')
    elemToFill = np.zeros(self.totElements, dtype='bool')
    fluid0 = (self.fluid==0)
    fluid1 = (self.fluid==1)
    done[arr] = True
    done[[-1,0]] = True
    elemToFill[arr[fluid0[arr]]] = True
    arr = arr[fluid1[arr]]
    
    temp = np.zeros(self.totElements, dtype='bool')
    while True:
        arrP = arr[arr<=self.nPores]
        temp[self.PTConnections[arrP][self.PTValid[arrP]]] = True
        temp[self.TPConnections[arr[arr>self.nPores]-self.nPores]] = True
        temp[done] = False
        elemToFill[temp & fluid0] = True
        arr = self.elementListS[(temp & fluid1)]
        if not any(arr):
            break            
        done[temp] = True

    tPhaseD.__update_PcD_ToFill__(self, self.elementListS[elemToFill])



