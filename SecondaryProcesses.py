import numpy as np
from sortedcontainers import SortedList
from functools import partial

from . import utilities as do
from . import tPhaseD
from . import tPhaseImb

class SecDrainage:
    def __init__(self, obj, writeData=False, writeTrappedData=True):
        obj.writeData = writeData
        obj.writeTrappedData = writeTrappedData
        tPhaseD.popUpdateOilInj = popUpdateOilInj

        obj.fluid[[-1, 0]] = 1, 0
        obj.capPresMax = obj.capPresMin
        obj.is_oil_inj = True
        obj.contactAng, obj.thetaRecAng, obj.thetaAdvAng = obj.prop_drainage.values()

        do.__initCornerApex__(obj)
        obj.Fd_Tr = do.__computeFd__(obj, obj.elemTriangle, obj.halfAnglesTr)
        obj.Fd_Sq = do.__computeFd__(obj, obj.elemSquare, obj.halfAnglesSq)
        tPhaseD.computePistonPc(obj)
        obj.centreEPOilInj[obj.elementLists] = 2*obj.sigma*np.cos(
            obj.thetaRecAng[obj.elementLists])/obj.Rarray[obj.elementLists]
        obj.PcD[:] = obj.PistonPcRec
        hasFluid1 = (obj.fluid==1)
        obj.PistonPcRec[hasFluid1] = obj.centreEPOilInj[hasFluid1]

        obj.ElemToFill = SortedList(key=lambda i: tPhaseD.LookupList(obj, i))
        obj.NinElemList[:] = True
        obj.prevFilled = hasFluid1
        populateToFill(obj, obj.conTToIn.copy())

        obj._cornArea = obj._areaWP.copy()
        obj._centerArea = obj._areaNWP.copy()
        obj._cornCond = obj._condWP.copy()
        obj._centerCond = obj._condNWP.copy()

        obj.areaWPhase = obj._cornArea.view()
        obj.areaNWPhase = obj._centerArea.view()
        obj.gWPhase = obj._cornCond.view()
        obj.gNWPhase = obj._centerCond.view()     
        obj.cycle += 1
        
        obj.results_dir = "quasi_static_results/"
        obj.results_str = ""
        if obj.writeData: do.__fileName__(obj)
        obj.primary = False
        obj.totNumFill = 0


def popUpdateOilInj(self):
    k = self.ElemToFill.pop(0)
    self.NinElemList[k] = True
    capPres = self.PcD[k]
    self.capPresMax = np.max([self.capPresMax, capPres])

    if not self.cWP.trapped[k]:
        arr = self.connectivity_graph[k]
        cID = np.unique(self.cNWP.clusterID[arr[self.cNWP.hasFluid[arr]]])
        self.cNWP.fill_with_phase(k, self.capPresMax, self)
        self.PistonPcRec[k] = self.centreEPOilInj[k]

        if self.isCircle[k]:
            self.cWP.unfill_phase(k, self.capPresMax)
    
        if cID.size==1:
            arr = arr[(self.fluid[arr]==0) & (~self.cWP.trapped[arr])]
            tPhaseD.update_PcD_ToFill(self, arr)
        else:
            kk = self.cNWP.clusterID[k]
            populateToFill(self, self.cNWP[kk].members)
        
        self.cnt += 1
        self.invInsideBox += self.isinsideBox[k]

    
def populateToFill(self, arr):
    arr0 = arr[(self.fluid[arr]==0)& ~(self.cWP.trapped[arr])]
    arr = arr[self.fluid[arr]==1]
    if arr.size>0:
        ids = self.cNWP.clusterID[arr]
        mem = []
        for k in np.unique(ids[ids>=0]):
            mem.extend(self.cNWP[k].members)

        arr = np.concatenate(self.connectivity_graph[mem])
        arr0 = np.union1d(arr0, arr[(self.fluid[arr]==0) & ~(self.cWP.trapped[arr])])
        
        
    if arr0.any():
        tPhaseD.update_PcD_ToFill(self, arr0)



class SecImbibition:
    def __init__(self, obj, writeData=False, writeTrappedData=True):
        obj.writeData = writeData
        obj.writeTrappedData = writeTrappedData
  
        obj.fluid[[-1, 0]] = 0, 1
        obj.ElemToFill = SortedList(key=lambda i: tPhaseImb.LookupList(i, obj.PcI, obj.nPores))
        obj.capPresMin = obj.maxPc
        
        obj.contactAng, obj.thetaRecAng, obj.thetaAdvAng = obj.prop_imbibition.values()
        obj.is_oil_inj = False

        do.__initCornerApex__(obj)
        obj.NWElemNotInToFill = obj.cNWP.hasFluid.copy()
        tPhaseImb.__computePistonPc__(obj)
        tPhaseImb.__computePc__(obj, obj.maxPc, obj.elementLists.copy(), update=False)

        obj._areaWP = obj.areaWPhase.copy()
        obj._areaNWP = obj.areaNWPhase.copy()
        obj._condWP = obj.gWPhase.copy()
        obj._condNWP = obj.gNWPhase.copy()
        obj.areaWPhase = obj._areaWP.view()
        obj.areaNWPhase = obj._areaNWP.view()
        obj.gWPhase = obj._condWP.view()
        obj.gNWPhase = obj._condNWP.view()

        obj.results_dir = "quasi_static_results/"
        obj.results_str = ""
        if obj.writeData: do.__fileName__(obj)
        obj.primary = False

        




