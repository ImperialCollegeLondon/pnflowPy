import numpy as np
from sortedcontainers import SortedList
from functools import partial

from tPhaseImb import TwoPhaseImbibition
import utilities as do
import tPhaseD
import tPhaseImb


class SecImbibition(TwoPhaseImbibition):
    def __init__(self, obj, writeData=False, writeTrappedData=True):
        obj.writeData = writeData
        obj.writeTrappedData = writeTrappedData

def initialize(self):   
    self.fluid[[-1, 0]] = 0, 1
    lookup_func = partial(tPhaseImb.LookupList, PcI=self.PcI, nPores=self.nPores)
    self.ElemToFill = SortedList(key=lookup_func)
    #self.ElemToFill = SortedList(key=lambda i: self.LookupList(i))
    self.capPresMin = self.maxPc
    
    self.contactAng, self.thetaRecAng, self.thetaAdvAng = self.prop_imbibition.values()
    self.is_oil_inj = False

    do.__initCornerApex__(self)
    tPhaseImb.__computePistonPc__(self)
    tPhaseImb.__computePc__(self, self.maxPc, self.elementLists, update=False)

    self._areaWP = self._cornArea.copy()
    self._areaNWP = self._centerArea.copy()
    self._condWP = self._cornCond.copy()
    self._condNWP = self._centerCond.copy()

    self.areaWPhase = self._areaWP.view()
    self.areaNWPhase = self._areaNWP.view()
    self.gWPhase = self._condWP.view()
    self.gNWPhase = self._condNWP.view()
    self.cornerArea = self._cornArea.view()
    self.centerArea = self._centerArea.view()
    self.cornerCond = self._cornCond.view()
    self.centerCond = self._centerCond.view()

    if self.writeData: tPhaseImb.__fileName__(self)
    self.primary = False

        


