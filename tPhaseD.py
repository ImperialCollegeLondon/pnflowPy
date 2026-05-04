import os
import warnings
from math import pi
from time import time
import numpy as np
import pandas as pd
from sortedcontainers import SortedList
from functools import partial
from . import utilities as do
from .compat import Cluster


class TwoPhaseDrainage:
    def __init__(self, obj, writeData=False):
        obj.writeData = writeData 
        obj.results_dir = "quasi_static_results/"
        obj.results_str = ""


def initialize(self):
    self._areaWP = self._cornArea = self.areaSPhase.copy()
    self._areaNWP = self._centerArea = np.zeros(self.totElements, dtype=np.float32) 
    self._condWP = self._cornCond = self.gwSPhase.copy()
    self._condNWP = self._centerCond = np.zeros(self.totElements, dtype=np.float32)

    self.areaWPhase = self._areaWP.view()
    self.areaNWPhase = self._areaNWP.view()
    self.gWPhase = self._condWP.view()
    self.gNWPhase = self._condNWP.view()
    self.maxCornerArea = np.zeros(self.totElements, dtype=np.float32)
    self.maxCornerCond = np.zeros(self.totElements, dtype=np.float32)

    self.cNWP = Cluster(self, 1)    # non-wetting phase cluster object
    hasNWFluid = self.cNWP.hasFluid
    self.cNWP.doClustering(
        np.flatnonzero(hasNWFluid).astype(np.int32), 0.0, True, True, True)
    
    self.contactAng, self.thetaRecAng, self.thetaAdvAng =\
        do.__wettabilityDistribution__(self)
    self.Fd_Tr = do.__computeFd__(self, self.elemTriangle, self.halfAnglesTr)
    self.Fd_Sq = do.__computeFd__(self, self.elemSquare, self.halfAnglesSq)
    
    self.nCorners_arr = np.zeros(self.totElements, dtype=np.int32)
    self.nCorners_arr[self.isTriangle] = 3
    self.nCorners_arr[self.isSquare] = 4

    do.__initCornerApex__(self)
    computePistonPc(self)
    self.PcD[:] = self.PistonPcRec
    self.centreEPOilInj = np.zeros(self.totElements, dtype=np.float32)
    self.centreEPOilInj[self.elementLists] = 2*self.sigma*np.cos(
        self.thetaRecAng[self.elementLists])/self.Rarray[self.elementLists]
    
    self.pop, self.update = 0, 0
    self.ElemToFill = SortedList(key=partial(LookupList, self))
    ElemToFill = self.conTToIn.copy()
    self.update += ElemToFill.size
    self.ElemToFill.update(ElemToFill)
    self.NinElemList = np.ones(self.totElements, dtype=np.bool_)
    self.NinElemList[ElemToFill] = False
    
    self.capPresMax = 0
    self.capPresMin = 0
    self.is_oil_inj = True
    self.cycle += 1
    self.qW, self.qNW = self.qwSPhase, 0.0
    self.krw, self.krnw = 1.0, 0.0
    self.totNumFill = 0

    
def LookupList(self, k):
    return (self.PcD[k], k > self.nPores, -k)


def drainage(self):
    start = time()
    print('---------------------------------------------------------------------------')
    print('-------------------------Two Phase Drainage Cycle {}------------------------'.format(self.cycle))

    if self.writeData:
        do.__fileName__(self)
        do.__writeHeaders__(self)
    self.results_str = ""

    self.SwTarget = max(self.finalSat, self.satW-self.dSw*0.5)
    self.PcTarget = min(self.maxPc, self.capPresMax+(
        self.minDeltaPc+abs(
            self.capPresMax)*self.deltaPcFraction)*0.1)
    self.oldPcTarget = 0
    
    while self.filling:
        self.oldSatW = self.satW
        __PDrainage__(self)
                    
        if (self.PcTarget > self.maxPc-0.001) or (
                self.satW < self.finalSat+0.00001):
            self.filling = False
            break
        
        self.oldPcTarget = self.capPresMax
        self.PcTarget = min(self.maxPc+1e-7, self.PcTarget+(
            self.minDeltaPc+abs(self.PcTarget)*self.deltaPcFraction))
        self.SwTarget = max(self.finalSat-1e-15, round((
            self.satW-self.dSw*0.75)/self.dSw)*self.dSw)

        if len(self.ElemToFill) == 0:
            self.filling = False
            self.cnt, self.totNumFill = 0, 0

            while (self.PcTarget < self.maxPc-1e-8) and (self.satW>self.finalSat):
                __CondTP_Drainage__(self)
                self.satW = do.Saturation(self, self.areaWPhase, self.areaSPhase)
                do.computePerm(self, self.capPresMax)
                self.results_str = do.writeResult(self, self.results_str, self.capPresMax)

                self.PcTarget = min(self.maxPc-1e-7, self.PcTarget+(
                    self.minDeltaPc+abs(self.PcTarget)*self.deltaPcFraction))
                if self.capPresMax == self.PcTarget: break
                else: self.capPresMax = self.PcTarget
            break

       
    if self.writeData:
        self.results_str += '\n\n'
        with open(self.file_name, 'a') as fQ:
            fQ.write(self.results_str)
                
    self.Pc = self.maxPc = self.capPresMax
    self.rpd = self.sigma/self.maxPc
    print("Number of trapped elements: W: {}  NW:{}".format(
        self.cWP.trapped.sum(), self.cNWP.trapped.sum()))
    print('No of W clusters: {}, No of NW clusters: {}'.format(
        np.count_nonzero(self.cWP.sizes), np.count_nonzero(self.cNWP.sizes)))
    self.is_oil_inj = False
   
    do.__finitCornerApex__(self, self.capPresMax)
    print('Time spent for the drainage process: ', time() - start)        
    print('==========================================================\n\n')
    print(f'no of pops: {self.pop}, no of updates: {self.update}')
    
    if self.writeData:
        os.makedirs(self.results_dir, exist_ok=True)
        filename = os.path.join(self.results_dir, 
            f"drainage_{self.title}_{int(self.capPresMax)}.pkl")
        do.saveState(self, filename)

    print('Im done with drainage!!!')

    
def __PDrainage__(self):
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    
    self.totNumFill = 0
    self.fillTarget = max(self.m_minNumFillings, int(
        self.m_initStepSize*self.totElements*(
            self.SwTarget-self.satW)))
    self.invInsideBox = 0
    endWhile = False

    while (self.PcTarget > self.capPresMax-1.0e-32) and (
            self.satW > self.SwTarget):
        self.oldSatW = self.satW
        self.invInsideBox = 0
        self.cnt = 0
        
        while self.ElemToFill and (self.invInsideBox < self.fillTarget) and (
            self.PcD[self.ElemToFill[0]] <= self.PcTarget):
            popUpdateOilInj(self)
        
        self.totNumFill += self.cnt
        if not self.ElemToFill:
            self.PcTarget = self.capPresMax
            break

        if (self.PcD[self.ElemToFill[0]] > self.PcTarget):
            self.capPresMax = max(self.capPresMax, self.PcTarget)
            endWhile = True
           
        __CondTP_Drainage__(self)
        self.satW = do.Saturation(self, self.areaWPhase, self.areaSPhase)
        if self.satW-self.oldSatW!=0.0:
            self.fillTarget = max(self.m_minNumFillings, int(min(
                self.fillTarget*self.m_maxFillIncrease,
                self.m_extrapCutBack*(self.invInsideBox / (
                    self.satW-self.oldSatW))*(self.SwTarget-self.satW))))
        
        if endWhile:
            break
    
    if endWhile:
        self.capPresMax = self.PcTarget
    else:
        self.PcTarget = self.capPresMax

    __CondTP_Drainage__(self)
    self.satW = do.Saturation(self, self.areaWPhase, self.areaSPhase)
    do.computePerm(self, self.capPresMax)
    
    self.results_str = do.writeResult(self, self.results_str, self.capPresMax)


def popUpdateOilInj(self):
    self.pop += 1
    k = self.ElemToFill.pop(0)
    self.NinElemList[k] = True
    capPres = self.PcD[k]
    self.capPresMax = max(self.capPresMax, capPres)
          
    if not self.cWP.trapped[k]:
        self.cNWP.fill_with_phase(k, self.capPresMax, self)
        self.PistonPcRec[k] = self.centreEPOilInj[k]
        arr = self.connectivity_graph[k]
        cond = (self.fluid[arr]==0) & (~self.cWP.trapped[arr])
        arr = arr[cond.astype(np.bool_)]
        if self.isCircle[k]:
            self.cWP.unfill_phase(k, self.capPresMax)
        self.cnt += 1
        self.invInsideBox += self.isinsideBox[k]
        update_PcD_ToFill(self, arr)


def __computePc__(self, arrr, Fd):
    Pc = self.sigma*(1+2*np.sqrt(pi*self.Garray[arrr]))*np.cos(
        self.contactAng[arrr])*Fd/self.Rarray[arrr]
    return Pc


def computePistonPc(self) -> None:
    self.PistonPcRec = np.zeros(self.totElements, dtype=np.float32)
    self.PistonPcRec[self.elemCircle] = 2*self.sigma*np.cos(
        self.contactAng[self.elemCircle])/self.Rarray[self.elemCircle]
    self.PistonPcRec[self.elemTriangle] = __computePc__(
        self, self.elemTriangle, self.Fd_Tr)
    self.PistonPcRec[self.elemSquare] = __computePc__(
        self, self.elemSquare, self.Fd_Sq)
    

def __func(self, i):
    '''returns the minimum receding Pc for pistonlike displacement'''
    
    arr = self.connectivity_graph[i]
    if arr.any():
        cond = self.cNWP.hasFluid[arr].astype(np.bool_)
        if cond.any():
            return self.PistonPcRec[arr[cond]].min()
    return 0


def update_PcD_ToFill(self, arr) -> None:
    minNeiPc = np.array([*map(lambda ar: __func(self, ar), arr)])
    entryPc = np.maximum(0.999*minNeiPc+0.001*self.PistonPcRec[
        arr], self.PistonPcRec[arr])
    
    ''' elements to be removed before updating PcD '''
    cond2 = (entryPc != self.PcD[arr])
    arr1 = arr[cond2 & (~self.NinElemList[arr])]
    [self.ElemToFill.remove(i) for i in arr1]
    self.NinElemList[arr1] = True

    ''' updating elements with new PcD '''
    self.PcD[arr[cond2]] = entryPc[cond2]

    ''' updating the ToFill elements '''
    cond3 = (self.NinElemList[arr])
    arr3 = arr[cond3]
    self.update += arr3.size
    self.ElemToFill.update(arr3)
    self.NinElemList[arr3] = False   



def __CondTP_Drainage__(self, Pc=None):
    # to suppress the FutureWarning and SettingWithCopyWarning respectively
    warnings.simplefilter(action='ignore', category=FutureWarning)
    pd.options.mode.chained_assignment = None

    arrr = (self.fluid==1)
    arrr[[0, -1]] = False
    if not np.any(arrr):
        return

    if Pc is None:
        Pc = np.full(self.totElements, self.capPresMax, dtype=np.float64)

    do.update_areas_conductances(self, arrr, Pc, False, True, False)
        

