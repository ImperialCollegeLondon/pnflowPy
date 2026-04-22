import os

import warnings
from time import time
from numba import njit, prange

import numpy as np
import pandas as pd
from sortedcontainers import SortedList
from functools import partial
from . import utilities as do


class TwoPhaseImbibition:    
    def __init__(self, obj, writeData=False, writeTrappedData=False, includeTrapping=True):
        obj.includeTrapping = includeTrapping
        obj.writeData = writeData
        obj.writeTrappedData = writeTrappedData
        obj.results_dir = "quasi_static_results/"
        obj.results_str = ""

def initialize(self):
    self.porebodyPc = np.zeros(self.totElements, dtype=np.float32)
    self.snapoffPc = np.zeros(self.totElements, dtype=np.float32)
    self.PistonPcAdv = np.zeros(self.totElements, dtype=np.float32)
    self.pistonPc_PHing = np.zeros(self.nPores+2, dtype=np.float32)
    self.fluid[[-1, 0]] = 0, 1  
    self.fillmech = np.full(self.totElements, -5, dtype=np.int32)
    self.capPresMin = self.maxPc
    
    self.maxCornerArea = np.zeros(self.totElements, dtype=np.float32)
    self.maxCornerCond = np.zeros(self.totElements, dtype=np.float32)
    
    self.contactAng, self.thetaRecAng, self.thetaAdvAng =\
       do.__wettabilityDistribution__(self)
    self.cosThetaAdvAng = np.cos(self.thetaAdvAng)
    self.sinThetaAdvAng = np.sin(self.thetaAdvAng)
    self.cosThetaRecAng = np.cos(self.thetaRecAng)
    self.sinThetaRecAng = np.sin(self.thetaRecAng)

    self.is_oil_inj = False
    do.__initCornerApex__(self)
    
    __computePistonPc__(self)
    self.randNum = self.rand(self.totElements)
    __computeSnapoffPc__(self)
    
    lookup_func = partial(LookupList, PcI=self.PcI, nPores=self.nPores)
    self.ElemToFill = SortedList(key=lookup_func)
    self.NWElemNotInToFill = self.cNWP.hasFluid.astype(np.bool_)
    __computePc__(self, self.maxPc, self.elementLists.copy(), update=False, 
                trapping=self.includeTrapping)

    self.pop, self.update = 0, 0
    self._cornArea = self._areaWP.copy()
    self._centerArea = self._areaNWP.copy()
    self._cornCond = self._condWP.copy()
    self._centerCond = self._condNWP.copy()
            
    self.specialPcD = np.zeros(self.totElements, dtype=np.float32)  
    

def imbibition(self):
    start = time()
    print('----------------------------------------------------------------------------------')
    print('---------------------------------Two Phase Imbibition Cycle {}---------------------'.format(self.cycle))
    
    if self.writeData:
        do.__fileName__(self)
        do.__writeHeaders__(self)
    else: 
        self.results_str = ""
        self.totNumFill = 0
        
    self.SwTarget = min(self.finalSat, self.satW+self.dSw*0.5)
    self.PcTarget = max(self.minPc, self.capPresMin-(
        self.minDeltaPc+abs(self.capPresMin)*self.deltaPcFraction)*0.1)
    self.fillTarget = max(self.m_minNumFillings, int(
        self.m_initStepSize*(self.totElements)*(
            self.satW-self.SwTarget)))
               
    while self.filling:
        
        __PImbibition__(self)        
        if (self.PcTarget < self.minPc+0.001) or (
                self.satW > self.finalSat-0.00001):
            self.filling = False
            break

        if (len(self.ElemToFill)==0):
            self.filling = False
            self.cnt, self.totNumFill = 0, 0
            _pclist = np.array([-1e-7, self.minPc])
            _pclist = np.sort(_pclist[_pclist<self.capPresMin])[::-1]
            for Pc in _pclist:
                self.capPresMin = Pc
                __CondTPImbibition__(self)
                self.satW = do.Saturation(self, self.areaWPhase, self.areaSPhase)
                do.computePerm(self, self.capPresMin)
                self.results_str = do.writeResult(self, self.results_str, self.capPresMin)
            break

        self.PcTarget = max(self.minPc+1e-7, self.PcTarget-(
            self.minDeltaPc+abs(
                self.PcTarget)*self.deltaPcFraction+1e-16))
        self.SwTarget = min(self.finalSat+1e-15, round((
            self.satW+self.dSw*0.75)/self.dSw)*self.dSw)

    if self.writeData:
        self.results_str += '\n\n'
        with open(self.file_name, 'a') as fQ:
            fQ.write(self.results_str)
        if self.writeTrappedData:
            do.__writeTrappedData__(self)

    print("Number of trapped elements: W: {}  NW:{}".format(
        self.cWP.trapped.sum(), self.cNWP.trapped.sum()))
    print('No of W clusters: {}, No of NW clusters: {}'.format(
        np.count_nonzero(self.cWP.sizes),
        np.count_nonzero(self.cNWP.sizes)))
    
    print('Time spent for the imbibition process: ', time() - start)
    print('===========================================================\n\n')

    print(f'no of pops: {self.pop}, no of updates: {self.update}')
    print('Im done with imbibition !!!')
        
    
def __PImbibition__(self):
    self.totNumFill = 0
    continue_to_fill = True
    while (self.PcTarget-1.0e-32 < self.capPresMin) and (
            self.satW <= self.SwTarget):
        self.oldSatW = self.satW
        self.invInsideBox = 0
        self.cnt = 0
        
        while (self.invInsideBox < self.fillTarget) and (
            len(self.ElemToFill) != 0) and (
                self.PcI[self.ElemToFill[0]] >= self.PcTarget):

            mem = self.cNWP[0].members
            if (not self.fillTillNWDisconnected) or (
                self.toInBdr[mem].any() and self.toOutBdr[mem].any()):
                popUpdateWaterInj(self)
                if not self.filling:  # remove later
                    return
            else:
                self.filling = False
                self.PcTarget = self.capPresMin
                break

        if len(self.ElemToFill) == 0:
            self.capPresMin = min(self.capPresMin, self.PcTarget)
            continue_to_fill = False
        elif (self.PcI[self.ElemToFill[0]] < self.PcTarget) and (
            self.capPresMin > self.PcTarget):
            self.capPresMin = self.PcTarget

        __CondTPImbibition__(self)
        self.satW = do.Saturation(self, self.areaWPhase, self.areaSPhase)
        self.totNumFill += self.cnt
        
        if not continue_to_fill or not self.filling or (
            self.PcI[self.ElemToFill[0]] < self.PcTarget):
            break
    

    if not continue_to_fill: pass
    elif (self.PcI[self.ElemToFill[0]] < self.PcTarget) and (
            self.capPresMin > self.PcTarget):
        self.capPresMin = self.PcTarget
    else:
        self.PcTarget = self.capPresMin
    
    __CondTPImbibition__(self)
    self.satW = do.Saturation(self, self.areaWPhase, self.areaSPhase)
    do.computePerm(self, self.capPresMin)
    self.results_str = do.writeResult(self, self.results_str, self.capPresMin)
    
    

def popUpdateWaterInj(self):

    self.pop += 1
    k = self.ElemToFill.pop(0)
    capPres = self.PcI[k]
    self.capPresMin = np.min([self.capPresMin, capPres])
    
    if not self.cNWP.trapped[k]:
        self.cWP.fill_with_phase(k, self.capPresMin, self)
        self.cNWP.unfill_phase(k, self.capPresMin)        
        neigh = self.connectivity_graph[k]
        cond = (self.cNWP.hasFluid[neigh] & ~self.cNWP.trapped[neigh])
        neigh = neigh[cond.astype(np.bool_)]
        if neigh.size>0:
            __computePc__(self, self.capPresMin, neigh, trapping=self.includeTrapping)
        
        self.specialPcD[k] = self.capPresMin
        self.fillmech[k] = 1*(self.PistonPcAdv[k]==capPres)+2*(
            self.porebodyPc[k]==capPres)+3*(self.snapoffPc[k]==capPres)
        self.cnt += 1
        self.invInsideBox += self.isinsideBox[k]
        
    
def __CondTPImbibition__(self, arrr=None, Pc=None, updateArea=True, overrideTrapping=False):
    if arrr is None:
        arrr = np.ones(self.totElements, dtype=bool)
        arrr[[-1,0]] = False

    if Pc is None:
        Pc = np.full(self.totElements, self.capPresMin, dtype=np.float64)
        
    do.update_areas_conductances(self, arrr, Pc, False, updateArea, overrideTrapping)
     

def __computePistonPc__(self):
    conda = (self.fluid == 0)
    condb = (self.fluid == 1) & (self.Garray < self.bndG2)  #polygons filled with w
    condc = (self.fluid == 1) & (self.Garray >= self.bndG2) #circles filled with nw
    condac = (conda | condc)
    condac[[-1,0]] = False

    self.PistonPcAdv[condac] = 2.0*self.sigma*self.cosThetaAdvAng[condac]/self.Rarray[condac]
    conda = conda & (self.maxPc<self.PistonPcRec)
    self.PistonPcAdv[conda] = self.maxPc*self.cosThetaAdvAng[conda]/self.cosThetaRecAng[conda]

    normThresPress = (self.Rarray*self.maxPc)/self.sigma
    angSum = np.zeros(self.totElements)
    angSum[self.elemTriangle] = np.cos(self.thetaRecAng[
        self.elemTriangle][:, np.newaxis] + self.halfAnglesTr).sum(axis=1)
    angSum[self.elemSquare] = np.cos(self.thetaRecAng[
        self.elemSquare][:, np.newaxis] + self.halfAnglesSq).sum(axis=1)
    rhsMaxAdvConAng = (-4.0*self.Garray*angSum)/(
        normThresPress-self.cosThetaRecAng+12.0*self.Garray*self.sinThetaRecAng)
    rhsMaxAdvConAng = np.clip(rhsMaxAdvConAng, -1.0, 1.0)
    m_maxConAngSpont = np.arccos(rhsMaxAdvConAng)

    condd = condb & (self.thetaAdvAng<m_maxConAngSpont) #calculate PHing
    __PistonPcHing__(self, condd)

    conde = np.zeros(self.totElements, dtype=np.bool_)
    conde[self.elemTriangle] = condb[self.elemTriangle] & (~condd[self.elemTriangle]) & (
        self.thetaAdvAng[self.elemTriangle] <= np.pi/2+self.halfAnglesTr[:, 0])
    self.PistonPcAdv[conde] = 2.0*self.sigma*self.cosThetaAdvAng[conde]/self.Rarray[conde]
    
    condf = condb & (~condd) & (~conde) 
    self.PistonPcAdv[condf] = 2.0*self.sigma*self.cosThetaAdvAng[condf]/self.Rarray[condf]
    

def __PistonPcHing__(self, arrr, accurate=True, overidetrapping=True):
    ''' compute entry capillary pressures for piston displacement '''
  
    arrrT = self.isTriangle & arrr
    arrrS = self.isSquare & arrr
    initialPc = np.divide(1.1*self.sigma*2.0*self.cosThetaAdvAng, self.Rarray, where=(self.Rarray!=0.0))
    delta = 0.0 if accurate else self._delta

    if np.any(arrrT):        
        self.PistonPcAdv[arrrT] = do.Pc_pistonHing_numba(
            arrrT, self.m_halfAngles, self.m_cornExists, self.m_initOrMaxPcHist, 
            self.m_initOrMinApexDistHist, self.m_advPc, self.m_recPc, 
            self.m_initedApexDist, initialPc, self.thetaAdvAng, self.thetaRecAng,
            self.cosThetaAdvAng, self.Rarray, self.Garray, self.cWP.trapped, self.cNWP.trapped, 
            self.cWP.pc, self.cNWP.pc, self.cWP.clusterID, self.cNWP.clusterID, self.sigma, 
            delta, self.EPSILON, self.MAX_ITER, self.MOLECULAR_LENGTH, 3)
    
    if np.any(arrrS):
        self.PistonPcAdv[arrrS] = do.Pc_pistonHing_numba(
            arrrS, self.m_halfAngles, self.m_cornExists, self.m_initOrMaxPcHist, 
            self.m_initOrMinApexDistHist, self.m_advPc, self.m_recPc, 
            self.m_initedApexDist, initialPc, self.thetaAdvAng, self.thetaRecAng,
            self.cosThetaAdvAng, self.Rarray, self.Garray, self.cWP.trapped, self.cNWP.trapped, 
            self.cWP.pc, self.cNWP.pc, self.cWP.clusterID, self.cNWP.clusterID, self.sigma, 
            delta, self.EPSILON, self.MAX_ITER, self.MOLECULAR_LENGTH, 4)
        
 

def __computeSnapoffPc__(self):
    ''' compute entry capillary pressure for Snap-off filling '''
    self.snapoffPc1 = self.sigma/self.Rarray[self.elemTriangle]*(self.cosThetaAdvAng[
        self.elemTriangle] - 2*self.sinThetaAdvAng[self.elemTriangle]/self.cotBetaTr[
            :, 0:2].sum(axis=1))
    
    apexDistTr = self.sigma*np.cos(self.thetaRecAng[
        self.elemTriangle][:, np.newaxis]+self.halfAnglesTr)/np.sin(
            self.halfAnglesTr)
    self._thetaHi_a = apexDistTr*np.sin(self.halfAnglesTr)/self.sigma
    
    self._snapoffPc2a = self.sigma/self.Rarray[self.elemTriangle]/(
        self.cotBetaTr[:, [0, 2]].sum(axis=1))
    self._snapoffPc2num = self.cosThetaAdvAng[self.elemTriangle]*self.cotBetaTr[:, 0] -\
        self.sinThetaAdvAng[self.elemTriangle]

    self.snapoffPc[self.elemSquare] = self.sigma/self.Rarray[self.elemSquare]*(
        self.cosThetaAdvAng[self.elemSquare] - self.sinThetaAdvAng[self.elemSquare])
    

def __updateSnapoffPc__(self, Pc: float):
    ''' update entry capillary pressure for Snap-off filling '''
    arrrTr = (self.fluid[self.elemTriangle] == 1)
    thetaHi = np.arccos(self._thetaHi_a[arrrTr]*Pc/self.maxPc)
    snapoffPc2 = self._snapoffPc2a[arrrTr]*(
        self._snapoffPc2num[arrrTr]+np.cos(thetaHi[:, 2])*self.cotBetaTr[arrrTr, 2]
        - np.sin(thetaHi[:, 2]))
    self.snapoffPc[self.elemTriangle[arrrTr]] = np.max(
        [self.snapoffPc1[arrrTr], snapoffPc2], axis=0)


@staticmethod
def LookupList(k, PcI, nPores):
    return (-int(PcI[k]*1e9), k<=nPores, -k)

def __computePc__(self, Pc, arr, update=True, trapping=True):
    entryPc = self.PistonPcAdv.copy()
    maxNeiPistonPrs = np.zeros(self.totElements, dtype=np.float32)
    _arr = arr[self.cNWP.hasFluid[arr].astype(np.bool_)] # elements filled with nw
    arrP = _arr[(_arr <= self.nPores)]   #pores filled with nw
    arrT = _arr[(_arr > self.nPores)]      #throats filled with nw

    hasOnlyWFluid = (self.fluid==0)
    valid_T_WF = hasOnlyWFluid[self.PTConnections]&(self.PTValid)
    if arrP.size>0:
        ''' identify pores where porebody filling could occur '''
        arr1 = arrP[np.sum(valid_T_WF[arrP], axis=1)>0]
        arr1 = arr1[(self.thetaAdvAng[arr1]<np.pi/2.0)]
        __porebodyFilling__(self, arr1)
        entryPc[arr1] = self.porebodyPc[arr1]
    
        ''' update the piston-like entry Pc '''
        maxNeiPistonPrs[arrP] = np.max(
            self.PistonPcAdv[self.PTConnections[arrP]], axis=1, initial=0.0,
            where=valid_T_WF[arrP])
    
    if arrT.size>0:
        ''' update the piston-like entry Pc '''
        _arrT = arrT-self.nPores
        maxNeiPistonPrs[arrT] = np.max(
            self.PistonPcAdv[self.TPConnections[_arrT]], axis=1, initial=0.0,
            where=(hasOnlyWFluid[self.TPConnections[_arrT]]))
        
    condb = (maxNeiPistonPrs > 0.0)
    entryPc[condb] = np.minimum(0.999*maxNeiPistonPrs[
        condb]+0.001*entryPc[condb], entryPc[condb])
    
    ''' Snap-off filling '''
    __updateSnapoffPc__(self, Pc)
    conda = (maxNeiPistonPrs > 0.0) & (entryPc>self.snapoffPc)
    toSnapoff = (~conda)&self.isPolygon
    entryPc[toSnapoff] = self.snapoffPc[toSnapoff]

    ''' update the toFill list '''
    if update:
        ''' update PcI '''   
        diff = (self.PcI[_arr]!=entryPc[_arr])
        changed = diff&(~self.NWElemNotInToFill[_arr])
        for i in _arr[changed]: self.ElemToFill.discard(i)
        self.PcI[arr] = entryPc[arr]
        ''' add to the toFill list '''        
        to_add = _arr[(diff|self.NWElemNotInToFill[_arr])]
        self.update += _arr.size
        self.ElemToFill.update(to_add)
        self.NWElemNotInToFill[to_add] = False
    else:
        self.PcI[arr] = entryPc[arr]
        _arr = __func4(self, _arr, trapping)
        self.update += _arr.size
        self.ElemToFill.update(_arr)
    

def __func4(self, arr, trapping=True):
    ''' ensures that all elements to be added to the tofill list have 
    (i) the non-wetting fluid; 
    (ii) the wetting fluid in the corners or a neighbouring element;
    (iii) the wetting fluid is not trapped.'''

    hasWFluid = self.cWP.hasFluid[arr].astype(np.bool_)
    arrW = arr[hasWFluid]

    arrNW = arr[~hasWFluid]
    arrP = arrNW[arrNW<=self.nPores]
    arrPT = self.PTConnections[arrP]
    arrT = arrNW[arrNW>self.nPores]
    arrTP = self.TPConnections[arrT-self.nPores]

    if trapping:
        hasValidNeighP = arrP[np.any(
            (self.cWP.hasFluid[arrPT])&(~self.cWP.trapped[arrPT])&self.PTValid[arrP], axis=1)]
        hasValidNeighT = arrT[np.any(
            (arrTP==-1) | ((self.cWP.hasFluid[arrTP])&(~self.cWP.trapped[arrTP])&(arrTP>0)), axis=1)]
    else:
        hasValidNeighP = arrP[np.any((self.cWP.hasFluid[arrPT])&self.PTValid[arrP], axis=1)]
        hasValidNeighT = arrT[np.any((arrTP==-1) | ((self.cWP.hasFluid[arrTP])&(arrTP>0)), axis=1)]
    
    arrr = np.concatenate((arrW, hasValidNeighP, hasValidNeighT))
    self.NWElemNotInToFill[arrr] = False
    return arrr
   

def __porebodyFilling__(self, ind):
    if ind.size > 0:
        arr = self.PTConnections[ind]
        cond = (self.fluid[arr]==1)&self.PTValid[ind]  
        arr2 = np.sort(np.where(cond, self.randNum[arr], np.nan))[:, :6]
        cond1 = (arr2!=np.nanmax(arr2, axis=1, initial=0.0)[:,np.newaxis])&(~np.isnan(arr2))
        sumrand = np.sum(arr2, where=cond1, axis=1)*15000

        #Blunt2
        self.porebodyPc[ind] = self.sigma*(
            2*self.cosThetaAdvAng[ind]/self.Rarray[ind] - sumrand)
