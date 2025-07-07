import os
import warnings
from math import pi
from time import time
import numpy as np
import pandas as pd
from sortedcontainers import SortedList
from functools import partial

from clustering import Cluster
import utilities as do


class TwoPhaseDrainage:
    def __init__(self, obj, writeData=False, writeTrappedData=False):
        obj.writeData = writeData
        obj.writeTrappedData = writeTrappedData       

def initialize(self):
    self.fluid = np.zeros(self.totElements, dtype='int')
    self.fluid[-1] = 1   # already filled
    self.hasWFluid = (self.fluid==0)|self.isPolygon
    self.hasWFluid[[-1,0]] = False
    self.hasNWFluid = (self.fluid==1)
    self.hasNWFluid[[-1,0]] = False
    self.trappedW = np.zeros(self.totElements, dtype='bool')
    self.trappedNW = np.zeros(self.totElements, dtype='bool')
    
    self._areaWP = self._cornArea = self.areaSPhase.copy()
    self._areaNWP = self._centerArea = np.zeros(self.totElements) 
    self._condWP = self._cornCond = self.gwSPhase.copy()
    self._condNWP = self._centerCond = np.zeros(self.totElements)

    self.areaWPhase = self._areaWP.view()
    self.areaNWPhase = self._areaNWP.view()
    self.gWPhase = self._condWP.view()
    self.gNWPhase = self._condNWP.view()
    
    self.clusterW = Cluster(self, 0)
    self.clusterNW = Cluster(self, 1)
    self.clusterW_ID = -5*np.ones(self.totElements, dtype='int')
    self.clusterNW_ID = -5*np.ones(self.totElements, dtype='int')
    
    self.connNW = np.zeros(self.totElements, dtype='bool')
    arrr = self.hasWFluid.copy()
    arrr[[0,-1]] = False
    do.check_Trapping_Clustering(
        self, self.elementListS[arrr], arrr.copy(), 0, 0, True, False)
   
    self.contactAng, self.thetaRecAng, self.thetaAdvAng =\
        do.__wettabilityDistribution__(self)
    self.Fd_Tr = do.__computeFd__(self, self.elemTriangle, self.halfAnglesTr)
    self.Fd_Sq = do.__computeFd__(self, self.elemSquare, self.halfAnglesSq)
       
    do.__initCornerApex__(self)
    __computePistonPc__(self)
    self.PcD[:] = self.PistonPcRec
    self.centreEPOilInj = np.zeros(self.totElements)
    self.centreEPOilInj[self.elementLists] = 2*self.sigma*np.cos(
        self.thetaRecAng[self.elementLists])/self.Rarray[self.elementLists]
    
    self.pop, self.update = 0, 0
    self.ElemToFill = SortedList(key=partial(LookupList, self))
    ElemToFill = self.conTToIn.copy()
    self.update += ElemToFill.size
    self.ElemToFill.update(ElemToFill)
    self.NinElemList = np.ones(self.totElements, dtype='bool')
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
        __fileName__(self)
        __writeHeadersD__(self)
    else: self.resultD_str = ""

    self.SwTarget = max(self.finalSat, self.satW-self.dSw*0.5)
    self.PcTarget = min(self.maxPc, self.capPresMax+(
        self.minDeltaPc+abs(
            self.capPresMax)*self.deltaPcFraction)*0.1)
    self.oldPcTarget = 0
    self.resultD_str = do.writeResult(self, self.resultD_str, self.capPresMin)
    
    # import dill
    # MEMORY_DIR = f"./saved_simulation_{self.title}"
    # os.makedirs(MEMORY_DIR, exist_ok=True)
    # targetFluid_pore = np.loadtxt('/home/aiadebimpe/PoreFlow/data/fPores_BentSepi_drainage_fluid_occupancy.dat', dtype=int)
    # targetFluid_throat = np.loadtxt('/home/aiadebimpe/PoreFlow/data/fThroats_BentSepi_drainage_fluid_occupancy.dat', dtype=int)
    # targetFluid = np.zeros_like(self.fluid)
    # targetFluid[self.poreList] = (targetFluid_pore==2)
    # targetFluid[self.tList] = (targetFluid_throat==2)
    
    while self.filling:
        self.oldSatW = self.satW
        __PDrainage__(self)
        
        #MAD = np.sum(np.abs(self.fluid-targetFluid)*self.volarray)/np.sum(self.volarray)*100
        #print(self.capPresMax, targetFluid.sum(), self.fluid.sum(), MAD)
        # with open(os.path.join(MEMORY_DIR, f"drainage_{self.capPresMax}.pkl"),"wb") as f:
            # dill.dump(self, f)
        
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
                self.resultD_str = do.writeResult(self, self.resultD_str, self.capPresMax)

                self.PcTarget = min(self.maxPc-1e-7, self.PcTarget+(
                    self.minDeltaPc+abs(self.PcTarget)*self.deltaPcFraction))
                if self.capPresMax == self.PcTarget: break
                else: self.capPresMax = self.PcTarget
            break
        
    if self.writeData:
        with open(self.file_name, 'a') as fQ:
            fQ.write(self.resultD_str)
        if self.writeTrappedData:
            __writeTrappedData__(self)

    self.maxPc = self.capPresMax
    self.rpd = self.sigma/self.maxPc
    print("Number of trapped elements: W: {}  NW:{}".format(
        self.trappedW.sum(), self.trappedNW.sum()))
    print('No of W clusters: {}, No of NW clusters: {}'.format(
        np.count_nonzero(self.clusterW.size), 
        np.count_nonzero(self.clusterNW.size)))
    self.is_oil_inj = False
    do.__finitCornerApex__(self, self.capPresMax)
    print('Time spent for the drainage process: ', time() - start)        
    print('==========================================================\n\n')

    print(f'no of pops: {self.pop}, no of updates: {self.update}')
    

def popUpdateOilInj(self):
    self.pop += 1
    k = self.ElemToFill.pop(0)
    capPres = self.PcD[k]
    self.capPresMax = max(self.capPresMax, capPres)
    if not self.trappedW[k]:
        self.fluid[k] = 1
        self.hasNWFluid[k] = True
        self.connNW[k] = True
        self.clusterNW_ID[k] = 0
        self.clusterNW.members[0, k] = True
        self.PistonPcRec[k] = self.centreEPOilInj[k]
        arr = self.elem[k].neighbours[self.elem[k].neighbours>0]
        arr = arr[(self.fluid[arr]==0) & (~self.trappedW[arr])]
        if self.isCircle[k]:
            kk = self.clusterW_ID[k]
            self.clusterW_ID[k] = -5
            self.clusterW.members[kk,k] = False
            self.connW[k] = False
            self.hasWFluid[k] = False
            do.check_Trapping_Clustering(
                self, arr.copy(), self.hasWFluid.copy(), 0, self.capPresMax, True)        
        self.cnt += 1
        self.invInsideBox += self.isinsideBox[k]
        __update_PcD_ToFill__(self, arr)            
   

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
    self.resultD_str = do.writeResult(self, self.resultD_str, self.capPresMax)


def __computePc__(self, arrr, Fd):
    Pc = self.sigma*(1+2*np.sqrt(pi*self.Garray[arrr]))*np.cos(
        self.contactAng[arrr])*Fd/self.Rarray[arrr]
    return Pc

def __computePistonPc__(self) -> None:
    self.PistonPcRec = np.zeros(self.totElements)
    self.PistonPcRec[self.elemCircle] = 2*self.sigma*np.cos(
        self.contactAng[self.elemCircle])/self.Rarray[self.elemCircle]
    self.PistonPcRec[self.elemTriangle] = __computePc__(
        self, self.elemTriangle, self.Fd_Tr)
    self.PistonPcRec[self.elemSquare] = __computePc__(
        self, self.elemSquare, self.Fd_Sq)
    

def __func(self, i):
    '''returns the minimum receding Pc for pistonlike displacement'''
    arr = self.elem[i].neighbours
    if arr.any():
        return self.PistonPcRec[arr[(arr>0) & self.hasNWFluid[arr]]].min()
    return 0

def __update_PcD_ToFill__(self, arr) -> None:
    minNeiPc = np.array([*map(lambda ar: __func(self, ar), arr)])
    entryPc = np.maximum(0.999*minNeiPc+0.001*self.PistonPcRec[
        arr], self.PistonPcRec[arr])
    
    ''' elements to be removed before updating PcD '''
    cond1 = (entryPc != self.PcD[arr]) & (~self.NinElemList[arr])
    [self.ElemToFill.discard(i) for i in arr[cond1]]
    self.NinElemList[arr[cond1]] = True

    ''' updating elements with new PcD '''
    cond2 = (entryPc != self.PcD[arr])
    self.PcD[arr[cond2]] = entryPc[cond2]

    ''' updating the ToFill elements '''
    cond3 = (self.NinElemList[arr])
    self.update += arr[cond3].size
    self.ElemToFill.update(arr[cond3])
    self.NinElemList[arr[cond3]] = False
    
        

def __CondTP_Drainage__(self):
    # to suppress the FutureWarning and SettingWithCopyWarning respectively
    warnings.simplefilter(action='ignore', category=FutureWarning)
    pd.options.mode.chained_assignment = None

    arrr = (self.fluid==1)
    arrr[[0, -1]] = False
    if not np.any(arrr):
        return
      
    arrrT = arrr & self.isTriangle
    arrrS = arrr & self.isSquare
    arrrC = arrr & self.isCircle

    Pc = np.full(self.totElements, self.capPresMax)
    if np.any(arrrT):
        do.createFilms(self, arrrT, self.PcD, 3)
        conAngPT, apexDistPT = do.cornerApex(self, arrrT, Pc, 
			self.contactAng.copy(), self.m_cornExists, 3)
       
        cornA, cornG = do.calcAreaW(self, arrrT, conAngPT, apexDistPT, 3)
        arrT = np.flatnonzero(arrrT)
        condlist = (cornA < self._cornArea[arrT])
        self._cornArea[arrT[condlist]] = cornA[condlist]

        condlist = (cornG < self._cornCond[arrT])
        self._cornCond[arrT[condlist]] = cornG[condlist]
    
    if np.any(arrrS):
        do.createFilms(self, arrrS, self.PcD, 4)
        conAngPS, apexDistPS = do.cornerApex(self, arrrS, Pc, 
			self.contactAng.copy(), self.m_cornExists, 4)
            
        cornA, cornG = do.calcAreaW(self, arrrS, conAngPS, apexDistPS, 4)
        arrS = np.flatnonzero(arrrS)
        condlist = (cornA < self._cornArea[arrS])
        self._cornArea[arrS[condlist]] = cornA[condlist]

        condlist = (cornG < self._cornCond[arrS])
        self._cornCond[arrS[condlist]] = cornG[condlist]
    
    if np.any(arrrC):
        arrrC = np.flatnonzero(arrrC)
        self._cornArea[arrrC] = 0.0
        self._cornCond[arrrC] = 0.0

    self._centerArea[arrr] = self.areaSPhase[arrr] - self._cornArea[arrr]
    self._centerCond[arrr] = self._centerArea[arrr]/self.areaSPhase[arrr]*self.gnwSPhase[arrr]

    
def __fileName__(self):
    result_dir = "./results_csv/"
    os.makedirs(os.path.dirname(result_dir), exist_ok=True)
    if not hasattr(self, '_num'):
        self._num = 1
        while True:         
            file_name = os.path.join(
                result_dir, "Flowmodel_"+self.title+"_Drainage_cycle"+str(
                    self.cycle)+"_"+str(self._num)+".csv")
            if os.path.isfile(file_name): self._num += 1
            else:
                break
        self.file_name = file_name
    else:
        self.file_name = os.path.join(
            result_dir, "Flowmodel_"+self.title+"_Drainage_cycle"+str(self.cycle)+\
                "_"+str(self._num)+".csv")
    

def __writeHeadersD__(self):
    self.resultD_str="======================================================================\n"
    self.resultD_str+="# Fluid properties:\nsigma (mN/m)  \tmu_w (cP)  \tmu_nw (cP)\n"
    self.resultD_str+="# \t%.6g\t\t%.6g\t\t%.6g" % (
        self.sigma*1000, self.muw*1000, self.munw*1000, )
    self.resultD_str+="\n# calcBox: \t %.6g \t %.6g" % (
        self.calcBox[0], self.calcBox[1], )
    self.resultD_str+="\n# Wettability:"
    self.resultD_str+="\n# model \tmintheta \tmaxtheta \tdelta \teta \tdistmodel"
    self.resultD_str+="\n# %.6g\t\t%.6g\t\t%.6g\t\t%.6g\t\t%.6g" % (
        self.wettClass, round(self.minthetai*180/np.pi,3), round(self.maxthetai*180/np.pi,3), self.delta, self.eta,) 
    self.resultD_str+=self.distModel
    self.resultD_str+="\nmintheta \tmaxtheta \tmean  \tstd"
    self.resultD_str+="\n# %3.6g\t\t%3.6g\t\t%3.6g\t\t%3.6g" % (
        round(self.contactAng.min()*180/np.pi,3), round(self.contactAng.max()*180/np.pi,3), 
        round(self.contactAng.mean()*180/np.pi,3), round(self.contactAng.std()*180/np.pi,3))
    
    self.resultD_str+="\nPorosity:  %3.6g" % (self.porosity)
    self.resultD_str+="\nMaximum pore connection:  %3.6g" % (self.maxPoreCon)
    self.resultD_str+="\nAverage pore-to-pore distance:  %3.6g" % (self.avgP2Pdist)
    self.resultD_str+="\nMean pore radius:  %3.6g" % (self.Rarray[self.poreList].mean())
    self.resultD_str+="\nAbsolute permeability:  %3.6g" % (self.absPerm)
    
    self.resultD_str+="\n======================================================================"
    self.resultD_str+="\n# Sw\t qW(m3/s)\t krw\t qNW(m3/s)\t krnw\t Pc\t Invasions"


def __writeTrappedData__(self):
    filename = os.path.join(
        "./results_csv/Flowmodel_{}_Drainage_{}_trappedDist.csv".format(
            self.title, self._num))
    data = [*zip(self.Rarray, self.volarray, self.fluid, self.trappedW, self.trappedNW)]
    np.savetxt(filename, data, delimiter=',', header='rad, volume, fluid, trappedW, trappedNW')


