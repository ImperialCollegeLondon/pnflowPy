import os
import warnings
from time import time

import numpy as np
import pandas as pd
from sortedcontainers import SortedList
from functools import partial

import utilities as do


class TwoPhaseImbibition:    
    def __init__(self, obj, writeData=False, writeTrappedData=False):
        obj.writeData = writeData
        obj.writeTrappedData = writeTrappedData

def initialize(self):
    self.porebodyPc = np.zeros(self.totElements)
    self.snapoffPc = np.zeros(self.totElements)
    self.PistonPcAdv = np.zeros(self.totElements)
    self.pistonPc_PHing = np.zeros(self.nPores+2)
    self.fluid[[-1, 0]] = 0, 1  
    self.fillmech = np.full(self.totElements, -5)
    self.capPresMin = self.maxPc
    
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
    __computePc__(self, self.maxPc, self.elementLists.copy(), False, True)
            
    self._areaWP = self._cornArea.copy()
    self._areaNWP = self._centerArea.copy()
    self._condWP = self._cornCond.copy()
    self._condNWP = self._centerCond.copy()
    
    #self.trappedPc = self.trappedNW_Pc.view()
    self.areaWPhase = self._areaWP.view()
    self.areaNWPhase = self._areaNWP.view()
    self.gWPhase = self._condWP.view()
    self.gNWPhase = self._condNWP.view()
    self.cornerArea = self._cornArea.view()
    self.centerArea = self._centerArea.view()
    self.cornerCond = self._cornCond.view()
    self.centerCond = self._centerCond.view()

def imbibition(self):
    start = time()
    print('----------------------------------------------------------------------------------')
    print('---------------------------------Two Phase Imbibition Cycle {}---------------------'.format(self.cycle))
    
    if self.writeData:
        self.__fileName__()
        self.__writeHeadersI__()
    else: 
        self.resultI_str = ""
        self.totNumFill = 0
        self.resultI_str = do.writeResult(self, self.resultI_str, self.capPresMax)
        
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
                self.resultI_str = do.writeResult(self, self.resultI_str, self.capPresMin)
                
            break

        self.PcTarget = max(self.minPc+1e-7, self.PcTarget-(
            self.minDeltaPc+abs(
                self.PcTarget)*self.deltaPcFraction+1e-16))
        self.SwTarget = min(self.finalSat+1e-15, round((
            self.satW+self.dSw*0.75)/self.dSw)*self.dSw)
        
    if self.writeData:
        with open(self.file_name, 'a') as fQ:
            fQ.write(self.resultI_str)
        if self.writeTrappedData:
            self.__writeTrappedData__()

    print("Number of trapped elements: W: {}  NW:{}".format(
        self.trappedW.sum(), self.trappedNW.sum()))
    print('No of W clusters: {}, No of NW clusters: {}'.format(
        np.count_nonzero(self.clusterW.size),
        np.count_nonzero(self.clusterNW.size)))
    #self.is_oil_inj = True
    #self.do.__finitCornerApex__(self.capPresMin)
    print('Time spent for the imbibition process: ', time() - start)
    print('===========================================================\n\n')    


def __PImbibition__(self):
    self.totNumFill = 0
    while (self.PcTarget-1.0e-32 < self.capPresMin) & (
            self.satW <= self.SwTarget):
        self.oldSatW = self.satW
        self.invInsideBox = 0
        self.cnt = 0
        try:
            while (self.invInsideBox < self.fillTarget) & (
                len(self.ElemToFill) != 0) & (
                    self.PcI[self.ElemToFill[0]] >= self.PcTarget):
                try:
                    assert not self.fillTillNWDisconnected
                    popUpdateWaterInj(self)
                except AssertionError:
                    try:
                        assert (self.clusterNW.members[0][self.conTToIn].any() and 
                                self.clusterNW.members[0][self.conTToOutletBdr].any())
                        popUpdateWaterInj(self)
                    except AssertionError:
                        self.filling = False
                        self.PcTarget = self.capPresMin
                        break

            assert (self.PcI[self.ElemToFill[0]] < self.PcTarget) & (
                    self.capPresMin > self.PcTarget)
            self.capPresMin = self.PcTarget
        except IndexError:
            self.capPresMin = min(self.capPresMin, self.PcTarget)
        except AssertionError:
            pass

        __CondTPImbibition__(self)
        self.satW = do.Saturation(self, self.areaWPhase, self.areaSPhase)
        self.totNumFill += self.cnt
        try:
            assert self.PcI[self.ElemToFill[0]] >= self.PcTarget
            assert self.filling
        except (AssertionError, IndexError):
            break
    try:
        assert (self.PcI[self.ElemToFill[0]] < self.PcTarget) & (
            self.capPresMin > self.PcTarget)
        self.capPresMin = self.PcTarget
    except AssertionError:
        self.PcTarget = self.capPresMin
    except IndexError:
        pass

    __CondTPImbibition__(self)
    self.satW = do.Saturation(self, self.areaWPhase, self.areaSPhase)
    do.computePerm(self, self.capPresMin)
    self.resultI_str = do.writeResult(self, self.resultI_str, self.capPresMin)
    

def fillWithWater(self, k):
    self.fluid[k] = 0
    try:
        assert self.hasWFluid[k]
    except AssertionError:
        try:
            neigh = self.elem[k].neighbours[self.elem[k].neighbours>0]
            self.hasWFluid[k] = True
            neighW = neigh[self.hasWFluid[neigh]]
            ids = self.clusterW_ID[neighW]
            ii = ids.min()
                
            ''' newly filled takes the properties of already filled neighbour '''
            self.clusterW_ID[k] = ii
            self.clusterW.members[ii,k] = True
            self.connW[k] = self.clusterW[ii].connected
            ids = ids[ids!=ii]                
            assert ids.size>0

            ''' need to coalesce '''
            mem = self.elementListS[self.clusterW.members[ids].any(axis=0)]
            self.clusterW.members[ii][mem] = True
            self.clusterW.members[ids] = False
            self.clusterW.availableID.update(ids)
            self.clusterW_ID[mem] = ii
        except (AssertionError, ValueError):
            pass

def unfillWithOil(self, k, Pc, updateCluster=False, updateConnectivity=False, 
                    updatePcClustConToInlet=True, updatePc=True):
    self.hasNWFluid[k] = False
    kk = self.clusterNW_ID[k]
    self.clusterNW_ID[k] = -5
    self.clusterNW.members[kk,k] = False        
    neigh = self.elem[k].neighbours[self.elem[k].neighbours>0]
    neigh = neigh[self.hasNWFluid[neigh]]
    do.check_Trapping_Clustering(
        self, neigh.copy(), self.hasNWFluid.copy(), 1, Pc, 
        updateCluster, updateConnectivity, updatePcClustConToInlet)
    try:
        assert updatePc
        neighb = neigh[~self.trappedNW[neigh]]
        __computePc__(self, self.capPresMin, neighb)
    except AssertionError:
        pass

def popUpdateWaterInj(self):
    k = self.ElemToFill.pop(0)
    capPres = self.PcI[k]
    self.capPresMin = np.min([self.capPresMin, capPres])

    try:
        assert not self.trappedNW[k]
        fillWithWater(self, k)
        unfillWithOil(self, k, self.capPresMin, True)
        self.fillmech[k] = 1*(self.PistonPcAdv[k]==capPres)+2*(
            self.porebodyPc[k]==capPres)+3*(self.snapoffPc[k]==capPres)
        self.cnt += 1
        self.invInsideBox += self.isinsideBox[k]
    except AssertionError:
        pass


def __CondTPImbibition__(self, arrr=None, Pc=None, updateArea=True):
    # to suppress the FutureWarning and SettingWithCopyWarning respectively
    warnings.simplefilter(action='ignore', category=FutureWarning)
    pd.options.mode.chained_assignment = None

    try:
        assert arrr is None
        arrrS = np.ones(self.elemSquare.size, dtype='bool')
        arrrT = np.ones(self.elemTriangle.size, dtype='bool')
        arrrC = np.ones(self.elemCircle.size, dtype='bool')
        Pc = np.full(self.totElements, self.capPresMin)
    except AssertionError:
        arrrS = arrr[self.isSquare]
        arrrT = arrr[self.isTriangle]
        arrrC = arrr[self.isCircle]

    try:
        curConAng = self.contactAng.copy()
        apexDist = np.empty_like(self.hingAngSq.T)
        conAngPS, apexDistPS = do.cornerApex(
            self, self.elemSquare, arrrS, self.halfAnglesSq[:, np.newaxis], Pc[self.elemSquare],
            curConAng, self.cornExistsSq.T, self.initOrMaxPcHistSq.T,
            self.initOrMinApexDistHistSq.T, self.advPcSq.T,
            self.recPcSq.T, apexDist, self.initedApexDistSq.T)
        
        cornA, cornG = do.calcAreaW(
            self, arrrS, self.halfAnglesSq, conAngPS, self.cornExistsSq, apexDistPS)
        elemSquare = self.elemSquare[arrrS]
        cond = (cornA<self.areaSPhase[elemSquare])
        self._cornArea[elemSquare[cond]] = cornA[cond]
        self._cornCond[elemSquare[cond]] = cornG[cond]
    except AssertionError:
        pass

    try:
        curConAng = self.contactAng.copy()
        apexDist = np.empty_like(self.hingAngTr.T)
        conAngPT, apexDistPT = do.cornerApex(
            self, self.elemTriangle, arrrT, self.halfAnglesTr.T, Pc[self.elemTriangle],
            curConAng, self.cornExistsTr.T, self.initOrMaxPcHistTr.T,
            self.initOrMinApexDistHistTr.T, self.advPcTr.T,
            self.recPcTr.T, apexDist, self.initedApexDistTr.T)
        
        cornA, cornG = do.calcAreaW(
            self, arrrT, self.halfAnglesTr, conAngPT, self.cornExistsTr, apexDistPT)
        elemTriangle = self.elemTriangle[arrrT]
        cond = (cornA<self.areaSPhase[elemTriangle])
        self._cornArea[elemTriangle[cond]] = cornA[cond]
        self._cornCond[elemTriangle[cond]] = cornG[cond]
    except AssertionError:
        pass

    try:
        assert arrrC.size>0
        arrrC = self.elemCircle[arrrC]
        self._cornArea[arrrC] = 0.0
        self._cornCond[arrrC] = 0.0
    except  AssertionError:
        pass

    self._centerArea = self.areaSPhase - self._cornArea
    self._centerCond = np.where(
        self.areaSPhase != 0.0, 
        self._centerArea/self.areaSPhase*self.gnwSPhase, 0.0)
    try:
        assert updateArea      
        __updateAreaCond__(self)
    except AssertionError:
        pass


def __updateAreaCond__(self):
    arrr = (~self.trappedNW)

    try:
        cond2 = arrr & (self.fluid == 0)
        assert np.any(cond2)
        self._areaWP[cond2] = self.areaSPhase[cond2]
        self._areaNWP[cond2] = 0.0
        self._condWP[cond2] = self.gwSPhase[cond2]
        self._condNWP[cond2] = 0.0
    except AssertionError:
        pass

    try:
        cond1 = arrr & (self.fluid==1) & (self.Garray<=self.bndG2)
        assert np.any(cond1)
        self._areaWP[cond1] = np.clip(self._cornArea[cond1], 0.0, self.areaSPhase[cond1])
        self._areaNWP[cond1] = np.clip(self._centerArea[cond1], 0.0, self.areaSPhase[cond1])
        self._condWP[cond1] = np.clip(self._cornCond[cond1], 0.0, self.gwSPhase[cond1])
        self._condNWP[cond1] = np.clip(self._centerCond[cond1], 0.0, self.gnwSPhase[cond1])
    except AssertionError:
        pass

    try:
        cond3 = arrr & (self.fluid==1) & (self.Garray>self.bndG2)
        assert np.any(cond3)
        self._areaWP[cond3] = 0.0
        self._areaNWP[cond3] = self.areaSPhase[cond3]
        self._condWP[cond3] = 0.0
        self._condNWP[cond3] = self.gnwSPhase[cond3]
    except AssertionError:
        pass
    

def __computePistonPc__(self):
    conda = (self.fluid == 0)
    condb = (self.fluid == 1) & (self.Garray < self.bndG2)  #polygons filled with w
    condc = (self.fluid == 1) & (self.Garray >= self.bndG2) #circles filled with nw
    condac = (conda | condc)

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

    conde = np.zeros(self.totElements, dtype='bool')
    conde[self.elemTriangle] = condb[self.elemTriangle] & (~condd[self.elemTriangle]) & (
        self.thetaAdvAng[self.elemTriangle] <= np.pi/2+self.halfAnglesTr[:, 0])
    self.PistonPcAdv[conde] = 2.0*self.sigma*self.cosThetaAdvAng[conde]/self.Rarray[conde]
    
    condf = condb & (~condd) & (~conde) 
    self.PistonPcAdv[condf] = 2.0*self.sigma*self.cosThetaAdvAng[condf]/self.Rarray[condf]
    

def __PistonPcHing__(self, arrr):
    ''' compute entry capillary pressures for piston displacement '''
    arrrS = arrr[self.elemSquare]
    arrrT = arrr[self.elemTriangle]
    
    try:
        assert np.any(arrrT)
        self.PistonPcAdv[self.elemTriangle[arrrT]] = Pc_pistonHing(
            self, self.elemTriangle, arrrT, self.halfAnglesTr.T, self.cornExistsTr,
            self.initOrMaxPcHistTr, self.initOrMinApexDistHistTr, self.advPcTr,
            self.recPcTr, self.initedApexDistTr)
    except AssertionError:
        pass
    try:
        assert np.any(arrrS)
        self.PistonPcAdv[self.elemSquare[arrrS]] = Pc_pistonHing(
            self, self.elemSquare, arrrS, self.halfAnglesSq, self.cornExistsSq,
            self.initOrMaxPcHistSq, self.initOrMinApexDistHistSq, self.advPcSq,
            self.recPcSq, self.initedApexDistSq)
    except AssertionError:
        pass
    

def Pc_pistonHing(self, arr, arrr, halfAng, m_exists, m_initOrMaxPcHist,
                    m_initOrMinApexDistHist, advPc, recPc, initedApexDist):
    
    newPc = 1.1*self.sigma*2.0*self.cosThetaAdvAng[arr]/self.Rarray[arr]
    
    arrr1 = arrr.copy()
    apexDist = np.zeros(arrr.size)
    counter = 0
    while True:
        oldPc = newPc.copy()
        sumOne, sumTwo = np.zeros(arrr.size), np.zeros(arrr.size)
        sumThree, sumFour = np.zeros(arrr.size), np.zeros(arrr.size)
        for i in range(m_exists.shape[1]):
            cond1 = arrr1 & m_exists[:, i]                
            conAng, apexDist = do.cornerApex(
                self, arr, cond1, halfAng[i], oldPc, self.thetaAdvAng.copy(), m_exists[:, i], m_initOrMaxPcHist[:, i], m_initOrMinApexDistHist[:, i], advPc[:, i],
                recPc[:, i], apexDist, initedApexDist[:, i], accurat=True, overidetrapping=True)

            partus = (apexDist*np.sin(halfAng[i])*oldPc/self.sigma)

            try:
                assert (abs(partus[cond1]) <= 1.0).all()
            except AssertionError:
                partus[cond1 & (abs(partus) > 1.0)] = 0.0          

            sumOne[cond1] += (apexDist*np.cos(conAng))[cond1]
            sumTwo[cond1] += (np.pi/2-conAng-halfAng[i])[cond1]
            sumThree[cond1] += (np.arcsin(partus[cond1]))
            sumFour[cond1] += apexDist[cond1]

        a = (2*sumThree-sumTwo)
        b = ((self.cosThetaAdvAng[arr]*self.Rarray[arr]/(
            2*self.Garray[arr])) - 2*sumFour + sumOne)
        c = (-pow(self.Rarray[arr], 2)/(4*self.Garray[arr]))

        arr1 = pow(b, 2)-np.array(4*a*c)
        cond = (arr1 > 0)
        newPc[arrr1] = (self.sigma*(2*a[arrr1])/(
            (-b+np.sqrt(arr1))*cond + (-b)*(~cond))[arrr1])
        err = 2.0*abs((newPc - oldPc)/(abs(oldPc)+abs(newPc)+1.0e-3))[arrr1]
        counter += 1
        try:
            assert (err < self.EPSILON).all() or (counter > self.MAX_ITER)
            break
        except AssertionError:
            arrr1[arrr1] = (err >= self.EPSILON)

    newPc[np.isnan(newPc)] = 0.0
    return newPc[arrr]


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
    return (-round(PcI[k], 9), k<=nPores, -k)

def __computePc__(self, Pc, arr, update=True, trapping=True):
    entryPc = self.PistonPcAdv.copy()
    maxNeiPistonPrs = np.zeros(self.totElements)
    _arr = arr[self.hasNWFluid[arr]] # & ~self.trappedNW[arr]]  # elements filled with nw
    arrP = _arr[(_arr <= self.nPores)]   #pores filled with nw
    arrT = _arr[(_arr > self.nPores)]      #throats filled with nw
    _arrT = arrT-self.nPores

    ''' identify pores where porebody filling could occur '''
    arr1 = np.sum(self.fluid[self.PTConnections[arrP]]==0, axis=1,
                    where=self.PTValid[arrP])
    cond1 = (arr1 > 0) & (self.thetaAdvAng[arrP] < np.pi/2.0) #pores for porebody filling
    __porebodyFilling__(self, arrP[cond1])
    entryPc[arrP[cond1]] = self.porebodyPc[arrP[cond1]]
    
    ''' update the piston-like entry Pc '''      
    maxNeiPistonPrs[arrP] = np.max(
        self.PistonPcAdv[self.PTConnections[arrP]], axis=1, initial=0.0,
        where=(self.PTValid[arrP]&(self.fluid[self.PTConnections[arrP]]==0)))
    maxNeiPistonPrs[arrT] = np.max(
        self.PistonPcAdv[self.TPConnections[_arrT]], axis=1, initial=0.0,
        where=((self.fluid[self.TPConnections[_arrT]]==0)))
    condb = (maxNeiPistonPrs > 0.0)
    entryPc[condb] = np.minimum(0.999*maxNeiPistonPrs[
        condb]+0.001*entryPc[condb], entryPc[condb])
    
    ''' Snap-off filling '''
    __updateSnapoffPc__(self, Pc)
    conda = (maxNeiPistonPrs > 0.0) & (entryPc>self.snapoffPc)
    entryPc[~conda&(self.Garray<self.bndG2)] = self.snapoffPc[~conda&(self.Garray<self.bndG2)]

    ''' update the toFill list '''
    try:
        assert update
        ''' update PcI '''
        diff = (self.PcI[arr] != entryPc[arr])
        [self.ElemToFill.discard(i) for i in arr[diff]]
        self.PcI[arr[diff]] = entryPc[arr[diff]]

        ''' add to the toFill list '''
        arrr = np.zeros(self.totElements, dtype=bool)
        arrr[_arr[self.hasWFluid[_arr]]] = True
        _arr = _arr[~self.hasWFluid[_arr]]
        arrP = _arr[_arr <= self.nPores]
        arrT = _arr[_arr > self.nPores]
        _arrT = arrT-self.nPores
        arrrP = (self.hasWFluid[self.PTConnections[arrP]]&self.PTValid[arrP]).any(axis=1)
        arrrT = (self.hasWFluid[self.TPConnections[_arrT]]|
                    (self.TPConnections[_arrT]==-1)).any(axis=1)
        arrr[arrP] = arrrP
        arrr[arrT] = arrrT
        arrr[self.ElemToFill] = False
        self.ElemToFill.update(self.elementListS[arrr])
    except AssertionError:
        self.PcI[arr] = entryPc[arr]
        _arr = __func4(self, _arr, trapping)
        self.ElemToFill.update(_arr)
    

def __func4(self, arr, trapping=True):
    ''' ensures that all elements to be added to the tofill list have 
    (i) the non-wetting fluid; 
    (ii) the wetting fluid in the corners or a neighbouring element;
    (iii) the wetting fluid is not trapped.'''

    arrr = np.zeros(self.totElements, dtype=bool)
    arrr[arr[self.hasWFluid[arr]]] = True
    arr = arr[~self.hasWFluid[arr]]
    arrP = arr[arr<=self.nPores]
    arrPT = self.PTConnections[arrP]
    arrT = arr[arr>self.nPores]
    arrTP = self.TPConnections[arrT-self.nPores]

    try:
        assert trapping
        conP = (self.hasWFluid[arrPT])&(~self.trappedW[arrPT])&self.PTValid[arrP]
        conT = (arrTP==-1) | ((self.hasWFluid[arrTP])&(~self.trappedW[arrTP])&(arrTP>0))
    except AssertionError:
        conP = (self.hasWFluid[arrPT])&self.PTValid[arrP]
        conT = (arrTP==-1) | ((self.hasWFluid[arrTP])&(arrTP>0))
    
    arrr[arrP[conP.any(axis=1)]] = True
    arrr[arrT[conT.any(axis=1)]] = True
    return self.elementListS[arrr]


def __porebodyFilling__(self, ind):
    try:
        assert ind.size > 0
        arr = self.PTConnections[ind]
        cond = (self.fluid[arr]==1)&self.PTValid[ind]  
        arr2 = np.sort(np.where(cond, self.randNum[arr], np.nan))[:, :6]
        cond1 = (arr2!=np.nanmax(arr2, axis=1)[:,np.newaxis])&(~np.isnan(arr2))
        sumrand = np.sum(arr2, where=cond1, axis=1)*15000

        #Blunt2
        self.porebodyPc[ind] = self.sigma*(
            2*self.cosThetaAdvAng[ind]/self.Rarray[ind] - sumrand)
    except AssertionError:
        pass


def __writeHeadersI__(self):
    self.resultI_str="======================================================================\n"
    self.resultI_str+="# Fluid properties:\nsigma (mN/m)  \tmu_w (cP)  \tmu_nw (cP)\n"
    self.resultI_str+="# \t%.6g\t\t%.6g\t\t%.6g" % (
        self.sigma*1000, self.muw*1000, self.munw*1000, )
    self.resultI_str+="\n# calcBox: \t %.6g \t %.6g" % (
        self.calcBox[0], self.calcBox[1], )
    self.resultI_str+="\n# Wettability:"
    self.resultI_str+="\n# model \tmintheta \tmaxtheta \tdelta \teta \tdistmodel"
    self.resultI_str+="\n# %.6g\t\t%.6g\t\t%.6g\t\t%.6g\t\t%.6g" % (
        self.wettClass, round(self.minthetai*180/np.pi,3), round(self.maxthetai*180/np.pi,3), self.delta, self.eta,)
    self.resultI_str+=self.distModel
    self.resultI_str+="\nmintheta \tmaxtheta \tmean  \tstd"
    self.resultI_str+="\n# %3.6g\t\t%3.6g\t\t%3.6g\t\t%3.6g" % (
        round(self.contactAng.min()*180/np.pi,3), round(self.contactAng.max()*180/np.pi,3), round(self.contactAng.mean()*180/np.pi,3), round(self.contactAng.std()*180/np.pi,3))
    
    self.resultI_str+="\nPorosity:  %3.6g" % (self.porosity)
    self.resultI_str+="\nMaximum pore connection:  %3.6g" % (self.maxPoreCon)
    self.resultI_str+="\nAverage pore-to-pore distance:  %3.6g" % (self.avgP2Pdist)
    self.resultI_str+="\nMean pore radius:  %3.6g" % (self.Rarray[self.poreList].mean())
    self.resultI_str+="\nAbsolute permeability:  %3.6g" % (self.absPerm)
    
    self.resultI_str+="\n======================================================================"
    self.resultI_str+="\n# Sw\t qW(m3/s)\t krw\t qNW(m3/s)\t krnw\t Pc\t Invasions"

    self.totNumFill = 0
    self.resultI_str = do.writeResult(self, self.resultI_str, self.capPresMax)


def __fileName__(self):
    result_dir = "./results_csv/"
    os.makedirs(os.path.dirname(result_dir), exist_ok=True)
    if not hasattr(self, '_num'):
        self._num = 1
        while True:         
            file_name = os.path.join(
                result_dir, "Flowmodel_"+self.title+"_Imbibition_cycle"+str(self.cycle)+\
                    "_"+str(self._num)+".csv")
            if os.path.isfile(file_name): self._num += 1
            else:
                break
        self.file_name = file_name
    else:
        self.file_name = os.path.join(
            result_dir, "Flowmodel_"+self.title+"_Imbibition_cycle"+str(self.cycle)+\
                "_"+str(self._num)+".csv")


def __writeTrappedData__(self):
    filename = os.path.join(
        "./results_csv/Flowmodel_{}_Imbibition_cycle{}_{}_trappedDist.csv".format(
            self.title, self.cycle, self._num))
    data = [*zip(self.Rarray, self.volarray, self.fluid, self.trappedW, self.trappedNW)]
    np.savetxt(filename, data, delimiter=',', header='rad, volume, fluid, trappedW, trappedNW')
        

        
        
        
        


