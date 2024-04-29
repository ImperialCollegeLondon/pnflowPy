import os
import sys
import warnings
from math import pi
from time import time
import numpy as np
import pandas as pd
from sortedcontainers import SortedList
from utilities import Computations
from sPhase import SinglePhase

class TwoPhaseDrainage(SinglePhase):
    cycle = 0
    def __new__(cls, obj, writeData=False, writeTrappedData=False):
        obj.__class__ = TwoPhaseDrainage
        return obj
    
    def __init__(self, obj, writeData=False, writeTrappedData=False):
        self.do = Computations(self)
        
        self.fluid = np.zeros(self.totElements, dtype='int')
        self.fluid[-1] = 1   # already filled
        self.trappedW = np.zeros(self.totElements, dtype='bool')
        self.trappedNW = np.zeros(self.totElements, dtype='bool')
        self.trappedW_Pc = np.zeros(self.totElements)
        self.trappedNW_Pc = np.zeros(self.totElements)
        self.trapCluster_W = np.zeros(self.totElements, dtype='int')
        self.trapCluster_NW = np.zeros(self.totElements, dtype='int')

        self.trapped = self.trappedW
        self.trappedPc = self.trappedW_Pc
        self.trapClust = self.trapCluster_W
        self.connW = self.connected.copy()
        self.connNW = np.zeros(self.totElements, dtype='bool')

        self.contactAng, self.thetaRecAng, self.thetaAdvAng =\
            self.do.__wettabilityDistribution__()
        self.Fd_Tr = self.do.__computeFd__(self.elemTriangle, self.halfAnglesTr)
        self.Fd_Sq = self.do.__computeFd__(self.elemSquare, self.halfAnglesSq)

        self.cornExistsTr = np.zeros([self.nTriangles, 3], dtype='bool')
        self.cornExistsSq = np.zeros([self.nSquares, 4], dtype='bool')
        self.initedTr = np.zeros([self.nTriangles, 3], dtype='bool')
        self.initedSq = np.zeros([self.nSquares, 4], dtype='bool')
        self.initOrMaxPcHistTr = np.zeros([self.nTriangles, 3])
        self.initOrMaxPcHistSq = np.zeros([self.nSquares, 4])
        self.initOrMinApexDistHistTr = np.zeros([self.nTriangles, 3])
        self.initOrMinApexDistHistSq = np.zeros([self.nSquares, 4])
        self.initedApexDistTr = np.zeros([self.nTriangles, 3])
        self.initedApexDistSq = np.zeros([self.nSquares, 4])
        self.advPcTr = np.zeros([self.nTriangles, 3])
        self.advPcSq = np.zeros([self.nSquares, 4])
        self.recPcTr = np.zeros([self.nTriangles, 3])
        self.recPcSq = np.zeros([self.nSquares, 4])
        self.hingAngTr = np.zeros([self.nTriangles, 3])
        self.hingAngSq = np.zeros([self.nSquares, 4])
    
        self.do.__initCornerApex__()
        self.__computePistonPc__()
        self.PcD = self.PistonPcRec.copy()
        self.PcI = np.zeros(self.totElements)
        self.centreEPOilInj = np.zeros(self.totElements)
        self.centreEPOilInj[self.elementLists] = 2*self.sigma*np.cos(
            self.thetaRecAng[self.elementLists])/self.Rarray[self.elementLists]
        
        self.ElemToFill = SortedList(key=lambda i: self.LookupList(i))
        ElemToFill = self.nPores+self.conTToIn
        self.ElemToFill.update(ElemToFill)
        self.NinElemList = np.ones(self.totElements, dtype='bool')
        self.NinElemList[ElemToFill] = False

        self._cornArea = self.AreaSPhase.copy()
        self._centerArea = np.zeros(self.totElements) 
        self._cornCond = self.gwSPhase.copy()
        self._centerCond = np.zeros(self.totElements)

        self.capPresMax = 0
        self.capPresMin = 0
        self.is_oil_inj = True
        self.cycle += 1
        self.writeData = writeData
        self.writeTrappedData = writeTrappedData

        self.qW, self.qNW = 0.0, 0.0
        self.krw, self.krnw = 0.0, 0.0
        self.totNumFill = 0
       

    @property
    def AreaWPhase(self):
        return self._cornArea
    
    @property
    def AreaNWPhase(self):
        return self._centerArea
    
    @property
    def gWPhase(self):
        return self._cornCond
    
    @property
    def gNWPhase(self):
        return self._centerCond
    
    def LookupList(self, k):
        return (self.PcD[k], k > self.nPores, -k)
    
    def drainage(self):
        start = time()
        print('---------------------------------------------------------------------------')
        print('-------------------------Two Phase Drainage Cycle {}------------------------'.format(self.cycle))

        if self.writeData:
            self.__fileName__()
            self.__writeHeadersD__()
        else: self.resultD_str = ""

        self.SwTarget = max(self.finalSat, self.satW-self.dSw*0.5)
        self.PcTarget = min(self.maxPc, self.capPresMax+(
            self.minDeltaPc+abs(
             self.capPresMax)*self.deltaPcFraction)*0.1)
        self.oldPcTarget = 0
        self.resultD_str = self.do.writeResult(self.resultD_str, self.capPresMin)

        while self.filling:
            self.oldSatW = self.satW
            self.__PDrainage__()
            
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
                    self.__CondTP_Drainage__()
                    self.satW = self.do.Saturation(self.AreaWPhase, self.AreaSPhase)
                    self.do.computePerm()
                    self.resultD_str = self.do.writeResult(self.resultD_str, self.capPresMax)

                    self.PcTarget = min(self.maxPc-1e-7, self.PcTarget+(
                        self.minDeltaPc+abs(self.PcTarget)*self.deltaPcFraction))
                    if self.capPresMax == self.PcTarget: break
                    else: self.capPresMax = self.PcTarget
                   
                break

        if self.writeData:
            with open(self.file_name, 'a') as fQ:
                fQ.write(self.resultD_str)
            if self.writeTrappedData:
                self.__writeTrappedData__()

        self.maxPc = self.capPresMax
        self.rpd = self.sigma/self.maxPc
        print("Number of trapped elements: W: {}  NW:{}".format(
            self.trappedW.sum(), self.trappedNW.sum()))
        print(self.rpd, self.sigma, self.maxPc)
        print(len(self.ElemToFill), self.PcD[self.ElemToFill[:10]])
        self.is_oil_inj = False
        self.do.__finitCornerApex__(self.capPresMax)
        print('Time spent for the drainage process: ', time() - start)        
        print('==========================================================\n\n')
        #from IPython import embed; embed()


    def popUpdateOilInj(self):
        k = self.ElemToFill.pop(0)
        capPres = self.PcD[k]
        self.capPresMax = np.max([self.capPresMax, capPres])

        try:
            assert not self.do.isTrapped(k, 0, self.capPresMax)
            self.fluid[k] = 1
            self.PistonPcRec[k] = self.centreEPOilInj[k]
            arr = self.elem[k].neighbours
            arr = arr[(self.fluid[arr] == 0) & ~(self.trappedW[arr]) & (arr>0)]
            [*map(lambda i: self.do.isTrapped(i, 0, self.capPresMax), arr)]

            self.cnt += 1
            self.invInsideBox += self.isinsideBox[k]
            self.__update_PcD_ToFill__(arr)
        except AssertionError:
            pass
    

    def __PDrainage__(self):
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        self.totNumFill = 0
        self.fillTarget = max(self.m_minNumFillings, int(
            self.m_initStepSize*(self.totElements)*(
             self.SwTarget-self.satW)))
        self.invInsideBox = 0

        while (self.PcTarget+1.0e-32 > self.capPresMax) & (
                self.satW > self.SwTarget):
            self.oldSatW = self.satW
            self.invInsideBox = 0
            self.cnt = 0
            try:
                while (self.invInsideBox < self.fillTarget) & (
                    len(self.ElemToFill) != 0) & (
                        self.PcD[self.ElemToFill[0]] <= self.PcTarget):
                    self.popUpdateOilInj()
            except IndexError:
                break

            try:
                assert (self.PcD[self.ElemToFill[0]] > self.PcTarget) & (
                        self.capPresMax < self.PcTarget)
                self.capPresMax = self.PcTarget
            except AssertionError:
                pass
            
            self.__CondTP_Drainage__()
            self.satW = self.do.Saturation(self.AreaWPhase, self.AreaSPhase)
            self.totNumFill += self.cnt
            try:
                self.fillTarget = max(self.m_minNumFillings, int(min(
                    self.fillTarget*self.m_maxFillIncrease,
                    self.m_extrapCutBack*(self.invInsideBox / (
                        self.satW-self.oldSatW))*(self.SwTarget-self.satW))))
            except OverflowError:
                pass
                
            try:
                assert self.PcD[self.ElemToFill[0]] <= self.PcTarget
            except AssertionError:
                break

        try:
            assert (self.PcD[self.ElemToFill[0]] > self.PcTarget)
            self.capPresMax = self.PcTarget
        except (AssertionError, IndexError):
            self.PcTarget = self.capPresMax
        
        self.__CondTP_Drainage__()
        self.satW = self.do.Saturation(self.AreaWPhase, self.AreaSPhase)
        self.do.computePerm()
        self.resultD_str = self.do.writeResult(self.resultD_str, self.capPresMax)

    
    def __computePc__(self, arrr, Fd):
        Pc = self.sigma*(1+2*np.sqrt(pi*self.Garray[arrr]))*np.cos(
            self.contactAng[arrr])*Fd/self.Rarray[arrr]
        return Pc
    
    def __computePistonPc__(self) -> None:
        self.PistonPcRec = np.zeros(self.totElements)
        self.PistonPcRec[self.elemCircle] = 2*self.sigma*np.cos(
            self.contactAng[self.elemCircle])/self.Rarray[self.elemCircle]
        self.PistonPcRec[self.elemTriangle] = self.__computePc__(
            self.elemTriangle, self.Fd_Tr)
        self.PistonPcRec[self.elemSquare] = self.__computePc__(
            self.elemSquare, self.Fd_Sq)
        
    
    def __func(self, i):
        try:
            arr = self.elem[i].neighbours
            return self.PistonPcRec[arr[(arr > 0) & (self.fluid[arr] == 1)]].min()
        except ValueError:
            return 0

    
    def __func3(self, i):
        try:
            self.ElemToFill.remove(i)
            self.NinElemList[i] = True
        except ValueError:
            pass

    def __update_PcD_ToFill__(self, arr) -> None:
        try:
            minNeiPc = np.array([*map(lambda ar: self.__func(ar), arr)])
            entryPc = np.maximum(0.999*minNeiPc+0.001*self.PistonPcRec[
                arr], self.PistonPcRec[arr])
            
            # elements to be removed before updating PcD
            cond1 = (entryPc != self.PcD[arr]) & (~self.NinElemList[arr])
            [*map(lambda i: self.__func3(i), arr[cond1])]

            # updating elements with new PcD
            cond2 = (entryPc != self.PcD[arr])
            self.PcD[arr[cond2]] = entryPc[cond2]

            # updating the ToFill elements
            cond3 = (self.NinElemList[arr])
            self.ElemToFill.update(arr[cond3])
            self.NinElemList[arr[cond3]] = False
        except:
            pass
            

    
    def __CondTP_Drainage__(self):
        # to suppress the FutureWarning and SettingWithCopyWarning respectively
        warnings.simplefilter(action='ignore', category=FutureWarning)
        pd.options.mode.chained_assignment = None

        try:
            arrr = (self.fluid==1)&(self.AreaSPhase!=0.0)
            assert np.any(arrr)
        except AssertionError:
            return
        
        arrrS = arrr[self.elemSquare]
        arrrT = arrr[self.elemTriangle]
        arrrC = arrr[self.elemCircle]

        try:
            assert np.any(arrrT)
            Pc = self.PcD[self.elemTriangle]
            curConAng = self.contactAng.copy()
            self.do.createFilms(self.elemTriangle, arrrT, self.halfAnglesTr, Pc,
                        self.cornExistsTr, self.initedTr,
                        self.initOrMaxPcHistTr,
                        self.initOrMinApexDistHistTr, self.advPcTr,
                        self.recPcTr, self.initedApexDistTr)
            
            apexDist = np.zeros(self.hingAngTr.T.shape)
            conAngPT, apexDistPT = self.do.cornerApex(
                self.elemTriangle, arrrT, self.halfAnglesTr.T, self.capPresMax,
                curConAng, self.cornExistsTr.T, self.initOrMaxPcHistTr.T,
                self.initOrMinApexDistHistTr.T, self.advPcTr.T,
                self.recPcTr.T, apexDist, self.initedApexDistTr.T)
            
            cornA, cornG = self.do.calcAreaW(
                arrrT, self.halfAnglesTr, conAngPT, self.cornExistsTr, apexDistPT)
            
            arrrT = self.elemTriangle[arrrT]
            condlist = (cornA < self._cornArea[arrrT])
            self._cornArea[arrrT[condlist]] = cornA[condlist]

            condlist = (cornG < self._cornCond[arrrT])
            self._cornCond[arrrT[condlist]] = cornG[condlist]
        except AssertionError:
            pass

        try:
            assert np.any(arrrS)
            Pc = self.PcD[self.elemSquare]
            curConAng = self.contactAng.copy()
            self.do.createFilms(self.elemSquare, arrrS, self.halfAnglesSq,
                           Pc, self.cornExistsSq, self.initedSq, self.initOrMaxPcHistSq,
                           self.initOrMinApexDistHistSq, self.advPcSq,
                           self.recPcSq, self.initedApexDistSq)

            apexDist = np.zeros(self.hingAngSq.T.shape)
            conAngPS, apexDistPS = self.do.cornerApex(
                self.elemSquare, arrrS, self.halfAnglesSq[:, np.newaxis], self.capPresMax,
                curConAng, self.cornExistsSq.T, self.initOrMaxPcHistSq.T,
                self.initOrMinApexDistHistSq.T, self.advPcSq.T,
                self.recPcSq.T, apexDist, self.initedApexDistSq.T)

            cornA, cornG = self.do.calcAreaW(
                arrrS, self.halfAnglesSq, conAngPS, self.cornExistsSq, apexDistPS)
            
            arrrS = self.elemSquare[arrrS]
            condlist = (cornA < self._cornArea[arrrS])
            self._cornArea[arrrS[condlist]] = cornA[condlist]

            condlist = (cornG < self._cornCond[arrrS])
            self._cornCond[arrrS[condlist]] = cornG[condlist]
        except AssertionError:
            pass
        try:
            assert np.any(arrrC)
            arrrC = self.elemCircle[arrrC]
            self._cornArea[arrrC] = 0.0
            self._cornCond[arrrC] = 0.0
        except  AssertionError:
            pass
        
        self._centerArea[arrr] = self.AreaSPhase[arrr] - self._cornArea[arrr]
        self._centerCond[arrr] = self._centerArea[arrr]/self.AreaSPhase[arrr]*self.gnwSPhase[arrr]


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


