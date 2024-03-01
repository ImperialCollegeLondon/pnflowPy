#import os
import numpy as np
from sortedcontainers import SortedList
#import warnings
#from itertools import chain

from tPhaseD import TwoPhaseDrainage
from tPhaseImb import TwoPhaseImbibition

#import sys
#sys.setrecursionlimit(2000)


class SecDrainage(TwoPhaseDrainage):
    def __new__(cls, obj, writeData=False, writeTrappedData=True):
        obj.__class__ = SecDrainage
        return obj
    
    def __init__(self, obj, writeData=False, writeTrappedData=True):
        self.fluid[[-1, 0]] = 1, 0
        
        self.capPresMax = self.capPresMin
        self.is_oil_inj = True
        self.contactAng, self.thetaRecAng, self.thetaAdvAng = self.prop_drainage.values()
        self.trapped = self.trappedW
        self.trappedPc = self.trappedW_Pc
        self.trapClust = self.trapCluster_W
        
        self.do.__initCornerApex__()
        self.Fd_Tr = self.do.__computeFd__(self.elemTriangle, self.halfAnglesTr)
        self.Fd_Sq = self.do.__computeFd__(self.elemSquare, self.halfAnglesSq)
        self.__computePistonPc__()
        self.centreEPOilInj[self.elementLists] = 2*self.sigma*np.cos(
           self.thetaRecAng[self.elementLists])/self.Rarray[self.elementLists]
        self.PcD[:] = self.PistonPcRec
        self.PistonPcRec[self.fluid==1] = self.centreEPOilInj[self.fluid==1]
        self.ElemToFill = SortedList(key=lambda i: self.LookupList(i))
        self.NinElemList[:] = True
        self.prevFilled = (self.fluid==1)
        self.populateToFill(self.conTToIn+self.nPores)

        self._cornArea = self._areaWP.copy()
        self._centerArea = self._areaNWP.copy()
        self._cornCond = self._condWP.copy()
        self._centerCond = self._condNWP.copy()
       
        self.cycle += 1
        self.writeData = writeData
        if self.writeData: self.__fileName__()
        self.primary = False
        self.writeTrappedData = writeTrappedData
        self.totNumFill = 0
        
    
    def popUpdateOilInj(self):
        k = self.ElemToFill.pop(0)
        capPres = self.PcD[k]
        self.capPresMax = np.max([self.capPresMax, capPres])

        try:
            assert not self.do.isTrapped(k, 0, self.capPresMax)

            self.fluid[k] = 1
            self.connNW[k] = True
            self.PistonPcRec[k] = self.centreEPOilInj[k]
            arr = self.elem[k].neighbours
            
            ar1 = arr[(self.fluid[arr]==0) & (arr>0)]
            [*map(lambda i: self.do.isTrapped(i, 0, self.capPresMax), ar1)]

            ar2 = arr[(self.fluid[arr]==1)&(arr>0)&(self.prevFilled[arr])]
            try:
                assert ar2.size > 0
                self.untrapNWElements(ar2)
            except (IndexError, AssertionError):
                pass

            self.cnt += 1
            self.invInsideBox += self.isinsideBox[k]
            self.__update_PcD_ToFill__(ar1)
    
        except (AssertionError, IndexError):
            pass


    def untrapNWElements(self, ind):
        idx = self.trapCluster_NW[ind]
        arrr = np.zeros(self.totElements, dtype='bool')
        arrr[ind[idx==0]] = True
        idx = idx[idx>0]
        while True:
            try:
                i = idx[0]
                arrr[(self.trapCluster_NW==i)] = True
                idx = idx[idx!=i]
            except IndexError:
                break
        
        self.trappedNW[arrr] = False
        self.connNW[arrr] = True
        self.trapCluster_NW[arrr] = 0
        self.trappedNW_Pc[arrr] = 0.0
        self.prevFilled[arrr] = False
        self.populateToFill(self.elementLists[arrr[1:-1]])


    def populateToFill(self, arr):
        condlist = np.zeros(self.totElements, dtype='bool')
        condlist[arr[self.fluid[arr]==0]] = True
        nPores = self.nPores

        Notdone = np.ones(self.totElements, dtype='bool')
        Notdone[condlist] = False
        Notdone[[-1,0]] = False

        arr = arr[self.fluid[arr]==1]
        arrP = arr[arr<=nPores]   #pores
        arr = list(arr[arr>nPores])  #throats
        try:
            assert arrP.size>0
            Notdone[arrP] = False
            self.prevFilled[arrP] = False
            arrT = self.PTConnections[arrP]
            condlist[arrT[(self.fluid[arrT]==0)&(arrT>0)]] = True
            arr.extend(arrT[(self.fluid[arrT]==1)&(arrT>0)])
        except AssertionError:
            pass

        while True:
            try:
                tt = arr.pop(0)
                Notdone[tt] = False
                self.prevFilled[tt] = False
                while True:
                    try:
                        tt = tt-nPores
                        arrr = np.zeros(nPores+2, dtype='bool')
                        arrr[self.TPConnections[tt]] = True
                        pp = self.poreList[(arrr&Notdone[self.poreListS])[1:-1]]
                        Notdone[pp] = False
                        self.prevFilled[pp] = False
                        condlist[pp[self.fluid[pp]==0]] = True
                        pp = pp[self.fluid[pp]==1]

                        arrr = np.zeros(self.totElements, dtype='bool')
                        arrr[self.PTConnections[pp]] = True
                        tt = self.elementLists[(arrr&Notdone)[1:-1]]
                        Notdone[tt] = False
                        self.prevFilled[tt] = False
                        condlist[tt[self.fluid[tt]==0]] = True
                        tt = tt[self.fluid[tt]==1]
                        
                        assert tt.size > 0

                    except (AssertionError, IndexError):
                        try:
                            arr = np.array(arr)
                            arr = list(arr[Notdone[arr]])
                        except IndexError:
                            arr=[]
                        break
            except IndexError:
                break
        ElemToFill = self.elementLists[condlist[1:-1]]
        self.__update_PcD_ToFill__(ElemToFill)
    

class SecImbibition(TwoPhaseImbibition):
    def __new__(cls, obj, writeData=False, writeTrappedData=True):
        obj.__class__ = SecImbibition
        return obj
    
    def __init__(self, obj, writeData=False, writeTrappedData=True):    
        self.fluid[[-1, 0]] = 0, 1  
        self.ElemToFill = SortedList(key=lambda i: self.LookupList(i))
        self.capPresMin = self.maxPc
        
        self.contactAng, self.thetaRecAng, self.thetaAdvAng = self.prop_imbibition.values()
        self.is_oil_inj = False
        self.trapped = self.trappedNW
        self.trappedPc = self.trappedNW_Pc
        self.trapClust = self.trapCluster_NW
        self.do.__initCornerApex__()
        self.__computePistonPc__()
        self.__computePc__(self.maxPc, self.elementLists, False)

        self._areaWP = self._cornArea.copy()
        self._areaNWP = self._centerArea.copy()
        self._condWP = self._cornCond.copy()
        self._condNWP = self._centerCond.copy()

        self.writeData = writeData
        if self.writeData: self.__fileName__()
        self.primary = False
        self.writeTrappedData = writeTrappedData
        


