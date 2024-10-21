import numpy as np
from sortedcontainers import SortedList

from tPhaseD import TwoPhaseDrainage
from tPhaseImb import TwoPhaseImbibition

class SecDrainage(TwoPhaseDrainage):
    def __new__(cls, obj, writeData=False, writeTrappedData=True):
        obj.__class__ = SecDrainage
        return obj
    
    def __init__(self, obj, writeData=False, writeTrappedData=True):
        self.fluid[[-1, 0]] = 1, 0
        self.capPresMax = self.capPresMin
        self.is_oil_inj = True
        self.contactAng, self.thetaRecAng, self.thetaAdvAng = self.prop_drainage.values()

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
        self.populateToFill(self.conTToIn.copy())

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
            assert not self.trappedW[k]
            self.fluid[k] = 1
            self.hasNWFluid[k] = True
            self.connNW[k] = True
            self.clusterNW_ID[k] = 0
            self.clusterNW.members[0, k] = True
            self.PistonPcRec[k] = self.centreEPOilInj[k]
            arr = self.elem[k].neighbours[self.elem[k].neighbours>0]
            arr0 = arr[(self.fluid[arr]==0) & (~self.trappedW[arr])]
            arr1 = arr[self.hasNWFluid[arr]] #&(self.clusterNW_ID[arr]>0)]
            try:
                assert self.isCircle[k]
                kk = self.clusterW_ID[k]
                self.clusterW_ID[k] = -5
                self.clusterW.members[kk,k] = False
                self.connW[k] = False
                self.hasWFluid[k] = False
                self.do.check_Trapping_Clustering(
                    arr0.copy(), self.hasWFluid.copy(), 0, self.capPresMax, True)
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
                self.populateToFill(mem)
            except AssertionError:
                self.__update_PcD_ToFill__(arr0)
            
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
            temp[self.PTConnections[arr[arr<=self.nPores]]] = True
            temp[self.TPConnections[arr[arr>self.nPores]-self.nPores]] = True
            temp[done] = False
            elemToFill[temp & fluid0] = True
            arr = self.elementListS[(temp & fluid1)]
            if not any(arr):
                break            
            done[temp] = True

        self.__update_PcD_ToFill__(self.elementListS[elemToFill])


    

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

        self.do.__initCornerApex__()
        self.__computePistonPc__()
        self.__computePc__(self.maxPc, self.elementLists, update=False)

        self._areaWP = self._cornArea.copy()
        self._areaNWP = self._centerArea.copy()
        self._condWP = self._cornCond.copy()
        self._condNWP = self._centerCond.copy()

        self.writeData = writeData
        if self.writeData: self.__fileName__()
        self.primary = False
        self.writeTrappedData = writeTrappedData
        


