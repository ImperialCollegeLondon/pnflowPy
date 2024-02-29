import os
import sys

import numpy as np
from itertools import chain
import numpy_indexed as npi
from scipy.sparse import csr_matrix
import warnings
import numba as nb
from numba import int64, float64
from sortedcontainers import SortedList
from solver import Solver


class Computations():

    def __init__(self, obj):
        self.obj = obj
        self.isPolygon = (self.Garray <= self.bndG2)
    
    def __getattr__(self, name):
        return getattr(self.obj, name)

    #@nb.jit(nopython=True)
    def matrixSolver(self, Amatrix, Cmatrix) -> np.array:
        return Solver(Amatrix, Cmatrix).solve()
    
    def computegL(self, g) -> np.array:
        gL = np.zeros(self.nThroats)
        cond = (g[self.tList] > 0.0) & (
            (g[self.P1array] > 0) | (self.P1array < 1)) & (
            (g[self.P2array] > 0) | (self.P2array < 1))
        cond3 = cond & (g[self.P1array] > 0) & (g[self.P2array] > 0)
        cond2 = cond & (g[self.P1array] == 0) & (g[self.P2array] > 0) & (
            self.LP2array_mod > 0)
        cond1 = cond & (g[self.P1array] > 0) & (g[self.P2array] == 0) & (
            self.LP1array_mod > 0)

        gL[cond3] = 1/(self.LP1array_mod[cond3]/g[self.P1array[cond3]] +
                    self.LTarray_mod[cond3]/g[self.tList[cond3]] + self.LP2array_mod[
                    cond3]/g[self.P2array[cond3]])
        gL[cond2] = 1/(self.LTarray_mod[cond2]/g[self.tList[cond2]] + self.LP2array_mod[
                    cond2]/g[self.P2array[cond2]])
        gL[cond1] = 1/(self.LTarray_mod[cond1]/g[self.tList[cond1]] + self.LP1array_mod[
                    cond1]/g[self.P1array[cond1]])

        return gL


    def isConnected3(self, indPS, indTS):
        connected = np.zeros(self.totElements, dtype='bool')
        Notdone = np.zeros(self.totElements, dtype='bool')
        isOnInletBdr = self.isOnInletBdr.copy()
        isOnInletBdr[self.poreList] = False
        isOnOutletBdr = self.isOnOutletBdr.copy()
        isOnOutletBdr[self.poreList] = False

        Notdone[indPS] = True
        Notdone[indTS+self.nPores] = True
        Notdone = (Notdone&(isOnInletBdr|isOnOutletBdr|self.isinsideBox))
        arrlist = list(self.elementLists[(Notdone&isOnInletBdr)[1:-1]])

        while True:
            conn = np.zeros(self.totElements, dtype='bool')
            arrr = np.zeros(self.totElements, dtype='bool')
            try:
                arr = arrlist.pop(0)
                while True:
                    arrr[:] = False
                    try:
                        Notdone[arr] = False
                        conn[arr] = True
                        try:
                            arrr[np.array([*chain(*self.PTConData[arr])])+self.nPores] = True
                        except TypeError:
                            arrr[self.PTConData[arr]+self.nPores]=True
                        except IndexError:
                            arrr[self.P1array[arr-self.nPores-1]] = True
                            arrr[self.P2array[arr-self.nPores-1]] = True
                        arr = self.elementLists[(arrr&Notdone)[1:-1]]
                        assert arr.size > 0
                    except AssertionError:
                        arrlist = np.array(arrlist)
                        arrlist = list(arrlist[Notdone[arrlist]])
                        try:
                            assert conn[isOnOutletBdr].sum()>0
                            connected[conn] = True
                        except AssertionError:
                            pass
                        break
            except IndexError:
                break

        return connected
    

    #def isConnected(self, indPS, indTS):
    def isConnected(self, Notdone):
        connected = np.zeros(self.totElements, dtype='bool')
        conTToInlet = self.conTToInlet+self.nPores
        conTToOutlet = self.conTToOutlet+self.nPores
    
        arrlist = list(conTToInlet[Notdone[conTToInlet]])
        conTToOutlet = conTToOutlet[Notdone[conTToOutlet]]
        arrr = np.zeros(self.totElements, dtype='bool')
        conn = np.zeros(self.totElements, dtype='bool')
        
        while True:
            conn[:] = False
            try:
                arr = arrlist.pop(np.argmin(self.distToExit[arrlist]))
                while True:
                    arrr[:] = False
                    try:
                        Notdone[arr] = False
                        conn[arr] = True
                        try:
                            arrr[self.PTConnections[arr]] = True
                            #arrr[np.array([*chain(*self.PTConData[arr])])+self.nPores] = True
                        except TypeError:
                            arrr[self.PTConData[arr]+self.nPores]=True
                        except IndexError:
                            arr = arr-self.nPores-1
                            arrr[self.P1array[arr]] = True
                            arrr[self.P2array[arr]] = True
                        arr = self.elementListS[(arrr&Notdone)]
                        assert arr.size > 0
                    except AssertionError:
                        try:
                            assert conn[conTToOutlet].sum()>0
                            connected[conn] = True
                        except AssertionError:
                            pass

                        arrlist = np.array(arrlist)
                        arrlist = list(arrlist[Notdone[arrlist]])

                        break
            except (IndexError, ValueError):
                break
    
        return connected


    def isConnected1(self, indPS, indTS) -> np.array:
        connected = np.zeros(self.totElements, dtype='bool')

        doneP = np.ones(self.nPores+2, dtype='bool')
        doneP[indPS] = False
        doneP[0] = False
        doneT = np.ones(self.nThroats+1, dtype='bool')
        doneT[indTS] = False

        #tin = list(self.conTToIn[~doneT[self.conTToIn]])
        #from IPython import embed; embed()
        tin = self.elementLists[self.isOnInletBdr[1:-1] & self.connected[1:-1]]-self.nPores
        tin = list(tin[~doneT[tin]])
        tout = self.elementLists[self.isOnOutletBdr[1:-1] & self.connected[1:-1]]
        tout = list(tout[~doneT[tout-self.nPores]])
        #from IPython import embed; embed()
        while True:
            try:
                conn = np.zeros(self.totElements, dtype='bool')
                doneP[0] = False
                t = tin.pop(0)
                doneT[t] = True
                conn[t+self.nPores] = True
                while True:
                    #from IPython import embed; embed()
                    
                    p = np.array([self.P1array[t-1], self.P2array[t-1]])

                    p = p[~doneP[p]]
                    doneP[p] = True
                    conn[p] = True

                    try:
                        tt = np.zeros(self.nThroats+1, dtype='bool')
                        tt[np.array([*chain(*self.PTConData[p])])] = True
                        t = self.throatList[tt[1:] & ~doneT[1:]]
                        assert t.size > 0
                        doneT[t] = True
                        conn[t+self.nPores] = True
                    except (AssertionError, IndexError):
                        try:
                            tin = np.array(tin)
                            tin = list(tin[~doneT[tin]])
                        except IndexError:
                            tin=[]
                        break
                try:
                    assert conn[tout].any()
                    connected[conn] = True
                except AssertionError:
                    pass
            except (AssertionError, IndexError):
                break

        connected = connected & self.isinsideBox
        return connected
                

    def isTrapped(self, i, fluid, Pc, args=None):
        try:
            assert not args
            (trapped, trappedPc, trapClust) = (
                self.trapped, self.trappedPc, self.trapClust)
        except AssertionError:
            (trapped, trappedPc, trapClust) = args
        
        try:
            assert trapped[i]
            return True
        except AssertionError:
            try:
                assert fluid
                Notdone = (self.fluid==1)
            except AssertionError:
                Notdone = (self.fluid==0)|self.isPolygon

        arr = Notdone.copy()
        Notdone[[-1, 0, i]] = True, True, False
        arrlist = [i]
        canAdd = Notdone.copy()

        while True:
            try:
                j = arrlist.pop(np.argmin(self.distToBoundary[arrlist]))
                assert j>0
                Notdone[j] = False
                pt = self.elem[j].neighbours
                arrlist.extend(pt[canAdd[pt]])
                canAdd[pt] = False
            except AssertionError:
                Notdone[arrlist] = False
                try:
                    arrlist = np.array(arrlist)[trapped[arrlist]]
                    arrl = []
                    [arrl.extend(self.elementLists[
                        (trapClust==k)[1:-1]]) for k in set(trapClust[arrlist])]
                    Notdone[arrl] = False
                except (IndexError, AssertionError):
                    pass
                
                arr = (arr & ~Notdone)
                trapped[arr] = False
                trappedPc[arr] = 0.0
                trapClust[arr] = 0
                return False
            except (IndexError, ValueError):
                arr = (arr & ~Notdone)
                try:
                    assert trapped[arr].sum()==0
                    trapped[arr] = True
                    trappedPc[arr] = Pc
                    trapClust[arr] = trapClust.max()+1
                except AssertionError:
                    trapped[arr] = True
                    trappedPc[arr] = trappedPc[
                        arr&(trapClust==trapClust[arr].max())][0]
                    trapClust[arr] = trapClust[arr].max()
                return True


    def getValue(self, arrr, gL):
        c = arrr[self.poreList].sum()
        indP = self.poreList[arrr[self.poreList]]
        Cmatrix = np.zeros(c)
        row, col, data = [], [], []
        nPores = self.nPores
        
        def worker(arr: np.array) -> float:
            return sum(gL[arr[arrr[arr+nPores]] - 1])
        
        cond = [*map(worker, self.PTConData[indP])]
        #cond = worker()   
        m = np.arange(c)
        row.extend(m)
        col.extend(m)
        data.extend(cond)

        arrT = arrr[self.tList] & arrr[self.P1array] & arrr[self.P2array]
        cond = -gL[arrT]
        j = npi.indices(indP, self.P1array[arrT])
        k = npi.indices(indP, self.P2array[arrT])
        row.extend(j)
        col.extend(k)
        data.extend(cond)
        row.extend(k)
        col.extend(j)
        data.extend(cond)

        # for entries on/in the inlet boundary
        arrT = arrr[self.tList] & self.isOnInletBdr[self.tList]
        arrP = self.P1array[arrT]*(arrr[self.P1array[arrT]]) +\
            self.P2array[arrT]*(arrr[self.P2array[arrT]])
        cond = gL[arrT]
        m = npi.indices(indP, arrP)

        Cmatrix = np.array([*map(lambda i: cond[m == i].sum(), range(c))])
        Amatrix = csr_matrix((data, (row, col)), shape=(c, c),
                            dtype=float)
        
        return Amatrix, Cmatrix

    def _getValue_(self, arrr, gL):
        row, col, data = [], [], []
        indP = self.poreList[arrr[self.poreList]]
        c = indP.size
        Cmatrix = np.zeros(c)
        nPores = self.nPores
        mList = dict(zip(indP, np.arange(c)))

        # throats within the calcBox
        cond1 = arrr[self.tList] & arrr[self.P1array] & arrr[self.P2array]
        indT1 = zip(self.throatList[cond1], self.P1array[cond1], self.P2array[cond1])

        # throats on the inletBdr
        cond2 = arrr[self.tList] & self.isOnInletBdr[self.tList]
        indP2 = self.P1array*(cond2 & arrr[self.P1array]) + self.P2array*(
            cond2 & arrr[self.P2array])
        indT2 = zip(self.throatList[cond2], indP2[cond2])

        # throats on the outletBdr
        cond3 = arrr[self.tList] & self.isOnOutletBdr[self.tList]
        indP3 = self.P1array*(cond3 & arrr[self.P1array]) + self.P2array*(
            cond3 & arrr[self.P2array])
        indT3 = zip(self.throatList[cond3], indP3[cond3])

        def worker1(t, P1, P2):
            cond = gL[t-1]
            P1, P2 = mList[P1], mList[P2]
            row.extend((P1, P2, P1, P2))
            col.extend((P2, P1, P1, P2))
            data.extend((-cond, -cond, cond, cond))
            return
        
        def worker2(t:int, P:int):
            cond = gL[t-1]
            P = mList[P]
            row.append(P)
            col.append(P)
            data.append(cond)
            Cmatrix[P] += cond

        def worker3(t:int, P:int):
            cond = gL[t-1]
            P = mList[P]
            row.append(P)
            col.append(P)
            data.append(cond)

        #from IPython import embed; embed()
        for arr in indT1: worker1(*arr)
        for arr in indT2: worker2(*arr)
        for arr in indT3: worker3(*arr)

        Amatrix = csr_matrix((data, (row, col)), shape=(c, c), dtype=float)
        
        return Amatrix, Cmatrix
    
    def __getValue__(self, arrr, gL):
        row, col, data = [], [], []
        indP = self.poreListS[arrr[self.poreListS]]
        c = indP.size
        Cmatrix = np.zeros(c)
        mList = dict(zip(indP, np.arange(c)))

        # throats within the calcBox
        cond1 = arrr[self.tList] & self.isinsideBox[self.P1array] & self.isinsideBox[self.P2array]
        indT1 = zip(self.throatList[cond1], self.P1array[cond1], self.P2array[cond1])

        # throats connected to the inletBdr
        condP1 = (arrr[self.tList])&(self.isOnInletBdr[self.P1array])
        condP2 = (arrr[self.tList])&(self.isOnInletBdr[self.P2array])
        indP2 = self.P2array*condP1 + self.P1array*condP2
        cond2 = (condP1 | condP2)
        indT2 = zip(self.throatList[cond2], indP2[cond2])

        # throats connectec to the outletBdr
        condP1 = (arrr[self.tList])&(self.isOnOutletBdr[self.P1array])
        condP2 = (arrr[self.tList])&(self.isOnOutletBdr[self.P2array])
        indP3 = self.P2array*condP1 + self.P1array*condP2
        cond3 = (condP1 | condP2)
        indT3 = zip(self.throatList[cond3], indP3[cond3])

        def worker1(t, P1, P2):
            #print(t, P1, P2)
            cond = gL[t-1]
            P1, P2 = mList[P1], mList[P2]
            row.extend((P1, P2, P1, P2))
            col.extend((P2, P1, P1, P2))
            data.extend((-cond, -cond, cond, cond))

        def worker2(t:int, P:int):
            cond = gL[t-1]
            P = mList[P]
            row.append(P)
            col.append(P)
            data.append(cond)
            Cmatrix[P] += cond

        def worker3(t:int, P:int):
            cond = gL[t-1]
            P = mList[P]
            row.append(P)
            col.append(P)
            data.append(cond)

        for arr in indT1: worker1(*arr)
        for arr in indT2: worker2(*arr)
        for arr in indT3: worker3(*arr)
        
        Amatrix = csr_matrix((data, (row, col)), shape=(c, c), dtype=float)
        
        return Amatrix, Cmatrix
    
    

    def Saturation(self, AreaWP, AreaSP):
        satWP = AreaWP/AreaSP
        num = (satWP[self.isinsideBox]*self.volarray[self.isinsideBox]).sum()
        return num/self.totVoidVolume
    

    def computeFlowrate(self, gL):
        arrr = np.zeros(self.totElements, dtype='bool')    
        arrr[self.P1array[(gL > 0.0)]] = True
        arrr[self.P2array[(gL > 0.0)]] = True
        arrr[self.tList[(gL > 0.0)]] = True
        arrr = (arrr & self.connected & self.isinsideBox)

        self.conn = self.isConnected(arrr)
        Amatrix, Cmatrix = self.__getValue__(self.conn, gL)

        pres = np.zeros(self.nPores+2)
        pres[self.conn[self.poreListS]] = self.matrixSolver(Amatrix, Cmatrix)
        pres[self.isOnInletBdr[self.poreListS]] = 1.0       
        delP = np.abs(pres[self.P1array] - pres[self.P2array])
        qp = gL*delP

        try:
            conTToInlet = self.conTToInlet[self.conn[self.conTToInlet+self.nPores]]
            conTToOutlet = self.conTToOutlet[self.conn[self.conTToOutlet+self.nPores]]
            qinto = qp[conTToInlet-1].sum()
            qout = qp[conTToOutlet-1].sum()
            assert np.isclose(qinto, qout, atol=1e-30)
            qout = (qinto+qout)/2
        except AssertionError:
            pass

        return qout
    

    def computeFlowrate1(self, gL):
        arrPoreList = np.zeros(self.nPores+2, dtype='bool')
        arrPoreList[self.P1array[(gL > 0.0)]] = True
        arrPoreList[self.P2array[(gL > 0.0)]] = True
        indPS = self.poreList[arrPoreList[1:-1]]
        indTS = self.throatList[(gL > 0.0)]
        #from IPython import embed; embed()
        self.conn = self.isConnected(indPS, indTS)
        Amatrix, Cmatrix = self.__getValue__(self.conn, gL)

        #from IPython import embed; embed()

        pres = np.zeros(self.nPores+2)
        pres[self.poreList[self.isOnInletBdr[self.poreList]]] = 1.0
        #print('*************************************')
        #import cProfile
        #import pstats
        #profiler = cProfile.Profile()
        #profiler.enable()
        pres[1:-1][self.conn[self.poreList]] = self.matrixSolver(Amatrix, Cmatrix)
        #profiler.disable()
        #stats = pstats.Stats(profiler).sort_stats('cumtime')
        #stats.print_stats()
        #pres[1:-1][self.conn[self.poreList]] = self.matrixSolver(Amatrix, Cmatrix)

        delP = np.abs(pres[self.P1array] - pres[self.P2array])
        qp = gL*delP
        qinto = qp[self.isOnInletBdr[self.tList] & self.conn[self.tList]].sum()
        qout = qp[self.isOnOutletBdr[self.tList] & self.conn[self.tList]].sum()
        try:
            assert np.isclose(qinto, qout, atol=1e-30)
            qout = (qinto+qout)/2
        except AssertionError:
            pass

        return qout
    
    def computePerm(self):
        gwL = self.computegL(self.gWPhase)
        self.obj.qW = self.qW = self.computeFlowrate(gwL)
        self.obj.krw = self.krw = self.krw = self.qW/self.qwSPhase
        self.trapCluster_W[self.conn] = 0
        self.trappedW[self.conn] = False
        self.connW[:] = self.conn
        
        try:
            assert self.fluid[self.conTToOutlet+self.nPores].sum() > 0
            gnwL = self.computegL(self.gNWPhase)
            self.obj.qNW = self.qNW = self.computeFlowrate(gnwL)
            self.obj.krnw = self.krnw = self.qNW/self.qnwSPhase
            self.trapCluster_NW[self.conn] = 0
            self.trappedNW[self.conn] = False
            self.connNW[:] = self.conn
        except AssertionError:
            self.qNW, self.krnw = 0.0, 0.0
        
        self.obj.fw = self.qW/(self.qW + self.qNW)
    

    def weibull(self) -> np.array:
        randNum = self.rand(self.nPores)
        if self.delta < 0 and self.eta < 0:              # Uniform Distribution
            return self.minthetai + (self.maxthetai-self.minthetai)*randNum
        else:                                  # Weibull Distribution
            return (self.maxthetai-self.minthetai)*pow(-self.delta*np.log(
                randNum*(1.0-np.exp(-1.0/self.delta))+np.exp(-1.0/self.delta)), 
                1.0/self.eta) + self.minthetai
        
    
    def __wettabilityDistribution__(self) -> np.array:
        # compute the distribution of contact angles in the network
        contactAng = np.zeros(self.totElements)
        conAng = self.weibull()        

        print(np.array([conAng[self.poreList-1].mean(), conAng[self.poreList-1].std(),
                        conAng[self.poreList-1].min(), conAng[self.poreList-1].max()])*180/np.pi)

        if self.distModel.lower() == 'rmax':
            sortedConAng = conAng[conAng.argsort()[::-1]]
            sortedPoreIndex = self.poreList[self.Rarray[self.poreList].argsort()[::-1]]
            print('rmax')
            from IPython import embed; embed()
        elif self.distModel.lower() == 'rmin':
            sortedConAng = conAng[conAng.argsort()[::-1]]
            sortedPoreIndex = self.poreList[self.Rarray[self.poreList].argsort()]
            print('rmin')
            from IPython import embed; embed()
        else:
            cond1 = (self.fluid[self.poreList] == 0)
            cond2 = (self.fluid[self.poreList] == 1)

            sortedPoreIndex = self.poreList.copy()
            self.shuffle(sortedPoreIndex)
            self.shuffle(conAng)
            contactAng[sortedPoreIndex] = conAng.copy()  #'''
            
        randNum = self.rand(self.nThroats)
        conda = (self.P1array > 0)
        condb = (self.P2array > 0)
        condc = (conda & condb)
        
        contactAng[self.tList[~conda]] = contactAng[self.P2array[~conda]]
        contactAng[self.tList[~condb]] = contactAng[self.P1array[~condb]]
        contactAng[self.tList[condc & (randNum > 0.5)]] = contactAng[
            self.P1array[condc & (randNum > 0.5)]]
        contactAng[self.tList[condc & (randNum <= 0.5)]] = contactAng[
            self.P2array[condc & (randNum <= 0.5)]]
        
        #contactAng[self.poreList[self.fluid[self.poreList]==0]] = 0
        #contactAng[self.tList[self.fluid[self.tList]==0]] = 0
        #if not self.is_oil_inj:
            #from IPython import embed; embed() 
         #   randNum = self.rand((self.PcD > self.maxPc).sum())
          #  contactAng[self.fluid==0] = 40/180*np.pi   #*randNum         # Uniform Distribution'''
        
        print(np.array([contactAng.mean(), contactAng.std(), contactAng.min(), contactAng.max()])*180/np.pi)
        #from IPython import embed; embed()
        thetaRecAng, thetaAdvAng = self.setContactAngles(contactAng)

        return contactAng, thetaRecAng, thetaAdvAng


    
    def setContactAngles(self, contactAng) -> np.array:
        if self.wettClass == 1:
            thetaRecAng = contactAng.copy()
            thetaAdvAng = contactAng.copy()
        elif self.wettClass == 2:
            growthExp = (np.pi+self.sepAng)/np.pi
            thetaRecAng = np.maximum(0.0, growthExp*contactAng - self.sepAng)
            thetaAdvAng = np.minimum(np.pi, growthExp*contactAng)
        elif self.wettClass == 3:
            thetaRecAng = np.zeros(contactAng.size)
            thetaAdvAng = np.zeros(contactAng.size)

            cond1 = (contactAng >= 0.38349) & (contactAng < 1.5289)
            cond2 = (contactAng >= 1.5289) & (contactAng < 2.7646)
            cond3 = (contactAng >= 2.7646)
            thetaRecAng[cond1] = (0.5*np.exp(
                0.05*contactAng[cond1]*180.0/np.pi)-1.5)*np.pi/180.0
            thetaRecAng[cond2] = 2.0*(contactAng[cond2]-1.19680)
            thetaRecAng[cond3] = np.pi

            cond4 = (contactAng >= 0.38349) & (contactAng < 1.61268)
            cond5 = (contactAng >= 1.61268) & (contactAng < 2.75805)
            cond6 = (contactAng >= 2.75805)
            thetaAdvAng[cond4] = 2.0*(contactAng[cond4]-0.38349)
            thetaAdvAng[cond5] = (181.5 - 4051.0*np.exp(
                -0.05*contactAng[cond5]*180.0/np.pi))*np.pi/180.0
            thetaAdvAng[cond6] = np.pi
        elif self.wettClass == 4:
            thetaAdvAng = contactAng.copy()
            thetaRecAng = pow(np.pi - 1.3834263 - pow(
                np.pi - thetaAdvAng + 0.004, 0.45), 1.0/0.45) - 0.004
        else:
            plusCoef = np.pi - (0.1171859*(self.sepAng**3) - 0.6614868*(
                self.sepAng**2) + 1.632065*self.sepAng)
            exponentCoef = 1.0 - (0.01502745*(self.sepAng**3) - 0.1015349*(
                self.sepAng**2) + 0.4734059*self.sepAng)
            thetaAdvAng = contactAng.copy()
            thetaRecAng = pow(plusCoef - pow(
                np.pi - thetaAdvAng + 0.004, exponentCoef), 1.0/exponentCoef) - 0.004
            
        return thetaRecAng, thetaAdvAng

    def __computeFd__(self, arrr, arrBeta) -> np.array:
        thet = self.contactAng[arrr, np.newaxis]
        cond = (arrBeta < (np.pi/2-thet))
        arr3 = np.cos(thet)*np.cos(thet + arrBeta)/np.sin(arrBeta)
        arr4 = np.pi/2 - thet - arrBeta
        arr1 = (arr3-arr4)/pow(np.cos(thet), 2)
        C1 = np.sum(arr1*cond, axis=1)

        num = 1 + np.sqrt(1 - 4*self.Garray[arrr]*C1)
        den = 1 + 2*np.sqrt(np.pi*self.Garray[arrr])

        Fd = num/den
        return Fd
    
    
    def createFilms(self, arr, arrr, halfAng, Pc, m_exists,
                m_inited, m_initOrMaxPcHist, m_initOrMinApexDistHist, advPc,
                recPc, m_initedApexDist):

        arrr = arrr[:, np.newaxis]
        Pc = Pc[:, np.newaxis]
        cond = (~(m_exists & m_inited) & arrr)

        try:
            assert cond.sum() > 0
            conAng = self.thetaRecAng[arr, np.newaxis] if self.is_oil_inj else self.thetaAdvAng[
                arr, np.newaxis]
            condf = cond & (conAng < (np.pi/2 - halfAng))
            assert condf.sum() > 0
            m_exists[condf] = True
            m_initedApexDist[condf] = np.maximum((self.sigma/Pc*np.cos(
                conAng+halfAng)/np.sin(halfAng))[condf], 0.0)

            advPc[condf] = np.where(m_initedApexDist[
                condf] != 0.0, self.sigma*np.cos((np.minimum(np.pi, self.thetaAdvAng[
                arr, np.newaxis])+halfAng)[condf])/(
                m_initedApexDist*np.sin(halfAng))[condf], 0.0)

            recPc[condf] = np.where(m_initedApexDist[
                condf] != 0.0, self.sigma*np.cos((np.minimum(np.pi, self.thetaRecAng[
                arr, np.newaxis])+halfAng)[condf])/(
                m_initedApexDist*np.sin(halfAng))[condf], 0.0)
            
            m_inited[condf] = True
            condu = condf & (Pc > m_initOrMaxPcHist)
            assert condu.sum() > 0
            m_initOrMinApexDistHist[condu] = m_initedApexDist[condu]
            m_initOrMaxPcHist[condu] = (Pc*condu)[condu]
        except AssertionError:
            pass


    def cornerApex(self, arr, arrr, halfAng, Pc, conAng, m_exists,
               m_initOrMaxPcHist, m_initOrMinApexDistHist, advPc,
               recPc, apexDist, initedApexDist, accurat=False,
               overidetrapping=False):
        warnings.simplefilter(action='ignore', category=RuntimeWarning)

        apexDist[~m_exists & arrr] = self.MOLECULAR_LENGTH
        delta = 0.0 if accurat else self._delta
        # update the apex dist and contact angle
        conAng = np.ones(m_exists.shape)*conAng[arr]

        try:
            assert not overidetrapping
            apexDist[:, arrr] = initedApexDist[:, arrr]
            assert self.trappedW[arr[arrr]].sum()+self.trappedNW[arr[arrr]].sum()>0

            trappedPc = self.trappedW_Pc[arr]
            cond = (~self.trappedW[arr]) & self.trappedNW[arr] & arrr
            trappedPc[cond] = self.trappedNW_Pc[arr[cond]]
            cond = (self.trappedW[arr]|self.trappedNW[arr])
            
            part = np.maximum(-0.999999, np.minimum(
                0.999999, (trappedPc*initedApexDist*np.sin(
                    halfAng)).T[cond]/self.sigma))
            try:
                conAng.T[cond] = np.minimum(np.maximum(np.arccos(part)-halfAng.T[cond], 0.0), np.pi)
            except IndexError:
                conAng.T[cond] = np.minimum(np.maximum(np.arccos(part)-halfAng.T, 0.0), np.pi)
            
        except AssertionError:
            pass

        # condition 1
        #print('condition 1')
        cond1a = m_exists & (advPc-delta <= Pc) & (Pc <= recPc+delta)
        cond1 = cond1a & arrr
        try:
            assert cond1.sum() > 0
            #if not self.is_oil_inj and 10761 in arr: print('  cond1  ')
            part = np.minimum(0.999999, np.maximum(
                Pc*initedApexDist*np.sin(halfAng)/self.sigma,
                -0.999999))
            hingAng = np.minimum(np.maximum(
                (np.arccos(part)-halfAng)[cond1], -self._delta),
                np.pi+self._delta)
            conAng[cond1] = np.minimum(np.maximum(hingAng, 0.0), np.pi)
            apexDist[cond1] = initedApexDist[cond1]
        except AssertionError:
            pass

        # condition 2
        #print('condition 2')
        cond2a = m_exists & ~cond1a & (Pc < advPc)
        cond2 = cond2a & arrr
        try:
            assert cond2.sum() > 0
            #if not self.is_oil_inj and 10761 in arr: print('  cond2  ')
            conAng[cond2] = (self.thetaAdvAng[arr]*cond2)[cond2]
            apexDist[cond2] = (self.sigma/Pc*np.cos(
                conAng+halfAng)/np.sin(halfAng))[cond2]

            cond2b = (apexDist < initedApexDist) & cond2
            assert cond2b.sum() > 0
            #print('  cond2b  ')
            part = Pc*initedApexDist*np.sin(halfAng)/self.sigma
            part = np.minimum(0.999999, np.maximum(part, -0.999999))
            hingAng = np.minimum(np.maximum(
                (np.arccos(part)-halfAng)[cond2b], 0.0), np.pi)
            
            conAng[cond2b] = hingAng
            apexDist[cond2b] = initedApexDist[cond2b]
        except AssertionError:
            pass

        # condition 3
        #print('  condition 3   ')
        cond3a = m_exists & ~cond1a & ~cond2a & (Pc > m_initOrMaxPcHist)
        cond3 = cond3a & arrr
        try:
            assert cond3.sum() > 0
            #if not self.is_oil_inj and 10761 in arr: print('  cond3  ')
            conAng[cond3] = np.minimum(np.pi, (self.thetaRecAng[arr]*cond3)[cond3])
            apexDist[cond3] = (self.sigma/Pc*np.cos(
                conAng+halfAng)/np.sin(halfAng))[cond3]
        except AssertionError:
            pass

        # condition 4
        #print('  condition 4  ')
        cond4a = m_exists & ~cond1 & ~cond2a & ~cond3a & (Pc > recPc)
        cond4 = (cond4a*arrr)
        try:
            assert cond4.sum() > 0
            #print('  condition 4  ')
            #if not self.is_oil_inj and 10761 in arr: print('  cond4  ')
            conAng[cond4] = (self.thetaRecAng[arr]*cond4)[cond4]
            apexDist[cond4] = (self.sigma/Pc*np.cos(conAng+halfAng)/np.sin(halfAng))[cond4]
            cond4b = cond4 & (apexDist > initedApexDist)
            cond4c = cond4 & (~cond4b) & (apexDist < m_initOrMinApexDistHist)
            try:
                assert cond4b.sum() > 0
                #print('cond4b')
                part = (Pc*initedApexDist*np.sin(halfAng)/self.sigma)
                part = np.maximum(np.minimum(part, 0.999999), -0.999999)
                hingAng = np.minimum(np.maximum((
                    np.arccos(part)-halfAng)[cond4b], 0.0), np.pi)
                conAng[cond4b] = hingAng
                apexDist[cond4b] = initedApexDist[cond4b]
            except AssertionError:
                pass
            try:
                assert cond4c.sum() > 0
                #print('cond4c')
                part = (Pc*m_initOrMinApexDistHist*np.sin(halfAng)/self.sigma)
                part = np.maximum(np.minimum(part, 0.999999), -0.999999)
                hingAng = np.minimum(np.maximum((
                    np.arccos(part)-halfAng)[cond4c], 0.0), np.pi)
                conAng[cond4c] = hingAng
                apexDist[cond4c] = m_initOrMinApexDistHist[cond4c]
            except AssertionError:
                pass
        except AssertionError:
            pass

        # condition 5
        #print('  condition 5  ')
        cond5 = m_exists & ~cond1 & ~cond2 & ~cond3a & ~cond4a
        cond5 = (cond5*arrr)
        try:
            assert cond5.sum() > 0
            if not self.is_oil_inj: print('  cond5  ')
            apexDist[cond5] = ((self.sigma/Pc)*np.cos(
                conAng+halfAng)/np.sin(halfAng))[cond5]
        except AssertionError:
            pass
        
        return conAng.T, apexDist.T


    def calcAreaW(self, arrr, halfAng, conAng, m_exists, apexDist):
        # -- obtain corner conductance -- #
        dimlessCornerA = np.zeros(m_exists.shape)

        cond1 = m_exists & (np.abs(conAng+halfAng-np.pi/2) < 0.01)
        try:
            dimlessCornerA[cond1] = np.sin(halfAng[cond1])*np.cos(halfAng[cond1])
        except IndexError:
            dimlessCornerA[cond1] = (np.sin(halfAng)*np.cos(halfAng)*cond1)[cond1]

        cond2 = m_exists & (np.abs(conAng+halfAng-np.pi/2) >= 0.01)
        dimlessCornerA[cond2] = pow((np.sin(halfAng)/np.cos(
            conAng + halfAng))[cond2], 2.0)*(np.cos(conAng)*np.cos(
            conAng + halfAng)/np.sin(halfAng)+conAng+halfAng-np.pi/2)[cond2]
        
        cornerGstar = (np.sin(halfAng)*np.cos(halfAng)/(
            4*pow(1+np.sin(halfAng), 2))*m_exists)
        cornerG = cornerGstar.copy()
        
        cond3 = m_exists & (np.abs(conAng+halfAng-np.pi/2) > 0.01)
        cornerG[cond3] = dimlessCornerA[cond3]/(4.0*pow((1 - np.sin(
            halfAng)/np.cos(conAng + halfAng)*(
                conAng + halfAng - np.pi/2))[cond3], 2.0))

        cFactor = np.where(cornerG != 0.0, 0.364+0.28*cornerGstar/cornerG, 0.0)
        conductance = cFactor*pow(apexDist, 4)*pow(
            dimlessCornerA, 2)*cornerG/self.muw
        area = apexDist*apexDist*dimlessCornerA
    
        cornerCond = conductance.sum(axis=1)
        cornerArea = area.sum(axis=1)

        return cornerArea[arrr], cornerCond[arrr]


    def __finitCornerApex__(self, pc):
        trapped = (self.trappedW | self.trappedNW)
        arrr = self.connected
        arrrS = arrr[self.elemSquare]
        arrrT = arrr[self.elemTriangle]
       
        apexDist = np.zeros(self.cornExistsTr.T.shape)
        self.finitCornerApex(
            self.elemTriangle, arrrT, self.halfAnglesTr.T, pc,
            self.cornExistsTr.T, self.initedTr.T, self.initOrMaxPcHistTr.T,
            self.initOrMinApexDistHistTr.T, self.advPcTr.T,
            self.recPcTr.T, apexDist, self.initedApexDistTr.T, trapped)
    
        apexDist = np.zeros(self.cornExistsSq.T.shape)
        self.finitCornerApex(
            self.elemSquare, arrrS, self.halfAnglesSq[:, np.newaxis], pc,
            self.cornExistsSq.T, self.initedSq.T, self.initOrMaxPcHistSq.T,
            self.initOrMinApexDistHistSq.T, self.advPcSq.T,
            self.recPcSq.T, apexDist, self.initedApexDistSq.T, trapped)


    def __initCornerApex__(self):
        trapped = (self.trappedW | self.trappedNW)
        arrr = self.connected
        arrrS = arrr[self.elemSquare]
        arrrT = arrr[self.elemTriangle]
        
        self.initCornerApex(
            self.elemTriangle, arrrT, self.halfAnglesTr, self.cornExistsTr, self.initedTr,
            self.recPcTr, self.advPcTr, self.initedApexDistTr, trapped)
    
        self.initCornerApex(
            self.elemSquare, arrrS, self.halfAnglesSq, self.cornExistsSq, self.initedSq,
            self.recPcSq, self.advPcSq, self.initedApexDistSq, trapped)

        
    def finitCornerApex(self, arr, arrr, halfAng, Pc, m_exists,
                    m_inited, m_initOrMaxPcHist, m_initOrMinApexDistHist,
                    advPc, recPc, apexDist, m_initedApexDist, trapped):

        cond = arrr & (m_inited | (~trapped[arr])) & m_exists
        conAng = self.thetaRecAng.copy() if self.is_oil_inj else self.thetaAdvAng.copy()
        conAng, apexDist = self.cornerApex(
            arr, arrr, halfAng, Pc, conAng, 
            cond, m_initOrMaxPcHist, m_initOrMinApexDistHist,
            advPc, recPc, apexDist, m_initedApexDist, overidetrapping=True)
        
        apexDist = apexDist.T
        recPc[cond] = self.sigma*np.cos((np.minimum(np.pi, self.thetaRecAng[
            arr])+halfAng)[cond])/((apexDist*np.sin(halfAng))[cond])
        advPc[cond] = self.sigma*np.cos((np.minimum(np.pi, self.thetaAdvAng[
            arr])+halfAng)[cond])/((apexDist*np.sin(halfAng))[cond])

        cond1 = cond & (Pc > m_initOrMaxPcHist)
        m_initOrMinApexDistHist[cond1] = apexDist[cond1]
        m_inited[cond] = False
        m_initedApexDist[cond] = apexDist[cond]
        
        try:
            m_initOrMaxPcHist[cond1] = Pc[cond1]
        except (TypeError, IndexError):
            m_initOrMaxPcHist[cond1] = Pc


    def initCornerApex(self, arr, arrr, halfAng, m_exists, m_inited,
                       recPc, advPc, m_initedApexDist, trapped):

        cond =  (m_exists & (arrr&~trapped[arr])[:, np.newaxis])
        try:
            assert cond.sum()>0
            m_inited[cond] = True
            Pc =np.zeros_like(m_initedApexDist)
            Pc[cond] = self.sigma*np.cos(np.minimum(
                np.pi, ((self.thetaRecAng[arr, np.newaxis]+halfAng)*cond)[cond]))/(
                    m_initedApexDist*np.sin(halfAng))[cond]

            recPc[cond & (recPc < Pc)] = Pc[cond & (recPc < Pc)]
            advPc[cond] = self.sigma*np.cos(np.minimum(
                np.pi, ((self.thetaAdvAng[arr, np.newaxis]+halfAng)*cond)[cond]))/(
                    (m_initedApexDist*np.sin(halfAng))[cond])
        except AssertionError:
            pass
        

    def writeResult(self, result_str, Pc):
        print('Sw: %7.6g  \tqW: %8.6e  \tkrw: %8.6g  \tqNW: %8.6e  \tkrnw:\
              %8.6g  \tPc: %8.6g\t %8.0f invasions' % (
              self.satW, self.qW, self.krw, self.qNW, self.krnw,
              Pc, self.totNumFill, ))
            
        if self.writeData:
            result_str+="\n%.6g,%.6e,%.6g,%.6e,%.6g,%.6g,%.0f" % (
                self.satW, self.qW, self.krw, self.qNW, self.krnw,
                Pc, self.totNumFill, )
        
        return result_str


        
