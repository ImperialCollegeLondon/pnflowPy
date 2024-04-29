import os
import numpy as np
import pandas as pd
from math import sqrt, pi
from time import time
import warnings
from itertools import chain
from joblib import Parallel, delayed
import concurrent.futures
import asyncio
from mpi4py.futures import MPIPoolExecutor

from inputData import InputData

class Network(InputData):

    def __init__(self, inputFile):
        st = time()        
        super().__init__(inputFile)

        print('Reading network files for {}'.format(self.network()))

        self.MAX_ITER = 1000
        self.EPSILON = 1.0e-6
        self._delta = 1e-7
        self.MOLECULAR_LENGTH = 1.0E-10
        self.satW = 1.0
        self.mu = 0.001
        self.RAND_MAX = 2147483647
        self.bndG1 = sqrt(3)/36+0.00001
        self.bndG2 = 0.07
        self.bndG3 = sqrt(3)/36
        self.pin_ = -1
        self.pout_ = 0

        (self.sigma, self.muw, self.munw, self.ohmw, self.ohmnw,
         self.rhow, self.rhonw) = self.fluidproperties()
        
        (self.m_minNumFillings, self.m_initStepSize, self.m_extrapCutBack,
         self.m_maxFillIncrease, self.m_StableFilling) = self.satCovergence()
        
        self.title = self.network()
        self.calcBox = self.__calcBox__()
        self.SEED = self.randSeed()
        np.random.seed(self.SEED)
        self.dirname = os.path.dirname(__file__)
        self.__readNetworkFiles__()
        self.PTConnections = np.zeros([self.nPores+2, self.maxPoreCon], dtype='int')
        self.TPConnections = np.zeros([self.nThroats+1, 2], dtype='int')
        self.xstart = self.calcBox[0]*self.xDim
        self.xend = self.calcBox[1]*self.xDim
        self.NetworkData()
        
        self.__identifyConnectedElements__()
        self.__computeHalfAng__(self.elemTriangle)
        self.halfAnglesSq = np.array([pi/4, pi/4, pi/4, pi/4])
        
        self.__isinsideBox__()
        self.__isOnBdr__()
        self.__modifyLength__()
        self.__computeDistToExit__()
        self.__computeDistToBoundary__()
        self.__porosity__()

        self.nTriangles = self.elemTriangle.size
        self.nSquares = self.elemSquare.size
        self.nCircles = self.elemCircle.size

        self.elem = self.formArray1(self.totElements, Outlet(self),
                                [NElement(self, i) for i in range(1,self.totElements-1)],
                                Inlet(self), dtype='object')
        
        
        #self.__elementList__()
        #[*map(lambda i: self.__elementList__(i), self.elementLists)]

        self.__writeData__()
        print('time taken:   ', time()-st)
        del self.pore, self.throat, self.poreCon, self.throatCon
        #from IPython import embed; embed()

        
    def __readNetworkFiles__(self):
        self.nThroats = int(open(self.cwd + '/' + str(self.title) + "_link1.dat").readline())
        arr1 = open(self.cwd + '/' + str(self.title) + "_node1.dat").readline().split()
        self.nPores = int(arr1[0])
        self.xDim, self.yDim, self.zDim = float(arr1[1]), float(arr1[2]), float(arr1[3])

        link1 = np.loadtxt(self.cwd + '/' + str(self.title) + "_link1.dat", skiprows=1)
        link2 = np.loadtxt(self.cwd + '/' + str(self.title) + "_link2.dat")
        Lines3 = open(self.cwd + '/' + str(self.title) + "_node1.dat").readlines()
        node2 = np.loadtxt(self.cwd + '/' + str(self.title) + "_node2.dat")

        self.throat = np.concatenate((link1[:,[0,1,2,3,4]], link2[:,[3,4,5]], 
                                      link1[:,[5]], link2[:,[6,7]]), axis=1)
    
        self.maxPoreCon = 0
        self.poreCon = [np.array([])]
        self.throatCon = [np.array([])]
        self.pore = [*map(self.__getDataP__, Lines3[1:], node2)]

        self.poreList = np.arange(1, self.nPores+1)
        self.throatList = np.arange(1, self.nThroats+1)
        self.tList = self.throatList+self.nPores
        self.Area_ = self.yDim*self.zDim
        self.Lnetwork = self.xDim
        self.totElements = self.nPores+self.nThroats+2
        self.poreListS = np.arange(self.nPores+2)
        self.poreListS[-1] = -1
        self.elementLists = np.arange(1, self.nPores+self.nThroats+1)
        self.elementListS = np.arange(self.totElements)
        self.elementListS[-1] = -1
        

    def __readNetworkFiles1__(self):
        # from IPython import embed; embed()
        # read the network files and process data       
        Lines1 = open(self.cwd + '/' + str(self.title) + "_link1.dat").readlines()
        Lines2 = open(self.cwd + '/' + str(self.title) + "_link2.dat").readlines()
        Lines3 = open(self.cwd + '/' + str(self.title) + "_node1.dat").readlines()
        Lines4 = open(self.cwd + '/' + str(self.title) + "_node2.dat").readlines()

        Lines1 = [*map(str.split, Lines1)]
        Lines2 = [*map(str.split, Lines2)]
        Lines3 = [*map(str.split, Lines3)]
        Lines4 = [*map(str.split, Lines4)]

        self.nThroats = int(Lines1[0][0])
        arr1 = Lines3[0]
        [self.nPores, self.xDim, self.yDim, self.zDim] = [
            int(arr1[0]), float(arr1[1]), float(arr1[2]), float(arr1[3])]
        del arr1
        self.maxPoreCon = max([*map(lambda x: int(x[4]), Lines3[1:])])

        self.throat = [*map(self.__getDataT__, Lines1[1:], Lines2)]
        self.pore = [*map(self.__getDataP__, Lines3[1:], Lines4)]
        self.poreCon = [*map(self.__getPoreCon__, Lines3[1:])]
        self.poreCon.insert(0, np.array([]))
        self.throatCon = [*map(self.__getThroatCon__, Lines3[1:])]
        self.throatCon.insert(0, np.array([]))

        self.poreList = np.arange(1, int(self.nPores)+1)
        self.throatList = np.arange(1, int(self.nThroats)+1)
        self.tList = self.nPores+self.throatList
        self.Area_ = self.yDim*self.zDim
        self.Lnetwork = self.xDim
        self.totElements = self.nPores+self.nThroats+2
        self.poreListS = np.arange(self.nPores+2)
        self.poreListS[-1] = -1
        self.elementLists = np.arange(1, self.nPores+self.nThroats+1)
        self.elementListS = np.arange(self.nPores+self.nThroats+2)
        self.elementListS[-1] = -1
        #from IPython import embed; embed()


    def __getDataT__(self, x, y):
        return [int(x[0]), int(x[1]), int(x[2]), float(x[3]), float(x[4]),
                float(y[3]), float(y[4]), float(y[5]), float(x[5]), float(y[6]),
                float(y[7])]


    def __getDataP__(self, x, y):
        x = x.split()
        a = 5+int(x[4])
        self.maxPoreCon = max(self.maxPoreCon, int(x[4]))
        self.poreCon.append(np.array([*map(int, x[5:a])], dtype='int'))
        self.throatCon.append(np.array([*map(int, x[a+2:])], dtype='int'))
        return [int(x[0]), float(x[1]), float(x[2]), float(x[3]), int(x[4]),
                float(y[1]), float(y[2]), float(y[3]), float(y[4]), bool(int(x[a])),
                bool(int(x[a+1]))]

    def __getPoreCon__(self, x):
        a = 5+int(x[4])
        return np.array([*map(int, x[5:a])], dtype='int')

    def __getThroatCon__(self, x):
        a = 5+int(x[4])
        return np.array([*map(int, x[a+2:])], dtype='int')

    def NetworkData(self):
        warnings.simplefilter(action='ignore', category=FutureWarning)
        pd.options.mode.chained_assignment = None

        PoreData = pd.DataFrame(self.pore, columns=[
            "P", "x", "y", "z", "connNum", "volume", "r", "shapeFact",
            "clayVol", "poreInletStat", "poreOutletStat"])

        ThroatData = pd.DataFrame(self.throat, columns=[
            "T", "P1", "P2", "r", "shapeFact", "LP1",
            "LP2", "LT", "lenTot", "volume", "clayVol"])
        
        self.P1array = ThroatData['P1'].values.astype('int')
        self.P2array = ThroatData['P2'].values.astype('int')
        self.LP1array = ThroatData['LP1'].values
        self.LP2array = ThroatData['LP2'].values
        self.LTarray = ThroatData['LT'].values

        
        self.x_array = self.formArray1(self.nPores+2, self.Lnetwork+1e-15,
                                      PoreData['x'].values, -1e-15)
        self.y_array = self.formArray1(self.nPores+2, self.yDim/2,
                                      PoreData['y'].values, self.yDim/2)
        self.z_array = self.formArray1(self.nPores+2, self.zDim/2,
                                      PoreData['z'].values, self.zDim/2)

        self.lenTotarray = ThroatData['lenTot'].values
        self.connNum_array = self.formArray1(self.nPores+2, 0,
                                      PoreData['connNum'].values, 0)      
        self.poreInletStat = self.formArray1(self.nPores+2, False,
                                      PoreData['poreInletStat'].values, False,
                                      dtype='bool')
        self.conPToIn = self.poreList[self.poreInletStat[1:-1]]
        self.poreOutletStat = self.formArray1(self.nPores+2, False,
                                      PoreData['poreOutletStat'].values, False,
                                      dtype='bool')
        
        self.conPToOut = self.poreList[self.poreOutletStat[1:-1]]
        self.conTToIn = self.throatList[(self.P1array == self.pin_) | (self.P2array == self.pin_)]
        self.conTToOut = self.throatList[(self.P1array == self.pout_) | (
            self.P2array == self.pout_)]
        
        
        self.poreCon.append(self.conPToIn)
        self.throatCon.append(self.conTToIn)
        self.PPConData = np.array(self.poreCon, dtype=object)
        self.PTConData = np.array(self.throatCon, dtype=object)

        self.Garray = self.formArray2(self.shapeFact(PoreData['shapeFact'].values),
                                      self.shapeFact(ThroatData['shapeFact'].values))
        self.volarray = self.formArray2(self.shapeFact(PoreData['volume'].values),
                                      self.shapeFact(ThroatData['volume'].values))
        self.Rarray = self.formArray2(self.shapeFact(PoreData['r'].values),
                                      self.shapeFact(ThroatData['r'].values))
        self.ClayVolarray = self.formArray2(self.shapeFact(PoreData['clayVol'].values),
                                      self.shapeFact(ThroatData['clayVol'].values))

        self.isTriangle = (self.Garray <= self.bndG1)
        self.elemTriangle = self.elementLists[self.isTriangle[1:-1]]
        self.isCircle = (self.Garray >= self.bndG2)
        self.elemCircle = self.elementLists[self.isCircle[1:-1]]
        self.isSquare = (self.Garray > self.bndG1) & (self.Garray < self.bndG2)
        self.elemSquare = self.elementLists[self.isSquare[1:-1]]

        del PoreData
        del ThroatData

    
    def formArray1(self, size, first, middle, last, dtype='float'):
        array = np.zeros(size, dtype=dtype)
        array[0] = first
        array[-1] = last
        array[1:-1] = middle
        return array
    

    def formArray2(self, pval, tval, dtype='float'):
        array = np.zeros(self.totElements, dtype=dtype)
        array[self.poreList] = pval
        array[self.tList] = tval
        return array


    def __isinsideBox__(self):
        self.isinsideBox = np.zeros(self.totElements, dtype='bool')
        self.isinsideBox[self.poreList] = (self.x_array[1:-1] >= self.xstart) & (
            self.x_array[1:-1] <= self.xend)
        self.isinsideBox[self.tList] = (
            self.isinsideBox[self.P1array] | self.isinsideBox[self.P2array])
        
    def __isOnBdr__(self):
        self.isOnInletBdr = np.zeros(self.totElements, dtype='bool')
        self.isOnOutletBdr = np.zeros(self.totElements, dtype='bool')

        condP1 = (self.isinsideBox[self.tList]) & (~self.isinsideBox[self.P1array])
        self.isOnInletBdr[self.P1array[condP1]] = (self.x_array[self.P1array[condP1]] < self.xstart)
        self.isOnOutletBdr[self.P1array[condP1]] = (self.x_array[self.P1array[condP1]] > self.xend)

        condP2 = (self.isinsideBox[self.tList]) & (~self.isinsideBox[self.P2array])
        self.isOnInletBdr[self.P2array[condP2]] = (self.x_array[self.P2array[condP2]] < self.xstart)
        self.isOnOutletBdr[self.P2array[condP2]] = (self.x_array[self.P2array[condP2]] > self.xend)

        self.isOnBdr = self.isOnInletBdr | self.isOnOutletBdr
        self.conTToInlet = self.throatList[(self.isinsideBox[self.tList]) & (
            self.isOnInletBdr[self.P1array] | self.isOnInletBdr[self.P2array])]
        self.conTToOutlet = self.throatList[(self.isinsideBox[self.tList]) & (
            self.isOnOutletBdr[self.P1array] | self.isOnOutletBdr[self.P2array])]



    def shapeFact(self, data):
        G = np.minimum(data, np.sqrt(3)/36-0.00005)*(data <= self.bndG1) + (1/16)*((
            data > self.bndG1) & (data < self.bndG2)) + (1/(4*np.pi))*(data >= self.bndG2)
        return G
    
    def rand(self, a=1):
        return np.random.randint(0, self.RAND_MAX, size=a)/self.RAND_MAX
    
    def shuffle(self, obj):
        np.random.shuffle(obj)

    def choice(self, obj, size=1):
        return np.random.choice(obj, size)
    
    def __identifyConnectedElements__(self):
        ttt = list(self.throatList[(self.P1array == self.pin_) | (self.P2array == self.pin_) |
                                    (self.P1array == self.pout_) | (self.P2array == self.pout_)])
        self.connected = np.zeros(self.totElements, dtype='bool')
        self.connected[-1] = True
    
        doneP = np.zeros(self.nPores+2, dtype='bool')
        doneP[[-1, 0]] = True
        doneT = np.zeros(self.nThroats+1, dtype='bool')
        while True:
            indexP = np.zeros(self.nPores, dtype='bool')
            indexT = np.zeros(self.nThroats, dtype='bool')
            try:
                assert len(ttt) > 0
            except AssertionError:
                break
            
            t = ttt.pop(0)
            while True:
                doneT[t] = True
                indexT[t-1] = True
                p = np.array([self.P1array[t-1], self.P2array[t-1]])
                p = p[~doneP[p]]
                doneP[p] = True
                indexP[p-1] = True

                try:
                    assert p.size > 0
                    tt = np.zeros(self.throatList.size+1, dtype='bool')
                    tt[np.array([*chain(*self.PTConData[p])])] = True
                    t = self.throatList[tt[1:] & ~doneT[1:]]
                    assert t.size > 0
                except AssertionError:
                    try:
                        #assert any(tout & indexT)
                        self.connected[self.poreList[indexP]] = True
                        self.connected[self.throatList[indexT]+self.nPores] = True
                        
                    except AssertionError:
                        pass

                    try:
                        assert len(ttt) > 0
                        ttt = np.array(ttt)
                        ttt = list(ttt[~doneT[ttt]])
                    except AssertionError:
                        break
                    break
                except:
                    print('an error occured!!!')
                    from IPython import embed; embed()
                

        self.connected[[0,-1]] = True
        #from IPython import embed; embed()

    def __computeDistToExit__(self):
        self.distToExit = np.zeros(self.totElements)
        self.distToExit[-1] = self.Lnetwork
        self.distToExit[self.poreList] = self.Lnetwork - self.x_array[1:-1]
        self.distToExit[self.tList] = np.minimum(
            self.distToExit[self.P1array], self.distToExit[self.P2array])
        
    def __computeDistToBoundary__(self):
        self.distToBoundary = np.zeros(self.totElements)
        self.distToBoundary[self.poreList] = np.minimum(self.x_array[
            1:-1], self.distToExit[self.poreList])
        self.distToBoundary[self.tList] = np.minimum(
            self.distToBoundary[self.P1array], self.distToBoundary[self.P2array])
        
    def __elementList__(self):
        for ind in self.elementLists:
            try:
                assert ind <= self.nPores
                pp = Pore(self, ind)
                try:
                    assert pp.G <= self.bndG1
                    el = Element(self, Triangle(self, pp))
                except AssertionError:
                    try:
                        assert pp.G > self.bndG2
                        el = Element(self, Circle(pp))
                    except AssertionError:
                        el = Element(self, Square(pp))
            except AssertionError:
                tt = Throat(self, ind)
                try:
                    assert tt.G <= self.bndG1
                    el = Element(self, Triangle(self, tt))
                except AssertionError:
                    try:
                        assert tt.G > self.bndG2
                        el = Element(self, Circle(tt))
                    except AssertionError:
                        el = Element(self, Square(tt))

            self.elem[el.indexOren] = el


    def __computeHalfAng__(self, arrr):
        angle = -12*np.sqrt(3)*self.Garray[arrr]
        assert (np.abs(angle) <= 1.0).all()

        beta2_min = np.arctan(2/np.sqrt(3)*np.cos((np.arccos(angle)/3)+(
            4*np.pi/3)))
        beta2_max = np.arctan(2/np.sqrt(3)*np.cos(np.arccos(angle)/3))

        randNum = self.rand(arrr.size)
        randNum = 0.5*(randNum+0.5)

        beta2 = beta2_min + (beta2_max - beta2_min)*randNum
        beta1 = 0.5*(np.arcsin((np.tan(beta2) + 4*self.Garray[arrr]) * np.sin(
            beta2) / (np.tan(beta2) - 4*self.Garray[arrr])) - beta2)
        beta3 = np.pi/2 - beta2 - beta1

        assert (beta1 <= beta2).all()
        assert (beta2 <= beta3).all()
        self.halfAnglesTr = np.column_stack((beta1, beta2, beta3))

    def __modifyLength__(self):
        cond1 = self.isinsideBox[self.tList] & ~(self.isinsideBox[self.P2array])
        cond2 = self.isinsideBox[self.tList] & (~self.isinsideBox[self.P1array])

        scaleFact = np.zeros(self.nThroats)
        bdr = np.zeros(self.nThroats)
        throatStart = np.zeros(self.nThroats)
        throatEnd = np.zeros(self.nThroats)

        self.LP1array_mod = self.LP1array.copy()
        self.LP2array_mod = self.LP2array.copy()
        self.LTarray_mod = self.LTarray.copy()

        try:
            assert cond1.sum() > 0
            scaleFact[cond1] = (self.x_array[self.P2array[cond1]]-self.x_array[
                self.P1array[cond1]])/(
                    self.LTarray[cond1]+self.LP1array[cond1]+self.LP2array[cond1])
            cond1a = cond1 & (self.P2array < 1)
            scaleFact[cond1a] = (self.x_array[self.P2array[cond1a]]-self.x_array[
                self.P1array[cond1a]])/abs(
                    self.x_array[self.P2array[cond1a]]-self.x_array[self.P1array[cond1a]])

            bdr[cond1 & (self.x_array[self.P2array] < self.xstart)] = self.xstart
            bdr[cond1 & (self.x_array[self.P2array] >= self.xstart)] = self.xend
            throatStart[cond1] = self.x_array[self.P1array[cond1]] + self.LP1array[
                cond1]*scaleFact[cond1]
            throatEnd[cond1] = throatStart[cond1] + self.LTarray[cond1]*scaleFact[cond1]

            cond1b = cond1 & (~cond1a) & (throatEnd > self.xstart) & (throatEnd < self.xend)
            cond1c = cond1 & (~cond1a) & (~cond1b) & (throatStart > self.xstart) & (
                throatStart < self.xend)
            cond1d = cond1 & (~cond1a) & (~cond1b) & (~cond1c)
            
            self.LP2array_mod[cond1a | cond1c | cond1d] = 0.0
            self.LP2array_mod[cond1b] = (bdr[cond1b] - throatEnd[cond1b])/scaleFact[
                cond1b]
            self.LTarray_mod[cond1c] = (bdr[cond1c] - throatStart[cond1c])/scaleFact[
                cond1c]
            self.LTarray_mod[cond1d] = 0.0
        except AssertionError:
            pass

        try:
            assert cond2.sum() > 0
            scaleFact[cond2] = (self.x_array[self.P1array[cond2]]-self.x_array[self.P2array[
                cond2]])/(self.LTarray[cond2]+self.LP1array[cond2]+self.LP2array[cond2])
            cond2a = cond2 & (self.P1array < 1)
            scaleFact[cond2a] = (self.x_array[self.P1array[cond2a]]-self.x_array[self.P2array[
                cond2a]])/abs(self.x_array[self.P1array[cond2a]]-self.x_array[self.P2array[cond2a]]
                            )

            bdr[cond2 & (self.x_array[self.P1array] < self.xstart)] = self.xstart
            bdr[cond2 & (self.x_array[self.P1array] >= self.xstart)] = self.xend
            throatStart[cond2] = self.x_array[self.P2array[cond2]] + self.LP2array[
                cond2]*scaleFact[cond2]
            throatEnd[cond2] = throatStart[cond2] + self.LTarray[cond2]*scaleFact[cond2]

            cond2b = cond2 & (~cond2a) & (throatEnd > self.xstart) & (throatEnd < self.xend)
            cond2c = cond2 & (~cond2a) & (~cond2b) & (throatStart > self.xstart) & (
                throatStart < self.xend)
            cond2d = cond2 & (~cond2a) & (~cond2b) & (~cond2c)
            
            self.LP1array_mod[cond2a | cond2c | cond2d] = 0.0
            self.LP1array_mod[cond2b] = (bdr[cond2b] - throatEnd[cond2b])/scaleFact[
                cond2b]
            self.LTarray_mod[cond2c] = (bdr[cond2c] - throatStart[cond2c])/scaleFact[
                cond2c]
            self.LTarray_mod[cond2d] = 0.0
        except AssertionError:
            pass
    
    def __porosity__(self):
        volTotal = self.volarray[self.isinsideBox].sum() 
        clayvolTotal = self.ClayVolarray[self.isinsideBox].sum()
        self.totBoxVolume = (self.xend-self.xstart)*self.Area_
        self.porosity = (volTotal+clayvolTotal)/self.totBoxVolume
        self.totVoidVolume = volTotal+clayvolTotal

    def __writeData__(self):
        print('porosity = ', self.porosity)
        print('maximum pore connection = ', self.maxPoreCon)
        #from IPython import embed; embed()
        delta_x = self.x_array[self.P2array]-self.x_array[self.P1array]
        delta_y = self.y_array[self.P2array]-self.y_array[self.P1array]
        delta_z = self.z_array[self.P2array]-self.z_array[self.P1array]
        self.avgP2Pdist = np.sqrt(pow(delta_x, 2) + pow(delta_y, 2) + pow(delta_z, 2)).mean()
        #print('Average pore-to-pore distance = ', np.mean(self.lenTotarray))
        print('Average pore-to-pore distance = ', self.avgP2Pdist)
        print('Mean pore radius = ', np.mean(self.Rarray[self.poreList]))



class NElement:
    iTr, iSq, iCi = -1, -1, -1
    def __init__(self, parent, ind):
        
        self.indexOren = ind
        self.isPore = ind<=parent.nPores
        self.volume = parent.volarray[ind]
        self.r = parent.Rarray[ind]
        self.G = parent.Garray[ind]
        self.clayVol = parent.ClayVolarray[ind]
        self.isTriangle = parent.isTriangle[ind]
        self.isSquare = parent.isSquare[ind]
        self.isCircle = parent.isCircle[ind]
        
        self.isinsideBox = parent.isinsideBox[ind]
        self.isOnInletBdr = parent.isOnInletBdr[ind]
        self.isOnOutletBdr = parent.isOnOutletBdr[ind]
        self.isOnBdr = self.isOnInletBdr | self.isOnOutletBdr
        self.isConnected = parent.connected[ind]

        try:
            assert self.isPore
            self.loadPoreProp(parent)
        except AssertionError:
            self.loadThroatProp(parent)
        
        if self.isTriangle: self.loadTriangleProp(parent)
        elif self.isSquare: self.loadSquareProp(parent)
        else: self.loadCircleProp(parent)


    def loadPoreProp(self, obj):
        self.index = self.indexOren
        self.x = obj.x_array[self.index]
        self.y = obj.y_array[self.index]
        self.z = obj.z_array[self.index]
        self.connNum = obj.connNum_array[self.index]
        self.poreInletStat = obj.poreInletStat[self.index]
        self.poreOutletStat = obj.poreOutletStat[self.index]
        self.connP = obj.PPConData[self.index]
        self.neighbours = self.connT = obj.PTConData[self.index]+obj.nPores
        obj.PTConnections[self.index,:self.connT.size]=self.connT


    def loadThroatProp(self, obj):
        self.index = self.indexOren-obj.nPores
        self.P1 = int(obj.P1array[self.index-1])
        self.P2 = int(obj.P2array[self.index-1])
        self.LP1 = obj.LP1array[self.index-1]
        self.LP2 = obj.LP2array[self.index-1]
        self.LT = obj.LTarray[self.index-1]
        self.lenTot = obj.lenTotarray[self.index-1]

        self.conToInlet = -1 in [self.P1, self.P2]
        self.conToOutlet = 0 in [self.P1, self.P2]
        self.conToExit = self.conToInlet | self.conToOutlet
        self.LP1mod = obj.LP1array_mod[self.index-1]
        self.LP2mod = obj.LP2array_mod[self.index-1]
        self.LTmod = obj.LTarray_mod[self.index-1]
        self.neighbours = np.array([self.P1, self.P2])
        obj.TPConnections[self.index]=self.neighbours       
        

    def loadTriangleProp(self, obj):
        NElement.iTr += 1
        self.indexTr = self.iTr
        self.halfAng = obj.halfAnglesTr[self.iTr]
        

    def loadSquareProp(self, obj):
        NElement.iSq += 1
        self.indexSq = self.iSq
        self.halfAng = np.array([pi/4, pi/4, pi/4, pi/4])


    def loadCircleProp(self, obj):
        NElement.iCi += 1
        self.indexCi = self.iCi



class Element:    
    def __init__(self, parent, obj):
        self.__dict__.update(vars(obj))
        self.isinsideBox = parent.isinsideBox[self.indexOren]
        self.isOnInletBdr = parent.isOnInletBdr[self.indexOren]
        self.isOnOutletBdr = parent.isOnOutletBdr[self.indexOren]
        self.isOnBdr = self.isOnInletBdr | self.isOnOutletBdr
        self.isConnected = parent.connected[self.indexOren]


class Pore:    
    def __init__(self, parent, ind):
        self.index = ind
        self.indexOren = ind
        self.x = parent.pore[ind-1][1]
        self.y = parent.pore[ind-1][2]
        self.z = parent.pore[ind-1][3]
        self.connNum = parent.pore[ind-1][4]
        self.volume = parent.pore[ind-1][5]
        self.r = parent.pore[ind-1][6]
        self.G = parent.pore[ind-1][7]
        self.clayVol = parent.pore[ind-1][8]
        self.poreInletStat = parent.pore[ind-1][9]
        self.poreOutletStat = parent.pore[ind-1][10]
        self.connP = parent.poreCon[ind]
        self.neighbours = self.connT = parent.throatCon[ind]+parent.nPores
        self.isPore = True
        parent.PTConnections[self.index][:self.connT.size]=self.connT


class Throat:    
    def __init__(self, parent, ind):
        self.index = ind-parent.nPores
        self.indexOren = ind
        self.P1 = int(parent.throat[self.index-1][1])
        self.P2 = int(parent.throat[self.index-1][2])
        self.r = parent.throat[self.index-1][3]
        self.G = parent.throat[self.index-1][4]
        self.LP1 = parent.throat[self.index-1][5]
        self.LP2 = parent.throat[self.index-1][6]
        self.LT = parent.throat[self.index-1][7]
        self.lenTot = parent.throat[self.index-1][8]
        self.volume = parent.throat[self.index-1][9]
        self.clayVol = parent.throat[self.index-1][10]
        
        # inlet = -1 and outlet = 0
        self.conToInlet = True if -1 in [self.P1, self.P2] else False
        self.conToOutlet =  True if 0 in [self.P1, self.P2] else False
        self.conToExit = self.conToInlet | self.conToOutlet
        self.isPore = False
        self.LP1mod = parent.LP1array_mod[self.index-1]
        self.LP2mod = parent.LP2array_mod[self.index-1]
        self.LTmod = parent.LTarray_mod[self.index-1]
        self.neighbours = np.array([self.P1, self.P2])
        parent.TPConnections[self.index]=self.neighbours       
 
        
class Triangle:  
    def __init__(self, parent, obj):
        self.__dict__.update(vars(obj))
        self.apexDist = np.zeros(3)
        self.c_exists = np.zeros(3, dtype='bool')
        self.hingAng = np.zeros(3)
        self.m_inited = np.zeros(3, dtype='bool')
        self.m_initOrMinApexDistHist = np.full(3, np.inf)
        self.m_initOrMaxPcHist = np.full(3, -np.inf)
        self.m_initedApexDist = np.zeros(3)
        self.indexTr = np.where(parent.elemTriangle == self.indexOren)
        self.halfAng = parent.halfAnglesTr[self.indexTr]


class Square:
    def __init__(self, obj):
        self.__dict__.update(vars(obj))
        self.halfAng = np.array([pi/4, pi/4, pi/4, pi/4])
        self.apexDist = np.zeros(4)
        self.c_exists = np.zeros(4, dtype='bool')
        self.hingAng = np.zeros(4)
        self.m_inited = np.zeros(4, dtype='bool')
        self.m_initOrMinApexDistHist = np.full(4, np.inf)
        self.m_initOrMaxPcHist = np.full(4, -np.inf)
        self.m_initedApexDist = np.zeros(4)
        

class Circle:    
    def __init__(self, obj):
        self.__dict__.update(vars(obj))
        pass
    

class Inlet:
    def __init__(self, parent):
        self.index = -1
        self.x = 0.0
        self.indexOren = -1
        self.connected = True
        self.isinsideBox = False
        self.neighbours = parent.conTToIn
        # to implement this later if need arise
        #parent.PTConnections[self.index][:parent.conTToIn.size]=parent.conTToIn


class Outlet:
    def __init__(self, parent):
        self.index = 0
        self.x = parent.Lnetwork
        self.indexOren = 0
        self.connected = False
        self.isinsideBox = False
        self.neighbours = parent.conTToOut+parent.nPores
        # to implement this if need arise
        #parent.PTConnections[self.index][:parent.conTToOut.size]=parent.conTToOut


#network = Network('./data/input_pnflow_bent.dat')
        

'''import cProfile
import pstats
profiler = cProfile.Profile()
profiler.enable()
self.__elementList__()
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats()
from IPython import embed; embed()'''
'''
%load_ext line_profiler
'''
