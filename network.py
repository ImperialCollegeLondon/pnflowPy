import os
import numpy as np
from math import sqrt, pi
from time import time

from inputData import InputData

class Network(InputData):

    def __init__(self, inputFile):
        print('------------------------------------------------------------------------')
        print('---------------------------Network properties---------------------------')
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
        self.NetworkData()
        
        self.__writeData__()
        print('time taken:   ', time()-st)
        
    def __readNetworkFiles__(self):
        self.nThroats = int(open(self.cwd + '/' + str(self.title) + "_link1.dat").readline())
        arr1 = open(self.cwd + '/' + str(self.title) + "_node1.dat").readline().split()
        self.nPores = int(arr1[0])
        self.xDim, self.yDim, self.zDim = float(arr1[1]), float(arr1[2]), float(arr1[3])

        link1 = np.loadtxt(self.cwd + '/' + str(self.title) + "_link1.dat", skiprows=1)
        link2 = np.loadtxt(self.cwd + '/' + str(self.title) + "_link2.dat")
        Lines3 = open(self.cwd + '/' + str(self.title) + "_node1.dat").readlines()
        node2 = np.loadtxt(self.cwd + '/' + str(self.title) + "_node2.dat")

        self.poreList = np.arange(1, self.nPores+1)
        self.throatList = np.arange(1, self.nThroats+1)
        self.tList = self.throatList+self.nPores
        self.Area_ = self.yDim*self.zDim
        self.Lnetwork = self.xDim
        self.totElements = self.nPores+self.nThroats+2
        self.poreListS = np.arange(self.nPores+2)
        self.poreListS[-1] = -1
        self.elementLists = np.arange(1, self.totElements-1)
        self.elementListS = np.arange(self.totElements)
        self.elementListS[-1] = -1

        def formArray(pval, tval, dtype='float'):
            arr = np.zeros(self.totElements, dtype=dtype)
            arr[self.poreList] = pval
            arr[self.tList] = tval
            return arr
        
        def shapeFact(data):
            return np.minimum(data, np.sqrt(3)/36-0.00005)*(data <= self.bndG1) \
                + (1/16)*((data > self.bndG1) & (data < self.bndG2)) \
                + (1/(4*np.pi))*(data >= self.bndG2)
        
        def getDataP(x):
            x = x.split()
            a = 5+int(x[4])
            self.maxPoreCon = max(self.maxPoreCon, int(x[4]))
            self.PPConData.append(np.array([*map(int, x[5:a])], dtype='int'))
            self.PTConData.append(np.array([*map(int, x[a+2:])], dtype='int'))
            i = int(x[0])
            self.x_array[i] = x[1]
            self.y_array[i] = x[2]
            self.z_array[i] = x[3]
            self.connNum_array[i] = x[4]
            self.poreInletStat[i] = int(x[a])
            self.poreOutletStat[i] = int(x[a+1])

        self.P1array = link1[:,1].astype('int')
        self.P2array = link1[:,2].astype('int')
        self.LP1array = link2[:,3]
        self.LP2array = link2[:,4]
        self.LTarray = link2[:,5]
        self.lenTotarray = link1[:,5]

        self.Rarray = formArray(node2[:,2], link1[:,3])
        self.volarray = formArray(node2[:,1], link2[:,6])
        self.Garray = formArray(shapeFact(node2[:,3]), shapeFact(link1[:,4]))
        self.ClayVolarray = formArray(node2[:,4], link2[:,7])

        self.maxPoreCon = 0
        self.PPConData = [np.array([], dtype='int8')]
        self.PTConData = [np.array([], dtype='int32')]
        self.x_array = np.zeros(self.nPores+2)
        self.x_array[[0,-1]] = self.Lnetwork+1e-15, -1e-15
        self.y_array = np.zeros(self.nPores+2)
        self.y_array[[0, -1]] = self.yDim/2
        self.z_array = np.zeros(self.nPores+2)
        self.z_array[[0, -1]] = self.zDim/2
        self.connNum_array = np.zeros(self.nPores+2, dtype='uint32')
        self.poreInletStat = np.zeros(self.nPores+2, dtype='bool')
        self.poreOutletStat = np.zeros(self.nPores+2, dtype='bool')
        [*map(getDataP, Lines3[1:])]

    def NetworkData(self):
        self.PPConData = np.asarray(self.PPConData, dtype=object)
        self.PTConData = np.asarray(self.PTConData, dtype=object)+self.nPores
        self.PTConnections = np.array([np.pad(
            subarray, (0, self.maxPoreCon - len(subarray)), 
            constant_values=-5) for subarray in self.PTConData])
        self.TPConnections = np.zeros([self.nThroats+1, 2], dtype='int32')
        self.TPConnections[1:,0] = self.P1array
        self.TPConnections[1:,1] = self.P2array
        self.TPConnections[0] = -5
        self.xstart = self.calcBox[0]*self.xDim
        self.xend = self.calcBox[1]*self.xDim

        self.conPToIn = self.poreList[self.poreInletStat[1:-1]]
        self.conPToOut = self.poreList[self.poreOutletStat[1:-1]]
        self.conTToIn = self.tList[(self.P1array == self.pin_) | (self.P2array == self.pin_)]
        self.conTToOut = self.tList[(self.P1array==self.pout_) | (self.P2array == self.pout_)]
        self.conTToExit = self.tList[(self.P1array == self.pin_) | (self.P2array == self.pin_)| 
                                     (self.P1array == self.pout_) | (self.P2array == self.pout_)]

        self.isTriangle = (self.Garray <= self.bndG1)
        self.elemTriangle = self.elementLists[self.isTriangle[1:-1]]
        self.isCircle = (self.Garray >= self.bndG2)
        self.elemCircle = self.elementLists[self.isCircle[1:-1]]
        self.isSquare = (self.Garray > self.bndG1) & (self.Garray < self.bndG2)
        self.elemSquare = self.elementLists[self.isSquare[1:-1]]
        self.isPolygon = (self.Garray <= self.bndG2)

        self.__identifyConnectedElements__()
        self.__computeHalfAng__(self.elemTriangle)
        self.halfAnglesSq = np.array([pi/4, pi/4, pi/4, pi/4])
        self.cotBetaSq = 1/np.tan(self.halfAnglesSq)
        
        self.__isinsideBox__()
        self.__isOnBdr__()
        self.__modifyLength__()
        self.__computeDistToInlet__()
        self.__computeDistToExit__()
        self.__computeDistToBoundary__()
        self.__porosity__()

        self.nTriangles = self.elemTriangle.size
        self.nSquares = self.elemSquare.size
        self.nCircles = self.elemCircle.size

        self.elem = np.zeros(self.totElements, dtype='object')
        self.elem[[0, -1]] = Outlet(self), Inlet(self)
        self.elem[1:-1] = [Element(self, i) for i in range(1,self.totElements-1)]

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
        self.conTToInletBdr = self.tList[(self.isinsideBox[self.tList]) & (
            self.isOnInletBdr[self.P1array] | self.isOnInletBdr[self.P2array])]
        self._conTToInletBdr = self.conTToInletBdr-self.nPores
        self.conTToOutletBdr = self.tList[(self.isinsideBox[self.tList]) & (
            self.isOnOutletBdr[self.P1array] | self.isOnOutletBdr[self.P2array])]
        self._conTToOutletBdr = self.conTToOutletBdr-self.nPores
        
    
    def rand(self, a=1):
        return np.random.randint(0, self.RAND_MAX, size=a)/self.RAND_MAX
    
    def shuffle(self, obj):
        np.random.shuffle(obj)

    def choice(self, obj, size=1):
        return np.random.choice(obj, size)
    
    def __identifyConnectedElements__(self):
        ttt = self.tList[(self.P1array<=0)|(self.P2array<-0)]
        self.connected = np.zeros(self.totElements, dtype='bool')
        notdone = np.ones(self.totElements, dtype='bool')
        notdone[[-1,0]] = False
        self.connected[ttt] = True
        notdone[ttt] = False
        ttt -= self.nPores
        while True:
            try:
                ppp = self.TPConnections[ttt]
                ppp = ppp[notdone[ppp]]
                assert ppp.size>0
                notdone[ppp] = False
                self.connected[ppp] = True
                ttt = self.PTConnections[ppp]
                ttt = ttt[notdone[ttt]&(ttt>0)]
                assert ttt.size>0
                notdone[ttt] = False
                self.connected[ttt] = True
                ttt -= self.nPores
            except AssertionError:
                break
        self.isolated = ~self.connected

    def __computeDistToInlet__(self):
        self.distToInlet = np.zeros(self.totElements)
        self.distToInlet[self.poreListS] = self.x_array
        self.distToInlet[self.tList] = np.minimum(
            self.distToInlet[self.P1array], self.distToInlet[self.P2array])

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
        self.cotBetaTr = 1/np.tan(self.halfAnglesTr)

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
        delta_x = self.x_array[self.P2array]-self.x_array[self.P1array]
        delta_y = self.y_array[self.P2array]-self.y_array[self.P1array]
        delta_z = self.z_array[self.P2array]-self.z_array[self.P1array]
        self.avgP2Pdist = np.sqrt(pow(delta_x, 2) + pow(delta_y, 2) + pow(delta_z, 2)).mean()
        print('Average pore-to-pore distance = ', self.avgP2Pdist)
        print('Mean pore radius = ', np.mean(self.Rarray[self.poreList]))



class Element:
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
        self.neighbours = self.connT = obj.PTConData[self.index]
        #obj.PTConnections[self.index,:self.connT.size]=self.connT


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
        self.conTToExit = self.conToInlet | self.conToOutlet
        self.LP1mod = obj.LP1array_mod[self.index-1]
        self.LP2mod = obj.LP2array_mod[self.index-1]
        self.LTmod = obj.LTarray_mod[self.index-1]
        self.neighbours = np.array([self.P1, self.P2])

    def loadTriangleProp(self, obj):
        Element.iTr += 1
        self.indexTr = self.iTr
        self.halfAng = obj.halfAnglesTr[self.iTr]
        
    def loadSquareProp(self, obj):
        Element.iSq += 1
        self.indexSq = self.iSq
        self.halfAng = np.array([pi/4, pi/4, pi/4, pi/4])

    def loadCircleProp(self, obj):
        Element.iCi += 1
        self.indexCi = self.iCi

class Inlet:
    def __init__(self, parent):
        self.index = -1
        self.x = 0.0
        self.indexOren = -1
        self.connected = True
        self.isinsideBox = False
        self.neighbours = parent.conTToIn


class Outlet:
    def __init__(self, parent):
        self.index = 0
        self.x = parent.Lnetwork
        self.indexOren = 0
        self.connected = False
        self.isinsideBox = False
        self.neighbours = parent.conTToOut


#network = Network('./data/input_pnflow_bent.dat')
