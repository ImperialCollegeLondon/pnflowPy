import numpy as np
from scipy.sparse import csr_matrix
import warnings
from solver import Solver
from functools import reduce

class Computations():

    def __init__(self, obj):
        self.obj = obj
        self.toInlet = np.zeros(self.totElements, dtype='bool')
        self.toInlet[self.conTToIn] = True
        self.toInBdr = self.toInlet.copy()
        self.toInBdr[self.conTToInletBdr] = True
        self.toOutlet = np.zeros(self.totElements, dtype='bool')
        self.toOutlet[self.conTToOut] = True
        self.toOutBdr = self.toOutlet.copy()
        self.toOutBdr[self.conTToOutletBdr] = True
        
    
    def __getattr__(self, name):
        return getattr(self.obj, name)
    
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
 
    def check_Trapping_Clustering(self, arr, notdone, fluid, Pc, updateCluster=False, 
                                  updateConnectivity=False):
        i = 0
        arrDict = {}
        arrr = np.zeros(self.totElements, dtype='bool')
        connectedCluster = []
        while True:
            try:
                i += 1
                ii = arr[0]
                done = np.zeros(self.totElements, dtype='bool')
                notdone[ii] = False
                done[ii] = True
                arrr[ii] = True
                trappedStatus, connStatus = True, False
                try:
                    assert ii<=self.nPores
                    arrr[self.PTConData[ii]] = True
                    ii = self.elementListS[(arrr&notdone)]
                    try:
                        assert ii.size > 0
                        done[ii] = True
                        notdone[ii] = False
                    except AssertionError:
                        cond1, cond2 = self.toInBdr[done].any(), self.toOutBdr[done].any()
                        assert cond1 or cond2
                        trappedStatus = False
                        assert cond1 and cond2
                        connStatus = True
                        connectedCluster.append(i)
                        arrDict[i] = {'members': done, 'connStatus': connStatus, 
                                        'trappedStatus': trappedStatus}
                        continue
                except AssertionError:
                    pass
                while True:
                    try:
                        ii = ii-self.nPores-1
                        p1, p2 = self.P1array[ii], self.P2array[ii]
                        arrr[p1[p1>0]] = True
                        arrr[p2[p2>0]] = True
                        ii = self.elementListS[(arrr&notdone)]
                        assert ii.size > 0
                        notdone[ii] = False
                        done[ii] = True
                        tt = self.PTConnections[ii]
                        arrr[tt[tt>0]] = True
                        ii = self.elementListS[(arrr&notdone)]
                        assert ii.size > 0
                        notdone[ii] = False
                        done[ii] = True
                    except AssertionError:
                        try:
                            assert self.toInlet[done].any() or self.toOutlet[done].any()
                            trappedStatus = False
                        except AssertionError:
                            pass
                        try:
                            assert self.toInBdr[done].any() and self.toOutBdr[done].any()
                            connStatus = True
                            connectedCluster.append(i)
                        except AssertionError:
                            pass
                        arrDict[i] = {'members': done, 'connStatus': connStatus, 
                                        'trappedStatus': trappedStatus}
                        arr = arr[notdone[arr]]
                        break
            except IndexError:
                break
    
        try:
            lenClust = len(connectedCluster)
            if lenClust==1:
                mem = arrDict[connectedCluster[0]]['members']
            elif lenClust==0:
                mem = np.zeros(self.totElements, dtype=bool)
            else:
                mem = reduce(np.logical_or, (arrDict[k]['members'] for k in connectedCluster))

            try:
                assert fluid==0
                cluster_ID, cluster, trapped = self.clusterW_ID, self.clusterW, self.trappedW
            except AssertionError:
                cluster_ID, cluster, trapped = self.clusterNW_ID, self.clusterNW, self.trappedNW

            assert not updateCluster
            cond = mem.any()
            cluster.connected[0] = cond
            cluster.clustConToInlet[0] = cond
            cluster.trappedStatus[0] = not cond
            try:
                ids = cluster_ID[mem][cluster_ID[mem]>=0]
                assert not (ids==0).all()
                mem1 = self.elementListS[mem][cluster_ID[mem]>=0]
                mem1 = mem1[ids!=0]
                ids = ids[ids!=0]
                ''' ensure connected cluster is cluster 0 '''
                cluster_ID[mem1] = 0
                cluster.members[:, mem1] = False
                cluster.members[0][mem1] = True
                trapped[mem1] = False
                ''' check which clusters are truly empty '''
                availClust = ids[~cluster.members[ids].any(axis=1)]
                cluster.availableID.update(availClust)
            except AssertionError:
                pass
        except AttributeError:
            pass
        except AssertionError:
            cluster.clustering(arrDict, Pc, cluster_ID, cluster, trapped)

        try:
            assert not updateConnectivity
            return
        except AssertionError:
            return mem

        
    def __getValue__(self, arrr, gL):
        row, col, data = [], [], []
        indP = self.poreList[arrr[self.poreList]]
        c = indP.size
        mList = -np.ones(self.nPores+2, dtype='int')
        mList[indP] = np.arange(c)

        arrrT = arrr[self.tList]
        arrrP1, arrrP2 = arrr[self.P1array], arrr[self.P2array]

        ''' throats within the calcBox '''
        cond1 = arrrT & arrrP1 & arrrP2
        arrr[self.tList[cond1]] = False
        t_1 = self.throatList[cond1]
        P1_1, P2_1 = mList[self.P1array[cond1]], mList[self.P2array[cond1]]
        cond_1 = gL[t_1-1]

        ''' throats connected to the inletBdr '''
        cond2a = arrrT & (self.isOnInletBdr[self.P1array]&arrrP2)
        cond2b = arrrT & (self.isOnInletBdr[self.P2array]&arrrP1)
        indP2 = np.concatenate((self.P2array[cond2a], self.P1array[cond2b]))
        P_2 = mList[indP2]
        t_2 = np.concatenate((self.throatList[cond2a], self.throatList[cond2b]))
        cond_2 = gL[t_2-1]
        Cmatrix = np.bincount(P_2, cond_2, c) #set up the Cmatrix

        ''' throats connected to the outletBdr '''
        cond3a = arrrT & (self.isOnOutletBdr[self.P1array]&arrrP2)
        cond3b = arrrT & (self.isOnOutletBdr[self.P2array]&arrrP1)
        indP3 = np.concatenate((self.P2array[cond3a], self.P1array[cond3b]))
        P_3 = mList[indP3]
        t_3 = np.concatenate((self.throatList[cond3a], self.throatList[cond3b]))
        cond_3 = gL[t_3-1]

        ''' set up the Amatrix '''
        row = np.concatenate((P1_1, P2_1, P1_1, P2_1, P_2, P_3))
        col = np.concatenate((P2_1, P1_1, P1_1, P2_1, P_2, P_3))
        data = np.concatenate((-cond_1, -cond_1, cond_1, cond_1, cond_2, cond_3))
        Amatrix = csr_matrix((data, (row, col)), shape=(c, c), dtype=float)

        return Amatrix, Cmatrix
    
    def Saturation(self, AreaWP, AreaSP):
        satWP = AreaWP/AreaSP
        num = (satWP[self.isinsideBox]*self.volarray[self.isinsideBox]).sum()
        return num/self.totVoidVolume
    

    def computeFlowrate(self, gL, fluid, Pc):
        conn = self.connW.copy() if fluid==0 else self.connNW.copy()
        conTToIn = self.conTToIn.copy()
        arrr = np.zeros(self.totElements, dtype='bool')    
        arrr[self.P1array[(gL > 0.0)]] = True
        arrr[self.P2array[(gL > 0.0)]] = True
        arrr[self.tList[(gL > 0.0)]] = True
        arrr = (arrr & self.connected)
        conn = self.check_Trapping_Clustering(
            conTToIn[arrr[conTToIn]], arrr.copy(), fluid, Pc, updateConnectivity=True)
        try:
            assert fluid==0
            self.obj.connW = conn
        except AssertionError:
            self.obj.connNW = conn
        conn = conn & self.isinsideBox
        Amatrix, Cmatrix = self.__getValue__(conn, gL)
           
        pres = np.zeros(self.nPores+2)
        pres[conn[self.poreListS]] = self.matrixSolver(Amatrix, Cmatrix)
        pres[self.isOnInletBdr[self.poreListS]] = 1.0       
        delP = np.abs(pres[self.P1array] - pres[self.P2array])
        qp = gL*delP
        
        try:
            conTToInletBdr = self._conTToInletBdr[conn[self.conTToInletBdr]]
            conTToOutletBdr = self._conTToOutletBdr[conn[self.conTToOutletBdr]]
            qinto = qp[conTToInletBdr-1].sum()
            qout = qp[conTToOutletBdr-1].sum()
            assert np.isclose(qinto, qout, atol=1e-30)
            qout = (qinto+qout)/2
        except AssertionError:
            pass

        return qout

    
    def computePerm(self, Pc):
        gwL = self.computegL(self.gWPhase)
        self.obj.qW = self.qW = self.computeFlowrate(gwL, 0, Pc)
        self.obj.krw = self.krw = self.qW/self.qwSPhase
       
        try:
            assert self.fluid[self.conTToOutletBdr].sum() > 0
            gnwL = self.computegL(self.gNWPhase)
            self.obj.qNW = self.qNW = self.computeFlowrate(gnwL, 1, Pc)
            self.obj.krnw = self.krnw = self.qNW/self.qnwSPhase
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
        ''' compute the distribution of contact angles in the network '''
        contactAng = np.zeros(self.totElements)
        conAng = self.weibull()        

        arr = np.array([conAng[self.poreList-1].mean(), conAng[self.poreList-1].std(),
            conAng[self.poreList-1].min(), conAng[self.poreList-1].max()])*180/np.pi
        print('contact Angles (only pores): mean: {}, std: {}, min: {}, max: {}'.format(
            np.round(arr[0],2), np.round(arr[1],2), np.round(arr[2],2), np.round(arr[3],2)))

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
        
        arr = np.array([contactAng.mean(), contactAng.std(), contactAng.min(), contactAng.max()]
                     )*180/np.pi
        print('contact Angles (all elements): mean: {}, std: {}, min: {}, max: {}'.format(
            np.round(arr[0],2), np.round(arr[1],2), np.round(arr[2],2), np.round(arr[3],2)))
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

        ''' update the apex dist and contact angle '''
        apexDist[~m_exists & arrr] = self.MOLECULAR_LENGTH
        delta = 0.0 if accurat else self._delta
        conAng = np.ones(m_exists.shape)*conAng[arr]

        try:
            assert not overidetrapping
            apexDist[:, arrr] = initedApexDist[:, arrr]
            assert self.trappedW[arr[arrr]].sum()+self.trappedNW[arr[arrr]].sum()>0
            arrr1 = arrr & self.trappedW[arr]
            arrr2 = arrr & ~self.trappedW[arr] & self.trappedNW[arr]
            trappedPc = np.zeros(arr.size)
            trappedPc[arrr1] = np.array(self.clusterW.pc)[self.clusterW_ID[arr[arrr1]]]
            trappedPc[arrr2] = np.array(self.clusterNW.pc)[self.clusterNW_ID[arr[arrr2]]]
            cond = (self.trappedW[arr]|self.trappedNW[arr])
            part = np.clip((trappedPc*initedApexDist*np.sin(halfAng)).T[cond]/self.sigma, 
                           -0.999999, 0.999999)
            try:
                conAng.T[cond] = np.clip(np.arccos(part)-halfAng.T[cond], 0.0, np.pi)
            except IndexError:
                conAng.T[cond] = np.clip(np.arccos(part)-halfAng.T, 0.0, np.pi)
        except AssertionError:
            pass
        
        ''' condition 1 '''
        cond1a = m_exists & (advPc-delta <= Pc) & (Pc <= recPc+delta)
        cond1 = cond1a & arrr
        try:
            assert cond1.sum() > 0
            part = np.clip(Pc*initedApexDist*np.sin(halfAng)/self.sigma, -0.999999, 0.999999)
            hingAng = np.clip((np.arccos(part)-halfAng)[cond1], -self._delta, np.pi+self._delta)
            conAng[cond1] = np.clip(hingAng, 0.0, np.pi)
            apexDist[cond1] = initedApexDist[cond1]
        except AssertionError:
            pass

        ''' condition 2 '''
        cond2a = m_exists & ~cond1a & (Pc < advPc)
        cond2 = cond2a & arrr
        try:
            assert cond2.sum() > 0
            conAng[cond2] = (self.thetaAdvAng[arr]*cond2)[cond2]
            apexDist[cond2] = (self.sigma/Pc*np.cos(
                conAng+halfAng)/np.sin(halfAng))[cond2]

            cond2b = (apexDist < initedApexDist) & cond2
            assert cond2b.sum() > 0
            part = np.clip(Pc*initedApexDist*np.sin(halfAng)/self.sigma, -0.999999, 0.999999)
            hingAng = np.clip((np.arccos(part)-halfAng)[cond2b], 0.0, np.pi)
            
            conAng[cond2b] = hingAng
            apexDist[cond2b] = initedApexDist[cond2b]
        except AssertionError:
            pass

        ''' condition 3 '''
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

        ''' condition 4 '''
        cond4a = m_exists & ~cond1 & ~cond2a & ~cond3a & (Pc > recPc)
        cond4 = (cond4a*arrr)
        try:
            assert cond4.sum() > 0
            conAng[cond4] = (self.thetaRecAng[arr]*cond4)[cond4]
            apexDist[cond4] = (self.sigma/Pc*np.cos(conAng+halfAng)/np.sin(halfAng))[cond4]
            cond4b = cond4 & (apexDist > initedApexDist)
            cond4c = cond4 & (~cond4b) & (apexDist < m_initOrMinApexDistHist)
            try:
                assert cond4b.sum() > 0
                part = np.clip(Pc*initedApexDist*np.sin(halfAng)/self.sigma, -0.999999, 0.999999)
                hingAng = np.clip((np.arccos(part)-halfAng)[cond4b], 0.0, np.pi)
                conAng[cond4b] = hingAng
                apexDist[cond4b] = initedApexDist[cond4b]
            except AssertionError:
                pass
            try:
                assert cond4c.sum() > 0
                part = np.clip(Pc*m_initOrMinApexDistHist*np.sin(halfAng)/self.sigma, 
                               -0.999999, 0.999999)
                hingAng = np.clip((np.arccos(part)-halfAng)[cond4c], 0.0, np.pi)
                conAng[cond4c] = hingAng
                apexDist[cond4c] = m_initOrMinApexDistHist[cond4c]
            except AssertionError:
                pass
        except AssertionError:
            pass

        ''' condition 5 '''
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
        ''' -- obtain corner conductance -- '''
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
        print('Sw: %10.6g  \tqW: %8.6e  \tkrw: %12.6g  \tqNW: %8.6e  \tkrnw:\
              %12.6g  \tPc: %8.6g\t %8.0f invasions' % (
              self.satW, self.qW, self.krw, self.qNW, self.krnw,
              Pc, self.totNumFill, ))
            
        if self.writeData:
            result_str+="\n%.6g,%.6e,%.6g,%.6e,%.6g,%.6g,%.0f" % (
                self.satW, self.qW, self.krw, self.qNW, self.krnw,
                Pc, self.totNumFill, )
        
        return result_str


        
