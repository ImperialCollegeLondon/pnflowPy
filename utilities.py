import numpy as np
from scipy.sparse import csr_matrix
import warnings
from solver import Solver
from functools import reduce
from numba import njit, prange

class Computations():
    def __init__(self, obj):
        obj.toInlet = np.zeros(obj.totElements, dtype='bool')
        obj.toInlet[obj.conTToIn] = True
        obj.toInBdr = obj.toInlet.copy()
        obj.toInBdr[obj.conTToInletBdr] = True
        obj.toOutlet = np.zeros(obj.totElements, dtype='bool')
        obj.toOutlet[obj.conTToOut] = True
        obj.toOutBdr = obj.toOutlet.copy()
        obj.toOutBdr[obj.conTToOutletBdr] = True


def matrixSolver(Amatrix, Cmatrix) -> np.array:
    return Solver(Amatrix, Cmatrix).solve()


def computegL(self, g) -> np.array:
    return compute_gL_numba(
        self.P1array, self.P2array, self.tList,
        self.LP1array_mod, self.LP2array_mod, self.LTarray_mod,
        g, self.nThroats
    )


@njit(parallel=True)
def compute_gL_numba(P1array, P2array, tList, LP1, LP2, LT, g, nThroats):
    gL = np.zeros(nThroats)
    for i in prange(nThroats):
        gT  = g[tList[i]]
        gP1 = g[P1array[i]]
        gP2 = g[P2array[i]]

        if (gT > 0.0) and ((gP1>0) or (P1array[i]<1)) and ((gP2>0) or (P2array[i]<1)):
            if (gP1 > 0) and (gP2 > 0):
                gL[i] = 1.0 / (LP1[i]/gP1 + LT[i]/gT + LP2[i]/gP2)
            elif (gP1 == 0) and (gP2 > 0) and (LP2[i] > 0):
                gL[i] = 1.0 / (LT[i]/gT + LP2[i]/gP2)
            elif (gP1 > 0) and (gP2 == 0) and (LP1[i] > 0):
                gL[i] = 1.0 / (LT[i]/gT + LP1[i]/gP1)
    return gL



def check_Trapping_Clustering(self, arr, notdone, fluid, Pc, updateCluster=False,
                                updateConnectivity=False, updatePcClustConToInlet=True):
    i = 0
    members = np.zeros(self.totElements, dtype=bool)
    arrDict = {}
    connectedCluster = []
    TValid = self.TValid[notdone[self.TValid]]
    TPValid = self.TPValid[notdone[self.TValid]]
    mem = np.zeros(self.totElements, dtype=bool)
    
    while arr.size:
        i += 1
        ii = arr[0]
        done = np.zeros(self.totElements, dtype=bool)
        done[ii] = True
        notdone[ii] = False
        trappedStatus, connStatus = True, False

        doPore = (ii<=self.nPores)
        while True:
            if doPore:
                ii_next = TValid[done[TPValid]]
                doPore = False
            else:
                ii_next = TPValid[done[TValid]]
                doPore = True

            ii_next = ii_next[notdone[ii_next]]
            if ii_next.size == 0:
                break

            done[ii_next] = True
            notdone[ii_next] = False
        
        TValid, TPValid = TValid[notdone[TValid]], TPValid[notdone[TValid]]
        trappedStatus = not (self.toInlet[done].any() or self.toOutlet[done].any())
        if self.toInBdr[done].any() and self.toOutBdr[done].any():
            connStatus = True
            connectedCluster.append(i)
            mem[done] = True

        arrDict[i] = {'members': done, 'connStatus': connStatus, 'trappedStatus': trappedStatus}
        arr = arr[notdone[arr]]
        members[done] = True

    try:
        if fluid == 0:
            cluster_ID, cluster, trapped = self.clusterW_ID, self.clusterW, self.trappedW
        else:
            cluster_ID, cluster, trapped = self.clusterNW_ID, self.clusterNW, self.trappedNW

        if not updateCluster:
            isConnected = mem.any()
            cluster.connected[0] = isConnected
            cluster.clustConToExit[0] = isConnected
            cluster.trappedStatus[0] = not isConnected
            ids = cluster_ID[mem][cluster_ID[mem] >= 0]
            if ids.size>0 and not (ids==0).all():
                mem1 = self.elementListS[mem][cluster_ID[mem] >= 0]
                mem1 = mem1[ids != 0]
                ids = ids[ids != 0]
                cluster_ID[mem1] = 0
                cluster.members[:, mem1] = False
                cluster.members[0][mem1] = True
                trapped[mem1] = False
                availClust = ids[~cluster.members[ids].any(axis=1)]
                cluster.availableID.update(availClust)
        else:
            members = self.elementListS[members]
            cluster.clustering(members, arrDict, Pc, cluster_ID, trapped, 
                               updatePcClustConToInlet)

    except AttributeError:
        pass

    if not updateConnectivity:
        return
    else:
        return mem
    

@njit
def build_Amatrix_data(
    throatList, P1array, P2array,
    isOnInletBdr, isOnOutletBdr, gL, mList):

    row = np.empty(4*throatList.size, dtype=np.int32)
    col = np.empty_like(row)
    data = np.empty(4*throatList.size, dtype=np.float64)
    count = 0

    c = np.sum(mList >= 0)  # number of active pores
    Cmatrix = np.zeros(c)

    for t in throatList:
        cond = gL[t]

        if cond == 0.0:
            continue

        P1_t, P2_t = P1array[t], P2array[t]
        P1, P2 = mList[P1_t], mList[P2_t]
        
        if (P1 >= 0) and (P2 >= 0):
            # internal connection
            row[count:count+4] = [P1, P2, P1, P2]
            col[count:count+4] = [P2, P1, P1, P2]
            data[count:count+4] = [-cond, -cond, cond, cond]
            count += 4

        elif (P1 >= 0) and (isOnInletBdr[P2_t]):
            # connection to inlet boundary
            row[count] = P1
            col[count] = P1
            data[count] = cond
            count += 1
            Cmatrix[P1] += cond

        elif (P2 >= 0) and (isOnInletBdr[P1_t]):
            # connection to inlet boundary
            row[count] = P2
            col[count] = P2
            data[count] = cond
            count += 1
            Cmatrix[P2] += cond

        elif (P1 >= 0) and (isOnOutletBdr[P2_t]):
            # connection to outlet boundary
            row[count] = P1
            col[count] = P1
            data[count] = cond
            count += 1

        elif (P2 >= 0) and (isOnOutletBdr[P1_t]):
            # connection to outlet boundary
            row[count] = P2
            col[count] = P2
            data[count] = cond
            count += 1

    return row[:count], col[:count], data[:count], Cmatrix


def __getValue__(self, arrr, gL):
    indP = self.poreList[arrr[self.poreList]]
    c = indP.size
    mList = -np.ones(self.nPores+2, dtype=np.int32)
    mList[indP] = np.arange(c)

    throatList = self.throatList[arrr[self.tList]]-1

    row, col, data, Cmatrix = build_Amatrix_data(
        throatList, self.P1array, self.P2array,
        self.isOnInletBdr, self.isOnOutletBdr, gL,
        mList
    )

    Amatrix = csr_matrix((data, (row, col)), shape=(c, c), dtype=float)

    return Amatrix, Cmatrix


@njit
def Saturation(self, AreaWP, AreaSP):
    satWP = AreaWP/AreaSP
    num = (satWP[self.isinsideBox]*self.volarray[self.isinsideBox]).sum()
    return num/self.totVoidVolume


def computeFlowrate(self, gL, fluid, Pc, vector=False):
    conn = self.connW.copy() if fluid==0 else self.connNW.copy()
    conTToIn = self.conTToIn.copy()
    arrr = np.zeros(self.totElements, dtype='bool')
    active = (gL > 0.0)
    arrr[self.P1array[active]] = True
    arrr[self.P2array[active]] = True
    arrr[self.tList[active]] = True
    arrr = (arrr & self.connected)
    conn = check_Trapping_Clustering(
        self, conTToIn[arrr[conTToIn]], arrr.copy(), fluid, Pc, updateConnectivity=True)
    if fluid == 0: self.connW = conn
    else: self.connNW = conn
    conn = conn & self.isinsideBox
    Amatrix, Cmatrix = __getValue__(self, conn, gL)
        
    pres = np.zeros(self.nPores+2)
    if conn.any():
        pres[conn[self.poreListS]] = matrixSolver(Amatrix, Cmatrix)
        pres[self.isOnInletBdr[self.poreListS]] = 1.0       
        qp = compute_qp(self.P1array, self.P2array, gL, pres)
    else:
        qp = np.zeros(self.nThroats)
    
    if not vector:
        conTToInletBdr = self._conTToInletBdr[conn[self.conTToInletBdr]]
        conTToOutletBdr = self._conTToOutletBdr[conn[self.conTToOutletBdr]]
        qinto = qp[conTToInletBdr-1].sum()
        qout = qp[conTToOutletBdr-1].sum()
        if np.isclose(qinto, qout, atol=1e-30):
            qout = (qinto+qout)/2
        return qout
    else:
        return (qp, (pres[self.P2array]<=pres[self.P1array]))
    

@njit
def compute_qp(P1array, P2array, gL, pres):
    n = len(gL)
    delP = np.empty(n)
    qp = np.empty(n)
    for i in range(n):
        delP[i] = abs(pres[P1array[i]] - pres[P2array[i]])
        qp[i] = gL[i] * delP[i]
    return qp
        

def computePerm(self, Pc):
    gwL = computegL(self, self.gWPhase)
    self.qW = self.qW = computeFlowrate(self, gwL, 0, Pc)
    self.krw = self.krw = self.qW/self.qwSPhase
    if self.fluid[self.conTToOutletBdr].sum() > 0:
        gnwL = computegL(self, self.gNWPhase)
        self.qNW = self.qNW = computeFlowrate(self, gnwL, 1, Pc)
        self.krnw = self.krnw = self.qNW/self.qnwSPhase
    else:
        self.qNW, self.krnw = 0.0, 0.0
    
    self.fw = self.qW/(self.qW + self.qNW)


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
    conAng = weibull(self)        

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
    thetaRecAng, thetaAdvAng = setContactAngles(self, contactAng)

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


@njit(parallel=True)
def create_films_numba(
    arr, arrr, halfAng, Pc, m_exists, m_inited, m_initOrMaxPcHist, m_initOrMinApexDistHist, advPc, recPc, m_initedApexDist, is_oil_inj, sigma, thetaAdvAng, thetaRecAng):

    n = arr.size
    nCorners = m_exists.shape[1]
    is_square = (nCorners == 4)
    for i in prange(n):
        idx = arr[i]

        if not arrr[i]:
            continue

        conAng = thetaRecAng[idx] if is_oil_inj else thetaAdvAng[idx]
        Pc_val = Pc[i]
        sigma_over_Pc = sigma / Pc_val
        for j in range(nCorners):
            if m_exists[i, j] and m_inited[i, j]:
                continue

            halfAng_ij = halfAng[0, j] if is_square else halfAng[i, j]

            if conAng >= (np.pi / 2.0 - halfAng_ij):
                continue

            m_exists[i, j] = True
            
            cosTerm = np.cos(conAng + halfAng_ij)
            sinTerm = np.sin(halfAng_ij)
            initedApexDist = max(sigma_over_Pc * cosTerm / sinTerm, 0.0)
            m_initedApexDist[i, j] = initedApexDist

            if initedApexDist != 0.0:
                advPc[i, j] = sigma * np.cos(min(np.pi, thetaAdvAng[idx]) + halfAng_ij) / (initedApexDist * sinTerm)
                recPc[i, j] = sigma * np.cos(min(np.pi, thetaRecAng[idx]) + halfAng_ij) / (initedApexDist * sinTerm)
            else:
                advPc[i, j] = 0.0
                recPc[i, j] = 0.0

            m_inited[i, j] = True

            if Pc_val > m_initOrMaxPcHist[i, j]:
                m_initOrMinApexDistHist[i, j] = initedApexDist
                m_initOrMaxPcHist[i, j] = Pc_val



def createFilms(self, arr, arrr, halfAng, Pc, m_exists,
                m_inited, m_initOrMaxPcHist, m_initOrMinApexDistHist, advPc,
                recPc, m_initedApexDist):
    create_films_numba(
        arr, arrr, halfAng, Pc,
        m_exists, m_inited, m_initOrMaxPcHist, m_initOrMinApexDistHist,
        advPc, recPc, m_initedApexDist, self.is_oil_inj,
        self.sigma, self.thetaAdvAng, self.thetaRecAng
    )


@njit
def corner_apex_numba(
    arr, arrr, halfAng, Pc, _conAng, m_exists,
    m_initOrMaxPcHist, m_initOrMinApexDistHist, advPc,
    recPc, apexDist, initedApexDist, trappedW, trappedNW, clusterW_pc, clusterNW_pc, 
    clusterW_ID, clusterNW_ID, sigma, thetaAdvAng, thetaRecAng, 
    delta, overidetrapping, MOLECULAR_LENGTH, is_square):

    n = arr.size
    nCorners = m_exists.shape[1]
    if is_square is None: is_square = (nCorners == 4)
    conAng = np.empty(m_exists.shape, dtype=np.float64)

    for i in prange(n):
        
        idx = arr[i]
        if not arrr[i]:
            continue

        Pc_val = Pc[0] if Pc.size==1 else Pc[i]
            
        sigma_over_Pc = sigma / Pc_val
        halfAng_i = halfAng[0] if is_square else halfAng[i]

        if not overidetrapping:
            apexDist[i] = initedApexDist[i]
            trapped = False
            if trappedW[idx]:
                cidx = clusterW_ID[idx]
                trappedPc = clusterW_pc[cidx]
                trapped = True
            elif trappedNW[idx]:
                cidx = clusterNW_ID[idx]
                trappedPc = clusterNW_pc[cidx]
                trapped = True

            if trapped:
                for j in range(nCorners):
                    apexDist[i, j] = initedApexDist[i, j]
                    part = trappedPc * initedApexDist[i, j] * np.sin(halfAng_i[j]) / sigma
                    part = min(0.999999, max(-0.999999, part))
                    conAng[i, j] = max(min(np.arccos(part) - halfAng_i[j], np.pi), 0.0)

        for j in range(nCorners):
            halfAng_ij = halfAng_i[j]
            sinHalfAng_ij = np.sin(halfAng_ij)
            initedApexDist_ij = initedApexDist[i, j]
            conAng_ij = _conAng[idx]

            # cond0
            if not m_exists[i, j]:
                if overidetrapping:
                    apexDist_ij = MOLECULAR_LENGTH

            # cond1
            elif (advPc[i, j] - delta <= Pc_val) and (Pc_val <= recPc[i, j] + delta):
                part = max(
                    min(initedApexDist_ij * sinHalfAng_ij / sigma_over_Pc, 0.999999), -0.999999)
                conAng_ij = max(min(np.arccos(part) - halfAng_ij, np.pi), 0.0)
                apexDist_ij = initedApexDist_ij

            # cond2
            elif Pc_val < advPc[i, j]:
                conAng_ij = thetaAdvAng[idx]
                apexDist_ij = sigma_over_Pc * np.cos(conAng_ij+halfAng_ij)/sinHalfAng_ij

                if apexDist_ij < initedApexDist_ij:
                    part = max(
                        min(initedApexDist_ij * sinHalfAng_ij / sigma_over_Pc, 0.999999), -0.999999)
                    conAng_ij = max(min(np.arccos(part) - halfAng_ij, np.pi), 0.0)
                    apexDist_ij = initedApexDist_ij
            
            # cond3
            elif Pc_val > m_initOrMaxPcHist[i, j]:
                conAng_ij = min(np.pi, thetaRecAng[idx])
                apexDist_ij = sigma_over_Pc*np.cos(conAng_ij+halfAng_ij)/sinHalfAng_ij

            # cond4
            elif Pc_val > recPc[i, j]:
                conAng_ij = thetaRecAng[idx]
                apexDist_ij = sigma_over_Pc*np.cos(conAng_ij+halfAng_ij)/sinHalfAng_ij
                m_initOrMinApexDistHist_ij = m_initOrMinApexDistHist[i, j]

                if apexDist_ij > initedApexDist_ij:
                    part = max(
                        min(initedApexDist_ij * sinHalfAng_ij / sigma_over_Pc, 0.999999), -0.999999)
                    conAng_ij = max(min(np.arccos(part) - halfAng_ij, np.pi), 0.0)
                    apexDist_ij = initedApexDist_ij

                elif apexDist_ij < m_initOrMinApexDistHist_ij:
                    part = max(
                        min(m_initOrMinApexDistHist_ij * sinHalfAng_ij / sigma_over_Pc, 0.999999), -0.999999)
                    conAng_ij = max(min(np.arccos(part) - halfAng_ij, np.pi), 0.0)
                    apexDist_ij = m_initOrMinApexDistHist_ij

            # cond5
            else:
                apexDist_ij = sigma_over_Pc*np.cos(conAng_ij+halfAng_ij)/sinHalfAng_ij

            conAng[i, j] = conAng_ij
            apexDist[i, j] = apexDist_ij

    return conAng, apexDist
    

def cornerApex(self, arr, arrr, halfAng, Pc, conAng, m_exists,
            m_initOrMaxPcHist, m_initOrMinApexDistHist, advPc,
            recPc, apexDist, initedApexDist, accurat=False,
            overidetrapping=False, is_square=None):
    
    delta = 0.0 if accurat else self._delta
    Pc = np.atleast_1d(Pc)
    #try:
    return corner_apex_numba(
        arr, arrr, halfAng, Pc, conAng, m_exists,
        m_initOrMaxPcHist, m_initOrMinApexDistHist, advPc,
        recPc, apexDist, initedApexDist, self.trappedW, self.trappedNW, 
        self.clusterW.pc, self.clusterNW.pc, self.clusterW_ID, self.clusterNW_ID, 
        self.sigma, self.thetaAdvAng, self.thetaRecAng, 
        delta,  overidetrapping, self.MOLECULAR_LENGTH, is_square=is_square)                                                




def initCornerApex(self, arr, arrr, halfAng, m_exists, m_inited,
                    recPc, advPc, m_initedApexDist, trapped):

    cond =  (m_exists & (arrr&~trapped[arr])[:, np.newaxis])
    if cond.sum()>0:
        m_inited[cond] = True
        Pc =np.zeros_like(m_initedApexDist)
        Pc[cond] = self.sigma*np.cos(np.minimum(
            np.pi, ((self.thetaRecAng[arr, np.newaxis]+halfAng)*cond)[cond]))/(
                m_initedApexDist*np.sin(halfAng))[cond]

        recPc[cond & (recPc < Pc)] = Pc[cond & (recPc < Pc)]
        advPc[cond] = self.sigma*np.cos(np.minimum(
            np.pi, ((self.thetaAdvAng[arr, np.newaxis]+halfAng)*cond)[cond]))/(
                (m_initedApexDist*np.sin(halfAng))[cond])


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


def updateObj(self, obj):
    selfDict = self.__dict__
    objDict = obj.__dict__

    for key, new_val in objDict.items():
        if key not in selfDict:
            setattr(self, key, new_val)
            continue

        old_val = selfDict[key]

        try:
            if isinstance(old_val, np.ndarray) and isinstance(new_val, np.ndarray):
                # Compare arrays by shape and content
                if old_val.shape != new_val.shape or not np.all(old_val == new_val):
                    # If both have bases and are arrays → update the base
                    if (isinstance(old_val.base, np.ndarray) and 
                        isinstance(new_val.base, np.ndarray)):
                        old_val.base[:] = new_val
                    else:
                        old_val[:] = new_val
            else:
                # If not equal, update attribute
                if old_val != new_val:
                    setattr(self, key, new_val)

        except (ValueError, TypeError, AttributeError):
            # Fallback to setattr if any issue in comparison or assignment
            setattr(self, key, new_val)




        
