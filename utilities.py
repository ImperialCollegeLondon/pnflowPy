import numpy as np
from scipy.sparse import csr_matrix
import warnings
from solver import Solver
from functools import reduce
from numba import njit, prange
import os

import utilities_cython as do


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

@njit(parallel=True, cache=True)
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
    cond = notdone[self.TValid]
    TValid = self.TValid[cond]
    TPValid = self.TPValid[cond]
    mem0 = np.zeros(self.totElements, dtype=bool)
    
    while arr.size:
        i += 1
        ii = arr[0]
        
        done, TValid, TPValid = check_Trapping_Clustering_numba(
            ii, TValid, TPValid, notdone, self.nPores, self.totElements)
        _done = np.flatnonzero(done)
        trappedStatus = not (self.toInlet[_done].any() or self.toOutlet[_done].any())
        if self.toInBdr[_done].any() and self.toOutBdr[_done].any():
            connStatus = True
            connectedCluster.append(i)
            mem0[_done] = True
        else:
            connStatus = False

        arrDict[i] = {'members': done, 'connStatus': connStatus, 'trappedStatus': trappedStatus}
        arr = arr[notdone[arr]]
        members[_done] = True

    try:
        if fluid == 0:
            cluster_ID, cluster, trapped = self.clusterW_ID, self.clusterW, self.trappedW
        else:
            cluster_ID, cluster, trapped = self.clusterNW_ID, self.clusterNW, self.trappedNW

        _mem0 = np.flatnonzero(mem0)
        if not updateCluster:
            isConnected = _mem0.any()
            cluster.connected[0] = isConnected
            cluster.clustConToExit[0] = isConnected
            cluster.trappedStatus[0] = not isConnected
            clustID = cluster_ID[_mem0]
            cond = (clustID != 0)
            ids = clustID[cond]
            if ids.any():
                mem1 = _mem0[cond]
                cluster_ID[mem1] = 0
                cluster.members[:, mem1] = False
                cluster.members[0][mem1] = True
                trapped[mem1] = False
                #availClust = ids[~cluster.members[ids].any(axis=1)]
                availClust = ids[cluster.size[ids] == 0]
                cluster.availableID.update(availClust)
        else:
            cluster.clustering( np.flatnonzero(members), arrDict, Pc, cluster_ID, trapped, 
                               updatePcClustConToInlet)

    except AttributeError:
        pass

    if not updateConnectivity:
        return
    else:
        return mem0
    

@njit(cache=True)
def check_Trapping_Clustering_numba(ii, TValid, TPValid, notdone, nPores, totElements):
    done = np.zeros(totElements, dtype=np.bool_)
    done[ii] = True
    notdone[ii] = False

    filterNext = np.zeros(totElements, dtype=np.bool_)
    filterNext[ii] = True
    doPore = ii <= nPores

    while True:
        if doPore:
            temp = np.flatnonzero(filterNext[TPValid])
            ii_next = TValid[temp]
            doPore = False
        else:
            temp = np.flatnonzero(filterNext[TValid])
            ii_next = TPValid[temp]
            doPore = True

        filter_ii = notdone[ii_next]
        ii_next = ii_next[filter_ii]
        if ii_next.size == 0:
                break

        filterNext[filterNext] = False
        filterNext[ii_next] = True
        done[ii_next] = True
        notdone[ii_next] = False

    temp = np.flatnonzero(notdone[TValid])
    TValid = TValid[temp]
    TPValid = TPValid[temp]

    return done, TValid, TPValid
    

@njit(parallel=True, cache=True)
def build_Amatrix_data(
    throatList, P1array, P2array,
    isOnInletBdr, isOnOutletBdr, gL, mList):
        
    tSize = throatList.size
    max_entries = 4 * tSize

    row_tmp = np.full(max_entries, -1, dtype=np.int32)
    col_tmp = np.full_like(row_tmp, -1)
    data_tmp = np.zeros_like(row_tmp, dtype=np.float64)
    entry_counts = np.zeros(tSize, dtype=np.int32)
    
    c = np.sum(mList >= 0)  # number of active pores
    Cmatrix_cond = np.zeros(tSize, dtype=np.float64)
    Cmatrix_ind = np.full(tSize, -1, dtype=np.int32)

    for i in prange(tSize):
        t = throatList[i]-1
        cond = gL[t]
        if cond == 0.0:
            continue
            
        P1_t, P2_t = P1array[t], P2array[t]
        P1, P2 = mList[P1_t], mList[P2_t]
        offset = i*4
        local_count = 0
        
        if (P1 >= 0) and (P2 >= 0):
            # internal connection
            row_tmp[offset] = P1
            row_tmp[offset + 1] = P2
            row_tmp[offset + 2] = P1
            row_tmp[offset + 3] = P2
            
            col_tmp[offset] = P2
            col_tmp[offset + 1] = P1
            col_tmp[offset + 2] = P1
            col_tmp[offset + 3] = P2
            
            data_tmp[offset] = -cond
            data_tmp[offset + 1] = -cond
            data_tmp[offset + 2] = cond
            data_tmp[offset + 3] = cond
            
            local_count = 4

        elif (P1 >= 0) and (isOnInletBdr[P2_t]):
            # connection to inlet boundary
            row_tmp[offset] = P1
            col_tmp[offset] = P1
            data_tmp[offset] = cond
            local_count = 1
            Cmatrix_cond[i] = cond
            Cmatrix_ind[i] = P1

        elif (P2 >= 0) and (isOnInletBdr[P1_t]):
            # connection to inlet boundary
            row_tmp[offset] = P2
            col_tmp[offset] = P2
            data_tmp[offset] = cond
            local_count = 1
            Cmatrix_cond[i] = cond
            Cmatrix_ind[i] = P2

        elif (P1 >= 0) and (isOnOutletBdr[P2_t]):
            # connection to outlet boundary
            row_tmp[offset] = P1
            col_tmp[offset] = P1
            data_tmp[offset] = cond
            local_count = 1

        elif (P2 >= 0) and (isOnOutletBdr[P1_t]):
            # connection to outlet boundary
            row_tmp[offset] = P2
            col_tmp[offset] = P2
            data_tmp[offset] = cond
            local_count = 1
            
        entry_counts[i] = local_count
        
    total_entries = np.sum(entry_counts)
    row = np.empty(total_entries, dtype=np.int32)
    col = np.empty_like(row)
    data = np.empty(total_entries, dtype=np.float64)
    Cmatrix = np.zeros(c)
    
    idx = 0
    for i in range(tSize):
        offset = i*4
        n = entry_counts[i]
        for j in range(n):
            row[idx] = row_tmp[offset+j]
            col[idx] = col_tmp[offset+j]
            data[idx] = data_tmp[offset+j]
            idx += 1

        if Cmatrix_cond[i]>0.0:
            P = Cmatrix_ind[i]
            Cmatrix[P] += Cmatrix_cond[i]

    return row, col, data, Cmatrix
    

def __getValue__(self, arrr, gL):
    indP = self.poreList[arrr[self.poreList]]
    c = indP.size
    mList = -np.ones(self.nPores+2, dtype=np.int32)
    mList[indP] = np.arange(c)

    throatList = self.throatList[arrr[self.tList]]

    row, col, data, Cmatrix = build_Amatrix_data(
        throatList, self.P1array, self.P2array,
        self.isOnInletBdr, self.isOnOutletBdr, gL,
        mList)

    Amatrix = csr_matrix((data, (row, col)), shape=(c, c), dtype=float)

    return Amatrix, Cmatrix


def Saturation(self, AreaWP, AreaSP):
    return Saturation_numba(
        self.isinsideBox, self.totElements, self.totVoidVolume, AreaWP, AreaSP, self.volarray)


@njit(parallel=True, cache=True)
def Saturation_numba(isinsideBox, totElements, totVoidVolume, AreaWP, AreaSP, volarray):
    vol = 0.0
    for i in prange(totElements):
        if isinsideBox[i] and AreaSP[i]!=0.0:
            vol += (AreaWP[i]/AreaSP[i]*volarray[i])
    return vol/totVoidVolume
        
        
def computeFlowrate(self, gL, fluid, Pc, vector=False):
    arrr, arr = computeFlowrate_numba_1(
        gL, self.totElements, self.P1array, self.P2array, 
        self.tList, self.conTToIn, self.connected)
   
    conn = check_Trapping_Clustering(
        self, arr, arrr, fluid, Pc, updateConnectivity=True)
        
    if fluid == 0: self.connW = conn
    else: self.connNW = conn
    mList, arrT, c, indP = computeFlowrate_numba_2(
        conn, self.nPores, self.poreList, self.tList, self.totElements, self.isinsideBox)

    if conn.any():
        row, col, data, Cmatrix = build_Amatrix_data(
            arrT, self.P1array, self.P2array,
            self.isOnInletBdr, self.isOnOutletBdr, gL,
            mList)       
       
        Amatrix = csr_matrix((data, (row, col)), shape=(c, c), dtype=np.float64)
        pres = np.zeros(self.nPores+2)
        pres[indP] = matrixSolver(Amatrix, Cmatrix)
        qout, qp, direction = compute_qp_numba(
            self.P1array, self.P2array, self.tList, gL, self.nThroats, pres, self.poreList, c, conn,
            self.isOnInletBdr, vector, self.is_conTToInletBdr.copy(), self.is_conTToOutletBdr.copy())

    else:
        qout, qp = 0.0, np.zeros(self.nThroats)
        direction = np.ones(self.nThroats, dtype=np.bool_)
    
    if not vector:
        return qout
    else:
        return qp, direction
        
    
@njit(parallel=True, cache=True)
def computeFlowrate_numba_1(gL, totElements, P1array, P2array, tList, conTToIn, connected):
        
    arrr = np.zeros(totElements, dtype='bool')
    
    active = (gL>0.0)
    arrP1 = P1array[active]
    arrP2 = P2array[active]
    arrT = tList[active]
    
    mask_P1 = connected[arrP1]
    mask_P2 = connected[arrP2]
    mask_T = connected[arrT]
    
    arrP1_con = arrP1[mask_P1]
    arrP2_con = arrP2[mask_P2]
    arrT_con = arrT[mask_T]
    
    for i in prange(arrP1_con.size):
        P1 = arrP1_con[i]
        arrr[P1] = True
    for i in prange(arrP2_con.size):
        P2 = arrP2_con[i]
        arrr[P2] = True
    for i in prange(arrT_con.size):
        T = arrT_con[i]
        arrr[T] = True
            
    mask_TToIn = arrr[conTToIn]
    arrTToIn = conTToIn[mask_TToIn]

    return arrr, arrTToIn
    
    
@njit(parallel=True, cache=True)
def computeFlowrate_numba_2(arrr, nPores, poreList, tList, totElements, isinsideBox):
    mList = -np.ones(nPores+2, dtype=np.int32)
    for i in prange(totElements):
        if not arrr[i]:
            continue
        if not isinsideBox[i] or i < 1:
            arrr[i] = False
            continue
    
    indP = np.flatnonzero(arrr[poreList])+1
    c = indP.size
    mList[indP] = np.arange(c)
    arrT = np.flatnonzero(arrr[tList])+1
            
    return mList, arrT, c, indP
            
            
@njit(parallel=True, cache=True)
def compute_qp_numba(P1array, P2array, tList, gL, nThroats, pres, poreList, c, arrr,
        isOnInletBdr, vector, conTToInletBdr, conTToOutletBdr):
            
    qp = np.zeros(nThroats)
    indP = np.flatnonzero(arrr[poreList])+1
    for i in prange(c):
        P = indP[i]
        if arrr[P] and isOnInletBdr[P]:
            pres[P] = 1.0

    direction = np.ones(nThroats, dtype=np.bool_)
    for i in prange(nThroats):
        P1, P2, t = P1array[i], P2array[i], tList[i]
        if conTToInletBdr[i] and not arrr[t]:
            conTToInletBdr[i] = False
        if conTToOutletBdr[i] and not arrr[t]:
            conTToOutletBdr[i] = False
            
        delP = abs(pres[P1] - pres[P2])
        qp[i] = gL[i] * delP
        if vector:
            direction[i] = pres[P1]<=pres[P2]
                
    qinto = np.sum(qp[conTToInletBdr])
    qout = np.sum(qp[conTToOutletBdr])           
    if not vector and abs(qinto - qout)<1e-30:
        qout = (qinto + qout)/2.0
        
    return qout, qp, direction
            

def computePerm(self, Pc):
    gwL = computegL(self, self.gWPhase)
    self.qW = computeFlowrate(self, gwL, 0, Pc)
    self.krw = self.qW/self.qwSPhase
    if self.fluid[self.conTToOutletBdr].sum() > 0:
        gnwL = computegL(self, self.gNWPhase)
        self.qNW = computeFlowrate(self, gnwL, 1, Pc)
        self.krnw = self.qNW/self.qnwSPhase
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
    

def __wettabilityDistribution__(self, conAng=None, shuffle=True, randNum=None) -> np.array:
    ''' compute the distribution of contact angles in the network '''
    contactAng = np.zeros(self.totElements)
    if conAng is None:
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
        if shuffle:
            self.shuffle(sortedPoreIndex)
            self.shuffle(conAng)
        contactAng[sortedPoreIndex] = conAng.copy()  #'''
        
    if randNum is None:
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

        
@njit(parallel=True, cache=True)
def create_films_numba(
    arrr, halfAng, Pc, m_exists, m_inited, m_initOrMaxPcHist,
    m_initOrMinApexDistHist, advPc, recPc, m_initedApexDist, is_oil_inj,
    sigma, thetaAdvAng, thetaRecAng, nCorners):

    arr = np.flatnonzero(arrr)
    n = arr.size
    thetaAng = thetaRecAng if is_oil_inj else thetaAdvAng
    half_pi = np.pi/2.0
    
    for i in prange(n):
        idx = arr[i]
        conAng = thetaAng[idx]
        Pc_val = Pc[i]
        sigma_over_Pc = sigma / Pc_val
        for j in prange(nCorners):
            if m_exists[idx, j] and m_inited[idx, j]:
                continue

            halfAng_ij = halfAng[idx, j]
            if conAng >= (half_pi - halfAng_ij):
                continue

            m_exists[idx, j] = True
            cosTerm = np.cos(conAng + halfAng_ij)
            sinTerm = np.sin(halfAng_ij)
            initedApexDist = max(sigma_over_Pc * cosTerm / sinTerm, 0.0)
            m_initedApexDist[idx, j] = initedApexDist
            if initedApexDist != 0.0:
                advPc[idx, j] = sigma * np.cos(min(np.pi, thetaAdvAng[idx]) + halfAng_ij) / (initedApexDist * sinTerm)
                recPc[idx, j] = sigma * np.cos(min(np.pi, thetaRecAng[idx]) + halfAng_ij) / (initedApexDist * sinTerm)
            else:
                advPc[idx, j] = 0.0
                recPc[idx, j] = 0.0

            m_inited[idx, j] = True
            if Pc_val > m_initOrMaxPcHist[idx, j]:
                m_initOrMinApexDistHist[idx, j] = initedApexDist
                m_initOrMaxPcHist[idx, j] = Pc_val
                

def createFilms(self, arrr, Pc, nCorners):
    create_films_numba(
        arrr, self.m_halfAngles, Pc, self.m_cornExists, self.m_inited, 
        self.m_initOrMaxPcHist, self.m_initOrMinApexDistHist,
        self.m_advPc, self.m_recPc, self.m_initedApexDist, self.is_oil_inj,
        self.sigma, self.thetaAdvAng, self.thetaRecAng, nCorners
    )

 
def calcAreaW(self, arrr, conAng, apexDist, nCorners):
    return calcAreaW_numba(arrr, self.m_halfAngles, conAng, self.m_cornExists,
                apexDist, self.muw, nCorners)
    
  
@njit(parallel=True, cache=True)
def calcAreaW_numba(arrr, halfAng, conAng, m_exists, apexDist, muw, nCorners):
    half_pi = np.pi/2.0
    arr = np.flatnonzero(arrr)
    n = arr.size
    area_tmp = np.zeros((n, nCorners), dtype=np.float64)
    conductance_tmp = np.zeros_like(area_tmp)
    
    for i in prange(n):
        idx = arr[i]
        for j in prange(nCorners):
            if not m_exists[idx,j]:
                continue
                
            conAng_ij = conAng[i, j]
            halfAng_ij = halfAng[idx, j]
            sin_halfAng_ij = np.sin(halfAng_ij)
            cos_halfAng_ij = np.cos(halfAng_ij)
        
            term0 = conAng_ij + halfAng_ij
            term1 = term0 - half_pi
            abs_term1 = abs(term1)          
            cos_term0 = np.cos(term0)
            term2 = sin_halfAng_ij*cos_halfAng_ij
            
            if abs_term1 < 0.01:
                dimlessCornerA_ij = term2
            else:
                dimlessCornerA_ij = (
                    (sin_halfAng_ij/cos_term0)**2.0 *
                    (np.cos(conAng_ij)*cos_term0/sin_halfAng_ij + term1))
        
            cornerGstar_ij = term2/(4.0 * (1 + sin_halfAng_ij)**2.0)
            if abs_term1 > 0.01:
                cornerG_ij = dimlessCornerA_ij/(4.0 * 
                    (1 - (sin_halfAng_ij/cos_term0)*term1)**2.0)
            else:
                cornerG_ij = cornerGstar_ij
                
            if cornerG_ij != 0.0:
                apexDist_ij = apexDist[i, j]
                cFactor_ij = 0.364 + 0.28*cornerGstar_ij/cornerG_ij
                conductance_tmp[i, j] = cFactor_ij * (apexDist_ij**4.0) *\
                    (dimlessCornerA_ij**2.0) * cornerG_ij/muw
                area_tmp[i, j] = (apexDist_ij**2.0)*dimlessCornerA_ij
            else:
                conductance_tmp[i, j] = 0.0
    
    cornerArea = np.sum(area_tmp, axis=1)
    cornerCond = np.sum(conductance_tmp, axis=1)
    
    return cornerArea, cornerCond
    

def __finitCornerApex__(self, Pc):
    trapped = (self.trappedW | self.trappedNW).reshape(-1,1)
    arrr = self.connected & (self.isSquare | self.isTriangle)
    arrrS = arrr & self.isSquare
    arrrT = arrr & self.isTriangle
    arr = np.flatnonzero(arrr)
    Pc = np.full(self.totElements, Pc)
    
    m_cornExists = self.m_cornExists.copy()
    m_cornExists[arr] = (self.m_inited[arr] | (~trapped[arr])) & m_cornExists[arr]
    contactAng = self.thetaRecAng if self.is_oil_inj else self.thetaAdvAng
    apexDist = np.zeros_like(self.m_initedApexDist)
   
    if np.any(arrrT):
        arrT = np.flatnonzero(arrrT)
        _, apexDist[arrT,:3] = cornerApex(
            self, arrrT, Pc, contactAng, m_cornExists, 3, overidetrapping=True)
        
        finitCornerApex_numba(arrrT, m_cornExists, self.m_halfAngles, Pc, self.m_inited, 
            self.m_initOrMaxPcHist, self.m_initOrMinApexDistHist, self.m_advPc, self.m_recPc, 
            apexDist, self.m_initedApexDist, self.thetaRecAng, self.thetaAdvAng, 
            self.sigma, 3)
    
    if np.any(arrrS):
        arrS = np.flatnonzero(arrrS)
        _, apexDist[arrS] = cornerApex(
            self, arrrS, Pc, contactAng, m_cornExists, 4, overidetrapping=True)
        
        finitCornerApex_numba(arrrS, m_cornExists, self.m_halfAngles, Pc, self.m_inited, 
            self.m_initOrMaxPcHist, self.m_initOrMinApexDistHist, self.m_advPc, self.m_recPc, 
            apexDist, self.m_initedApexDist, self.thetaRecAng, self.thetaAdvAng, 
            self.sigma, 4)

        
@njit(parallel=True, cache=True)
def finitCornerApex_numba(arrr, m_cornExists, halfAng, Pc, m_inited, m_initOrMaxPcHist, 
    m_initOrMinApexDistHist, advPc, recPc, apexDist, m_initedApexDist,
    thetaRecAng, thetaAdvAng, sigma, nCorners):
    
    arr = np.flatnonzero(arrr)
    n = arr.size
    for i in prange(n):
        idx = arr[i]
        Pc_i = Pc[idx]
        for j in prange(nCorners):
            if not m_cornExists[idx,j]:
                continue
            halfAng_ij = halfAng[idx,j]
            sin_halfAng_ij = np.sin(halfAng_ij)
            apexDist_ij = apexDist[idx,j]
            recPc[idx,j] = sigma*np.cos((min(np.pi, thetaRecAng[idx])+halfAng_ij))/(
                apexDist_ij*sin_halfAng_ij)
            advPc[idx,j] = sigma*np.cos((min(np.pi, thetaAdvAng[idx])+halfAng_ij))/(
                apexDist_ij*sin_halfAng_ij)
            if Pc_i > m_initOrMaxPcHist[idx,j]:
                m_initOrMinApexDistHist[idx,j] = apexDist_ij
                m_initOrMaxPcHist[idx,j] = Pc_i
            m_inited[idx,j] = False
            m_initedApexDist[idx,j] = apexDist_ij
            
    

def cornerApex(self, arrr, Pc, contactAng, m_cornExists, nCorners, accurat=False,                  
               overidetrapping=False):
    
    delta = 0.0 if accurat else self._delta
    # print('Im in cornerApex!!!  ')
    # from IPython import embed; embed()
    return corner_apex_numba(
        arrr, self.m_halfAngles, Pc, contactAng, m_cornExists,
        self.m_initOrMaxPcHist, self.m_initOrMinApexDistHist, self.m_advPc,
        self.m_recPc, self.m_initedApexDist, self.trappedW, self.trappedNW, 
        self.clusterW.pc, self.clusterNW.pc, self.clusterW_ID, self.clusterNW_ID, 
        self.sigma, self.thetaAdvAng, self.thetaRecAng, 
        delta,  overidetrapping, self.MOLECULAR_LENGTH, nCorners)

    # return do.corner_apex_cython(
    #     arrr, self.m_halfAngles, Pc, m_cornExists,
    #     self.m_initOrMaxPcHist, self.m_initOrMinApexDistHist, self.m_advPc,
    #     self.m_recPc, self.m_initedApexDist, self.trappedW, self.trappedNW, 
    #     self.clusterW.pc, self.clusterNW.pc, self.clusterW_ID.astype(np.int32), 
    #     self.clusterNW_ID.astype(np.int32), 
    #     self.sigma, self.thetaAdvAng, self.thetaRecAng, 
    #     delta,  overidetrapping, self.MOLECULAR_LENGTH, nCorners)


@njit(fastmath=True, cache=True)
def corner_apex_1D_numba(
    arrr, halfAng, Pc, _conAng, m_cornExists, m_initOrMaxPcHist,
    m_initOrMinApexDistHist, advPc, recPc, apexDist, initedApexDist, 
    trappedW, trappedNW, clusterW_pc, clusterNW_pc, clusterW_ID, 
    clusterNW_ID, sigma, thetaAdvAng, thetaRecAng, 
    delta, overidetrapping, MOLECULAR_LENGTH):
        
    arr = np.flatnonzero(arrr)
    n = arr.size
    conAng = np.zeros(arrr.size, dtype=np.float64)
    for i in prange(n):
        idx = arr[i]
        if not arrr[idx]:
            continue

        Pc_i = Pc[idx]
        sigma_over_Pc = sigma / Pc_i
        halfAng_i = halfAng[idx]
        sin_h_i = np.sin(halfAng_i)
        initedApexDist_i = initedApexDist[idx]
        conAng_i = _conAng[idx]
        
        if not overidetrapping:
            apexDist_i = initedApexDist_i
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
                part = trappedPc * initedApexDist_i * sin_h_i / sigma
                part = min(0.999999, max(-0.999999, part))
                conAng_i = max(min(np.arccos(part) - halfAng_i, np.pi), 0.0)
            
        # cond0
        if not m_cornExists[idx]:
            if overidetrapping:
                apexDist_i = MOLECULAR_LENGTH

        # cond1
        elif (advPc[idx] - delta <= Pc_i) and (Pc_i <= recPc[idx] + delta):
            part = max(
                min(initedApexDist_i * sin_h_i / sigma_over_Pc, 0.999999), -0.999999)
            conAng_i = max(min(np.arccos(part) - halfAng_i, np.pi), 0.0)
            apexDist_i = initedApexDist_i

        # cond2
        elif Pc_i < advPc[idx]:
            conAng_i = thetaAdvAng[idx]
            apexDist_i = sigma_over_Pc * np.cos(conAng_i+halfAng_i)/sin_h_i

            if apexDist_i < initedApexDist_i:
                part = max(
                    min(initedApexDist_i * sin_h_i / sigma_over_Pc, 0.999999), -0.999999)
                conAng_i = max(min(np.arccos(part) - halfAng_i, np.pi), 0.0)
                apexDist_i = initedApexDist_i
        
        # cond3
        elif Pc_i > m_initOrMaxPcHist[idx]:
            conAng_i = min(np.pi, thetaRecAng[idx])
            apexDist_i = sigma_over_Pc*np.cos(conAng_i+halfAng_i)/sin_h_i

        # cond4
        elif Pc_i > recPc[idx]:
            conAng_i = thetaRecAng[idx]
            apexDist_i = sigma_over_Pc*np.cos(conAng_i+halfAng_i)/sin_h_i
            m_initOrMinApexDistHist_i = m_initOrMinApexDistHist[idx]

            if apexDist_i > initedApexDist_i:
                part = max(
                    min(initedApexDist_i * sin_h_i / sigma_over_Pc, 0.999999), -0.999999)
                conAng_i = max(min(np.arccos(part) - halfAng_i, np.pi), 0.0)
                apexDist_i = initedApexDist_i

            elif apexDist_i < m_initOrMinApexDistHist_i:
                part = max(
                    min(m_initOrMinApexDistHist_i * sin_h_i / sigma_over_Pc, 0.999999), -0.999999)
                conAng_i = max(min(np.arccos(part) - halfAng_i, np.pi), 0.0)
                apexDist_i = m_initOrMinApexDistHist_i

        # cond5
        else:
            apexDist_i = sigma_over_Pc*np.cos(conAng_i+halfAng_i)/sin_h_i

        conAng[idx] = conAng_i
        apexDist[idx] = apexDist_i

    return conAng, apexDist


@njit(fastmath=True, cache=True)
def corner_apex_numba(
    arrr, halfAng, Pc, _conAng, m_cornExists, m_initOrMaxPcHist,
    m_initOrMinApexDistHist, advPc, recPc, initedApexDist, 
    trappedW, trappedNW, clusterW_pc, clusterNW_pc, clusterW_ID, 
    clusterNW_ID, sigma, thetaAdvAng, thetaRecAng, 
    delta, overidetrapping, MOLECULAR_LENGTH, nCorners):

    arr = np.flatnonzero(arrr)
    n = arr.size
    conAng = np.zeros((n, nCorners), dtype=np.float64)
    apexDist = np.zeros((n, nCorners), dtype=np.float64)

    for i in prange(n):
        idx = arr[i]
        if not arrr[idx]:
            continue

        Pc_i = Pc[idx]
        sigma_over_Pc = sigma / Pc_i
        halfAng_i = halfAng[idx]
        if not overidetrapping:
            apexDist[i] = initedApexDist[idx,:nCorners]
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
                    apexDist[i, j] = initedApexDist[idx, j]
                    part = trappedPc * initedApexDist[idx, j] * np.sin(halfAng_i[j]) / sigma
                    part = min(0.999999, max(-0.999999, part))
                    conAng_ij = max(min(np.arccos(part) - halfAng_i[j], np.pi), 0.0)
                    conAng[i, j] = conAng_ij

        for j in range(nCorners):
            halfAng_ij = halfAng_i[j]
            sinHalfAng_ij = np.sin(halfAng_ij)
            if np.abs(sinHalfAng_ij) < 1e-10:
                continue
            initedApexDist_ij = initedApexDist[idx, j]

            # cond0
            if not m_cornExists[idx, j]:
                if overidetrapping:
                    apexDist_ij = MOLECULAR_LENGTH

            # cond1
            elif (advPc[idx, j] - delta <= Pc_i) and (Pc_i <= recPc[idx, j] + delta):
                part = max(
                    min(initedApexDist_ij * sinHalfAng_ij / sigma_over_Pc, 0.999999), -0.999999)
                conAng_ij = max(min(np.arccos(part) - halfAng_ij, np.pi), 0.0)
                apexDist_ij = initedApexDist_ij

            # cond2
            elif Pc_i < advPc[idx, j]:
                conAng_ij = thetaAdvAng[idx]
                apexDist_ij = sigma_over_Pc * np.cos(conAng_ij+halfAng_ij)/sinHalfAng_ij

                if apexDist_ij < initedApexDist_ij:
                    part = max(
                        min(initedApexDist_ij * sinHalfAng_ij / sigma_over_Pc, 0.999999), -0.999999)
                    conAng_ij = max(min(np.arccos(part) - halfAng_ij, np.pi), 0.0)
                    apexDist_ij = initedApexDist_ij
            
            # cond3
            elif Pc_i > m_initOrMaxPcHist[idx, j]:
                conAng_ij = min(np.pi, thetaRecAng[idx])
                apexDist_ij = sigma_over_Pc*np.cos(conAng_ij+halfAng_ij)/sinHalfAng_ij

            # cond4
            elif Pc_i > recPc[idx, j]:
                conAng_ij = thetaRecAng[idx]
                apexDist_ij = sigma_over_Pc*np.cos(conAng_ij+halfAng_ij)/sinHalfAng_ij
                m_initOrMinApexDistHist_ij = m_initOrMinApexDistHist[idx, j]

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
        
    
def initCornerApex(self, arr, arrr, halfAng, m_cornExists, m_inited,
                    recPc, advPc, m_initedApexDist, trapped):

    cond =  (m_cornExists & (arrr&~trapped[arr]).reshape(-1,1))
    # do.initCornerApex_cython(arr.astype(np.int32), cond, halfAng, m_inited, recPc, advPc,
    #                         m_initedApexDist, self.thetaRecAng, self.thetaAdvAng, self.sigma)

    initCornerApex_numba(arr, cond, halfAng, m_inited, recPc, advPc, m_initedApexDist,
                         self.thetaRecAng, self.thetaAdvAng, self.sigma)
                                        

def __initCornerApex__(self):
    trapped = (self.trappedW | self.trappedNW)
    arrr = self.connected
    arrrS = arrr[self.elemSquare]
    arrrT = arrr[self.elemTriangle]
    
    initCornerApex(
        self, self.elemTriangle, arrrT, self.halfAnglesTr, self.cornExistsTr, self.initedTr,
        self.recPcTr, self.advPcTr, self.initedApexDistTr, trapped)
    initCornerApex(
        self, self.elemSquare, arrrS, self.halfAnglesSq.reshape(-1,1), self.cornExistsSq, self.initedSq,
        self.recPcSq, self.advPcSq, self.initedApexDistSq, trapped)
        
        
@njit(parallel=True, cache=True)
def initCornerApex_numba(arr, cond, halfAng, m_inited, recPc, advPc, 
                        m_initedApexDist, thetaRecAng, thetaAdvAng, sigma):
    n, m = cond.shape
    for i in prange(n):
        idx = arr[i]
        for j in prange(m):
            if not cond[i,j]:
                continue
            
            halfAng_ij = halfAng[i,j]
            sin_halfAng_ij = np.sin(halfAng_ij)        
            m_inited[i,j] = True
            pc_ij = sigma*np.cos(min(np.pi, thetaRecAng[idx]+halfAng_ij))/(
                m_initedApexDist[i,j]*sin_halfAng_ij)
            recPc[i,j] = max(recPc[i,j], pc_ij)
            advPc[i,j] = sigma*np.cos(min(np.pi, thetaAdvAng[idx]+halfAng_ij))/(
                m_initedApexDist[i,j]*sin_halfAng_ij)
            

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




        
