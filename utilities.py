import numpy as np
from scipy.sparse import csr_matrix
import warnings
from functools import reduce
from numba import njit, prange
from numba.types import int32, float32, boolean, void, Tuple, int64, float64
import os
import dill
import sys
import joblib
from .solver import Solver



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
        obj.toExit = obj.toInlet|obj.toOutlet
        self.done = np.zeros(obj.totElements, dtype=np.bool_)
        self.filterNext = np.zeros(obj.totElements, dtype=np.bool_)
        obj.conAng_cur = np.zeros(obj.totElements*4, dtype=np.float32)
        obj.apexDist_cur = np.zeros(obj.totElements*4, dtype=np.float32)


def matrixSolver(Amatrix, Cmatrix) -> np.array:
    return Solver(Amatrix, Cmatrix).solve()


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
    Cmatrix = np.zeros(c, dtype=np.float64)
    
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
    mList[indP] = np.arange(c, dtype=np.int32)

    throatList = self.throatList[arrr[self.tList]]

    row, col, data, Cmatrix = build_Amatrix_data(
        throatList, self.P1array, self.P2array,
        self.isOnInletBdr, self.isOnOutletBdr, gL,
        mList)

    Amatrix = csr_matrix((data, (row, col)), shape=(c, c))

    return Amatrix, Cmatrix


def computegL(self, g) -> np.array:
    return compute_gL_numba(
        self.P1array, self.P2array, self.tList,
        self.LP1array_mod, self.LP2array_mod, self.LTarray_mod,
        g, self.nThroats
    )

@njit(parallel=True, cache=True)
def compute_gL_numba(P1array, P2array, tList, LP1, LP2, LT, g, nThroats):
    gL = np.zeros(nThroats, dtype=np.float64)
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


def computePerm0(self, Pc):
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


def computePerm(self, Pc):
    self.cWP.computeFlowrate(self.gWPhase)
    self.qW = self.cWP.flowrate
    self.krw = self.cWP.flowrate/self.qwSPhase

    self.cNWP.computeFlowrate(self.gNWPhase)
    self.qNW = self.cNWP.flowrate
    self.krnw = self.cNWP.flowrate/self.qnwSPhase

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
    contactAng = np.zeros(self.totElements, dtype=np.float32)

    if conAng is None:
        conAng = weibull(self)        

    arr = np.array([conAng[self.poreList-1].mean(), conAng[self.poreList-1].std(),
        conAng[self.poreList-1].min(), conAng[self.poreList-1].max()])*180/np.pi
    print('contact Angles (only pores): mean: {}, std: {}, min: {}, max: {}'.format(
        np.round(arr[0],2), np.round(arr[1],2), np.round(arr[2],2), np.round(arr[3],2)))

    if self.distModel.lower() == 'rmax':
        sortedConAng = conAng[conAng.argsort()[::-1]]
        sortedPoreIndex = self.poreList[self.Rarray[self.poreList].argsort()[::-1]]
        contactAng[sortedPoreIndex] = sortedConAng
        print('rmax')
        
    elif self.distModel.lower() == 'rmin':
        sortedConAng = conAng[conAng.argsort()[::-1]]
        sortedPoreIndex = self.poreList[self.Rarray[self.poreList].argsort()]
        contactAng[sortedPoreIndex] = sortedConAng
        print('rmin')
        
    else:
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
        
    if self.title=='test1D':
        contactAng[[3, 13,16,19,22,25]] = 0.0
        
    
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

        
@njit(void(
    boolean[:], float32[:], float64[:], boolean[:], boolean[:], float32[:],
    float32[:], float32[:], float32[:], float32[:], boolean,
    int64, float64, float32[:], float32[:], int32[:]), parallel=True, cache=True)
def create_films_numba(
    arrr, halfAng, Pc, m_exists, m_inited, m_initOrMaxPcHist,
    m_initOrMinApexDistHist, advPc, recPc, m_initedApexDist, is_oil_inj,
    totElements, sigma, thetaAdvAng, thetaRecAng, nCorners_arr):

    
    half_pi = np.pi / 2.0
    for i in prange(totElements):
        if not arrr[i]:
            continue

        nCorners = nCorners_arr[i]
        offset = i * 4 
        current_Pc = Pc[i]
        conAng = thetaRecAng[i] if is_oil_inj else thetaAdvAng[i]
        sigma_over_Pc = sigma / current_Pc

        for j in range(nCorners):
            f_idx = offset + j
            if m_exists[f_idx] and m_inited[f_idx]:
                continue

            halfAng_ij = halfAng[f_idx]
            if conAng >= (half_pi - halfAng_ij):
                continue

            m_exists[f_idx] = True
            cosTerm = np.cos(conAng + halfAng_ij)
            sinTerm = np.sin(halfAng_ij)
            dist = max(float(sigma_over_Pc * cosTerm / sinTerm), 0.0)
            m_initedApexDist[f_idx] = dist

            if dist > 0.0:
                common = sigma / (dist * sinTerm)
                adv_angle = min(np.pi, thetaAdvAng[i])
                advPc[f_idx] = common * np.cos(adv_angle + halfAng_ij)
                rec_angle = min(np.pi, thetaRecAng[i])
                recPc[f_idx] = common * np.cos(rec_angle + halfAng_ij)
            else:
                advPc[f_idx] = 0.0
                recPc[f_idx] = 0.0

            m_inited[f_idx] = True
            if current_Pc > m_initOrMaxPcHist[f_idx]:
                m_initOrMinApexDistHist[f_idx] = dist
                m_initOrMaxPcHist[f_idx] = current_Pc
                

def createFilms(self, arrr, Pc, nCorners):
    create_films_numba(
        arrr, self.m_halfAngles, Pc, self.m_cornExists, self.m_inited, 
        self.m_initOrMaxPcHist, self.m_initOrMinApexDistHist,
        self.m_advPc, self.m_recPc, self.m_initedApexDist, self.is_oil_inj,
        self.sigma, self.thetaAdvAng, self.thetaRecAng, nCorners
    )

 
def calcAreaW(self, arrr, conAng, apexDist, nCorners):
    return calcAreaW_numba(np.flatnonzero(arrr), self.m_halfAngles, conAng, self.m_cornExists,
                apexDist, self.muw, nCorners)
    
  
@njit(void(
    boolean[:], float32[:], int32[:], float32[:], float32[:], boolean[:], 
    int64, float64, boolean[:], float32[:], float32[:], float32[:],
    float32[:], float32[:], float32[:], float32[:],
    float32[:], float32[:], float32[:], float32[:], float32[:], float32[:],
    boolean, int32[:], boolean, boolean), parallel=True, cache=True)
def calcAreaW_numba(
    arrr, halfAng, fluid_arr, conAng_arr, apexDist_arr, m_exists, 
    totElements, muw, trappedNW, areaSPhase, gwSPhase, gnwSPhase, 
    maxCornerArea, maxCornerCond, _cornArea, _cornCond,
     _centerArea, _centerCond, _areaWP, _areaNWP, _condWP, _condNWP,
     is_oil_inj, nCorners_arr, updateArea, overideTrapping):

    half_pi = np.pi / 2.0
    for i in prange(totElements):
        if not arrr[i]:
            continue

        cornA = 0.0
        cornG = 0.0
        nCorners = nCorners_arr[i]

        if nCorners > 0:
            offset = i * 4
            for j in range(nCorners):
                f_idx = offset + j
                
                if not m_exists[f_idx]:
                    continue

                halfAng_ij = halfAng[f_idx]
                conAng_ij = conAng_arr[f_idx]
                sin_ha = np.sin(halfAng_ij)
                cos_ha = np.cos(halfAng_ij)
                
                term0 = conAng_ij + halfAng_ij
                term1 = term0 - half_pi
                abs_term1 = abs(term1)
                cos_term0 = np.cos(term0)
                term2 = sin_ha * cos_ha

                cornerGstar_ij = term2 / (4.0 * ((1.0 + sin_ha) ** 2))
                if abs_term1 < 0.01:
                    dimlessCornerA_ij = term2
                else:
                    dimlessCornerA_ij = ((sin_ha / cos_term0) ** 2) * (
                        (np.cos(conAng_ij) * cos_term0 / sin_ha) + term1)

                if abs_term1 > 0.01:
                    cornerG_ij = dimlessCornerA_ij / (
                        4.0 * ((1.0 - (sin_ha / cos_term0) * term1) ** 2))
                else:
                    cornerG_ij = cornerGstar_ij

                if cornerG_ij != 0.0:
                    apexDist_ij = apexDist_arr[f_idx]
                    cFactor_ij = 0.364 + 0.28 * cornerGstar_ij / cornerG_ij
                    cornG += (cFactor_ij * (apexDist_ij ** 4) * (dimlessCornerA_ij ** 2) * cornerG_ij / muw)
                    cornA += (apexDist_ij ** 2) * dimlessCornerA_ij

            if is_oil_inj:
                if cornA < _cornArea[i]: 
                    _cornArea[i] = cornA
                if cornG < _cornCond[i]: 
                    _cornCond[i] = cornG
                cornA = _cornArea[i]
                cornG = _cornCond[i]
            else:
                if cornA > areaSPhase[i]:
                    cornA = _cornArea[i]
                    cornG = _cornCond[i]
                elif cornA > maxCornerArea[i]:
                    maxCornerArea[i] = cornA
                    maxCornerCond[i] = cornG
                
                _cornArea[i] = cornA
                _cornCond[i] = cornG

        else:
            if fluid_arr[i] == 1:
                _cornArea[i] = 0.0
                _cornCond[i] = 0.0
                cornA = 0.0
                cornG = 0.0
            else:
                cornA = areaSPhase[i]
                cornG = gnwSPhase[i]
                _cornArea[i] = cornA
                _cornCond[i] = cornG

        # Update Bulk Phase Occupancy and Conductance
        if updateArea and (overideTrapping or not trappedNW[i]):
            if fluid_arr[i] == 0:  # Wetting phase fills center
                _areaWP[i] = areaSPhase[i]
                _areaNWP[i] = 0.0
                _condWP[i] = gwSPhase[i]
                _condNWP[i] = 0.0
            else:  # Non-Wetting phase in center, Wetting phase in films
                _areaWP[i] = cornA
                _areaNWP[i] = areaSPhase[i] - cornA
                _condWP[i] = cornG
                if areaSPhase[i] > 1e-30:
                    _condNWP[i] = (_areaNWP[i] / areaSPhase[i]) * gnwSPhase[i]
                else:
                    _condNWP[i] = 0.0

        _centerArea[i] = areaSPhase[i] - cornA
        if areaSPhase[i] > 1e-30:
            _centerCond[i] = (_centerArea[i] / areaSPhase[i]) * gnwSPhase[i]
        else:
            _centerCond[i] = 0.0


@njit(parallel=True, cache=True)
def finiteCornerHelper(arrr, isPolygon, trappedW, trappedNW, m_exists, m_inited, nCorners_arr):
    n = arrr.size
    for i in prange(n):
        if not arrr[i]: continue
        if not isPolygon[i]:
            arrr[i] = False
            continue

        trapped = trappedW[i] or trappedNW[i]
        nCorners = nCorners_arr[i]
        base = i * 4
        for j in range(nCorners):
            k = base + j
            if not m_exists[k]: continue
            if not m_inited[k] and trapped:
                m_exists[k] = False


def __finitCornerApex__(self, Pc):
    arrr = self.connected.copy()
    m_exists = self.m_cornExists.copy()
    finiteCornerHelper(arrr, self.isPolygon, self.cWP.trapped, self.cNWP.trapped,
        m_exists, self.m_inited, self.nCorners_arr)
    Pc = np.full(self.totElements, Pc, dtype=np.float64)
    corner_apex_numba(
        arrr, self.m_halfAngles, Pc, m_exists, self.m_inited, self.m_initOrMaxPcHist,
        self.m_initOrMinApexDistHist, self.m_advPc, self.m_recPc, self.m_initedApexDist, 
        self.is_oil_inj, self.totElements, self.sigma, self.contactAng, self.thetaAdvAng, 
        self.thetaRecAng, self.nCorners_arr, self.cWP.trapped, self.cWP.clusterID, 
        self.cWP.pc, self.cNWP.trapped, self.cNWP.clusterID, self.cNWP.pc, self.conAng_cur,
        self.apexDist_cur, self.MOLECULAR_LENGTH, self._delta, False, True)

    finitCornerApex_numba(arrr, m_exists, self.m_halfAngles, Pc, self.m_inited, 
        self.m_initOrMaxPcHist, self.m_initOrMinApexDistHist, self.m_advPc, self.m_recPc, 
        self.apexDist_cur, self.m_initedApexDist, self.thetaRecAng, self.thetaAdvAng, 
        self.sigma, self.nCorners_arr)


@njit(parallel=True, cache=True, fastmath=True)
def finitCornerApex_numba(
    arrr, m_cornExists, halfAng, Pc, m_inited, m_initOrMaxPcHist, m_initOrMinApexDistHist,
    advPc, recPc, apexDist, m_initedApexDist, thetaRecAng, thetaAdvAng, sigma, nCorners_arr):

    n = arrr.size
    for i in prange(n):
        if not arrr[i]: continue
        Pc_i = Pc[i]
        nCorners = nCorners_arr[i]
        theta_rec = thetaRecAng[i]
        theta_adv = thetaAdvAng[i]
        base = i * 4

        for j in range(nCorners):
            k = base + j
            if not m_cornExists[k]: continue

            ha = halfAng[k]
            sin_ha = np.sin(ha) 
            ad = apexDist[k]
            denom = ad * sin_ha

            recPc[k] = sigma * np.cos(min(np.pi, theta_rec) + ha) / denom
            advPc[k] = sigma * np.cos(min(np.pi, theta_adv) + ha) / denom

            if Pc_i > m_initOrMaxPcHist[k]:
                m_initOrMaxPcHist[k] = Pc_i
                m_initOrMinApexDistHist[k] = ad

            m_inited[k] = False
            m_initedApexDist[k] = ad

            
    

def cornerApex(self, arrr, Pc, m_cornExists, nCorners, accurate=False,                  
               overidetrapping=False):
    
    corner_apex_numba(
        arrr, self.m_halfAngles, Pc, self.m_cornExists, self.m_inited, self.m_initOrMaxPcHist,
        self.m_initOrMinApexDistHist, self.m_advPc, self.m_recPc, self.m_initedApexDist, self.is_oil_inj,
        self.totElements, self.sigma, self.contactAng, self.thetaAdvAng, self.thetaRecAng, 
        self.nCorners_arr, self.cWP.trapped, self.cWP.clusterID, self.cWP.pc, self.cNWP.trapped, 
        self.cNWP.clusterID, self.cNWP.pc, self.conAng_cur, self.apexDist_cur, self.MOLECULAR_LENGTH, 
        self._delta, accurate, overidetrapping)

  
@njit(fastmath=True, cache=True, parallel=True)
def corner_apex_1D_numba(
    arr, halfAng, Pc, _conAng, m_cornExists, m_initOrMaxPcHist,
    m_initOrMinApexDistHist, advPc, recPc, conAng, apexDist, initedApexDist, 
    trappedW, trappedNW, clusterW_pc, clusterNW_pc, clusterW_ID, 
    clusterNW_ID, sigma, thetaAdvAng, thetaRecAng, 
    delta, overidetrapping, MOLECULAR_LENGTH, c):
        
    n = arr.size
    for i in prange(n):
        idx = arr[i]
        ind = idx*4 + c
        Pc_i = Pc[idx]
        sigma_over_Pc = sigma / Pc_i
        halfAng_i = halfAng[ind]
        sin_h_i = np.sin(halfAng_i)
        initedApexDist_i = initedApexDist[ind]
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
        if not m_cornExists[ind]:
            if overidetrapping:
                apexDist_i = MOLECULAR_LENGTH

        # cond1
        elif (advPc[ind] - delta <= Pc_i) and (Pc_i <= recPc[ind] + delta):
            part = max(
                min(initedApexDist_i * sin_h_i / sigma_over_Pc, 0.999999), -0.999999)
            conAng_i = max(min(np.arccos(part) - halfAng_i, np.pi), 0.0)
            apexDist_i = initedApexDist_i

        # cond2
        elif Pc_i < advPc[ind]:
            conAng_i = thetaAdvAng[idx]
            apexDist_i = sigma_over_Pc * np.cos(conAng_i+halfAng_i)/sin_h_i

            if apexDist_i < initedApexDist_i:
                part = max(
                    min(initedApexDist_i * sin_h_i / sigma_over_Pc, 0.999999), -0.999999)
                conAng_i = max(min(np.arccos(part) - halfAng_i, np.pi), 0.0)
                apexDist_i = initedApexDist_i
        
        # cond3
        elif Pc_i > m_initOrMaxPcHist[ind]:
            conAng_i = min(np.pi, thetaRecAng[idx])
            apexDist_i = sigma_over_Pc*np.cos(conAng_i+halfAng_i)/sin_h_i

        # cond4
        elif Pc_i > recPc[ind]:
            conAng_i = thetaRecAng[idx]
            apexDist_i = sigma_over_Pc*np.cos(conAng_i+halfAng_i)/sin_h_i
            m_initOrMinApexDistHist_i = m_initOrMinApexDistHist[ind]

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


@njit(void(
    boolean[:], float32[:], float64[:], boolean[:], boolean[:], float32[:],
    float32[:], float32[:], float32[:], float32[:], boolean,
    int64, float64, float32[:], float32[:], float32[:], int32[:],
    boolean[:], int32[:], float32[:], boolean[:], int32[:], float32[:],
    float32[:], float32[:], float64, float64, boolean, boolean), parallel=True, cache=True)
def corner_apex_numba(
    arrr, halfAng, Pc_values, m_exists, m_inited, m_initOrMaxPcHist,
    m_initOrMinApexDistHist, advPc, recPc, m_initedApexDist, is_oil_inj,
    totElements, sigma, contactAng, thetaAdvAng, thetaRecAng, nCorners_arr, 
    trappedW, clusterW_ID, clusterW_pc, trappedNW, clusterNW_ID, clusterNW_pc,
    conAng_out, apexDist_out, molecular_length, _delta, accurate, overidetrapping):

    delta = 0.0 if accurate else _delta
    
    for i in prange(totElements):
        if not arrr[i]:
            continue

        offset = i * 4
        current_Pc = Pc_values[i]
        sigma_over_Pc = sigma / current_Pc

        trapped = False
        trappedPc_val = 0.0
        if not overidetrapping:
            if trappedW[i]:
                trappedPc_val = clusterW_pc[clusterW_ID[i]]
                trapped = True
            elif trappedNW[i]:
                trappedPc_val = clusterNW_pc[clusterNW_ID[i]]
                trapped = True

        nCorners = nCorners_arr[i]
        for j in range(nCorners):
            f_idx = offset + j
            ha_ij = halfAng[f_idx]
            sinHa_ij = np.sin(ha_ij)

            if abs(sinHa_ij) < 1e-10:
                conAng_out[f_idx] = 0.0
                apexDist_out[f_idx] = 0.0
                continue

            conAng_ij = 0.0
            apexDist_ij = 0.0

            # 1. Initial Trapping/Contact Angle Logic
            if not overidetrapping:
                apexDist_ij = m_initedApexDist[f_idx]
                if trapped:
                    part = trappedPc_val * apexDist_ij * sinHa_ij / sigma
                    part = max(-0.999999, min(0.999999, part))
                    conAng_ij = max(0.0, min(np.pi, np.arccos(part) - ha_ij))
                else:
                    conAng_ij = contactAng[i]

            # 2. Main Geometric Logic Branches
            if not m_exists[f_idx]:
                if overidetrapping:
                    apexDist_ij = molecular_length
            
            elif (advPc[f_idx] - delta <= current_Pc) and (current_Pc <= recPc[f_idx] + delta):
                apexDist_ij = m_initedApexDist[f_idx]
                part = max(-0.999999, min(0.999999, (apexDist_ij * sinHa_ij / sigma_over_Pc)))
                conAng_ij = max(0.0, min(np.pi, np.arccos(part) - ha_ij))

            elif current_Pc < advPc[f_idx]:
                conAng_ij = thetaAdvAng[i]
                apexDist_ij = sigma_over_Pc * np.cos(conAng_ij + ha_ij) / sinHa_ij
                if apexDist_ij < m_initedApexDist[f_idx]:
                    apexDist_ij = m_initedApexDist[f_idx]
                    part = max(-0.999999, min(0.999999, (apexDist_ij * sinHa_ij / sigma_over_Pc)))
                    conAng_ij = max(0.0, min(np.pi, np.arccos(part) - ha_ij))

            elif current_Pc > m_initOrMaxPcHist[f_idx]:
                conAng_ij = min(np.pi, thetaRecAng[i])
                apexDist_ij = sigma_over_Pc * np.cos(conAng_ij + ha_ij) / sinHa_ij

            elif current_Pc > recPc[f_idx]:
                conAng_ij = thetaRecAng[i]
                apexDist_ij = sigma_over_Pc * np.cos(conAng_ij + ha_ij) / sinHa_ij
                minApexHist = m_initOrMinApexDistHist[f_idx]
                
                if apexDist_ij > m_initedApexDist[f_idx]:
                    apexDist_ij = m_initedApexDist[f_idx]
                    part = max(-0.999999, min(0.999999, (apexDist_ij * sinHa_ij / sigma_over_Pc)))
                    conAng_ij = max(0.0, min(np.pi, np.arccos(part) - ha_ij))
                elif apexDist_ij < minApexHist:
                    part = max(-0.999999, min(0.999999, (minApexHist * sinHa_ij / sigma_over_Pc)))
                    conAng_ij = max(0.0, min(np.pi, np.arccos(part) - ha_ij))
                    apexDist_ij = minApexHist
            else:
                apexDist_ij = sigma_over_Pc * np.cos(conAng_ij + ha_ij) / sinHa_ij

            # Final assignment
            conAng_out[f_idx] = conAng_ij
            apexDist_out[f_idx] = apexDist_ij


@njit(void(
    boolean[:], float32[:], float64[:], boolean[:], boolean[:], 
    float32[:], float32[:], float32[:], float32[:], float32[:], 
    boolean, int64, float64, float32[:], float32[:], float32[:], int32[:],
    boolean[:], int32[:], float32[:], boolean[:], int32[:], float32[:], int32[:],
    float32[:], float32[:], float32[:], float32[:], float32[:], float32[:], float32[:],
    float32[:], float32[:], float32[:], float32[:], float32[:], float32[:],
    float32[:], float32[:], float64, float64, float64, boolean, boolean, boolean), cache=True)
def update_areas_conductances_numba(arrr, halfAng, Pc_values, m_exists, m_inited, 
    m_initOrMaxPcHist, m_initOrMinApexDistHist, advPc, recPc, m_initedApexDist,
    is_oil_inj, totElements, sigma, contactAng, thetaAdvAng, thetaRecAng, nCorners_arr, 
    trappedW, clusterW_ID, clusterW_pc, trappedNW, clusterNW_ID, clusterNW_pc, fluid_arr,
    areaSPhase, gwSPhase, gnwSPhase, maxCornerArea, maxCornerCond, _cornArea, _cornCond,
    _centerArea, _centerCond, _areaWP, _areaNWP, _condWP, _condNWP,
    conAng_cur, apexDist_cur, molecular_length, muw, _delta, accurate, updateArea, overideTrapping):

    create_films_numba(
        arrr, halfAng, Pc_values, m_exists, m_inited, m_initOrMaxPcHist,
        m_initOrMinApexDistHist, advPc, recPc, m_initedApexDist, is_oil_inj,
        totElements, sigma, thetaAdvAng, thetaRecAng, nCorners_arr)

    corner_apex_numba(
        arrr, halfAng, Pc_values, m_exists, m_inited, m_initOrMaxPcHist,
        m_initOrMinApexDistHist, advPc, recPc, m_initedApexDist, is_oil_inj,
        totElements, sigma, contactAng, thetaAdvAng, thetaRecAng, nCorners_arr, 
        trappedW, clusterW_ID, clusterW_pc, trappedNW, clusterNW_ID, clusterNW_pc,
        conAng_cur, apexDist_cur, molecular_length, _delta, accurate, overideTrapping)
    
    calcAreaW_numba(
        arrr, halfAng, fluid_arr, conAng_cur, apexDist_cur, m_exists, 
        totElements, muw, trappedNW, areaSPhase, gwSPhase, gnwSPhase, 
        maxCornerArea, maxCornerCond, _cornArea, _cornCond,
        _centerArea, _centerCond, _areaWP, _areaNWP, _condWP, _condNWP, 
        is_oil_inj, nCorners_arr, updateArea, overideTrapping)


def update_areas_conductances(self, arrr, Pc_values, accurate, updateArea, overidetrapping):
    cWP = self.cWP
    cNWP = self.cNWP
    
    update_areas_conductances_numba(
        arrr, self.m_halfAngles, Pc_values, self.m_cornExists, self.m_inited, 
        self.m_initOrMaxPcHist, self.m_initOrMinApexDistHist, self.m_advPc, self.m_recPc, 
        self.m_initedApexDist, self.is_oil_inj, self.totElements, self.sigma, self.contactAng, 
        self.thetaAdvAng, self.thetaRecAng, self.nCorners_arr, cWP.trapped, cWP.clusterID, 
        cWP.pc, cNWP.trapped, cNWP.clusterID, cNWP.pc, self.fluid, self.areaSPhase, self.gwSPhase, 
        self.gnwSPhase, self.maxCornerArea, self.maxCornerCond, self._cornArea, self._cornCond, 
        self._centerArea, self._centerCond, self._areaWP, self._areaNWP, self._condWP, self._condNWP, 
        self.conAng_cur, self.apexDist_cur, self.MOLECULAR_LENGTH, self.muw, self._delta, accurate, 
        updateArea, overidetrapping)


def __initCornerApex__(self):
    initCornerApex_numba(self.connected, self.m_halfAngles, self.m_cornExists, self.m_inited, 
        self.m_recPc, self.m_advPc, self.m_initedApexDist, self.thetaRecAng, self.thetaAdvAng, 
        self.cWP.trapped, self.cNWP.trapped, self.sigma, self.nCorners_arr)


@njit(parallel=True, cache=True)
def initCornerApex_numba(
    arrr, halfAng, m_exists, m_inited, recPc, advPc, m_initedApexDist, 
    thetaRecAng, thetaAdvAng, trappedW, trappedNW, sigma, nCorners_arr):
    
    n = arrr.size

    for i in prange(n):
        if not arrr[i] or trappedW[i] or trappedNW[i]: continue
        nCorners = nCorners_arr[i]
        base = i * 4
        theta_rec = thetaRecAng[i]
        theta_adv = thetaAdvAng[i]
        

        for j in range(nCorners):
            k = base + j
            if not m_exists[k]: continue

            ha = halfAng[k]
            sin_ha = np.sin(ha)
            m_inited[k] = True
            denom = m_initedApexDist[k] * sin_ha
            pc = sigma * np.cos(min(np.pi, theta_rec + ha)) / denom
            if pc > recPc[k]: recPc[k] = pc
            advPc[k] = sigma * np.cos(min(np.pi, theta_adv + ha)) / denom



@njit(cache=True, parallel=True)
def Pc_pistonHing_numba(
    arrr, halfAng, m_cornExists, m_initOrMaxPcHist, m_initOrMinApexDistHist,
    advPc, recPc, initedApexDist, initialPc, thetaAdvAng, thetaRecAng,
    cosThetaAdvAng, Rarray, Garray, trappedW, trappedNW, clusterW_pc, clusterNW_pc,
    clusterW_ID, clusterNW_ID, sigma, delta, EPSILON, MAX_ITER, MOLECULAR_LENGTH,
    nCorners):

    N = arrr.size
    active = np.empty(N, dtype=np.int32)
    n_active = 0
    for i in range(N):
        if arrr[i]:
            active[n_active] = i
            n_active += 1

    apexDist = np.zeros(N*4, dtype=np.float32)
    conAng = np.zeros(N, dtype=np.float32)
    sumOne   = np.zeros(N, dtype=np.float32)
    sumTwo   = np.zeros(N, dtype=np.float32)
    sumThree = np.zeros(N, dtype=np.float32)
    sumFour  = np.zeros(N, dtype=np.float32)
    oldPc    = np.empty(N, dtype=np.float32)

    counter = 0

    while n_active > 0 and counter < MAX_ITER:
        for k in prange(n_active):
            i = active[k]
            oldPc[i] = initialPc[i]
            sumOne[i] = 0.0
            sumTwo[i] = 0.0
            sumThree[i] = 0.0
            sumFour[i] = 0.0

        for c in range(nCorners):
            corner_apex_1D_numba(
                active[:n_active], halfAng, oldPc, thetaAdvAng, m_cornExists,
                m_initOrMaxPcHist, m_initOrMinApexDistHist, advPc, recPc,
                conAng, apexDist, initedApexDist, trappedW, trappedNW,
                clusterW_pc, clusterNW_pc, clusterW_ID, clusterNW_ID,
                sigma, thetaAdvAng, thetaRecAng, delta, True,
                MOLECULAR_LENGTH, c)

            for k in prange(n_active):
                i = active[k]
                idx = i*4 + c

                if not m_cornExists[idx]: continue

                ha = halfAng[idx]
                sin_ha = np.sin(ha)
                ad = apexDist[i]

                part = ad * sin_ha * oldPc[i] / sigma
                if part < -1.0 or part > 1.0:
                    part = 0.0


                ca = conAng[i]
                sumOne[i]   += ad * np.cos(ca)
                sumTwo[i]   += (np.pi * 0.5 - ha - ca)
                sumThree[i] += np.arcsin(part)
                sumFour[i]  += ad

        new_n_active = 0

        for k in range(n_active):
            i = active[k]

            a = 2.0 * sumThree[i] - sumTwo[i]
            b = ((cosThetaAdvAng[i] * Rarray[i] / (2.0 * Garray[i]))
                 - 2.0 * sumFour[i] + sumOne[i])
            c = -(Rarray[i] * Rarray[i]) / (4.0 * Garray[i])

            disc = b * b - 4.0 * a * c
            if disc > 0.0:
                Pc_new = sigma * (2.0 * a) / (-b + np.sqrt(disc))
            else:
                Pc_new = sigma * (2.0 * a) / (-b)

            err = 2.0 * abs(
                (Pc_new - oldPc[i]) /
                (abs(oldPc[i]) + abs(Pc_new) + 1e-3)
            )

            initialPc[i] = Pc_new

            if err >= EPSILON:
                active[new_n_active] = i
                new_n_active += 1

        n_active = new_n_active
        counter += 1

    count = 0
    for i in range(N):
        if np.isnan(initialPc[i]):
            initialPc[i] = 0.0

    return initialPc[arrr]


def updateObj(self, obj):
    list_of_attr = []
    selfDict = self.__dict__
    try:
        objDict = obj.__dict__
    except AttributeError:
        objDict = obj

    for key, new_val in objDict.items():
        if key not in selfDict:
            setattr(self, key, new_val)
            list_of_attr.append(key)
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
                    list_of_attr.append(key)
            else:
                # If not equal, update attribute
                if old_val != new_val:
                    setattr(self, key, new_val)
                    list_of_attr.append(key)

        except (ValueError, TypeError, AttributeError):
            # Fallback to setattr if any issue in comparison or assignment
            setattr(self, key, new_val)
            list_of_attr.append(key)
            
        if key=='cWP' or key=='cNWP':
            setattr(getattr(self, key), 'network', self) 
            
    return list_of_attr


def writeResult(self, result_str, Pc):
    print('Sw: %10.6g  \tqW: %8.6e  \tkrw: %12.6g  \tqNW: %8.6e  \tkrnw:\
            %12.6g  \tPc: %8.6g\t %8.0f invasions' % (
            self.satW, self.qW, self.krw, self.qNW, self.krnw,
            Pc, self.totNumFill, ))
        
    if self.writeData:
        result_str+="\n%10.6g,\t%8.6e,\t%12.6g,\t\t%8.6e,\t%12.6g,\t\t%8.6g,\t%8.0f" % (
            self.satW, self.qW, self.krw, self.qNW, self.krnw, Pc, self.totNumFill, )
    
    return result_str


def __writeHeaders__(self):
    self.header_str="======================================================================\n"
    self.header_str+="# Fluid properties:\nsigma (mN/m)  \tmu_w (cP)  \tmu_nw (cP)\n"
    self.header_str+="# \t%.6g\t\t%.6g\t\t%.6g" % (
        self.sigma*1000, self.muw*1000, self.munw*1000, )
    self.header_str+="\n# calcBox: \t %.6g \t %.6g" % (
        self.calcBox[0], self.calcBox[1], )
    self.header_str+="\n# Wettability:"
    self.header_str+="\n# model \tmintheta \tmaxtheta \tdelta \teta \tdistmodel"
    self.header_str+="\n# %.6g\t\t%.6g\t\t%.6g\t\t%.6g\t%.6g\t" % (
        self.wettClass, round(self.minthetai*180/np.pi,3), round(self.maxthetai*180/np.pi,3), 
        self.delta, self.eta,)
    self.header_str+=self.distModel
    self.header_str+="\nmintheta \tmaxtheta \tmean  \t\tstd"
    self.header_str+="\n# %3.6g\t\t%3.6g\t\t%3.6g\t\t%3.6g" % (
        round(self.contactAng.min()*180/np.pi,3), round(self.contactAng.max()*180/np.pi,3), 
        round(self.contactAng.mean()*180/np.pi,3), round(self.contactAng.std()*180/np.pi,3))
    
    self.header_str+="\nPorosity:  %3.6g" % (self.porosity)
    self.header_str+="\nMaximum pore connection:  %3.6g" % (self.maxPoreCon)
    self.header_str+="\nAverage pore-to-pore distance:  %3.6g" % (self.avgP2Pdist)
    self.header_str+="\nMean pore radius:  %3.6g" % (self.Rarray[self.poreList].mean())
    self.header_str+="\nAbsolute permeability:  %3.6g" % (self.absPerm)
    
    self.header_str+="\n======================================================================"
    self.header_str+="\n# Sw\t\tqW(m3/s)\t\tkrw\t\tqNW(m3/s)\t\tkrnw\t\tPc\t\tInvasions"

    self.totNumFill = 0
    with open(self.file_name, 'a') as fQ:
        fQ.write(self.header_str)
    

def __writeTrappedData__(self):
    displacement_type = 'Drainage' if self.is_oil_inj else 'Imbibition'
    filename = os.path.join(
        self.results_dir, "{}_{}_cycle_{}_trappedDist_{}.csv".format(
            self.title, displacement_type, self.cycle, self._num))
    data = [*zip(self.Rarray, self.volarray, self.fluid, self.cWP.trapped, self.cNWP.trapped)]
    np.savetxt(filename, data, delimiter=',', 
                header='rad, volume, fluid, trappedW, trappedNW',
                fmt=['%.6e', '%.6e', '%d', '%d', '%d'])


def __fileName__(self):
    os.makedirs(os.path.dirname(self.results_dir), exist_ok=True)
    displacement_type = 'Drainage' if self.is_oil_inj else 'Imbibition'
    if not hasattr(self, '_num'):
        self._num = 1
        while True:         
            file_name = os.path.join(
                self.results_dir, "{}_{}_cycle_{}_{}.csv".format(
                self.title, displacement_type, self.cycle, self._num))
            if os.path.isfile(file_name): self._num += 1
            else:
                break
        self.file_name = file_name
    else:
        self.file_name = os.path.join(
            self.results_dir,"{}_{}_cycle_{}_{}.csv".format(
                self.title, displacement_type, self.cycle, self._num))


def sizeof(obj):
    # numpy arrays: include data buffer
    if isinstance(obj, np.ndarray):
        return obj.nbytes

    # python containers: estimate recursively (shallow)
    if isinstance(obj, dict):
        return sys.getsizeof(obj) + sum(sizeof(k) + sizeof(v) for k, v in obj.items())

    if isinstance(obj, (list, tuple, set)):
        return sys.getsizeof(obj) + sum(sizeof(x) for x in obj)

    # fallback: shallow size only
    try:
        return sys.getsizeof(obj)
    except Exception:
        return 0


def saveState(self, fname):

    state_attrs = ['is_oil_inj', 'maxPc', 'wettClass', 'minthetai', 'maxthetai', 'delta', 'eta',
                   'distModel', 'sepAng', 'results_dir', 'results_str', '_areaWP', '_cornArea', 
                   '_areaNWP', '_centerArea', '_condWP', '_cornCond', '_condNWP', '_centerCond',
                   'areaWPhase', 'areaNWPhase', 'gWPhase', 'gNWPhase', 'contactAng', 
                   'thetaRecAng', 'thetaAdvAng', 'Fd_Tr', 'Fd_Sq', 'PistonPcRec', 'centreEPOilInj', 
                   'pop', 'update', 'NinElemList', 'capPresMax', 'capPresMin', 'qW', 'qNW', 'krw', 'krnw',
                   'totNumFill', 'prop_drainage', 'SwTarget', 'PcTarget', 'oldPcTarget', 'oldSatW', 'fillTarget',
                   'invInsideBox', 'cnt', 'fw', 'rpd', 'satW', 'rng', 'm_cornExists', 
                   'm_initOrMaxPcHist', 'm_initOrMinApexDistHist', 'm_initedApexDist', 'm_advPc', 'm_recPc', 
                   'PcD',  'fluid', 'cWP', 'cycle', 'cNWP']

    state = {attr: getattr(self, attr) for attr in state_attrs}

    additional_attrs = ['maxCornerCond', 'maxCornerArea', 'PcI', 'minPc', 'PistonPcAdv', 'porebodyPc', 'snapoffPc']
    for attr in additional_attrs:
        if hasattr(self, attr):
            state[attr] = getattr(self, attr)
    
    joblib.dump(state, fname, compress=3)
    
        
