import numpy as np
from sortedcontainers import SortedList
from numba import njit, prange, jit
from numba.types import int32, float32, boolean, void, Tuple, int64, float64
from numba.typed import Dict
import pnflowPy.temp as temp
import pnflowPy.utilities as do
from scipy.sparse import csr_matrix
from time import time
import os
from IPython import embed

_temp = temp.TempArrays()


class Cluster():
    def __init__(self, network, phase, nClusters=None):
        self.network = network
        totElements = network.totElements
        if nClusters is None:
            if totElements < 1000:
               nClusters = totElements
            else:
               nClusters = network.totElements//8
            
        
        self.time_doClustering = 0.0
        self.phase = phase
        self.values = [ClusterObj(0, self, network)]
        self.keys = [0]
        self.drainEvents = 0
        self.imbEvents = 0
        _temp.formTempNetworkArrays(
            network.nPores, totElements, network.connectivity_graph_flat.size)

        self.pc = np.zeros(nClusters, dtype=np.float32)
        self.drainEvents = 0
        self.imbEvents = 0
        
        self.members = np.full(totElements, -5, dtype=np.int32)
        self.mem_offsets = np.zeros(nClusters+1, dtype=np.int32)
        self.heads_arr = np.full(nClusters, -5, dtype=np.int32) 
        self.next_elem_arr = np.full(totElements, -5, dtype=np.int32)
        self.neighbours_updated = np.zeros(nClusters, dtype=np.bool_)
        self._neighbours = [[] for _ in range(nClusters)]
        
        self._visited_by = np.full(totElements, -1, dtype=np.int32)
        self._parent = np.arange(totElements, dtype=np.int32)

        self.trappedStatus = np.zeros(nClusters, dtype=np.bool_)
        self.connected = np.zeros(nClusters, dtype=np.bool_)
        self.sizes = np.zeros(nClusters, dtype=np.int32)
        self.nClusters = nClusters  
        
        # elements
        self.clustConToExit = np.zeros(totElements, dtype=np.bool_)
        self.clusterID  = np.full(totElements, -5, dtype=np.int32)
        self.hasFluid = (network.fluid==phase)
        if phase==0:
            self.hasFluid |= network.isPolygon
        self.hasFluid[[-1,0]] = False
        self.conn = np.zeros(totElements, dtype=np.bool_)
        self.trapped = np.zeros(totElements, dtype=np.bool_)

        self.flowrate = 0.0
        self.flow_vec = np.zeros(totElements, dtype=np.float64)
        self.flow_dir = np.zeros(totElements, dtype=np.bool_)
        self.gL = np.zeros(network.nThroats, dtype=np.float64)


    def __getitem__(self, key):
        if key < len(self.values):
            return self.values[key]
        else:
            self[key] = None 
            return self.values[key]

    def __setitem__(self, key, value):
        while len(self.values) <= key:
            new_key = len(self.values)
            self.keys.append(new_key)
            self.values.append(ClusterObj(new_key, self, self.network))

    def __delitem__(self, key):
        try:
            self.pc[key] = 0.0
            if hasattr(self, 'moles') :
                self.moles[key] = 0.0
            if hasattr(self, 'volume'):
                self.volume[key] = 0.0  
        except IndexError:
            raise KeyError(f'Key "{key}" not found')

    def items(self):
        return zip(self.keys, self.values)

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'network' in state: 
            del state['network']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def resizeClusters(self, size):
        totElements = self.network.totElements
    
        oldSize = self.pc.size
        newSize = oldSize + size
        self.heads_arr = np.concatenate((self.heads_arr, np.full(size, -5, dtype=np.int32)))
        self.mem_offsets = np.concatenate((self.mem_offsets, np.full(size, -5, dtype=np.int32)))
        self._neighbours.extend([[] for _ in range(size)])

        for attr in ['pc', 'trappedStatus', 'connected', 'sizes']:
            tempArr = getattr(self, attr)
            newArr = np.zeros(newSize, dtype=tempArr.dtype)
            newArr[:oldSize] = tempArr
            setattr(self, attr, newArr)
        
        for c in np.arange(self.nClusters, self.nClusters+size):
            self[c] = {'key': c, 'parent':self.network}
        
        self.nClusters += size     

    
    def fill_with_phase(self, i, Pc, network):
        network.fluid[i] = self.phase
        if self.hasFluid[i]: return
        self.hasFluid[i] = True
        self.doClustering(np.asarray([i]), Pc, True, True, True)
       
       
    def unfill_phase(self, i, Pc):
        ntwk = self.network
        k = self.clusterID[i]
        self.sizes[k] -= 1
        self.hasFluid[i] = False
        self.clusterID[i] = -5
        self.trapped[i] = False

        start, end = ntwk.cg_offsets[i], ntwk.cg_offsets[i+1]
        neigh = ntwk.connectivity_graph_flat[start:end]
        neigh = neigh[self.hasFluid[neigh]]
        self.doClustering(neigh, Pc, True, True, True)
        

    def computeFlowrate(self, conductance, vector_mode=False):
        
        ntwk = self.network
        conn = (self.conn & ntwk.isinsideBox).astype(np.bool_)

        compute_gL_numba(ntwk.P1array, ntwk.P2array, ntwk.tList, 
            ntwk.LP1array_mod, ntwk.LP2array_mod, ntwk.LTarray_mod,
            conductance, ntwk.nThroats, self.gL)
        
        row, col, data, Cmatrix, indP = build_Amatrix_Cmatrix(
            conn, ntwk.nPores, ntwk.nThroats, ntwk.poreList, ntwk.throatList, ntwk.tList, 
            ntwk.P1array, ntwk.P2array, ntwk.isOnInletBdr, ntwk.isOnOutletBdr, self.gL)
        c = indP.size
        Amatrix = csr_matrix((data, (row, col)), shape=(c, c))

        pres = np.zeros(ntwk.nPores+2, dtype=np.float64)
        
        pres[indP] = do.matrixSolver(Amatrix, Cmatrix)
        self.flowrate = compute_qp_numba(
            ntwk.P1array, ntwk.P2array, ntwk.tList, self.gL, ntwk.nThroats, pres, ntwk.poreList, 
            c, conn, ntwk.isOnInletBdr, vector_mode, ntwk.is_conTToInletBdr.copy(), 
                ntwk.is_conTToOutletBdr.copy(), self.flow_vec, self.flow_dir)
        
        

    def doClustering(self, arr, pc_val, updateCluster=False, updateConnectivity=False, updatePcClustConToExit=True):
        if arr.size==0: return
        ntwk = self.network
        totElements = ntwk.totElements
        
        temp_clusterID = np.full(totElements, -5, dtype=np.int32)
        if np.ndim(pc_val) == 0:
            pc_val = np.full(arr.size, pc_val, dtype=np.float32)
        
        try:
            parallel_exploration_dsu_with_assigning(arr, pc_val, self.hasFluid, ntwk.connectivity_graph_flat, 
                ntwk.cg_offsets, self.sizes, self.clusterID, self.pc, temp_clusterID, ntwk.toInlet, ntwk.toOutlet, 
                ntwk.toInBdr, ntwk.toOutBdr, _temp.done, _temp.visited, self.members, self.mem_offsets, 
                self.heads_arr, self.next_elem_arr, self.trapped, self.trappedStatus, self.conn, self.connected,
                self.neighbours_updated, updateConnectivity)
        except RuntimeError:
            self.resizeClusters(min(200, (~self.hasFluid[arr]).sum()+1))
            temp_clusterID[arr] = -5
            parallel_exploration_dsu_with_assigning(arr, pc_val, self.hasFluid, ntwk.connectivity_graph_flat, 
                ntwk.cg_offsets, self.sizes, self.clusterID, self.pc, temp_clusterID, ntwk.toInlet, ntwk.toOutlet, 
                ntwk.toInBdr, ntwk.toOutBdr, _temp.done, _temp.visited, self.members, self.mem_offsets, 
                self.heads_arr, self.next_elem_arr, self.trapped, self.trappedStatus, self.conn, self.connected,
                self.neighbours_updated, updateConnectivity)
            
            

class ClusterObj:
    def __init__(self, key, parent, network):
        self.network = network
        self.key = key
        self.parent = parent
        self._neighbours = np.array([])
        self._members = np.array([])
        
    @property
    def phase(self):
        return self.parent.phase
    
    @property
    def pc(self):
        return self.parent.pc[self.key]
    
    @property
    def trapped(self):
        return self.parent.trappedStatus[self.key]
    
    @property
    def connected(self):
        return self.parent.connected[self.key]
    
    @property
    def members(self):
        '''returns the surrounding elements to this cluster'''
        p = self.parent
        k = self.key
        start, end = p.mem_offsets[k], p.mem_offsets[k+1]
        return p.members[start:end]
            

    @property
    def neighbours(self):
        '''returns the surrounding elements to this cluster'''
        parent = self.parent
        if parent.neighbours_updated[self.key]:
            return self._neighbours
        else:
            self._neighbours = getNeighbours(self.key, self.members, parent.hasFluid, 
                parent.network.connectivity_graph_flat, parent.network.cg_offsets, 
                parent.network.totElements)
            parent.neighbours_updated[self.key] = True
            return self._neighbours
    
    @property
    def volume(self):
        '''returns the volume of a cluster'''
        try:
            return self.parent.volume[self.key]
        except AttributeError:
            return (self.parent.volarray[self.members]).sum()
    
    @property
    def moles(self):
        '''returns the mole of a cluster'''
        return self.parent.cWP[self.key]
              
    def items(self):
        items = {k: v for k, v in self.__dict__.items() if k != "network"}
        return items

    def __str__(self):
        return f'{self.items()}'
    
    def __repr__(self):
        return self.__str__()



@njit(cache=True)
def getNeighbours(k, mem, hasFluid, cg_data, cg_offsets, totElements):
    
    neigh = np.empty(totElements, dtype=np.int32)
    done = np.zeros(totElements, dtype=np.bool_)
    n = mem.size
    j = 0
    for i in range(n):
        curr = mem[i]
        start, end = cg_offsets[curr], cg_offsets[curr + 1]

        for curr in cg_data[start:end]:
            if not hasFluid[curr] and not done[curr]:
                neigh[j] = curr
                done[curr] = True
                j += 1

    return neigh[:j]



@njit(int32[:]
    (int32, boolean[:], boolean[:], int32[:], int32[:], int32[:]), cache=True)
def find_cluster_members_numba(ii, valid, done, cg_data, cg_offsets, visited):
    visited[0] = ii
    done[done] = False
    done[ii] = True

    i, j = 0, 1
    while i<j:
        current = visited[i]
        i += 1
        arr = cg_data[cg_offsets[current]:cg_offsets[current+1]]
        arr = arr[valid[arr] & ~done[arr]]

        visited[j : j + arr.size] = arr
        done[arr] = True
        j += arr.size
       
    return visited[:j]


@njit(void
    (int32[:], int32[:], int32[:], float64[:], float64[:], float64[:], float32[:], int32, float64[:]
    ), parallel=True, cache=True)
def compute_gL_numba(P1array, P2array, tList, LP1, LP2, LT, g, nThroats, gL):
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


@njit(Tuple((int32[:], int32[:], float64[:], float64[:], int32[:]))
    (boolean[:], int32, int32, int32[:], int32[:], int32[:], 
    int32[:], int32[:], boolean[:], boolean[:], float64[:]), parallel=True, cache=True)
def build_Amatrix_Cmatrix(arrr, nPores, nThroats, poreList, throatList, tList, 
    P1array, P2array, isOnInletBdr, isOnOutletBdr, gL):
    
    indP = poreList[arrr[poreList]]
    c = indP.size
    mList = -np.ones(nPores+2, dtype=np.int32)
    mList[indP] = np.arange(c, dtype=np.int32)
    indT = throatList[arrr[tList]]        
    tSize = indT.size
    max_entries = 4 * tSize

    row_tmp = np.full(max_entries, -1, dtype=np.int32)
    col_tmp = np.full(max_entries, -1, dtype=np.int32)
    data_tmp = np.zeros(max_entries, dtype=np.float64)
    entry_counts = np.zeros(tSize, dtype=np.int32)
    Cmatrix_cond = np.zeros(tSize, dtype=np.float64)
    Cmatrix_ind = np.full(tSize, -1, dtype=np.int32)

    for i in prange(tSize):
        t = indT[i]-1
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
    col = np.empty(total_entries, dtype=np.int32)
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

    return row, col, data, Cmatrix, indP
    

@njit(float64
    (int32[:], int32[:], int32[:], float64[:], int64, float64[:], int32[:], int64, boolean[:], 
    boolean[:], boolean, boolean[:], boolean[:], float64[:], boolean[:]), parallel=True, cache=True)
def compute_qp_numba(P1array, P2array, tList, gL, nThroats, pres, poreList, c, arrr,
    isOnInletBdr, vector_mode, conTToInletBdr, conTToOutletBdr, flow_vec, flow_dir):
            
    indP = np.flatnonzero(arrr[poreList])+1
    for i in prange(c):
        P = indP[i]
        if arrr[P] and isOnInletBdr[P]:
            pres[P] = 1.0

    for i in prange(nThroats):
        P1, P2, t = P1array[i], P2array[i], tList[i]
        if conTToInletBdr[i] and not arrr[t]:
            conTToInletBdr[i] = False
        if conTToOutletBdr[i] and not arrr[t]:
            conTToOutletBdr[i] = False
            
        delP = abs(pres[P1] - pres[P2])
        flow_vec[i] = gL[i] * delP
        if vector_mode:
            flow_dir[i] = pres[P1]<=pres[P2]
                
    qinto = np.sum(flow_vec[conTToInletBdr])
    qout = np.sum(flow_vec[conTToOutletBdr])           
    if not vector_mode and abs(qinto - qout)<1e-30:
        qout = (qinto + qout)/2.0
        
    return qout


@njit(cache=True)
def parallel_exploration_dsu_with_assigning(arr, pc_val, hasFluid, cg_data, cg_offsets, 
    sizes, clusterID, clusterPc, temp_clusterID, toInlet, toOutlet, toInBdr, toOutBdr, 
    done, visited, members, mem_offsets, heads_arr, next_elem_arr, trapped_arr, 
    trappedStatus, conn_arr, connected, neighbours_updated, updateConnectivity):
    
    nRoots = arr.size
    free = 1
    nClusters = sizes.size
    for i in range(nRoots):
        ii = arr[i]
        if temp_clusterID[ii] != -5: continue
        has_in, has_out, has_inB, has_outB = False, False, False, False
        curr = doClustering_numba(ii, hasFluid, done, cg_data, cg_offsets, visited)
        
        temp_clusterID[curr] = i
        if toInlet[curr].any(): has_in = True
        if toOutlet[curr].any(): has_out = True
        if toInBdr[curr].any(): has_inB = True
        if toOutBdr[curr].any(): has_outB = True 

        is_connected = has_inB and has_outB
        is_trapped = not (has_in or has_out)
        
        if is_connected:
            cid = 0
        else:
            while sizes[free] > 0:
                free += 1
                if free == nClusters: 
                    raise RuntimeError("More clusters need to be formed !!!")
            cid = free
            free += 1

        heads_arr[cid] = curr[0]
        next_elem_arr[curr[:-1]] = curr[1:]
        next_elem_arr[curr[-1]] = -5
        old_cid = clusterID[curr]
        
        neighbours_updated[old_cid] = False
        clusterID[curr] = cid
        sizes[cid] += curr.size
        connected[cid] = is_connected
        trappedStatus[cid] = is_trapped
        trapped_arr[curr] = is_trapped
        if updateConnectivity: conn_arr[curr] = is_connected
        
        neighbours_updated[cid] = False
        clusterPc[cid] = pc_val[i]

        old_cid = old_cid[old_cid >= 0]
        while old_cid.size > 0:
            cid0 = old_cid[0]
            cond = (old_cid == cid0)
            sizes[cid0] -= cond.sum()
            old_cid = old_cid[~cond]

    j = 0
    n = clusterID.size
    for k in range(nClusters):
        k_size = sizes[k]
        curr = heads_arr[k]
        mem_offsets[k] = j
        if k_size==0: continue
        while curr>0 and j<n:
            members[j] = curr
            curr = next_elem_arr[curr]
            j += 1

        if k==nClusters:
            mem_offsets[k+1] = j

    members[j:] = -5

       

@njit(parallel=True, cache=True)
def doClustering_numba_parallel(arr, valid, connectivity_graph, cg_offsets, totElements):
    
    n_seeds = arr.size
    clusters = np.full((n_seeds, totElements), -5, dtype=np.int32)
    cluster_sizes = np.zeros(n_seeds, dtype=np.int32)

    for s in prange(n_seeds):
        seed = arr[s]
        visited = np.empty(totElements, dtype=np.int32)
        done = np.zeros(totElements, dtype=np.bool_)

        visited[0] = seed
        done[seed] = True
        i, j = 0, 1

        while i < j:
            current = visited[i]
            i += 1

            start = cg_offsets[current]
            end   = cg_offsets[current + 1]

            for n in connectivity_graph[start:end]:
                if valid[n] and not done[n]:
                    visited[j] = n
                    done[n] = True
                    j += 1

        clusters[s, :j] = visited[:j]
        cluster_sizes[s] = j

    return clusters, cluster_sizes


@njit(parallel=True)
def doClustering_batch(batch, valid, cg_data, cg_offsets, visited, done, 
    cluster_sizes, totElements):

    n_batch = batch.size

    for b in prange(n_batch):
        seed = batch[b]
        visited[b, 0] = seed
        done[b, seed] = True

        i, j = 0, 1
        while i < j:
            current = visited[b, i]
            i += 1

            start = cg_offsets[current]
            end   = cg_offsets[current + 1]

            for n in cg_data[start:end]:
                if valid[n] and not done[b, n]:
                    visited[b, j] = n
                    done[b, n] = True
                    j += 1

        cluster_sizes[b] = j

    return n_batch



@njit(cache=True)
def filter_and_assign_clusters(clusters, cluster_sizes, temp_clusterID):
    
    n_seeds = clusters.shape[0]
    valid_seeds = np.full(n_seeds, -5, dtype=np.int32)
    k = 0

    for cid in range(n_seeds):
        size = cluster_sizes[cid]
        if size == 0:
            continue

        mem = clusters[cid, :size]
        is_valid = True
        for m in mem:
            if temp_clusterID[m] != -5:
                is_valid = False
                break

        if is_valid:
            temp_clusterID[mem] = k
            valid_seeds[k] = cid
            k += 1

    return valid_seeds[:k], k


@njit(cache=True)
def filter_and_assign_clusters_batch(visited, cluster_sizes, 
    temp_clusterID, valid_seeds, start, n_seeds):
    
    k = start
    j = 0
    for i in range(n_seeds):
        size = cluster_sizes[i]
        if size == 0: continue

        mem = visited[i, :size]
        is_valid = True
        for m in mem:
            if temp_clusterID[m] != -5:
                is_valid = False
                break

        if is_valid:
            for m in mem:
                temp_clusterID[m] = k
            valid_seeds[j] = i
            k += 1
            j += 1

    return j

        

   

@njit(cache=True)
def doClustering_numba(ii, valid, done, connectivity_graph, cg_offsets, visited):
    visited[:] = -5
    visited[0] = ii
    done[:] = False
    done[ii] = True

    i, j = 0, 1
    while i<j:
        current = visited[i]

        i += 1
        arr = connectivity_graph[cg_offsets[current]:cg_offsets[current+1]]
        k = 0
        for n in arr:
            if valid[n] and not done[n]:
                visited[j + k] = n
                done[n] = True
                k += 1

        j += k

    return visited[:j].copy()

