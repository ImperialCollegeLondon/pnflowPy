import numpy as np
from sortedcontainers import SortedList
from numba import njit, prange


class LookupClass:
    def __init__(self, totElements, nPores, conn_graph_size):
        self.done = np.zeros(totElements, dtype=np.bool_)
        self.filterNext = np.zeros(totElements, dtype=np.bool_)
        self.mList = np.zeros(nPores+2, dtype=np.int32)
        self.visited = np.empty(conn_graph_size, dtype=np.int32)


class Cluster():
    def __init__(self, obj, phase, numClusters=200):
        self.obj = obj
        self.nClusters = numClusters
        self.phase = phase
        self.keys = [0]*numClusters
        self.values = [ClusterObj(0, self, obj)]
        self.pc = np.zeros(numClusters)
        self.drainEvents = 0
        self.imbEvents = 0
        self.availableID = SortedList()
        self.availableID.update(np.arange(1,numClusters))
        self.members = np.zeros([numClusters, obj.totElements], dtype=np.bool_)
        self.trappedStatus = np.zeros(numClusters, dtype=np.bool_)
        self.connected = np.zeros(numClusters, dtype=np.bool_)
        self.size = np.zeros(numClusters, dtype=np.int32)
        self.fluid = obj.fluid.view()
        self.temp = LookupClass(obj.totElements, obj.nPores, obj.connectivity_graph_flat.size)
        
        # elements
        self.clustConToExit = np.zeros(obj.totElements, dtype=np.bool_)
        
        if phase==0:
            self.trapped = obj.trappedW.view()
            self.conn = obj.connW.view()
            self.hasFluid = obj.hasWFluid.view()
            self.clusterID = obj.clusterW_ID.view()
        else:
            self.trapped = obj.trappedNW.view()
            self.conn = obj.connNW.view()
            self.hasFluid = obj.hasNWFluid.view()
            self.clusterID = obj.clusterNW_ID.view()
        

    def __getitem__(self, key):
        if key in self.keys:
            index = self.keys.index(key)
            return self.values[index]
        else:
            self[key] = {'key': key}
    
    def __setitem__(self, key, value):
        if key not in self.keys:
            try:
                self.keys[key] = key
            except IndexError:
                self.keys.append(key)
            self.values.append(ClusterObj(key, self, self.obj))
            
    def __delitem__(self, key):
        if key in self.keys:
            index = self.keys.index(key)
            self.pc[index] = 0.0
            if hasattr(self, 'moles'):
                self.moles[index] = 0.0
            if hasattr(self, 'volume'):
                self.volume[index] = 0.0     
        else:
            raise KeyError(f'Key "{key}" not found')
        
    def items(self):
        return zip(self.keys, self.values)
    
    
    def doClustering(self, arr, notdone, Pc, updateCluster=False,
                    updateConnectivity=False, updatePcClustConToExit=True):

        if updateConnectivity:
            self.conn.fill(False)
        
        done = self.temp.done
        while arr.size:
            ii = arr[0]
            doClustering_numba(ii, notdone, done, self.obj.connectivity_graph_flat, 
                               self.obj.cg_offsets, self.temp.visited)

            _done = np.flatnonzero(done)
            trappedStatus = not (self.obj.toInlet[_done].any() or self.obj.toOutlet[_done].any())
            connStatus = self.obj.toInBdr[_done].any() and self.obj.toOutBdr[_done].any()
            
            if updateCluster:
                self.clustering(_done, Pc, trappedStatus, connStatus, updatePcClustConToExit)
                

            if updateConnectivity and connStatus:
                self.conn[_done] = True
                
            arr = arr[~done[arr]]

    
    def clustering(self, mem, Pc, trappedStatus, connStatus, updatePcClustConToInlet):
        oldkeys = self.clusterID[mem]
        if np.all(oldkeys==oldkeys[0]) and (mem.size == self.size[oldkeys[0]]):
            return
        
        oldMem = mem[oldkeys>=0]
        oldkeys = oldkeys[oldkeys>=0]
        
        if oldkeys.size>0:
            emptyKeys = removeMembers(
                oldkeys, oldMem, self.members, self.trapped, trappedStatus, self.size)
            if emptyKeys.size>0:
                self.availableID.update(emptyKeys)

        if connStatus:
            self.clusterID[mem] = 0
            addMembers(0, mem, self.members, self.trapped, trappedStatus, self.size)
            clustConToExit = [0]
            self.trappedStatus[0] = False
            self.connected[0] = True
        else:
            if len(self.availableID)==0:
                addSize = min(self.nClusters, 200)
                self.resizeClusters(addSize)

            ct = self.availableID.pop(0)
            self.clusterID[mem] = ct
            self[ct] = {'key':ct, 'parent':self}
            addMembers(ct, mem, self.members, self.trapped, trappedStatus, self.size)
                
            self.pc[ct] = Pc
            if not trappedStatus: clustConToExit = [ct]
            self.trappedStatus[ct] = trappedStatus
            self.connected[ct] = False
        
        if updatePcClustConToInlet and not trappedStatus:
            self.pc[clustConToExit] = Pc

        return
    

    def fill_with_phase(self, i, Pc):
        self.fluid[i] = self.phase
        if self.hasFluid[i]: return
        self.hasFluid[i] = True
        self.doClustering(np.asarray([i]), self.hasFluid, Pc, True, False, True)


    def unfill_phase(self, i, Pc):
        oldID = self.clusterID[i]
        self.members[oldID, i] = False
        self.size[oldID] -= 1
        self.hasFluid[i] = False
        self.clusterID[i] = -5
        neigh = self.obj.connectivity_graph[i]
        neigh = neigh[self.hasFluid[neigh] & (neigh>0)]
        if neigh.any():
            self.doClustering(neigh, self.hasFluid, Pc, True, False, True)


    def computeFlowrate(self, gL):
        active = gL>0.0
        return computeFlowrate_numba(active, self.obj.poreList, self.obj.throatList, 
                                     self.obj.tList, self.obj.conTToIn, 
                                     self.hasFluid, self.obj.connected, self.conn,
                                     self.obj.toInBdr, self.obj.toOutBdr, self.obj.isinsideBox,
                                     self.obj.connectivity_graph_flat, self.obj.cg_offsets,
                                     self.temp.done, self.temp.mList, self.temp.visited)
    


    def resizeClusters(self, size):
        self.members = np.vstack(
            (self.members, np.zeros([size,self.obj.totElements], dtype=np.bool_)))
        self.pc = np.concatenate((self.pc, np.zeros(size)))
        self.trappedStatus = np.concatenate(
            (self.trappedStatus, np.zeros(size, dtype=bool)))
        self.connected = np.concatenate(
            (self.connected, np.zeros(size, dtype=bool)))
        self.size = np.concatenate((self.size, np.zeros(size, dtype=bool)))
        for c in np.arange(len(self.keys), self.pc.size):
            self[c] = {'key': c}
        self.availableID.update(np.flatnonzero(self.size==0))

            
            
    def updateNeighMatrix(self, other, cond=None):
        '''This updates the neighMatrix!!! might be later revised!!!'''
        if cond is None:
            cond = np.ones(other.nThroats, dtype=bool)
        
        cluster_ID =  other.clusterNW_ID if self.phase==1 else other.clusterW_ID
        def _f(cond):
            P1array = other.P1array[cond]
            P2array = other.P2array[cond]
            tList = other.tList[cond]
        
            clustP1 = cluster_ID[P1array]
            clustP2 = cluster_ID[P2array]
            clustT = cluster_ID[tList]

            condT = (clustT>=0)
            condP1 = (P1array>0) & (clustP1!=clustT)
            condP1_P1 = condP1 & (clustP1>=0) # T is neighbour to P1
            condP1_T = condP1 & condT   # P1 is neighbour to T
            condP2 = (P2array>0) & (clustP2!=clustT)
            condP2_P2 = condP2 & (clustP2>=0) & (clustP2 != clustP1) # T is neighbour to P2
            condP2_T = condP2 & condT   # P2 is neighbour to T

            ''' check if there is any coalescence '''
            condP1_P1_T = condP1_P1 & condT # P1 and T should coalesce together
            condP2_P2_T = condP2 & condT & (clustP2>=0) # P2 and T should coalesce together
            return (clustP1, clustP2, clustT, condP1_P1, condP1_T, 
                    condP2_P2, condP2_T, condP1_P1_T, condP2_P2_T, P1array, P2array, tList)
        
        while True:
            (clustP1, clustP2, clustT, condP1_P1, condP1_T, 
            condP2_P2, condP2_T, condP1_P1_T, condP2_P2_T,
            P1array, P2array, tList) = _f(cond)
            if condP1_P1_T.any() or condP2_P2_T.any():
                arr = np.sort(np.concatenate((
                    np.array([clustP1[condP1_P1_T], clustT[condP1_P1_T]]).T,
                    np.array([clustP2[condP2_P2_T], clustT[condP2_P2_T]]).T)), axis=1)
                arr = list(set(map(tuple, arr)))
                
                _arr = np.unique(arr)
                neigh = self.neighbours[_arr].any(axis=0)
                self.coalesceClusters(arr, cluster_ID, other)
                    
                neigh = neigh|self.members[_arr].any(axis=0)
                cond = cond | neigh[other.tList]
            else:
                keysToUpdate = np.unique(cluster_ID[other.tList[cond]])
                keysToUpdate = keysToUpdate[keysToUpdate>=0]
                break

        clust = np.concatenate((clustP1[condP1_P1],  clustT[condP1_T], 
                                clustP2[condP2_P2], clustT[condP2_T]))
        neigh = np.concatenate((tList[condP1_P1],  P1array[condP1_T], 
                                tList[condP2_P2],  P2array[condP2_T]))

        self.neighbours[keysToUpdate] = False
        self.neighbours[clust, neigh] = True

        return


    def coalesceClusters(self, arr, cluster_ID, other):
        ''' coalesce clusters together '''
        arr = [*map(np.array, arr)]
        values, counts = np.unique(arr, return_counts=True)
        
        def _f1(c, ar, other):
            # compute new moles, volume and pc
            print(f'coalesced clusters: {ar}')
            _mem = self.members[ar].any(axis=0)
            mem = other.elementListS[_mem]
            if _mem[other.conTToIn].any() and _mem[other.conTToOut].any():
                c=0
                ar = ar[ar!=0]
                self.moles[c] += self.moles[ar].sum()
                self.volume[c] += self.volume[ar].sum()
            else:
                self.moles[c] = self.moles[ar].sum()
                self.volume[c] = self.volume[ar].sum()
            
            pc = np.append(self.pc[c], self.pc[ar])
            pc = pc[pc>other.Pc]
            if pc.size==0:
                self.pc[c] = other.Pc
            else:
                self.pc[c] = pc[pc>other.Pc].min()
        
            mem1 = mem[cluster_ID[mem]!=c]
            clustID = cluster_ID[mem1]
            cluster_ID[mem1] = c
            self.members[clustID, mem1] = False
            self.members[c, mem1] = True
            return c
        
        while counts.size>0:
            c = values[np.argmax(counts)]
            arrC = [ar[ar!=c][0] for ar in arr if c in ar]
            self.neighbours[arrC] = False
            arrC.append(c)
            c1 = _f1(c, np.array(arrC), other)
            if c1 in arrC: arrC.remove(c1)
            for c2 in arrC: del self[c2]
            arr = [ar for ar in arr if c not in ar]
            values, counts = np.unique(arr, return_counts=True)
            
        return
    
    
class ClusterObj:
    def __init__(self, key, parent, obj):
        self.obj = obj
        self.key = key
        self.parent = parent
        
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
        return self.obj.elementListS[self.parent.members[self.key]]

    @property
    def neighbours(self):
        '''returns the surrounding elements to this cluster'''
        try:
            return self.obj.elementListS[self.parent.neighbours[self.key]]
        except AttributeError:
            return np.array([], dtype=int)
    
    @property
    def volume(self):
        '''returns the volume of a cluster'''
        try:
            return self.parent.volume[self.key]
        except AttributeError:
            return (self.parent.volarray[self.members]).sum()
    
    @property
    def moles(self):
        '''returns the volume of a cluster'''
        return self.parent.moles[self.key]
              
    def items(self):
        items = {k: v for k, v in self.__dict__.items() if k != "obj"}
        return items

    def __str__(self):
        return f'{self.items()}'
    
    def __repr__(self):
        return self.__str__()
    


@njit(cache=True)
def doClustering_numba(ii, valid, done, connectivity_graph, cg_offsets, visited):
    visited[0] = ii
    done.fill(False)
    done[ii] = True

    i, j = 0, 1
    while i<j:
        current = visited[i]
        i += 1
        for k in connectivity_graph[cg_offsets[current]:cg_offsets[current+1]]:
            if valid[k] and not done[k]:
                visited[j] = k
                done[k] = True
                j += 1   
    return


@njit(cache=True)
def removeMembers(keys, mem, members, trapped, trappedStatus, size):
    n = keys.size
    for i in range(n): members[keys[i], mem[i]] = False
    trapped[mem] = trappedStatus
    kk = np.unique(keys)
    size[kk] -= np.bincount(keys)[kk]
    emptyKeys = kk[(size[kk]==0)&(kk!=0)]
    return emptyKeys


@njit(cache=True)
def addMembers(k, mem, members, trapped, trappedStatus, size):
    members[k, mem] = True
    trapped[mem] = trappedStatus
    size[k] += mem.size
            

@njit(parallel=True, cache=True)
def computeFlowrate_numba(active, poreList, throatList, tList, conTToIn, hasFluid, 
                          connected, conn, toInBdr, toOutBdr, isinsideBox, connectivity_graph,
                          cg_offsets, done, mList, visited):
    

    arrr = hasFluid & connected
    arrr[tList] &= active
    arrTToIn = conTToIn[arrr[conTToIn]]
    conn.fill(False)

    i = 0
    for i in range(arrTToIn.size):
        ii = arrTToIn[i]
        doClustering_numba(ii, arrr, done, connectivity_graph, cg_offsets, visited)
        _done = np.flatnonzero(done)
        connStatus = toInBdr[_done].any() and toOutBdr[_done].any()
        if connStatus: 
            conn[_done] = True
            break
        
    
    conn &= isinsideBox
    indP = poreList[conn[poreList]]
    indT = throatList[conn[tList]]
    c = indP.size
    mList.fill(-1)
    mList[indP] = np.arange(c)

    return mList, indT, c, indP
    

    



