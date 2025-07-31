import numpy as np
from sortedcontainers import SortedList
import tPhaseImb as tPhaseImb

class Cluster():
    def __init__(self, obj, fluid=1, numClusters=200):
        self.obj = obj
        self.fluid = fluid
        self.keys = [0]*numClusters
        self.values = [ClusterObj(0, self, obj)]
        self.pc = np.zeros(numClusters)
        self.drainEvents = 0
        self.imbEvents = 0
        self.availableID = SortedList()
        self.availableID.update(np.arange(1,numClusters))
        self.members = np.zeros([numClusters, obj.totElements], dtype=bool)
        self.trappedStatus = np.zeros(numClusters, dtype=bool)
        self.connected = np.zeros(numClusters, dtype=bool)
        self.clustConToExit = self.members[:, obj.conTToExit].any(axis=1)
        
    @property
    def size(self):
        '''returns the number of elements in a cluster'''
        return self.members.sum(axis=1)
    
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
    
    def clustering(self, mem, arrDict, Pc, cluster_ID, trapped,        
                   updatePcClustConToInlet):
        oldkeys = cluster_ID[mem]
        oldMem = mem[oldkeys>=0]
        oldkeys = oldkeys[oldkeys>=0]
        self.members[oldkeys, oldMem] = False #uncluster previously clustered elements
        arrDictKeys = np.fromiter(arrDict.keys(), dtype=int)
        oldkeys = oldkeys[oldkeys>0]
        if oldkeys.size>0:
            oldkeys = np.unique(oldkeys)
            availClust = oldkeys[~self.members[oldkeys].any(axis=1)] #newly available clusters
            newID = np.setdiff1d(availClust,self.availableID)
            self.availableID.update(newID)
            arrDictKeys.sort()
           
        for k in arrDictKeys:
            members = self.obj.elementListS[arrDict[k]['members']]
            if arrDict[k]['connStatus']:
                cluster_ID[members] = 0
                self.members[0][members] = True
                trapped[members] = False
                self.clustConToExit[0] = True
                self.trappedStatus[0] = False
                self.connected[0] = True
            else:
                if len(self.availableID)==0:
                    # double previous size/add 500 new clusters
                    oldSize = self.pc.size
                    addSize = min(oldSize, 200)
                    self.resizeClusters(addSize)
                    id = np.setdiff1d(np.where(self.size==0)[0], self.availableID)
                    self.availableID.update(id[id>0])

                ct = self.availableID.pop(0)
                cluster_ID[members] = ct
                self.members[ct][members] = True
                self[ct] = {'key':ct, 'parent':self}
                self.pc[ct] = Pc
                trapped[members] = arrDict[k]['trappedStatus']
                self.clustConToExit[ct] = arrDict[k]['members'][self.obj.conTToExit].any()
                self.trappedStatus[ct] = arrDict[k]['trappedStatus']
                self.connected[ct] = False
                
        if updatePcClustConToInlet:
            self.pc[self.clustConToExit] = Pc

        return
    
    def resizeClusters(self, size):
        self.members = np.vstack(
            (self.members, np.zeros([size,self.obj.totElements], dtype=bool)))
        self.pc = np.concatenate((self.pc, np.zeros(size)))
        self.trappedStatus = np.concatenate(
            (self.trappedStatus, np.zeros(size, dtype=bool)))
        self.connected = np.concatenate(
            (self.connected, np.zeros(size, dtype=bool)))
        self.clustConToExit = np.concatenate(
            (self.clustConToExit, np.zeros(size, dtype=bool)))
        for c in np.arange(len(self.keys), self.pc.size):
            self[c] = {'key': c}
            
            
    def updateNeighMatrix(self, other, cond=None):
        '''This updates the neighMatrix!!! might be later revised!!!'''
        if cond is None:
            cond = np.ones(other.nThroats, dtype=bool)
        
        cluster_ID =  other.clusterNW_ID if self.fluid==1 else other.clusterW_ID
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
    def fluid(self):
        return self.parent.fluid
    
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
    
    

