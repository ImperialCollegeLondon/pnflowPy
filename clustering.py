import numpy as np
from sortedcontainers import SortedList
    

class Cluster():
    def __init__(self, obj, fluid=1, numClusters=200):
        self.obj = obj
        self.fluid = fluid
        self.keys = [0]
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
            self.keys.append(key)
            self.values.append(ClusterObj(key, self, self.obj))
            if self[key].key!=self.keys[key]:
                print('index is not same as key, there is a problem here!!!')
                from IPython import embed; embed()

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
    
    def clustering(self, mem, arrDict, Pc, cluster_ID, cluster, trapped,        
                   updatePcClustConToInlet):
        oldkeys = cluster_ID[mem]
        oldMem = mem[oldkeys>=0]
        oldkeys = oldkeys[oldkeys>=0]
        cluster.members[oldkeys, oldMem] = False #uncluster previously clustered elements
        arrDictKeys = np.fromiter(arrDict.keys(), dtype=int)
        try:
            oldkeys = oldkeys[oldkeys>0]
            assert oldkeys.size==0
        except AssertionError:
            oldkeys = np.array(list(set(oldkeys)))
            availClust = oldkeys[~cluster.members[oldkeys].any(axis=1)] #newly available clusters
            cluster.availableID.update(np.setdiff1d(availClust,self.availableID))
            try:
                assert oldkeys.size<=1
            except:
                arrDictKeys = arrDictKeys[oldkeys.argsort()]
       
        for k in arrDictKeys:
            members = self.obj.elementListS[arrDict[k]['members']]
            try:
                assert arrDict[k]['connStatus']
                cluster_ID[members] = 0
                cluster.members[0][members] = True
                trapped[members] = False
                cluster.clustConToExit[0] = True
                cluster.trappedStatus[0] = False
            except AssertionError:
                try:
                    ct = cluster.availableID.pop(0)
                except IndexError:
                    # double previous size/add 500 new clusters
                    oldSize = cluster.pc.size
                    addSize = min(oldSize, 500)
                    self.resizeClusters(addSize, cluster)
                    id = np.setdiff1d(np.where(cluster.size==0)[0], cluster.availableID)
                    cluster.availableID.update(id[id>0])
                    ct = cluster.availableID.pop(0)

                cluster_ID[members] = ct
                cluster.members[ct][members] = True
                cluster[ct] = {'key':ct, }
                cluster.pc[ct] = Pc
                trapped[members] = arrDict[k]['trappedStatus']
                cluster.clustConToExit[ct] = arrDict[k]['members'][self.obj.conTToExit].any()
                cluster.trappedStatus[ct] = arrDict[k]['trappedStatus']
                cluster.connected[ct] = False
                
        try:
            assert updatePcClustConToInlet
            cluster.pc[cluster.clustConToExit] = Pc
        except AssertionError:
            pass

        return
    
    def resizeClusters(self, size, cluster):
        cluster.members = np.vstack(
            (cluster.members, np.zeros([size,self.obj.totElements], dtype=bool)))
        cluster.pc = np.concatenate((cluster.pc, np.zeros(size)))
        cluster.trappedStatus = np.concatenate(
            (cluster.trappedStatus, np.zeros(size, dtype=bool)))
        cluster.connected = np.concatenate(
            (cluster.connected, np.zeros(size, dtype=bool)))
        cluster.clustConToExit = np.concatenate(
            (cluster.clustConToExit, np.zeros(size, dtype=bool)))
        for c in np.arange(len(self.keys), self.pc.size):
            self[c] = {'key': c}
            
            

    def updateNeighMatrix(self, cond=None):
        '''This updates the neighMatrix!!! might be later revised!!!'''
        try:
            assert cond is None
            cond = np.ones(self.obj.nThroats, dtype=bool)
        except AssertionError:
            pass
        
        try:
            assert self.fluid==1
            cluster_ID = self.obj.clusterNW_ID
        except AssertionError:
            cluster_ID = self.obj.clusterW_ID
       
        def _f(cond):
            P1array = self.obj.P1array[cond]
            P2array = self.obj.P2array[cond]
            tList = self.obj.tList[cond]
        
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
            try:
                (clustP1, clustP2, clustT, condP1_P1, condP1_T, 
                condP2_P2, condP2_T, condP1_P1_T, condP2_P2_T,
                P1array, P2array, tList) = _f(cond)
                assert condP1_P1_T.any() or condP2_P2_T.any()
                arr = np.sort(np.concatenate((
                    np.array([clustP1[condP1_P1_T], clustT[condP1_P1_T]]).T,
                    np.array([clustP2[condP2_P2_T], clustT[condP2_P2_T]]).T)), axis=1)
                arr = list(set(map(tuple, arr)))
                
                _arr = np.unique(arr)
                neigh = self.neighbours[_arr].any(axis=0)
                print(arr)
                self.coalesceClusters(arr, cluster_ID)
                neigh = neigh|self.members[_arr].any(axis=0)
                cond = cond | neigh[self.obj.tList]
            except AssertionError:
                keysToUpdate = np.unique(cluster_ID[self.obj.tList[cond]])
                keysToUpdate = keysToUpdate[keysToUpdate>=0]
                break

        clust = np.concatenate((clustP1[condP1_P1],  clustT[condP1_T], 
                                clustP2[condP2_P2], clustT[condP2_T]))
        neigh = np.concatenate((tList[condP1_P1],  P1array[condP1_T], 
                                tList[condP2_P2],  P2array[condP2_T]))

        self.neighbours[keysToUpdate] = False
        self.neighbours[clust, neigh] = True


    def coalesceClusters(self, arr, cluster_ID):
        ''' coalesce clusters together '''
        arr = [*map(np.array, arr)]
        values, counts = np.unique(arr, return_counts=True)
        
        def _f1(c, ar):
            # compute new moles, volume and pc
            term1 = self.volume[ar]/self.moles[ar]
            self.pc[c] = (self.pc[ar]*term1).sum()/term1.sum()
            self.moles[c] = self.moles[ar].sum()
            self.totalVolume[c] = self.totalVolume[ar].sum()
        
            _mem = self.members[ar].any(axis=0)
            mem = self.obj.elementListS[_mem]
            mem1 = mem[cluster_ID[mem]!=c]
            clustID = cluster_ID[mem1]
            cluster_ID[mem1] = c
            self.members[clustID, mem1] = False
            self.members[c, mem1] = True

            _f2(c, _mem, self.pc, self.obj)
            vol = (1-self.obj.cornerArea[mem]/self.obj.areaSPhase[mem])*self.obj.volarray[mem]
            self.volume[c] = vol.sum()

        def _f2(key, arr, newPc, other): 
            # compute the corner areas at the new pc and restore back the trapped status
            oldWStatus, oldNWStatus = other.trappedW[arr], other.trappedNW[arr]
            other.trappedW[arr], other.trappedNW[arr] = False, True
            oldPc = self.pc[key]
            self.pc[key] = newPc[key]
            pc = newPc[other.clusterNW_ID]
            other.__CondTPImbibition__(arr, pc, False)
            other.trappedW[arr], other.trappedNW[arr] = oldWStatus, oldNWStatus
            self.pc[key] = oldPc

        while True:
            try:
                c = values[np.argmax(counts)]
                arrC = [ar[ar!=c][0] for ar in arr if c in ar]
                self.neighbours[arrC] = False
                arrC1 = arrC.copy()
                arrC.append(c)
                _f1(c, arrC)
                for c1 in arrC1: del self[c1]
                arr = [ar for ar in arr if c not in ar]
                values, counts = np.unique(arr, return_counts=True)
            except ValueError:
                break
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
            #return (self.obj.volarray[self.members]*self.parent.satList[self.members]).sum()
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
    
    

