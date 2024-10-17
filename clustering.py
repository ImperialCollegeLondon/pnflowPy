import numpy as np
from sortedcontainers import SortedList
    

class ClusterManipulation():
    def __init__(self):
        pass

    def clustering(self, arrDict, fluid, Pc):
        try:
            assert fluid==0
            cluster_ID, cluster, trapped = self.clusterW_ID, self.clusterW, self.trappedW
        except AssertionError:
            cluster_ID, cluster, trapped = self.clusterNW_ID, self.clusterNW, self.trappedNW

        for k in arrDict.keys():
            try:
                assert arrDict[k]['connStatus']
                members = self.elementListS[arrDict[k]['members']]
                members1 = members[(cluster_ID[members]!=0)]
                members2 = members[(cluster_ID[members]>0)]
                availClust = np.array(list(set(cluster_ID[members2])), dtype=int)
                cluster_ID[members1] = 0
                cluster.members[:, members2] = False
                cluster.members[0][members] = True
                availClust = availClust[~cluster.members[availClust].any(axis=1)]
                cluster.availableID.update(availClust)
                trapped[members] = arrDict[k]['trappedStatus']
                cluster.clustConToInlet[0] = True
            except AssertionError:
                try:
                    ct = cluster.availableID.pop(0)
                except IndexError:
                    ''' more cluster rows need to be created '''
                    cluster.availableID.update(
                        np.where(~cluster.members.any(axis=1))[0])                    
                    try:
                        ct = cluster.availableID.pop(0)
                    except IndexError:
                        ct = cluster.pc.size
                        ct1 = min(ct, 500)  # do not create more than 500 new clusters at a time
                        cluster.members = np.vstack(
                            (cluster.members, np.zeros([ct1,self.totElements], dtype=bool)))
                        cluster.pc = np.concatenate((cluster.pc, np.zeros(ct1)))
                        cluster.trappedStatus = np.concatenate(
                            (cluster.trappedStatus, np.zeros(ct1, dtype=bool)))
                        cluster.connected = np.concatenate(
                            (cluster.connected, np.zeros(ct1, dtype=bool)))
                        cluster.clustConToInlet = np.concatenate(
                            (cluster.clustConToInlet, np.zeros(ct1, dtype=bool)))
                        cluster.availableID.update(np.arange(ct+1, ct+ct1))
    
                members = self.elementListS[arrDict[k]['members']]
                members1 = members[(cluster_ID[members]>=0)]
                members2 = members1[(cluster_ID[members1]>0)&(cluster_ID[members1]!=ct)]
                availClust = np.array(list(set(cluster_ID[members2])), dtype=int)
                cluster_ID[members] = ct
                cluster.members[:, members1] = False
                cluster.members[ct][members] = True
                availClust = availClust[~cluster.members[availClust].any(axis=1)]
                cluster.availableID.update(availClust)
                cluster.trappedStatus[ct] = arrDict[k]['trappedStatus']
                cluster.connected[ct] = arrDict[k]['connStatus']
                trapped[members] = arrDict[k]['trappedStatus']
                cluster[ct] = {'key':ct, 'fluid':fluid}
                cluster.pc[ct] = Pc
                cluster.clustConToInlet[ct] = arrDict[k]['members'][self.conTToIn].any()

        cluster.pc[cluster.clustConToInlet] = Pc
        return
        

    def updateNeighMatrix(self, cond=None):
        '''This updates the neighMatrix!!! might be later revised!!!'''
        try:
            assert cond is None
            cond = np.ones(self.nThroats, dtype=bool)
            P1array = self.P1array
            P2array = self.P2array
            tList = self.tList 
        except AssertionError:
            P1array = self.P1array[cond]
            P2array = self.P2array[cond]
            tList = self.tList[cond]

        def _f():
            clustP1 = self.clusterNW_ID[P1array]
            clustP2 = self.clusterNW_ID[P2array]
            clustT = self.clusterNW_ID[tList]

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
                    condP2_P2, condP2_T, condP1_P1_T, condP2_P2_T)
        
        while True:
            try:
                (clustP1, clustP2, clustT, condP1_P1, condP1_T, 
                condP2_P2, condP2_T, condP1_P1_T, condP2_P2_T) = _f()
                assert condP1_P1_T.any() or condP2_P2_T.any()
                arr = np.sort(np.concatenate((
                    np.array([clustP1[condP1_P1_T], clustT[condP1_P1_T]]).T,
                    np.array([clustP2[condP2_P2_T], clustT[condP2_P2_T]]).T)), axis=1)
                arr = list(set(map(tuple, arr)))
                self.coalesceClusters(arr)
            except AssertionError:
                break
        clust = np.concatenate((clustP1[condP1_P1],  clustT[condP1_T], 
                                clustP2[condP2_P2], clustT[condP2_T]))
        neigh = np.concatenate((tList[condP1_P1],  P1array[condP1_T], 
                                tList[condP2_P2],  P2array[condP2_T]))
        
        self.neighbours[clust] = False
        self.neighbours[clust, neigh] = True


    def coalesceClusters(self, arr):
        ''' coalesce clusters together '''
        arr = np.array([*map(np.array, arr)])
        values, counts = np.unique(arr, return_counts=True)
        c = values[np.argmax(counts)]
       
        def _f1(ar, c, self):
            ar = np.array(ar)
            kk = ar[ar!=c][0]
            self.clusterNW_ID[self.clusterNW_ID==kk] = c
            return kk
        
        def _f2(c, ar, self):
            # compute new moles, volume and pc
            newMoles = self.clusterNW[c].moles+self.clusterNW.moles[ar].sum()
            newVolume = self.clusterNW[c].volume+self.clusterNW.volume[ar].sum()
            newPc = ((newMoles/self.clusterNW[c].moles)*
                     (self.clusterNW[c].volume/newVolume)*self.clusterNW[c].pc)
            self.clusterNW.moles[c] = newMoles
            self.clusterNW.volume[c] = newVolume
            self.clusterNW.pc[c] = newPc
            self.clusterNW.members[:,ar] = False
            self.clusterNW.members[c, ar] = True

        while True:
            try:
                arrC = []
                [arrC.append(_f1(ar,c,self)) for ar in arr if c in ar]
                cond = ~(arr==c).any(axis=1)
                arr = arr[cond]
                _f2(c, arrC, self)
                assert arr.size>0
                values, counts = np.unique(arr, return_counts=True)
                c = values[np.argmax(counts)]
            except AssertionError:
                break
    
        return
    
    
class Cluster(ClusterManipulation):
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
        self.members = np.zeros([numClusters, self.totElements], dtype=bool)
        self.trappedStatus = np.zeros(numClusters, dtype=bool)
        self.connected = np.zeros(numClusters, dtype=bool)
        self.clustConToInlet = self.members[:, self.conTToIn].any(axis=1)
        
    @property
    def size(self):
        '''returns the number of elements in a cluster'''
        return self.members.sum(axis=1)
    
    def __getattr__(self, name):
        return getattr(self.obj, name)
    
    def __getitem__(self, key):
        if key in self.keys:
            index = self.keys.index(key)
            return self.values[index]
        else:
            raise KeyError(f'Key "{key}" not found')
    
    def __setitem__(self, key, value):
        if key not in self.keys:
            self.keys.append(key)
            self.values.append(ClusterObj(key, self, self.obj))

    def __delitem__(self, key):
        if key in self.keys:
            index = self.keys.index(key)
            #self.keys[index] = None
            self.values[index] = None
            self.Pc[index] = None
        else:
            raise KeyError(f'Key "{key}" not found')
        
    def items(self):
        return zip(self.keys, self.values)


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
        return self.parent.connStatus[self.key]
    
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
    def pcLow(self):
        '''returns the Pc value for imbibing the toImb element'''
        try:
            return self.parent.PcLow[self.key]
        except AttributeError:
            return -np.inf
        
    @property
    def pcHigh(self):
        '''returns the Pc value for draining the toDrain element'''
        try:
            return self.parent.PcHigh[self.key]
        except AttributeError:
            return np.inf
    
    @property
    def toImbibe(self):
        ''' returns a member of the cluster with the highest entry capillary pressure for imbibition'''
        try:
            return self.parent.toImbibe[self.key]
        except AttributeError:
            return -5
        
    @property
    def toDrain(self):
        '''returns a neighbouring element with the least entry capillary pressure  for drainage'''
        try:
            return self.parent.toDrain[self.key]
        except AttributeError:
            return -5
    
    @property
    def volume(self):
        '''returns the volume of a cluster'''
        try:
            return self.parent.volume[self.key]
        except AttributeError:
            #return (self.obj.volarray[self.members]*self.parent.satList[self.members]).sum()
            return (self.obj.volarray[self.members]).sum()
    
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
    

def returnIndex(arr):
    for idx, row in enumerate(arr):
        if np.all(row == False):  # Efficiently check if all values in the row are False
            return idx
    return None  # If no such row exists, return None
    

