import numpy as np


class TempArrays:
	def __init__(self):
	    pass

	@classmethod
	def formTempNetworkArrays(cls, nPores, totElements, conn_graph_size):
		cls.done = np.zeros(totElements, dtype=np.bool_)
		cls.filterNext = np.zeros(totElements, dtype=np.bool_)
		cls.mList = np.zeros(nPores+2, dtype=np.int32)
		cls.visited = np.empty(conn_graph_size, dtype=np.int32)