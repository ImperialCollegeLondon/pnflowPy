import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np

SCALE_FACTOR = 1e12

class Solver():
    def __init__(self, Amat, Cmat):
        comm = PETSc.COMM_WORLD
        #A = PETSc.Mat()
        #A.create(comm)
        m = Cmat.size
        A = PETSc.Mat().createAIJ(size=(m, m), csr=(
            Amat.indptr, Amat.indices, Amat.data*SCALE_FACTOR), comm=comm)
        self.C = PETSc.Vec().createWithArray(Cmat*SCALE_FACTOR, comm=comm)
    
        A.setUp()

        self.ksp = PETSc.KSP().create(comm=comm)
        # ksp.create()
        self.ksp.setType('cg')
        self.ksp.setTolerances(atol=1e-12, rtol=1e-12)
        # ILU
        #self.ksp.getPC().setType('jacobi')
        self.ksp.getPC().setType('hypre')
        opts = PETSc.Options()
        opts['pc_hypre_type'] = 'boomeramg'

        # CREATE INITIAL GUESS
        self.psol = PETSc.Vec().createWithArray(np.ones(m))
        # SOLVE
        self.ksp.setOperators(A)
        self.ksp.setFromOptions()

    def solve(self):
        self.ksp.solve(self.C, self.psol)
        return self.psol.getArray()
        


