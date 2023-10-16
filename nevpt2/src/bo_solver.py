import numpy as np
import copy
from scipy import linalg as La

class BO_solver:

      def __init__(self,BO=None):
          self.BO = BO

      def solve_with_scf(self):
          from pyscf import gto,scf,ao2mo
          mol = gto.M(verbose=4)
          mol.nelectron     = self.BO.ne[0]+self.BO.ne[1]
          mol.spin          = self.BO.ne[0]-self.BO.ne[1]
          mol.incore_anyway = True
          mol.nao_nr        = lambda *args : self.BO.no
          mol.energy_nuc    = lambda *args : self.BO.h0
          if(mol.spin==0): mf = scf.RHF(mol)
          else:            mf = scf.ROHF(mol)
          mf.get_hcore = lambda *args: self.BO.h1
          mf.get_ovlp  = lambda *args: self.BO.s1
          mf._eri      = ao2mo.restore(1,self.BO.h2,self.BO.no)
      
          mf = scf.newton(mf)
          E0 = mf.kernel(self.BO.dm)
          if(not mf.converged):
             mf = scf.newton(mf)
             E0 = mf.kernel(mf.make_rdm1())
          return mf,E0

      def solve_with_fci(self,nroots=1):
          from pyscf import fci
          cisolver           = fci.direct_nosym.FCI()
          cisolver.max_cycle = 100
          cisolver.conv_tol  = 1e-8
          cisolver.nroots    = nroots
          ECI,fcivec         = cisolver.kernel(self.BO.h1,self.BO.h2,self.BO.no,(self.BO.ne[0],self.BO.ne[1]),ecore=self.BO.h0)
          if(nroots==1):
             return ECI,fcivec
          else:
             nroots = len(fcivec)
             dims   = fcivec[0].shape
             fcimat = np.zeros((dims[0],dims[1],nroots))
             evec   = np.zeros(nroots)
             for m,(E,f) in enumerate(zip(ECI,fcivec)):
                 evec[m]       = E
                 fcimat[:,:,m] = f
          return evec,fcimat

