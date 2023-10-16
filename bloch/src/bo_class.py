import numpy as np
import copy
from scipy import linalg as La

class BO_class:

      def __init__(self,mol=None,mf=None):
          from pyscf import ao2mo
          self.h0 = mol.energy_nuc()
          self.s1 = mf.get_ovlp()
          self.f1 = mf.get_fock()
          self.h1 = mf.get_hcore()
          self.h2 = ao2mo.restore(1,mf._eri,mol.nao_nr())
          self.no = mol.nao_nr()
          self.ne = ((mol.nelectron+mol.spin)//2,(mol.nelectron-mol.spin)//2)
          self.dm = mf.make_rdm1()
          if(self.ne[0]==self.ne[1] and len(self.dm.shape)==3):
             self.dm = self.dm[0,:,:] + self.dm[1,:,:]

      def copy(self):
          return copy.deepcopy(self)

      def transform_integrals(self,C):
          X = np.dot(self.s1,C)
          if(len(self.dm.shape)==2): self.dm = np.einsum('pi,pq,qj->ij',X,self.dm,X,optimize=True)
          else:                      self.dm = np.einsum('pi,xpq,qj->xij',X,self.dm,X,optimize=True)
          self.s1 = np.einsum('pi,pq,qj->ij',C,self.s1,C,optimize=True)
          self.f1 = np.einsum('pi,pq,qj->ij',C,self.f1,C,optimize=True)
          self.h1 = np.einsum('pi,pq,qj->ij',C,self.h1,C,optimize=True)
          self.h2 = np.einsum('pi,rj,qk,sl,prqs->ijkl',C,C,C,C,self.h2,optimize=True)
          self.no = C.shape[1]

      def downfold(self,to_remove):
          obj = self.copy()
          n   = obj.no
          to_keep = [i for i in range(n) if i not in to_remove]
          dne     = len([i for i in to_remove if i<min(self.ne)])
          obj.s1  = obj.s1[np.ix_(to_keep,to_keep)]
          obj.f1  = obj.f1[np.ix_(to_keep,to_keep)]
          obj.h1  = obj.h1[np.ix_(to_keep,to_keep)]
          obj.h2  = obj.h2[np.ix_(to_keep,to_keep,to_keep,to_keep)]
          obj.no  = obj.no-len(to_remove)
          obj.ne  = (obj.ne[0]-dne,obj.ne[1]-dne)
          if(len(obj.dm.shape)==2): obj.dm = obj.dm[np.ix_(to_keep,to_keep)]
          else:                     obj.dm = obj.dm[np.ix_([0,1],to_keep,to_keep)]
          return obj

      def freeze_orbitals(self,to_freeze):
          obj = self.copy()
          n   = obj.no
          rho_up = np.zeros((n,n))
          rho_dn = np.zeros((n,n))
          for i in to_freeze: rho_up[i,i]=1.0
          for i in to_freeze: rho_dn[i,i]=1.0
          dE  = np.einsum('ij,ji',obj.h1,rho_up+rho_dn)
          dE += 0.5*np.einsum('prqs,pr,qs',obj.h2,rho_up+rho_dn,rho_up+rho_dn)
          dE -= 0.5*np.einsum('prqs,ps,qr',obj.h2,rho_up,rho_up)
          dE -= 0.5*np.einsum('prqs,ps,qr',obj.h2,rho_dn,rho_dn)
          dV  =     np.einsum('prqs,pr->qs',obj.h2,rho_up+rho_dn)
          dV -= 0.5*np.einsum('prqs,ps->qr',obj.h2,rho_up)
          dV -= 0.5*np.einsum('prqs,ps->qr',obj.h2,rho_dn)
          obj.h0 += dE
          obj.h1 += dV
          return obj.downfold(to_freeze)

