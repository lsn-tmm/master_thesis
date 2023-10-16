from abc import ABC,abstractmethod
from nevpt2_solver import NEVPT2_solver
from bo_solver import BO_solver
import numpy as np
from scipy import linalg as LA
from subroutines import h_op
from FCI_types import my_wavefunction,my_omega
from QSE_types import my_data,excited_states
#from pyscf.lib.linalg_helper import safe_eigh
from qiskit_subroutines import prepare_operators,run_vqe,measure_operators,measure_operators_from_bloch_vector

def safe_eigh(h,s,lindep):
    from functools import reduce
    seig, t = LA.eigh(s)
    idx = seig >= lindep
    t = t[:,idx] * (1/np.sqrt(seig[idx]))
    if t.size > 0:
        heff = reduce(np.dot, (t.T.conj(), h, t))
        w, v = LA.eigh(heff)
        v = np.dot(t, v)
    else:
        w = np.zeros((0,))
        v = t
    return w,v,seig

class VQE_QSE_solver(NEVPT2_solver):

      def set_quantum_variables(self,quantum_variables,with_bloch_vector=False,bloch_vector=None,lindep=1e-15):
          self.quantum_variables = quantum_variables
          self.with_bloch_vector = with_bloch_vector
          self.bloch_vector = bloch_vector
          self.lindep = lindep 

      def preparation_of_operators(self,read_from_file=False,write_to_file=False,filename=None):
          self.operators,self.quantum_variables = prepare_operators(self.BO_IAO,self.quantum_variables,read_from_file,write_to_file,filename)
          self.prepared = True

      def compute_ground_state(self):
          SOLVER = BO_solver(self.BO_IAO)
          if(not self.prepared):
             self.operators,self.quantum_variables = prepare_operators(self.BO_IAO,self.quantum_variables)
             self.prepared = True
          Psi_g = run_vqe(self.BO_IAO,self.operators,self.quantum_variables)
          return Psi_g

      def compute_excited_states(self,ground_state):
          from pyscf import fci
          n     = self.BO_IAO.no
          na,nb = self.BO_IAO.ne
          a_orb    = [r for r in range(n)]
          aa_pairs = [(s,r) for s in range(n) for r in range(n) if s<r]
          ab_pairs = [(s,r) for s in range(n) for r in range(n)]
          Psi0  = ground_state.circuit
          Psi_e =  excited_states()
          for k in ['u','d','uu','dd','ud']:
              Sk = self.operators['qse_s_'+k]
              Hk = self.operators['qse_h_'+k]
              if(self.with_bloch_vector):
                Sk = measure_operators_from_bloch_vector(Sk,Psi0,self.bloch_vector)
                Hk = measure_operators_from_bloch_vector(Hk,Psi0,self.bloch_vector)
              else:
                Sk = measure_operators(Sk,Psi0,ground_state.instance)
                Hk = measure_operators(Hk,Psi0,ground_state.instance)
              if(k=='u' or k=='d'):
                 E_operators = a_orb
                 C = np.zeros((n,len(E_operators)))
              if(k=='uu' or k=='dd'):
                 E_operators = aa_pairs
                 C = np.zeros((n,n,len(E_operators)))
              if(k=='ud'):
                 E_operators = ab_pairs
                 C = np.zeros((n,n,len(E_operators)))
              n_op = len(E_operators)
              S = np.zeros((n_op,n_op))
              H = np.zeros((n_op,n_op))
              for m,(a,b) in enumerate([(a,b) for a in range(n_op) for b in range(n_op) if a<=b]):
                  S[a,b] = Sk[m][0]
                  H[a,b] = Hk[m][0]
                  S[b,a] = Sk[m][0]
                  H[b,a] = Hk[m][0]
              sgm,V = LA.eigh(S)
              e,U = safe_eigh(H,S,self.lindep)[:2]
              print("lindep ",self.lindep)
              print(" >>> overlap matrix eigenvalues "+" ".join(["%.7f" % x for x in sgm])+" \n")
              print(" saved energies "+" ".join(["%.7f" % x for x in e])+" \n")
              #print('sector: ', k)
              #print('H shape: ', H.shape)
              #print('H: ', H)
              #print('S shape: ', S.shape)
              #print('S: ', S)
              for i_psi in range(len(e)):
                  if(k=='u' or k=='d'):
                     for j,r in enumerate(a_orb):
                         C[r,i_psi] = U[j,i_psi]
                     C = C[:,:len(e)]
                  if(k=='uu' or k=='dd'):
                     for j,(s,r) in enumerate(aa_pairs):
                      C[s,r,i_psi] = U[j,i_psi]
                     C = C[:,:,:len(e)]
                  if(k=='ud'):
                     for j,(s,r) in enumerate(ab_pairs):
                         C[s,r,i_psi] = U[j,i_psi]
                     C = C[:,:,:len(e)]
              Psi_e.set_excited_states(k,S,C,e)
          # --------------------------------------------------------------------------------
          return Psi_e

      def compute_omega_coefficients(self,ground_state,Psi_e):
          from pyscf import fci
          n     = self.BO_IAO.no
          na,nb = self.BO_IAO.ne
          a_orb    = [r for r in range(n)]
          aa_pairs = [(s,r) for s in range(n) for r in range(n) if s<r]
          ab_pairs = [(s,r) for s in range(n) for r in range(n)]
          omega = my_omega()
          dm1,dm2 = {},{}
          dm1['u'] = Psi_e.u.overlap
          dm1['d'] = Psi_e.d.overlap
          dm2['uu'] = np.zeros((n,n,n,n))
          dm2['dd'] = np.zeros((n,n,n,n))
          dm2['ud'] = np.zeros((n,n,n,n))
          for a,(q,p) in enumerate(ab_pairs):
              for b,(s,r) in enumerate(ab_pairs):
                  dm2['ud'][q,s,p,r] = Psi_e.ud.overlap[a,b]
          for a,(q,p) in enumerate(aa_pairs):
              for b,(s,r) in enumerate(aa_pairs):
                  dm2['uu'][p,r,q,s] = Psi_e.uu.overlap[a,b]
                  dm2['uu'][q,r,p,s] = -dm2['uu'][p,r,q,s]
                  dm2['uu'][p,s,q,r] = -dm2['uu'][p,r,q,s]
                  dm2['uu'][q,s,p,r] =  dm2['uu'][p,r,q,s]
                  dm2['dd'][p,r,q,s] = Psi_e.dd.overlap[a,b]
                  dm2['dd'][q,r,p,s] = -dm2['dd'][p,r,q,s]
                  dm2['dd'][p,s,q,r] = -dm2['dd'][p,r,q,s]
                  dm2['dd'][q,s,p,r] =  dm2['dd'][p,r,q,s]
          # ------------------------------------------------------------
          for k in ['u','d']:
              omega.set_tensor(1,k,np.einsum('pm,pr->mr',Psi_e.get_excited_states_c(k),dm1[k]))
              omega.set_tensor(3,k,np.einsum('pm,prqs->mqsr',Psi_e.get_excited_states_c(k),dm2[k+k]))
              if(k=='u'): omega.set_tensor(3,k, omega.get_tensor(3,k)+np.einsum('pm,prqs->mqsr',Psi_e.get_excited_states_c(k),dm2['ud']))
              if(k=='d'): omega.set_tensor(3,k, omega.get_tensor(3,k)+np.einsum('qm,prqs->mprs',Psi_e.get_excited_states_c(k),dm2['ud']))
          for k in ['uu','ud','dd']:
              omega.set_tensor(2,k,np.einsum('pqm,prqs->msr',Psi_e.get_excited_states_c(k),dm2[k]))
          return omega
