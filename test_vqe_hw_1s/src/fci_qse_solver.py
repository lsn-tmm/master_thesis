from abc import ABC,abstractmethod
from nevpt2_solver import NEVPT2_solver
from bo_solver import BO_solver
import numpy as np
from scipy import linalg as LA
from subroutines import h_op
from FCI_types import my_wavefunction,my_omega
from QSE_types import my_data,excited_states
from pyscf.lib.linalg_helper import safe_eigh

class FCI_QSE_solver(NEVPT2_solver):

    
      def compute_ground_state(self):
          SOLVER = BO_solver(self.BO_IAO)
          result = SOLVER.solve_with_fci(nroots=1)
          Psi_g  = my_wavefunction(result[0],result[1])
          return Psi_g        
        
      
      def compute_excited_states(self,ground_state):
          from pyscf import fci
          n     = self.BO_IAO.no
          na,nb = self.BO_IAO.ne
          Psi0  = ground_state.get_wavefunction()
          Psi_e = excited_states()
          aa_pairs = [(s,r) for s in range(n) for r in range(n) if s<r]
          ab_pairs = [(s,r) for s in range(n) for r in range(n)]
          des_a_gs = [fci.addons.des_a(Psi0,n,(na,nb),r) for r in range(n)]
          des_b_gs = [fci.addons.des_b(Psi0,n,(na,nb),r) for r in range(n)]
          des_aa_gs = [fci.addons.des_a(fci.addons.des_a(Psi0,n,(na,nb),r),n,(na-1,nb),s) for (s,r) in aa_pairs]
          des_bb_gs = [fci.addons.des_b(fci.addons.des_b(Psi0,n,(na,nb),r),n,(na,nb-1),s) for (s,r) in aa_pairs]
          des_ab_gs = [fci.addons.des_b(fci.addons.des_a(Psi0,n,(na,nb),r),n,(na-1,nb),s) for (r,s) in ab_pairs]
          for k in ['u','d']:
              S = np.zeros((n,n))
              H = np.zeros((n,n))
              C = np.zeros((n,n))
              for r in range(n):
                  if(k=='u'):
                     cr_gs = des_a_gs[r]
                     H_cr_gs = h_op(cr_gs,self.BO_IAO,(na-1,nb))
                  if(k=='d'):
                     cr_gs = des_b_gs[r]
                     H_cr_gs = h_op(cr_gs,self.BO_IAO,(na,nb-1))
                  for p in range(n):
                      if(k=='u'): cp_gs = des_a_gs[p]
                      if(k=='d'): cp_gs = des_b_gs[p]
                      S[p,r] = np.dot(cp_gs.flatten(),cr_gs.flatten())
                      H[p,r] = np.dot(cp_gs.flatten(),H_cr_gs.flatten())
              e,C = LA.eigh(H,S)      
              Psi_e.set_excited_states(k,S,C,e+self.BO_IAO.h0)
          # --------------------------------------------------------------------------------
          for k in ['uu','dd']:
              n_psi = len(aa_pairs)
              S = np.zeros((n_psi,n_psi))
              H = np.zeros((n_psi,n_psi))
              C = np.zeros((n,n,n_psi))
              for j,(s,r) in enumerate(aa_pairs):
                  if(k=='uu'):
                     csr_gs = des_aa_gs[j]
                     try:    H_csr_gs = h_op(csr_gs,self.BO_IAO,(na-2,nb))
                     except: H_csr_gs = np.zeros(csr_gs.shape)
                  if(k=='dd'): 
                     csr_gs = des_bb_gs[j]
                     try:    H_csr_gs = h_op(csr_gs,self.BO_IAO,(na,nb-2))
                     except: H_csr_gs = np.zeros(csr_gs.shape)
                  for m,(p,q) in enumerate(aa_pairs):
                      if(k=='uu'): cpq_gs = des_aa_gs[m]
                      if(k=='dd'): cpq_gs = des_bb_gs[m]
                      S[m,j] = np.dot(cpq_gs.flatten(),csr_gs.flatten())
                      H[m,j] = np.dot(cpq_gs.flatten(),H_csr_gs.flatten())
              e,U = safe_eigh(H,S)[:2]
              C = C[:,:,:len(e)]
              for i_psi in range(len(e)):
                  for j,(s,r) in enumerate(aa_pairs):
                      C[s,r,i_psi] = U[j,i_psi]
              Psi_e.set_excited_states(k,S,C,e+self.BO_IAO.h0)
          # --------------------------------------------------------------------------------
          for k in ['ud']:
              n_psi = len(ab_pairs)
              S = np.zeros((n_psi,n_psi))
              H = np.zeros((n_psi,n_psi))
              C = np.zeros((n,n,n_psi))
              for j,(s,r) in enumerate(ab_pairs):
                  csr_gs = des_ab_gs[j]
                  H_csr_gs = h_op(csr_gs,self.BO_IAO,(na-1,nb-1))
                  for m,(p,q) in enumerate(ab_pairs):
                      cpq_gs = des_ab_gs[m]
                      S[m,j] = np.dot(cpq_gs.flatten(),csr_gs.flatten())
                      H[m,j] = np.dot(cpq_gs.flatten(),H_csr_gs.flatten())
              sgm,V = LA.eigh(S)
              e,U = safe_eigh(H,S,lindep=1e-14)[:2]
              C = C[:,:,:len(e)]
              for i_psi in range(len(e)):
                  for j,(s,r) in enumerate(ab_pairs):
                      C[s,r,i_psi] = U[j,i_psi]                    
              Psi_e.set_excited_states(k,S,C,e+self.BO_IAO.h0)
          return Psi_e

      def compute_omega_coefficients(self,ground_state,Psi_e):
          from pyscf import fci
          n     = self.BO_IAO.no
          na,nb = self.BO_IAO.ne
          Psi0  = ground_state.get_wavefunction()
          omega = my_omega()
          dm1,dm2 = fci.direct_spin1.make_rdm12s(Psi0,n,(na,nb))
          dm1 = {'u':dm1[0],'d':dm1[1]}
          dm2 = {'uu':dm2[0],'ud':dm2[1],'dd':dm2[2]}
          # ------------------------------------------------------------
          for k in ['u','d']:
              omega.set_tensor(1,k,np.einsum('pm,pr->mr',Psi_e.get_excited_states_c(sector=k),dm1[k]))
              omega.set_tensor(3,k,np.einsum('pm,prqs->mqsr',Psi_e.get_excited_states_c(k),dm2[k+k]))
              if(k=='u'): omega.set_tensor(3,k, omega.get_tensor(3,k)+np.einsum('pm,prqs->mqsr',Psi_e.get_excited_states_c(k),dm2['ud']))
              if(k=='d'): omega.set_tensor(3,k, omega.get_tensor(3,k)+np.einsum('qm,prqs->mprs',Psi_e.get_excited_states_c(k),dm2['ud']))
          for k in ['uu','ud','dd']:
              omega.set_tensor(2,k,np.einsum('pqm,prqs->msr',Psi_e.get_excited_states_c(k),dm2[k]))
          
          return omega                                 
          
