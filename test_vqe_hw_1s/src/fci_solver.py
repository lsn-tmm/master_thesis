from abc import ABC,abstractmethod
from nevpt2_solver import NEVPT2_solver
from bo_solver import BO_solver
import numpy as np
from FCI_types import my_wavefunction,excited_states,my_omega

class FCI_solver(NEVPT2_solver):

      # FCI_solver eredita dalla classe base NEVPT2_solver e calcola coefficienti omega 
      # ed energie di stato fondamentale/eccitati secondo lo schema "tutto esatto"

      def compute_ground_state(self):
          SOLVER = BO_solver(self.BO_IAO)
          result = SOLVER.solve_with_fci(nroots=1)
          Psi_g  = my_wavefunction(result[0],result[1])
          return Psi_g

      def compute_excited_states(self,ground_state):
          Psi_e = excited_states()
          for (k,dna,dnb) in zip(['u','d','uu','ud','dd'],
                                 [1,0,2,1,0],
                                 [0,1,0,1,2]):
              BO_k = self.BO_IAO.copy()
              BO_k.ne = (BO_k.ne[0]-dna,BO_k.ne[1]-dnb)
              SOLVER = BO_solver(BO_k)
              try:
                  result = SOLVER.solve_with_fci(nroots=1000000)
                  Psi_e.set_excited_states(k,result[0],result[1])
              except:
                 Psi_e.set_excited_states(k,[BO_k.h0],[1.0])
          return Psi_e

      def compute_omega_coefficients(self,Psi_g,Psi_e):
          from pyscf import fci
          n     = self.BO_IAO.no
          na,nb = self.BO_IAO.ne
          Psi0  = Psi_g.wavefunction
          omega = my_omega()
          # ------------------------------------------------------------
          for k in ['u','d']:
              n_psi = Psi_e.get_num_psi(k)
              omega.init_tensor(1,k,n_psi,n)
              for p in range(n):
                  if(k=='u'): cp_gs = fci.addons.des_a(Psi0,n,(na,nb),p)
                  if(k=='d'): cp_gs = fci.addons.des_b(Psi0,n,(na,nb),p)
                  for m in range(n_psi):
                      Psim = Psi_e.get_excited_state_wf(k,m)
                      omega.set_element(1,k,(m,p),np.dot(cp_gs.flatten(),Psim.flatten()))
          # ------------------------------------------------------------
          for k in ['uu','ud','dd']:
              n_psi = Psi_e.get_num_psi(k)
              omega.init_tensor(2,k,n_psi,n)
              for r in range(n):
                  if(k=='uu' or k=='ud'): cr_gs = fci.addons.des_a(Psi0,n,(na,nb),r)
                  else:                   cr_gs = fci.addons.des_b(Psi0,n,(na,nb),r)
                  for s in range(n):
                      if(k=='uu'): csr_gs = fci.addons.des_a(cr_gs,n,(na-1,nb),s)
                      if(k=='dd'): csr_gs = fci.addons.des_b(cr_gs,n,(na,nb-1),s)
                      if(k=='ud'): csr_gs = fci.addons.des_b(cr_gs,n,(na-1,nb),s)
                      for m in range(n_psi):
                        Psim = Psi_e.get_excited_state_wf(k,m)
                        try:
                          omega.set_element(2,k,(m,s,r),np.dot(csr_gs.flatten(),Psim.flatten()))
                        except:
                          omega.set_element(2,k,(m,s,r),0)
          # ------------------------------------------------------------
          for k in ['u','d']:
              n_psi = Psi_e.get_num_psi(k)
              omega.init_tensor(3,k,n_psi,n)
              for r in range(n):
                  if(k=='u'): cr_gs = fci.addons.des_a(Psi0,n,(na,nb),r)
                  else:       cr_gs = fci.addons.des_b(Psi0,n,(na,nb),r)
                  for s in range(n):
                      for q in range(n):
                          cqsr_gs = np.zeros(cr_gs.shape)
                          if(k=='u'):
                             try: cqsr_gs += fci.addons.cre_a(fci.addons.des_a(cr_gs,n,(na-1,nb),s),n,(na-2,nb),q)
                             except: pass
                             try: cqsr_gs += fci.addons.cre_b(fci.addons.des_b(cr_gs,n,(na-1,nb),s),n,(na-1,nb-1),q)
                             except: pass
                          else:
                             try: cqsr_gs += fci.addons.cre_a(fci.addons.des_a(cr_gs,n,(na,nb-1),s),n,(na-1,nb-1),q)
                             except: pass 
                             try: cqsr_gs += fci.addons.cre_b(fci.addons.des_b(cr_gs,n,(na,nb-1),s),n,(na,nb-2),q)
                             except: pass
                          for m in range(n_psi):
                            Psim = Psi_e.get_excited_state_wf(k,m)
                            omega.set_element(3,k,(m,q,s,r), np.dot(cqsr_gs.flatten(),Psim.flatten()))
          # ------------------------------------------------------------
          return omega

