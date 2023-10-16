from abc import ABC,abstractmethod
from time import time
import numpy as np

class NEVPT2_solver(ABC):

      # classe base NEVPT2_solver con metodi astratti
      # - compute_ground_state (restituisce energia e funzione d'onda di GS)
      # - compute_excited_states (restituisce energie e funzioni d'onda di stati eccitati)
      # - compute_omega_coefficients (restituisce i coefficienti omega)

      def __init__(self,BO_IAO,BO_iao_external):
          self.prepared = False                    # for VQE only
          self.BO_IAO = BO_IAO
          self.BO_iao_external = BO_iao_external

      @abstractmethod
      def compute_ground_state(self):
          # lo stato fondamentale va restituito come dizionario a due entrate GS['energy'], GS['wavefunction']
          pass

      @abstractmethod
      def compute_excited_states(self,ground_state):
          # gli stati fondamentali vanno restituiti come dizionari EXCITED['u'], EXCITED['d'], EXCITED['uu'], EXCITED['dd'], EXCITED['ud']
          # ciascuno di questi dizionari e' a sua volta un dizionario EXCITED['u']['energies'] e EXCITED['u']['wavefunctions']
          pass

      @abstractmethod
      def compute_omega_coefficients(self,ground_state,excited_states):
          # i coefficienti omega vanno restituiti come dizionario omega['1'],omega['2'],omega['3']
          # ciascuno degli omega['k'] e' un dizionario
          # omega['1']['u']  e omega['1']['d']
          # omega['3']['u']  e omega['3']['d']
          # omega['2']['uu'] e omega['2']['ud'] omega['2']['dd']
          pass

      def compute_nevpt2_energy(self):
          ground_state   = self.compute_ground_state()
        
          start = time()
            
          excited_states = self.compute_excited_states(ground_state)
          omega          = self.compute_omega_coefficients(ground_state,excited_states)

          n_val = self.BO_IAO.no
          n_ext = self.BO_iao_external.no-self.BO_IAO.no
          h1    = self.BO_iao_external.h1
          f1    = self.BO_iao_external.f1
          h2    = self.BO_iao_external.h2

          # contrazione dei coefficienti omega con gli elementi di matrice di H, equazioni (32),(34),(39),(42) e (47)
          # e calcolo del contributo NEVPT2 all'energia di stato fondamentale, equazione (49)
          # qui indici P e Q indicano orbitali esterni, f1[P,P] ed f1[Q,Q] indicano autovalori dell'operatore di Fock
          # Evec[k][mu] = E[k][mu]-EGS+f[P,P]+f[Q,Q]
          dE = 0.0
          E_GS = ground_state.energy

          for C in range(n_val,n_val+n_ext):
              fC = f1[C,C]
              for k in ['u','d']:
                  mkC_V_g = np.einsum('q,mq->m',h1[C,:n_val],omega.get_tensor(1,k)) \
                          + np.einsum('rqs,mqsr->m',h2[C,:n_val,:n_val,:n_val],omega.get_tensor(3,k)) # Eq. (34)
                  Evec    = np.array(excited_states.get_excited_state_en(k))+fC-E_GS
                  dE     -= sum(mkC_V_g**2/Evec)                                                      # Eq. (49), termini u/d
          print("u+d contributions ",dE)
          for C in range(n_val,n_val+n_ext):
              for D in range(C+1,n_val+n_ext):
                  fCD = f1[C,C]+f1[D,D]
                  for k in ['uu','dd']:
                      mkCD_V_g = np.einsum('rs,msr->m',h2[C,:n_val,D,:n_val],omega.get_tensor(2,k))   # Eq. (42)
                      Evec     = np.array(excited_states.get_excited_state_en(k))+fCD-E_GS
                      dE      -= sum(mkCD_V_g**2/Evec)                                                # Eq. (49), termini uu/dd
          print("uu+dd contributions ",dE)
          for C in range(n_val,n_val+n_ext):
              for D in range(n_val,n_val+n_ext):
                  fCD = f1[C,C]+f1[D,D]
                  for k in ['ud']:
                      mkCD_V_g = np.einsum('rs,msr->m',h2[C,:n_val,D,:n_val],omega.get_tensor(2,k))   # Eq. (47)
                      Evec     = np.array(excited_states.get_excited_state_en(k))+fCD-E_GS
                      dE      -= sum(mkCD_V_g**2/Evec)                                                # Eq. (49), termine ud
          print("ud contributions ",dE)
          
          end = time()  
          print('NEVPT2 time: %.2f s' % (end-start))
      
          return E_GS+dE
 
