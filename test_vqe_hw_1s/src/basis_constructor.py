import numpy as np
from bo_class import BO_class
from bo_solver import BO_solver
from scipy import linalg as LA
from subroutines import project_outside

class Basis_Constructor:

      # divisione degli orbitali in core, attivi, ed esterni
      def __init__(self,mol,mf):
          self.mol = mol
          self.mf  = mf

      def compute_core_valence(self):
          from pyscf import lo
          # costruzione degli orbitali IAO (vedi Knizia, 2013)
          mo_occ = self.mf.mo_coeff[:,self.mf.mo_occ>0]
          C_val  = lo.iao.iao(self.mol,mo_occ)
          C_val  = lo.vec_lowdin(C_val,self.mf.get_ovlp())
          BO_IAO = BO_class(self.mol,self.mf)
          BO_IAO.transform_integrals(C_val)
          # esegue un conto RHF/ROHF nella base di orbitali IAO, producendo orbitali di valenza occupati oppure virtuali
          SOLVER = BO_solver(BO_IAO)
          mf,_   = SOLVER.solve_with_scf()
          C_val  = np.dot(C_val,mf.mo_coeff)
          self.valence = C_val

      def compute_external(self):
          # costruzione del complemento ortogonale alla base di core+valenza
          C_ext  = project_outside(self.mf.mo_coeff,self.valence,self.mf.get_ovlp())
          BO_EXT = BO_class(self.mol,self.mf)
          BO_EXT.transform_integrals(C_ext)
          # costruzione dell'operatore di Fock degli orbitali esterni F(ext)
          # diagonalizzazione di F(ext), e definizione degli orbitali esterni come autovettori di F(ext)
          f_ext  = BO_EXT.f1
          e,w    = LA.eigh(f_ext)
          self.external = np.dot(C_ext,w)

      def return_basis(self):
          nv = self.valence.shape[1]
          n  = nv+self.external.shape[1]
          C  = np.zeros((n,n))
          C[:,:nv] = self.valence
          C[:,nv:] = self.external
          return C
