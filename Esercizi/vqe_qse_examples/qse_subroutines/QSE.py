def safe_eigh(h,s,lindep=1e-15):
    from scipy import linalg as LA
    import numpy as np
    from functools import reduce
    seig,t = LA.eigh(s)
    if seig[0]<lindep:
        idx  = seig >= lindep
        t    = t[:,idx]*(1/np.sqrt(seig[idx]))
        heff = reduce(np.dot,(t.T.conj(),h,t))
        w,v  = LA.eigh(heff)
        v    = np.dot(t,v)
    else:
        w,v  = LA.eigh(h, s)
    return w,v

def adjoint(WPO):
    import numpy as np
    ADJ = WPO.copy()
    ADJ._paulis = [[np.conj(weight),pauli] for weight,pauli in WPO._paulis]
    return ADJ

class QSE:

      def __init__(self,psi=None,operators=None,mol_info=None,instance_dict={}):
          from qiskit                            import Aer
          from qiskit.aqua                       import QuantumInstance
          self.psi         = psi
          self.operators   = operators
          self.mol_info    = mol_info
          self.matrices    = {'S':None,'H':None}
          self.result      = {}
          if(mol_info['operators']['tapering']):
             self.hamiltonian   = mol_info['operators']['untapered_h_op']
             self.tapering      = mol_info['operators']['tapering']
             self.z2syms        = mol_info['operators']['tapering_info'][0]
             self.target_sector = mol_info['operators']['target_sector']
          else:
             self.hamiltonian   = mol_info['operators']['h_op']
             self.tapering      = mol_info['operators']['tapering']
          backend          = Aer.get_backend(instance_dict['instance'])
          quantum_instance = QuantumInstance(backend=backend,shots=instance_dict['shots'])
          self.instance    = quantum_instance

      def construct_qse_matrices(self,eps=1e-6):
          import numpy as np
          import sys
          sys.path.append('../commons/')
          from tapering import taper_auxiliary_operators
          from harvest import measure_operators
          import time

          n = len(self.operators)
          H = np.zeros((n,n,2))
          S = np.zeros((n,n,2))
          for i,Pi in enumerate(self.operators):
              for j,Pj in enumerate(self.operators):
                  if(i<=j):
                     t0 = time.time()
                     Sij_oper = adjoint(Pi)*Pj                    # S(I,J)
                     Hij_oper = adjoint(Pi)*(self.hamiltonian*Pj) # H(I,J)
                     Sij_oper = Sij_oper.chop(eps)
                     Hij_oper = Hij_oper.chop(eps)
                     t1 = time.time()
                     if(self.tapering):
                        Sij_oper,Hij_oper = taper_auxiliary_operators([Sij_oper,Hij_oper],self.z2syms,self.target_sector)
                     t2 = time.time()
                     S[i,j,:] = measure_operators([Sij_oper],self.psi,self.instance)[0]
                     H[i,j,:] = measure_operators([Hij_oper],self.psi,self.instance)[0]
                     t3 = time.time()
                     S[j,i,:] = S[i,j,:]
                     H[j,i,:] = H[i,j,:]
                     print("matrix elements (%d/%d,%d/%d) computed " % (i+1,n,j+1,n,))
                     print("times op,taper,measure %.6f %.6f %.6f  " % (t1-t0,t2-t1,t3-t2))
          self.matrices['S'] = S
          self.matrices['H'] = H 

      def run(self,lindep=1e-15,outfile=None,resfile='qse_result'):
          import numpy as np
          result = {}
          self.construct_qse_matrices()
          result['overlap']      = self.matrices['S'][:,:,0]
          result['hamiltonian']  = self.matrices['H'][:,:,0]
          eta,V                  = safe_eigh(result['hamiltonian'],result['overlap'],lindep=lindep)
          result['eigenvalues']  = eta
          result['eigenvectors'] = V
          self.result            = result
          for i,ei in enumerate(result['eigenvalues']):
              outfile.write('eigenvalue %d = %.8f \n' % (i,ei))
              for j,vji in enumerate(result['eigenvectors'][:,i]):
                  outfile.write('eigenvector %d, component %d = %.8f \n' % (i,j,vji))
              outfile.write('-'*53+'\n')
          np.save(resfile,result,allow_pickle=True)
          return result




