import numpy as np
import pyscf
import itertools

def get_integrals(mol,mf,C,n):
    from pyscf import ao2mo
    h0  = mol.energy_nuc()
    h1  = np.einsum('ab,ai,bk->ik',mf.get_hcore(),C,C)    
    h2  = ao2mo.restore(1,ao2mo.kernel(mol,C),C.shape[1])
    no  = h1.shape[0]
    ne  = ((mol.nelectron+mol.spin)//2,(mol.nelectron-mol.spin)//2)
    irrep_names = {y:x for x,y in zip(mol.irrep_name,mol.irrep_id)}
    mo_irreps   = [irrep_names[x] for x in pyscf.symm.label_orb_symm(mol,mol.irrep_id,mol.symm_orb,C)]
    return {'no':no,'ne':ne,'h0':h0,'h1':h1,'h2':h2,'irr':mo_irreps}

# --------------------------------------------------------------------------------------

def downfold(mol_dict,to_remove,virtual=False):
    to_keep = [i for i in range(mol_dict['no']) if i not in to_remove]
    mol_dict['h1'] = mol_dict['h1'][np.ix_(to_keep,to_keep)]
    mol_dict['h2'] = mol_dict['h2'][np.ix_(to_keep,to_keep,to_keep,to_keep)]
    mol_dict['no'] = mol_dict['no'] - len(to_remove)
    if(not virtual):
       mol_dict['ne'] = (mol_dict['ne'][0]-len(to_remove),mol_dict['ne'][1]-len(to_remove))
    mol_dict['irr']   = [mol_dict['irr'][x] for x in to_keep]

def freeze_orbitals(mol_data,to_freeze):
    import numpy as np
    n  = mol_data['no']
    na = mol_data['ne'][0]
    nb = mol_data['ne'][1]
    h0 = mol_data['h0']
    h1 = mol_data['h1']
    h2 = mol_data['h2']
    rho_up = np.zeros((n,n))
    rho_dn = np.zeros((n,n))
    for i in to_freeze: rho_up[i,i]=1.0
    for i in to_freeze: rho_dn[i,i]=1.0
    Enuc  = np.einsum('ij,ji',h1,rho_up+rho_dn)
    Enuc += 0.5*np.einsum('prqs,pr,qs',h2,rho_up+rho_dn,rho_up+rho_dn)
    Enuc -= 0.5*np.einsum('prqs,ps,qr',h2,rho_up,rho_up)
    Enuc -= 0.5*np.einsum('prqs,ps,qr',h2,rho_dn,rho_dn)
    V1    = np.einsum('prqs,pr->qs',h2,rho_up+rho_dn)
    V1   -= 0.5*np.einsum('prqs,ps->qr',h2,rho_up)
    V1   -= 0.5*np.einsum('prqs,ps->qr',h2,rho_dn)
    h1   += V1
    mol_data['h0'] = h0 + Enuc
    mol_data['h1'] = h1
    mol_data['h2'] = h2
    return mol_data

def build_active_space(mol_info,mol,mf,to_keep):
    n  = mol_info['no']
    na = mol_info['ne'][0]
    nb = mol_info['ne'][1]
    not_to_keep = [x for x in range(n) if x not in to_keep]
    to_freeze   = [x for x in not_to_keep if x<nb]
    to_discard  = [x for x in not_to_keep if x>=na]
    freeze_orbitals(mol_info,to_freeze)
    downfold(mol_info,to_discard,virtual=True)
    downfold(mol_info,to_freeze,virtual=False)
    return mol_info

# --------------------------------------------------------------------------------------

def do_scf(mol_data,rho_0=None):
    from pyscf import gto,scf,ao2mo
    n     = mol_data['no']
    na,nb = mol_data['ne'][0],mol_data['ne'][1]
    mol               = gto.M(verbose=2)
    mol.nelectron     = na+nb
    mol.spin          = na-nb
    mol.incore_anyway = True
    mol.nao_nr        = lambda *args : n
    mol.energy_nuc    = lambda *args : mol_data['h0']
    if(na==nb):
       mf            = scf.RHF(mol)
       mf.get_hcore  = lambda *args: mol_data['h1']
       mf.get_ovlp   = lambda *args: np.eye(n)
       mf._eri       = ao2mo.restore(1,mol_data['h2'],n)
       if(rho_0 is None):
          rho_0 = np.zeros((n,n))
          for i in range(na): rho_0[i,i] = 2.0
    else:
       mf            = scf.ROHF(mol)
       mf.get_hcore  = lambda *args: mol_data['h1']
       mf.get_ovlp   = lambda *args: np.eye(n)
       mf._eri       = ao2mo.restore(1,mol_data['h2'],n)
       if(rho_0 is None):
          rho_0 = np.zeros((2,n,n))
          for i in range(na): rho_0[0,i,i] = 1.0
          for i in range(nb): rho_0[1,i,i] = 1.0
    E0 = mf.kernel(rho_0)
    if(not mf.converged):
       mf = scf.newton(mf)
       E0 = mf.kernel(mf.make_rdm1())
    return E0

# --------------------------------------------------------------------------------------

def print_dictionary(mol_info,verbose=False):
    print("-"*53)
    print("number of orbitals  ",mol_info['no'])
    print("number of electrons ",mol_info['ne']) 
    print("energy offset       ",mol_info['h0'])
    print("irreps of orbitals  ",mol_info['irr'])
    if(verbose):
       n = mol_info['no']
       for p,r in itertools.product(range(n),repeat=2):
           print("h1(%d,%d) = %.8f " % (p,r,mol_info['h1'][p,r]))
       for p,r,q,s in itertools.product(range(n),repeat=4):
           print("h1(%d,%d,%d,%d) = %.8f " % (p,r,q,s,mol_info['h2'][p,r,q,s]))
    print("-"*53)

def dump_on_file(mol_info,fname='dictionary_output'):
    np.save(fname,mol_info,allow_pickle=True)

def read_from_file(fname='dictionary_output'):
    return np.load(fname,allow_pickle=True).item()

