import numpy as np
import pyscf
import itertools

# tutte queste subroutines potrebbero/dovrebbero diventare una classe
# questa classe potrebbe chiamarsi Born_Oppenheimer o qualcosa del genere, e dovrebbe avere
# 1. dei data members
# ............ (guardiamo dentro mol_data: abbiamo tutto? o servira' dell'altro?)
# ............ 
# ............ 
# ............ 
# ............ 
# ............ 
# ............ 
# 2. dei metodi
# __init__
# transform_integrals
#      passaggio dalla base A alla base B
#      <Bi|h|Bj> = \sum_{kl} V(ki) V(lj) <Ak|h|Al>
#      tensori di rango 4
# do_scf esegui un conto Hartee-Fock nella base desiderata
# do_electronic_structure esegui tutti i conti immaginabili (HF.....CASCI) nella base desiderata
# freeze_orbitals
# remove_orbitals
# ............ 

def get_integrals(mol,mf,C,n):
    from pyscf import ao2mo
    from scipy import linalg as LA
    # l'Hamiltoniano di Born-Oppenheimer ha la forma seguente
    # H = E0 
    #   + \sum_{pq \sigma} h(pq) a*(p\sigma) a(q\sigma) 
    #   + \sum_{prqs \sigma\tau} (pr|qs)/2 a*(p\sigma) a*(q\tau) a(s\tau) a(r\sigma)
    # (pr|qs) = \int dxdy f(p,x) f(r,x) 1/|x-y| f(q,y) f(s,y)
    # p\sigma \     / q\tau
    #          ----- 
    # r\sigma /     \ s\tau
    # ------------------------------------------------------------------------------
    h0 = mol.energy_nuc()                                # E0
    s1 = np.einsum('ab,ai,bk->ik',mf.get_ovlp(),C,C)     # overlap (ci|ck) = \sum_{ab} C(ai) S(ab) C(bk)
    h1 = np.einsum('ab,ai,bk->ik',mf.get_hcore(),C,C)    # h1      h(ik)   = \sum_{ab} C(ai) h(ab) C(bk)
    h2 = ao2mo.restore(1,ao2mo.kernel(mol,C),C.shape[1]) # h2      (pr|qs) = \sum_{ab} C(pi) C(rj) C(qk) C(sl) (ij|kl)
    no = h1.shape[0]
    ne = ((mol.nelectron+mol.spin)//2,(mol.nelectron-mol.spin)//2)
    na = ne[0]
    nb = ne[1]
    mol_data = {'no':no, # numero di orbitali 
                'ne':ne, # numero di elettroni (na,nb) a spin-up e spin-down
                'h0':h0, # costante (energy offset)
                's1':s1, # matrice di overlap       nella base di orbitali selezionati
                'h1':h1, # hamiltoniano a un corpo  nella base di orbitali selezionati
                'h2':h2} # hamiltoniano a due corpi nella base di orbitali selezionati
    # ------------------------------------------------------------------------------
    # quando si fa un conto HF si ottengono degli orbitali occupati |Mk> = \sum_{a} X_{ak} |a>
    # alcuni orbitali sono occupati da particelle sia a spin up sia a spin down (i primi nb orbitali)
    # altri orbitali sono occupati da particelle solo a spin up (orbitali da nb ad na)
    # d'altra parte noi abbiamo una base |Ci> = \sum_{a} C_{ai} |a>
    # |Mk> = \sum_i Y_{ik} |Ci>
    # come si fa? anzitutto si proietta sulla base |Cj>
    # <Cj|Mk> = \sum_i Y_{ik} <Cj|Ci>
    # poi si calcolano i prodotti interni
    # <Cj|Mk> = \sum_{ab} C_{aj} X_{bk} <a|b> = T(C)*S*X
    # <Cj|Ci> = \sum_{ab} C_{ai} C_{bj} <a|b> = T(C)*S*C = S_bar
    # poi si riscrive la proiezione come prodotto di matrici
    # [T(C)*S*X] = S_bar*Y
    # Y = (S_bar)^{-1} [T(C)*S*X]
    S         = mf.get_ovlp()
    S_bar     = np.dot(C.T,np.dot(S,C)) # S_bar
    S_bar_inv = LA.inv(S_bar)           # inversa di S_bar
    X        = mf.mo_coeff[:,:na]
    X        = np.dot(S,X)
    X        = np.dot(C.T,X)
    #S        = mf.get_ovlp()
    #S_cross  = np.dot(C.T,S)
    #S_bar    = np.dot(C.T,np.dot(S,C))                   # S_bar
    #S_bar    = LA.inv(S_bar)
    Y        = np.dot(S_bar_inv,X)
    if(mol.spin==0):                               # shell chiusa, na=nb
       rho_0 = 2*np.dot(Y[:,:nb],Y[:,:nb].T)       # \rho(pr) = 2*\sum_{i=1}^{na=nb} Y(pi)Y(ri)
    else:                                          # shell aperta
       rho_0 = np.zeros((2,Y.shape[0],Y.shape[0])) # \rho(up,pr)   = \sum_{i=1}^{na} Y(pi)Y(ri)
       rho_0[0,:,:] = np.dot(Y[:,:na],Y[:,:na].T)  # \rho(down,pr) = \sum_{i=1}^{nb} Y(pi)Y(ri)
       rho_0[1,:,:] = np.dot(Y[:,:nb],Y[:,:nb].T)
    #print(rho_0.shape,np.einsum('pr,pr',rho_0,s1))
    #exit()
    E,V = do_scf(mol_data,rho_0=rho_0)[:2]         # <=== eseguiamo un conto RHF nella base IAO
    # ------------------------------------------------------------
    # passaggio alla base degli orbitali di Hartree-Fock (occupati e virtuali) espressi nella base Ck
    C          = np.dot(C,V)
    #print(mol.irrep_name)
    C[:,:nb]   = pyscf.symm.symmetrize_orb(mol,C[:,:nb])
    if(na>nb): C[:,nb:na] = pyscf.symm.symmetrize_orb(mol,C[:,nb:na])
    C[:,na:]   = pyscf.symm.symmetrize_orb(mol,C[:,na:])
    # ------------------------------------------------------------   <==== QUESTA E' UNA RIPETIZIONE DELLA PARTE INIZIALE
    h0          = mol.energy_nuc()
    s1          = np.einsum('ab,ai,bk->ik',mf.get_ovlp(),C,C)
    h1          = np.einsum('ab,ai,bk->ik',mf.get_hcore(),C,C)
    h2          = ao2mo.restore(1,ao2mo.kernel(mol,C),C.shape[1])
    no          = h1.shape[0]
    ne          = ((mol.nelectron+mol.spin)//2,(mol.nelectron-mol.spin)//2)
    mol_data    = {'no':no,'ne':ne,'h0':h0,'s1':s1,'h1':h1,'h2':h2}
    E,V         = do_scf(mol_data)[:2]
    irrep_names = {y:x for x,y in zip(mol.irrep_name,mol.irrep_id)}
    mo_irreps   = [irrep_names[x] for x in pyscf.symm.label_orb_symm(mol,mol.irrep_id,mol.symm_orb,C)]
    return {'no':no,'ne':ne,'h0':h0,'s1':s1,'h1':h1,'h2':h2,'irr':mo_irreps,'C':C}

# --------------------------------------------------------------------------------------

def downfold(mol_dict,to_remove,virtual=False):
    to_keep = [i for i in range(mol_dict['no']) if i not in to_remove]
    mol_dict['s1'] = mol_dict['s1'][np.ix_(to_keep,to_keep)]
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
    mol               = gto.M(verbose=4)
    mol.nelectron     = na+nb
    mol.spin          = na-nb
    mol.incore_anyway = True
    mol.nao_nr        = lambda *args : n
    mol.energy_nuc    = lambda *args : mol_data['h0']
    if(na==nb):
       mf            = scf.RHF(mol)
       mf.get_hcore  = lambda *args: mol_data['h1']
       mf.get_ovlp   = lambda *args: mol_data['s1']
       mf._eri       = ao2mo.restore(1,mol_data['h2'],n)
       if(rho_0 is None):
          rho_0 = np.zeros((n,n))
          for i in range(na): rho_0[i,i] = 2.0
    else:
       mf            = scf.ROHF(mol)
       mf.get_hcore  = lambda *args: mol_data['h1']
       mf.get_ovlp   = lambda *args: mol_data['s1']
       mf._eri       = ao2mo.restore(1,mol_data['h2'],n)
       if(rho_0 is None):
          rho_0 = np.zeros((2,n,n))
          for i in range(na): rho_0[0,i,i] = 1.0
          for i in range(nb): rho_0[1,i,i] = 1.0
    mf = scf.newton(mf)
    E0 = mf.kernel(rho_0)
    if(not mf.converged):
       mf = scf.newton(mf)
       E0 = mf.kernel(mf.make_rdm1())
    #a  = mf.stability()[0]
    #E0 = mf.kernel(a,mf.mo_occ)
    #exit()
    return E0,mf.mo_coeff,mol,mf

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

# --------------------------------------------------------------------------------------

def do_electronic_structure(mol_data):
    import numpy as np
    from pyscf import gto,scf,lo,tools  
    from pyscf import mp,ci,cc,mcscf
    EHF,C,mol,mf = do_scf(mol_data)
    nf     = 0                        
    mymp   = mp.MP2(mf,frozen=nf)     
    EMP    = EHF + mymp.kernel()[0]
    mcisd  = ci.CISD(mf,frozen=nf)
    ECISD  = EHF   + mcisd.kernel()[0]
    mc     = cc.CCSD(mf,frozen=nf)
    ECCSD  = EHF   + mc.kernel()[0]
    ECCSDT = ECCSD + mc.ccsd_t()
    na,nb       = (mol.nelectron+mol.spin)//2-nf,(mol.nelectron-mol.spin)//2-nf
    nact        = mol.nao_nr()-nf                                              
    mc          = mcscf.CASCI(mf,nact,(na,nb))
    mc.frozen   = nf
    ECASCI      = mc.kernel()[0]
    print("Hartree-Fock   ",EHF)
    print("Moller-Plesset ",EMP)
    print("CISD           ",ECISD)
    print("CCSD           ",ECCSD)
    print("CCSD(T)        ",ECCSDT)
    print("CASCI          ",ECASCI)

