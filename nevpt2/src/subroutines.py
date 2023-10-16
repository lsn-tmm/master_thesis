import numpy as np
from scipy import linalg as LA

#def ovlp(S,A,B=None):
#    if(B is None): B=A
#    M = np.dot(np.dot(A.T,S),B)
#    U,s,V = LA.svd(M,full_matrices=True)
#    return np.prod(s)

def project_outside(A,B,S):
    n = B.shape[1]
    M = np.dot(np.dot(A.T,S),B)
    U,s,V = LA.svd(M,full_matrices=True)
    return np.dot(A,U[:,n:])

def h_op(c,BO_obj,ne):
    from pyscf     import fci
    from pyscf.fci import fci_slow
    h2e = fci_slow.absorb_h1e(BO_obj.h1,BO_obj.h2,BO_obj.no,ne,0.5) 
    return fci_slow.contract_2e(h2e,c,BO_obj.no,ne)

def get_ao_integrals(mol,mf):
    from pyscf import ao2mo
    return [mol.energy_nuc(),mf.get_ovlp(),mf.get_fock(),mf.get_hcore(),ao2mo.restore(1,mf._eri,mol.nao_nr())]

'''
def get_info(mol,mf,C):
    n = C.shape[1]
    na,nb = (mol.nelectron+mol.spin)//2,(mol.nelectron-mol.spin)//2
    S = mf.get_ovlp()
    mo_occ = mf.mo_coeff[:,mf.mo_occ>0]
    mo_occ = np.einsum('pr,pq,qi->ri',C,S,mo_occ,optimize=True)
    if(na==nb):
       rho = 2*np.einsum('pi,qi->pq',mo_occ,mo_occ)
    else:
       rho = np.zeros((2,n,n))
       rho[0,:,:] = np.einsum('pi,qi->pq',mo_occ,mo_occ)
       rho[1,:,:] = np.einsum('pi,qi->pq',mo_occ[:,:nb],mo_occ[:,:nb])
    return [n,(mol.nelectron+mol.spin)//2,(mol.nelectron-mol.spin)//2,rho]

def transform_integrals(mol,mf,C):
    h0,s1,f1,h1,h2 = get_ao_integrals(mol,mf)
    s1 = np.einsum('pi,pq,qj->ij',C,s1,C,optimize=True)
    f1 = np.einsum('pi,pq,qj->ij',C,f1,C,optimize=True)
    h1 = np.einsum('pi,pq,qj->ij',C,h1,C,optimize=True)
    h2 = np.einsum('pi,rj,qk,sl,prqs->ijkl',C,C,C,C,h2,optimize=True)
    return [h0,s1,f1,h1,h2]

def freeze_orbitals(integrals,to_freeze):
    h0,s1,f1,h1,h2 = integrals
    n = s1.shape[0]
    rho_up = np.zeros((n,n))
    rho_dn = np.zeros((n,n))
    for i in to_freeze: rho_up[i,i]=1.0
    for i in to_freeze: rho_dn[i,i]=1.0
    dE  = np.einsum('ij,ji',h1,rho_up+rho_dn)
    dE += 0.5*np.einsum('prqs,pr,qs',h2,rho_up+rho_dn,rho_up+rho_dn)
    dE -= 0.5*np.einsum('prqs,ps,qr',h2,rho_up,rho_up)
    dE -= 0.5*np.einsum('prqs,ps,qr',h2,rho_dn,rho_dn)
    V1    =     np.einsum('prqs,pr->qs',h2,rho_up+rho_dn)
    V1   -= 0.5*np.einsum('prqs,ps->qr',h2,rho_up)
    V1   -= 0.5*np.einsum('prqs,ps->qr',h2,rho_dn)
    h1   += V1
    return downfold([h0+dE,s1,f1,h1,h2],n,to_freeze)

def downfold(integrals,n,to_remove):
    h0,s1,f1,h1,h2 = integrals
    to_keep = [i for i in range(n) if i not in to_remove]
    print(s1.shape,to_remove,to_keep)
    s1 = s1[np.ix_(to_keep,to_keep)]
    f1 = f1[np.ix_(to_keep,to_keep)]
    h1 = h1[np.ix_(to_keep,to_keep)]
    h2 = h2[np.ix_(to_keep,to_keep,to_keep,to_keep)]
    return [h0,s1,f1,h1,h2]

def do_scf(integrals,info):
    from pyscf import gto,scf,ao2mo
    h0,s1,f1,h1,h2 = integrals
    n,na,nb,rho = info
    mol = gto.M(verbose=4)
    mol.nelectron     = na+nb 
    mol.spin          = na-nb
    mol.incore_anyway = True
    mol.nao_nr        = lambda *args : n
    mol.energy_nuc    = lambda *args : h0
    if(na==nb): mf = scf.RHF(mol)
    else:       mf = scf.ROHF(mol)
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp  = lambda *args: s1
    mf._eri      = ao2mo.restore(1,h2,n)

    mf = scf.newton(mf)
    E0 = mf.kernel(rho)
    if(not mf.converged):
       mf = scf.newton(mf)
       E0 = mf.kernel(mf.make_rdm1())
    return mf,E0

def construct_iao(mol,mf):
    from pyscf import lo
    mo_occ = mf.mo_coeff[:,mf.mo_occ>0] # recuperano orbitali occupati
    a = lo.iao.iao(mol,mo_occ)          # costruzione IAO
    a = lo.vec_lowdin(a,mf.get_ovlp())  # ortonormalizzazione IAO
    integrals = transform_integrals(mol,mf,a) # costruisce la BO hamiltonian nella base IAO
    info = get_info(mol,mf,a)                 # 
    mf,_ = do_scf(integrals,info)             # eseguiamo un conto HF nella base IAO -> occupati+virt,valenza
    return np.dot(a,mf.mo_coeff)              # restituiamo orbitali occupati+virt,valenza 

def complement_iao(mol,mf,iao):
    from scipy import linalg as LA
    # orbitali molecolari proiettati fuori da occ+virt,valenza
    c = project_outside(mf.mo_coeff,iao,mf.get_ovlp()) # c = orbitali virtuali espansi nella base AO
    f = transform_integrals(mol,mf,c)[2] # proiettiamo l'operatore di Fock fuori da occ+virt,valenza
    e,w = LA.eigh(f)                     # lo diagonalizziamo
    print("spettro dell'operatore di Fock proiettato fuori da occ+vrt,valenza ",e) 
    return np.dot(c,w)                   # restituiamo orbitali virt,non-valenza

def do_fci(integrals,info,case='n'):
    from pyscf import fci

    h0,s1,f1,h1,h2 = integrals
    n,na,nb,rho    = info

    if(case=='n'):    dna,dnb,nroots=0,0,1
    elif(case=='u'):  dna,dnb,nroots=1,0,10000000
    elif(case=='d'):  dna,dnb,nroots=0,1,10000000
    elif(case=='uu'): dna,dnb,nroots=2,0,10000000
    elif(case=='ud'): dna,dnb,nroots=1,1,10000000
    elif(case=='dd'): dna,dnb,nroots=0,2,10000000
    else:             assert(False)

    cisolver           = fci.direct_nosym.FCI()
    cisolver.max_cycle = 100
    cisolver.conv_tol  = 1e-8
    cisolver.nroots    = nroots
    ECI,fcivec         = cisolver.kernel(h1,h2,n,(na-dna,nb-dnb),ecore=h0)
    # ECI = [E1,E2,E3,...]
    # fcivec = [Psi1,Psi2,Psi3,...]
    if(nroots==1): ECI,fcivec = [ECI],[fcivec]
    return [(E,f) for (E,f) in zip(ECI,fcivec)]

def compute_transitions(A,B,info,case,rank):
    from pyscf import fci
    n,na,nb,rho = info
    m = len(B)
    # <Psi(mu)|****|Psi0>
    # funzione: des_a = distrugge una particella a spin up
    # funzione: des_b = distrugge una particella a spin down
    if(rank==1):
       # <Psi(mu)|a(p,up)|Psi0>                # rank=1
       # <Psi(mu)|a(p,down)|Psi0>              # rank=1
       omega = np.zeros((m,n))
       for p in range(n):
           if(case=='u'): cp_gs = fci.addons.des_a(A[0][1],n,(na,nb),p)
           if(case=='d'): cp_gs = fci.addons.des_b(A[0][1],n,(na,nb),p)
           for m,(Em,Bm) in enumerate(B):                        # overlap con stati eccitati
               omega[m,p] = np.dot(cp_gs.flatten(),Bm.flatten()) # overlap (flatten->1D)
    elif(rank==2):
       # <Psi(mu)|a(s,up)a(r,up)|Psi0>         # rank=2
       # <Psi(mu)|a(s,down)a(r,down)|Psi0>     # rank=2
       # <Psi(mu)|a(s,up)a(r,down)|Psi0>       # rank=2
       omega = np.zeros((m,n,n))
       for r in range(n):
           if(case=='uu' or case=='ud'): cr_gs = fci.addons.des_a(A[0][1],n,(na,nb),r)
           else:                         cr_gs = fci.addons.des_b(A[0][1],n,(na,nb),r)
           for s in range(n):
               if(case=='uu'): csr_gs = fci.addons.des_a(cr_gs,n,(na-1,nb),s)
               if(case=='dd'): csr_gs = fci.addons.des_b(cr_gs,n,(na,nb-1),s)
               if(case=='ud'): csr_gs = fci.addons.des_b(cr_gs,n,(na-1,nb),s)
               for m,(Em,Bm) in enumerate(B):
                   omega[m,s,r] = np.dot(csr_gs.flatten(),Bm.flatten())
    elif(rank==3):
       # <Psi(mu)|[a*(q,up)a(s,up)+a*(q,down)a(s,down)]a(r,up)|Psi0> # rank=3
       # <Psi(mu)|[a*(q,up)a(s,up)+a*(q,down)a(s,down)]a(r,down)|Psi0> # rank=3
       omega = np.zeros((m,n,n,n))
       for r in range(n): # distruggi r
           if(case=='u'): cr_gs = fci.addons.des_a(A[0][1],n,(na,nb),r)
           else:          cr_gs = fci.addons.des_b(A[0][1],n,(na,nb),r)
           for s in range(n): # distruggi s
               for q in range(n): # crei q
                   if(case=='u'):
                                # a*(q,up)       a(s,up)
                      cqsr_gs = fci.addons.cre_a(fci.addons.des_a(cr_gs,n,(na-1,nb),s),n,(na-2,nb),q) \
                              + fci.addons.cre_b(fci.addons.des_b(cr_gs,n,(na-1,nb),s),n,(na-1,nb-1),q)
                                # a*(q,down)     a(s,down)
                   else:
                      cqsr_gs = fci.addons.cre_a(fci.addons.des_a(cr_gs,n,(na,nb-1),s),n,(na-1,nb-1),q) \
                              + fci.addons.cre_b(fci.addons.des_b(cr_gs,n,(na,nb-1),s),n,(na,nb-2),q)
                   for m,(Em,Bm) in enumerate(B):
                       omega[m,q,s,r] = np.dot(cqsr_gs.flatten(),Bm.flatten())
    else:
          assert(False)
    return omega

def nevpt2_hamiltonian(mol,mf,c,to_freeze):
    c1,c2 = c # base di valenza e base "fuori di valenza"
    n  = mol.nao_nr()
    m  = c1.shape[1]
    nf = len(to_freeze)
    basis = np.zeros((n,n))
    basis[:,:m] = c1
    basis[:,m:] = c2
    S = mf.get_ovlp()
    print("c1 orthonormal  ",ovlp(S,c1)) # ortonormalita'
    print("c2 orthonormal  ",ovlp(S,c2)) # ortonormalita'
    print("c1 ovlp c2      ",ovlp(S,c1,c2)) # ortogonalita' delle due basi
    print("mo_occ along c1 ",ovlp(S,c1,mf.mo_coeff[:,mf.mo_occ>0])) # controllo occupati generati da c1
    print("mo_occ along c2 ",ovlp(S,c2,mf.mo_coeff[:,mf.mo_occ>0])) # controllo occupati ortogonali a c2
    integrals  = transform_integrals(mol,mf,basis)     # passo nella base (c1,c2)
    integrals  = freeze_orbitals(integrals,to_freeze)  # congelamento degli orbitali di core 
    info       = [n-nf,(mol.nelectron+mol.spin)//2-nf,(mol.nelectron-mol.spin)//2-nf,None]

    if(n-nf<15):
       print("FCI, full basis")
       print(do_fci(integrals,info,'n')[0][0])

    to_remove  = [x for x in range(m-nf,n-nf)]
    nd         = len(to_remove)
    integrals1 = downfold(integrals,n-nf,to_remove)   # rimozione degli orbitali in c2
    info1      = [n-nf-nd,(mol.nelectron+mol.spin)//2-nf,(mol.nelectron-mol.spin)//2-nf,None]

    # ---------------------------------------------------------------------------------------

    # risoluzione dell'eq di Schrodinger ESATTA nello spazio di valenza con
    # n = (Na,Nb)
    # u = (Na-1,Nb)
    # d = (Na,Nb-1)
    # uu = (Na-2,Nb)    ----> (E(mu),Psi(mu))
    # ud = (Na-1,Nb-1)
    # dd = (Na,Nb-2)
    results   = {k:do_fci(integrals1,info1,k) for k in ['n','u','d','uu','ud','dd']}
    omega_q   = {k:compute_transitions(results['n'],results[k],info1,k,1) for k in ['u','d']}
    omega_qsr = {k:compute_transitions(results['n'],results[k],info1,k,3) for k in ['u','d']}
    omega_sr  = {k:compute_transitions(results['n'],results[k],info1,k,2) for k in ['uu','ud','dd']}

    # ---------------------------------------------------------------------------------------

    print("FCI, IAO ")
    print(do_fci(integrals1,info1,'n')[0][0])
    #exit()

    n_low  = info1[0]
    n_high = n-nf-n_low
    h0,s1,f1,h1,h2 = integrals
    t1 = h1

    dE = 0.0
    E0 = results['n'][0][0]

    for P in range(n_low,n_low+n_high):
        fP = f1[P,P]
        for k in ['u','d']:
            mkP_V_g = np.einsum('q,mq->m',t1[P,:n_low],omega_q[k])+np.einsum('rqs,mqsr->m',h2[P,:n_low,:n_low,:n_low],omega_qsr[k])
            Evec    = np.array([E+fP-E0 for (E,c) in results[k]])
            dE     -= sum(mkP_V_g**2/Evec)
    #        print("one-particle ",P,k,sum(mkP_V_g**2/Evec))
    #print(dE)
    for P in range(n_low,n_low+n_high):
        for Q in range(P+1,n_low+n_high):
            fPQ = f1[P,P]+f1[Q,Q]
            for k in ['uu','dd']:
                mkPQ_V_g = np.einsum('rs,msr->m',(h2[P,:n_low,Q,:n_low]-h2[Q,:n_low,P,:n_low])/2.0,omega_sr[k])
                Evec     = np.array([E+fPQ-E0 for (E,c) in results[k]])
                dE     -= sum(mkPQ_V_g**2/Evec)
    #            print("two-particle ",P,Q,k,sum(mkPQ_V_g**2/Evec))
    #print(dE)
    for P in range(n_low,n_low+n_high):
        for Q in range(n_low,n_low+n_high):
            fPQ = f1[P,P]+f1[Q,Q]
            for k in ['ud']:
                mkPQ_V_g = np.einsum('rs,msr->m',h2[P,:n_low,Q,:n_low],omega_sr[k])
                Evec     = np.array([E+fPQ-E0 for (E,c) in results[k]])
                dE      -= sum(mkPQ_V_g**2/Evec)
    #            print("two-particle ",P,Q,k,sum(mkPQ_V_g**2/Evec))
    print(E0+dE)
'''

