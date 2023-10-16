class Born_Oppenheimer:
    
    
    # Data Members
    
    _mol = ''
    _mf = ''
    
    _h0 = ''
    _s1 = ''
    _h1 = ''
    _h2 = ''
    
    _no = ''
    _ne = ''
    _na = ''
    _nb = ''
    
    _C = ''
    _irr = ''
    
    _res = {}
    
    
    # Methods:
    
    def __init__(self,mol,mf,C,n):  
        import numpy as np
        import pyscf
        import itertools
        from scipy import linalg as LA
        from pyscf import ao2mo
        
        self._mol = mol
        self._mf = mf
        
        self._h0 = mol.energy_nuc()                                  # E0
        
        self.transform_integrals(C)
        # self._s1 = np.einsum('ab,ai,bk->ik',mf.get_ovlp(),C,C)     # overlap (ci|ck) = \sum_{ab} C(ai) S(ab) C(bk)
        # self._h1 = np.einsum('ab,ai,bk->ik',mf.get_hcore(),C,C)    # h1   h(ik)   = \sum_{ab} C(ai) h(ab) C(bk)
        # self._h2 = ao2mo.restore(1,ao2mo.kernel(mol,C),C.shape[1]) # h2  (pr|qs) = \sum_{ab} C(pi) C(rj) C(qk) C(sl)(ij|kl)
        self._no = self._h1.shape[0]
        self._ne = ((mol.nelectron+mol.spin)//2,(mol.nelectron-mol.spin)//2)
        self._na = self._ne[0]
        self._nb = self._ne[1]
        
        S         = mf.get_ovlp()
        S_bar     = np.dot(C.T,np.dot(S,C)) # S_bar
        S_bar_inv = LA.inv(S_bar)           # inversa di S_bar
        X         = mf.mo_coeff[:,:self._na]
        X         = np.dot(S,X)
        X         = np.dot(C.T,X) 
        Y         = np.dot(S_bar_inv,X)
        
        if(mol.spin==0):                               # shell chiusa, na=nb
           rho_0 = 2*np.dot(Y[:,:self._nb],Y[:,:self._nb].T)       # \rho(pr) = 2*\sum_{i=1}^{na=nb} Y(pi)Y(ri)
        else:                                          # shell aperta
           rho_0 = np.zeros((2,Y.shape[0],Y.shape[0])) # \rho(up,pr)   = \sum_{i=1}^{na} Y(pi)Y(ri)
           rho_0[0,:,:] = np.dot(Y[:,:self._na],Y[:,:self._na].T)  # \rho(down,pr) = \sum_{i=1}^{nb} Y(pi)Y(ri)
           rho_0[1,:,:] = np.dot(Y[:,:self._nb],Y[:,:self._nb].T)
        
        E,V = self.do_scf(rho_0=rho_0)[:2]         # <=== eseguiamo un conto RHF nella base IAO
        # ------------------------------------------------------------
        # passaggio alla base degli orbitali di Hartree-Fock (occupati e virtuali) espressi nella base Ck
        self._C                = np.dot(C,V)
        #print(mol.irrep_name)
        self._C[:,:self._nb]         = pyscf.symm.symmetrize_orb(mol,self._C[:,:self._nb])
        if(self._na>self._nb): C[:,self._nb:self._na]  = pyscf.symm.symmetrize_orb(mol,self._C[:,self._nb:self._na])
        self._C[:,self._na:]         = pyscf.symm.symmetrize_orb(mol,self._C[:,self._na:])
        
        self.transform_integrals(self._C)
        
        #E,V         = do_scf()[:2]
        
        irrep_names = {y:x for x,y in zip(self._mol.irrep_name,self._mol.irrep_id)}
        self._irr   = [irrep_names[x] for x in pyscf.symm.label_orb_symm(self._mol,self._mol.irrep_id,self._mol.symm_orb,self._C)]
        
    def transform_integrals(self, C = None):
        import numpy as np
        from pyscf import ao2mo
        
        if(C is not None): self._C = C
        
        self._s1          = np.einsum('ab,ai,bk->ik',self._mf.get_ovlp(),self._C,self._C)
        self._h1          = np.einsum('ab,ai,bk->ik',self._mf.get_hcore(),self._C,self._C)
        self._h2          = ao2mo.restore(1,ao2mo.kernel(self._mol,self._C),self._C.shape[1])
        self._no          = self._h1.shape[0]
        
    #---------------------------------------Calculation-----------------------------------------------
        
    def do_scf(self, rho_0=None):
        import numpy as np
        from pyscf import gto,scf,ao2mo
        
        n     = self._no
        mol   = gto.M(verbose=4)
        mol.nelectron     = self._na+self._nb
        mol.spin          = self._na-self._nb
        mol.incore_anyway = True
        mol.nao_nr        = lambda *args : n
        mol.energy_nuc    = lambda *args : self._h0
        if(self._na==self._nb):
           mf            = scf.RHF(mol)
           mf.get_hcore  = lambda *args: self._h1
           mf.get_ovlp   = lambda *args: self._s1
           mf._eri       = ao2mo.restore(1,self._h2,n)
           if(rho_0 is None):
              rho_0 = np.zeros((n,n))
              for i in range(self._na): rho_0[i,i] = 2.0
        else:
           mf            = scf.ROHF(mol)
           mf.get_hcore  = lambda *args: self._h1
           mf.get_ovlp   = lambda *args: self._s1
           mf._eri       = ao2mo.restore(1,self._h2,n)
           if(rho_0 is None):
              rho_0 = np.zeros((2,n,n))
              for i in range(self._na): rho_0[0,i,i] = 1.0
              for i in range(self._nb): rho_0[1,i,i] = 1.0
        mf = scf.newton(mf)
        E0 = mf.kernel(rho_0)
        if(not mf.converged):
           mf = scf.newton(mf)
           E0 = mf.kernel(mf.make_rdm1())
        a  = mf.stability()[0]
        E0 = mf.kernel(a,mf.mo_occ)

        return E0,mf.mo_coeff,mol,mf        

        
    # ---------------------------------------Utility-------------------------------------------

    def print_mol_data(self,verbose=False):
        import itertools
        
        print("-"*53)
        print("number of orbitals  ",self._no)
        print("number of electrons ",self._ne) 
        print("energy offset       ",self._h0)
        print("irreps of orbitals  ",self._irr)
        if(verbose):
           n = self._no
           for p,r in itertools.product(range(n),repeat=2):
               print("h1(%d,%d) = %.8f " % (p,r,self._h1[p,r]))
           for p,r,q,s in itertools.product(range(n),repeat=4):
               print("h1(%d,%d,%d,%d) = %.8f " % (p,r,q,s,self._h2[p,r,q,s]))
        print("-"*53)

    def dump_on_file(self,fname='dictionary_output'):
        import numpy as np
        
        mol_info = {'no':self._no, # numero di orbitali 
                    'ne':self._ne, # numero di elettroni (na,nb) a spin-up e spin-down
                    'h0':self._h0, # costante (energy offset)
                    's1':self._s1, # matrice di overlap       nella base di orbitali selezionati
                    'h1':self._h1, # hamiltoniano a un corpo  nella base di orbitali selezionati
                    'h2':self._h2} # hamiltoniano a due corpi nella base di orbitali selezionati
        np.save(fname,mol_info,allow_pickle=True)

    def read_from_file(self,fname='dictionary_output'):
        return np.load(fname,allow_pickle=True).item()

    # ----------------------------------------Active Space----------------------------------------------
    
    
    def downfold(self,to_remove,virtual=False):
        import numpy as np
        
        to_keep = [i for i in range(self._no) if i not in to_remove]  
        self._s1 = self._s1[np.ix_(to_keep,to_keep)]
        self._h1 = self._h1[np.ix_(to_keep,to_keep)]
        self._h2 = self._h2[np.ix_(to_keep,to_keep,to_keep,to_keep)]
        self._no = self._no - len(to_remove)       
        if(not virtual):
           self._ne = (self._ne[0]-len(to_remove),self._ne[1]-len(to_remove))
           self._na, self._nb = self._ne[0], self._ne[1]
        self._irr   = [self._irr[x] for x in to_keep]
    
    def freeze_orbitals(self,to_freeze):
        import numpy as np
        
        n  = self._no
        h0 = self._h0
        h1 = self._h1
        h2 = self._h2
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
        self._h0 = h0 + Enuc
        self._h1 = h1
        self._h2 = h2

    def build_active_space(self,to_keep):
        n = self._no
        #na = mol_info['ne'][0]
        #nb = mol_info['ne'][1]
        not_to_keep = [x for x in range(n) if x not in to_keep]
        to_freeze   = [x for x in not_to_keep if x<self._nb]
        to_discard  = [x for x in not_to_keep if x>=self._na]
        self.freeze_orbitals(to_freeze)
        self.downfold(to_discard,virtual=True)
        self.downfold(to_freeze,virtual=False)
    
    # --------------------------------------------------------------------------------------

    def do_electronic_structure(self):
        import numpy as np
        from pyscf import gto,scf,lo,tools  
        from pyscf import mp,ci,cc,mcscf
        
        EHF,C,mol,mf = self.do_scf()
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
        mc          = mcscf.CASSCF(mf,nact,(na,nb))
        mc.frozen   = nf
        #ECASSCF     = mc.kernel()[0]
        
        print("Hartree-Fock   ",EHF)
        print("Moller-Plesset ",EMP)
        print("CISD           ",ECISD)
        print("CCSD           ",ECCSD)
        print("CCSD(T)        ",ECCSDT)
        print("CASCI          ",ECASCI)
        #print("CASSCF         ",ECASSCF)  
        
        self._res['E_HF'] = EHF
        self._res['E_MP'] = EMP
        self._res['E_CISD'] = ECISD
        self._res['E_CCSD'] = ECCSD
        self._res['E_CCSD(T)'] = ECCSDT
        self._res['E_CASCI'] = ECASCI
        #self._res['E_CASSCF'] = ECASSCF

    
    
    
    
    
    
    
    
    
    
    
