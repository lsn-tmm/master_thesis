import numpy     as np
from pyscf       import gto,scf,lo,tools  # librerie di PySCF
from pyscf.tools import cubegen           # gto: Gaussian Type Orbital
                                          # scf: Self-Consistent Field
from pyscf       import mp,ci,cc,mcscf

R_rad = 0.9697
R_an  = 0.9640
R     = R_an
geo   = []
geo.append(['O',(0,0,0)]) # geometria: lista di elementi della forma [A,(x,y,z)] xyz in Angstrom
geo.append(['H',(0,0,R)])

res = {}
res['basis'] = ['sto-6g','6-31g','6-31++g','6-31g**','6-31++g**','cc-pvdz','cc-pvtz','cc-pvqz','aug-cc-pvdz','aug-cc-pvtz','aug-cc-pvqz']
res['E_HF_anion'] = []
res['E_MP_anion'] = []
res['E_CISD_anion'] = []
res['E_CCSD_anion'] = []
res['E_CCSD(T)_anion'] = []
res['E_CASCI_anion'] = []
res['E_CASSCF_anion'] = []


for i in res['basis']:
    
    print("#"*53)
    print("START:", i)
    print("#"*53)

    mol = gto.M(atom     = geo,
                basis    = i,         # Slater Type Orbital - 6g = 6 Gaussiane per orbitale
                charge   = -1,        # carica totale (nucleare + elettronica)
                spin     = 0,         # STRANO MA VERO: N(up)-N(down)
                verbose  = 4,         # quanto e cosa stampare
                symmetry = True)      # simmetrie molecolari (rotazione attorno all'asse OH)

    mf  = scf.RHF(mol)
    mf  = scf.newton(mf)         # solutore al secondo ordine E(C+dC) = E(C) + dC*g(C) + dC H(C) dC
    E   = mf.kernel()            # kernel = esegui ROHF
    a   = mf.stability()[0]      # calcolo g(C) ed H(C), proposta di nuovi orbitali possibilmente migliori (a)
    EHF = mf.kernel(a,mf.mo_occ) # kernel = esegui ROHF partendo dagli orbitali a
    mf.analyze()

    #
    # categoria 1: espansioni intorno all'Hartree-Fock
    #

    # ----- Moller-Plesset second order perturbation theory
    #
    # O----H               O[1s,2s,2px,2py,2pz] H[1s] # se la base e' cc-pCVnZ allora si possono studiare gli elettroni nell'1s
    nf     = 1                                        # altrimenti, l'orbitale 1s va congelato (rimane doppiamente occupato)
    mymp   = mp.MP2(mf,frozen=nf)                     # H = F + (H-F); F ha autofunzioni note, (H-F) e' perturbazione
    EMP    = EHF + mymp.kernel()[0]                   # Rayleigh-Schrodinger perturbation theory E = E(HF) - \sum_m |<m|H-F|0>|^2/(Em-E0)
    mcisd  = ci.CISD(mf,frozen=nf)                    # CISD: CI = configuration interaction; S = singole eccitazioni sull'Hartree-Fock
    ECISD  = EHF   + mcisd.kernel()[0]                # D = doppie eccitazioni sull'Hartree-Fock
    # CISD: H | Psi(HF) >      = T |Psi(HF) > + \sum_{ai} V(ai) c*(a)c(i) |Psi(HF)> + \sum_{abij} W(abij) c*(a)c*(b)c(j)c(i) |Psi(HF)>
    # CCSD: exp(T) | Psi(HF) >
    # Cluster operator: T = \sum_{ai} t(ai) c*(a)c(i) + \sum_{abij} t(abij) c*(a)c*(b)c(j)c(i) + ...
    mc     = cc.CCSD(mf,frozen=nf)
    ECCSD  = EHF   + mc.kernel()[0]
    ECCSDT = ECCSD + mc.ccsd_t()
    # CCSD(T): si calcola il CCSD e poi si stima il contributo all'energia di correlazione dalle triple eccitazioni perturbativamente
    # CCSD(T) + cc-pVTZ e' chiamato il "gold standard"

    # 
    # categoria 2: diagonalizzazione piu' o meno esatta
    #

    # CAS = complete active space
    # CI  = configuration interaction
    # SCF = self-consistent field
    # CASCI: si sceglie un insieme di orbitali e di elettroni, si costruiscono tutti i possibili determinanti di Slater 
    # che coinvolgono eccitazioni di questi orbitali ed elettroni
    # e si risolve l'eq di Schrodinger in questo sottospazio
    # 150 -------- INATTIVI
    # ............ INATTIVI
    #  7  -------- INATTIVI
    #  6  -------- ATTIVI
    #  5  ---ud--- ATTIVI
    #  4  ---ud--- INATTIVI
    #  3  ---ud--- INATTIVI
    #  2  ---ud--- INATTIVI
    #  1  ---ud--- CONGELATI (FROZEN)

    #  6  ---- --d- --u- -ud- complete active space
    #  5  -ud- -u-- -d-- ----

    nel_frozen  = (nf,nf)  # congelare  l'1s
    orb_frozen  = nf       # coongelare l'1s
    na,nb       = (mol.nelectron+mol.spin)//2-nf,(mol.nelectron-mol.spin)//2-nf # tutti elettroni attivi tranne 1s
    nao_minimal = 6                                                             # prendere 6 orbitali di Hartree-Fock
    nact        = nao_minimal-nf                                                # tutti questi 6 orbitali attivi tranne l'1s
    mc          = mcscf.CASCI(mf,nact,(na,nb))
    mc.frozen   = orb_frozen
    ECASCI      = mc.kernel()[0]

    # CASSCF: come il CASCI, ma gli orbitali non sono orbitali di campo medio, sono orbitali ottimizzati variazionalmente

    mc          = mcscf.CASSCF(mf,nact,(na,nb))
    mc.frozen   = orb_frozen
    ECASSCF     = mc.kernel()[0]

    print("Hartree-Fock   ",EHF)
    print("Moller-Plesset ",EMP)
    print("CISD           ",ECISD)
    print("CCSD           ",ECCSD)
    print("CCSD(T)        ",ECCSDT)
    print("CASCI          ",ECASCI)
    print("CASSCF         ",ECASSCF)
    
    res['E_HF_anion'].append(EHF)
    res['E_MP_anion'].append(EMP)
    res['E_CISD_anion'].append(ECISD)
    res['E_CCSD_anion'].append(ECCSD)
    res['E_CCSD(T)_anion'].append(ECCSDT)
    res['E_CASCI_anion'].append(ECASCI)
    res['E_CASSCF_anion'].append(ECASSCF)
    
    print("#"*53)
    print("END:", i)
    print("#"*53)
    
    
try :
    data = np.load('../oh_data.npy',allow_pickle=True).item()
    for i in res:
        data[i] = res[i]
    np.save('../oh_data',data,allow_pickle=True)
except:
    np.save('../oh_data',res,allow_pickle=True)
    



    
    

# Sistema S1, Sistema S2
# E(S1,metodo) = E(S1,esatta) + BIAS(S1,metodo)
# E(S2,metodo) = E(S2,esatta) + BIAS(S2,metodo)
# E(S1,metodo)-E(S2,metodo) = E(S1,esatta)-E(S2,esatta) + BIAS(S1,metodo)-BIAS(S2,metodo)
# quando   BIAS(S1,metodo),BIAS(S2,metodo) "grande"  (rispetto E(S1,esatta)-E(S2,esatta) o a certe soglie come la chemical accuracy)
# tuttavia BIAS(S1,metodo)-BIAS(S2,metodo) "piccola" (rispetto E(S1,esatta)-E(S2,esatta) o a certe soglie come la chemical accuracy)
# allora "cancellation of errors"

#---------------------------------------------------------------------#
# metodo  | size-extensive E(A+B)=E(A)+E(B) | variational | cost      #
#---------------------------------------------------------------------#
# HF      | SI                              | SI          | M**4      #
# MP2     | NO                              | NO          | M**5      #
# CISD    | NO                              | SI          | M**7      #
# CCSD    | SI                              | NO          | M**6      #
# CCSD(T) | SI                              | NO          | M**9      #
# CASCI   | NO                              | SI          | exp(Nact) #
# CASSCF  | NO                              | SI          | exp(Nact) #
# FCI     | SI                              | SI          | exp(M)    #
#---------------------------------------------------------------------#

#------------------------------------------#
# ORBITALI      CORRELAZIONE   COMPUTER    #
# VIRTUALI      DINAMICA       CLASSICO    #
# VALENZA       STATICA        QUANTISTICO #
#------------------------------------------#

# E(R,basis,method) "potential energy surface = energia versus bondlength"
# R_{eq}(basis,method) = argmin_R E(R,basis,method)
