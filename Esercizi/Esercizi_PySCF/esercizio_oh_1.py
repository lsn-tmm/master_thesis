import numpy     as np
from pyscf       import gto,scf,lo,tools  # librerie di PySCF
from pyscf.tools import cubegen           # gto: Gaussian Type Orbital
                                          # scf: Self-Consistent Field

# ---------------------------------------- GTO

R   = 0.9697              # experimental OH bondlength in Angstrom
geo = []
geo.append(['O',(0,0,0)]) # geometria: lista di elementi della forma [A,(x,y,z)] xyz in Angstrom
geo.append(['H',(0,0,R)])

mol = gto.M(atom     = geo,
            basis    = 'sto-6g', # Slater Type Orbital - 6g = 6 Gaussiane per orbitale
            charge   = 0,        # carica totale (nucleare + elettronica)
            spin     = 1,        # STRANO MA VERO: N(up)-N(down)
            verbose  = 4,        # quanto e cosa stampare
            symmetry = True)     # simmetrie molecolari (rotazione attorno all'asse OH)

# ---------------------------------------- GTO

n  = mol.nao_nr()               # orbitali atomici
AO = np.eye(n)                  # |a(i) > = \sum_{j} delta(ji) |a(j)>

for i in range(n):
    tools.cubegen.orbital(mol,'AO_%d.cube'% i,AO[:,i])

for i,ai in enumerate(mol.ao_labels()):
    print("orbitale atomico ",i," = ",ai)

# ---------------------------------------- SCF

# RHF
# N_up = N_down
# --------
# ---ud---
# ---ud---
# ---ud---

# ROHF
# N_up > N_down
# --------
# ---u----
# ---ud---
# ---ud---

# UHF   XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# -------   --------
# ---u---   ---d----
# ---u---   ---d----
# ---u---   ---d----
# GHF   XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

mf  = scf.ROHF(mol)
mf  = scf.newton(mf)         # solutore al secondo ordine E(C+dC) = E(C) + dC*g(C) + dC H(C) dC
E   = mf.kernel()            # kernel = esegui ROHF
a   = mf.stability()[0]      # calcolo g(C) ed H(C), proposta di nuovi orbitali possibilmente migliori (a)
E   = mf.kernel(a,mf.mo_occ) # kernel = esegui ROHF partendo dagli orbitali a
mf.analyze()

# ---------------------------------------- SCF

C   = mf.mo_coeff            # orbitali di HF, nella letteratura chiamati MOs (molecular orbitals)
for i in range(n):
    tools.cubegen.orbital(mol,'MO_%d.cube'% i,C[:,i])
    for k,ak in enumerate(mol.ao_labels()):
        print("MO ",i," CONTRIBUTION FROM ",ak,C[k,i])
    print("---------")



