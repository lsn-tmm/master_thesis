import numpy as np
import pyscf
from   pyscf import gto,scf,mcscf

# ----- define a molecular system

basis  = '6-31g'
sym    = 'Coov'
charge = 0
spin   = 1

mol = gto.Mole()
mol.build(atom     = [['O',(0,0,0)],['H',(0,0,0.9697)]],
          basis    = basis,
          symmetry = sym,
          charge   = charge,
          spin     = spin,
          verbose  = 4)

# ----- carry out a HF calculation

mf             = scf.ROHF(mol)
mf.irrep_nelec = {'A1':(3,3),'E1x':(1,1),'E1y':(1,0)}
E              = mf.kernel()
mf.stability()
mf.analyze()

