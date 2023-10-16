import numpy as np
import pyscf
from   pyscf import gto,scf,mcscf

# ----- define a molecular system

basis  = '6-31g'
sym    = 'Coov'
charge = -1
spin   = 0

mol = gto.Mole()
mol.build(atom     = [['O',(0,0,0)],['H',(0,0,0.9640)]],
          basis    = basis,
          symmetry = sym,
          charge   = charge,
          spin     = spin,
          verbose  = 4)

# ----- carry out a HF calculation

mf             = scf.ROHF(mol)
mf.irrep_nelec = {'A1':6,'E1x':2,'E1y':2}
E              = mf.kernel()
mf.stability()
mf.analyze()

