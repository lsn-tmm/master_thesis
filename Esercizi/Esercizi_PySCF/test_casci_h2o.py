from pyscf import gto,scf,fci,mcscf
  
mol = gto.Mole()
mol.build(verbose  = 4,
          atom     = [['O',(0.0000, 0.0000, 0.1173)],
                      ['H',(0.0000, 0.7572,-0.4692)],
                      ['H',(0.0000,-0.7572,-0.4692)]],
          basis    = 'cc-pvdz',
          charge   = 0,
          spin     = 0,
          symmetry = 'C2v')

myhf = mol.RHF()

myhf.kernel()
myhf.analyze()

# active space of H 1s and O 2s, 2p orbitals
mymc = myhf.CASCI(nelecas=(4,4),ncas=6)
cas_space_symmetry = {'A1':3,'B1':2,'B2':1}
mo = mcscf.sort_mo_by_irrep(mymc,myhf.mo_coeff,cas_space_symmetry)
mymc.fcisolver.wfnsym = 'A1'

mymc.kernel(mo)
mymc.analyze()
