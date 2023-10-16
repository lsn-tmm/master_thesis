import numpy as np
import pyscf
from   pyscf import gto,scf,mcscf,mp,cc,ci

def complete_es(mol,mf,EHF,nf):
    nel_frozen  = (nf,nf)
    orb_frozen  = nf     
    na,nb       = (mol.nelectron+mol.spin)//2-nf,(mol.nelectron-mol.spin)//2-nf
    nact        = mol.nao_nr()-nf
    mc          = mcscf.CASCI(mf,nact,(na,nb))
    mc.frozen   = orb_frozen
    ECASCI      = mc.kernel()[0]
    return ECASCI

def geometry(d):
    return [['O',(0,0,0)],['H',(0,0,d)]]

outf = open('radical_results.txt','w')
for R in ['0.70','0.72','0.74','0.76','0.78',
          '0.80','0.82','0.84','0.86','0.88',
          '0.90','0.92','0.94','0.96','0.98',
          '1.00','1.02','1.04','1.06','1.08',
          '1.10','1.12','1.14','1.16','1.18','1.20']:
    mol = gto.Mole()
    mol.build(atom     = geometry(float(R)),
              basis    = '6-31++g**',
              symmetry = 'coov',
              charge   = 0,
              spin     = 1,
              max_memory = 16000,
              verbose  = 6)
    mf             = scf.ROHF(mol)
    mf             = scf.newton(mf)
    EHF            = mf.kernel()
    a              = mf.stability()[0]
    EHF            = mf.kernel(a,mf.mo_occ)
    mf.analyze()
    ECASCI = complete_es(mol,mf,EHF,nf=1)
    outf.write('%.12f %.12f %.12f \n' % (float(R),EHF,ECASCI))

