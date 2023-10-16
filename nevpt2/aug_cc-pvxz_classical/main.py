import pyscf
from pyscf import gto,scf,mp,cc,mcscf

def compute(b,R,kind):
    if(kind=='anion'):
       mol = pyscf.M(atom=[['O',(0,0,0)],['H',(0,0,R)]],charge=-1,spin=0,basis=b,symmetry=True)
       mf  = mol.RHF()
    else:
       mol = pyscf.M(atom=[['O',(0,0,0)],['H',(0,0,R)]],charge=0,spin=1,basis=b,symmetry=True)
       mf  = mol.ROHF()
    mf  = scf.newton(mf)
    E0  = mf.kernel()
    if(not mf.converged):
       E0  = mf.kernel(mf.make_rdm1())
    cm  = mp.MP2(mf,frozen=1)
    E1  = cm.kernel()[0]
    cm  = cc.CCSD(mf,frozen=1)
    E2  = cm.kernel()[0]
    if(not cm.converged):
       E2  = cm.kernel(t1=cm.t1,t2=cm.t2)[0]
    E3 = cm.ccsd_t()

    cas_space_symmetry = {'A1':3,'E1x':1,'E1y':1}
    na    = (mol.nelectron+mol.spin)//2-1
    nb    = (mol.nelectron-mol.spin)//2-1
    mycas = mf.CASSCF(ncas=5,nelecas=(na,nb))
    mo    = mcscf.sort_mo_by_irrep(mycas,mf.mo_coeff,cas_space_symmetry)
    mycas.frozen = 1
    mycas.max_cycle_macro = 100
    mycas.conv_tol        = 1e-9
    mycas.conv_tol_grad   = 1e-6
    E4 = mycas.kernel(mo)[0]
    return E0,E1,E2,E2+E3,E4

R_list = [0.80,0.82,0.84,0.86,0.88,0.90,0.92,0.94,0.96,0.98,1.00,1.02,1.04,1.06,1.08,1.10,1.12,1.14,1.16,1.18,1.20]

outf = open('E_anion_aug-cc-pvxz.txt','w')
for R in R_list:
    for x,b in zip([2,3,4],['aug-cc-pvdz','aug-cc-pvtz','aug-cc-pvqz']):
        E0,E1,E2,E3,E4 = compute(b,R,'anion')
        outf.write('%d %f %f %f %f %f \n' % (x,E0,E1,E2,E3,E4))
outf.close()

outf = open('E_radical_aug-cc-pvxz.txt','w')
for R in R_list:
    for x,b in zip([2,3,4],['aug-cc-pvdz','aug-cc-pvtz','aug-cc-pvqz']):
        E0,E1,E2,E3,E4 = compute(b,R,'radical')
        outf.write('%d %f %f %f %f %f \n' % (x,E0,E1,E2,E3,E4))
outf.close()

