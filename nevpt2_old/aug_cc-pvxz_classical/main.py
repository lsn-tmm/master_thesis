import pyscf
from pyscf import gto,scf,mp,cc

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
    cm  = mp.MP2(mf,frozen=2)
    E1  = cm.kernel()[0]
    cm  = cc.CCSD(mf,frozen=2)
    E2  = cm.kernel()[0]
    print(cm.converged)
    if(not cm.converged):
       E2  = cm.kernel(t1=cm.t1,t2=cm.t2)[0]
    return E0,E1,E2

R_list = [0.70,0.72,0.74,0.76,0.78,0.80,0.82,0.84,0.86,0.88,
          0.90,0.92,0.94,0.96,0.98,1.00,1.02,1.04,1.06,1.08,
          1.10,1.12,1.14,1.16,1.18,1.20]

outf = open('E_anion_aug-cc-pvxz.txt','w')
for R in R_list:
    for x,b in zip([2,3,4],['aug-cc-pvdz','aug-cc-pvtz','aug-cc-pvqz']):
        E0,E1,E2 = compute(b,R,'anion')
        outf.write('%d %f %f %f \n' % (x,E0,E1,E2))
outf.close()

outf = open('E_radical_aug-cc-pvxz.txt','w')
for R in R_list:
    for x,b in zip([2,3,4],['aug-cc-pvdz','aug-cc-pvtz','aug-cc-pvqz']):
        E0,E1,E2 = compute(b,R,'radical')
        outf.write('%d %f %f %f \n' % (x,E0,E1,E2))
outf.close()

