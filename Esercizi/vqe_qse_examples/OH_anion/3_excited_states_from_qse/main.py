import numpy as np
import sys
sys.path.append('../../commons/')
sys.path.append('../../qse_subroutines')
from subroutines import build_qse_operators,excitation_numbers
from QSE         import QSE
from job_details import write_job_details

outfile  = open('qse.txt','w')
write_job_details(outfile)

# prima si fa un conto di stato fondamentale, e.g. con il VQE; questo ci da' una funzione d'onda Psi
vqe_res   = np.load('../2_ground_state_from_vqe/vqe_q_uccsd_results.npy',allow_pickle=True).item()

# si definisce un insieme {E_i}_i di operatori, che il QSE chiama "excitation operators" --- QUESTI LI SCEGLIAMO NOI
# a noi interesseranno 5 casi per l'NEVPT2
# caso 1) E_i = a_{p,up}              (distruzione di una particella a spin-up)                 u
# caso 2) E_i = a_{p,down}            (distruzione di una particella a spin-down)               d 
# caso 3) E_i = a_{p,up} a_{q,up}     (distruzione di due particelle a spin-up) p<q             uu
# caso 4) E_i = a_{p,down} a_{q,down} (distruzione di due particelle a spin-down) p<q           dd
# caso 5) E_i = a_{p,up} a_{q,down}   (distruzione di due particelle a spin opposti) q,p liberi ud
# funzione che restituisca una lista di operatori corrispondenti al caso 1...5
for c in ['u','d','uu','dd','ud']:
    operators = build_qse_operators(class_of_operators=c,mol_info=vqe_res)
    print("Case ",c)
    print(len(operators))

operators = build_qse_operators(class_of_operators='u',mol_info=vqe_res)

# QSE: produzione di due matrici S_{ij} = <Psi|E_i* E_j|Psi>
#      H_{ij} = <Psi|E_i* H E_j|Psi>
#      HC = eSC
QSE_calc  = QSE(psi           = vqe_res['vqe_circuit'],
                operators     = operators,
                mol_info      = vqe_res,
                instance_dict = {'instance':'statevector_simulator','shots':1})
QSE_calc.run(outfile=outfile,resfile='qse_results.npy')

ns,nc = excitation_numbers(vqe_res)
H,S   = QSE_calc.matrices['H'],QSE_calc.matrices['S']

