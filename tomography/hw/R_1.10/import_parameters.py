import numpy as np

par = {}
for species in ['radical','anion']:
    for basis in ['aug-cc-pvtz','aug-cc-pvqz']:
        x = np.load('../../../nevpt2_old/%s/%s/R_1.10/vqe_su2_output.npy'%(basis,species),allow_pickle=True).item()
        x = x['opt_params']
        par[species+'_'+basis] = np.array(x)

np.save('parameters.npy',par,allow_pickle=True)

#anion bs 0110
#radical bs 0100 

