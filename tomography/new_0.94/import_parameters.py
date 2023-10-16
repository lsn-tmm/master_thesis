import numpy as np

par = {}
for species in ['radical','anion']:
    for basis in ['aug-cc-pvqz']:
        for R in ['0.94']:
            x = np.load('../../nevpt2/%s/%s/R_%s/vqe_su2_output.npy'
                        % (basis,species,R),allow_pickle=True).item()
            x = x['opt_params']
            par[species+'_'+basis+'_'+R] = np.array(x)

np.save('parameters.npy',par,allow_pickle=True)

print(par.keys())

#anion bs 101110
#radical bs 101100

