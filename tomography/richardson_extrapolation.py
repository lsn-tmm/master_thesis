import numpy as np
import os

def ric(v1,v2):
    ve = np.zeros(v1.shape)
    ve[:,0] = v1[:,0]-(v2[:,0]-v1[:,0])/2.0
    for m in range(ve.shape[0]):
        if(ve[m,0]>1): ve[m,0]=1
        if(ve[m,0]<-1): ve[m,0]=-1
    ve[:,1] = np.sqrt(v1[:,1]**2+v2[:,1]**2)/2.0
    return ve

RSH_list = sorted(os.listdir('../01_preprocessing/geometries/'))
RSH_list = [filename[7:len(filename)-4] for filename in RSH_list]
RSH_list = [float(R)/1000.0 for R in RSH_list]
RSH_list = ['%s'%R for R in RSH_list]

for R in RSH_list:
    bloch_1 = np.load('nm_em/R_%s_nm_em_bloch.npy'%R,allow_pickle=True).item()
    bloch_3 = np.load('nm_em_re_3/R_%s_nm_em_re_3_bloch.npy'%R,allow_pickle=True).item()
    bloch_r = {k:ric(bloch_1[k],bloch_3[k]) for k in bloch_1.keys()}
    np.save('nm_em_re/R_%s_nm_em_re_bloch.npy'%R,bloch_r,allow_pickle=True)

