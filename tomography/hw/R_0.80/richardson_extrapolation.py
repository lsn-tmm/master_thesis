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

dir = 'manila'

bloch_1 = np.load('tomo_nm_em_bloch.npy',allow_pickle=True).item()
bloch_3 = np.load('tomo_nm_em_re_bloch.npy',allow_pickle=True).item()
bloch_r = {k:ric(bloch_1[k],bloch_3[k]) for k in bloch_1.keys()}
np.save('tomo_nm_em_re_extrapolated_bloch.npy',bloch_r,allow_pickle=True)

bloch_1 = np.load(dir+'_em_bloch.npy',allow_pickle=True).item()
bloch_3 = np.load(dir+'_em_re_bloch.npy',allow_pickle=True).item()
bloch_r = {k:ric(bloch_1[k],bloch_3[k]) for k in bloch_1.keys()}
np.save(dir+'_em_re_extrapolated_bloch.npy',bloch_r,allow_pickle=True)

