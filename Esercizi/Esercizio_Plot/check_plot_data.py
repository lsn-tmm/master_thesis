import numpy as np
import matplotlib.pyplot as plt

res = {}
res['basis'] = ['sto-6g','6-31g','6-31++g','6-31g**','6-31++g**','cc-pvdz','cc-pvtz','cc-pvqz','aug-cc-pvdz','aug-cc-pvtz']
res['E_HF'] = []
res['E_MP'] = []
res['E_CISD'] = []
res['E_CCSD'] = []
res['E_CCSD(T)'] = []
res['E_CASCI'] = []
res['E_CASSCF'] = []
    
data = np.load('oh_data.npy', allow_pickle=True).item()
iao_data = np.load('iao_oh_data.npy', allow_pickle=True).item()

print('#'*100)
print('ORIGINAL BASIS')
print('#'*100)
for i in data:
    print(i,data[i])
    if i == 'E_CASSCF_anion' :  print('\n') 
    
print('#'*100)
print('IAO')
print('#'*100)
    
for i in iao_data:   
    print(i,iao_data[i])
    if i == 'E_CASSCF_anion' :  print('\n') 