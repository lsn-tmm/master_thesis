import numpy as np

class my_wavefunction:
    
    def __init__(self, energy = None, wavefunction = None):
        self.energy = np.asarray(energy)
        self.wavefunction = np.asarray(wavefunction)
        
    def get_num_psi(self):
        return len(self.energy)
    
    def get_wavefunction(self,m = None):
        if (m == None):
            return self.wavefunction
        else:
            d = len(self.wavefunction.shape)
            command = ':,' * (d-1)
            return eval('self.wavefunction[%sm]'%command)
    
    def get_energy(self):
        return self.energy
    
    def print(self):
        print(self.energy, self.wavefunction)
        
class excited_states:
    
    def __init__(self):
        self.u = my_wavefunction()
        self.d = my_wavefunction()
        self.uu = my_wavefunction()
        self.dd = my_wavefunction()
        self.ud = my_wavefunction()
        
    def set_u(self, energies, wavefunctions):
        self.u = my_wavefunction(energies,wavefunctions)
    
    def set_d(self, energies, wavefunctions):
        self.d = my_wavefunction(energies,wavefunctions)
        
    def set_uu(self, energies, wavefunctions):
        self.uu = my_wavefunction(energies,wavefunctions)
    
    def set_dd(self, energies, wavefunctions):
        self.dd = my_wavefunction(energies,wavefunctions)
    
    def set_ud(self, energies, wavefunctions):
        self.ud = my_wavefunction(energies,wavefunctions)

    def set_excited_states(self, sector, energies, wavefunctions):
        if(sector=='u'):    self.set_u(energies, wavefunctions)
        elif(sector=='d'):  self.set_d(energies, wavefunctions)
        elif(sector=='uu'): self.set_uu(energies, wavefunctions)
        elif(sector=='ud'): self.set_ud(energies, wavefunctions)
        elif(sector=='dd'): self.set_dd(energies, wavefunctions)
        else:               raise NameError('%s is not a valid sector: u,d,uu,ud or dd expected'%sector)

    def get_num_psi(self,sector):
        if(sector=='u'):    return self.u.get_num_psi()
        elif(sector=='d'):  return self.d.get_num_psi()
        elif(sector=='uu'): return self.uu.get_num_psi()
        elif(sector=='ud'): return self.ud.get_num_psi()
        elif(sector=='dd'): return self.dd.get_num_psi()
        else:               raise NameError('%s is not a valid sector: u,d,uu,ud or dd expected'%sector)

    def get_excited_state_wf(self, sector, m = None):
        if(sector=='u'):    return self.u.get_wavefunction(m)
        elif(sector=='d'):  return self.d.get_wavefunction(m)
        elif(sector=='uu'): return self.uu.get_wavefunction(m)
        elif(sector=='ud'): return self.ud.get_wavefunction(m)
        elif(sector=='dd'): return self.dd.get_wavefunction(m)
        else:               raise NameError('%s is not a valid sector: u,d,uu,ud or dd expected'%sector)
            
    def get_excited_state_en(self, sector):
        if(sector=='u'):    return self.u.get_energy()
        elif(sector=='d'):  return self.d.get_energy()
        elif(sector=='uu'): return self.uu.get_energy()
        elif(sector=='ud'): return self.ud.get_energy()
        elif(sector=='dd'): return self.dd.get_energy()
        else:               raise NameError('%s is not a valid sector: u,d,uu,ud or dd expected'%sector)

class my_omega:
    
    def __init__(self):
        self.one_u = None
        self.one_d = None
        self.three_u = None
        self.three_d = None
        self.two_uu = None
        self.two_dd = None
        self.two_ud = None
        
    def set_dim_1u(self, n_psi, n):
        self.one_u = np.zeros((n_psi,n))
        
    def set_dim_1d(self, n_psi, n):
        self.one_d = np.zeros((n_psi,n))
        
    def set_dim_2uu(self, n_psi, n):
        self.two_uu = np.zeros((n_psi,n,n))
        
    def set_dim_2dd(self, n_psi, n):
        self.two_dd = np.zeros((n_psi,n,n))
        
    def set_dim_2ud(self, n_psi, n):
        self.two_ud = np.zeros((n_psi,n,n))
        
    def set_dim_3u(self, n_psi, n):
        self.three_u =  np.zeros((n_psi,n,n,n))
        
    def set_dim_3d(self, n_psi, n):
        self.three_d =  np.zeros((n_psi,n,n,n))  
        
    def init_tensor(self,k,sector,n_psi,n):
        if(k==1):
           if(sector=='u'):   self.set_dim_1u(n_psi, n)
           elif(sector=='d'): self.set_dim_1d(n_psi, n)
           else:              raise NameError('%s is not a valid sector: u or d expected'%sector)
        elif(k==2):
           if(sector=='uu'):   self.set_dim_2uu(n_psi, n)
           elif(sector=='ud'): self.set_dim_2ud(n_psi, n)
           elif(sector=='dd'): self.set_dim_2dd(n_psi, n)
           else:               raise NameError('%s is not a valid sector: uu, ud or dd expected'%sector)
        elif(k==3):
           if(sector=='u'):   self.set_dim_3u(n_psi, n)
           elif(sector=='d'): self.set_dim_3d(n_psi, n)
           else:              raise NameError('%s is not a valid sector: u or d expected'%sector)
        else:
           raise ValueError('%i is not a valid number: 1,2 or 3 expected'%k)
        
    def set_tensor(self,k,sector,tensor):
        if(k==1):
           if(sector=='u'):   self.one_u = tensor
           elif(sector=='d'): self.one_d = tensor
           else:              raise NameError('%s is not a valid sector: u or d expected'%sector)
        elif(k==2):
           if(sector=='uu'):   self.two_uu = tensor
           elif(sector=='ud'): self.two_ud = tensor
           elif(sector=='dd'): self.two_dd = tensor
           else:               raise NameError('%s is not a valid sector: uu, ud or dd expected'%sector)
        elif(k==3):
           if(sector=='u'):   self.three_u = tensor
           elif(sector=='d'): self.three_d = tensor
           else:              raise NameError('%s is not a valid sector: u or d expected'%sector)
        else:
           raise ValueError('%i is not a valid number: 1,2 or 3 expected'%k)        

    def set_element(self,k,sector,idx,val):
        if(k==1):
           if(sector=='u'):   self.one_u[idx] = val
           elif(sector=='d'): self.one_d[idx] = val
           else:              raise NameError('%s is not a valid sector: u or d expected'%sector)
        elif(k==2):
           if(sector=='uu'):   self.two_uu[idx] = val
           elif(sector=='ud'): self.two_ud[idx] = val
           elif(sector=='dd'): self.two_dd[idx] = val
           else:               raise NameError('%s is not a valid sector: uu, ud or dd expected'%sector)
        elif(k==3):
           if(sector=='u'):   self.three_u[idx] = val
           elif(sector=='d'): self.three_d[idx] = val
           else:              raise NameError('%s is not a valid sector: u or d expected'%sector)
        else:
           raise ValueError('%i is not a valid number: 1,2 or 3 expected'%k)

    def get_tensor(self,k,sector):
        if(k==1):
           if(sector=='u'):   return self.one_u
           elif(sector=='d'): return self.one_d
           else:              raise NameError('%s is not a valid sector: u or d expected'%sector)
        elif(k==2):
           if(sector=='uu'):   return self.two_uu
           elif(sector=='ud'): return self.two_ud
           elif(sector=='dd'): return self.two_dd
           else:               raise NameError('%s is not a valid sector: uu, ud or dd expected'%sector)
        elif(k==3):
           if(sector=='u'):   return self.three_u
           elif(sector=='d'): return self.three_d
           else:              raise NameError('%s is not a valid sector: u or d expected'%sector)
        else:
           raise ValueError('%i is not a valid number: 1,2 or 3 expected'%k)
    
