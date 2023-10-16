import numpy as np

class my_data:
    
    def __init__(self, overlap = None, coefficients = None, energy = None):
        self.overlap = overlap
        self.coefficients = coefficients
        self.energy = energy
        
        
    def get_overlap(self):
        return self.overlap
    
    def get_coefficients(self):
        return self.coefficients
        
    def get_energy(self):
        return self.energy
    
    def print(self):
        print(self.overlap, self.coefficients, self.energy)
        

class excited_states:
    
    def __init__(self):
        self.u = my_data()
        self.d = my_data()
        self.uu = my_data()
        self.dd = my_data()
        self.ud = my_data()
        
        
    def set_u(self, overlap, coefficients, energy):
        self.u = my_data(overlap, coefficients, energy)
    
    def set_d(self, overlap, coefficients, energy):
        self.d = my_data(overlap, coefficients, energy)
        
    def set_uu(self, overlap, coefficients, energy):
        self.uu = my_data(overlap, coefficients, energy)
    
    def set_dd(self, overlap, coefficients, energy):
        self.dd = my_data(overlap, coefficients, energy)
    
    def set_ud(self, overlap, coefficients, energy):
        self.ud = my_data(overlap, coefficients, energy)

    def set_excited_states(self, sector, overlap, coefficients, energy):
        if(sector=='u'):    self.set_u(overlap, coefficients, energy)
        elif(sector=='d'):  self.set_d(overlap, coefficients, energy)
        elif(sector=='uu'): self.set_uu(overlap, coefficients, energy)
        elif(sector=='ud'): self.set_ud(overlap, coefficients, energy)
        elif(sector=='dd'): self.set_dd(overlap, coefficients, energy)
        else:               raise NameError('%s is not a valid sector: u,d,uu,ud or dd expected'%sector)
            
    def get_excited_states_c(self, sector):
        if(sector=='u'):    return self.u.get_coefficients()
        elif(sector=='d'):  return self.d.get_coefficients()
        elif(sector=='uu'): return self.uu.get_coefficients()
        elif(sector=='ud'): return self.ud.get_coefficients()
        elif(sector=='dd'): return self.dd.get_coefficients()
        else:               raise NameError('%s is not a valid sector: u,d,uu,ud or dd expected'%sector)
            
            
    def get_excited_state_en(self, sector):
        if(sector=='u'):    return self.u.get_energy()
        elif(sector=='d'):  return self.d.get_energy()
        elif(sector=='uu'): return self.uu.get_energy()
        elif(sector=='ud'): return self.ud.get_energy()
        elif(sector=='dd'): return self.dd.get_energy()
        else:               raise NameError('%s is not a valid sector: u,d,uu,ud or dd expected'%sector)
