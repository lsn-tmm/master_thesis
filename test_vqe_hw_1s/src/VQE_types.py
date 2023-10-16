class vqe_data:
    
    def __init__(self, target_sector = None, optimizer = 'bfgs', max_iter = 1000, instance = 'statevector_simulator', 
                       shots = 1000, ansatz = 'q_uccsd', initial_point = None):
        
        self.target_sector = target_sector
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.instance = instance
        self.shots = shots
        self.ansatz = ansatz
        self.initial_point = initial_point
        self.tapering_info = None
        self.reps = 0
        
        self.print()
        
                
    def set_target_sector(self, target_sector):
        self.target_sector = target_sector
        
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        
    def set_max_iter(self, max_iter):
        self.max_iter = max_iter
        
    def set_instance(self, instance):
        self.instance = instance
        
    def set_shots(self, shots):
        self.shots = shots
        
    def set_ansatz(self, ansatz):
        self.ansatz = ansatz
        
    def set_initial_point(self, initial_point):
        self.initial_point = initial_point
        
    def set_tapering_info(self, tapering_info):
        self.tapering_info = tapering_info  
    
    def print(self):
        
        print('----- VQE data ------')
        print('target_sector: ', self.target_sector)
        print('optimizer: ', self.optimizer)
        print('max_iter: ', self.max_iter)
        print('instance: ', self.instance)
        print('shots: ', self.shots)
        print('ansatz: ', self.ansatz)
        print('initial_point: ', self.initial_point)
        print('\n')
