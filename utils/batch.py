import numpy as np


class Batch(object):
    def __init__(self,num_instances,batch_size,shuffle=True):
        # parameters
        self.num_instances   = num_instances
        self.batch_size      = batch_size
        self.shuffle         = shuffle
        self.start           = 0
        self.epoch_completed = False
        
        # initialization
        self.indices = np.arange(0,num_instances)
        self.initialize_epoch()
            
    def initialize_epoch(self):
        self.initialize_next_epoch()
            
    # initialize for next epoch for the same training (get new shuffle if needed)
    def initialize_next_epoch(self):
        self.epoch_completed = False
        self.start = 0
        if self.shuffle == True:
            np.random.shuffle(self.indices)    
            
    # get next batch indices
    def get_next_batch_indices(self):
        start = self.start   
        batch_size = self.batch_size
        if start + batch_size < (self.num_instances-1):
            end = start + batch_size
            self.start = end
        else:            
            end = self.num_instances
            self.epoch_completed = True
        return self.indices[start:end]
    
    def has_next_batch(self):
        return self.epoch_completed == False
