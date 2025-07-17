import numpy as np
class EEGPackage():
    '''
    Define package for different experiments
    '''
    def __init__(self, eeg=None, events=None, numOfTrial=1, numOfStimuli=0, name=None, path=None):
        self.numOfTrial = numOfTrial
        self.numOfStimuli = numOfStimuli
        self.name = name
        self.eeg = eeg
        self.events = events
        self.path = path
        self.trials = []
        self.valid = True
        
    def toTrials(self):
        self.trials = []
        if self.name == None:
            return
            
        
            
