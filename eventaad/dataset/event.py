import sys
import json
from eventaad.utils import *

def correct_events(s_events, d_events, epsilon = 1):
    sys.setrecursionlimit(3100)
    s_len = len(s_events)
    d_len = len(d_events)
    before_list = d_events.copy()
    print(s_len)
    print(d_len)
    s_samples = []
    d_samples = []
    s_codes = []
    d_codes = []
    for i in range(s_len):
        s_codes.append(s_events[i].code)
        s_samples.append(s_events[i].toNextEvent)

    for i in range(d_len):
        d_codes.append(d_events[i].code)
        d_samples.append(d_events[i].toNextEvent)

    print(s_samples[490:501])
    print(d_samples[489:499])
    sample_diff = find_errors(s_samples[490:501], d_samples[489:499], epsilon=epsilon)
    print(sample_diff)
    
    for i in range(len(sample_diff)):
        if sample_diff[i] == -1:
            toNextEvent = 0
            code = -1
            event = Event(code, toNextEvent)
            d_events.insert(i, event)
            d_samples.insert(i, toNextEvent)        
            print('inserted')
    
    i = 0
    while i < len(sample_diff):
        if sample_diff[i] != 0:
            j = i
            s_sum = 0
            d_sum = 0
            while sample_diff[j]!=0:
                s_sum = s_sum + s_samples[j]
                d_sum = d_sum + d_samples[j]
                d_events[j].code = s_events[j].code
                d_events[j].toNextEvent = s_events[j].toNextEvent
                print('j= ',j)
                j+=1
            diff = d_sum-s_sum
            print('diff= ', diff)
            d_events[j-1].toNextEvent = d_events[j-1].toNextEvent+diff
            if abs(diff) > epsilon:
                for m in range(i, j):
                    print('m= ',m)
                    d_events[m].code = -1 #not be used
            i = j
        else:
            i+=1
    
    d_len = len(d_events)
    print(d_len)
    print(len(sample_diff))
        
    for i in range(d_len):
        d_codes.append(d_events[i].code)
        d_samples.append(d_events[i].toNextEvent)
    
    return sample_diff

class Event:
    '''
    Structure for storing event information for the experiment
    '''
    def __init__(self, code, sampleth=0, toNextEvent=0, stimuli_idx=0):
        self.code = code
        self.sampleth = sampleth
        self.toNextEvent = toNextEvent
        self.stimuli_idx = stimuli_idx

