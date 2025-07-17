import os
import json
from datetime import datetime, timezone
import pytz
import numpy as np
import pandas as pd
import json
from numpy.core.records import fromarrays
from numpy.lib.recfunctions import append_fields
from scipy.io import savemat, loadmat
from mne_bids import read_raw_bids, write_raw_bids, BIDSPath, print_dir_tree
from mne_bids.stats import count_events
from mne.io import Raw, read_raw_eeglab, read_epochs_eeglab
from mne.channels import read_custom_montage

from .event import *
from .EEGPackage import *
from SerialTriggerDecoder import *

RANDOM_SENTENCE_PATCH = 1

def write_meta_bids(part, bids_path, events):
    participants_tsv_fname = os.path.join(bids_path.root, 'participants.tsv')
    participants_json_fname = participants_tsv_fname.replace('.tsv', '.json')
    sidecar_path = str(bids_path.fpath).replace('.set', '.json')
    events_path = (str(bids_path.fpath)).replace('eeg.set', 'events.tsv')
    events_stimuli_path = events_path.replace('.tsv', '.json')
    channels_path = str(bids_path.fpath).replace('eeg.set', 'channels.tsv')
    
    events_df = pd.read_csv(events_path, sep='\t')
    if len(events) > 0 and all(k in events.dtype.names for k in ('attended_spk', 'attended_stimulus', 'attended_event','stimuli_idx','epoch')):
        events_df['attended_spk'] = events['attended_spk']
        events_df['attended_stimulus'] = events['attended_stimulus']
        events_df['attended_event'] = events['attended_event']
        events_df['stimuli_idx'] = events['stimuli_idx']
        events_df['epoch'] = events['epoch']
        events_df['duration'] = len(events_df['duration'])*[0.4]        
        #fix bug of wrong encoded events in Kristian pilot recording
        '''
        if part.name == 'multiT~spoken~count':
            events_df.loc[(events_df['value'] == 2) & (events_df['attended_event'] == 5), 'trial_type'] = 'color'
            events_df.loc[(events_df['value'] == 2) & (events_df['attended_event'] == 5), 'value'] = 5
        if part.name == 'discrete~discrete~count':
            events_df.loc[(events_df['value'] == 2) & (events_df['stimuli_idx'] == 1), 'trial_type'] = 'color'
            events_df.loc[(events_df['value'] == 2) & (events_df['stimuli_idx'] == 1), 'value'] = 5
        '''
        events_df.to_csv(events_path, sep='\t', index=False)    
        with open(events_stimuli_path, "w") as jsonFile:
            json.dump(part.trials, jsonFile, separators=(',\n', ': '))
            jsonFile.close()

def read_meta_bids():
    return None

def makeEEGLABEvents(part):
    latency = []
    duration = []
    type_ = []
    position = []
    attended_event = []
    attended_stimulus = []
    attended_spk = []
    stimuli_idx=[]
    epoch=[]
    trial = -1

    #for e, e2 in zip(part.events[0], part.events[1]):
    for e in part.events:
        if(e.code==sti.TRIAL_START_CODE):
            trial+=1
        latency.append(e.sampleth)
        type_.append(e.code)
        if len(part.trials)>0:
            duration.append(400)        
            position.append(e.stimuli_idx)
            if 'attended_event' in part.trials[trial].keys():
                attended_event.append(part.trials[trial]['attended_event'])
            else:
                attended_event.append(sti.TARGET)
                
            if 'attended_stimulus' in part.trials[trial].keys():
                attended_stimulus.append(part.trials[trial]['attended_stimulus'])
            else:
                attended_stimulus.append(0)
                
            if 'attended_spk' in part.trials[trial].keys():
                attended_spk.append(part.trials[trial]['attended_spk'])
            else:
                attended_spk.append(0)                
            stimuli_idx.append(e.stimuli_idx)
            epoch.append(trial)
            
        '''
        latency.append(e2.sampleth)
        duration.append(e2.toNextEvent)
        type_.append(e2.code)
        position.append(e2.stimuli_idx)
        attended_event.append(part.trials[trial]['attended_event'])
        stimuli_idx.append(e2.stimuli_idx)
        epoch.append(trial)
        '''    
    
    if len(attended_event) > 0:
        events = fromarrays([latency, duration, type_, attended_event, stimuli_idx, epoch], names=['latency','duration','type', 'attended_event', 'stimuli_idx', 'epoch'])
    else:
        events = fromarrays([latency, type_], names=['latency','type'])

    if len(attended_stimulus) > 0:
        events = append_fields(events, 'attended_stimulus', attended_stimulus, dtypes=np.int16, usemask=False, asrecarray=True)  
    if len(attended_spk) > 0:
        events = append_fields(events, 'attended_spk', attended_spk, dtypes=np.int16, usemask=False, asrecarray=True)          
    
    events.sort(order='latency')
    return events

def makeBIDSEvents(part):
    events = []
    for e in part.events:
        events.append([e.sampleth, 0, e.code])

    return events
    
def makeBIDSEvents(eeglab_events):
    type_ = eeglab_events['type']
    latency = eeglab_events['latency']
    events = []
    for i in range(len(eeglab_events)):
        events.append([latency[i], 0, type_[i]])

    return events    


def makeEEGLABEpochs(part, events):
    '''
    i_event = part.numOfTrial*[[]]
    e_eventlatency = part.numOfTrial*[[]]
    e_eventposition = part.numOfTrial*[[]]
    e_eventtype = part.numOfTrial*[[]]
    for j in range(events.size):
        epch = events[j]['epoch']
        e_event[epch].append(j)
        e_eventlatency[epch].append(events[j]['latency'])
        e_eventposition[epch].append(events[j]['position'])
        e_eventtype[epch].append(events[i]['type'])
    epochs = fromarrays([e_event, e_eventlatency, e_eventposition, e_eventtype], names=['event','eventlatency','eventposition', 'eventtype'])

    #epochs = fromarrays([list(np.array(e_event)), list(np.array(e_eventlatency)), list(np.array(e_eventposition)), list(np.array(e_eventtype))], names=['event','eventlatency','eventposition', 'eventtype'])
    '''
    
    part.eeg = part.eeg.reshape(part.eeg.shape[0], -1, part.numOfTrial)
    e_event = [np.array([[]], dtype=object) for i in range(part.numOfTrial)]
    e_eventlatency = [np.array([[]], dtype=object) for i in range(part.numOfTrial)]
    e_eventposition = [np.array([[]], dtype=object) for i in range(part.numOfTrial)]
    e_eventtype = [np.array([[]], dtype=object) for i in range(part.numOfTrial)]
    for j in range(events.size):
        epch = events[j]['epoch']
        e_event[epch] = np.append(e_event[epch], np.array([[j]]), axis=1)
        e_eventlatency[epch] = np.append(e_eventlatency[epch], np.array([[events[j]['latency']]]), axis=1)
        e_eventlatency[epch][0,-1] = np.array([[events[j]['latency']]])
        e_eventposition[epch] = np.append(e_eventposition[epch], np.array([[events[j]['position']]]), axis=1)
        e_eventposition[epch][0,-1] = np.array([[events[j]['position']]])
        e_eventtype[epch] = np.append(e_eventtype[epch], np.array([[events[j]['type']]]), axis=1)
        e_eventtype[epch][0,-1] = np.array([[events[j]['type']]])

    return fromarrays([e_event, e_eventlatency, e_eventposition, e_eventtype], names=['event','eventlatency','eventposition', 'eventtype'])


def synchronize(eeg1, eeg2):
    eeg = EEGData()
    eeg.start_time = min((eeg1.start_time, eeg2.start_time))
    if eeg1.sample_rate != eeg2.sample_rate:
        raise RuntimeError('Sampling rate is not matched!')
    eeg.sample_rate = eeg1.sample_rate
    eeg.num_channels = eeg1.num_channels
    eeg.channels = eeg1.channels[0:32].copy()
    eeg.channels = eeg.channels + eeg2.channels[0:14].copy()
    eeg.numOfExp = eeg1.numOfExp
    eeg.parts = eeg1.parts.copy()
    for i in range(eeg.numOfExp):
        events = []
        for e1,e2 in zip(eeg1.parts[i].events, eeg2.parts[i].events):
            if (e1.code != e2.code) or e1.code == -1 or e2.code == -1:
                e1.code = -1
            events.append(e1)
        eeg.parts[i].events = events
        '''
        eeg.parts[i].events = [eeg.parts[i].events]
        eeg.parts[i].events.append(eeg2.parts[i].events)
        '''
        scalp = np.delete(eeg.parts[i].eeg, list(range(32,eeg.num_channels)), axis=0) #scalp
        ear = np.delete(eeg2.parts[i].eeg, list(range(14,eeg2.num_channels)), axis=0) #scalp
        eeg.parts[i].eeg = np.concatenate((scalp, ear), axis=0)

    spon_scalp = np.zeros(shape=(32,0))
    spon_ear = np.zeros(shape=(14,0))
    events = []
    for i in range(eeg.numOfExp, len(eeg.parts)): #spontaneous part
        scalp = np.delete(eeg.parts[i].eeg, list(range(32,eeg.num_channels)), axis=0) #scalp
        ear = np.delete(eeg2.parts[i].eeg, list(range(14,eeg2.num_channels)), axis=0) #scalp
        sampleth = spon_scalp.shape[1]
        toNextEvent = scalp.shape[1]
        spon_scalp =  np.concatenate((spon_scalp, scalp), axis=1)
        spon_ear =  np.concatenate((spon_ear, ear), axis=1) 
        events.append(Event(code=0, sampleth=sampleth, toNextEvent=toNextEvent))
    min_len = min(spon_scalp.shape[1], spon_ear.shape[1])
    spon_scalp = np.delete(spon_scalp, list(range(min_len, spon_scalp.shape[1])), axis=1)
    spon_ear = np.delete(spon_ear, list(range(min_len, spon_ear.shape[1])), axis=1)
    eeg.parts[eeg.numOfExp].eeg = np.concatenate((spon_scalp, spon_ear), axis=0)
    eeg.parts[eeg.numOfExp].events = events
    del eeg.parts[eeg.numOfExp+1:len(eeg.parts)]
    eeg.num_channels = 46
    '''
    with open('channels', 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        eeg.channels[i].name = lines[i].strip()
    '''
    for i in range(len(Channel.AAD_channels)):
        eeg.channels[i].name = Channel.AAD_channels[i]
    for i in range(len(Channel.AAD_channel_type)):
        eeg.channels[i].ch_type = Channel.AAD_channel_type[i]
    return eeg

class EEGData:
    LOCS_PATH = 'eventaad_chan32.locs'
    def __init__(self):
        self.name = None
        self.start_time = None
        self.sample_rate = None
        self.num_samples = 0
        self.samples = None        
        self.num_channels = 0
        self.channels = []
        self.raw_events = None
        self.exp_names = []
        self.parts = []

    def addChannel(self, name, unit_name, channel_type='EEG'):
        ch = Channel(name, unit_name, channel_type)
        self.channels.append(ch)

    def toJSON(self):
        return json.dumps(self.__dict__)
        
    def decode_events(self, triggerClk=8, thrError=-0.3, transError=0.1):
        self.trigger = (self.samples[35, :]==0).astype(int)
        decoder = SerialTriggerDecoder(self.trigger, self.sample_rate, triggerClk, thrError, transError)
        self.raw_events = decoder.decode()
        print('number of events: ', len(self.raw_events))
        #fix trigger bug s009
        #self.raw_events[684]['pattern'] = '1000'
        #self.raw_events[684]['code'] = sti.TRIAL_START_CODE
        
    #for bug fix of Yousef2 recording
    def decode_events_2(self, triggerClk=8, thrError=-0.3, transError=0.1):
        self.trigger = (self.samples[35, :]==0).astype(int)
        decoder1 = SerialTriggerDecoder(self.trigger[0:2000000], self.sample_rate, triggerClk, thrError, transError)
        events1 = decoder1.decode_2()
        decoder2 = SerialTriggerDecoder(self.trigger[2000000:], self.sample_rate, triggerClk, thrError, transError)
        events2 = decoder2.decode()
        
        for i in range(len(events2)):
            events2[i]['sample_idx'] = events2[i]['sample_idx'] + 2000000
        self.raw_events = events1 + events2
        

    def __getEventsFromFile(self, filepath):
        events = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                #numbers = [int(num) for num in line.split()]
                numbers = []
                for i in range(min(4,len(line.split()))):
                    numbers.append(int(line.split()[i]))
                '''
                if(len(numbers) == 4):
                    event = Event(code=numbers[0], sampleth=int(numbers[3]*self.sample_rate/1000), stimuli_idx=numbers[2])
                else:
                    event = Event(code=numbers[0], sampleth=int(numbers[1]*self.sample_rate/1000), stimuli_idx=numbers[2])
                '''
                event = Event(code=numbers[0], sampleth=int(numbers[1]*self.sample_rate/1000), stimuli_idx=numbers[2])    
                events.append(event)
            f.close()
        for i in range(len(events)-1):
            events[i].toNextEvent = events[i+1].sampleth - events[i].sampleth
        events[len(events)-1].toNextEvent = self.sample_rate
        return events

    def __correctEvents(self, eventFiles, events, start_idx, end_idx, eeg, epsilon, attended_stimulus=0):
        if isinstance(eventFiles, list):
            s_events = self.__getEventsFromFile(eventFiles[attended_stimulus])
        elif isinstance(eventFiles, str):
            s_events = self.__getEventsFromFile(eventFiles)
            
        for i in range(start_idx,end_idx+1):
            events[i].stimuli_idx = attended_stimulus
        
        d_events = events[start_idx:end_idx+1]
        s_len = len(s_events)
        d_len = len(d_events)
        s_positions = []
        d_positions = []
        s_codes = []
        d_codes = []
        s_distances = []
        d_distances = []
        for i in range(s_len):
            s_events[i].stimuli_idx = attended_stimulus
            s_codes.append(s_events[i].code)
            s_positions.append(s_events[i].sampleth)
            s_distances.append(s_events[i].toNextEvent)

        for i in range(d_len):      
            d_codes.append(d_events[i].code)
            d_positions.append(d_events[i].sampleth)
            d_distances.append(d_events[i].toNextEvent)

        dist_diff = find_errors(s_distances, d_distances, epsilon=0)
        code_diff = find_errors(s_codes, d_codes, epsilon=0)
        
        for i in range(len(dist_diff)):
            #if (dist_diff[i] == -1) and (code_diff[i] == -1):
            if (code_diff[i] == -1):
                toNextEvent = 0
                code = -1
                event = Event(code=code, sampleth=d_events[i+1].sampleth, toNextEvent=toNextEvent, stimuli_idx=attended_stimulus)
                d_events.insert(i, event)
                events.insert(start_idx + i, event)
                d_distances.insert(i, toNextEvent)
                d_codes.insert(i, code)
                print('inserted 1 sample')
        
        i = 0
        while i < len(dist_diff):
            if dist_diff[i] != 0:
                j = i
                s_sum = 0
                d_sum = 0
                while j < s_len and dist_diff[j]!=0 and d_events[j].code != sti.TRIAL_END_CODE:
                    s_sum = s_sum + s_distances[j]
                    d_sum = d_sum + d_distances[j]
                    d_events[j].code = s_events[j].code
                    d_events[j].toNextEvent = s_events[j].toNextEvent
                    if j>0:
                        d_events[j].sampleth = d_events[j-1].sampleth + d_events[j-1].toNextEvent
                    #d_distances[j] = d_events[j].toNextEvent
                    #d_codes[j] = d_events[j].code
                    j+=1
                diff = d_sum-s_sum
                d_events[j-1].toNextEvent = d_events[j-1].toNextEvent+diff
                if abs(diff) > epsilon:
                    for m in range(i, j):
                        if d_events[m].code != sti.TRIAL_START_CODE and d_events[m].code != sti.TRIAL_END_CODE:
                            d_events[m].code = -1 #not be used
                if d_events[j].code == sti.TRIAL_END_CODE:
                    j+=1
                i = j
            else:
                i+=1

        for i in range(s_len):
            d = d_events[i]
            s = s_events[i]
            d.stimuli_idx = s.stimuli_idx
            diff = d.toNextEvent - s.toNextEvent
            if diff > 0: #remove diff samples from d_events
                psts = [p for p in range(d.sampleth+d.toNextEvent-diff, d.sampleth+d.toNextEvent)]
                eeg = np.delete(eeg, psts, axis=1)
            elif diff < 0: #insert diff samples to d_events
                j = d.sampleth+d.toNextEvent-abs(diff)
                while (j < d.sampleth+s.toNextEvent):
                    if (j+1) < eeg.shape[1]:
                        inserted_samples = (eeg[:,j] + eeg[:,j+1])/2
                    else:
                        inserted_samples = eeg[:,j]
                    eeg = np.insert(eeg, j, inserted_samples, axis=1)
                    j+=2
            d.toNextEvent-= diff
            if(diff != 0):
                for j in range(start_idx + i+1, len(events)):
                    events[j].sampleth -= diff
            if d.code == sti.TRIAL_END_CODE:
                eeg = np.delete(eeg, [p for p in range(d.sampleth, d.sampleth + d.toNextEvent-self.sample_rate)], axis=1)
                d.toNextEvent = self.sample_rate
                for j in range(start_idx + i+1, len(events)):
                    events[j].sampleth -= d.toNextEvent-self.sample_rate
        
        #to-do: insert events from 2nd stream. take care of stimuli_idx and attended stream
        if isinstance(eventFiles, list):
            for i in range(len(eventFiles)):
                if i == attended_stimulus:
                    continue
                s_events = self.__getEventsFromFile(eventFiles[i])
                for j in range(1,len(s_events)-1):
                    s_events[j].stimuli_idx = i
                    s_events[j].sampleth += events[start_idx].sampleth
                    s_events[j].toNextEvent = 0
                    events.insert(start_idx + j, s_events[j])
        return eeg

    def __correctEvents_random_sentence(self, eventFiles, events, start_idx, end_idx, eeg, epsilon, attended_stimulus=0):     
        d_events = events[start_idx:end_idx+1]
        s_diff = s_events[-1].sampleth - s_events[0].sampleth
        d_diff = d_events[-1].sampleth - d_events[0].sampleth
        diff = s_diff-d_diff
        total_diff = diff + s_events[-1].toNextEvent - d_events[-1].toNextEvent
        
        events[start_idx].toNextEvent = s_events[0].toNextEvent
        del events[end_idx]
        for i in range(1,len(s_events)):
            e = s_events[i]
            e.sampleth = events[start_idx+i-1].sampleth + events[start_idx+i-1].toNextEvent
            if (np.abs(diff) > 50 and e.code != sti.TRIAL_START_CODE and e.code != sti.TRIAL_END_CODE):
                e.code = -1
            events.insert(start_idx+i, e)
        
        if (total_diff < 0):
            print(f'deleted {-total_diff} samples')
            psts = [p for p in range(d_events[-1].sampleth+d_events[-1].toNextEvent+total_diff, d_events[-1].sampleth+d_events[-1].toNextEvent)]
            eeg = np.delete(eeg, psts, axis=1)
        elif total_diff > 0:
            print(f'inserted {total_diff} samples')
            inserted_samples = eeg[:,d_events[-1].sampleth-total_diff:d_events[-1].sampleth]
            for i in range(total_diff):
                eeg = np.insert(eeg, d_events[-1].sampleth + i, inserted_samples[:,i], axis=1)
        #correct the sampleth of following events after delete/insert
        for j in range(start_idx + len(s_events), len(events)):
            events[j].sampleth += total_diff
        
        return eeg

    def __correctEvents_story(self, eventFiles, events, start_idx, end_idx, eeg, epsilon, attended_stimulus=0):
        if isinstance(eventFiles, list):
            s_events = self.__getEventsFromFile(eventFiles[attended_stimulus])
        elif isinstance(eventFiles, str):
            s_events = self.__getEventsFromFile(eventFiles)
    
        d_events = events[start_idx:end_idx+1]
        s_diff = s_events[-1].sampleth - s_events[0].sampleth
        d_diff = d_events[-1].sampleth - d_events[0].sampleth
        diff = s_diff-d_diff
        total_diff = diff + s_events[-1].toNextEvent - d_events[-1].toNextEvent
        
        events[start_idx].toNextEvent = s_events[0].toNextEvent
        del events[end_idx]
        for i in range(1,len(s_events)):
            e = s_events[i]
            e.sampleth = events[start_idx+i-1].sampleth + events[start_idx+i-1].toNextEvent
            e.stimuli_idx = attended_stimulus
            if (np.abs(diff) > 50 and e.code != sti.TRIAL_START_CODE and e.code != sti.TRIAL_END_CODE):
                e.code = -1
            events.insert(start_idx+i, e)
        
        if (total_diff < 0):
            print(f'deleted {-total_diff} samples')
            psts = [p for p in range(d_events[-1].sampleth+d_events[-1].toNextEvent+total_diff, d_events[-1].sampleth+d_events[-1].toNextEvent)]
            eeg = np.delete(eeg, psts, axis=1)
        elif total_diff > 0:
            print(f'inserted {total_diff} samples')
            inserted_samples = eeg[:,d_events[-1].sampleth-total_diff:d_events[-1].sampleth]
            for i in range(total_diff):
                eeg = np.insert(eeg, d_events[-1].sampleth + i, inserted_samples[:,i], axis=1)
        #correct the sampleth of following events after delete/insert
        for j in range(start_idx + len(s_events), len(events)):
            events[j].sampleth += total_diff
            
        if isinstance(eventFiles, list):
            for i in range(len(eventFiles)):
                if i == attended_stimulus:
                    continue
                s_events = self.__getEventsFromFile(eventFiles[i])
                for j in range(1, len(s_events)-1):
                    s_events[j].stimuli_idx = i
                    s_events[j].sampleth += events[start_idx].sampleth
                    s_events[j].toNextEvent = 0
                    events.insert(start_idx + j, s_events[j])
        return eeg

        
    def parseExperiments(self, exp_path):
        exp_data_path = os.path.join(exp_path, 'experiments.txt')
        assert os.path.isfile(exp_data_path), "Experiments' data file does not exist."
        exp_paths = []
        with open(exp_data_path) as f:
            lines = f.readlines()
            for line in lines:
                self.exp_names.append(line.strip())
        #parse events
        self.numOfExp = len(self.exp_names)
        for i in range(self.numOfExp):
            path = os.path.join(exp_path, self.exp_names[i])
            pkg = EEGPackage(name=self.exp_names[i], path=path)
            jsonPath = os.path.join(path, 'experiment_stimuli.json')
            if not os.path.isfile(jsonPath):
                pkg.valid = False
                self.parts.append(pkg)
                print(f'Invalid: {jsonPath}')
                continue
            expJSONfile = open(os.path.join(path, 'experiment_stimuli.json'))
            expJSON = json.load(expJSONfile)
            pkg.numOfTrial = expJSON['trial_number']
            pkg.trials = expJSON['trials']
            self.parts.append(pkg)
        
        #fixing bug
        tmp_events = self.raw_events.copy()
        #for i in range(1,len(self.raw_events)-1):
        i = 1
        while i<len(tmp_events)-1:
            code_b1 = tmp_events[i-1]['code']
            code = tmp_events[i]['code']
            code_a1 = tmp_events[i+1]['code']
            sample_idx = tmp_events[i+1]['sample_idx']
            if ((code == sti.PART_START_CODE) and (code_b1 != sti.PART_END_CODE)):
                e = {'sample_idx': tmp_events[i-1]['sample_idx'] + int(self.sample_rate*2.0), 'pattern': '1111', 'code': sti.PART_END_CODE}
                tmp_events.insert(i, e)
                if (code_b1 != sti.TRIAL_END_CODE):
                    e = {'sample_idx': tmp_events[i-1]['sample_idx'] + int(self.sample_rate*1.0), 'pattern': '1111', 'code': sti.TRIAL_END_CODE}
                    tmp_events.insert(i, e)
            if ((code == sti.TRIAL_END_CODE) and (code_a1 != sti.TRIAL_START_CODE) and (code_a1 != sti.PART_END_CODE) and (code_a1 != sti.PART_START_CODE)):
                e = {'sample_idx': tmp_events[i+1]['sample_idx'] - int(self.sample_rate*1.0), 'pattern': '1000', 'code': sti.TRIAL_START_CODE}
                tmp_events.insert(i+1, e)
            
            i+=1
            
        self.raw_events = tmp_events.copy()
        
        start_e = self.raw_events[0]
        if start_e['code'] != sti.PART_START_CODE:
            e = {'sample_idx': start_e['sample_idx'] - int(self.sample_rate*1.0), 'pattern': '1110', 'code': sti.PART_START_CODE}
            self.raw_events.insert(0, e)
            
        end_e = self.raw_events[-1]
        if end_e['code'] != sti.PART_END_CODE:
            e = {'sample_idx': end_e['sample_idx'] + int(self.sample_rate*1.0), 'pattern': '1111', 'code': sti.PART_END_CODE}
            self.raw_events.append(e)
        print(f'raw events (after corrected): {len(self.raw_events)}')
        #split eeg
        start_idx = 0
        end_idx = 0
        events = []
        pkg_idx = 0
        for i in range(len(self.raw_events)):
            e = self.raw_events[i]
            if e['code'] == sti.PART_START_CODE:
                start_idx = i+1
                #store spontaneous eeg
                if i==0:
                    eeg = self.samples[:,0:self.raw_events[start_idx]['sample_idx']]
                else:
                    eeg = self.samples[:,self.raw_events[end_idx]['sample_idx']:self.raw_events[start_idx]['sample_idx']]
                pkg = EEGPackage(name='spontaneous', eeg=eeg) #spontaneous package
                self.parts.append(pkg)
            elif e['code'] == sti.PART_END_CODE:
                end_idx = i
                eeg = self.samples[:,self.raw_events[start_idx]['sample_idx']:self.raw_events[end_idx]['sample_idx']]
                self.parts[pkg_idx].eeg = np.copy(eeg)
                self.parts[pkg_idx].events = events[start_idx:end_idx]
                pkg_idx+=1

            if i == (len(self.raw_events) - 1):
                event = Event(code=e['code'], sampleth=e['sample_idx']-self.raw_events[start_idx]['sample_idx'], toNextEvent=self.sample_rate) #1 second more
            else:
                event = Event(code=e['code'], sampleth=e['sample_idx']-self.raw_events[start_idx]['sample_idx'], toNextEvent=self.raw_events[i+1]['sample_idx'] - e['sample_idx'])
            events.append(event)
            
        #parts refining
        for i in range(self.numOfExp):
            exp = self.parts[i]
            if exp.valid:
                trials = exp.trials
                events = exp.events
                trial_idx = 0
                start_idx = 0
                end_idx = -1
                has_started = False
                j = 0
                while j < len(events):
                    e = events[j]
                    if e.code == sti.TRIAL_START_CODE:
                        start_idx = j
                        has_started = True
                    if (e.code == sti.TRIAL_END_CODE and has_started):
                        #originalEvents = self.__getEventsFromFile(trials[trial_idx]["trigger_log"])
                        end_idx = j
                        if 'attended_stimulus' in trials[trial_idx].keys():
                            attended_stimulus = trials[trial_idx]['attended_stimulus']
                        else:
                            attended_stimulus = 0
                        if RANDOM_SENTENCE_PATCH:
                            print(f'{exp.name} - trial_idx: {trial_idx} events: {start_idx} - {end_idx} attended_stimulus: {attended_stimulus}')
                            if exp.name == 'singleT~randomsentence~mem':
                                exp.eeg = self.__correctEvents_random_sentence(trials[trial_idx]["trigger_log"], events,start_idx, end_idx, exp.eeg, epsilon=1)
                            elif exp.name == 'multiT~storysnippet~focus' or exp.name == 'speech~speech~focus' or exp.name == 'speech~speech~attend':
                                exp.eeg = self.__correctEvents_story(trials[trial_idx]["trigger_log"], events,start_idx, end_idx, exp.eeg, epsilon=1, attended_stimulus=attended_stimulus)
                            else:
                                exp.eeg = self.__correctEvents(trials[trial_idx]["trigger_log"], events,start_idx, end_idx, exp.eeg, epsilon=1, attended_stimulus=attended_stimulus)
                        else:
                            exp.eeg = self.__correctEvents(trials[trial_idx]["trigger_log"], events,start_idx, end_idx, exp.eeg, epsilon=1, attended_stimulus=attended_stimulus)

                        has_started = False
                        trial_idx+=1                            
                    j += 1

    def saveEEGLAB(self, path, sbj_id):
        out_path = os.path.abspath(os.path.join(path, sbj_id))
        out_files = []
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        channels = self.channels

        theta = [0]*self.num_channels
        radius = [0]*self.num_channels
        labels = []
        for ch in channels:
            labels.append(ch.name)
        sph_theta = [0]*self.num_channels
        sph_phi = [0]*self.num_channels
        sph_radius = [0]*self.num_channels
        X = [0]*self.num_channels
        Y = [0]*self.num_channels
        Z = [0]*self.num_channels

        chanlocs = fromarrays([theta, radius, labels, sph_theta, sph_phi, sph_radius, X, Y, Z], names=['theta', 'radius', 'labels', 'sph_theta', 'sph_phi', 'sph_radius', 'X', 'Y', 'Z'])

        for i in range(self.numOfExp + 1):
            if self.parts[i].valid:
                part = self.parts[i]
                name = part.name
                if i < self.numOfExp:
                    trials = part.trials
                else:
                    trials = []
                filename = out_path + '/{}.set'.format(name)
                num_samples = part.eeg.shape[1]
                times = np.arange(0,num_samples, dtype=np.float)/self.sample_rate
                eeg_dict = dict(data=part.eeg,
                        setname=name,
                        nbchan=float(self.num_channels),
                        pnts=float(num_samples),
                        trials=float(len(trials)),
                        srate=float(self.sample_rate),
                        xmin=times[0],
                        xmax=times[-1],
                        chanlocs=chanlocs,
                        event=[],
                        icawinv=[],
                        icasphere=[],
                        icaweights=[])

                savemat(filename, dict(EEG=eeg_dict), appendmat=False)
                out_files.append(filename)
        return out_files

    def saveBIDS(self, bidsPath, bidsName, eegPath, sbj_id, sbj_age, sbj_sex, sbj_hand, meas_date):
        eeg_path = os.path.join(eegPath, sbj_id)
        out_files = []
        if not os.path.exists(eeg_path):
            os.makedirs(eeg_path)

        channels = self.channels

        theta = []
        radius = []
        labels = []
        sph_theta = []
        sph_phi = []
        sph_radius = []
        X = []
        Y = []
        Z = []
        ch_types = []
        for ch in channels:
            labels.append(ch.name)
            theta.append(ch.theta)
            radius.append(ch.radius)
            sph_theta.append(ch.sph_theta)
            sph_phi.append(ch.sph_phi)
            sph_radius.append(ch.sph_radius)
            X.append(ch.X)
            Y.append(ch.Y)
            Z.append(ch.Z)
            ch_types.append(ch.ch_type)

        chanlocs = fromarrays([theta, radius, labels, sph_theta, sph_phi, sph_radius, X, Y, Z, ch_types], names=['theta', 'radius', 'labels', 'sph_theta', 'sph_phi', 'sph_radius', 'X', 'Y', 'Z', 'type'])
        #re-reference to common reference channel Fpz for Scalp EEG and cRef for Ear EEG
        ref_idx1 = labels.index('Fpz')
        ref_idx2 = labels.index('cRef')
        for i in range(self.numOfExp + 1):
            part = self.parts[i]
            ref1_data = part.eeg[ref_idx1,...]
            ref2_data = part.eeg[ref_idx2,...]
            part.eeg[:31,...] = part.eeg[:31,...] - ref1_data
            part.eeg[32:,...] = part.eeg[32:,...] - ref2_data            
        
        for i in range(self.numOfExp + 1):
            if self.parts[i].valid:
                part = self.parts[i]
                name = part.name
                if i < self.numOfExp:
                    trials = part.trials
                else:
                    trials = []
                filename = os.path.join(eeg_path, '{}.set'.format(name))
                
                #creating events and epochs data
                eeglab_events = []
                eeglab_epochs = []
                bids_events = []
                if (part.events != None):
                    eeglab_events = makeEEGLABEvents(part)
                    #eeglab_epochs = makeEEGLABEpochs(part, eeglab_events)
                    #bids_events = makeBIDSEvents(part)
                    bids_events = makeBIDSEvents(eeglab_events)
                
                num_samples = part.eeg.shape[1]
                times = np.arange(0,num_samples, dtype=np.float)/self.sample_rate
                eeg_dict = dict(data=part.eeg,
                        setname=name,
                        nbchan=float(self.num_channels),
                        pnts=float(num_samples),
                        trials=1.0,
                        srate=float(self.sample_rate),
                        xmin=times[0],
                        xmax=times[-1],
                        chanlocs=chanlocs,
                        event=eeglab_events,
                        epoch=eeglab_epochs,
                        icawinv=[],
                        icasphere=[],
                        icaweights=[])
                savemat(filename, dict(EEG=eeg_dict), appendmat=False)
                out_files.append(filename)
                eeglab_raw = read_raw_eeglab(filename)
                eeglab_raw.preload = False
                #update EEG data
                channel_types = {}
                for j in range(len(Channel.AAD_channel_type)):
                    channel_types.update({Channel.AAD_channels[j]:Channel.AAD_channel_type[j]})
                channel_types.update({'EOG': 'eog'})
                eeglab_raw.set_channel_types(channel_types)
                montage = read_custom_montage(os.path.abspath(self.LOCS_PATH))
                eeglab_raw.set_montage(montage, on_missing='ignore')
                #eeglab_raw.set_meas_date(datetime(meas_date[0], meas_date[1], meas_date[2]).astimezone(pytz.UTC))
                eeglab_raw.set_meas_date(datetime(meas_date[0], meas_date[1], meas_date[2]).replace(tzinfo=timezone.utc))
                eeglab_raw.info['subject_info'] = {"sex": sbj_sex, "hand": sbj_hand, "birthday": (meas_date[0]-sbj_age, 6, 1)} #default birthday 1/6/<birthyear>
                #
                bids_root = os.path.join(bidsPath, bidsName)
                bids_path = BIDSPath(subject=sbj_id, task=name,root=bids_root)
                if(part.events!=None):
                    events_data = np.array(bids_events)
                    #write_raw_bids(eeglab_raw, bids_path, events_data=events_data[events_data[:, 0].argsort()], event_id=sti.EVENT_ID[name], overwrite=True)
                    write_raw_bids(eeglab_raw, bids_path, events_data=events_data, event_id=sti.EVENT_ID[name], overwrite=True)
                    #write_raw_bids(eeglab_raw, bids_path, overwrite=True)
                else:
                    write_raw_bids(eeglab_raw, bids_path, overwrite=True)
                write_meta_bids(part, bids_path, eeglab_events)
        return out_files
        
    def readFromBIDS(self, bidsPath, sbj_id):

        return True

    def print_data(self, name):
        print('name: ', self.name)
        print('start_time: ', self.start_time)
        print('num_samples: ', self.num_samples)
        print('num_channels: ', self.num_channels)
        for i in range(self.numOfExp):
            exp = self.parts[i]
            print('experiment: ', exp.name)
            print('path: ', exp.path)
            print('numofTrial: ', exp.numOfTrial)
            print('trials: ', exp.trials)
            print('len: ', exp.eeg.shape)
            print('num of events: {}'.format(len(exp.events)))
            file_name = 'events_{}_{}.txt'.format(name, exp.name)
            with open(file_name, "w") as f:
                for e in exp.events:
                    f.write('{}\t{}\t{}\t{}\n'.format(e.sampleth, e.code, e.toNextEvent, e.stimuli_idx))
                f.close()
        
            
class Channel:
    """ 'Channel' represents a device channel. It has the next properties:

        name : 'string' The name of the channel.

        unit_name : 'string' The name of the unit (e.g. 'μVolt)  of the sample-data of the channel.
    """
    
    AAD_channels = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7-T3', 'C3', 'Cz', 'C4', 'T8-T4', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7-T5', 'P3', 'Pz', 'P4', 'P8-T6', 'POz', 'O1', 'Oz', 'O2', 'ELA', 'ELB', 'ELC', 'ELE', 'ELI', 'ELT', 'ERA', 'ERB', 'ERC', 'ERE', 'ERI', 'ERT', 'EOG', 'cRef']
    AAD_channel_type = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg']
    def __init__(self, name=None, unit_name=None, ch_type='EEG'):
        self.name = name
        self.unit_name = unit_name
        self.ch_type = ch_type
        self.theta = 0
        self.radius = 0
        self.sph_theta = 0
        self.sph_phi = 0
        self.sph_radius = 0
        self.X = 0
        self.Y = 0
        self.Z = 0
