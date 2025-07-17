import os
import sys
from scipy.io import savemat, loadmat
import numpy as np
from numpy.core.records import fromarrays
import mne
from mne.io import Raw, read_raw_eeglab
#from mne import export
sys.path.append('../..')
from TMSiSDK.file_readers import Poly5Reader

from eventaad.dataset.EEGData import EEGData, Channel

def poly5ToEEG(poly5_path, name=None):
    eeg = EEGData()
    try:
        data = Poly5Reader(poly5_path)
    except:
        print('Error in reading poly5 file.')
        return None
    if name!=None:
        eeg.name = name
    else:
        eeg.name = data.name
    eeg.start_time = data.start_time
    eeg.sample_rate = data.sample_rate
    eeg.num_samples = data.num_samples
    eeg.samples = data.samples
    eeg.num_channels = data.num_channels
    for ch in data.channels:
        eeg.addChannel(ch.name, ch.unit_name)
    
    #Fixing Preben3 pilot recording
    '''
    eeg.samples = np.delete(eeg.samples, range(4600000, 5600000), axis=1)
    eeg.num_samples = eeg.num_samples - (5600000 - 4600000)    
    '''
    
    #Fixing Kaare2 pilot recording
    '''
    if '1_scalp' in poly5_path:
        eeg.samples = np.delete(eeg.samples, range(2887530, 3000000), axis=1)
        eeg.num_samples = eeg.num_samples - (3000000 - 2887530)
    elif '2_ear' in poly5_path:
        eeg.samples = np.delete(eeg.samples, range(2898140, 3010610), axis=1)
        eeg.num_samples = eeg.num_samples - (3010610 - 2898140)    
    '''
    
    #Fixing Kaare3 pilot recording
    '''
    if '1_scalp' in poly5_path:
        eeg.samples = np.delete(eeg.samples, range(300000, 687000), axis=1)
        eeg.num_samples = eeg.num_samples - (687000 - 300000)
    elif '2_ear' in poly5_path:
        eeg.samples = np.delete(eeg.samples, range(300000, 700000), axis=1)
        eeg.num_samples = eeg.num_samples - (700000 - 300000)    
    '''
    
    #Fixing Preben4 pilot recording
    '''
    eeg.samples = np.delete(eeg.samples, range(1100000, 1187400), axis=1)
    eeg.num_samples = eeg.num_samples - (1187400 - 1100000)   
    '''

    #Fixing Preben5 pilot recording
    '''
    if '1_scalp' in poly5_path:
        eeg.samples = np.delete(eeg.samples, range(1940000, 2000000), axis=1)
        eeg.num_samples = eeg.num_samples - (2000000 - 1940000) 
    elif '2_ear' in poly5_path:
        eeg.samples = np.delete(eeg.samples, range(3840000, 3940000), axis=1)
        eeg.num_samples = eeg.num_samples - (3940000 - 3840000)
    ''' 
    
    #Fixing Kristian pilot recording
    '''
    if '1_scalp' in poly5_path:
        eeg.samples = np.delete(eeg.samples, range(4470000, 5020000), axis=1)
        eeg.num_samples = eeg.num_samples - (5020000 - 4470000) 
    elif '2_ear' in poly5_path:
        eeg.samples = np.delete(eeg.samples, range(3705000, 4242000), axis=1)
        eeg.num_samples = eeg.num_samples - (4242000 - 3705000)
    ''' 
    
    return eeg
    
def matToEEG(mat_path): 
    eeg = EEGData()
    try:
        mat = loadmat(mat_path)
    except:
        print('Error in reading mat file.')
        return None
    eeg.name = mat['name']
    start_time = mat['start_time']
    eeg.start_time = datetime.datetime(start_time[0], start_time[1], start_time[2], start_time[4], start_time[5], start_time[6])
    eeg.sample_rate = np.asscalar(mat['sample_rate'])
    eeg.num_samples = mat['num_samples']
    eeg.samples = mat['samples']
    eeg.num_channels = mat['num_channels']
    for ch in mat['channels']:
        eeg.addChannel(ch.name, ch.unit_name)

    return eeg

def poly5ToMat(poly5_path, mat_path=None):
    if mat_path==None:
        filename, file_extension = os.path.splitext(poly5_path)
        mat_path = os.path.abspath(filename + ".mat")
    try:
        data = Poly5Reader(poly5_path)
    except:
        print('Error in reading poly5 file.')
        return None
    eeg_dict = {}

    eeg_dict['name'] = data.name
    eeg_dict['start_time'] = data.start_time
    eeg_dict['sample_rate'] = data.sample_rate
    eeg_dict['num_samples'] = data.num_samples
    eeg_dict['samples'] = data.samples
    eeg_dict['num_channels'] = data.num_channels
    eeg_dict['channels'] = data.channels
    savemat(mat_path, eeg_dict)


def toRawEEGLAB(eeg_data):
    '''
    create mne.Raw data from EEGData
    '''
    return None 

def toEEGLABFile(eeg_data, filename, configfile=None):
    """Export EEG data to EEGLAB .set file."""
    #read config json from file
    if configfile != None:
        print('Config file: ' + configfile)

    data = eeg_data.samples * 1e6  #convert to microvolts
    fs = eeg_data.sample_rate 
    times = np.arange(0,eeg_data.num_samples, dtype=np.float)/fs
    channels = eeg_data.channels

    theta = [0]*eeg_data.num_channels
    radius = [0]*eeg_data.num_channels
    labels = []
    for ch in channels:
        labels.append(ch.name)
    sph_theta = [0]*eeg_data.num_channels
    sph_phi = [0]*eeg_data.num_channels
    sph_radius = [0]*eeg_data.num_channels
    X = [0]*eeg_data.num_channels
    Y = [0]*eeg_data.num_channels
    Z = [0]*eeg_data.num_channels

    chanlocs = fromarrays([theta, radius, labels, sph_theta, sph_phi, sph_radius, X, Y, Z], names=['theta', 'radius', 'labels', 'sph_theta', 'sph_phi', 'sph_radius', 'X', 'Y', 'Z'])
    eeg_dict = dict(data=data,
                    setname=eeg_data.name,
                    nbchan=eeg_data.num_channels,
                    pnts=eeg_data.num_samples,
                    trials=1,
                    srate=fs,
                    xmin=times[0],
                    xmax=times[-1],
                    chanlocs=chanlocs,
                    event=[],
                    icawinv=[],
                    icasphere=[],
                    icaweights=[])


    savemat(filename, dict(EEG=eeg_dict), appendmat=False)

def rawToEEGLABFile(eeglab_raw, filename, verbose=None):
    eeglab_raw.export(filename, fmt='eeglab', verbose=verbose)

def fromEEGLAB(eeglab_filename):
    raw = read_raw_eeglab(eeglab_filename)
    return raw


def getEpochs():
    return None