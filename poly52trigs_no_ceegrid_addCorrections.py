import scipy.io
from SerialTriggerDecoder import *
from eventaad.dataset.converter import *
import matplotlib.pyplot as plt
import numpy as np
import csv
from expectedTriggerCalculator import *

def poly52trigs_no_ceegrid_addCorrections(basePath, participantNumber,filterBufferPeriod):
    rawDataPath = basePath + 'sourcedata\P' + participantNumber + '\\'
    
    subjFolder = "sub-" + participantNumber
    
    correctionsFile_scalp = rawDataPath + 'P' + participantNumber + '_scalpCorrTrigs.txt'
    
    scalp_eegCodes = ([x.split()[0] for x in open(correctionsFile_scalp).readlines()])
    scalp_eegLatencies = ([x.split()[1] for x in open(correctionsFile_scalp).readlines()])#Need to adjust this stuff..
    
    scalp_eegCodes.remove(scalp_eegCodes[0])
    scalp_eegLatencies.remove(scalp_eegLatencies[0])

    #Convert all to ints:
    scalp_eegCodes = [int(i) for i in scalp_eegCodes]
    scalp_eegLatencies = [int(i) for i in scalp_eegLatencies]

#####################################################################################################################################################################
    #Good to check that the number of trigs is as expected:

    num_scalp_eeg_events = len(scalp_eegCodes)

    #Calculate no. of expected trigs, compare to actual:  
    expectedTrigs = expectedTriggerCalculator(basePath, participantNumber)

    if expectedTrigs > num_scalp_eeg_events:
        diff = str(expectedTrigs - num_scalp_eeg_events)
        print("WARNING - " + diff + " SCALP EVENTS MISSING")#
    if expectedTrigs < num_scalp_eeg_events:
        diff = str(num_scalp_eeg_events - expectedTrigs)
        print("WARNING - " + diff + " EXCESS SCALP EVENTS DETECTED")


#######################################################################################################################################################################################################
#######################################################################################################################################################################################################
    #Save to files, in a BIDS-friendly format:
    
    outputDir = basePath + "bids_dataset\sub-" + participantNumber + "\eeg\\"
    outputDirExists = os.path.exists(outputDir)
    
    if not outputDirExists:
        os.makedirs(outputDir)
#######################################################################################################################################################################################################   
    
    sfreq = 1000
    
    #Do this "task by task", i.e emotion decoding first etc:
    task1 = "emotion"
    filename = subjFolder + "_task-" + task1 + "_acq-scalp_events.tsv"
    scalpEvents_BIDS = (outputDir + filename)

    P1practiceEnded = False
    P1ended = False
    P2practiceEnded = False
    P2ended = False
    P3practiceEnded = False
    P3ended = False
    
    P1P3trialStartVals = np.arange(1, 73, 2) #Can reuse these later
    P1P3trialEndVals = np.arange(2, 74, 2)
    
    with open(scalpEvents_BIDS, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        writer.writerow(["onset", "duration", "value", "significance", "trial"])
        i = 0
        j = 1 #For keeping track of main trials
        
        P1startLatency_scalp = scalp_eegLatencies[i]/sfreq - filterBufferPeriod
        duration = "0" #Throughout P1
        
        while P1ended == False:
            onset = scalp_eegLatencies[i]/sfreq - P1startLatency_scalp #So that the first trigger is at t = 0
            value = scalp_eegCodes[i]
                
            if P1practiceEnded == False:
                trial = "prac"
                if value in P1P3trialStartVals:
                    significance = "trial_start"
                else:
                    significance = "trial_end"
                    P1practiceEnded = True
                    
            elif value == 154:
                trial = "N/A"
                significance = "main_trials_start"
            elif value == 155:
                trial = "N/A"
                significance = "main_trials_end"
                
            else:
                trial = str(j)
                if value in P1P3trialStartVals:
                    significance = "trial_start"
                elif value in P1P3trialEndVals:
                    significance = "trial_end"
                    j += 1
                
            if value == 46: #For these trigs in particular, started about 42.7ms late for some reason
                onset -= 0.043
                    
            writer.writerow([str(onset), duration, str(value), significance, trial])
            i += 1
            
            if significance == "main_trials_end":
                P1endLatency_scalp = onset + P1startLatency_scalp + filterBufferPeriod
                P1ended = True  
                         
        tsvfile.close
    
    
    #P2:
    task2 = "attnMultInstOBs"    
    filename = subjFolder + "_task-" + task2 + "_acq-scalp_events.tsv"
    scalpEvents_BIDS = (outputDir + filename)

    P2trialStartVals = np.arange(73, 144, 2) #Can reuse these later
    P2trialEndVals = np.arange(74, 145, 2)
    P2attendedOBvals = np.array([145, 149, 153])
    P2unattendedOBvals = np.array([146, 147, 148, 150, 151, 152])    
    
    with open(scalpEvents_BIDS, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        writer.writerow(["onset", "duration", "value", "significance", "trial"])
        
        #Continue to use i from before.
        j = 1 #For keeping track of prac trials
        k = 1 #"     "       "   "  main   "
        
        P2startLatency_scalp = scalp_eegLatencies[i]/sfreq - filterBufferPeriod
        
        while P2ended == False:
            onset = scalp_eegLatencies[i]/sfreq - P2startLatency_scalp
            value = scalp_eegCodes[i]
            
            if value == 156:
                trial = "N/A"
                significance = "main_trials_start"
                P2practiceEnded = True
                
            elif P2practiceEnded == False:
                trial = "prac_" + str(j)
                if value in P2trialStartVals:
                    significance = "trial_start"
                elif value in P2attendedOBvals:
                    significance = "attended_OB"
                elif value in P2unattendedOBvals:
                    significance = "unattended_OB"
                elif value in P2trialEndVals:
                    significance = "trial_end"
                    j += 1
                    
            elif value == 157:
                trial = "N/A"
                significance = "main_trials_end"
                
            else:
                trial = str(k)
                if value in P2trialStartVals:
                    significance = "trial_start"
                elif value in P2attendedOBvals:
                    significance = "attended_OB"
                elif value in P2unattendedOBvals:
                    significance = "unattended_OB"
                elif value in P2trialEndVals:
                    significance = "trial_end"
                    k += 1
                
            if "OB" in significance: #Oddballs
                duration = "0.49197278911"
            else:
                duration = "0"
            writer.writerow([str(onset), duration, str(value), significance, trial])
            i += 1
            
            if significance == "main_trials_end":
                P2endLatency_scalp = onset + P2startLatency_scalp + filterBufferPeriod
                P2ended = True  
                
        tsvfile.close
        
    #P3:
    task3 = "attnOneInstNoOBs"    
    filename = subjFolder + "_task-" + task3 + "_acq-scalp_events.tsv"
    scalpEvents_BIDS = (outputDir + filename)
        
    with open(scalpEvents_BIDS, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        writer.writerow(["onset", "duration", "value", "significance", "trial"])
        j = 1 #For keeping track of main trials
        
        P3startLatency_scalp = scalp_eegLatencies[i]/sfreq - filterBufferPeriod
        duration = "0" #Should already be set, but if e.g triggers above are missed, this acts as a failsafe.
        
        while P3ended == False:
            onset = scalp_eegLatencies[i]/sfreq - P3startLatency_scalp #So that the first trigger is at t = 0
            value = scalp_eegCodes[i]
                
            if P3practiceEnded == False:
                trial = "prac"
                if value in P1P3trialStartVals:
                    significance = "trial_start"
                else:
                    significance = "trial_end"
                    P3practiceEnded = True
                    
            elif value == 158:
                trial = "N/A"
                significance = "main_trials_start"
            elif value == 159:
                trial = "N/A"
                significance = "main_trials_end"
                
            else:
                trial = str(j)
                if value in P1P3trialStartVals:
                    significance = "trial_start"
                elif value in P1P3trialEndVals:
                    significance = "trial_end"
                    j += 1
                    
            if value == 46: #For these trigs in particular, started about 42.7ms late for some reason
                onset -= 0.043
                
            writer.writerow([str(onset), duration, str(value), significance, trial])
            i += 1
            
            if significance == "main_trials_end" or i == len(scalp_eegLatencies):
                P3endLatency_scalp = onset + P3startLatency_scalp + filterBufferPeriod
                P3ended = True  
                         
        tsvfile.close
        
    partStartLatencies_scalp =[P1startLatency_scalp, P2startLatency_scalp, P3startLatency_scalp]
    partEndLatencies_scalp = [P1endLatency_scalp, P2endLatency_scalp, P3endLatency_scalp]
    partStartEndLatencies_scalp = np.stack([partStartLatencies_scalp, partEndLatencies_scalp])
        
    return partStartEndLatencies_scalp