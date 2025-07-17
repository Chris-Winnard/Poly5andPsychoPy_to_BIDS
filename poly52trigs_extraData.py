import scipy.io
from SerialTriggerDecoder import *
from eventaad.dataset.converter import *
import matplotlib.pyplot as plt
import numpy as np
import csv
from expectedTriggerCalculator import *


def poly52trigs_extraData(basePath, participantNumber, filterBufferPeriod):
    rawDataPath = basePath + 'sourcedata\P' + participantNumber + '\\'
    
    subjFolder = "sub-" + participantNumber
    
    correctionsFile_scalp = rawDataPath + 'P' + participantNumber + '_scalpCorrTrigs_extraData.txt'
    
    scalp_eegCodes = ([x.split()[0] for x in open(correctionsFile_scalp).readlines()])
    scalp_eegLatencies = ([x.split()[1] for x in open(correctionsFile_scalp).readlines()])#Need to adjust this stuff..
    
    scalp_eegCodes.remove(scalp_eegCodes[0])
    scalp_eegLatencies.remove(scalp_eegLatencies[0])
    
    correctionsFile_ceegrid = rawDataPath + 'P' + participantNumber + '_ceegridCorrTrigs_extraData.txt'
    
    ceegridCodes = ([x.split()[0] for x in open(correctionsFile_ceegrid).readlines()])
    ceegridLatencies = ([x.split()[1] for x in open(correctionsFile_ceegrid).readlines()])
    
    ceegridCodes.remove(ceegridCodes[0])
    ceegridLatencies.remove(ceegridLatencies[0])

    #Convert all to ints:
    scalp_eegCodes = [int(i) for i in scalp_eegCodes]
    scalp_eegLatencies = [int(i) for i in scalp_eegLatencies]
    ceegridCodes = [int(i) for i in ceegridCodes]
    ceegridLatencies = [int(i) for i in ceegridLatencies]

#####################################################################################################################################################################
    #Good to run various checks: that there are the same number of events for the scalp/ceegrid files; that these do not contradict; and separately checking that events in
    #each are sensible. These are only rough and not exhaustive

    #Check same number of trigs recorded for scalp and ceegrid:
    num_scalp_eeg_events = len(scalp_eegCodes)
    num_ceegrid_events = len(ceegridCodes)
    difference = num_scalp_eeg_events - num_ceegrid_events
    if difference > 0:
        print("WARNING - MORE SCALP THAN CEEGRID EVENTS DETECTED")
        lesserEventCount = num_ceegrid_events
    elif difference < 0:
        print("WARNING - MORE CEEGRID EVENTS THAN SCALP EVENTS DETECTED")
        lesserEventCount = num_scalp_eeg_events
    else:
        print("Equal number of scalp and ceegrid events detected.")
        lesserEventCount = num_scalp_eeg_events #could use either 
        
    
    #Calculate no. of expected trigs, compare to recorded:   
    expectedTrigs = expectedTriggerCalculator(basePath, participantNumber)
    if expectedTrigs > num_scalp_eeg_events:
        diff = str(expectedTrigs - num_scalp_eeg_events)
        print("WARNING - " + diff + " SCALP EVENTS MISSING")#
    if expectedTrigs < num_scalp_eeg_events:
        diff = str(num_scalp_eeg_events - expectedTrigs)
        print("WARNING - " + diff + " EXCESS SCALP EVENTS DETECTED")
    if expectedTrigs > num_ceegrid_events:
        diff = str(expectedTrigs - num_ceegrid_events)
        print("WARNING - " + diff + " CEEGRID EVENTS MISSING")
    if expectedTrigs < num_ceegrid_events:
        diff = str(num_ceegrid_events - expectedTrigs)
        print("WARNING - " + diff + " EXCESS CEEGRID EVENTS DETECTED")
    if expectedTrigs == num_scalp_eeg_events and difference == 0:
        print("Good news: the number of trigs detected for both scalp and cEEGrid is what should be expected.")
    

    #Finally, check individual scalp and ceegrid trigs match up:
    x = 0
    for i in range(0, lesserEventCount):
        if scalp_eegCodes[i] != ceegridCodes[i]:
            print("WARNING - EVENT CODES DISCREPENCY")
            print("This is at the following sample in the scalp data: " + str(scalp_eegLatencies[i]))
            print("Alternatively, this is the following sample for the ceegrid data: " + str(ceegridLatencies[i]))
            
            
            #Single event checks. Note, if scalp/ceegrid EEG does have one or two more triggers, then will need to check the extra ones at the end
            #separately
         #  scalp_pattern = scalp_eeg.raw_events[i].get("pattern")
            scalp_code = scalp_eegCodes[i]
            
            if int(scalp_code) > 159:
                print("Scalp code value is too large.")
            
            ceegrid_code = ceegridCodes[i]
            
            if int(ceegrid_code) > 159:
                print("ceegrid code value is too large.")
                
            
            print("It is the " + str(i+1) + "th event in each data file. The scalp code is: " + str(scalp_code) + " and the ceegrid code is: " + str(ceegrid_code))
        if scalp_eegCodes[i] == ceegridCodes[i]:
            x += 1
            
    print(str(x) + " out of " + str(lesserEventCount) + " codes are the same between both data files. Note this assumes that the code numbers are the same, OR that if one file has more "
          + "events detected, these are at the end.")
    

#######################################################################################################################################################################################################
#######################################################################################################################################################################################################
    #Save to files
    
    outputDir = basePath + "bids_dataset\misc\\"
    outputDirExists = os.path.exists(outputDir)
    
    if not outputDirExists:
        os.makedirs(outputDir)
#######################################################################################################################################################################################################       
    sfreq = 1000
    i = 0
    partStartLatency_scalp = scalp_eegLatencies[i]/sfreq #Note- here we are only assuming up to one 'part' of the experiment recorded per participant (e.g part 2 scalp+ceegrid for P09)
    
    if participantNumber == "09": #Extra data for Part 2, scalp then ceegrid:
        P2practiceEnded = False
        P2ended = False

        task = "attnMultInstOBsExtra"    
        
        filename = subjFolder + "_task-" + task + "_acq-scalp_events.tsv"
        scalpEvents_BIDS = (outputDir + "sub-09\\eeg\\" + filename)
        os.makedirs(os.path.dirname(scalpEvents_BIDS), exist_ok=True)
        
        P2trialStartVals = np.arange(73, 144, 2)
        P2trialEndVals = np.arange(74, 145, 2)
        P2attendedOBvals = np.array([145, 149, 153])
        P2unattendedOBvals = np.array([146, 147, 148, 150, 151, 152])    
        
        for item in reversed(scalp_eegCodes):
            if item in P2trialEndVals:
                final_trig = item
                break
    
        with open(scalpEvents_BIDS, 'w', newline='') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
            writer.writerow(["onset", "duration", "value", "significance", "trial"])
            
            #Continue to use i from before.
            j = 1 #For keeping track of prac trials
            k = 1 #"     "       "   "  main   "
            
            
            while P2ended == False:
                onset = scalp_eegLatencies[i]/sfreq - partStartLatency_scalp
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
                
                if significance == "main_trials_end" or value == final_trig:
                    partEndLatency_scalp = onset + partStartLatency_scalp
                    P2ended = True  
                    
            tsvfile.close
        
#ceegrid:
        P2practiceEnded = False
        P2ended = False
        i = 0
        filename = subjFolder + "_task-" + task + "_acq-ceegrid_events.tsv"
        ceegridEvents_BIDS = (outputDir + "sub-09\\eeg\\" + filename)
        os.makedirs(os.path.dirname(ceegridEvents_BIDS), exist_ok=True)
        
        for item in reversed(ceegridCodes):
            if item in P2trialEndVals:
                final_trig = item
                break
        
        with open(ceegridEvents_BIDS, 'w', newline='') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
            writer.writerow(["onset", "duration", "value", "significance", "trial"])
            
            #Continue to use i from before.
            j = 1 #For keeping track of prac trials
            k = 1 #"     "       "   "  main   "
            
            partStartLatency_ceegrid = ceegridLatencies[i]/sfreq
            
            while P2ended == False:
                onset = ceegridLatencies[i]/sfreq - partStartLatency_ceegrid
                value = ceegridCodes[i]
                
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
                
                if significance == "main_trials_end" or value == final_trig:
                    partEndLatency_ceegrid = onset + partStartLatency_ceegrid
                    P2ended = True  
                    
            tsvfile.close
        
#######################################################################################################################################################################################################
    elif participantNumber == "28":        
        i = 0
        P3practiceEnded = False
        P3ended = False
        P1P3trialStartVals = np.arange(1, 73, 2)
        P1P3trialEndVals = np.arange(2, 74, 2)
        
        for item in reversed(scalp_eegCodes):
            if item in P1P3trialEndVals:
                final_trig = item
                break
            
        task = "attnOneInstNoOBsExtra"    
        filename = subjFolder + "_task-" + task + "_acq-scalp_events.tsv"
        scalpEvents_BIDS = (outputDir + "sub-28\\eeg\\" + filename)
        os.makedirs(os.path.dirname(scalpEvents_BIDS), exist_ok=True)
            
        with open(scalpEvents_BIDS, 'w', newline='') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
            writer.writerow(["onset", "duration", "value", "significance", "trial"])
            j = 1 #For keeping track of main trials
            
            partStartLatency_scalp = scalp_eegLatencies[i]/sfreq
            duration = "0" #Should already be set, but if e.g triggers above are missed, this acts as a failsafe.
            
            while P3ended == False:
                onset = scalp_eegLatencies[i]/sfreq - partStartLatency_scalp #So that the first trigger is at t = 0
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
                
                if significance == "main_trials_end" or value == final_trig:
                    partEndLatency_scalp = onset + partStartLatency_scalp
                    P3ended = True  
                             
            tsvfile.close
#ceegrid:       
        P3practiceEnded = False
        P3ended = False
        i = 0
        
        P1P3trialStartVals = np.arange(1, 73, 2) #Can reuse these later
        P1P3trialEndVals = np.arange(2, 74, 2)
            
        #P3:
        filename = subjFolder + "_task-" + task + "_acq-ceegrid_events.tsv"
        ceegridEvents_BIDS = (outputDir + "sub-28\eeg\\" + filename)
        os.makedirs(os.path.dirname(ceegridEvents_BIDS), exist_ok=True)
            
        with open(ceegridEvents_BIDS, 'w', newline='') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
            writer.writerow(["onset", "duration", "value", "significance", "trial"])
            j = 1 #For keeping track of main trials
            
            partStartLatency_ceegrid = ceegridLatencies[i]/sfreq
            duration = "0" #Should already be set, but if e.g triggers above are missed, this acts as a failsafe.
            
            while P3ended == False:
                onset = ceegridLatencies[i]/sfreq - partStartLatency_ceegrid #So that the first trigger is at t = 0
                value = ceegridCodes[i]
                    
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
                
                if significance == "main_trials_end" or value == final_trig:
                    partEndLatency_ceegrid = onset + partStartLatency_ceegrid
                    P3ended = True  
                             
            tsvfile.close

    partStartEndLatencies_scalp = np.stack([partStartLatency_scalp-filterBufferPeriod, partEndLatency_scalp+filterBufferPeriod])
    partStartEndLatencies_ceegrid = np.stack([partStartLatency_ceegrid-filterBufferPeriod, partEndLatency_ceegrid+filterBufferPeriod])
        
    return partStartEndLatencies_scalp, partStartEndLatencies_ceegrid