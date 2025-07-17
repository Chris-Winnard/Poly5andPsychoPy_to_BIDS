import scipy.io
from SerialTriggerDecoder import *
from eventaad.dataset.converter import *
import matplotlib.pyplot as plt
import numpy as np
import csv
from expectedTriggerCalculator import *


def poly52trigs_partial_ceegrid_addCorrections(basePath, participantNumber, filterBufferPeriod):
    """This has been set up for P06 in particular. Keeping in the scalp vs. ceegrid trig checks early on, as all of 
    the cEEGrid trigs recorded OK, and these are useful to have. Also, we do have #SOME P2, P3 ceegrid data, so leaving
    the code generalisable as much as possible."""
    
    
    rawDataPath = basePath + 'sourcedata\P' + participantNumber + '\\'
    
    subjFolder = "sub-" + participantNumber
    
    correctionsFile_scalp = rawDataPath + 'P' + participantNumber + '_scalpCorrTrigs.txt'
    
    scalp_eegCodes = ([x.split()[0] for x in open(correctionsFile_scalp).readlines()])
    scalp_eegLatencies = ([x.split()[1] for x in open(correctionsFile_scalp).readlines()]) #Need to adjust this stuff..
    
    scalp_eegCodes.remove(scalp_eegCodes[0])
    scalp_eegLatencies.remove(scalp_eegLatencies[0])
    
    correctionsFile_ceegrid = rawDataPath + 'P' + participantNumber + '_ceegridCorrTrigs.txt'
    
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
        print("WARNING - CEEGRID EVENTS MISSING")
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
    #Save to files, in a BIDS-friendly format:
    
    outputDir = basePath + "bids_dataset\sub-" + participantNumber + "\eeg\\"
    outputDirExists = os.path.exists(outputDir)
    
    if not outputDirExists:
        os.makedirs(outputDir)
#######################################################################################################################################################################################################   
    #First, scalp:
    
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
        duration = "0" #Should already be set, but if e.g., triggers above are missed, this acts as a failsafe.
        
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
        
#######################################################################################################################################################################################################
    #ceegrid:
        
    sfreq = 1000
    
    #From careful inspection, already found which trials have no ground so will just input start/end latencies here manually:
    #For P2endLatency_ceegrid, choosing 1s AFTER the last OK trial ends.
    #For P3startLatency_ceegrid, choosing 1s BEFORE the first OK trial starts.
    #We also add/subtract filterBufferPeriod to those respectively (these periods don't include data without the ground).
    P2earlyStopLatency_ceegrid = 1544141/sfreq
    P3lateStartLatency_ceegrid = 2694658/sfreq
    P2endLatency_ceegrid = P2earlyStopLatency_ceegrid + filterBufferPeriod
    P3startLatency_ceegrid = P3lateStartLatency_ceegrid - filterBufferPeriod
    
    #Do this "task by task", i.e emotion decoding first etc:
    task1 = "emotion"
    filename = subjFolder + "_task-" + task1 + "_acq-ceegrid_events.tsv"
    ceegridEvents_BIDS = (outputDir + filename)

    P1practiceEnded = False
    P1ended = False
    P2practiceEnded = False
    P2ended = False
    P3practiceEnded = False
    P3ended = False
    
    P1P3trialStartVals = np.arange(1, 73, 2) #Can reuse these later
    P1P3trialEndVals = np.arange(2, 74, 2)
    
    with open(ceegridEvents_BIDS, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        writer.writerow(["onset", "duration", "value", "significance", "trial"])
        i = 0
        j = 1 #For keeping track of main trials
        
        P1startLatency_ceegrid = ceegridLatencies[i]/sfreq - filterBufferPeriod
        
        while P1ended == False:
            onset = ceegridLatencies[i]/sfreq - P1startLatency_ceegrid #So that the first trigger is at t = 0
            value = ceegridCodes[i]
                
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
                P1endLatency_ceegrid = onset + P1startLatency_ceegrid + filterBufferPeriod
                P1ended = True  

    #P2. We don't use P2 data but kept here as a 'placeholder'/to prevent confusion w/ P3.
    task2 = "attnMultInstOBs"    
    filename = subjFolder + "_task-" + task2 + "_acq-ceegrid_events.tsv"
    ceegridEvents_BIDS = (outputDir + filename)

    P2trialStartVals = np.arange(73, 144, 2) #Can reuse these later
    P2trialEndVals = np.arange(74, 145, 2)
    P2attendedOBvals = np.array([145, 149, 153])
    P2unattendedOBvals = np.array([146, 147, 148, 150, 151, 152])    
    
    with open(ceegridEvents_BIDS, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        writer.writerow(["onset", "duration", "value", "significance", "trial"])
        
        #Continue to use i from before.
        j = 1 #For keeping track of prac trials
        k = 1 #"     "       "   "  main   "
        
        P2startLatency_ceegrid = ceegridLatencies[i]/sfreq - filterBufferPeriod
        
        while P2ended == False:
            onset = ceegridLatencies[i]/sfreq - P2startLatency_ceegrid
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
                    
                
            elif onset + P2startLatency_ceegrid > P2endLatency_ceegrid:  #Once past the early stopping point, we should ignore 
            #the current trig, and go back to the early stopping point, as the ground comes loose after it.
                onset = P2earlyStopLatency_ceegrid - P2startLatency_ceegrid
                trial = "N/A"
                significance = "main_trials_end" #Give it this significance/value for simplicity.
                value = 157
                
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
            
            if significance == "main_trials_end": #Set to stop here as the ground comes loose after.
                P2ended = True  
                
        tsvfile.close

    #P3:
    task3 = "attnOneInstNoOBs"    
    filename = subjFolder + "_task-" + task3 + "_acq-ceegrid_events.tsv"
    ceegridEvents_BIDS = (outputDir + filename)
        
    with open(ceegridEvents_BIDS, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        writer.writerow(["onset", "duration", "value", "significance", "trial"])
        j = 3 #For keeping track of main trials. Start from 3rd trial since we miss the pract one/first two mains.
        P3lateStartOccurred = False
        
        while P3lateStartOccurred == False: #We keep moving up in latency indices, but don't leave this loop
        #until we've passed the late start latency.
        #Don't consider 'P3startLatency_ceegrid' here, comparing to overall latencies.
            onset = ceegridLatencies[i]/sfreq
            i += 1
            if onset > P3lateStartLatency_ceegrid:
                onset = float(filterBufferPeriod) #Equivalent to what's used in all other events files.
                value = 158
                trial = "N/A"
                significance = "main_trials_start"
                writer.writerow([str(onset), duration, str(value), significance, trial])
                P3lateStartOccurred = True
                
                i -= 1 #Otherwise we would miss the trig just after P3lateStartLatency_ceegrid
        
        duration = "0" #Should already be set, but if e.g., triggers above are missed, this acts as a failsafe.

        while P3ended == False:
            onset = ceegridLatencies[i]/sfreq - P3startLatency_ceegrid #So that the first trigger is at t = 0
            value = ceegridCodes[i]
            
            #Already passed pract trials and what would have been 'main_trials_start'
            if value == 159:
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
            
            if significance == "main_trials_end" or i == len(ceegridLatencies):
                P3endLatency_ceegrid = onset + P3startLatency_ceegrid + filterBufferPeriod
                P3ended = True  
                
        tsvfile.close
    
    partStartLatencies_ceegrid = [P1startLatency_ceegrid, P2startLatency_ceegrid, P3startLatency_ceegrid]
    partEndLatencies_ceegrid = [P1endLatency_ceegrid, P2endLatency_ceegrid, P3endLatency_ceegrid]
    partStartEndLatencies_ceegrid = np.stack([partStartLatencies_ceegrid, partEndLatencies_ceegrid])
        
    return partStartEndLatencies_scalp, partStartEndLatencies_ceegrid