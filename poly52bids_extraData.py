from poly52bids_partial_ceegrid_addCorrections import *
from poly52bids_no_ceegrid import *
import pathlib
import matlab.engine
from poly52trigs_extraData import *
from setAndTrigs2bids_extraData import *
from additionalDataReader import *

def poly52bids_extraData(basePath, participantNumber, handedness, filterBufferPeriod):
    #Stage one has already been completed. Below are stages 2-4. Stages 5-6 completed after,
    #in poly52bids.py.
        
    partStartEndLatencies_scalp, partStartEndLatencies_ceegrid = poly52trigs_extraData(basePath,
                                                                                       participantNumber,filterBufferPeriod)
        
    additionalData = additionalDataReader(basePath, participantNumber, handedness)
    
    setAndTrigs2bids_extraData(basePath, participantNumber, partStartEndLatencies_scalp, partStartEndLatencies_ceegrid,
                               additionalData)