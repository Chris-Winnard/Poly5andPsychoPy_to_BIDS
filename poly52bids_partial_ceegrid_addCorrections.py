import pathlib
import matlab.engine
from poly52trigs_partial_ceegrid_addCorrections import *
from setAndTrigs2bids_partial_ceegrid import *
from additionalDataReader import *

def poly52bids_partial_ceegrid_addCorrections(basePath, participantNumber, handedness, filterBufferPeriod):
    """For a recording where cEEGrid ground came loose (only P1 FULLY recorded), and also trigs needed corrections."""
    
    eng = matlab.engine.start_matlab()
    eng.poly52set(basePath, participantNumber, nargout=0)
    eng.quit()   
    print("cEEGrid arrays OK for " + participantNumber + ".")
    print("Stage 1 of conversion complete.")
    
    partStartEndLatencies_scalp, partStartEndLatencies_ceegrid = poly52trigs_partial_ceegrid_addCorrections(basePath, 
                                                                                                            participantNumber, filterBufferPeriod)
    print("This file had missing or incorrect triggers, which have been corrected manually. These may be a little less precise than"
          + " otherwise.")
    print("Stage 2 of conversion complete.")
    
    additionalData = additionalDataReader(basePath, participantNumber, handedness)
    print("Stage 3 of conversion complete.")
    
    setAndTrigs2bids_partial_ceegrid(basePath, participantNumber, partStartEndLatencies_scalp, partStartEndLatencies_ceegrid,
                                     additionalData)
    print("Stage 4 of conversion complete.")