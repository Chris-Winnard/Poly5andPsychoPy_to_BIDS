import pathlib
import matlab.engine
from poly52trigs_no_ceegrid import *
from poly52trigs_no_ceegrid_addCorrections import *
from setAndTrigs2bids_no_ceegrid import *
from additionalDataReader import *

def poly52bids_no_ceegrid(basePath, participantNumber, handedness, filterBufferPeriod):
    """For a few recordings where cEEGrid data was not recorded/was of insufficient quality."""
    
    eng = matlab.engine.start_matlab()
    eng.poly52set_no_ceegrid(basePath, participantNumber, nargout=0)
    eng.quit()   
    print("cEEGrid arrays OK for " + participantNumber + ".")
    print("Stage 1 of conversion complete.")
    
    partStartEndLatencies_scalp = poly52trigs_no_ceegrid(basePath, participantNumber,
                                                         filterBufferPeriod)
    print("Stage 2 of conversion complete.")
    
    additionalData = additionalDataReader(basePath, participantNumber, handedness)
    print("Stage 3 of conversion complete.")
    
    setAndTrigs2bids_no_ceegrid(basePath, participantNumber, partStartEndLatencies_scalp,
                                additionalData)
    print("Stage 4 of conversion complete.")