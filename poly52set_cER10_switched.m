function poly52set(basePath, participantNumber)
    
    if class(basePath) == 'char'
        basePath = convertCharsToStrings(basePath);
    end

    if class(participantNumber) == 'char'
        participantNumber = convertCharsToStrings(participantNumber);
    end

    rawDataPath = basePath + '/sourcedata/P' + participantNumber + '/';

    fileName = 'P' + participantNumber + '_scalp';
    fileNamePath_full = rawDataPath + fileName + '.Poly5';
    fileNamePath_full = convertStringsToChars(fileNamePath_full);

    ChannelLabels = {'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7',...
    'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2'};

    addpath('C:\Users\cjwin\OneDrive - Queen Mary, University of London\Documents\eeglab2023.0 MINIMAL PLUGINS_old')
    EEG_scalp = pop_loadpoly5_2('filepath','','filename',fileNamePath_full,'ref_ch',[1:32],'NaNdiscSamples',true,'ChannelsDiscThd',1,'EnableSaturationDisc',true,'EnableSameValueDisc',false,'ChannelLabels',ChannelLabels);
    %Need to set to average reference. This is without irrelevant chans (and discarded chans not included), and also more accurate than the average referencing done within the Mobita.

    EEG_scalp=pop_select(EEG_scalp,'channel',1:32);
    
    %save as EEGLAB struct (useful for Python work):
    setFilePath = char(basePath + 'EEG Set Files (Unprocessed)\');
    outputFileName= char('P' + participantNumber + '_scalp.set');
    pop_saveset(EEG_scalp, outputFileName, setFilePath);
    

    %Now something very similar for the ceegrid data:
    fileName = 'P' + participantNumber + '_ceegrid';
    fileNamePath_full = rawDataPath + fileName + '.Poly5';
    fileNamePath_full = convertStringsToChars(fileNamePath_full);

    %ceegrid - Fpz is reference:
    ChannelLabels = {'cEL1', 'cEL2', 'cEL3', 'cEL4', 'cEL5', 'cEL6', 'cEL7', 'cEL8','cEL9','cEL10','Fpz', '', '', '', '', '', ...
                     'cER1', 'cER2', 'cER3', 'cER4', 'cER5', 'cER6', 'cER7', 'cER8', 'cER9', '', '','','','','','cER10'};


    EEG_ceegrid = pop_loadpoly5_2('filepath','','filename',fileNamePath_full,'ref_ch',[1:11, 17:25, 32],'NaNdiscSamples',true,'ChannelsDiscThd',1,'EnableSaturationDisc',true,'EnableSameValueDisc',false,'ChannelLabels',ChannelLabels);

    %We remove empty channels:
    EEG_ceegrid=pop_select(EEG_ceegrid,'channel',[1:11, 17:25, 32]);
    
    %save as EEGLAB struct (useful for Python work):
    setFilePath = char(basePath + 'EEG Set Files (Unprocessed)\');
    outputFileName= char('P' + participantNumber + '_ceegrid.set');
    pop_saveset(EEG_ceegrid, outputFileName, setFilePath);
end