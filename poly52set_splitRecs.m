function poly52set_splitRecs(basePath, participantNumber, rec1, rec2)
    
    if class(basePath) == 'char'
        basePath = convertCharsToStrings(basePath);
    end

    if class(participantNumber) == 'char'
        participantNumber = convertCharsToStrings(participantNumber);
    end

    if class(rec1) == 'char'
        rec1 = convertCharsToStrings(rec1);
    end

    if class(rec2) == 'char'
        rec2 = convertCharsToStrings(rec2);
    end

    rawDataPath = basePath + '/sourcedata/P' + participantNumber + '/';
    
    %Scalp:

    %First recording:
    fileName_rec1 = 'P' + participantNumber + '_' + rec1 +  '_scalp';
    fileNamePath_full = rawDataPath + fileName_rec1 + '.Poly5';
    fileNamePath_full = convertStringsToChars(fileNamePath_full);

    ChannelLabels = {'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'M1', 'T7',...
    'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2'}; 
    
    addpath('C:\Users\cjwin\OneDrive - Queen Mary, University of London\Documents\eeglab2023.0 MINIMAL PLUGINS_old')
    EEG_scalp_rec1 = pop_loadpoly5_2('filepath','','filename',fileNamePath_full,'ref_ch',[1:32],'NaNdiscSamples',true,'ChannelsDiscThd',1,'EnableSaturationDisc',true,'EnableSameValueDisc',false,'ChannelLabels',ChannelLabels);
    %Need to set to average reference. This is without irrelevant chans (and discarded chans not included), and also more accurate than the average referencing done within the Mobita.
    
    EEG_scalp_rec1 = pop_select(EEG_scalp_rec1,'channel',1:32);

    %Second recording:
    fileName_rec2 = 'P' + participantNumber + '_' + rec2 +  '_scalp';
    fileNamePath_full = rawDataPath + fileName_rec2 + '.Poly5';
    fileNamePath_full = convertStringsToChars(fileNamePath_full);

    EEG_scalp_rec2 = pop_loadpoly5_2('filepath','','filename',fileNamePath_full,'ref_ch',[1:32],'NaNdiscSamples',true,'ChannelsDiscThd',1,'EnableSaturationDisc',true,'EnableSameValueDisc',false,'ChannelLabels',ChannelLabels);
    EEG_scalp_rec2 = pop_select(EEG_scalp_rec2,'channel',1:32);

    %Merge the two together, and then save:
    EEG_merged_scalp = pop_mergeset(EEG_scalp_rec1, EEG_scalp_rec2);

    setFilePath = char(basePath + 'EEG Set Files (Unprocessed)\');
    outputFileName = char('P' + participantNumber + '_scalp.set');
    pop_saveset(EEG_merged_scalp, outputFileName, setFilePath);

    

    %Now something very similar for the ceegrid data:
    fileName_rec1 = 'P' + participantNumber + '_' + rec1 +  '_ceegrid';
    fileNamePath_full = rawDataPath + fileName_rec1 + '.Poly5';
    fileNamePath_full = convertStringsToChars(fileNamePath_full);

    %Fpz is reference:
    ChannelLabels = {'cEL1', 'cEL2', 'cEL3', 'cEL4', 'cEL5', 'cEL6', 'cEL7', 'cEL8','cEL9','cEL10','Fpz', '', '', '', '', '', ...
                     'cER1', 'cER2', 'cER3', 'cER4', 'cER5', 'cER6', 'cER7', 'cER8', 'cER9', 'cER10', '','','','','',''};
    
    EEG_ceegrid_rec1 = pop_loadpoly5_2('filepath','','filename',fileNamePath_full,'ref_ch',[1:11, 17:26],'NaNdiscSamples',true,'ChannelsDiscThd',1,'EnableSaturationDisc',true,'EnableSameValueDisc',false,'ChannelLabels',ChannelLabels);
    EEG_ceegrid_rec1 = pop_select(EEG_ceegrid_rec1,'channel',[1:11, 17:26]);
    
    %Second recording:
    fileName_rec2 = 'P' + participantNumber + '_' + rec2 +  '_ceegrid';
    fileNamePath_full = rawDataPath + fileName_rec2 + '.Poly5';
    fileNamePath_full = convertStringsToChars(fileNamePath_full);

    EEG_ceegrid_rec2 = pop_loadpoly5_2('filepath','','filename',fileNamePath_full,'ref_ch',[1:11, 17:26],'NaNdiscSamples',true,'ChannelsDiscThd',1,'EnableSaturationDisc',true,'EnableSameValueDisc',false,'ChannelLabels',ChannelLabels);
    EEG_ceegrid_rec2 = pop_select(EEG_ceegrid_rec2,'channel',[1:11, 17:26]);

    %Merge the two together, and then save:
    EEG_merged_ceegrid = pop_mergeset(EEG_ceegrid_rec1, EEG_ceegrid_rec2);

    setFilePath = char(basePath + 'EEG Set Files (Unprocessed)\');
    outputFileName= char('P' + participantNumber + '_ceegrid.set');
    pop_saveset(EEG_merged_ceegrid, outputFileName, setFilePath);
end