import pynwb
from allensdk.brain_observatory.ecephys.dynamic_gating_ecephys_session import DynamicGatingEcephysSession
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import os
from pathlib import Path


def get_good_units_with_structure(dynamic_gating_session, structures=('LGd', 'VISp', 'VISl')):
    """
    Same as get_good_units but returns structure information.
    
    Returns:
    --------
    tuple:
        - pandas.DataFrame: DataFrame containing filtered good units from specified structures
        - dict: Dictionary of spike times for each good unit
        - dict: Dictionary mapping unit_ids to their structure acronyms
    """
    # Original filtering code
    units = dynamic_gating_session.get_units()
    channels = dynamic_gating_session.get_channels()
    
    channels = channels[
        (channels['structure_acronym'] != 'out of brain') & 
        (channels['structure_acronym'] != 'No Area') &
        (channels['structure_acronym'] != 'Track not annotated')
    ]
    
    units_channels = units.merge(channels, left_on='peak_channel_id', right_index=True)
    units_channels = units_channels.sort_values('probe_vertical_position', ascending=False)
    
    good_unit_filter = (
        (units_channels['snr'] > 1) &
        (units_channels['isi_violations'] < 1) &
        (units_channels['firing_rate'] > 0.1)
    )
    
    good_units = units_channels.loc[good_unit_filter]
    
    structure_filter = good_units['structure_acronym'].str.startswith(structures)
    good_units_visual = good_units[structure_filter]
    
    # Get spike times and structure info
    spike_times_dict = {}
    structure_dict = {}
    for unit_id in good_units_visual.index:
        spike_times_dict[unit_id] = dynamic_gating_session.spike_times[unit_id]
        structure_dict[unit_id] = good_units_visual.loc[unit_id, 'structure_acronym']
    
    return good_units_visual, spike_times_dict, structure_dict



def maxInterval(spiketrain, max_begin_ISI, max_end_ISI, min_IBI, min_burst_duration,
                min_spikes_in_burst, pre_burst_silence=0.1):
    
    allBurstData = {}
    '''
    Phase 1 - Burst Detection
    Here a burst is defined as starting when two consecutive spikes have an
    ISI less than max_begin_ISI apart, and there's at least pre_burst_silence
    of no spikes before the burst. The end of the burst is given when two
    spikes have an ISI greater than max_end_ISI.
    '''
    inBurst = False
    burstNum = 0
    currentBurst = []
    last_spike_time = -np.inf  # Initialize to negative infinity

    for n in range(1, len(spiketrain)):
        ISI = spiketrain[n] - spiketrain[n - 1]
        time_since_last_spike = spiketrain[n - 1] - last_spike_time

        if inBurst:
            if ISI > max_end_ISI:  # end the burst
                currentBurst = np.append(currentBurst, spiketrain[n - 1])
                allBurstData[burstNum] = currentBurst
                currentBurst = []
                burstNum += 1
                inBurst = False
                last_spike_time = spiketrain[n - 1]
            elif (ISI < max_end_ISI) & (n == len(spiketrain) - 1):
                currentBurst = np.append(currentBurst, spiketrain[n])
                allBurstData[burstNum] = currentBurst
                burstNum += 1
            else:
                currentBurst = np.append(currentBurst, spiketrain[n - 1])
        else:
            if ISI < max_begin_ISI and time_since_last_spike >= pre_burst_silence:
                currentBurst = np.append(currentBurst, spiketrain[n - 1])
                inBurst = True
            else:
                last_spike_time = spiketrain[n - 1]

    # Calculate IBIs
    IBI = []
    for b in range(1, burstNum):
        prevBurstEnd = allBurstData[b - 1][-1]
        currBurstBeg = allBurstData[b][0]
        IBI = np.append(IBI, (currBurstBeg - prevBurstEnd))

    '''
    Phase 2 - Merging of Bursts
    Here we see if any pair of bursts have an IBI less than min_IBI; if so,
    we then merge the bursts. We specifically need to check when say three
    bursts are merged into one.
    '''
    tmp = allBurstData
    allBurstData = {}
    burstNum = 0
    for b in range(1, len(tmp)):
        prevBurst = tmp[b - 1]
        currBurst = tmp[b]
        if IBI[b - 1] < min_IBI:
            prevBurst = np.append(prevBurst, currBurst)
        allBurstData[burstNum] = prevBurst
        burstNum += 1
    if burstNum >= 2:
        allBurstData[burstNum] = currBurst

    '''
    Phase 3 - Quality Control
    Remove small bursts less than min_bursts_duration or having too few
    spikes less than min_spikes_in_bursts. In this phase we have the
    possibility of deleting all spikes.
    '''
    tooShort = 0
    tmp = allBurstData
    allBurstData = {}
    burstNum = 0
    if len(tmp) > 1:
        for b in range(len(tmp)):
            currBurst = tmp[b]
            if len(currBurst) <= min_spikes_in_burst:
                tooShort +=1
            elif currBurst[-1] - currBurst[0] <= min_burst_duration:
                tooShort += 1
            else:
                allBurstData[burstNum] = currBurst
                burstNum += 1

    return allBurstData, tooShort


def get_burst_times_dict(spike_times, burst_params):
    """Same as before, no changes needed as it just processes spike times"""
    burst_times_dict = {}
    
    for unit_id, spikes in spike_times.items():
        if len(spikes) >= burst_params['min_spikes_in_burst']:
            burst_data, _ = maxInterval(
                spikes,
                burst_params['max_begin_ISI'],
                burst_params['max_end_ISI'],
                burst_params['min_IBI'],
                burst_params['min_burst_duration'],
                burst_params['min_spikes_in_burst'],
                burst_params['pre_burst_silence']
            )
            
            if burst_data:
                burst_times = [(burst[0], burst[-1]) for burst in burst_data.values()]
                burst_times_dict[unit_id] = burst_times
            else:
                burst_times_dict[unit_id] = []
    
    return burst_times_dict

def count_bursts_and_spikes_after_stim(burst_times_dict, spike_times, structure_dict, presentation_times, window_duration=1.0):
    """
    Count bursts and spikes after each stimulus presentation, organized by trial.
    
    Returns:
    --------
    pandas.DataFrame with columns:
        - trial: stimulus presentation number
        - onset_time: stimulus presentation time
        - structure: brain structure
        - spike_count: number of spikes in the window
        - burst_count: number of bursts in the window
    """
    # Get unique structures
    unique_structures = set(structure_dict.values())
    
    # Initialize list to store all rows
    rows = []
    
    # For each stimulus presentation
    for trial, pres_time in enumerate(presentation_times):
        window_start = pres_time
        window_end = pres_time + window_duration
        
        # Initialize structure-wise counters for this presentation
        for structure in unique_structures:
            spike_count = 0
            burst_count = 0
            
            # Count spikes for all units of this structure
            for unit_id, spikes in spike_times.items():
                if structure_dict[unit_id] == structure:
                    spikes = np.array(spikes)
                    window_mask = (spikes >= window_start) & (spikes < window_end)
                    spike_count += np.count_nonzero(window_mask)
            
            # Count bursts for all units of this structure
            for unit_id, bursts in burst_times_dict.items():
                if structure_dict[unit_id] == structure:
                    burst_count += sum(1 for burst_start, burst_end in bursts 
                                     if window_start <= burst_start < window_end)
            
            # Add row for this structure in this trial
            rows.append({
                'trial': trial,
                'onset_time': pres_time,
                'structure': structure,
                'spike_count': spike_count,
                'burst_count': burst_count
            })
    
    # Create DataFrame
    counts_df = pd.DataFrame(rows)
    
    return counts_df

def add_metadata_to_df(df, metadata, session, subject_id):
    # Create a copy of the metadata dict to avoid modifying the original
    meta_dict = metadata.copy()
    
    # Convert datetime to string to avoid serialization issues
    if 'date_of_acquisition' in meta_dict:
        meta_dict['date_of_acquisition'] = str(meta_dict['date_of_acquisition'])
    
    # Convert UUID to string
    if 'behavior_session_uuid' in meta_dict:
        meta_dict['behavior_session_uuid'] = str(meta_dict['behavior_session_uuid'])
    
    # Convert list to string for driver_line
    if 'driver_line' in meta_dict:
        meta_dict['driver_line'] = ','.join(meta_dict['driver_line'])
    
    # Add each metadata field as a column
    for key, value in meta_dict.items():
        df[key] = value
    
    # Add the additional variables
    df['session'] = session
    df['subject_id'] = subject_id
    
    return df