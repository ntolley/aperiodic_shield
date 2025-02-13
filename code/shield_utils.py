""" Functions to organize SHIELD data """

import pynwb
from allensdk.brain_observatory.ecephys.dynamic_gating_ecephys_session import DynamicGatingEcephysSession
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import signal
import xarray as xr
import json

def find_animals(meta_path, roi=['LGd', 'VISp', 'VISl']):

    sessions_metadata_table = pd.read_csv(os.path.join(meta_path,'metadata_tables','dynamic_gating_session_metadata.csv'))

    contains_all_roi = sessions_metadata_table['structure_acronyms'].apply(lambda x: all(item in x for item in roi))
    filtered_df = sessions_metadata_table[contains_all_roi & sessions_metadata_table['has_lfp']]
    mice = filtered_df['mouse_id'].unique()

    return mice

def load_animals_oi():

    # load animals of interest

    with open('session_info.json', 'r') as file:
        session_info = json.load(file)


    # reorganize a bit for ease
    mice_sess = dict()

    for session in session_info['all_sessions']:
        mice_sess[session[0:6]] = []


    for session in session_info['all_sessions']:
        mice_sess[session[0:6]].append(session[8:])

    return mice_sess



def downsample(trace, original_fs, target_fs):
    """Downsample the data with anti-aliasing filter"""
    nyq = target_fs / 2
    b, a = signal.butter(4, nyq, fs=original_fs, btype='low')
    filtered = signal.filtfilt(b, a, trace)
    down = int(original_fs / target_fs)
    downsampled = signal.decimate(filtered, down, ftype='fir')
    return downsampled

def align_lfp(lfp, trial_window, alignment_times, trial_ids = None):
    '''
    Aligns the LFP data array to experiment times of interest
    INPUTS:
        lfp: data array containing LFP data for one probe insertion
        trial_window: vector specifying the time points to excise around each alignment time
        alignment_times: experiment times around which to excise data
        trial_ids: indices in the session stim table specifying which stimuli to use for alignment.
                    None if aligning to non-stimulus times
    
    OUTPUT:
        aligned data array with dimensions channels x trials x time
    '''
    
    time_selection = np.concatenate([trial_window + t for t in alignment_times])
    
    if trial_ids is None:
        trial_ids = np.arange(len(alignment_times))
        
    inds = pd.MultiIndex.from_product((trial_ids, trial_window), 
                                      names=('presentation_id', 'time_from_presentation_onset'))

    ds = lfp.sel(time = time_selection, method='nearest').to_dataset(name = 'aligned_lfp')

    # Create xarray coordinates from the pandas MultiIndex to avoid deprecation break down
    mindex_coords = xr.Coordinates.from_pandas_multiindex(inds, 'time')
    
    # assign new coordinates and unstack
    ds = ds.assign_coords(mindex_coords).unstack('time')

    return ds['aligned_lfp']


def get_lfp_dict(subj, data_path, lfp_files, session_file, toi=[0, 1], down_srate=500, roi=['LGd', 'VISp', 'VISl']):

    """ Inputs:
    subj: subject ID
    session: session ID
    toi: time window of interest (in s) [start_time, end_time]
    """

    # initialize LFP dictionary
    layer_data = dict()

    print(f'loading session: {session_file}', flush=True)

    nwb_file_asset = pynwb.NWBHDF5IO(f'{data_path}/sub-{subj}/{session_file}', mode='r', load_namespaces=True)
    nwb_file = nwb_file_asset.read()
    dynamic_gating_session = DynamicGatingEcephysSession.from_nwb(nwb_file)

    # probe map
    probe_index = dynamic_gating_session.probes.index
    probe_map = {}
    for p in probe_index:
        probe_name = dynamic_gating_session.probes.name[p]
        filename = list(filter(lambda s: '-'+str(p)+'_' in s, lfp_files))
        probe_map[probe_name] = os.path.join(os.path.join(data_path,f'sub-{subj}'),filename[0])

    # add the LFP data to the session object
    dynamic_gating_session = DynamicGatingEcephysSession.from_nwb(nwb_file, probe_data_path_map=probe_map)

    # get the channels
    sess_units = dynamic_gating_session.get_units()

    # find the different layers in the brain area (e.g. VISpl2/3. VISl4, VISpl5)

    areas = np.unique(sess_units.structure_layer.values)
    area_layers = [name for name in areas if any(r in name for r in roi)]

    # extract layer for each unit
    layer_areas_units = dict()

    for al in area_layers:
        layer_areas_units[al] = sess_units[sess_units.structure_layer.str.contains(al)].index

    # get stimulus presentations
    stim_presentations = dynamic_gating_session.stimulus_presentations
    flashes = stim_presentations[stim_presentations['stimulus_name'].str.contains('flash')]
    presentation_times = flashes.start_time.values
    flash_end_times = presentation_times + flashes.duration
    presentation_ids = flashes.index.values

    srate = dynamic_gating_session.probes.sampling_rate.values[0]
    dt = 1/srate

    # load LFP for each probe
    for pi in probe_index:
        lfp = dynamic_gating_session.get_lfp(pi)

        # align LFP to stimulus presentations
        aligned_lfp = align_lfp(lfp, np.arange(toi[0], toi[1], dt), presentation_times, presentation_ids)

        # downsample
        if down_srate < aligned_lfp.data.shape[-1]-1/toi[1]:
            # Calculate the new number of time points after downsampling
            new_time_points = int(down_srate * toi[1] + 1)
            
            # Create a new array with the downsampled shape
            new_data = np.zeros((aligned_lfp.data.shape[0], aligned_lfp.data.shape[1], new_time_points))

            for i, channel_lfp in enumerate(aligned_lfp.data):

                # Downsample the current channel
                downsampled_channel = downsample(channel_lfp, srate, down_srate)
                
                # Assign the downsampled channel to the new array
                new_data[i] = downsampled_channel
        
            # Replace the DataArray with the new downsampled data
            aligned_lfp = xr.DataArray(
                data=new_data,
                dims=['channel', 'presentation_id', 'time'],
                coords={
                    'channel': aligned_lfp.coords['channel'],
                    'trial': aligned_lfp.coords['presentation_id'],
                    'time': np.linspace(0, toi[1], int(down_srate * toi[1] + 1))
                },
                attrs=aligned_lfp.attrs  # Preserve original attributes if needed
            )
        
        # extract lfp in the different brain areas
        lfp_channel_layer = dict()
        for layer, channels in zip(layer_areas_units.keys(), layer_areas_units.values()):
            lfp_channel_layer[layer] = [chan for chan in aligned_lfp.channel.values if chan in channels]


        for layer, channels in lfp_channel_layer.items():
            if channels:  # Only process layers with channels
                if layer not in layer_data:
                    layer_data[layer] = []
                layer_data[layer] = aligned_lfp.sel(channel=channels)

    return layer_data

def get_behav_df(subj, data_path, session_file, toi=[0, 1]):

    # load NWB file

    nwb_file_asset = pynwb.NWBHDF5IO(f'{data_path}/sub-{subj}/{session_file}', mode='r', load_namespaces=True)
    nwb_file = nwb_file_asset.read()
    dynamic_gating_session = DynamicGatingEcephysSession.from_nwb(nwb_file)

