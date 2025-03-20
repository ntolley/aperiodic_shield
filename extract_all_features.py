# packages
import pynwb
from allensdk.brain_observatory.ecephys.dynamic_gating_ecephys_session import DynamicGatingEcephysSession
from spike_utils import get_good_units_with_structure, maxInterval, get_burst_times_dict, count_bursts_and_spikes_after_stim, add_metadata_to_df

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import xarray as xr
import sys
import pickle

from pathlib import Path
import pandas as pd
import glob

# Add the path you want to include
sys.path.append('code')
# load custom scripts
from shield_utils import find_animals, get_lfp_dict, downsample, align_lfp, load_animals_oi
from tfr_utils import compute_tfr

def apply_specparam(spectra, freqs, specparam_settings, n_jobs=-1):
    """
    Apply spectral parameterization to 3D array of power spectra.

    Parameters
    ----------
    spectra : 3d array
        Power spectra, with dimensions [n_conditions, n_spectra, n_freqs].
    freqs : 1d array
        Frequency values for the power spectra.
    specparam_settings : dict
        Settings for the spectral parameterization.
    n_jobs : int
        Number of parallel jobs to run.

    Returns
    -------
    exponent : 2d array
        Aperiodic exponents for each power spectrum. Dimensions are 
        [n_spectra, n_times].
    """
    
    # imports
    from specparam import SpectralGroupModel
    from specparam.objs import fit_models_3d, combine_model_objs
    
    sm = SpectralGroupModel(**specparam_settings, verbose=False)
    sgm = fit_models_3d(sm, freqs, spectra, n_jobs=n_jobs)
    sgm = combine_model_objs(sgm)
    
    return sgm


def compute_flattened_spectra(sgm):
    """
    Compute flattened power spectra from SpectralGroupModel object.

    Parameters
    ----------
    sgm : SpectralGroupModel object
        SpectralGroupModel object.

    Returns
    -------
    spectra_flat : 2d array
        Flattened power spectra.    
    """
    
    spectra_list = []
    for ii in range(len(sgm)):
        sm = sgm.get_model(ii)
        spectra_list.append(sm.power_spectrum - sm._ap_fit)
    spectra_flat = np.array(spectra_list)
    
    return spectra_flat


if __name__ == "__main__":
    data_path = '/oscar/data/sjones/kduecker/shield_data/'
    meta_path = 'externals/SHIELD_Dynamic_Gating_Analysis'
    save_path = '/oscar/data/sjones/ntolley/shield_data'
    lfp_data_path = '/oscar/data/sjones/ntolley/shield_data/results_lfp_layer'


    toi = [0, 2]                # time window around flash onset

    window_duration = 1.0 # used for spike detection (should it be longer to match behavior?)

    # Define burst parameters (same as before)
    burst_params = {
        'max_begin_ISI': 0.004,  # 4 ms
        'max_end_ISI': 0.02,     # 20 ms
        'min_IBI': 0.1,          # 100 ms
        'min_burst_duration': 0.008,  # 8 ms
        'min_spikes_in_burst': 3,
        'pre_burst_silence': 0.1  # 100 ms
    }

    FS_LFP = 1250 # approximate sampling frequency for LFP
    # spectrogram hyperparameters
    FREQS = [10, 100, 128] # [start, stop, n_freqs] (Hz)
    N_CYCLES = 5 # for Morlet decomp

    # SpecParam hyperparameters
    SPECPARAM_SETTINGS = {
        'peak_width_limits' :   [2, 20], # default : (0.5, 12.0) - recommends at least frequency resolution * 2
        'min_peak_height'   :   0, # default : 0
        'max_n_peaks'       :   4, # default : inf
        'peak_threshold'    :   3, # default : 2.0
        'aperiodic_mode'    :   'knee'} # 'fixed' or 'knee'
    N_JOBS = 20 # number of jobs for parallel processing

    mice_sess = load_animals_oi()       # animals of interest

    # load the sessions that have the ROIs
    for subj in list(mice_sess.keys()):
        ses_files = os.listdir(os.path.join(data_path,f'sub-{subj}'))
        print(f'Subject {subj}')

        for session in mice_sess[subj]:
            print(f'loading session: {session}', flush=True)
            ses_file = list(filter(lambda s: session in s, ses_files))

            # Load LFP data
            lfp_fname = f'{lfp_data_path}/lfp_{subj}_{session}.pkl'
            with open(lfp_fname, 'rb') as f:
                lfp_dict = pickle.load(f)

            subject_id = f'sub-{subj}'
            nwb_file_asset = pynwb.NWBHDF5IO(f'{data_path}/{subject_id}/{ses_file[0]}', mode='r', load_namespaces=True)
            nwb_file = nwb_file_asset.read()
            dynamic_gating_session = DynamicGatingEcephysSession.from_nwb(nwb_file)

            # extract running, blinks, licks
            eye_tracking = dynamic_gating_session.eye_tracking
            running_speed = dynamic_gating_session.running_speed
            licks = dynamic_gating_session.licks

            # get stimulus presentations
            stim_presentations = dynamic_gating_session.stimulus_presentations
            flashes = stim_presentations[stim_presentations['stimulus_name'].str.contains('flash')]
            presentation_times = flashes.start_time.values
            flash_end_times = presentation_times + flashes.duration
            presentation_ids = flashes.index.values

            # Get good units with structure info
            good_units_visual, spike_times_dict, structure_dict = get_good_units_with_structure(dynamic_gating_session)
            unique_structures = set(structure_dict.values())
            
            # Get burst times
            burst_times_dict = get_burst_times_dict(spike_times_dict, burst_params)

            # Columns of dataframe
            trial_list = list()
            subject_id_list = list()
            session_list = list()
            mean_speed = list()
            num_licks = list()
            num_rewards = list()
            num_blinks = list()
            pupil_dil = list()
            onset_time_list = list()
            structure_list = list()
            spike_count_list = list()
            burst_count_list = list()
            exponent_list = list()
            spectra_flat_list = list()
            

            # Iterate over presentation times to get behavior
            for trial, trial_start in enumerate(presentation_times):
                # mean running speed
                flash_running = running_speed.query('timestamps >= {} and timestamps <= {} '.format(trial_start-toi[0], trial_start+toi[1]))
                # number of licks
                flash_licks = licks.query('timestamps >= {} and timestamps <= {} '.format(trial_start-toi[0], trial_start+toi[1]))
                # check if this makes sense: rewards?
                reward_time = dynamic_gating_session.rewards.query('timestamps >= {} and timestamps <= {} '.
                                                format(trial_start-toi[0], trial_start+toi[1]))['timestamps']

                # number of blinks
                eye_blinks = eye_tracking[eye_tracking['likely_blink']].query('timestamps >= {} and timestamps <= {} '.format(trial_start-toi[0], trial_start+toi[1]))

                # dilation
                eye_trial = eye_tracking.query('timestamps >= {} and timestamps <= {} '.format(trial_start-toi[0], trial_start+toi[1]))

                window_start = trial_start
                window_end = trial_start + window_duration
                
                # Initialize structure-wise counters for this presentation
                for structure in unique_structures:
                    spike_count = 0
                    burst_count = 0
                    
                    # Count spikes for all units of this structure
                    for unit_id, spikes in spike_times_dict.items():
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
                    trial_list.append(trial)
                    mean_speed.append(np.mean(flash_running.speed.values))
                    num_licks.append(len(flash_licks))
                    num_rewards.append(len(reward_time))
                    num_blinks.append(len(eye_blinks))
                    pupil_dil.append(np.mean(eye_trial.pupil_width.values))

                    onset_time_list.append(trial_start)
                    structure_list.append(structure)
                    spike_count_list.append(spike_count)
                    burst_count_list.append(burst_count)
                    subject_id_list.append(subject_id)
                    session_list.append(session)

            for structure in unique_structures:
                # Extract specparams features
                if structure in lfp_dict.keys():
                    lfp_epochs = lfp_dict[structure]
                    lfp_epochs = np.expand_dims(lfp_epochs, 0)

                    # compute tfr
                    tfr, tfr_freqs = compute_tfr(lfp_epochs, FS_LFP, FREQS, method='morlet', 
                                                    n_morlet_cycle=N_CYCLES, n_jobs=N_JOBS)

                    tfr = np.mean(tfr, axis=0) # average over channels
                    tfr = np.mean(tfr, axis=2)
                    # parameterize spectra, compute aperiodic exponent and total power
                    sgm = apply_specparam(tfr[None,:,:], tfr_freqs, SPECPARAM_SETTINGS, N_JOBS)
                    exponent = sgm.get_params('aperiodic', 'exponent')
                    exponent_list.extend(exponent.squeeze())

                    spectra_flat = compute_flattened_spectra(sgm)
                    spectra_flat = [spectra_flat[idx, :] for idx in range(spectra_flat.shape[0])]
                    spectra_flat_list.extend(spectra_flat)
                else:
                    exponent_list.extend(np.repeat(np.nan, len(presentation_times)))
                    spectra_flat_list.extend(np.repeat(np.nan, len(presentation_times)))


        #     break
        # break

    df_dict = {
        'trial': trial_list,
        'subject_id': subject_id_list,
        'session': session_list,
        'onset_time': onset_time_list,
        'mean_speed': mean_speed,
        'num_licks': num_licks,
        'num_rewards': num_rewards,
        'num_blinks': num_blinks,
        'pupil_dil': pupil_dil,
        'structure': structure_list,
        'spike_count': spike_count_list,
        'burst_count': burst_count_list,
        'exponent': exponent_list,
        'spectra_flat': spectra_flat_list
    }
    df = pd.DataFrame(df_dict)

    df.to_csv('all_features.csv')
