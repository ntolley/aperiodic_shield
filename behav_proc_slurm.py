
# packages
import pynwb
from allensdk.brain_observatory.ecephys.dynamic_gating_ecephys_session import DynamicGatingEcephysSession
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import xarray as xr
import sys
import pickle

# Add the path you want to include
sys.path.append('code')
# load custom scripts
from shield_utils import find_animals, get_lfp_dict, downsample, align_lfp, load_animals_oi


data_path = '/oscar/data/sjones/kduecker/shield_data/'
meta_path = 'externals/SHIELD_Dynamic_Gating_Analysis'

toi = [0, 2]                # time window around flash onset

mice_sess = load_animals_oi()       # animals of interest

subj_id = int(sys.argv[1])
print(subj_id, flush=True)
# current subject (ID from array job)
subj = list(mice_sess.keys())[subj_id]


# loop over sessions
ses_files = os.listdir(os.path.join(data_path,f'sub-{subj}'))           # sessions per mouse

# load the sessions that have the ROIs
for session in mice_sess[subj]:

    print(f'loading session: {session}', flush=True)
    ses_file = list(filter(lambda s: session in s, ses_files))

    nwb_file_asset = pynwb.NWBHDF5IO(f'{data_path}/sub-{subj}/{ses_file[0]}', mode='r', load_namespaces=True)
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

    mean_speed = np.zeros_like(presentation_times)
    num_licks = np.zeros_like(presentation_times)
    num_rewards = np.zeros_like(presentation_times)
    num_blinks = np.zeros_like(presentation_times)
    pupil_dil = np.zeros_like(presentation_times)

    for i,trial_start in enumerate(presentation_times):

        # mean running speed
        flash_running = running_speed.query('timestamps >= {} and timestamps <= {} '.format(trial_start-toi[0], trial_start+toi[1]))
        mean_speed[i] = np.mean(flash_running.speed.values)
        # number of licks
        flash_licks = licks.query('timestamps >= {} and timestamps <= {} '.format(trial_start-toi[0], trial_start+toi[1]))
        num_licks[i] = len(flash_licks)
        # check if this makes sense: rewards?
        reward_time = dynamic_gating_session.rewards.query('timestamps >= {} and timestamps <= {} '.
                                        format(trial_start-toi[0], trial_start+toi[1]))['timestamps']

        num_rewards[i] = len(reward_time)

        # number of blinks
        eye_blinks = eye_tracking[eye_tracking['likely_blink']].query('timestamps >= {} and timestamps <= {} '.format(trial_start-toi[0], trial_start+toi[1]))
        num_blinks[i] = len(eye_blinks)

        # dilation
        eye_trial = eye_tracking.query('timestamps >= {} and timestamps <= {} '.format(trial_start-toi[0], trial_start+toi[1]))
        pupil_dil[i] = np.mean(eye_trial.pupil_width.values)

    df = pd.DataFrame()
    df['mean_speed'] = mean_speed
    df['num_licks'] = num_licks
    df['num_rewards'] = num_rewards
    df['num_blinks'] = num_blinks
    df['pupil_dil'] = num_blinks


    df.to_csv(os.path.join(data_path,'results_behavior',f'behav_{subj}_{session}.csv'))
