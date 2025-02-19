
# imports
import sys
import pynwb
from allensdk.brain_observatory.ecephys.dynamic_gating_ecephys_session import DynamicGatingEcephysSession
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import xarray as xr
import sys
import pickle

# include code path
sys.path.append('code')

# load custom shield scripts
from shield_utils import find_animals, get_lfp_dict, downsample, align_lfp, load_animals_oi

# paths
data_path = '/oscar/data/sjones/kduecker/shield_data'
meta_path = 'externals/SHIELD_Dynamic_Gating_Analysis'

# subj_id from array job
subj_id = int(sys.argv[1])

down_srate = 500            # downsampling
roi = ['LGd', 'VISp']       # regions of interest
toi = [0, 2]                # time window around 

mice_sess = load_animals_oi()  # load subject and sesson IDs

# loop over mice here and store
subj = list(mice_sess.keys())[subj_id]

ses_files = os.listdir(os.path.join(data_path,f'sub-{subj}'))           # sessions per mouse

# get lfp files and spike files
lfp_files = list(filter(lambda s: 'None' in s, ses_files))

# load the sessions that have the ROIs
for session in mice_sess[subj]:
    ses_file = list(filter(lambda s: session in s, ses_files))

    layer_lfp = get_lfp_dict(subj, data_path, lfp_files, ses_file[0], toi, down_srate, roi)         # align and downsample LFP

    with open(os.path.join(data_path,'results_lfp_layer', f'lfp_{subj}_{session}.pkl'), 'wb') as f:
        pickle.dump(layer_lfp, f)