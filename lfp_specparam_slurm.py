import sys
sys.path.append('code/')

import os
from shield_utils import get_lfp_dict
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tfr_utils import compute_tfr

lfp_data_path = '/oscar/data/sjones/kduecker/shield_data/results_lfp_layer'
lfp_files = os.listdir(lfp_data_path)

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
    num_trials = 150
    session_list, area_list, exponent_list = list(), list(), list()

    for lfp_fname in lfp_files:
        fname = f'{lfp_data_path}/{lfp_fname}'
        with open(fname, 'rb') as f:
            lfp_dict = pickle.load(f)

        area_names = list(lfp_dict.keys())
        for area in area_names:
            lfp_epochs = lfp_dict[area]
            session_list.extend(np.repeat(lfp_fname, num_trials))
            area_list.extend(np.repeat(area, num_trials))

            # compute tfr
            tfr, tfr_freqs = compute_tfr(lfp_epochs, FS_LFP, FREQS, method='morlet', 
                                            n_morlet_cycle=N_CYCLES, n_jobs=N_JOBS)

            tfr = np.mean(tfr, axis=0) # average over channels
            tfr = np.mean(tfr, axis=2)
            # parameterize spectra, compute aperiodic exponent and total power
            sgm = apply_specparam(tfr[None,:,:], tfr_freqs, SPECPARAM_SETTINGS, N_JOBS)
            exponent = sgm.get_params('aperiodic', 'exponent')
            exponent_list.extend(exponent)

        


