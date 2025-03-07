"""MNE Sample Data: M/EEG combined processing."""

import mne
import mne_bids

bids_root = "~/mne_data/ds000248"
deriv_root = "~/mne_data/derivatives/mne-bids-pipeline/ds000248_base"
subjects_dir = f"{bids_root}/derivatives/freesurfer/subjects"

subjects = ["01"]
rename_events = {"Smiley": "Emoji", "Button": "Switch"}
conditions = ["Auditory", "Visual", "Auditory/Left", "Auditory/Right"]
epochs_metadata_query = "index > 0"  # Just for testing!
contrasts = [("Visual", "Auditory"), ("Auditory/Right", "Auditory/Left")]

time_frequency_conditions = ["Auditory", "Visual"]

ch_types = ["meg", "eeg"]
mf_reference_run = "01"
find_flat_channels_meg = True
find_noisy_channels_meg = True
use_maxwell_filter = True


def noise_cov(bp: mne_bids.BIDSPath) -> mne.Covariance:
    """Estimate the noise covariance."""
    # Use pre-stimulus period as noise source
    bp = bp.copy().update(suffix="epo")
    if not bp.fpath.exists():
        bp.update(split="01")
    epo = mne.read_epochs(bp)
    cov = mne.compute_covariance(epo, rank="info", tmax=0)
    return cov


spatial_filter = "ssp"
n_proj_eog = dict(n_mag=1, n_grad=1, n_eeg=1)
n_proj_ecg = dict(n_mag=1, n_grad=1, n_eeg=0)
ssp_meg = "combined"
ecg_proj_from_average = True
eog_proj_from_average = False
epochs_decim = 4

bem_mri_images = "FLASH"
recreate_bem = True

n_jobs = 2


def mri_t1_path_generator(bids_path: mne_bids.BIDSPath) -> mne_bids.BIDSPath:
    """Return the path to a T1 image."""
    # don't really do any modifications – just for testing!
    return bids_path
