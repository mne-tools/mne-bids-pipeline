"""ds003104: Somatosensory.

See [OpenNeuro](https://openneuro.org/datasets/ds003104) for more information.
"""

bids_root = "~/mne_data/ds003104"
deriv_root = "~/mne_data/derivatives/mne-bids-pipeline/ds003104"
subjects_dir = f"{bids_root}/derivatives/freesurfer/subjects"

conditions = ["somato_event1"]
ch_types = ["meg"]

# shorten the raw file for speed
crop_runs = (0, 20)

# use oversampled temporal projection to clean sensor noise
use_otp_denoising = True
otp_duration = 10.0
