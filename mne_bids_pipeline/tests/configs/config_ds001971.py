"""Mobile brain body imaging (MoBI) gait adaptation experiment.

See ds001971 on OpenNeuro: https://github.com/OpenNeuroDatasets/ds001971
"""

from mne_bids_pipeline.typing import ArbitraryContrast

bids_root = "~/mne_data/ds001971"
deriv_root = "~/mne_data/derivatives/mne-bids-pipeline/ds001971"

task = "AudioCueWalkingStudy"
interactive = False
ch_types = ["eeg"]
reject = {"eeg": 150e-6}
conditions = ["AdvanceTempo", "DelayTempo"]
contrasts = [
    ArbitraryContrast(
        name="AdvanceMinusDelay",
        conditions=["AdvanceTempo", "DelayTempo"],
        weights=[1, -1],
    ),
]

subjects = ["001"]
runs = ["01"]
epochs_decim = 5  # to 100 Hz

# This is mostly for testing purposes!
decode = True
decoding_time_generalization = True
decoding_time_generalization_decim = 2
decoding_csp = True
decoding_csp_freqs = {
    "beta": [13, 20, 30],
}
decoding_csp_times = [-0.2, 0.0, 0.2, 0.4]

# Just to test that MD5 works
memory_file_method = "hash"
