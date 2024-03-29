#%%
import mne
from mne.minimum_norm import (
    apply_inverse,
    make_inverse_operator,
    write_inverse_operator,
)
from mne_bids import BIDSPath
#%%
# noise cov file
fname_info = '/Volumes/neurospin/meg/meg_tmp/TimeInWM_Izem_2019/BIDS_anonymized/derivatives/sub-748/meg/sub-748_task-rest_proc-clean_raw.fif'
fname_cov = '/Volumes/neurospin/meg/meg_tmp/TimeInWM_Izem_2019/BIDS_anonymized/derivatives/sub-748/meg/sub-748_task-rest_proc-clean_cov.fif'
fname_fwd = '/Volumes/neurospin/meg/meg_tmp/TimeInWM_Izem_2019/BIDS_anonymized/derivatives/sub-748/meg/sub-748_task-tiwm_fwd.fif'

info = mne.io.read_info(fname_info)
cov = mne.read_cov(fname_cov)
forward = mne.read_forward_solution(fname_fwd)

#%%
inverse_operator = make_inverse_operator(
        info, forward, cov, loose=cfg.loose, depth=cfg.depth, rank="info"

#%%
import config_sophie_mac as config

#%%
bids_path = BIDSPath(
        subject='sub-748',
        session='',
        task=cfg.task,
        acquisition=cfg.acq,
        run=None,
        recording=cfg.rec,
        space=cfg.space,
        extension=".fif",
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False,
    )
# %%
