"""
============================================
02. Extract events from the stimulus channel
============================================

The events are extracted from stimulus channel 'STI101'. The events are saved
to the subject's MEG directory.
"""

import os.path as op

import mne
from mne.parallel import parallel_func

from library.config import study_path, meg_dir, N_JOBS


def run_events(subject_id):
    subject = "sub%03d" % subject_id
    print("processing subject: %s" % subject)
    in_path = op.join(study_path, 'ds117', subject, 'MEG')
    out_path = op.join(meg_dir, subject)
    for run in range(1, 7):
        run_fname = op.join(in_path, 'run_%02d_raw.fif' % (run,))
        raw = mne.io.read_raw_fif(run_fname)
        mask = 4096 + 256  # mask for excluding high order bits
        events = mne.find_events(raw, stim_channel='STI101',
                                 consecutive='increasing', mask=mask,
                                 mask_type='not_and', min_duration=0.003)

        print("  S %s - R %s" % (subject, run))

        fname_events = op.join(out_path, 'run_%02d-eve.fif' % run)
        mne.write_events(fname_events, events)


parallel, run_func, _ = parallel_func(run_events, n_jobs=N_JOBS)
parallel(run_func(subject_id) for subject_id in range(1, 20))
