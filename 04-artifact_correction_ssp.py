"""
===========
05. Run ICA
===========

ICA decomposition using fastICA.
"""

import os.path as op

import mne
from mne.parallel import parallel_func

from config import meg_dir, N_JOBS


def run_ssp(subject_id, tsss=None):
    subject = "sub%03d" % subject_id
    print("Processing subject: %s%s"
          % (subject, (' (tSSS=%d)' % tsss) if tsss else ''))
    data_path = op.join(meg_dir, subject)
    raws = list()
    print("  Loading runs")
    for run in range(1, 7):
        if tsss:
            run_fname = op.join(data_path, 'run_%02d_filt_tsss_%d_raw.fif'
                                % (run, tsss))
        else:
            run_fname = op.join(data_path, 'run_%02d_filt_sss_highpass-%sHz'
                                '_raw.fif' % (run, 1))
        raws.append(mne.io.read_raw_fif(run_fname))
    raw = mne.concatenate_raws(raws)

    # XXX


# Memory footprint: around n_jobs * 4 GB
parallel, run_func, _ = parallel_func(run_ssp, n_jobs=N_JOBS)
parallel(run_func(subject_id) for subject_id in range(1, 20))
parallel(run_func(3, tsss) for tsss in (10, 1))  # Maxwell filtered data
