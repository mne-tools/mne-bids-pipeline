"""
===========
05. Run ICA
===========

ICA decomposition using fastICA.
"""

import os.path as op

import mne
from mne.preprocessing import ICA
from mne.parallel import parallel_func

import config

# Here we always process with the 1 Hz highpass data (instead of using
# l_freq) because ICA needs a highpass.


def run_ica(subject, tsss=None):
    print("Processing subject: %s%s"
          % (subject, (' (tSSS=%d)' % tsss) if tsss else ''))
    meg_subject_dir = op.join(config.meg_dir, subject)
    raw_fnames = op.join(meg_subject_dir, '%s_audvis_filt_sss_raw.fif' % subject)
    print("  Loading runs")
    raws = [mne.io.read_raw_fif(raw_fnames)]

    raw = mne.concatenate_raws(raws)
    # SSS reduces the data rank and the noise levels, so let's include
    # components based on a higher proportion of variance explained (0.999)
    # than we would otherwise do for non-Maxwell-filtered raw data (0.98)
    n_components = 0.999  # XXX: This can bring troubles to ICA
    if tsss:
        ica_name = op.join(config.meg_dir, subject,
                           'run_concat-tsss_%d-ica.fif' % tsss)
    else:
        ica_name = op.join(config.meg_dir, subject, 'run_concat-ica.fif')
    # Here we only compute ICA for MEG because we only eliminate ECG artifacts,
    # which are not prevalent in EEG (blink artifacts are, but we will remove
    # trials with blinks at the epoching stage).
    print('  Fitting ICA')
    ica = ICA(method='fastica', random_state=config.random_state,
              n_components=n_components)
    picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
                           stim=False, exclude='bads')
    ica.fit(raw, picks=picks, reject=dict(grad=4000e-13, mag=4e-12),
            decim=11)
    print('  Fit %d components (explaining at least %0.1f%% of the variance)'
          % (ica.n_components_, 100 * n_components))
    ica.save(ica_name)


# Memory footprint: around n_jobs * 4 GB
parallel, run_func, _ = parallel_func(run_ica, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
