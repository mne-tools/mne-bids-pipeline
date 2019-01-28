"""
===============
07. Evoked data
===============

The evoked data sets are created by averaging different categories of epochs.
The evoked data is saved using categories 'famous', 'scrambled', 'unfamiliar',
'contrast' and 'faces'.
"""

import os.path as op

import mne
from mne.parallel import parallel_func

from library.config import meg_dir, N_JOBS, l_freq


def run_evoked(subject_id, tsss=False):
    subject = "sub%03d" % subject_id
    print("Processing subject: %s%s"
          % (subject, (' (tSSS=%d)' % tsss) if tsss else ''))

    data_path = op.join(meg_dir, subject)
    if tsss:
        fname_epo = op.join(data_path, '%s-tsss_%d-epo.fif' % (subject, tsss))
        fname_evo = op.join(data_path, '%s-tsss_%d-ave.fif' % (subject, tsss))
    else:
        fname_epo = op.join(data_path, '%s_highpass-%sHz-epo.fif'
                            % (subject, l_freq))
        fname_evo = op.join(data_path, '%s_highpass-%sHz-ave.fif'
                            % (subject, l_freq))
    print('  Creating evoked datasets')
    epochs = mne.read_epochs(fname_epo, preload=True)

    evoked_famous = epochs['face/famous'].average()
    evoked_scrambled = epochs['scrambled'].average()
    evoked_unfamiliar = epochs['face/unfamiliar'].average()

    # Simplify comment
    evoked_famous.comment = 'famous'
    evoked_scrambled.comment = 'scrambled'
    evoked_unfamiliar.comment = 'unfamiliar'

    contrast = mne.combine_evoked([evoked_famous, evoked_unfamiliar,
                                   evoked_scrambled], weights=[0.5, 0.5, -1.])
    contrast.comment = 'contrast'
    faces = mne.combine_evoked([evoked_famous, evoked_unfamiliar], 'nave')
    faces.comment = 'faces'

    # let's make trial-count-normalized ones for group statistics
    epochs_eq = epochs.copy().equalize_event_counts(['face', 'scrambled'])[0]
    evoked_faces_eq = epochs_eq['face'].average()
    evoked_scrambled_eq = epochs_eq['scrambled'].average()
    assert evoked_faces_eq.nave == evoked_scrambled_eq.nave
    evoked_faces_eq.comment = 'faces_eq'
    evoked_scrambled_eq.comment = 'scrambled_eq'

    mne.evoked.write_evokeds(fname_evo, [evoked_famous, evoked_scrambled,
                                         evoked_unfamiliar, contrast, faces,
                                         evoked_faces_eq, evoked_scrambled_eq])


parallel, run_func, _ = parallel_func(run_evoked, n_jobs=N_JOBS)
parallel(run_func(subject_id) for subject_id in range(1, 20))
if l_freq is None:
    parallel(run_func(3, tsss) for tsss in (10, 1))  # Maxwell filtered data
