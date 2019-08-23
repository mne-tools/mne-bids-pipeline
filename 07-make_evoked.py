"""
===============
06. Evoked data
===============

The evoked data sets are created by averaging different conditions.
"""

import os.path as op
import itertools

import mne
from mne.parallel import parallel_func

from mne_bids import make_bids_basename

import config


def run_evoked(subject, session=None):
    print("Processing subject: %s" % subject)

    # Construct the search path for the data file. `sub` is mandatory
    subject_path = op.join('sub-{}'.format(subject))
    # `session` is optional
    if session is not None:
        subject_path = op.join(subject_path, 'ses-{}'.format(session))

    subject_path = op.join(subject_path, config.kind)

    bids_basename = make_bids_basename(subject=subject,
                                       session=session,
                                       task=config.task,
                                       acquisition=config.acq,
                                       run=None,
                                       processing=config.proc,
                                       recording=config.rec,
                                       space=config.space
                                       )

    if config.use_ica or config.use_ssp:
        extension = '_cleaned-epo'
    else:
        extension = '-epo'

    fpath_deriv = op.join(config.bids_root, 'derivatives',
                          config.PIPELINE_NAME, subject_path)
    fname_in = \
        op.join(fpath_deriv, bids_basename + '%s.fif' % extension)

    fname_out = \
        op.join(fpath_deriv, bids_basename + '-ave.fif')

    print("Input: ", fname_in)
    print("Output: ", fname_out)

    print('  Creating evoked datasets')
    epochs = mne.read_epochs(fname_in, preload=True)

    evokeds = []
    for condition in config.conditions:
        evokeds.append(epochs[condition].average())
    mne.evoked.write_evokeds(fname_out, evokeds)

    if config.plot:
        for evoked in evokeds:
            evoked.plot()

        # What's next heeds channel locations
        # ts_args = dict(gfp=True, time_unit='s')
        # topomap_args = dict(time_unit='s')

        # for condition, evoked in zip(config.conditions, evokeds):
        #     evoked.plot_joint(title=condition, ts_args=ts_args,
        #                       topomap_args=topomap_args)


def main():
    """Run evoked."""
    parallel, run_func, _ = parallel_func(run_evoked, n_jobs=config.N_JOBS)
    parallel(run_func(subject, session) for subject, session in
             itertools.product(config.subjects_list, config.sessions))


if __name__ == '__main__':
    main()
