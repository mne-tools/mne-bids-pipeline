"""
=================================
13. Group average on source level
=================================

Source estimates are morphed to the ``fsaverage`` brain.
"""

import itertools
import logging

import mne
from mne.parallel import parallel_func

from mne_bids import BIDSPath

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-study-template')


def morph_stc(subject, session=None):
    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=config.get_task(),
                         acquisition=config.acq,
                         run=None,
                         recording=config.rec,
                         space=config.space,
                         datatype=config.get_datatype(),
                         root=config.deriv_root,
                         check=False)

    fs_subject = config.get_fs_subject(subject)
    fs_subjects_dir = config.get_fs_subjects_dir()

    morphed_stcs = []
    for condition in config.conditions:
        method = config.inverse_method
        cond_str = config.sanitize_cond_name(condition)
        inverse_str = method
        hemi_str = 'hemi'  # MNE will auto-append '-lh' and '-rh'.
        morph_str = 'morph2fsaverage'

        fname_stc = bids_path.copy().update(
            suffix=f'{cond_str}+{inverse_str}+{hemi_str}')
        fname_stc_fsaverage = bids_path.copy().update(
            suffix=f'{cond_str}+{inverse_str}+{morph_str}+{hemi_str}')

        stc = mne.read_source_estimate(fname_stc)
        morph = mne.compute_source_morph(
            stc, subject_from=fs_subject, subject_to='fsaverage',
            subjects_dir=fs_subjects_dir)
        stc_fsaverage = morph.apply(stc)
        stc_fsaverage.save(fname_stc_fsaverage)
        morphed_stcs.append(stc_fsaverage)

        del fname_stc, fname_stc_fsaverage

    return morphed_stcs


@failsafe_run(on_error=on_error)
def main():
    """Run grp ave."""
    msg = 'Running Step 13: Grand-average source estimates'
    logger.info(gen_log_message(step=13, message=msg))

    mne.datasets.fetch_fsaverage(subjects_dir=config.get_fs_subjects_dir())

    parallel, run_func, _ = parallel_func(morph_stc, n_jobs=config.N_JOBS)
    all_morphed_stcs = parallel(run_func(subject, session)
                                for subject, session in
                                itertools.product(config.get_subjects(),
                                                  config.get_sessions()))
    all_morphed_stcs = [morphed_stcs for morphed_stcs, subject in
                        zip(all_morphed_stcs, config.get_subjects())]
    mean_morphed_stcs = map(sum, zip(*all_morphed_stcs))

    subject = 'average'
    # XXX to fix
    if config.get_sessions():
        session = config.get_sessions()[0]
    else:
        session = None

    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=config.get_task(),
                         acquisition=config.acq,
                         run=None,
                         processing=config.proc,
                         recording=config.rec,
                         space=config.space,
                         datatype=config.get_datatype(),
                         root=config.deriv_root,
                         check=False)

    for condition, this_stc in zip(config.conditions, mean_morphed_stcs):
        this_stc /= len(all_morphed_stcs)

        method = config.inverse_method
        cond_str = config.sanitize_cond_name(condition)
        inverse_str = method
        hemi_str = 'hemi'  # MNE will auto-append '-lh' and '-rh'.
        morph_str = 'morph2fsaverage'

        fname_stc_avg = bids_path.copy().update(
            suffix=f'{cond_str}+{inverse_str}+{morph_str}+{hemi_str}')
        this_stc.save(fname_stc_avg)

    msg = 'Completed Step 13: Grand-average source estimates'
    logger.info(gen_log_message(step=13, message=msg))


if __name__ == '__main__':
    main()
