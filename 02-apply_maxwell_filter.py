"""
===================================
03. Maxwell filter using MNE-Python
===================================

The data are Maxwell filtered using SSS or tSSS (if config.mf_st_duration
is not None) and movement compensation.

Using tSSS with a short duration can be used as an alternative to highpass
filtering. For instance, a duration of 10 s acts like a 0.1 Hz highpass.

The head position of all runs is corrected to the run specified in
config.mf_reference_run.
It is critical to mark bad channels before Maxwell filtering.

The function loads machine-specific calibration files from the paths set for
config.mf_ctc_fname  and config.mf_cal_fname.
"""  # noqa: E501

import os.path as op
import itertools

import mne
from mne.parallel import parallel_func
from mne_bids import make_bids_basename

import config


def run_maxwell_filter(subject, session=None):
    print("Processing subject: %s" % subject)

    # Construct the search path for the data file. `sub` is mandatory
    subject_path = op.join('sub-{}'.format(subject))
    # `session` is optional
    if session is not None:
        subject_path = op.join(subject_path, 'ses-{}'.format(session))

    subject_path = op.join(subject_path, config.kind)

    for run_idx, run in enumerate(config.runs):

        bids_basename = make_bids_basename(subject=subject,
                                           session=session,
                                           task=config.task,
                                           acquisition=config.acq,
                                           run=run,
                                           processing=config.proc,
                                           recording=config.rec,
                                           space=config.space
                                           )

        # Prepare a name to save the data
        fpath_deriv = op.join(config.bids_root, 'derivatives',
                              config.PIPELINE_NAME, subject_path)
        raw_fname_in = op.join(fpath_deriv, bids_basename + '_filt_raw.fif')
        raw_fname_out = op.join(fpath_deriv, bids_basename + '_sss_raw.fif')

        print("Input: ", raw_fname_in)
        print("Output: ", raw_fname_out)

        raw = mne.io.read_raw_fif(raw_fname_in, allow_maxshield=True)
        raw.fix_mag_coil_types()

        if run_idx == 0:
            destination = raw.info['dev_head_t']

        if config.mf_st_duration:
            print('    st_duration=%d' % (config.mf_st_duration,))

        raw_sss = mne.preprocessing.maxwell_filter(
            raw,
            calibration=config.mf_cal_fname,
            cross_talk=config.mf_ctc_fname,
            st_duration=config.mf_st_duration,
            origin=config.mf_head_origin,
            destination=destination)

        raw_sss.save(raw_fname_out, overwrite=True)

        if config.plot:
            # plot maxfiltered data
            raw_sss.plot(n_channels=50, butterfly=True)


def main():
    """Run maxwell_filter."""
    parallel, run_func, _ = parallel_func(run_maxwell_filter,
                                          n_jobs=config.N_JOBS)
    parallel(run_func(subject, session) for subject, session in
             itertools.product(config.subjects_list, config.sessions))


if __name__ == '__main__':
    if config.use_maxwell_filter:
        main()
