"""
===================================
03. Maxwell filter using MNE-python
===================================
The data are Maxwell filtered using SSS or tSSS (if config.mf_st_duration is not None)
and movement compensation.

Using tSSS with a short duration can be used as an alternative to highpass
filtering. For instance, a duration of 10 s acts like a 0.1 Hz highpass.

The head position of all runs is corrected to the run specified in
config.mf_reference_run.
It is critical to mark bad channels before Maxwell filtering.

The function loads machine-specific calibration files from the paths set for
config.mf_ctc_fname  and config.mf_cal_fname.
"""  # noqa: E501

import os.path as op

import mne
from mne.parallel import parallel_func

import config


def run_maxwell_filter(subject):
    print("processing subject: %s" % subject)
    # XXX : put the study-specific names in the config file
    meg_subject_dir = op.join(config.meg_dir, subject)

    # To match their processing, transform to the head position of the
    # defined run
    run = config.runs[config.mf_reference_run] + '_filt'
    raw_fname_in = config.base_raw_fname.format(**locals())
    info = mne.io.read_info(op.join(meg_subject_dir, raw_fname_in))
    destination = info['dev_head_t']

    for run in config.runs:
        run += '_filt'
        raw_fname_in = config.base_raw_fname.format(**locals())
        run += '_sss'
        raw_fname_out = config.base_raw_fname.format(**locals())

        raw_fname_in = op.join(meg_subject_dir, raw_fname_in)
        raw_fname_out = op.join(meg_subject_dir, raw_fname_out)

        print("Reading: ", raw_fname_in)
        print("Writing: ", raw_fname_out)

        raw = mne.io.read_raw_fif(raw_fname_in)

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

        # XXX if we add multiple runs, this should probably plot an appended
        # version of the data
        if config.plot:
            # plot maxfiltered data
            figure = raw_sss.plot(
                n_channels=50, butterfly=True, group_by='position')
            figure.show()


parallel, run_func, _ = parallel_func(run_maxwell_filter, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
