"""
===================================
03. Maxwell filter using MNE-python
===================================

XXX: this needs update: congig.mf_reference_run, config.mf_st_duration
config.mf_cal_fname, config.mf_ctc_fname

The data are Maxwell filtered using tSSS and movement compensation.

Using tSSS with a short duration can be used as an alternative to highpass
filtering. Here we will use the default (10 sec) and a short window (1 sec).

It is critical to mark bad channels before Maxwell
filtering. Here for consistency we exploit the MaxFilter log files for
determining the bad channels.

The data are also lowpass filtered at 40 Hz using linear-phase FIR filter with
delay compensation. The transition bandwidth is automatically defined. See
`Background information on filtering <http://mne-tools.github.io/dev/auto_tutorials/plot_background_filtering.html>`_
for more. The filtered data are saved to separate files to the subject's'MEG'
directory.
"""  # noqa: E501

import os.path as op

import mne
from mne.parallel import parallel_func

import config


def run_maxwell_filter(subject):
    print("processing subject: %s" % subject)
    # XXX : put the study-specific names in the config file
    meg_subject_dir = op.join(config.meg_dir, subject)
    raw_fnames_in = [op.join(meg_subject_dir, '%s_audvis_filt_raw.fif' % subject)]

    raw_fnames_out = [op.join(meg_subject_dir, '%s_audvis_filt_sss_raw.fif' % subject)]

    # To match their processing, transform to the head position of the defined run
    info = mne.io.read_info(raw_fnames_in[config.mf_reference_run])
    destination = info['dev_head_t']

    for raw_fname_in, raw_fname_out in zip(raw_fnames_in, raw_fnames_out):
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


parallel, run_func, _ = parallel_func(run_maxwell_filter, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
