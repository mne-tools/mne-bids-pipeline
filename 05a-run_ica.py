"""
===========
05. Run ICA
===========
This fits ICA on epoched data filtered with 1 Hz highpass,
for this purpose only using fastICA. Separate ICAs are fitted and stored for
MEG and EEG data.
To actually remove designated ICA components from your data, you will have to
run 06a-apply_ica.py.
"""

import os.path as op

import mne
from mne.report import Report
from mne.preprocessing import ICA
from mne.parallel import parallel_func

import config


def run_ica(subject, tsss=config.mf_st_duration):
    print("Processing subject: %s" % subject)

    meg_subject_dir = op.join(config.meg_dir, subject)

    raw_list = list()
    events_list = list()
    print("  Loading raw data")

    for run in config.runs:
        extension = run + '_sss_raw'
        raw_fname_in = op.join(meg_subject_dir,
                               config.base_fname.format(**locals()))
        eve_fname = op.splitext(raw_fname_in)[0] + '-eve.fif'
        print("Input: ", raw_fname_in, eve_fname)

        raw = mne.io.read_raw_fif(raw_fname_in, preload=True)

        events = mne.read_events(eve_fname)
        events_list.append(events)

        # mark bads from any run
        bads = config.bads[subject]
        if isinstance(bads, dict):
            bads = set(sum(bads.values(), []))

        raw.info['bads'] = bads
        print("added bads: ", raw.info['bads'])

        raw_list.append(raw)

    print('  Concatenating runs')
    raw, events = mne.concatenate_raws(raw_list, events_list=events_list)
    if config.eeg:
        raw.set_eeg_reference(projection=True)
    del raw_list

    # don't reject based on EOG to keep blink artifacts
    # in the ICA computation.
    reject_ica = config.reject
    if reject_ica and 'eog' in reject_ica:
        reject_ica = dict(reject_ica)
        del reject_ica['eog']

    # produce high-pass filtered version of the data for ICA
    raw_ica = raw.copy().filter(l_freq=1., h_freq=None)

    print("  Running ICA...")
    epochs_for_ica = mne.Epochs(raw_ica,
                                events, config.event_id, config.tmin,
                                config.tmax, proj=True,
                                baseline=config.baseline,
                                preload=True, decim=config.decim,
                                reject=reject_ica)

    # run ICA on MEG and EEG
    picks_meg = mne.pick_types(epochs_for_ica.info, meg=True, eeg=False,
                               eog=False, stim=False, exclude='bads')
    picks_eeg = mne.pick_types(epochs_for_ica.info, meg=False, eeg=True,
                               eog=False, stim=False, exclude='bads')
    all_picks = {'meg': picks_meg, 'eeg': picks_eeg}

    # get number of components for ICA
    # compute_rank requires 0.18
    # n_components_meg = (mne.compute_rank(epochs_for_ica.copy()
    #                        .pick_types(meg=True)))['meg']

    n_components_meg = 0.999

    n_components = {'meg': n_components_meg, 'eeg': 0.999}

    if config.eeg:
        ch_types = ['meg', 'eeg']
    else:
        ch_types = ['meg']

    for ch_type in ch_types:
        print('Running ICA for ' + ch_type)

        ica = ICA(method='fastica', random_state=config.random_state,
                  n_components=n_components[ch_type])

        picks = all_picks[ch_type]

        ica.fit(epochs_for_ica, picks=picks, decim=config.ica_decim)

        print('  Fit %d components (explaining at least %0.1f%% of the'
              ' variance)' % (ica.n_components_, 100 * n_components[ch_type]))

        ica_fname = \
            '{0}_{1}_{2}-ica.fif'.format(subject, config.study_name, ch_type)
        ica_fname = op.join(meg_subject_dir, ica_fname)
        ica.save(ica_fname)

        if config.plot:
            # plot ICA components to html report
            report_fname = \
                '{0}_{1}_{2}-ica.html'.format(subject, config.study_name,
                                              ch_type)
            report_fname = op.join(meg_subject_dir, report_fname)
            report = Report(report_fname, verbose=False)

            for idx in range(0, ica.n_components_):
                figure = ica.plot_properties(epochs_for_ica,
                                             picks=idx,
                                             psd_args={'fmax': 60},
                                             show=False)

                report.add_figs_to_section(figure, section=subject,
                                           captions=(ch_type.upper() +
                                                     ' - ICA Components'))

            report.save(report_fname, overwrite=True, open_browser=False)


parallel, run_func, _ = parallel_func(run_ica, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
