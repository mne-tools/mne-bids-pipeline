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
import itertools

import mne
from mne.report import Report
from mne.preprocessing import ICA
from mne.parallel import parallel_func

from mne_bids import make_bids_basename

import config


def run_ica(subject, session=None):
    """Run ICA."""
    print("Processing subject: %s" % subject)

    # Construct the search path for the data file. `sub` is mandatory
    subject_path = op.join('sub-{}'.format(subject))
    # `session` is optional
    if session is not None:
        subject_path = op.join(subject_path, 'ses-{}'.format(session))

    subject_path = op.join(subject_path, config.kind)

    fpath_deriv = op.join(config.bids_root, 'derivatives',
                          config.PIPELINE_NAME, subject_path)

    raw_list = list()
    print("  Loading raw data")

    for run in config.runs:
        # load first run of raw data for ecg /eog epochs
        print("  Loading one run from raw data")

        bids_basename = make_bids_basename(subject=subject,
                                           session=session,
                                           task=config.task,
                                           acquisition=config.acq,
                                           run=run,
                                           processing=config.proc,
                                           recording=config.rec,
                                           space=config.space
                                           )

        if config.use_maxwell_filter:
            raw_fname_in = \
                op.join(fpath_deriv, bids_basename + '_sss_raw.fif')
        else:
            raw_fname_in = \
                op.join(fpath_deriv, bids_basename + '_filt_raw.fif')

        eve_fname = \
            op.join(fpath_deriv, bids_basename + '-eve.fif')

        print("Input: ", raw_fname_in, eve_fname)

        raw = mne.io.read_raw_fif(raw_fname_in, preload=True)

        raw_list.append(raw)

    print('  Concatenating runs')
    raw = mne.concatenate_raws(raw_list)

    events, event_id = mne.events_from_annotations(raw)

    if "eeg" in config.ch_types or config.kind == 'eeg':
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
                                events, event_id, config.tmin,
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

    ch_types = []
    if 'eeg' in config.ch_types or config.kind == 'eeg':
        ch_types.append('eeg')
    if set(config.ch_types).intersection(('meg', 'grad', 'mag')):
        ch_types.append('meg')

    for ch_type in ch_types:
        print('Running ICA for ' + ch_type)

        ica = ICA(method='fastica', random_state=config.random_state,
                  n_components=n_components[ch_type])

        picks = all_picks[ch_type]
        if picks.size == 0:
            ica.fit(epochs_for_ica, decim=config.ica_decim)
        else:
            ica.fit(epochs_for_ica, picks=picks, decim=config.ica_decim)

        print('  Fit %d components (explaining at least %0.1f%% of the'
              ' variance)' % (ica.n_components_, 100 * n_components[ch_type]))

        # Load ICA
        ica_fname = \
            op.join(fpath_deriv, bids_basename + '_%s-ica.fif' % ch_type)

        ica.save(ica_fname)

        if config.plot:
            # plot ICA components to html report
            report_fname = \
                op.join(fpath_deriv,
                        bids_basename + '_%s-ica.html' % ch_type)

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


def main():
    """Run ICA."""
    parallel, run_func, _ = parallel_func(run_ica, n_jobs=config.N_JOBS)
    parallel(run_func(subject, session) for subject, session in
             itertools.product(config.subjects_list, config.sessions))


if __name__ == '__main__':
    if config.use_ica:
        main()
