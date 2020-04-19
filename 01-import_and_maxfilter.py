"""
========================
01. Apply Maxwell filter
========================

The data are imported from the BIDS folder.

If you chose to run Maxwell filter (config.use_maxwell_filter = True),
the data are Maxwell filtered using SSS or tSSS (if config.mf_st_duration
is not None) and movement compensation.

Using tSSS with a short duration can be used as an alternative to highpass
filtering. For instance, a duration of 10 s acts like a 0.1 Hz highpass.

The head position of all runs is corrected to the run specified in
config.mf_reference_run.
It is critical to mark bad channels before Maxwell filtering.

The function loads machine-specific calibration files from the paths set for
config.mf_ctc_fname  and config.mf_cal_fname.

Notes
-----
This is the first step of the pipeline, so it will also write a
`dataset_description.json` file to the root of the pipeline derivatives, which
are stored in bids_root/derivatives/PIPELINE_NAME. PIPELINE_NAME is defined in
the config.py file. The `dataset_description.json` file is formatted according
to the WIP specification for common BIDS derivatives, see this PR:

https://github.com/bids-standard/bids-specification/pull/265
"""  # noqa: E501

import os
import os.path as op
import glob
import itertools
import mne
from mne.preprocessing import find_bad_channels_maxwell
from mne.parallel import parallel_func
from mne_bids.read import reader as mne_bids_readers
from mne_bids import make_bids_basename, read_raw_bids
from mne_bids.config import BIDS_VERSION
from mne_bids.utils import _write_json

import config


# XXX currently only tested with use_maxwell_filter = False
def run_maxwell_filter(subject, session=None):
    print("Processing subject: %s" % subject)

    # Construct the search path for the data file. `sub` is mandatory
    subject_path = op.join('sub-{}'.format(subject))
    # `session` is optional
    if session is not None:
        subject_path = op.join(subject_path, 'ses-{}'.format(session))

    subject_path = op.join(subject_path, config.kind)
    data_dir = op.join(config.bids_root, subject_path)

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
        # Find the data file
        search_str = op.join(data_dir, bids_basename) + '_' + config.kind + '*'
        fnames = sorted(glob.glob(search_str))
        fnames = [f for f in fnames
                  if op.splitext(f)[1] in mne_bids_readers]

        if len(fnames) == 1:
            bids_fpath = fnames[0]
        elif len(fnames) == 0:
            raise ValueError('Could not find input data file matching: '
                             '"{}"'.format(search_str))
        elif len(fnames) > 1:
            raise ValueError('Expected to find a single input data file: "{}" '
                             ' but found:\n\n{}'
                             .format(search_str, fnames))

        if run_idx == 0:  # XXX does this work when no runs are specified?
            # Prepare the pipeline directory in /derivatives
            deriv_path = op.join(config.bids_root, 'derivatives',
                                 config.PIPELINE_NAME)
            fpath_out = op.join(deriv_path, subject_path)
            if not op.exists(fpath_out):
                os.makedirs(fpath_out)

            # Write a dataset_description.json for the pipeline
            ds_json = dict()
            ds_json['Name'] = config.PIPELINE_NAME + ' outputs'
            ds_json['BIDSVersion'] = BIDS_VERSION
            ds_json['PipelineDescription'] = {
                'Name': config.PIPELINE_NAME,
                'Version': config.VERSION,
                'CodeURL': config.CODE_URL,
            }
            ds_json['SourceDatasets'] = {
                'URL': 'n/a',
            }

            fname = op.join(deriv_path, 'dataset_description.json')
            _write_json(fname, ds_json, overwrite=True, verbose=True)

        # read_raw_bids automatically
        # - populates bad channels using the BIDS channels.tsv
        # - sets channels types according to BIDS channels.tsv `type` column
        # - sets raw.annotations using the BIDS events.tsv
        _, bids_fname = op.split(bids_fpath)
        raw = read_raw_bids(bids_fname, config.bids_root)

        # XXX hack to deal with dates that fif files cannot handle
        if config.daysback is not None:
            raw.anonymize(daysback=config.daysback)

        if config.crop is not None:
            raw.crop(*config.crop)

        raw.load_data()
        if hasattr(raw, 'fix_mag_coil_types'):
            raw.fix_mag_coil_types()

        if config.find_flat_channels_meg or config.find_noisy_channels_meg:
            if (config.find_flat_channels_meg and
                    not config.find_noisy_channels_meg):
                msg = 'Finding flat channels.'
            elif (config.find_noisy_channels_meg and
                  not config.find_flat_channels_meg):
                msg = 'Finding noisy channels using Maxwell filtering.'
            else:
                msg = ('Finding flat channels, and noisy channels using '
                       'Maxwell filtering.')

            print(msg)
            raw_lp_filtered_for_maxwell = (raw.copy()
                                           .filter(l_freq=None,
                                                   h_freq=40,
                                                   verbose=True))
            auto_noisy_chs, auto_flat_chs = find_bad_channels_maxwell(
                raw=raw_lp_filtered_for_maxwell,
                calibration=config.mf_cal_fname,
                cross_talk=config.mf_ctc_fname,
                verbose=True)
            del raw_lp_filtered_for_maxwell

            bads = raw.info['bads'].copy()
            if config.find_flat_channels_meg:
                print(f'Found {len(auto_flat_chs)} flat channels.')
                bads.extend(auto_flat_chs)
            if config.find_noisy_channels_meg:
                print(f'Found {len(auto_noisy_chs)} noisy channels.')
                bads.extend(auto_noisy_chs)

            bads = sorted(set(bads))
            raw.info['bads'] = bads
            print(f'Marked {len(raw.info["bads"])} channels as bad.')
            del bads, auto_flat_chs, auto_noisy_chs, msg

        if config.use_maxwell_filter:
            print('Applying maxwell filter.')

            # Warn if no bad channels are set before Maxfilter
            if not raw.info['bads']:
                print('\n Warning: Found no bad channels. \n ')

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

            # Prepare a name to save the data
            raw_fname_out = op.join(fpath_out, bids_basename + '_sss_raw.fif')
            raw_sss.save(raw_fname_out, overwrite=True)

            if config.plot:
                # plot maxfiltered data
                raw_sss.plot(n_channels=50, butterfly=True)

        else:
            print('Not applying maxwell filter.\n'
                  'If you wish to apply it set config.use_maxwell_filter=True')
            # Prepare a name to save the data
            raw_fname_out = op.join(fpath_out, bids_basename +
                                    '_nosss_raw.fif')
            raw.save(raw_fname_out, overwrite=True)

            if config.plot:
                # plot raw data
                raw.plot(n_channels=50, butterfly=True)


def main():
    """Run maxwell_filter."""
    parallel, run_func, _ = parallel_func(run_maxwell_filter,
                                          n_jobs=config.N_JOBS)
    parallel(run_func(subject, session) for subject, session in
             itertools.product(config.subjects_list, config.sessions))


if __name__ == '__main__':
    main()
