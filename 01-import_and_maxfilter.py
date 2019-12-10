"""
===================================
01. Maxwell filter using MNE-Python
===================================

The data are imported from the BIDS folder. 
The script also sets up the derivative structure. 

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
"""  # noqa: E501

import os
import os.path as op
import glob
import itertools

import mne
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

        if run_idx==0: # XXX does this when no runs are specified?            
            # Prepare the pipeline directory in /derivatives
            deriv_path = op.join(config.bids_root, 'derivatives', config.PIPELINE_NAME)
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

        if config.crop is not None:
            raw.crop(*config.crop)
            
        raw.load_data()
        raw.fix_mag_coil_types()
               
        if config.use_maxwell_filter:
            print('Applying maxwell filter.')
            
            # Warn if no bad channels are set before Maxfilter
            if raw.info['bads'] is None: # XXX is this None of no bads were set?
                print('\n Warning: Found no bad channels. \n ')

            if run_idx == 0:
                destination = raw.info['dev_head_t']

            if config.mf_st_duration:
                print('    st_duration=%d' % (config.mf_st_duration,))

            raw_sss = mne.preprocessing.maxwell_filter(
                raw,
                calibration=config.mf_cal_fname,#XXX what to do with these files for sample?
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
            raw_fname_out = op.join(fpath_out, bids_basename + '_nosss_raw.fif')
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
