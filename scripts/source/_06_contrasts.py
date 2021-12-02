"""In this file, we analyze the contrast in the source space.

- We apply the inverse operator to the covariances for different conditions.
- We subtract the source estimates between the two conditions.
- We average across subjects this contrast.

For now, the inverse operator uses the empty room covariance.

Inspired from:
https://mne.tools/stable/auto_examples/inverse/mne_cov_power.html

(mne_dev) csegerie@drago2:~/Desktop/mne-bids-pipeline$ nice -n 5 xvfb-run python run.py --config=/storage/store2/data/time_in_wm_new/config_charbel.py --steps=source/06-source_contrast

To then explore the results go to the source_analysis workspace
which is in my local computer /home/charb/Desktop/parietal/other/source_analysis/source_visu.py
"""

from pathlib import Path
import itertools
import logging
from typing import Optional
from mne.source_estimate import SourceEstimate
import numpy as np

import mne
from mne.epochs import BaseEpochs, read_epochs
from mne.minimum_norm.inverse import apply_inverse_cov
from mne.utils import BunchConst
from mne.parallel import parallel_func
from mne.minimum_norm import read_inverse_operator
from mne_bids import BIDSPath

import config
from os.path import join

logger = logging.getLogger('mne-bids-pipeline')

# You have to  create the result folder beforehand
freq_bands = {
    'alpha': {
        'range': (10, 15),
        'outdir': Path('/storage/store2/derivatives/time_in_wm/source_contrast/res_3s_alpha')
    },
    'beta': {
        'range': (15, 20),
        'outdir': Path('/storage/store2/derivatives/time_in_wm/source_contrast/res_3s_beta')
    }

}



def fname(subject, session, res_path):
    """Get name of source file."""
    filename = f"brain_contrast_morphed_sub-{subject}-ses-{session}.stc"
    return join(res_path, filename)


def plot_source(stc, filename):
    """Plot and save the source estimate."""
    print('plotting source estimate :-)')
    # return None
    brain = stc.plot(
        subjects_dir=config.get_fs_subjects_dir(),
        hemi="split", size=(1600, 800))

    print(f'saving image to: {filename}')
    brain.save_image(filename=filename, mode='rgb')


def one_subject(subject, session, cfg, freq_band):
    """Compute the contrast and morph it to the fsavg."""
    print(f'processing subject {subject}')
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        run=None,
        recording=cfg.rec,
        space=cfg.space,
        extension='.fif',
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False)

    fname_inv = bids_path.copy().update(suffix='inv')
    inverse_operator = read_inverse_operator(fname_inv)

    fname_epoch = BIDSPath(
        subject=subject,
        session=session,
        task=cfg.task,
        acquisition=cfg.acq,
        run=None,
        recording=cfg.rec,
        space=cfg.space,
        suffix='epo',
        extension='.fif',
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        processing='clean',
        check=False)

    epochs = read_epochs(fname_epoch)
    # epochs.decimate(5)
    # l_freq, h_freq = 15, 20
    # l_freq, h_freq = 10, 15
    l_freq, h_freq = freq_band['range']
    res_path = freq_band['outdir']
    res_path.mkdir(exist_ok=True)

    stc_cond = []
    for cond in config.contrasts[0]:  # FIXME iterate over ALL contrasts
        print(cond)

        # import ipdb
        # ipdb.set_trace()
        epochs_filter = epochs[cond].crop(tmin=1, tmax=4)
        epochs_filter.filter(l_freq, h_freq)

        data_cov = mne.compute_covariance(epochs_filter)

        # Here the inverse uses the empty room recording.
        stc_data = apply_inverse_cov(
            data_cov, epochs.info, inverse_operator,
            nave=len(epochs), method='dSPM', verbose=False)
        stc_data.data = np.log10(stc_data.data)
        stc_cond.append(stc_data)
        filename = f"brain_{cond.replace('/', '+')}-sub-{subject}-ses-{session}.png"

        # import ipdb
        # ipdb.set_trace()

        plot_source(stc_data, join(res_path, filename))

    # Taking the difference of the log
    # stc_cond[1].data = 10**stc_cond[1].data
    # stc_cond[0].data = 10**stc_cond[0].data

    # stc_contrast = stc_cond[1] / stc_cond[0]
    stc_contrast = stc_cond[1] - stc_cond[0]

    # import ipdb
    # ipdb.set_trace()
    print("subject", subject, np.max(stc_contrast.data), np.min(stc_contrast.data))

    filename = f"brain_contrast_sub-{subject}-ses-{session}.png"
    plot_source(stc_contrast, join(res_path, filename))

    morph = mne.compute_source_morph(
        stc_contrast,
        subject_from=config.get_fs_subject(subject), subject_to='fsaverage',
        subjects_dir=cfg.fs_subjects_dir
    )
    stc_fsaverage: SourceEstimate = morph.apply(stc_contrast)  # type: ignore

    filename = f"brain_contrast_morphed_sub-{subject}-ses-{session}.png"
    plot_source(stc_fsaverage, join(res_path, filename))

    stc_fsaverage.save(fname=fname(subject, session, res_path=res_path))

    return stc_fsaverage


def group_analysis(subjects, sessions, cfg, freq_band):
    """Take the average of the source estimates."""
    res_path = freq_band['outdir']

    tab_stc = [[None for ses in sessions] for sub in subjects]
    for sub, subject in enumerate(subjects):
        for ses, session in enumerate(sessions):
            tab_stc[sub][ses] = mne.read_source_estimate(
                fname=fname(subject, session, res_path=res_path), subject=subject).data

    from mne.source_estimate import SourceEstimate
    stc_avg: SourceEstimate = mne.read_source_estimate(
        fname=fname(subjects[0], sessions[0], res_path=res_path))

    # Average subject ###################################################
    # Save mean subject
    stc_avg.data = np.mean(np.array(tab_stc), axis=(0, 1))
    print(stc_avg.data.shape)
    print(type(stc_avg))

    stc_avg.save(join(res_path, "stc_avg.stc"))

    # TODO: Not elegant
    subject = "fsaverage"
    stc_avg.subject = subject
    brain = stc_avg.plot(
        subjects_dir="/storage/store2/derivatives/time_in_wm/freesurfer/subjects",
        hemi="split", size=(1600, 800), backend="pyvistaqt",
        colormap="seismic",
        # No need to calibrate the colorbar here, you can just use the visualization script
        clim=dict(kind="percent", pos_lims=[30, 80, 95])
    )
    filename = f"brain_contrast_morphed_sub-{subject}.png"
    brain.save_image(
        filename=join(res_path, filename),
        mode='rgb')

    # Every subject ###################################################
    # Hack in order to see each subject on different time frame on freeview
    temp = np.mean(np.array(tab_stc), axis=(1, 3))
    temp = np.transpose(temp)
    print(np.array(tab_stc).shape)
    print(temp.shape)
    stc_avg.data = temp
    stc_avg.time = np.linspace(0, 1, len(subjects))
    
    stc_avg.save(join(res_path, "stc_all.stc"))


def get_config(
    subject: Optional[str] = None,
    session: Optional[str] = None
) -> BunchConst:
    cfg = BunchConst(
        task=config.get_task(),
        datatype=config.get_datatype(),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        ch_types=config.ch_types,
        conditions=config.conditions,
        inverse_method=config.inverse_method,
        deriv_root=config.get_deriv_root(),
        fs_subjects_dir=config.get_fs_subjects_dir()
    )
    return cfg


def main():
    """Source space contrast."""
    import os
    os.environ['OMP_NUM_THREADS'] = str(config.get_n_jobs())
    os.environ['NUMEXPR_NUM_THREADS'] = str(config.get_n_jobs())
    os.environ['MKL_NUM_THREADS'] = str(config.get_n_jobs())

    subjects = config.get_subjects()
    sessions = config.get_sessions()
    cfg = get_config()

    # Usefull for debugging
    # for sub, ses in itertools.product(subjects, sessions):
    #     one_subject(sub, ses, cfg)

    parallel, run_func, _ = parallel_func(one_subject,
                                          n_jobs=config.get_n_jobs())
    parallel(
        run_func(cfg=cfg, subject=subject, session=session, freq_band=freq_band)
        for subject, session, freq_band in
        itertools.product(subjects, sessions, [freq_bands['alpha'], freq_bands['beta']])
    )

    parallel, run_func, _ = parallel_func(group_analysis,
                                          n_jobs=config.get_n_jobs())
    parallel(
        run_func(subjects, sessions, cfg, freq_band)
        for freq_band in [freq_bands['alpha'], freq_bands['beta']]
        # for subjects, sessions, cfg in subjects, sessions, cfg)
    )


if __name__ == '__main__':
    main()
