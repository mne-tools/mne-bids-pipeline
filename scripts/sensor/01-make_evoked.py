"""
===============
06. Evoked data
===============

The evoked data sets are created by averaging different conditions.
"""

import itertools
import logging

import mne
from mne.utils._bunch import BunchConst
from mne.parallel import parallel_func

from mne_bids import BIDSPath

import config
from config import gen_log_message, on_error, failsafe_run

logger = logging.getLogger('mne-bids-pipeline')


@failsafe_run(on_error=on_error)
def run_evoked(cfg, subject, session=None):
    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=cfg.task,
                         acquisition=cfg.acq,
                         run=None,
                         recording=cfg.rec,
                         space=cfg.space,
                         extension='.fif',
                         datatype=cfg.datatype,
                         root=cfg.deriv_root)

    processing = 'clean'  # always use the clean epochs

    fname_in = bids_path.copy().update(processing=processing, suffix='epo',
                                       check=False)
    fname_out = bids_path.copy().update(suffix='ave', check=False)

    msg = f'Input: {fname_in}, Output: {fname_out}'
    logger.info(gen_log_message(message=msg, step=6, subject=subject,
                                session=session))

    epochs = mne.read_epochs(fname_in, preload=True)

    msg = 'Creating evoked data based on experimental conditions …'
    logger.info(gen_log_message(message=msg, step=6, subject=subject,
                                session=session))
    all_evoked = dict()

    if isinstance(cfg.conditions, dict):
        for new_cond_name, orig_cond_name in cfg.conditions.items():
            evoked = epochs[orig_cond_name].average()
            evoked.comment = evoked.comment.replace(orig_cond_name,
                                                    new_cond_name)
            all_evoked[new_cond_name] = evoked
    else:
        for condition in cfg.conditions:
            evoked = epochs[condition].average()
            all_evoked[condition] = evoked

    if cfg.contrasts:
        msg = 'Contrasting evoked responses …'
        logger.info(gen_log_message(message=msg, step=6, subject=subject,
                                    session=session))

        for contrast in cfg.contrasts:
            cond_1, cond_2 = contrast
            evoked_diff = mne.combine_evoked([all_evoked[cond_1],
                                              all_evoked[cond_2]],
                                             weights=[1, -1])
            all_evoked[contrast] = evoked_diff

    evokeds = list(all_evoked.values())
    mne.write_evokeds(fname_out, evokeds)

    if cfg.interactive:
        for evoked in evokeds:
            evoked.plot()

        # What's next needs channel locations
        # ts_args = dict(gfp=True, time_unit='s')
        # topomap_args = dict(time_unit='s')

        # for condition, evoked in zip(config.conditions, evokeds):
        #     evoked.plot_joint(title=condition, ts_args=ts_args,
        #                       topomap_args=topomap_args)


def get_config(subject, session):
    cfg = BunchConst(
        task=config.get_task(),
        datatype=config.get_datatype(),
        session=session,
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        deriv_root=config.get_deriv_root(),
        conditions=config.conditions,
        contrasts=config.contrasts,
        interactive=config.interactive
    )
    return cfg


def main():
    """Run evoked."""
    msg = 'Running Step 6: Create evoked data'
    logger.info(gen_log_message(step=6, message=msg))

    parallel, run_func, _ = parallel_func(run_evoked, n_jobs=config.N_JOBS)
    parallel(run_func(get_config(subject, session), subject, session)
             for subject, session in
             itertools.product(config.get_subjects(),
                               config.get_sessions()))

    msg = 'Completed Step 6: Create evoked data'
    logger.info(gen_log_message(step=6, message=msg))


if __name__ == '__main__':
    main()
