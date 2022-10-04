"""Extract evoked data for each condition."""

import itertools
import logging
from typing import Optional
from types import SimpleNamespace

import mne
from mne_bids import BIDSPath

import config
from config import gen_log_kwargs, failsafe_run
from config import parallel_func


logger = logging.getLogger('mne-bids-pipeline')


def get_input_fnames_evoked(**kwargs):
    cfg = kwargs.pop('cfg')
    subject = kwargs.pop('subject')
    session = kwargs.pop('session')
    assert len(kwargs) == 0, kwargs.keys()
    del kwargs
    fname_epochs = BIDSPath(subject=subject,
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
                            processing='clean',  # always use clean epochs
                            check=False)
    in_files = dict()
    in_files['epochs'] = fname_epochs
    return in_files


@failsafe_run(script_path=__file__,
              get_input_fnames=get_input_fnames_evoked)
def run_evoked(*, cfg, subject, session, in_files):
    out_files = dict()
    out_files['evoked'] = in_files['epochs'].copy().update(
        suffix='ave', processing=None, check=False)

    msg = f'Input: {in_files["epochs"].basename}'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))
    msg = f'Output: {out_files["evoked"].basename}'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                 session=session))

    epochs = mne.read_epochs(in_files.pop("epochs"), preload=True)

    msg = 'Creating evoked data based on experimental conditions …'
    logger.info(**gen_log_kwargs(message=msg, subject=subject,
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
        logger.info(**gen_log_kwargs(message=msg, subject=subject,
                                     session=session))

        for contrast in cfg.contrasts:
            evoked_list = [epochs[x].average() for x in contrast["conditions"]]
            evoked_diff = mne.combine_evoked(evoked_list,
                                             weights=contrast["weights"])
            all_evoked[contrast["name"]] = evoked_diff

    evokeds = list(all_evoked.values())
    mne.write_evokeds(out_files['evoked'], evokeds, overwrite=True)

    if cfg.interactive:
        for evoked in evokeds:
            evoked.plot()

        # What's next needs channel locations
        # ts_args = dict(gfp=True, time_unit='s')
        # topomap_args = dict(time_unit='s')

        # for condition, evoked in zip(config.conditions, evokeds):
        #     evoked.plot_joint(title=condition, ts_args=ts_args,
        #                       topomap_args=topomap_args)
    assert len(in_files) == 0, in_files.keys()
    return out_files


def get_config(
    subject: Optional[str] = None,
    session: Optional[str] = None
) -> SimpleNamespace:
    cfg = SimpleNamespace(
        task=config.get_task(),
        datatype=config.get_datatype(),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        deriv_root=config.get_deriv_root(),
        conditions=config.conditions,
        contrasts=config.get_all_contrasts(),
        interactive=config.interactive,
    )
    return cfg


def main():
    """Run evoked."""
    if config.task_is_rest:
        msg = '    … skipping: for resting-state task.'
        logger.info(**gen_log_kwargs(message=msg))
        return

    with config.get_parallel_backend():
        parallel, run_func = parallel_func(run_evoked)
        logs = parallel(
            run_func(cfg=get_config(), subject=subject, session=session)
            for subject, session in
            itertools.product(
                config.get_subjects(),
                config.get_sessions()
            )
        )

        config.save_logs(logs)


if __name__ == '__main__':
    main()
