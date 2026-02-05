"""Utilities for mangling config vars."""

import copy
import functools
import pathlib
from collections.abc import Iterable, Sequence, Sized
from inspect import signature
from types import ModuleType, SimpleNamespace
from typing import Any, Literal, TypeVar

import mne
import mne_bids
import numpy as np
from mne_bids import BIDSPath

from ._logging import gen_log_kwargs, logger
from .typing import ArbitraryContrast, BaselineTypeT, ConditionsTypeT, ContrastSequenceT

try:
    _set_keys_arbitrary_contrast = set(ArbitraryContrast.__required_keys__)
except Exception:
    _set_keys_arbitrary_contrast = set(ArbitraryContrast.__annotations__.keys())


def get_fs_subjects_dir(config: SimpleNamespace) -> pathlib.Path:
    if not config.subjects_dir and config.deriv_root is not None:
        # We do this check here (and not in our regular checks section) to
        # avoid an error message when a user doesn't intend to run the source
        # analysis steps anyway.
        raise ValueError(
            'When specifying a "deriv_root", you must also supply a "subjects_dir".'
        )

    if not config.subjects_dir:
        assert isinstance(config.bids_root, pathlib.Path)
        return config.bids_root / "derivatives" / "freesurfer" / "subjects"
    else:
        return pathlib.Path(config.subjects_dir).expanduser().resolve()


def get_fs_subject(
    config: SimpleNamespace, subject: str, session: str | None = None
) -> str:
    subjects_dir = get_fs_subjects_dir(config)

    if config.use_template_mri is not None:
        assert isinstance(config.use_template_mri, str), type(config.use_template_mri)
        return config.use_template_mri

    if session is not None:
        return f"sub-{subject}_ses-{session}"
    elif (pathlib.Path(subjects_dir) / subject).exists():
        return subject
    else:
        return f"sub-{subject}"


def _has_session_specific_anat(
    subject: str, session: str | None, subjects_dir: pathlib.Path
) -> bool:
    return (subjects_dir / f"sub-{subject}_ses-{session}").exists()


@functools.cache
def _get_entity_vals_cached(
    *args: list[Any],
    **kwargs: dict[str, Any],
) -> tuple[str, ...]:
    return tuple(str(x) for x in mne_bids.get_entity_vals(*args, **kwargs))


def get_datatype(config: SimpleNamespace) -> Literal["meg", "eeg"]:
    # Content of ch_types should be sanitized already, so we don't need any
    # extra sanity checks here.
    if config.data_type == "meg":
        return "meg"
    if config.data_type == "eeg":
        return "eeg"
    if config.data_type is None:
        if config.ch_types == ["eeg"]:
            return "eeg"
        if any(t in ["meg", "mag", "grad"] for t in config.ch_types):
            return "meg"
    raise RuntimeError(
        "This probably shouldn't happen, got "
        f"config.data_type={repr(config.data_type)} and "
        f"config.ch_types={repr(config.ch_types)} "
        "but could not determine the datatype. Please contact "
        "the MNE-BIDS-pipeline developers. Thank you."
    )


@functools.cache
def _get_datatypes_cached(root: pathlib.Path) -> tuple[str, ...]:
    return tuple(mne_bids.get_datatypes(root=root))


def _get_ignore_datatypes(config: SimpleNamespace) -> tuple[str, ...]:
    _all_datatypes = _get_datatypes_cached(root=config.bids_root)
    _ignore_datatypes = set(_all_datatypes) - set([get_datatype(config)])
    return tuple(sorted(_ignore_datatypes))


def get_subjects(config: SimpleNamespace) -> list[str]:
    _valid_subjects = _get_entity_vals_cached(
        root=config.bids_root,
        entity_key="subject",
        ignore_datatypes=_get_ignore_datatypes(config),
    )
    if config.subjects == "all":
        s = _valid_subjects
    else:
        s = config.subjects
        missing_subjects = set(s) - set(_valid_subjects)
        if missing_subjects:
            raise FileNotFoundError(
                "The following requested subjects were not found in the dataset: "
                f"{', '.join(missing_subjects)}"
            )

    # Preserve order and remove excluded subjects
    subjects = [
        subject
        for subject in s
        if subject not in config.exclude_subjects and subject != "emptyroom"
    ]

    return subjects


def get_sessions(config: SimpleNamespace) -> tuple[None] | tuple[str, ...]:
    sessions = _get_sessions(config)
    if not sessions:
        return (None,)
    else:
        return sessions


def _get_sessions(config: SimpleNamespace) -> tuple[str, ...]:
    sessions = copy.deepcopy(config.sessions)
    _all_sessions = _get_entity_vals_cached(
        root=config.bids_root,
        entity_key="session",
        ignore_datatypes=_get_ignore_datatypes(config),
    )
    if sessions == "all":
        sessions = _all_sessions

    return tuple(str(x) for x in sessions)


def _get_task_float(val: dict[str, float] | float, task: str | None) -> float:
    """Get the float value for a task."""
    if isinstance(val, dict):
        assert task is not None
        task_float = val[task]
    else:
        task_float = val
    return task_float


def _get_task_average_epochs_tlims(
    *,
    config: SimpleNamespace,
) -> tuple[float, float]:
    # Not necessarily the right thing to do here, but since we can't concat epochs of
    # different lengths, let's use the average tmin/tmax across tasks
    vals = tuple(
        np.mean([_get_task_float(epochs_lim, task=task) for task in config.all_tasks])
        for epochs_lim in (config.epochs_tmin, config.epochs_tmax)
    )
    out = tuple(float(v) for v in vals)
    assert len(out) == 2  # just to make mypy happy
    return out


def _get_task_baseline(
    baseline: BaselineTypeT | dict[str, BaselineTypeT], task: str | None
) -> BaselineTypeT:
    """Get the baseline value for a task."""
    if isinstance(baseline, dict):
        assert task is not None
        task_baseline = baseline[task]
    else:
        task_baseline = baseline
    return task_baseline


def _get_task_conditions_dict(
    *,
    conditions: ConditionsTypeT | dict[str, ConditionsTypeT],
    task: str | None,
) -> dict[str, str]:
    """Get conditions for the current task as a dict mapping new to old names."""
    out_conditions: dict[str, str] = dict()
    if isinstance(conditions, Sequence):
        for cond in conditions:
            out_conditions[cond] = cond
    else:
        # This somewhat clunky looping (rather than just checking the first item)
        # is needed to make mypy happy about the nested dicts
        for key, val in conditions.items():
            assert isinstance(key, str)
            # if it's a string, it's a remapping
            if isinstance(val, str):
                out_conditions[key] = val
            else:
                # otherwise, it's a sequence or a remapping itself, and it's
                # task-specific
                if key == task:
                    if isinstance(val, Sequence):
                        for cond in val:
                            out_conditions[cond] = cond
                    else:
                        out_conditions.update(val)
    del task
    return out_conditions


def get_subjects_sessions(
    config: SimpleNamespace,
) -> dict[str, tuple[None] | tuple[str, ...]]:
    subjects = get_subjects(config)
    cfg_sessions = _get_sessions(config)
    # easy case first: datasets that don't have (named) sessions
    if not cfg_sessions:
        return {subj: (None,) for subj in subjects}

    # find which tasks to ignore when deciding if a subj has data for a session
    ignore_datatypes = _get_ignore_datatypes(config)
    tasks = get_tasks(config=config)
    all_tasks = _get_entity_vals_cached(
        root=config.bids_root,
        entity_key="task",
        ignore_datatypes=ignore_datatypes,
    )
    ignore_tasks = tuple(set(all_tasks) - set(tasks))

    # loop over subjs and check for available sessions
    subj_sessions: dict[str, tuple[None] | tuple[str, ...]] = dict()
    kwargs = (
        dict(ignore_suffixes=("scans", "coordsystem"))
        if "ignore_suffixes" in signature(mne_bids.get_entity_vals).parameters
        else dict()
    )
    for subject in subjects:
        subj_folder = config.bids_root / f"sub-{subject}"
        valid_sessions_subj = _get_entity_vals_cached(
            subj_folder,
            entity_key="session",
            ignore_tasks=ignore_tasks,
            ignore_acquisitions=("calibration", "crosstalk"),
            ignore_datatypes=ignore_datatypes,
            **kwargs,
        )
        keep_sessions: tuple[str, ...]
        # if valid_sessions_subj is empty, it might be because the dataset just doesn't
        # have `session` subfolders, or it might be that none of the sessions in config
        # are available for this subject.
        if not valid_sessions_subj:
            if any([x.name.startswith("ses") for x in subj_folder.iterdir()]):
                keep_sessions = ()  # has `ses-*` folders, just not the ones we want
            else:
                keep_sessions = cfg_sessions  # doesn't have `ses-*` folders
        else:
            missing_sessions = sorted(set(cfg_sessions) - set(valid_sessions_subj))
            if missing_sessions and not config.allow_missing_sessions:
                raise RuntimeError(
                    f"Subject {subject} is missing session{_pl(missing_sessions)} "
                    f"{missing_sessions}, and `config.allow_missing_sessions` is False"
                )
            keep_sessions = tuple(sorted(set(cfg_sessions) & set(valid_sessions_subj)))
        if len(keep_sessions):
            subj_sessions[subject] = keep_sessions
    return subj_sessions


def get_subjects_given_session(
    config: SimpleNamespace, session: str | None
) -> tuple[str, ...]:
    """Get the subjects who actually have data for a given session."""
    sub_ses = get_subjects_sessions(config)
    subjects = (
        tuple(sub for sub, ses in sub_ses.items() if session in ses)
        if config.allow_missing_sessions
        else config.subjects
    )
    assert not isinstance(subjects, str), subjects  # make sure it's not "all"
    return subjects


def _get_runs_sst(
    *,
    config: SimpleNamespace,
    subject: str,
    session: str | None,
    task: str | None,
) -> list[str | None]:
    """Return a list of runs for a given task in the BIDS input data."""
    if subject == "average":  # Used when creating the report
        return [None]

    kwargs = dict(ignore_datatypes=_get_ignore_datatypes(config))
    # Only traverse through the current subject's directory
    subj_root = config.bids_root / f"sub-{subject}"
    # Limit to session of interest
    if session is not None:
        sessions = _get_entity_vals_cached(subj_root, entity_key="session", **kwargs)
        kwargs["ignore_sessions"] = tuple(set(sessions) - set([session]))
        del session
    # Limit to task of interest
    tasks = _get_entity_vals_cached(subj_root, entity_key="task", **kwargs)
    if task:
        kwargs["ignore_tasks"] = tuple(set(tasks) - set([task]))
    del tasks
    # Get the runs for that task
    runs = _get_entity_vals_cached(subj_root, entity_key="run", **kwargs)

    # If we don't have any `run` entities, just set it to None, as we
    # commonly do when creating a BIDSPath.
    valid_runs: tuple[str, ...] | tuple[None]
    if runs:
        valid_runs = tuple(
            r for r in runs if r not in (config.exclude_runs or {}).get(subject, [])
        )
    else:
        valid_runs = (None,)
    del runs

    runs_sst: list[str | None] = list()
    if config.runs == "all":
        runs_sst[:] = valid_runs
    elif isinstance(config.runs, dict):
        runs_sst[:] = config.runs[task]
    else:
        assert isinstance(config.runs, list)
        runs_sst[:] = config.runs

    inclusion = set(runs_sst).issubset(set(valid_runs))
    if not inclusion:
        raise ValueError(
            f"Invalid run. It can be a subset of {valid_runs} but got {runs_sst}"
        )

    return runs_sst


def get_runs_tasks(
    *,
    config: SimpleNamespace,
    subject: str,
    session: str | None,
    which: tuple[str, ...] = ("runs", "noise", "rest"),
) -> tuple[tuple[str | None, str | None], ...]:
    """Get (run, task) tuples for all requested tasks and runs plus (maybe) rest."""
    from ._import_data import _get_noise_path, _get_rest_path

    assert isinstance(which, tuple)
    assert all(isinstance(inc, str) for inc in which)
    assert all(inc in ("runs", "noise", "rest") for inc in which)
    runs_tasks: list[tuple[str | None, str | None]] = list()
    if "runs" in which:
        for task in get_tasks(config=config):
            runs_tasks += [
                (run, task)
                for run in _get_runs_sst(
                    config=config,
                    subject=subject,
                    session=session,
                    task=task,
                )
            ]
    if "rest" in which:
        rest_path = _get_rest_path(
            cfg=config,
            subject=subject,
            session=session,
            kind="orig",
            add_bads=False,
        )
        if rest_path:
            runs_tasks.append((None, "rest"))
    if "noise" in which:
        mf_reference_run, mf_reference_task = get_mf_reference_run_task(
            config=config,
            subject=subject,
            session=session,
        )
        noise_path = _get_noise_path(
            mf_reference_run=mf_reference_run,
            mf_reference_task=mf_reference_task,
            cfg=config,
            subject=subject,
            session=session,
            kind="orig",
            add_bads=False,
        )
        if noise_path:
            runs_tasks.append((None, "noise"))
    return tuple(runs_tasks)


def get_mf_reference_run_task(
    config: SimpleNamespace,
    subject: str,
    session: str | None,
) -> tuple[str | None, str | None]:
    # Retrieve to run identifier (number, name) of the reference run
    run: str | None
    task: str | None
    if config.mf_reference_task is not None:
        task = config.mf_reference_task
    else:
        task = get_tasks(config=config)[0]
    task_runs = _get_runs_sst(
        config=config,
        subject=subject,
        session=session,
        task=task,
    )
    if config.mf_reference_run is not None:
        assert isinstance(config.mf_reference_run, str), type(config.mf_reference_run)
        run = config.mf_reference_run
        if run not in task_runs:
            raise ValueError(
                f"You set mf_reference_run={config.mf_reference_run}, but your "
                "dataset only contains the following runs for "
                f"{config.mf_reference_task=}: {task_runs}"
            )
    else:
        run = task_runs[0]
    return run, task


def get_tasks(config: SimpleNamespace) -> list[str | None]:
    # We should only need to compute this once for a given run of the pipeline
    if hasattr(config, "all_tasks"):
        return list(config.all_tasks)
    _valid_tasks = _get_entity_vals_cached(
        root=config.bids_root,
        entity_key="task",
        ignore_datatypes=_get_ignore_datatypes(config),
    )
    tasks: list[str | None] = list()
    if config.task:
        if isinstance(config.task, str):
            tasks.append(config.task)
        else:
            tasks.extend(config.task)
        if set(tasks) - set(_valid_tasks):
            raise ValueError(
                "The following requested tasks were not found in the dataset: "
                f"{set(tasks) - set(_valid_tasks)}"
            )
    else:
        if not _valid_tasks:
            tasks.append(None)
        else:
            tasks.extend(_valid_tasks)
    return tasks


def _get_ss(
    *,
    config: SimpleNamespace,
) -> list[tuple[str, str | None]]:
    return [
        (subject, session)
        for subject, sessions in get_subjects_sessions(config).items()
        for session in sessions
    ]


def _get_sst(
    *,
    config: SimpleNamespace,
) -> list[tuple[str, str | None, str | None]]:
    return [
        (subject, session, task)
        for subject, session in _get_ss(config=config)
        for task in get_tasks(config=config)
    ]


def _get_ssrt(
    *,
    config: SimpleNamespace,
    which: tuple[str, ...] | None = None,
) -> list[tuple[str, str | None, str | None, str | None]]:
    kwargs = dict()
    if which is not None:
        kwargs["which"] = which
    return [
        (subject, session, run, task)
        for subject, session in _get_ss(config=config)
        for run, task in get_runs_tasks(
            config=config,
            subject=subject,
            session=session,
            **kwargs,
        )
    ]


def _limit_which_clean(*, config: SimpleNamespace) -> tuple[str, ...]:
    which: tuple[str, ...] = ()
    if config.process_raw_clean:
        which += ("runs",)
    if config.process_empty_room:
        which += ("noise",)
    if config.process_rest:
        which += ("rest",)
    return which


def _get_channels_generic(
    channels: Any,
    subject: str = "",
    session: str | None = "",
    *,
    variable_name: str = "_unspecified_",
) -> Any:
    if not isinstance(channels, dict):
        return channels

    assert isinstance(channels, dict), "channels must be dict or concrete value"

    # session specific ch definition supersedes subject-level ch definition
    for key in (f"sub-{subject}_ses-{session}", f"sub-{subject}"):
        # empty list and None are explicitly allowed
        if key in channels:
            return channels[key]

    # use try/catch to allow for pickleable defaultdict implementation
    try:
        return channels["default"]
    except KeyError as e:
        raise KeyError(
            f"Could not find appropriate channel setting for {subject=} "
            f"and {session=} in config.{variable_name}, set it explicitly "
            'or set a default using the string key "default".'
        ) from e


def get_ecg_channel(
    ecg_channel: str | dict[str, str],
    subject: str = "",
    session: str | None = "",
) -> str:
    out = _get_channels_generic(
        ecg_channel,
        subject,
        session,
        variable_name="ssp_ecg_channel",
    )
    assert isinstance(out, str)  # mypy
    return out


def get_eog_channels(
    eog_channels: Sequence[str] | None | dict[str, Sequence[str] | None],
    subject: str = "",
    session: str | None = "",
) -> Sequence[str] | None:
    out = _get_channels_generic(
        eog_channels,
        subject,
        session,
        variable_name="eog_channels",
    )
    assert isinstance(out, Sequence | None)  # mypy
    return out


def sanitize_cond_name(cond: str) -> str:
    cond = cond.replace("/", "").replace("_", "").replace("-", "").replace(" ", "")
    return cond


def get_mf_cal_fname(
    *, config: SimpleNamespace, subject: str, session: str | None
) -> pathlib.Path | None:
    msg = "Could not find Maxwell Filter calibration file {where}."
    if config.mf_cal_fname is None:
        bids_path = BIDSPath(
            subject=subject,
            session=session,
            suffix="meg",
            datatype="meg",
            root=config.bids_root,
        )
        bids_match = bids_path.match()
        mf_cal_fpath = None
        if len(bids_match) > 0:
            mf_cal_fpath = bids_match[0].meg_calibration_fpath
        if mf_cal_fpath is None:
            msg = msg.format(where=f"from BIDSPath {bids_path}")
            if config.mf_cal_missing == "raise":
                raise ValueError(msg)
            elif config.mf_cal_missing == "warn":
                msg = f"WARNING: {msg} Set to None."
                logger.info(**gen_log_kwargs(message=msg))
    else:
        mf_cal_fpath = pathlib.Path(config.mf_cal_fname).expanduser().absolute()
        if not mf_cal_fpath.exists():
            msg = msg.format(where=f"at {str(config.mf_cal_fname)}")
            if config.mf_cal_missing == "raise":
                raise ValueError(msg)
            else:
                mf_cal_fpath = None
                if config.mf_cal_missing == "warn":
                    msg = f"WARNING: {msg} Set to None."
                    logger.info(**gen_log_kwargs(message=msg))

    assert isinstance(mf_cal_fpath, pathlib.Path | None), type(mf_cal_fpath)
    return mf_cal_fpath


def get_mf_ctc_fname(
    *, config: SimpleNamespace, subject: str, session: str | None
) -> pathlib.Path | None:
    msg = "Could not find Maxwell Filter cross-talk file {where}."
    if config.mf_ctc_fname is None:
        bids_path = BIDSPath(
            subject=subject,
            session=session,
            suffix="meg",
            datatype="meg",
            root=config.bids_root,
        )
        bids_match = bids_path.match()
        mf_ctc_fpath = None
        if len(bids_match) > 0:
            mf_ctc_fpath = bids_match[0].meg_crosstalk_fpath
        if mf_ctc_fpath is None:
            msg = msg.format(where=f"from BIDSPath {bids_path}")
            if config.mf_ctc_missing == "raise":
                raise ValueError(msg)
            elif config.mf_ctc_missing == "warn":
                msg = f"WARNING: {msg} Set to None."
                logger.info(**gen_log_kwargs(message=msg))

    else:
        mf_ctc_fpath = pathlib.Path(config.mf_ctc_fname).expanduser().absolute()
        if not mf_ctc_fpath.exists():
            msg = msg.format(where=f"at {str(config.mf_ctc_fname)}")
            if config.mf_ctc_missing == "raise":
                raise ValueError(msg)
            else:
                mf_ctc_fpath = None
                if config.mf_ctc_missing == "warn":
                    msg = f"WARNING: {msg} Set to None."
                    logger.info(**gen_log_kwargs(message=msg))

    assert isinstance(mf_ctc_fpath, pathlib.Path | None), type(mf_ctc_fpath)
    return mf_ctc_fpath


RawEpochsEvokedT = TypeVar(
    "RawEpochsEvokedT", bound=mne.io.BaseRaw | mne.BaseEpochs | mne.Evoked
)


def _restrict_analyze_channels(
    inst: RawEpochsEvokedT, cfg: SimpleNamespace
) -> RawEpochsEvokedT:
    analyze_channels = cfg.analyze_channels
    if cfg.analyze_channels == "ch_types":
        analyze_channels = cfg.ch_types
        inst.apply_proj()
    # We special-case the average reference here to work around a situation
    # where e.g. `analyze_channels` might contain only a single channel:
    # `concatenate_epochs` below will then fail when trying to create /
    # apply the projection. We can avoid this by removing an existing
    # average reference projection here, and applying the average reference
    # directly – without going through a projector.
    elif "eeg" in cfg.ch_types and cfg.eeg_reference == "average":
        inst.set_eeg_reference("average")
    else:
        inst.apply_proj()
    inst.pick(analyze_channels)
    return inst


def _get_bem_conductivity(cfg: SimpleNamespace) -> tuple[tuple[float, ...] | None, str]:
    conductivity: tuple[float, ...] | None = None  # should never be used
    if cfg.fs_subject in ("fsaverage", cfg.use_template_mri):
        pass
        tag = "5120-5120-5120"
    elif "eeg" in cfg.ch_types:
        conductivity = (0.3, 0.006, 0.3)
        tag = "5120-5120-5120"
    else:
        conductivity = (0.3,)
        tag = "5120"
    return conductivity, tag


def _meg_in_ch_types(ch_types: str) -> bool:
    return "mag" in ch_types or "grad" in ch_types or "meg" in ch_types


def get_noise_cov_bids_path(
    cfg: SimpleNamespace, subject: str, session: str | None
) -> BIDSPath:
    """Retrieve the path to the noise covariance file.

    Parameters
    ----------
    cfg
        The local configuration.
    subject
        The subject identifier.
    session
        The session identifier.

    Returns
    -------
    BIDSPath
        _description_
    """
    noise_cov_bp = BIDSPath(
        subject=subject,
        session=session,
        task=None,
        acquisition=cfg.acq,
        run=None,
        processing="clean",
        recording=cfg.rec,
        space=cfg.space,
        suffix="cov",
        extension=".fif",
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        check=False,
    )
    noise_cov = cfg.noise_cov
    if callable(noise_cov):
        noise_cov_bp.processing = "custom"
    elif noise_cov == "emptyroom":
        noise_cov_bp.task = "noise"
    elif noise_cov == "ad-hoc":
        noise_cov_bp.processing = "adhoc"
    elif noise_cov == "rest":
        noise_cov_bp.task = "rest"
    else:  # estimated from a time period
        pass

    return noise_cov_bp


def _get_task_contrasts(
    *, contrasts: ContrastSequenceT | dict[str, ContrastSequenceT], task: str | None
) -> Iterable[ArbitraryContrast]:
    normalized_contrasts = []
    # contrasts is either a dict[str(task),Sequence] or a Sequence
    use_contrasts: ContrastSequenceT
    if isinstance(contrasts, dict):
        assert task is not None
        use_contrasts = contrasts.get(task, [])
    else:
        use_contrasts = contrasts
    del contrasts
    for contrast in use_contrasts:
        if isinstance(contrast, tuple):
            assert len(contrast) == 2
            normalized_contrasts.append(
                ArbitraryContrast(
                    name=f"{contrast[0]}-{contrast[1]}",
                    conditions=list(contrast),
                    weights=[1, -1],
                )
            )
        else:
            normalized_contrasts.append(contrast)
    return normalized_contrasts


def _get_task_decoding_contrasts(
    config: SimpleNamespace, *, task: str | None
) -> Iterable[tuple[str, str]]:
    normalized_contrasts = []
    for contrast in _get_task_contrasts(contrasts=config.contrasts, task=task):
        # If a contrast is an `ArbitraryContrast` and satisfies
        # * has exactly two conditions (`check_len`)
        # * weights sum to 0 (`check_sum`)
        # Then the two conditions are used to perform decoding
        check_len = len(contrast["conditions"]) == 2
        check_sum = np.isclose(np.sum(contrast["weights"]), 0)
        if check_len and check_sum:
            cond_1 = contrast["conditions"][0]
            cond_2 = contrast["conditions"][1]
            normalized_contrasts.append((cond_1, cond_2))
    return normalized_contrasts


# Map _config.decoding_which_epochs to a BIDS proc- entity
_EPOCHS_DESCRIPTION_TO_PROC_MAP = {
    "uncleaned": None,
    "after_ica": "ica",
    "after_ssp": "ssp",
    "cleaned": "clean",
}


def _get_decoding_proc(config: SimpleNamespace) -> str | None:
    return _EPOCHS_DESCRIPTION_TO_PROC_MAP[config.decoding_which_epochs]


def get_eeg_reference(
    config: SimpleNamespace,
) -> Literal["average"] | Iterable[str]:
    if config.eeg_reference == "average":
        return "average"
    elif isinstance(config.eeg_reference, str):
        return [config.eeg_reference]
    else:
        assert isinstance(config.eeg_reference, Iterable)
        assert all(isinstance(x, str) for x in config.eeg_reference)
        return config.eeg_reference


def _validate_contrasts(
    contrasts: ContrastSequenceT | dict[str, ContrastSequenceT],
    *,
    tasks: str | None | list[str],
) -> None:
    tasks_list: list[str] | list[None]
    if tasks is None:
        tasks_list = [None]
    elif isinstance(tasks, str):
        tasks_list = [tasks]
    else:
        tasks_list = tasks
    contrasts_dict: dict[str, ContrastSequenceT]
    if not isinstance(contrasts, dict):
        if len(tasks_list) > 1:
            raise ValueError(
                f"When multiple tasks are defined via:\nconfig.task={tasks!r}\n"
                "contrasts must be a dict whose keys are task names and values are "
                f"sequences of contrasts, but got:\n{contrasts!r}"
            )
        contrasts_dict = {str(tasks_list[0]): contrasts}
    else:
        contrasts_dict = contrasts
    del contrasts
    # Now it's always a dict
    bad_keys = set(contrasts_dict) - set(tasks_list)
    if bad_keys:
        raise ValueError(
            f"config.contrasts has keys {sorted(contrasts_dict)} which do not match "
            f"the defined tasks {tasks_list} via config.task={tasks!r}."
        )
    for task, these_contrasts in contrasts_dict.items():
        assert isinstance(these_contrasts, Sequence)
        for ci, contrast in enumerate(these_contrasts):
            where = f"for contrast for task={task!r} at index {ci}"
            if isinstance(contrast, tuple):
                if len(contrast) != 2:
                    raise ValueError(
                        "Contrasts' tuples MUST be two conditions, got "
                        f"{len(contrast)=} {where}"
                    )
            elif isinstance(contrast, dict):
                missing = ", ".join(
                    repr(m)
                    for m in sorted(_set_keys_arbitrary_contrast - set(contrast))
                )
                if missing:
                    raise ValueError(
                        f"Missing key{_pl(missing)} in contrast {contrast} {where}: "
                        f"{missing}"
                    )
                if len(contrast["conditions"]) != len(contrast["weights"]):
                    raise ValueError(
                        f"Contrast {contrast['name']} {where} has an "
                        f"inconsistent number of conditions/weights"
                    )
            else:
                raise ValueError("Contrasts must be tuples or well-formed dicts")


def _get_step_modules() -> dict[str, tuple[ModuleType, ...]]:
    from .steps import freesurfer, init, preprocessing, sensor, source

    INIT_STEPS = init._STEPS
    PREPROCESSING_STEPS = preprocessing._STEPS
    SENSOR_STEPS = sensor._STEPS
    SOURCE_STEPS = source._STEPS
    FREESURFER_STEPS = freesurfer._STEPS

    STEP_MODULES = {
        "init": INIT_STEPS,
        "preprocessing": PREPROCESSING_STEPS,
        "sensor": SENSOR_STEPS,
        "source": SOURCE_STEPS,
        "freesurfer": FREESURFER_STEPS,
    }

    # Do not include the FreeSurfer steps in "all" – we don't intend to run
    # recon-all by default!
    STEP_MODULES["all"] = (
        STEP_MODULES["init"]
        + STEP_MODULES["preprocessing"]
        + STEP_MODULES["sensor"]
        + STEP_MODULES["source"]
    )

    return STEP_MODULES


def _bids_kwargs(*, config: SimpleNamespace) -> dict[str, str | None]:
    """Get the standard BIDS config entries."""
    return dict(
        proc=config.proc,
        datatype=get_datatype(config),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        bids_root=config.bids_root,
        deriv_root=config.deriv_root,
        all_tasks=config.all_tasks,  # we compute this once and store it for brevity
    )


def _do_mf_autobad(*, cfg: SimpleNamespace) -> bool:
    return bool(cfg.find_noisy_channels_meg or cfg.find_flat_channels_meg)


# Adapted from MNE-Python
def _pl(x: int | np.generic | Sized, *, non_pl: str = "", pl: str = "s") -> str:
    """Determine if plural should be used."""
    len_x = x if isinstance(x, int | np.generic) else len(x)
    return non_pl if len_x == 1 else pl


def _proj_path(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
) -> BIDSPath:
    return BIDSPath(
        subject=subject,
        session=session,
        acquisition=cfg.acq,
        recording=cfg.rec,
        space=cfg.space,
        datatype=cfg.datatype,
        root=cfg.deriv_root,
        extension=".fif",
        suffix="proj",
        check=False,
    )


def _get_rank(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: str | None,
    inst: mne.io.BaseRaw | mne.BaseEpochs | mne.Covariance,
    info: mne.Info | None = None,
    log: bool = True,
) -> dict[str, int]:
    if cfg.cov_rank == "info":
        kwargs = dict(rank="info")
        from_where = "from info"
    else:
        assert isinstance(cfg.cov_rank, dict)
        kwargs = cfg.cov_rank
        from_where = "compute from data"
    if info is None:
        assert not isinstance(inst, mne.Covariance)
        info = inst.info
    rank = mne.compute_rank(inst, info=info, **kwargs)
    assert isinstance(rank, dict)
    if log:
        msg = f"Using rank {from_where}: {rank}"
        logger.info(**gen_log_kwargs(message=msg))
    return rank
