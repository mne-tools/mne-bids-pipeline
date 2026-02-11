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
from .typing import ArbitraryContrast

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
    if config.task == "":
        ignore_tasks = None
    else:
        all_tasks = _get_entity_vals_cached(
            root=config.bids_root,
            entity_key="task",
            ignore_datatypes=ignore_datatypes,
        )
        ignore_tasks = tuple(set(all_tasks) - set([config.task]))

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


def get_runs_all_subjects(
    config: SimpleNamespace,
) -> dict[str, tuple[None] | tuple[str, ...]]:
    """Give the mapping between subjects and their runs.

    Returns
    -------
    a dict of runs present in the bids_path
    for each subject asked in the configuration file
    (and not for each subject present in the bids_path).
    """
    # Use caching under the hood for speed
    return _get_runs_all_subjects_cached(
        bids_root=config.bids_root,
        data_type=config.data_type,
        ch_types=tuple(config.ch_types),
        subjects=tuple(config.subjects) if config.subjects != "all" else "all",
        exclude_subjects=tuple(config.exclude_subjects),
        exclude_runs=tuple(config.exclude_runs) if config.exclude_runs else None,
    )


@functools.cache
def _get_runs_all_subjects_cached(
    **config_dict: dict[str, Any],
) -> dict[str, tuple[None] | tuple[str, ...]]:
    config = SimpleNamespace(**config_dict)
    # Sometimes we check list equivalence for ch_types, so convert it back
    config.ch_types = list(config.ch_types)
    subj_runs: dict[str, tuple[None] | tuple[str, ...]] = dict()
    for subject in get_subjects(config):
        # Only traverse through the current subject's directory
        valid_runs_subj = _get_entity_vals_cached(
            config.bids_root / f"sub-{subject}",
            entity_key="run",
            ignore_datatypes=_get_ignore_datatypes(config),
        )

        # If we don't have any `run` entities, just set it to None, as we
        # commonly do when creating a BIDSPath.
        if valid_runs_subj:
            if subject in (config.exclude_runs or {}):
                valid_runs_subj = tuple(
                    r for r in valid_runs_subj if r not in config.exclude_runs[subject]
                )
            subj_runs[subject] = valid_runs_subj
        else:
            subj_runs[subject] = (None,)

    return subj_runs


def get_intersect_run(config: SimpleNamespace) -> list[str | None]:
    """Return the intersection of all the runs of all subjects."""
    subj_runs = get_runs_all_subjects(config)
    # Do not use something like:
    # list(set.intersection(*map(set, subj_runs.values())))
    # as it will not preserve order. Instead just be explicit and preserve order.
    # We could use "sorted", but it's probably better to use the order provided by
    # the user (if they want to put `runs=["02", "01"]` etc. it's better to use "02")
    all_runs: list[str | None] = list()
    for runs in subj_runs.values():
        for run in runs:
            if run not in all_runs:
                all_runs.append(run)
    return all_runs


def get_runs(
    *,
    config: SimpleNamespace,
    subject: str,
    verbose: bool = False,
) -> list[str] | list[None]:
    """Return a list of runs in the BIDS input data.

    Parameters
    ----------
    subject
        Returns a list of the runs of this subject.
    verbose
        Notify if different subjects do not share the same runs.

    Returns
    -------
    The list of runs of the subject. If no BIDS `run` entity could be found,
    returns `[None]`.
    """
    if subject == "average":  # Used when creating the report
        return [None]

    runs = copy.deepcopy(config.runs)
    subj_runs = get_runs_all_subjects(config=config)
    valid_runs = subj_runs[subject]

    if len(get_subjects(config)) > 1:
        # Notify if different subjects do not share the same runs

        same_runs = True
        for runs_sub_i in subj_runs.values():
            if set(runs_sub_i) != set(list(subj_runs.values())[0]):
                same_runs = False

        if not same_runs and verbose:
            msg = (
                "Extracted all the runs. "
                "Beware, not all subjects share the same "
                "set of runs."
            )
            logger.info(**gen_log_kwargs(message=msg))

    if runs == "all":
        runs = list(valid_runs)

    if not runs:
        runs = [None]
    else:
        inclusion = set(runs).issubset(set(valid_runs))
        if not inclusion:
            raise ValueError(
                f"Invalid run. It can be a subset of {valid_runs} but got {runs}"
            )
    runs_out = list(runs)
    if runs_out != [None]:
        runs_out = list(str(x) for x in runs_out)
    return runs_out


def get_runs_tasks(
    *,
    config: SimpleNamespace,
    subject: str,
    session: str | None,
    which: tuple[str, ...] = ("runs", "noise", "rest"),
) -> tuple[tuple[str | None, str | None], ...]:
    """Get (run, task) tuples for all runs plus (maybe) rest."""
    from ._import_data import _get_noise_path, _get_rest_path

    assert isinstance(which, tuple)
    assert all(isinstance(inc, str) for inc in which)
    assert all(inc in ("runs", "noise", "rest") for inc in which)
    runs: list[str | None] = list()
    tasks: list[str | None] = list()
    if "runs" in which:
        runs.extend(get_runs(config=config, subject=subject))
        tasks.extend([get_task(config=config)] * len(runs))
    if "rest" in which:
        rest_path = _get_rest_path(
            cfg=config,
            subject=subject,
            session=session,
            kind="orig",
        )
        if rest_path:
            runs.append(None)
            tasks.append("rest")
    if "noise" in which:
        mf_reference_run = get_mf_reference_run(config=config)
        noise_path = _get_noise_path(
            mf_reference_run=mf_reference_run,
            cfg=config,
            subject=subject,
            session=session,
            kind="orig",
        )
        if noise_path:
            runs.append(None)
            tasks.append("noise")
    return tuple(zip(runs, tasks))


def get_mf_reference_run(config: SimpleNamespace) -> str | None:
    # Retrieve to run identifier (number, name) of the reference run
    if config.mf_reference_run is not None:
        assert isinstance(config.mf_reference_run, str), type(config.mf_reference_run)
        return config.mf_reference_run
    # Use the first run
    inter_runs = get_intersect_run(config)
    mf_ref_error = (config.mf_reference_run is not None) and (
        config.mf_reference_run not in inter_runs
    )
    if mf_ref_error:
        msg = (
            f"You set mf_reference_run={config.mf_reference_run}, but your "
            f"dataset only contains the following runs: {inter_runs}"
        )
        raise ValueError(msg)
    if not inter_runs:
        raise ValueError(
            f"The intersection of runs by subjects is empty. "
            f"Check the list of runs: "
            f"{get_runs_all_subjects(config)}"
        )
    return inter_runs[0]


def get_task(config: SimpleNamespace) -> str | None:
    task = config.task
    if task:
        assert isinstance(task, str), type(task)
        return task
    _valid_tasks = _get_entity_vals_cached(
        root=config.bids_root,
        entity_key="task",
        ignore_datatypes=_get_ignore_datatypes(config),
    )
    if not _valid_tasks:
        return None
    else:
        return _valid_tasks[0]


def _get_ss(
    *,
    config: SimpleNamespace,
) -> list[tuple[str, str | None]]:
    return [
        (subject, session)
        for subject, sessions in get_subjects_sessions(config).items()
        for session in sessions
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
        task=cfg.task,
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


def get_all_contrasts(config: SimpleNamespace) -> Iterable[ArbitraryContrast]:
    _validate_contrasts(config.contrasts)
    normalized_contrasts = []
    for contrast in config.contrasts:
        if isinstance(contrast, tuple):
            normalized_contrasts.append(
                ArbitraryContrast(
                    name=(contrast[0] + "+" + contrast[1]),
                    conditions=list(contrast),
                    weights=[1, -1],
                )
            )
        else:
            normalized_contrasts.append(contrast)
    return normalized_contrasts


def get_decoding_contrasts(config: SimpleNamespace) -> Iterable[tuple[str, str]]:
    _validate_contrasts(config.contrasts)
    normalized_contrasts = []
    for contrast in config.contrasts:
        if isinstance(contrast, tuple):
            normalized_contrasts.append(contrast)
        else:
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


def _validate_contrasts(contrasts: list[tuple[str, str] | dict[str, Any]]) -> None:
    for contrast in contrasts:
        if isinstance(contrast, tuple):
            if len(contrast) != 2:
                raise ValueError("Contrasts' tuples MUST be two conditions")
        elif isinstance(contrast, dict):
            missing = ", ".join(
                repr(m) for m in sorted(_set_keys_arbitrary_contrast - set(contrast))
            )
            if missing:
                raise ValueError(
                    f"Missing key{_pl(missing)} in contrast {contrast}: {missing}"
                )
            if len(contrast["conditions"]) != len(contrast["weights"]):
                raise ValueError(
                    f"Contrast {contrast['name']} has an "
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
        task=get_task(config),
        datatype=get_datatype(config),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        bids_root=config.bids_root,
        deriv_root=config.deriv_root,
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
        task=cfg.task,
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
