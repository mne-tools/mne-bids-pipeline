"""Utilities for mangling config vars."""

import copy
import functools
import pathlib
from collections.abc import Iterable
from types import ModuleType, SimpleNamespace
from typing import Any, Literal, Optional, TypeVar, Union

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
            'When specifying a "deriv_root", you must also supply a ' '"subjects_dir".'
        )

    if not config.subjects_dir:
        return config.bids_root / "derivatives" / "freesurfer" / "subjects"
    else:
        return pathlib.Path(config.subjects_dir).expanduser().resolve()


def get_fs_subject(config: SimpleNamespace, subject: str) -> str:
    subjects_dir = get_fs_subjects_dir(config)

    if config.use_template_mri is not None:
        return config.use_template_mri

    if (pathlib.Path(subjects_dir) / subject).exists():
        return subject
    else:
        return f"sub-{subject}"


@functools.cache
def _get_entity_vals_cached(*args, **kwargs) -> list[str]:
    return mne_bids.get_entity_vals(*args, **kwargs)


def get_datatype(config: SimpleNamespace) -> Literal["meg", "eeg"]:
    # Content of ch_types should be sanitized already, so we don't need any
    # extra sanity checks here.
    if config.data_type is not None:
        return config.data_type
    elif config.data_type is None and config.ch_types == ["eeg"]:
        return "eeg"
    elif config.data_type is None and any(
        [t in ["meg", "mag", "grad"] for t in config.ch_types]
    ):
        return "meg"
    else:
        raise RuntimeError(
            "This probably shouldn't happen, got "
            f"config.data_type={repr(config.data_type)} and "
            f"config.ch_types={repr(config.ch_types)} "
            "but could not determine the datatype. Please contact "
            "the MNE-BIDS-pipeline developers. Thank you."
        )


@functools.cache
def _get_datatypes_cached(root):
    return mne_bids.get_datatypes(root=root)


def _get_ignore_datatypes(config: SimpleNamespace) -> tuple[str]:
    _all_datatypes: list[str] = _get_datatypes_cached(root=config.bids_root)
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

    # Preserve order and remove excluded subjects
    subjects = [
        subject
        for subject in s
        if subject not in config.exclude_subjects and subject != "emptyroom"
    ]

    return subjects


def get_sessions(config: SimpleNamespace) -> Union[list[None], list[str]]:
    sessions = copy.deepcopy(config.sessions)
    _all_sessions = _get_entity_vals_cached(
        root=config.bids_root,
        entity_key="session",
        ignore_datatypes=_get_ignore_datatypes(config),
    )
    if sessions == "all":
        sessions = _all_sessions

    if not sessions:
        return [None]
    else:
        return sessions


def get_runs_all_subjects(
    config: SimpleNamespace,
) -> dict[str, Union[list[None], list[str]]]:
    """Give the mapping between subjects and their runs.

    Returns
    -------
    a dict of runs present in the bids_path
    for each subject asked in the configuration file
    (and not for each subject present in the bids_path).
    """
    # Use caching under the hood for speed
    return copy.deepcopy(
        _get_runs_all_subjects_cached(
            bids_root=config.bids_root,
            data_type=config.data_type,
            ch_types=tuple(config.ch_types),
            subjects=tuple(config.subjects) if config.subjects != "all" else "all",
            exclude_subjects=tuple(config.exclude_subjects),
            exclude_runs=tuple(config.exclude_runs) if config.exclude_runs else None,
        )
    )


@functools.cache
def _get_runs_all_subjects_cached(
    **config_dict: dict[str, Any],
) -> dict[str, Union[list[None], list[str]]]:
    config = SimpleNamespace(**config_dict)
    # Sometimes we check list equivalence for ch_types, so convert it back
    config.ch_types = list(config.ch_types)
    subj_runs = dict()
    for subject in get_subjects(config):
        # Only traverse through the current subject's directory
        valid_runs_subj = _get_entity_vals_cached(
            config.bids_root / f"sub-{subject}",
            entity_key="run",
            ignore_datatypes=_get_ignore_datatypes(config),
        )

        # If we don't have any `run` entities, just set it to None, as we
        # commonly do when creating a BIDSPath.
        if not valid_runs_subj:
            valid_runs_subj = [None]

        if subject in (config.exclude_runs or {}):
            valid_runs_subj = [
                r for r in valid_runs_subj if r not in config.exclude_runs[subject]
            ]
        subj_runs[subject] = valid_runs_subj

    return subj_runs


def get_intersect_run(config: SimpleNamespace) -> list[str]:
    """Return the intersection of all the runs of all subjects."""
    subj_runs = get_runs_all_subjects(config)
    # Do not use something like:
    # list(set.intersection(*map(set, subj_runs.values())))
    # as it will not preserve order. Instead just be explicit and preserve order.
    # We could use "sorted", but it's probably better to use the order provided by
    # the user (if they want to put `runs=["02", "01"]` etc. it's better to use "02")
    all_runs = list()
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
) -> Union[list[str], list[None]]:
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
                f"Invalid run. It can be a subset of {valid_runs} but " f"got {runs}"
            )
    return runs


def get_runs_tasks(
    *,
    config: SimpleNamespace,
    subject: str,
    session: Optional[str],
    which: tuple[str] = ("runs", "noise", "rest"),
) -> list[tuple[str]]:
    """Get (run, task) tuples for all runs plus (maybe) rest."""
    from ._import_data import _get_noise_path, _get_rest_path

    assert isinstance(which, tuple)
    assert all(isinstance(inc, str) for inc in which)
    assert all(inc in ("runs", "noise", "rest") for inc in which)
    runs = list()
    tasks = list()
    if "runs" in which:
        runs.extend(get_runs(config=config, subject=subject))
        tasks.extend([get_task(config=config)] * len(runs))
    kwargs = dict(
        cfg=config,
        subject=subject,
        session=session,
        kind="orig",
        add_bads=False,
    )
    if "rest" in which and _get_rest_path(**kwargs):
        runs.append(None)
        tasks.append("rest")
    if "noise" in which:
        mf_reference_run = get_mf_reference_run(config=config)
        if _get_noise_path(mf_reference_run=mf_reference_run, **kwargs):
            runs.append(None)
            tasks.append("noise")
    return tuple(zip(runs, tasks))


def get_mf_reference_run(config: SimpleNamespace) -> str:
    # Retrieve to run identifier (number, name) of the reference run
    if config.mf_reference_run is not None:
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
    if inter_runs:
        return inter_runs[0]
    else:
        raise ValueError(
            f"The intersection of runs by subjects is empty. "
            f"Check the list of runs: "
            f"{get_runs_all_subjects(config)}"
        )


def get_task(config: SimpleNamespace) -> Optional[str]:
    task = config.task
    if task:
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


def get_channels_to_analyze(info: mne.Info, config: SimpleNamespace) -> list[str]:
    # Return names of the channels of the channel types we wish to analyze.
    # We also include channels marked as "bad" here.
    # `exclude=[]`: keep "bad" channels, too.
    kwargs = dict(eog=True, ecg=True, exclude=())
    if get_datatype(config) == "meg" and _meg_in_ch_types(config.ch_types):
        pick_idx = mne.pick_types(info, **kwargs)

        if "mag" in config.ch_types:
            pick_idx = np.concatenate(
                [pick_idx, mne.pick_types(info, meg="mag", exclude=[])]
            )
        if "grad" in config.ch_types:
            pick_idx = np.concatenate(
                [pick_idx, mne.pick_types(info, meg="grad", exclude=[])]
            )
        if "meg" in config.ch_types:
            pick_idx = mne.pick_types(info, meg=True, exclude=[])
        pick_idx.sort()
    elif config.ch_types == ["eeg"]:
        pick_idx = mne.pick_types(
            info, meg=False, eeg=True, eog=True, ecg=True, exclude=[]
        )
    else:
        raise RuntimeError(
            "Something unexpected happened. Please contact "
            "the mne-bids-pipeline developers. Thank you."
        )

    ch_names = [info["ch_names"][i] for i in pick_idx]
    return ch_names


def sanitize_cond_name(cond: str) -> str:
    cond = cond.replace("/", "").replace("_", "").replace("-", "").replace(" ", "")
    return cond


def get_mf_cal_fname(
    *, config: SimpleNamespace, subject: str, session: str
) -> pathlib.Path:
    if config.mf_cal_fname is None:
        bids_path = BIDSPath(
            subject=subject,
            session=session,
            suffix="meg",
            datatype="meg",
            root=config.bids_root,
        ).match()[0]
        mf_cal_fpath = bids_path.meg_calibration_fpath
        if mf_cal_fpath is None:
            raise ValueError(
                "Could not determine Maxwell Filter Calibration file from BIDS "
                f"definition for file {bids_path}."
            )
    else:
        mf_cal_fpath = pathlib.Path(config.mf_cal_fname).expanduser().absolute()
        if not mf_cal_fpath.exists():
            raise ValueError(
                f"Could not find Maxwell Filter Calibration "
                f"file at {str(mf_cal_fpath)}."
            )

    return mf_cal_fpath


def get_mf_ctc_fname(
    *, config: SimpleNamespace, subject: str, session: str
) -> pathlib.Path:
    if config.mf_ctc_fname is None:
        mf_ctc_fpath = BIDSPath(
            subject=subject,
            session=session,
            suffix="meg",
            datatype="meg",
            root=config.bids_root,
        ).meg_crosstalk_fpath
        if mf_ctc_fpath is None:
            raise ValueError("Could not find Maxwell Filter cross-talk " "file.")
    else:
        mf_ctc_fpath = pathlib.Path(config.mf_ctc_fname).expanduser().absolute()
        if not mf_ctc_fpath.exists():
            raise ValueError(
                f"Could not find Maxwell Filter cross-talk "
                f"file at {str(mf_ctc_fpath)}."
            )

    return mf_ctc_fpath


RawEpochsEvokedT = TypeVar(
    "RawEpochsEvokedT", bound=Union[mne.io.BaseRaw, mne.BaseEpochs, mne.Evoked]
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


def _get_bem_conductivity(cfg: SimpleNamespace) -> tuple[tuple[float], str]:
    if cfg.fs_subject in ("fsaverage", cfg.use_template_mri):
        conductivity = None  # should never be used
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
    cfg: SimpleNamespace, subject: str, session: Optional[str]
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


def _get_decoding_proc(config: SimpleNamespace) -> Optional[str]:
    return _EPOCHS_DESCRIPTION_TO_PROC_MAP[config.decoding_which_epochs]


def get_eeg_reference(
    config: SimpleNamespace,
) -> Union[Literal["average"], Iterable[str]]:
    if config.eeg_reference == "average":
        return config.eeg_reference
    elif isinstance(config.eeg_reference, str):
        return [config.eeg_reference]
    else:
        return config.eeg_reference


def _validate_contrasts(contrasts: SimpleNamespace) -> None:
    for contrast in contrasts:
        if isinstance(contrast, tuple):
            if len(contrast) != 2:
                raise ValueError("Contrasts' tuples MUST be two conditions")
        elif isinstance(contrast, dict):
            if not _set_keys_arbitrary_contrast.issubset(set(contrast.keys())):
                raise ValueError(f"Missing key(s) in contrast {contrast}")
            if len(contrast["conditions"]) != len(contrast["weights"]):
                raise ValueError(
                    f"Contrast {contrast['name']} has an "
                    f"inconsistent number of conditions/weights"
                )
        else:
            raise ValueError("Contrasts must be tuples or well-formed dicts")


def _get_step_modules() -> dict[str, tuple[ModuleType]]:
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


def _bids_kwargs(*, config: SimpleNamespace) -> dict:
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
    return cfg.find_noisy_channels_meg or cfg.find_flat_channels_meg


# Adapted from MNE-Python
def _pl(x, *, non_pl="", pl="s"):
    """Determine if plural should be used."""
    len_x = x if isinstance(x, (int, np.generic)) else len(x)
    return non_pl if len_x == 1 else pl


def _proj_path(
    *,
    cfg: SimpleNamespace,
    subject: str,
    session: Optional[str],
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
