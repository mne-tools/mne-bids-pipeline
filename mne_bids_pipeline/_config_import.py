import ast
import copy
import difflib
import importlib.util
import os
import pathlib
import re
import types
from dataclasses import field
from functools import partial
from inspect import signature
from types import SimpleNamespace
from typing import Any

import matplotlib
import mne
import numpy as np
from mne_bids import get_entity_vals
from pydantic import BaseModel, ConfigDict, ValidationError

from ._config_utils import get_eog_channels, get_subjects_sessions
from ._logging import gen_log_kwargs, logger
from .typing import PathLike


class ConfigError(ValueError):
    pass


def _import_config(
    *,
    config_path: PathLike | None,
    overrides: SimpleNamespace | None = None,
    check: bool = True,
) -> SimpleNamespace:
    """Import the default config and the user's config."""
    # Get the default
    config = _get_default_config()
    # Public names users generally will have in their config
    valid_names = [d for d in dir(config) if not d.startswith("_")]
    # Names that we will reduce the SimpleConfig to before returning
    # (see _update_with_user_config)
    keep_names = [d for d in dir(config) if not d.startswith("__")] + [
        "config_path",
        "PIPELINE_NAME",
        "VERSION",
        "CODE_URL",
        "_raw_split_size",
        "_epochs_split_size",
    ]

    # Update with user config
    user_names = _update_with_user_config(
        config=config,
        config_path=config_path,
        overrides=overrides,
    )

    extra_exec_params_keys: tuple[str, ...] = ()
    extra_config = os.getenv("_MNE_BIDS_STUDY_TESTING_EXTRA_CONFIG", "")
    if extra_config:
        msg = f"With testing config: {extra_config}"
        logger.info(**gen_log_kwargs(message=msg, emoji="override"))
        extra_names = _update_config_from_path(
            config=config, config_path=extra_config, include_private=True
        )
        # Update valid_extra_names as needed if test configs in tests/test_run.py change
        valid_extra_names = set(
            (
                "_n_jobs",
                "_raw_split_size",
                "_epochs_split_size",
                "subjects_dir",  # test_session_specific_mri
                "deriv_root",  # test_session_specific_mri
                "Path",  # test_session_specific_mri
            )
        )
        assert set(extra_names) - valid_extra_names == set(), extra_names
        extra_exec_params_keys = tuple(set(["_n_jobs"]) & set(extra_names))
    keep_names.extend(extra_exec_params_keys)

    # Check it
    if check:
        _check_config(config, config_path)
        _check_misspellings_removals(
            valid_names=valid_names,
            user_names=user_names,
            config_validation=config.config_validation,
        )

    # Finally, reduce to our actual supported params (all keep_names should be present)
    config = SimpleNamespace(**{k: getattr(config, k) for k in keep_names})

    # Take some standard actions
    mne.set_log_level(verbose=config.mne_log_level.upper())

    # Take variables out of config (which affects the pipeline outputs) and
    # put into config.exec_params (which affect the pipeline execution methods,
    # but not the outputs)
    keys = (
        # Parallelization
        "n_jobs",
        "parallel_backend",
        "dask_temp_dir",
        "dask_worker_memory_limit",
        "dask_open_dashboard",
        # Interaction
        "on_error",
        "interactive",
        # Caching
        "memory_location",
        "memory_subdir",
        "memory_verbose",
        "memory_file_method",
        # Misc
        "deriv_root",
        "config_path",
    ) + extra_exec_params_keys
    in_both = {"deriv_root"}
    exec_params = SimpleNamespace(**{k: getattr(config, k) for k in keys})
    override_keys = ("subjects",)
    exec_params.overrides = SimpleNamespace(
        **{k: getattr(overrides, k) for k in override_keys if hasattr(overrides, k)}
    )
    for k in keys:
        if k not in in_both:
            delattr(config, k)
    config.exec_params = exec_params
    return config


def _get_default_config() -> SimpleNamespace:
    from . import _config

    # Don't use _config itself as it's mutable -- make a new object
    # with deepcopies of vals (keys are immutable strings so no need to copy)
    # except modules and imports
    tree = ast.parse(pathlib.Path(_config.__file__).read_text())
    ignore_keys = {
        name.asname or name.name
        for element in tree.body
        if isinstance(element, ast.Import | ast.ImportFrom)
        for name in element.names
    }
    config = SimpleNamespace(
        **{
            key: copy.deepcopy(val)
            for key, val in _config.__dict__.items()
            if not (key.startswith("__") or key in ignore_keys)
        }
    )
    return config


def _update_config_from_path(
    *,
    config: SimpleNamespace,
    config_path: PathLike,
    include_private: bool = False,
) -> list[str]:
    user_names = list()
    config_path = pathlib.Path(config_path).expanduser().resolve(strict=True)
    # Import configuration from an arbitrary path without having to fiddle
    # with `sys.path`.
    spec = importlib.util.spec_from_file_location(
        name="custom_config", location=config_path
    )
    assert spec is not None
    assert spec.loader is not None
    custom_cfg = importlib.util.module_from_spec(spec)
    assert custom_cfg is not None
    spec.loader.exec_module(custom_cfg)
    for key in dir(custom_cfg):
        if not key.startswith("__"):
            # don't validate private vars, but do add to config
            # (e.g., so that our hidden _raw_split_size is included)
            if include_private or not key.startswith("_"):
                user_names.append(key)
            val = getattr(custom_cfg, key)
            # Don't log about class types like ArbirtaryContrast, Ge, Path, etc.
            if not (
                isinstance(val, type)
                or isinstance(val, types.ModuleType)
                # there would be a way to detect if something is a type annotation but
                # I couldn't figure it out easily. This is debug level anyway so
                # hopefully not too bad to just maintain a blocklist here.
                or key in ("Annotated", "Path", "PathLike", "Literal", "FloatArrayLike")
            ):
                # Spacing here to match "Overriding" messages
                rep = repr(val).replace("\n", " ")
                rep = rep[:100] + "…" if len(rep) > 100 else rep
                msg = f"Assigning  {key} = {rep}"
                logger.debug(**gen_log_kwargs(message=msg, emoji="📝"))
            setattr(config, key, val)
    return user_names


def _update_with_user_config(
    *,
    config: SimpleNamespace,  # modified in-place
    config_path: PathLike | None,
    overrides: SimpleNamespace | None,
) -> list[str]:
    # 1. Basics and hidden vars
    from . import __version__

    config.PIPELINE_NAME = "mne-bids-pipeline"
    config.VERSION = __version__
    config.CODE_URL = "https://github.com/mne-tools/mne-bids-pipeline"
    config._raw_split_size = "2GB"
    config._epochs_split_size = "2GB"

    # 2. User config
    user_names = list()
    if config_path is not None:
        user_names.extend(
            _update_config_from_path(
                config=config,
                config_path=config_path,
            )
        )
    config.config_path = config_path

    # 3. Overrides via command-line switches
    overrides = overrides or SimpleNamespace()
    for name in dir(overrides):
        if not name.startswith("__"):
            val = getattr(overrides, name)
            msg = f"Overriding config.{name} = {repr(val)}"
            logger.info(**gen_log_kwargs(message=msg, emoji="override"))
            setattr(config, name, val)

    # 4. Env vars and other triaging
    if not config.bids_root:
        root = os.getenv("BIDS_ROOT", None)
        if root is None:
            raise ValueError(
                "You need to specify `bids_root` in your configuration, or "
                "define an environment variable `BIDS_ROOT` pointing to the "
                "root folder of your BIDS dataset"
            )
        config.bids_root = root
    config.bids_root = pathlib.Path(config.bids_root).expanduser().resolve()
    if config.deriv_root is None:
        config.deriv_root = config.bids_root / "derivatives" / config.PIPELINE_NAME
    config.deriv_root = pathlib.Path(config.deriv_root).expanduser().resolve()

    # 5. Consistency
    log_kwargs = dict(emoji="override")
    if config.interactive:
        if config.on_error != "debug":
            msg = 'Setting config.on_error="debug" because of interactive mode'
            logger.info(**gen_log_kwargs(message=msg, **log_kwargs))
        config.on_error = "debug"
    else:
        matplotlib.use("Agg")  # do not open any window  # noqa
    if config.on_error == "debug":
        if config.n_jobs != 1:
            msg = 'Setting config.n_jobs=1 because config.on_error="debug"'
            logger.info(**gen_log_kwargs(message=msg, **log_kwargs))
        config.n_jobs = 1
        if config.parallel_backend != "loky":
            msg = (
                'Setting config.parallel_backend="loky" because config.on_error="debug"'
            )
            logger.info(**gen_log_kwargs(message=msg, **log_kwargs))
        config.parallel_backend = "loky"
    return user_names


def _check_config(config: SimpleNamespace, config_path: PathLike | None) -> None:
    _pydantic_validate(config=config, config_path=config_path)

    # Eventually all of these could be pydantic-validated, but for now we'll
    # just change the ones that are easy

    config.bids_root.resolve(strict=True)
    if config.bids_root == config.deriv_root:
        raise ValueError(
            "bids_root and deriv_root cannot be the same directory "
            f"({config.bids_root})."
        )

    if (
        config.use_maxwell_filter
        and len(set(config.ch_types).intersection(("meg", "grad", "mag"))) == 0
    ):
        raise ValueError("Cannot use Maxwell filter without MEG channels.")
    mf_reserved_kwargs = (
        "raw",
        "calibration",
        "cross_talk",
        "st_duration",
        "st_correlation",
        "origin",
        "coord_frame",
        "destination",
        "head_pos",
        "extended_proj",
        "int_order",
        "ext_order",
    )
    # check `mf_extra_kws` for things that shouldn't be in there
    if duplicates := (set(config.mf_extra_kws) & set(mf_reserved_kwargs)):
        raise ConfigError(
            f"`mf_extra_kws` contains keys {', '.join(sorted(duplicates))} that are "
            "handled by dedicated config keys. Please remove them from `mf_extra_kws`."
        )
    # if `destination="twa"` make sure `mf_mc=True`
    if (
        isinstance(config.mf_destination, str)
        and config.mf_destination == "twa"
        and not config.mf_mc
    ):
        raise ConfigError(
            "cannot compute time-weighted average head position (mf_destination='twa') "
            "without movement compensation. Please set `mf_mc=True` in your config."
        )
    # if `dict` passed for ssp_ecg_channel, make sure its keys are valid
    if config.ssp_ecg_channel and isinstance(config.ssp_ecg_channel, dict):
        pattern = re.compile(r"^sub-[A-Za-z\d]+(_ses-[A-Za-z\d]+)?$")
        matches = set(filter(pattern.match, config.ssp_ecg_channel)).union("default")
        newline_indent = "\n  "
        if mismatch := (set(config.ssp_ecg_channel) - matches):
            raise ConfigError(
                "Malformed keys in ssp_ecg_channel dict:\n  "
                f"{newline_indent.join(sorted(mismatch))}"
            )
        # also make sure there are values for all subjects/sessions:
        missing = list()
        subjects_sessions = get_subjects_sessions(config)
        for sub, sessions in subjects_sessions.items():
            for ses in sessions:
                if (
                    config.ssp_ecg_channel.get(f"sub-{sub}") is None
                    and config.ssp_ecg_channel.get(f"sub-{sub}_ses-{ses}") is None
                ):
                    missing.append(
                        f"sub-{sub}" if ses is None else f"sub-{sub}_ses-{ses}"
                    )
        if missing:
            raise ConfigError(
                f"Missing entries in ssp_ecg_channel:\n  {newline_indent.join(missing)}"
            )

    # if `dict` passed for eog_channel, make sure its keys are valid
    if config.eog_channels and isinstance(config.eog_channels, dict):
        pattern = re.compile(r"^sub-[A-Za-z\d]+(_ses-[A-Za-z\d]+)?$")
        matches = set(filter(pattern.match, config.eog_channels)).union({"default"})
        newline_indent = "\n  "

        if mismatch := (set(config.eog_channels) - matches):
            raise ConfigError(
                "Malformed keys in eog_channels dict, "
                "must be sub-<subject> or sub-<subject>_ses-<session>:\n  "
                f"{newline_indent.join(sorted(repr(miss) for miss in mismatch))}"
            )
        # also make sure there are values for all subjects/sessions:
        missing = list()
        subjects_sessions = get_subjects_sessions(config)
        for sub, sessions in subjects_sessions.items():
            try:
                for sess in sessions:
                    get_eog_channels(config.eog_channels, subject=sub, session=sess)
            except KeyError:
                missing.append(f"sub-{sub}" if ses is None else f"sub-{sub}_ses-{ses}")

        if missing:
            raise ConfigError(
                f"Missing entries in eog_channels:\n  {newline_indent.join(missing)}"
            )

    reject = config.reject
    ica_reject = config.ica_reject
    if config.spatial_filter == "ica":
        effective_ica_l_freq = max([config.ica_l_freq or 0.0, config.l_freq or 0.0])
        if effective_ica_l_freq < 1:
            raise ValueError(
                "You requested to high-pass filter the data before ICA with "
                f"ica_l_freq={config.ica_l_freq} Hz. Please increase this "
                "setting to 1 Hz or above to ensure reliable ICA function."
            )
        if (
            config.ica_l_freq is not None
            and config.l_freq is not None
            and config.ica_l_freq < config.l_freq
        ):
            raise ValueError(
                "You requested a lower high-pass filter cutoff frequency for "
                f"ICA than for your raw data: ica_l_freq = {config.ica_l_freq}"
                f" < l_freq = {config.l_freq}. Adjust the cutoffs such that "
                "ica_l_freq >= l_freq, or set ica_l_freq to None if you do "
                "not wish to apply an additional high-pass filter before "
                "running ICA."
            )
        if (
            ica_reject is not None
            and reject is not None
            and reject not in ["autoreject_global", "autoreject_local"]
        ):
            for ch_type in reject:
                if ch_type in ica_reject and reject[ch_type] > ica_reject[ch_type]:
                    raise ValueError(
                        f'Rejection threshold in reject["{ch_type}"] '
                        f"({reject[ch_type]}) must be at least as stringent "
                        "as that in "
                        f'ica_reject["{ch_type}"] ({ica_reject[ch_type]})'
                    )

    if config.noise_cov == "emptyroom" and "eeg" in config.ch_types:
        raise ValueError(
            "You requested to process data that contains EEG channels. In "
            "this case, noise covariance can only be estimated from the "
            "experimental data, e.g., the pre-stimulus period. Please set "
            "noise_cov to (tmin, tmax)"
        )

    if config.noise_cov == "emptyroom" and not config.process_empty_room:
        raise ValueError(
            "You requested noise covariance estimation from empty-room "
            'recordings by setting noise_cov = "emptyroom", but you did not '
            "enable empty-room data processing. "
            "Please set process_empty_room = True"
        )

    if config.noise_cov == "raw" and not config.process_raw_clean:
        raise ValueError(
            "You requested noise covariance estimation from raw data by "
            '"setting noise_cov = "raw", but you did not enable '
            "writing cleaned raw data. Please set process_raw_clean = True"
        )

    if (
        config.allow_missing_sessions
        and "ignore_suffixes" not in signature(get_entity_vals).parameters
    ):
        raise ConfigError(
            "You've requested to `allow_missing_sessions`, but this functionality "
            "requires a newer version of `mne_bids` than you have available. Please "
            "update MNE-BIDS (or if on the latest version, install the dev version)."
        )

    bl = config.baseline
    if bl is not None:
        if (bl[0] is not None and bl[0] < config.epochs_tmin) or (
            bl[1] is not None and bl[1] > config.epochs_tmax
        ):
            raise ValueError(
                f"baseline {bl} outside of epochs interval "
                f"{[config.epochs_tmin, config.epochs_tmax]}."
            )

        if bl[0] is not None and bl[1] is not None and bl[0] >= bl[1]:
            raise ValueError(
                f"The end of the baseline period must occur after its start, "
                f"but you set baseline={bl}"
            )

    # check cluster permutation parameters
    if config.cluster_n_permutations < 10 / config.cluster_permutation_p_threshold:
        raise ValueError(
            "cluster_n_permutations is not big enough to calculate "
            "the p-values accurately."
        )

    # Another check that depends on some of the functions defined above
    if not config.task_is_rest and config.conditions is None:
        raise ValueError(
            "Please indicate the name of your conditions in your "
            "configuration. Currently the `conditions` parameter is empty. "
            "This is only allowed for resting-state analysis."
        )

    if not isinstance(config.mf_destination, str):
        destination = np.array(config.mf_destination, float)
        if destination.shape != (4, 4):
            raise ValueError(
                "config.mf_destination, if array-like, must have shape (4, 4) "
                f"but got shape {destination.shape}"
            )

    # MNE-ICALabel
    if config.ica_use_icalabel:
        pre = "When using MNE-ICALabel, you must set"
        if config.ica_l_freq != 1.0 or config.ica_h_freq != 100.0:
            raise ValueError(
                f"{pre} ica_l_freq=1 and h_freq=100, "
                f"but got: ica_l_freq={config.ica_l_freq} and "
                f"ica_h_freq={config.ica_h_freq}"
            )
        if config.eeg_reference != "average":
            raise ValueError(
                f'{pre} eeg_reference="average", but got: '
                f"eeg_reference={config.eeg_reference}"
            )
        if config.ica_algorithm not in ("picard-extended_infomax", "extended_infomax"):
            raise ValueError(
                f'{pre} ica_algorithm="picard-extended_infomax" or "extended_infomax", '
                f"but got: ica_algorithm={repr(config.ica_algorithm)}"
            )


def _default_factory(key: str, val: Any) -> Any:
    # convert a default to a default factory if needed, having an explicit
    # allowlist of non-empty ones
    allowlist = [
        {"n_mag": 1, "n_grad": 1, "n_eeg": 1},  # n_proj_*
        {"custom": (8, 24.0, 40)},  # decoding_csp_freqs
        ["evoked"],  # inverse_targets
        [4, 8, 16],  # autoreject_n_interpolate
        ("brain", "other"),  # ica_icalabel_include
        {
            "brain": 0.8,
            "muscle artifact": 0.8,
            "eye blink": 0.8,
            "heart beat": 0.8,
            "line noise": 0.8,
            "channel noise": 0.8,
            "other": 0.8,
        },  # ica_exclusion_thresholds
        {
            "brain": 0.3,
            "muscle artifact": 0.3,
            "eye blink": 0.3,
            "heart beat": 0.3,
            "line noise": 0.3,
            "channel noise": 0.3,
            "other": 0.3,
        },  # ica_class_thresholds
    ]

    def default_factory() -> Any:
        return val

    for typ in (dict, list):
        if isinstance(val, typ):
            try:
                idx = allowlist.index(val)
            except ValueError:
                assert val == typ(), (key, val)
                default_factory = typ
            else:
                if typ is dict:
                    default_factory = partial(typ, **allowlist[idx])  # type: ignore
                else:
                    assert typ is list
                    default_factory = partial(typ, allowlist[idx])
            return field(default_factory=default_factory)
    return val


def _pydantic_validate(
    config: SimpleNamespace,
    config_path: PathLike | None,
) -> None:
    """Create dataclass from config type hints and validate with pydantic."""
    # https://docs.pydantic.dev/latest/usage/dataclasses/
    from . import _config as root_config

    # Modify annotations to add nested strict parsing
    annotations = dict()
    attrs = dict()
    for key, annot in root_config.__annotations__.items():
        annotations[key] = annot
        attrs[key] = _default_factory(key, root_config.__dict__[key])
    name = "user configuration"
    if config_path is not None:
        name += f" from {config_path}"
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # needed in 2.6.0 to allow DigMontage for example
        validate_assignment=True,
        strict=True,  # do not allow float for int for example
        extra="forbid",
    )
    UserConfig = type(
        name,
        (BaseModel,),
        {"__annotations__": annotations, "model_config": model_config, **attrs},
    )
    # Now use pydantic to automagically validate
    user_vals = {key: val for key, val in config.__dict__.items() if key in annotations}
    try:
        UserConfig.model_validate(user_vals)  # type: ignore[attr-defined]
    except ValidationError as err:
        raise ValueError(str(err)) from None


_REMOVED_NAMES: dict[str, dict[str, str | None]] = {
    "debug": dict(
        new_name="on_error",
        instead='use on_error="debug" instead',
    ),
    "decim": dict(
        new_name="epochs_decim",
        instead=None,
    ),
    "resample_sfreq": dict(
        new_name="raw_resample_sfreq",
    ),
    "N_JOBS": dict(
        new_name="n_jobs",
    ),
    "ica_ctps_ecg_threshold": dict(
        new_name="ica_ecg_threshold",
    ),
}


def _check_misspellings_removals(
    *,
    valid_names: list[str],
    user_names: list[str],
    config_validation: str,
) -> None:
    # for each name in the user names, check if it's in the valid names but
    # the correct one is not defined
    for user_name in user_names:
        if user_name not in valid_names:
            # find the closest match
            closest_match = difflib.get_close_matches(user_name, valid_names, n=1)
            msg = f"Found a variable named {repr(user_name)} in your custom config,"
            if closest_match and closest_match[0] not in user_names:
                this_msg = (
                    f"{msg} did you mean {repr(closest_match[0])}? "
                    "If so, please correct the error. If not, please rename "
                    "the variable to reduce ambiguity and avoid this message, "
                    "or set config.config_validation to 'warn' or 'ignore'."
                )
                _handle_config_error(this_msg, config_validation)
            if user_name in _REMOVED_NAMES:
                new = _REMOVED_NAMES[user_name]["new_name"]
                if new not in user_names:
                    instead = _REMOVED_NAMES[user_name].get("instead", None)
                    if instead is None:
                        instead = f"use {new} instead"
                    this_msg = (
                        f"{msg} this variable has been removed as a valid "
                        f"config option, {instead}."
                    )
                    _handle_config_error(this_msg, config_validation)


def _handle_config_error(
    msg: str,
    config_validation: str,
) -> None:
    if config_validation == "raise":
        raise ValueError(msg)
    logger_call = logger.warning if config_validation == "warn" else logger.debug
    logger_call(**gen_log_kwargs(message=msg, emoji="🛟"))
