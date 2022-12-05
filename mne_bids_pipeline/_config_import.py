import ast
import copy
import difflib
import importlib
import os
import pathlib
from types import SimpleNamespace
from typing import Optional, List

import matplotlib
import mne
from mne.utils import _check_option, _validate_type

from ._logging import logger, gen_log_kwargs
from .typing import PathLike


def _import_config(
    *,
    config_path: Optional[PathLike],
    overrides: Optional[SimpleNamespace] = None,
    check: bool = True,
    log: bool = True,
) -> SimpleNamespace:
    """Import the default config and the user's config."""
    # Get the default
    config = _get_default_config()
    valid_names = [d for d in dir(config) if not d.startswith('_')]

    # Update with user config
    user_names = _update_with_user_config(
        config=config,
        config_path=config_path,
        overrides=overrides,
        log=log,
    )

    # Check it
    if check:
        _check_config(config)
        _check_misspellings_removals(
            config,
            valid_names=valid_names,
            user_names=user_names,
            log=log,
        )

    # Take some standard actions
    mne.set_log_level(verbose=config.mne_log_level.upper())

    # Take variables out of config (which affects the pipeline outputs) and
    # put into config.exec_params (which affect the pipeline execution methods,
    # but not the outputs)
    keys = (
        # Parallelization
        'N_JOBS', 'parallel_backend',
        'dask_temp_dir', 'dask_worker_memory_limit', 'dask_open_dashboard',
        # Interaction
        'on_error', 'interactive',
        # Caching
        'memory_location', 'memory_verbose', 'memory_file_method',
        # Misc
        'deriv_root', 'config_path',
    )
    in_both = {'deriv_root'}
    exec_params = SimpleNamespace(**{k: getattr(config, k) for k in keys})
    for k in keys:
        if k not in in_both:
            delattr(config, k)
    config.exec_params = exec_params
    return config


def _get_default_config():
    from . import _config
    # Don't use _config itself as it's mutable -- make a new object
    # with deepcopies of vals (keys are immutable strings so no need to copy)
    # except modules and imports
    tree = ast.parse(pathlib.Path(_config.__file__).read_text())
    ignore_keys = {
        name.asname or name.name
        for element in tree.body
        if isinstance(element, (ast.Import, ast.ImportFrom))
        for name in element.names
    }
    config = SimpleNamespace(
        **{key: copy.deepcopy(val) for key, val in _config.__dict__.items()
           if not (key.startswith('__') or key in ignore_keys)})
    return config


def _update_with_user_config(
    *,
    config: SimpleNamespace,  # modified in-place
    config_path: Optional[PathLike],
    overrides: Optional[SimpleNamespace],
    log: bool = False,
) -> List[str]:
    # 1. Basics and hidden vars
    from . import __version__
    config.PIPELINE_NAME = 'mne-bids-pipeline'
    config.VERSION = __version__
    config.CODE_URL = 'https://github.com/mne-tools/mne-bids-pipeline'
    config._raw_split_size = '2GB'
    config._epochs_split_size = '2GB'

    # 2. User config
    user_names = list()
    if config_path is not None:
        config_path = pathlib.Path(
            config_path).expanduser().resolve(strict=True)
        # Import configuration from an arbitrary path without having to fiddle
        # with `sys.path`.
        spec = importlib.util.spec_from_file_location(
            name='custom_config', location=config_path)
        custom_cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_cfg)
        for key in dir(custom_cfg):
            if not key.startswith('__'):
                # don't validate private vars, but do add to config
                # (e.g., so that our hidden _raw_split_size is included)
                if not key.startswith('_'):
                    user_names.append(key)
                val = getattr(custom_cfg, key)
                logger.debug('Overwriting: %s -> %s' % (key, val))
                setattr(config, key, val)
    config.config_path = config_path

    # 3. Overrides via command-line switches
    overrides = overrides or SimpleNamespace()
    for name in dir(overrides):
        if not name.startswith('__'):
            val = getattr(overrides, name)
            if log:
                msg = f'Overridding config.{name} = {repr(val)}'
                logger.info(
                    **gen_log_kwargs(
                        message=msg, step='', emoji='override', box='╶╴'
                    )
                )
            setattr(config, name, val)

    # 4. Env vars and other triaging
    if not config.bids_root:
        root = os.getenv('BIDS_ROOT', None)
        if root is None:
            raise ValueError(
                'You need to specify `bids_root` in your configuration, or '
                'define an environment variable `BIDS_ROOT` pointing to the '
                'root folder of your BIDS dataset')
        config.bids_root = root
    config.bids_root = \
        pathlib.Path(config.bids_root).expanduser().resolve()
    if config.deriv_root is None:
        config.deriv_root = \
            config.bids_root / 'derivatives' / config.PIPELINE_NAME
    config.deriv_root = pathlib.Path(config.deriv_root).expanduser().resolve()

    # 5. Consistency
    log_kwargs = dict(emoji='override', box='  ', step='')
    if config.interactive:
        if log and config.on_error != 'debug':
            msg = 'Setting config.on_error="debug" because of interactive mode'
            logger.info(**gen_log_kwargs(message=msg, **log_kwargs))
        config.on_error = 'debug'
    else:
        matplotlib.use('Agg')  # do not open any window  # noqa
    if config.on_error == 'debug':
        if log and config.N_JOBS != 1:
            msg = 'Setting config.N_JOBS=1 because config.on_error="debug"'
            logger.info(**gen_log_kwargs(message=msg, **log_kwargs))
        config.N_JOBS = 1
        if log and config.parallel_backend != 'loky':
            msg = (
                'Setting config.parallel_backend="loky" because '
                'config.on_error="debug"'
            )
            logger.info(**gen_log_kwargs(message=msg, **log_kwargs))
        config.parallel_backend = 'loky'
    return user_names


def _check_config(config: SimpleNamespace) -> None:
    # TODO: Use pydantic to do these validations
    # https://github.com/mne-tools/mne-bids-pipeline/issues/646
    _check_option(
        'config.parallel_backend', config.parallel_backend, ('dask', 'loky'))

    config.bids_root.resolve(strict=True)

    if (config.use_maxwell_filter and
            len(set(
                config.ch_types).intersection(('meg', 'grad', 'mag'))) == 0):
        raise ValueError('Cannot use Maxwell filter without MEG channels.')

    reject = config.reject
    ica_reject = config.ica_reject
    if config.spatial_filter == 'ica':
        _check_option(
            'config.ica_algorithm', config.ica_algorithm,
            ('picard', 'fastica', 'extended_infomax'))

        if config.ica_l_freq < 1:
            raise ValueError(
                'You requested to high-pass filter the data before ICA with '
                f'ica_l_freq={config.ica_l_freq} Hz. Please increase this '
                'setting to 1 Hz or above to ensure reliable ICA function.')
        if (config.ica_l_freq is not None and
                config.l_freq is not None and
                config.ica_l_freq < config.l_freq):
            raise ValueError(
                'You requested a lower high-pass filter cutoff frequency for '
                f'ICA than for your raw data: ica_l_freq = {config.ica_l_freq}'
                f' < l_freq = {config.l_freq}. Adjust the cutoffs such that '
                'ica_l_freq >= l_freq, or set ica_l_freq to None if you do '
                'not wish to apply an additional high-pass filter before '
                'running ICA.')
        if (ica_reject is not None and
                reject is not None and
                reject != 'autoreject_global'):
            for ch_type in reject:
                if (ch_type in ica_reject and
                        reject[ch_type] > ica_reject[ch_type]):
                    raise ValueError(
                        f'Rejection threshold in reject["{ch_type}"] '
                        f'({reject[ch_type]}) must be at least as stringent '
                        'as that in '
                        f'ica_reject["{ch_type}"] ({ica_reject[ch_type]})')

    if not config.ch_types:
        raise ValueError('Please specify ch_types in your configuration.')

    _VALID_TYPES = ('meg', 'mag', 'grad', 'eeg')
    if any(ch_type not in _VALID_TYPES for ch_type in config.ch_types):
        raise ValueError(
            'Invalid channel type passed. Please adjust `ch_types` in your '
            f'configuration, got {config.ch_types} but supported types are '
            f'{_VALID_TYPES}')

    _check_option(
        'config.on_error', config.on_error, ('continue', 'abort', 'debug'))
    _check_option(
        'config.memory_file_method', config.memory_file_method,
        ('mtime', 'hash'))

    if isinstance(config.noise_cov, str):
        _check_option(
            'config.noise_cov', config.noise_cov,
            ('emptyroom', 'ad-hoc', 'rest'), extra='when a string')

    if config.noise_cov == 'emptyroom' and 'eeg' in config.ch_types:
        raise ValueError(
            'You requested to process data that contains EEG channels. In '
            'this case, noise covariance can only be estimated from the '
            'experimental data, e.g., the pre-stimulus period. Please set '
            'noise_cov to (tmin, tmax)')

    if config.noise_cov == 'emptyroom' and not config.process_empty_room:
        raise ValueError(
            'You requested noise covariance estimation from empty-room '
            'recordings by setting noise_cov = "emptyroom", but you did not '
            'enable empty-room data processing. '
            'Please set process_empty_room = True')

    _check_option(
        'config.bem_mri_images', config.bem_mri_images,
        ('FLASH', 'T1', 'auto'))

    bl = config.baseline
    if bl is not None:
        if ((bl[0] is not None and bl[0] < config.epochs_tmin) or
                (bl[1] is not None and bl[1] > config.epochs_tmax)):
            raise ValueError(
                f'baseline {bl} outside of epochs interval '
                f'{[config.epochs_tmin, config.epochs_tmax]}.')

        if (bl[0] is not None and bl[1] is not None and bl[0] >= bl[1]):
            raise ValueError(
                f'The end of the baseline period must occur after its start, '
                f'but you set baseline={bl}')

    # check decoding parameters
    if config.decoding_n_splits < 2:
        raise ValueError('decoding_n_splits should be at least 2.')

    # check cluster permutation parameters
    if not 0 < config.cluster_permutation_p_threshold < 1:
        raise ValueError(
            "cluster_permutation_p_threshold should be in the (0, 1) interval."
        )

    if config.cluster_n_permutations < \
            10 / config.cluster_permutation_p_threshold:
        raise ValueError(
            "cluster_n_permutations is not big enough to calculate "
            "the p-values accurately."
        )

    # Another check that depends on some of the functions defined above
    if (not config.task_is_rest and config.conditions is None):
        raise ValueError(
            'Please indicate the name of your conditions in your '
            'configuration. Currently the `conditions` parameter is empty. '
            'This is only allowed for resting-state analysis.')

    _check_option(
        'config.on_rename_missing_events', config.on_rename_missing_events,
        ('raise', 'warn', 'ignore'))

    _validate_type(config.N_JOBS, int, 'N_JOBS')

    _check_option(
        'config.config_validation', config.config_validation,
        ('raise', 'warn', 'ignore'))


_REMOVED_NAMES = {
    'debug': dict(
        new_name='on_error',
        instead='use on_error="debug" instead',
    ),
    'decim': dict(
        new_name='epochs_decim',
        instead=None,
    ),
    'resample_sfreq': dict(
        new_name='raw_resample_sfreq',
    ),
}


def _check_misspellings_removals(
    config: SimpleNamespace,
    *,
    valid_names: List[str],
    user_names: List[str],
    log: bool,
) -> None:
    # for each name in the user names, check if it's in the valid names but
    # the correct one is not defined
    valid_names = set(valid_names)
    for user_name in user_names:
        if user_name not in valid_names:
            # find the closest match
            closest_match = difflib.get_close_matches(
                user_name, valid_names, n=1)
            msg = (
                f'Found a variable named {repr(user_name)} in your custom '
                'config,'
            )
            if closest_match and closest_match[0] not in user_names:
                this_msg = (
                    f'{msg} did you mean {repr(closest_match[0])}? '
                    'If so, please correct the error. If not, please rename '
                    'the variable to reduce ambiguity and avoid this message, '
                    "or set config.config_validation to 'warn' or 'ignore'."
                )
                _handle_config_error(this_msg, log, config)
            if user_name in _REMOVED_NAMES:
                new = _REMOVED_NAMES[user_name]['new_name']
                if new not in user_names:
                    instead = _REMOVED_NAMES[user_name].get("instead", None)
                    if instead is None:
                        instead = f'use {new} instead'
                    this_msg = (
                        f'{msg} this variable has been removed as a valid '
                        f'config option, {instead}.'
                    )
                    _handle_config_error(this_msg, log, config)


def _handle_config_error(
    msg: str,
    log: bool,
    config: SimpleNamespace,
) -> None:
    if config.config_validation == 'raise':
        raise ValueError(msg)
    elif config.config_validation == 'warn':
        if log:
            logger.warning(
                **gen_log_kwargs(message=msg, step='', emoji='🛟')
            )
