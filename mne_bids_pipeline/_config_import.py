from types import ModuleType
import os
import pathlib
import importlib

import matplotlib
import mne
from mne.utils import _check_option

from ._logging import logger


def _import_config() -> ModuleType:
    """Import the default config and the user's config."""
    from . import __version__
    from . import _config  # get the default
    _config.PIPELINE_NAME = 'mne-bids-pipeline'
    _config.VERSION = __version__
    _config.CODE_URL = 'https://github.com/mne-tools/mne-bids-pipeline'
    os.environ['MNE_BIDS_STUDY_SCRIPT_PATH'] = str(__file__)
    mne.set_log_level(verbose=_config.mne_log_level.upper())
    _config._raw_split_size = '2GB'
    _config._epochs_split_size = '2GB'
    # Now update with user config
    _update_with_user_config(_config)
    # And then check it
    _check_config(_config)
    return _config


def _update_with_user_config(config: ModuleType) -> ModuleType:
    if "MNE_BIDS_STUDY_CONFIG" in os.environ:
        cfg_path = pathlib.Path(os.environ['MNE_BIDS_STUDY_CONFIG'])

        if not cfg_path.exists():
            raise ValueError(
                'The custom configuration file specified in the '
                'MNE_BIDS_STUDY_CONFIG environment variable could not be '
                f'found: {cfg_path}')

        # Import configuration from an arbitrary path without having to fiddle
        # with `sys.path`.
        spec = importlib.util.spec_from_file_location(
            name='custom_config', location=cfg_path)
        custom_cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_cfg)
        for key in dir(custom_cfg):
            if not key.startswith('__'):
                val = getattr(custom_cfg, key)
                logger.debug('Overwriting: %s -> %s' % (key, val))
                setattr(config, key, val)

    config.interactive = bool(int(os.getenv(
        'MNE_BIDS_STUDY_INTERACTIVE', config.interactive)))
    if not config.interactive:
        matplotlib.use('Agg')  # do not open any window  # noqa

    if os.getenv('MNE_BIDS_STUDY_USE_CACHE', '') == '0':
        config.memory_location = False


def _check_config(config: ModuleType) -> None:
    # TODO: Use pydantic to do these validations
    # https://github.com/mne-tools/mne-bids-pipeline/issues/646
    _check_option(
        'config.parallel_backend', config.parallel_backend, ('dask', 'loky'))

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
