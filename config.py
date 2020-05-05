"""Set the configuration parameters for the study.

You need to define an environment variable `BIDS_ROOT` to point to the root
of your BIDS dataset to be analyzed.

"""
import importlib
import os
from collections import defaultdict
import copy

import numpy as np
import mne
from mne_bids.utils import get_entity_vals

# Name, version, and hosting location of the pipeline
PIPELINE_NAME = 'mne-study-template'
VERSION = '0.1.dev0'
CODE_URL = 'https://github.com/mne-tools/mne-study-template'


# ``study_name`` : str
#   Specify the name of your study. It will be used to populate filenames for
#   saving the analysis results.
#
# Example
# ~~~~~~~
# >>> study_name = 'my-study'

study_name = ''

# ``bids_root`` : str or None
#   Speficy the BIDS root directory. Pass an empty string or ```None`` to use
#   the value specified in the ``BIDS_ROOT`` environment variable instead.
#   Raises an exception if the BIDS root has not been specified.
#
# Example
# ~~~~~~~
# >>> bids_root = '/path/to/your/bids_root'  # Use this to specify a path here.
# or
# >>> bids_root = None  # Make use of the ``BIDS_ROOT`` environment variable.

bids_root = None

# ``subjects_dir`` : str or None
#   Path to the directory that contains the MRI data files and their
#   derivativesfor all subjects. Specifically, the ``subjects_dir`` is the
#   $SUBJECTS_DIR used by the Freesurfer software. If ``None``, will use
#   ``'bids_root/derivatives/freesurfer/subjects'``.

subjects_dir = None

# ``daysback``  : int
#   If not None apply a time shift to dates to adjust for limitateions
#   of fif files

daysback = None

# ``plot`` : boolean
#   If True, the scripts will generate plots.
#   If running the scripts from a notebook or spyder
#   run %matplotlib qt in the command line to get the plots in extra windows

plot = False

# ``crop`` : tuple or None
# If tuple, (tmin, tmax) to crop the raw data
# If None (default), do not crop.
crop = None

# BIDS params
# see: bids-specification.rtfd.io/en/latest/99-appendices/04-entity-table.html

# ``sessions`` : iterable or 'all'
#   The sessions to process.
sessions = 'all'

# ``task`` : str
#   The task to process.
task = ''

# ``runs`` : iterable or 'all'
#   The runs to process.
runs = 'all'

acq = None

proc = None

rec = None

space = None

# ``subjects_list`` : list of str
#   To define the list of participants, we use a list with all the anonymized
#   participant names. Even if you plan on analyzing a single participant, it
#   needs to be set up as a list with a single element, as in the 'example'
#   subjects_list = ['SB01']

subjects_list = 'all'

# ``exclude_subjects`` : list of str
#   Now you can specify subjects to exclude from the group study:
#
# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# Keep track of the criteria leading you to exclude
# a participant (e.g. too many movements, missing blocks, aborted experiment,
# did not understand the instructions, etc, ...)

exclude_subjects = ['emptyroom']

# ``ch_types``  : list of st
#    The list of channel types to consider.
#
# Example
# ~~~~~~~
# >>> ch_types = ['meg', 'eeg']  # to use MEG and EEG channels
# or
# >>> ch_types = ['meg']  # to use only MEG
# or
# >>> ch_types = ['grad']  # to use only gradiometer MEG channels

# Note: If `kind` is 'eeg', EEG ch_types will be used regardless of whether
# specified here or not
ch_types = []

###############################################################################
# DEFINE ADDITIONAL CHANNELS
# --------------------------
# needed for 01-import_and_maxfilter.py

# ``rename_channels`` : dict rename channels
#    Here you name or replace extra channels that were recorded, for instance
#    EOG, ECG.
#
# Example
# ~~~~~~~
# Here rename EEG061 to EOG061, EEG062 to EOG062, EEG063 to ECG063:
# >>> rename_channels = {'EEG061': 'EOG061', 'EEG062': 'EOG062',
#                        'EEG063': 'ECG063'}

# XXX should be done automatically from BIDS ?
rename_channels = None

# ``set_channel_types``: dict
#   Here you define types of channels to pick later.
#
# Example
# ~~~~~~~
# >>> set_channel_types = {'EEG061': 'eog', 'EEG062': 'eog',
#                          'EEG063': 'ecg', 'EEG064': 'misc'}

# XXX should not be necessary
set_channel_types = None

###############################################################################
# MAXWELL FILTER PARAMETERS
# -------------------------
# done in 01-import_and_maxfilter.py
#
# Note: For any of this to work, you must set ``mf_ctc_fname`` and
# ``mf_cal_fname`` above.
#
# "Bad", i.e. flat and overly noisy channels, can be automatically detected
# using a procedure inspired by the commercial MaxFilter by Elekta. First,
# a copy of the data is low-pass filtered at 40 Hz. Then, channels with
# unusually low variability are flagged as "flat", while channels with
# excessively high variability are flagged as "noisy". Flat and noisy channels
# are marked as "bad" and excluded from subsequent analysis. See
# :func:`mne.preprocssessing.find_bad_channels_maxwell` for more information
# on this procedure. The list of bad channels detected through this procedure
# will be merged with the list of bad channels already present in the dataset,
# if any.
#
# ``find_flat_channels_meg`` : bool
#   Auto-detect "flat" channels and mark them as bad.
#
# ``find_noisy_channels_meg`` : bool
#   Auto-detect "noisy" channels and mark them as bad.

find_flat_channels_meg = False
find_noisy_channels_meg = False

# ``use_maxwell_filter`` : bool
#   Use or not maxwell filter to preprocess the data.
#
# Warning
# ~~~~~~~
# If the data were recorded with internal active compensation (MaxShield),
# they need to be run through Maxwell filter to avoid distortions.
# Bad channels need to be set through BIDS channels.tsv and / or via the
# ``find_flat_channels_meg`` and ``find_noisy_channels_meg`` options above
# before applying Maxwell filter.

use_maxwell_filter = False

# There are two kinds of maxfiltering: SSS and tSSS
# [SSS = signal space separation ; tSSS = temporal signal space separation]
# (Taulu et al, 2004): http://cds.cern.ch/record/709081/files/0401166.pdf
#
# ``mf_st_duration`` : float | None
#    If not None, apply spatiotemporal SSS (tSSS) with specified buffer
#    duration (in seconds). MaxFilter™'s default is 10.0 seconds in v2.2.
#    Spatiotemporal SSS acts as implicitly as a high-pass filter where the
#    cut-off frequency is 1/st_dur Hz. For this (and other) reasons, longer
#    buffers are generally better as long as your system can handle the
#    higher memory usage. To ensure that each window is processed
#    identically, choose a buffer length that divides evenly into your data.
#    Any data at the trailing edge that doesn't fit evenly into a whole
#    buffer window will be lumped into the previous buffer.
#
# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# If you are interested in low frequency activity (<0.1Hz), avoid using tSSS
# and set mf_st_duration to None
#
# If you are interested in low frequency above 0.1 Hz, you can use the
# default mf_st_duration to 10 s meaning it acts like a 0.1 Hz highpass filter.
#
# Example
# ~~~~~~~
# >>> mf_st_duration = None
# or
# >>> mf_st_duration = 10.  # to apply tSSS with 0.1Hz highpass filter.

mf_st_duration = None

# ``mf_head_origin`` : array-like, shape (3,) | 'auto'
#   Origin of internal and external multipolar moment space in meters.
#   If 'auto', it will be estimated from headshape points.
#   If automatic fitting fails (e.g., due to having too few digitization
#   points), consider separately calling the fitting function with different
#   options or specifying the origin manually.
#
# Example
# ~~~~~~~
# >>> mf_head_origin = 'auto'

mf_head_origin = 'auto'

# ``cross talk`` : str
#   Path to the cross talk file
#
#
# ``calibration`` : str
#   Path to the calibration file.
#
#
# These 2 files should be downloaded and made available for running
# maxwell filtering.
#
# Example
# ~~~~~~~
# >>> cal_files_path = os.path.join(study_path, 'SSS')
# >>> mf_ctc_fname = os.path.join(cal_files_path, 'ct_sparse_mgh.fif')
# >>> mf_cal_fname = os.path.join(cal_files_path, 'sss_cal_mgh.dat')
#
# Warning
# ~~~~~~~
# These 2 files are site and machine specific files that provide information
# about the environmental noise. For practical purposes, place them in your
# study folder.
#
# At NeuroSpin: ct_sparse and sss_call are on the meg_tmp server

# cal_files_path = os.path.join(study_path, 'SSS')
# mf_ctc_fname = os.path.join(cal_files_path, 'ct_sparse_mgh.fif')
# mf_cal_fname = os.path.join(cal_files_path, 'sss_cal_mgh.dat')

mf_ctc_fname = ''
mf_cal_fname = ''

# Despite all possible care to avoid movements in the MEG, the participant
# will likely slowly drift down from the Dewar or slightly shift the head
# around in the course of the recording session. Hence, to take this into
# account, we are realigning all data to a single position. For this, you need
# to define a reference run (typically the one in the middle of
# the recording session).
#
# ``mf_reference_run``  : int
#   Which run to take as the reference for adjusting the head position of all
#   runs.
#
# Example
# ~~~~~~~
# >>> mf_reference_run = 0  # to use the first run

mf_reference_run = 0

###############################################################################
# FREQUENCY FILTERING
# -------------------
# done in 02-frequency_filter.py

# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# It is typically better to set your filtering properties on the raw data so
# as to avoid what we call border (or edge) effects.
#
# If you use this pipeline for evoked responses, you could consider
# a low-pass filter cut-off of h_freq = 40 Hz
# and possibly a high-pass filter cut-off of l_freq = 1 Hz
# so you would preserve only the power in the 1Hz to 40 Hz band.
# Note that highpass filtering is not necessarily recommended as it can
# distort waveforms of evoked components, or simply wash out any low
# frequency that can may contain brain signal. It can also act as
# a replacement for baseline correction in Epochs. See below.
#
# If you use this pipeline for time-frequency analysis, a default filtering
# coult be a high-pass filter cut-off of l_freq = 1 Hz
# a low-pass filter cut-off of h_freq = 120 Hz
# so you would preserve only the power in the 1Hz to 120 Hz band.
#
# If you need more fancy analysis, you are already likely past this kind
# of tips! :)


# ``l_freq`` : float
#   The low-frequency cut-off in the highpass filtering step.
#   Keep it None if no highpass filtering should be applied.

l_freq = 1.

# ``h_freq`` : float
#   The high-frequency cut-off in the lowpass filtering step.
#   Keep it None if no lowpass filtering should be applied.

h_freq = 40.

###############################################################################
# RESAMPLING
# ----------
#
# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# If you have acquired data with a very high sampling frequency (e.g. 2 kHz)
# you will likely want to downsample to lighten up the size of the files you
# are working with (pragmatics)
# If you are interested in typical analysis (up to 120 Hz) you can typically
# resample your data down to 500 Hz without preventing reliable time-frequency
# exploration of your data
#
# ``resample_sfreq``  : float
#   Specifies at which sampling frequency the data should be resampled.
#   If None then no resampling will be done.
#
# Example
# ~~~~~~~
# >>> resample_sfreq = None  # no resampling
# or
# >>> resample_sfreq = 500  # resample to 500Hz

resample_sfreq = None

# ``decim`` : int
#   Says how much to decimate data at the epochs level.
#   It is typically an alternative to the `resample_sfreq` parameter that
#   can be used for resampling raw data. 1 means no decimation.
#
# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# Decimation requires to lowpass filtered the data to avoid aliasing.
# Note that using decimation is much faster than resampling.
#
# Example
# ~~~~~~~
# >>> decim = 1  # no decimation
# or
# >>> decim = 4  # decimate by 4 ie devide sampling frequency by 4

decim = 1

###############################################################################
# AUTOMATIC REJECTION OF ARTIFACTS
# --------------------------------
#
# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# Have a look at your raw data and train yourself to detect a blink, a heart
# beat and an eye movement.
# You can do a quick average of blink data and check what the amplitude looks
# like.
#
#  ``reject`` : dict | None
#    The rejection limits to make some epochs as bads.
#    This allows to remove strong transient artifacts.
#    If you want to reject and retrieve blinks later, e.g. with ICA,
#    don't specify a value for the eog channel (see examples below).
#    Make sure to include values for eeg if you have EEG data
#
# Note
# ~~~~
# These numbers tend to vary between subjects.. You might want to consider
# using the autoreject method by Jas et al. 2018.
# See https://autoreject.github.io
#
# Example
# ~~~~~~~
# >>> reject = {'grad': 4000e-13, 'mag': 4e-12, 'eog': 150e-6}
# >>> reject = {'grad': 4000e-13, 'mag': 4e-12, 'eeg': 200e-6}
# >>> reject = None

reject = {'grad': 4000e-13, 'mag': 4e-12, 'eeg': 150e-6}


###############################################################################
# RENAME EXPERIMENTAL EVENTS
# --------------------------
#
# ``rename_events`` : dict
#   A dictionary specifying which events in the BIDS dataset to rename upon
#   loading, and before processing begins.
#
#   Pass an empty dictionary to not perform any renaming.
#
# Example
# ~~~~~~~
# Rename ``audio_left`` in the BIDS dataset to ``audio/left`` in the pipeline:
# >>> rename_events = {'audio_left': 'audio/left'}

rename_events = dict()


###############################################################################
# EPOCHING
# --------
#
# ``tmin``: float
#    A float in seconds that gives the start time before event of an epoch.
#
# Example
# ~~~~~~~
# >>> tmin = -0.2  # take 200ms before event onset.

tmin = -0.2

# ``tmax``: float
#    A float in seconds that gives the end time before event of an epoch.
#
# Example
# ~~~~~~~
# >>> tmax = 0.5  # take 500ms after event onset.

tmax = 0.5

# ``trigger_time_shift`` : float | None
#    If float it specifies the offset for the trigger and the stimulus
#    (in seconds). You need to measure this value for your specific
#    experiment/setup.
#
# Example
# ~~~~~~~
# >>> trigger_time_shift = 0  # don't apply any offset

trigger_time_shift = 0.

# ``baseline`` : tuple
#    It specifies how to baseline the epochs; if None, no baseline is applied.
#
# Example
# ~~~~~~~
# >>> baseline = (None, 0)  # baseline between tmin and 0

baseline = (None, 0)

#  `conditions`` : list
#    The condition names to consider. This can either be the keys of
#    ``event_id``, or – if event names were specified with ``/`` for
#    grouping – the name of the *grouped* condition (i.e., the
#    condition name before or after that ``/`` that is shared between the
#    respective conditions you wish to group). See the "Subselecting epochs"
#    tutorial for more information: https://mne.tools/stable/auto_tutorials/epochs/plot_10_epochs_overview.html#subselecting-epochs  # noqa: 501
#
# Example
# ~~~~~~~
# >>> conditions = ['auditory/left', 'visual/left']
# or
# >>> conditions = ['auditory/left', 'auditory/right']
# or
# >>> conditions = ['auditory']
# or
# >>> conditions = ['auditory', 'visual']
# or
# >>> conditions = ['left', 'right']

conditions = ['left', 'right']

###############################################################################
# ARTIFACT REMOVAL
# ----------------
#
# You can choose between ICA and SSP to remove eye and heart artifacts.
# SSP: https://mne-tools.github.io/stable/auto_tutorials/plot_artifacts_correction_ssp.html?highlight=ssp # noqa
# ICA: https://mne-tools.github.io/stable/auto_tutorials/plot_artifacts_correction_ica.html?highlight=ica # noqa
# if you choose ICA, run scripts 5a and 6a
# if you choose SSP, run scripts 5b and 6b
#
# Currently you cannot use both.
#
# ``use_ssp`` : bool
#    If True ICA should be used or not.

use_ssp = True

# ``use_ica`` : bool
#    If True ICA should be used or not.

use_ica = False

# ``ica_decim`` : int
#    The decimation parameter to compute ICA. If 5 it means
#    that 1 every 5 sample is used by ICA solver. The higher the faster
#    it is to run but the less data you have to compute a good ICA.

ica_decim = 11


# ``default_reject_comps_factory`` : callable
#    A factory function that returns a default rejection component dictionary:
#    A dictionary that specifies the indices of the ICA components to reject
#    for each subject. For example you can use:
#    rejcomps_man['subject01'] = dict(eeg=[12], meg=[7])

def default_reject_comps_factory():
    """Return the default rejection component dictionary."""
    return dict(meg=[], eeg=[])


rejcomps_man = defaultdict(default_reject_comps_factory)

# ``ica_ctps_ecg_threshold``: float
#    The threshold parameter passed to `find_bads_ecg` method.

ica_ctps_ecg_threshold = 0.1

###############################################################################
# DECODING
# --------
#
# ``decoding_conditions`` : list
#    List of conditions to be classified.
#
# Example
# ~~~~~~~
# >>> decoding_conditions = []  # don't do decoding
# or
# >>> decoding_conditions = [('auditory', 'visual'), ('left', 'right')]

decoding_conditions = []
# decoding_conditions = [('left', 'right')]

###############################################################################
# GROUP AVERAGE SENSORS
# ---------------------
#
# ``interpolate_bads_grand_average`` : bool
#    Interpolate bad sensors in each dataset before calculating the grand
#    average. This parameter is passed to the `mne.grand_average` function via
#    the keyword argument `interpolate_bads`. It requires to have channel
#    locations set.
#
# Example
# ~~~~~~~
# >>> interpolate_bads_grand_average = True

interpolate_bads_grand_average = True

# ``decoding_metric`` : str
#    The metric to use for cross-validation. It can be 'roc_auc' or 'accuracy'
#    or any metric supported by scikit-learn.

decoding_metric = 'roc_auc'

# ``decoding_n_splits`` : int
#    The number of folds (a.k.a. splits) to use in the cross-validation.

decoding_n_splits = 5

###############################################################################
# TIME-FREQUENCY
# --------------
#
# ``time_frequency_conditions`` : list
#    The conditions to compute time-frequency decomposition on.

# time_frequency_conditions = ['left', 'right']
time_frequency_conditions = []

###############################################################################
# SOURCE SPACE PARAMETERS
# -----------------------
#

# ``spacing`` : str
#    The spacing to use. Can be ``'ico#'`` for a recursively subdivided
#    icosahedron, ``'oct#'`` for a recursively subdivided octahedron,
#    ``'all'`` for all points, or an integer to use appoximate
#    distance-based spacing (in mm).

spacing = 'oct6'

# ``mindist`` : float
#    Exclude points closer than this distance (mm) to the bounding surface.

mindist = 5

# ``loose`` : float in [0, 1] | 'auto'
#    Value that weights the source variances of the dipole components
#    that are parallel (tangential) to the cortical surface. If loose
#    is 0 then the solution is computed with fixed orientation,
#    and fixed must be True or "auto".
#    If loose is 1, it corresponds to free orientations.
#    The default value ('auto') is set to 0.2 for surface-oriented source
#    space and set to 1.0 for volumetric, discrete, or mixed source spaces,
#    unless ``fixed is True`` in which case the value 0. is used.

loose = 0.2

# ``depth`` : None | float | dict
#    If float (default 0.8), it acts as the depth weighting exponent (``exp``)
#    to use (must be between 0 and 1). None is equivalent to 0, meaning no
#    depth weighting is performed. Can also be a `dict` containing additional
#    keyword arguments to pass to :func:`mne.forward.compute_depth_prior`
#    (see docstring for details and defaults).

depth = 0.8

# inverse_method : "MNE" | "dSPM" | "sLORETA" | "eLORETA"
#    Use minimum norm, dSPM (default), sLORETA, or eLORETA.

inverse_method = 'dSPM'

# smooth : int | None
#    Number of iterations for the smoothing of the surface data.
#    If None, smooth is automatically defined to fill the surface
#    with non-zero values. The default is spacing=None.

smooth = 10

fsaverage_vertices = [np.arange(10242), np.arange(10242)]

###############################################################################
# ADVANCED
# --------
#
# ``l_trans_bandwidth`` : float | 'auto'
#    A float that specifies the transition bandwidth of the
#    highpass filter. By default it's `'auto'` and uses default mne
#    parameters.

l_trans_bandwidth = 'auto'

#  ``h_trans_bandwidth`` : float | 'auto'
#    A float that specifies the transition bandwidth of the
#    lowpass filter. By default it's `'auto'` and uses default mne
#    parameters.

h_trans_bandwidth = 'auto'

#  ``N_JOBS`` : int
#    An integer that specifies how many subjects you want to run in parallel.

N_JOBS = 1

# ``random_state`` : None | int | np.random.RandomState
#    To specify the random generator state. This allows to have
#    the results more reproducible between machines and systems.
#    Some methods like ICA need random values for initialisation.

random_state = 42

# ``shortest_event`` : int
#    Minimum number of samples an event must last. If the
#    duration is less than this an exception will be raised.

shortest_event = 1

# ``allow_maxshield``  : bool
#    To import data that was recorded with Maxshield on before running
#    maxfilter set this to True.

allow_maxshield = False

###############################################################################
# Retrieve custom configuration options
# -------------------------------------
#
# For testing a specific dataset, create a Python file with a name of your
# liking (e.g., ``mydataset-template-config.py``), and set an environment
# variable ``MNE_BIDS_STUDY_CONFIG`` to that file.
#
# Example
# ~~~~~~~
# ``export MNE_BIDS_STUDY_CONFIG=/data/mystudy/mydataset-template-config.py``

if "MNE_BIDS_STUDY_CONFIG" in os.environ:
    cfg_path = os.environ['MNE_BIDS_STUDY_CONFIG']

    if os.path.exists(cfg_path):
        print('Using custom configuration specified in MNE_BIDS_STUDY_CONFIG.')
    else:
        msg = ('The custom configuration file specified in the '
               'MNE_BIDS_STUDY_CONFIG environment variable could not be '
               'found: {cfg_path}'.format(cfg_path=cfg_path))
        raise ValueError(msg)

    # Import configuration from an arbitrary path without having to fiddle
    # with `sys.path`.
    spec = importlib.util.spec_from_file_location(name='custom_config',
                                                  location=cfg_path)
    custom_cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_cfg)
    del spec, cfg_path

    new = None
    for val in dir(custom_cfg):
        if not val.startswith('__'):
            exec("new = custom_cfg.%s" % val)
            print('Overwriting: %s -> %s' % (val, new))
            exec("%s = custom_cfg.%s" % (val, val))


# BIDS_ROOT environment variable takes precedence over any configuration file
# values.
if os.getenv('BIDS_ROOT') is not None:
    bids_root = os.getenv('BIDS_ROOT')

# If we don't have a bids_root until now, raise an exeception as we cannot
# proceed.
if not bids_root:
    msg = ('You need to specify `bids_root` in your configuration, or '
           'define an environment variable `BIDS_ROOT` pointing to the '
           'root folder of your BIDS dataset')
    raise ValueError(msg)


###############################################################################
# CHECKS
# ------
#
# --- --- You should not touch the next lines --- ---

if (use_maxwell_filter and
        len(set(ch_types).intersection(('meg', 'grad', 'mag'))) == 0):
    raise ValueError('Cannot use maxwell filter without MEG channels.')

if use_ssp and use_ica:
    raise ValueError('Cannot use both SSP and ICA.')

if not ch_types:
    msg = 'Please specify ch_types in your configuration.'
    raise ValueError(msg)

if ch_types == ['eeg']:
    pass
elif 'eeg' in ch_types and len(ch_types) > 1:  # EEG + some other channel types
    msg = ('EEG data can only be analyzed separately from other channel '
           'types. Please adjust `ch_types` in your configuration.')
    raise ValueError(msg)
elif any([ch_type not in ('meg', 'mag', 'grad') for ch_type in ch_types]):
    msg = ('Invalid channel type passed. Please adjust `ch_types` in your '
           'configuration.')
    raise ValueError(msg)

if 'eeg' in ch_types:
    if use_ssp:
        msg = ('You requested SSP for EEG data via use_ssp=True. However, '
               'this is not presently supported. Please use ICA instead by '
               'setting use_ssp=False and use_ica=True.')
        raise ValueError(msg)
    if not use_ica:
        msg = ('You did not request ICA artifact correction for your data. '
               'To turn it on, set use_ica=True.')
        mne.utils.logger.info(msg)


###############################################################################
# Helper functions
# ----------------

def get_sessions():
    sessions_ = copy.deepcopy(sessions)  # Avoid clash with global variable.

    if sessions_ == 'all':
        sessions_ = get_entity_vals(bids_root, entity_key='ses')

    if not sessions_:
        return [None]
    else:
        return sessions_


def get_runs():
    runs_ = copy.deepcopy(runs)  # Avoid clash with global variable.

    if runs_ == 'all':
        runs_ = get_entity_vals(bids_root, entity_key='run')

    if not runs_:
        return [None]
    else:
        return runs_


def get_subjects():
    if subjects_list == 'all':
        s = get_entity_vals(bids_root, entity_key='sub')
    else:
        s = subjects_list

    return list(set(s) - set(exclude_subjects))


def get_task():
    if not task:
        return get_entity_vals(bids_root, entity_key='task')[0]
    else:
        return task


def get_kind():
    # Content of ch_types should be sanitized already, so we don't need any
    # extra sanity checks here.
    if ch_types == ['eeg']:
        return 'eeg'
    else:
        return 'meg'


def get_reject():
    reject_ = reject.copy()  # Avoid clash with global variable.
    kind = get_kind()

    if kind == 'eeg':
        ch_types_to_remove = ('mag', 'grad')
    else:
        ch_types_to_remove = ('eeg',)

    for ch_type in ch_types_to_remove:
        try:
            del reject_[ch_type]
        except KeyError:
            pass
    return reject_


def get_subjects_dir():
    if not subjects_dir:
        return os.path.join(bids_root, 'derivatives', 'freesurfer', 'subjects')
    else:
        return subjects_dir
