"""Set the configuration parameters for the study.

You need to define an environment variable `BIDS_ROOT` to point to the root
of your BIDS dataset to be analyzed.

"""
import importlib
import functools
import os
from collections import defaultdict
import copy
import coloredlogs
import logging

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

# ``interactive`` : boolean
#   If True, the scripts will provide some interactive elements, such as
#   figures. If running the scripts from a notebook or Spyder,
#   run %matplotlib qt in the command line to open the figures in a separate
#   window.

interactive = False

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

# ``subjects_list`` : 'all' | list of str
#   Subjects to analyze. If ``'all``, include all subjects. To only
#   include a subset of subjects, pass a list of their identifiers. Even
#   if you plan on analyzing only a single subject, pass their identifier
#   as a list.
#
#   Please note that if you intend to EXCLUDE only a few subjects, you
#   should consider setting ``subjects_list = 'all'`` and adding the
#   identifiers of the excluded subjects to ``exclude_subjects`` (see next
#   section).
#
# Example
# ~~~~~~~
# >>> subjects_list = 'all'  # Include all subjects.
# >>> subjects_list = ['05']  # Only include subject 05.
# >>> subjects_list = ['01', '02']  # Only include subjects 01 and 02.

subjects_list = 'all'

# ``exclude_subjects`` : list of str
#   Specify subjects to exclude from analysis. The MEG empty-room mock-subject
#   is automatically excluded from regular analysis.
#
# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# Keep track of the criteria leading you to exclude
# a participant (e.g. too many movements, missing blocks, aborted experiment,
# did not understand the instructions, etc, ...)
# The ``emptyroom`` subject will be excluded automatically.

exclude_subjects = []

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
#  `conditions`` : list
#    The condition names to consider. This can either be name of the
#    experimental condition as specified in the BIDS ``events.tsv`` file; or
#    the name of condition *grouped*, if the condition names contain the
#    (MNE-specific) group separator, ``/``. See the "Subselecting epochs"
#    tutorial for more information: https://mne.tools/stable/auto_tutorials/epochs/plot_10_epochs_overview.html#subselecting-epochs  # noqa: 501
#
# Example
# ~~~~~~~
# >>> conditions = ['auditory/left', 'visual/left']
# or
# >>> conditions = ['auditory/left', 'auditory/right']
# or
# >>> conditions = ['auditory']  # All "auditory" conditions (left AND right)
# or
# >>> conditions = ['auditory', 'visual']
# or
# >>> conditions = ['left', 'right']

conditions = ['left', 'right']

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

# ``baseline`` : tuple
#    It specifies how to baseline the epochs; if None, no baseline is applied.
#
# Example
# ~~~~~~~
# >>> baseline = (None, 0)  # baseline between tmin and 0

baseline = (None, 0)


#  ``contrasts`` : list of tuples
#    The conditions to contrast via a subtraction of ERPs / ERFs. Each tuple
#    in the list corresponds to one contrast. The condition names must be
#    specified in ``conditions`` above. Pass an empty list to avoid calculation
#    of contrasts.
#
# Example
# ~~~~~~~
# Contrast the "left" and the "right" conditions by calculating "left - right"
# at every time point of the evoked responses:
# >>> conditions = ['left', 'right']
# >>> contrasts = [('left', 'right')]  # Note we pass a tuple inside the list!
#
# Contrast the "left" and the "right" conditions within the "auditory" and
# the "visual" modality, and "auditory" vs "visual" regardless of side:
# >>> conditions = ['auditory/left', 'auditory/right',
#                   'visual/left', 'visual/right']
# >>> contrasts = [('auditory/left', 'auditory/right'),
#                  ('visual/left', 'visual/right'),
#                  ('auditory', 'visual')]

contrasts = []

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

# SSP
# ~~~
#
# ``use_ssp`` : bool
#    If True ICA should be used or not.

use_ssp = True

# ICA
# ~~~
# ``use_ica`` : bool
#    If True ICA should be used or not.

use_ica = False

# ``ica_algorithm`` : 'picard' | 'fastica' | 'extended_infomax'
#   The ICA algorithm to use.

ica_algorithm = 'picard'

# ``ica_max_iterations`` : int
#   Maximum number of iterations to decompose the data into independent
#   components. A low number means to finish earlier, but the consequence is
#   that the algorithm may not have finished converging. To ensure
#   convergence, pick a high number here (e.g. 3000); yet the algorithm will
#   terminate as soon as it determines that is has successfully converged, and
#   not necessarily exhaust the maximum number of iterations. Note that the
#   default of 200 seems to be sufficient for Picard in many datasets, because
#   it converges quicker than the other algorithms; but e.g. for FastICA, this
#   limit may be too low to achieve convergence.

ica_max_iterations = 200

# ``ica_decim`` : None | None
#    The decimation parameter to compute ICA. If 5 it means
#    that 1 every 5 sample is used by ICA solver. The higher the faster
#    it is to run but the less data you have to compute a good ICA. Set to
#    ``1`` ``None`` to not perform an decimation.

ica_decim = None


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
# ``decode`` : bool
#    Whether to perform decoding (MVPA) on the contrasts specified above as
#    "contrasts". MVPA will be performed on the level of individual epochs.

decode = True

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

# noise_cov : (None, 0) | ‘emptyroom’
#   Specify how to estimate the noise covariance matrix, which is used in
#   inverse modeling.
#
#   If a tuple, it takes the form ``(tmin, tmax)`` with the time specified in
#   seconds. If the first value of the tuple is ``None``, the considered
#   period starts at the beginning of the epoch. If the second value of the
#   tuple is ``None``, the considered period ends at the end of the epoch.
#   The default, ``(None, 0)``, includes the entire period before the event,
#   which is typically the pre-stimulus period.
#
#   If ``emptyroom``, the noise covariance matrix will be estimated from an
#   empty-room MEG recording. The empty-room recording will be automatically
#   selected based on recording date and time.
#
#   Please note that when processing data that contains EEG channels, the noise
#   covariance can ONLY be estimated from the pre-stimulus period.
#
# Example
# ~~~~~~~
# Use the period from start of the epoch until 100 ms before the experimental
# event:
# >>> noise_cov = (None, -0.1)
#
# Use the time period from the experimental event until the end of the epoch:
# >>> noise_cov = (0, None)
#
# Use an empty-room recording:
# >>> noise_cov = 'emptyroom'

noise_cov = (None, 0)

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

log_level = 'info'
mne_log_level = 'error'

# ``on_abort`` : 'continue' | 'abort'
#    Whether to abort processing as soon as an error occurs, or whether to
#    continue with all other processing steps for as long as possible.

on_error = 'abort'


###############################################################################
#                                                                             #
#                      CUSTOM CONFIGURATION ENDS HERE                         #
#                                                                             #
###############################################################################


###############################################################################
# Logger
# ------

logger = logging.getLogger('mne-study-template')

log_fmt = '%(asctime)s %(message)s'
log_date_fmt = coloredlogs.DEFAULT_DATE_FORMAT = '%H:%M:%S'
coloredlogs.install(level=log_level, logger=logger, fmt=log_fmt,
                    date_fmt=log_date_fmt)

mne.set_log_level(verbose=mne_log_level.upper())

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
        msg = f'Using custom configuration: {cfg_path}'
        logger.info(msg)
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
            logger.debug('Overwriting: %s -> %s' % (val, new))
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
# Derivates root
# --------------
deriv_root = os.path.join(bids_root, 'derivatives', PIPELINE_NAME)


###############################################################################
# CHECKS
# ------

if (use_maxwell_filter and
        len(set(ch_types).intersection(('meg', 'grad', 'mag'))) == 0):
    raise ValueError('Cannot use maxwell filter without MEG channels.')

if use_ssp and use_ica:
    raise ValueError('Cannot use both SSP and ICA.')

if use_ica and ica_algorithm not in ('picard', 'fastica', 'extended_infomax'):
    msg = (f"Invalid ICA algorithm requested. Valid values for ica_algorithm "
           f"are: 'picard', 'fastica', and 'extended_infomax', but received "
           f"{ica_algorithm}.")
    raise ValueError(msg)

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
        logger.info(msg)

if on_error not in ('continue', 'abort'):
    msg = (f"on_error must be one of 'continue' or 'abort', but received "
           f"{on_error}.")
    logger.info(msg)

if isinstance(noise_cov, str) and noise_cov != 'emptyroom':
    msg = (f"noise_cov must be a tuple or 'emptyroom', but received "
           f"{noise_cov}")
    raise ValueError(msg)

if noise_cov == 'emptyroom' and 'eeg' in ch_types:
    msg = ('You requested to process data that contains EEG channels. In this '
           'case, noise covariance can only be estimated from the '
           'experimental data, e.g., the pre-stimulus period. Please set '
           'noise_cov to (tmin, tmax)')
    raise ValueError(msg)


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

    subjects = set(s) - set(exclude_subjects)
    # Drop empty-room subject.
    subjects = subjects - set(['emptyroom'])

    return list(subjects)


def get_task():
    if not task:
        tasks = get_entity_vals(bids_root, entity_key='task')
        if not tasks:
            return None
        else:
            return tasks[0]
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


def get_fs_subjects_dir():
    if not subjects_dir:
        return os.path.join(bids_root, 'derivatives', 'freesurfer', 'subjects')
    else:
        return subjects_dir


def get_subject_path(subject, session, kind):
    subject_path = f'sub-{subject}'
    if session is not None:
        subject_path = os.path.join(subject_path, f'ses-{session}')
    subject_path = os.path.join(subject_path, kind)
    return subject_path


def get_subject_deriv_path(subject, session, kind):
    subject_path = get_subject_path(subject=subject, session=session,
                                    kind=kind)
    deriv_path = os.path.join(deriv_root, subject_path)
    return deriv_path


def gen_log_message(message, step=None, subject=None, session=None, run=None):
    if subject is not None:
        subject = f'sub-{subject}'
    if session is not None:
        session = f'ses-{session}'
    if run is not None:
        run = f'run-{run}'

    prefix = ', '.join([item for item in [subject, session, run]
                        if item is not None])
    if prefix:
        prefix = f'[{prefix}]'

    if step is not None:
        prefix = f'[Step-{step:02}]{prefix}'

    return prefix + ' ' + message


def failsafe_run(on_error):
    def failsafe_run_decorator(func):
        @functools.wraps(func)  # Preserve "identity" of original function
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                message = 'A critical error occurred.'
                message = gen_log_message(message=message)

                if on_error == 'abort':
                    logger.critical(message)
                    raise(e)
                else:
                    message = f'{message} The error message was:\n{str(e)}'
                    logger.critical(message)
        return wrapper
    return failsafe_run_decorator
