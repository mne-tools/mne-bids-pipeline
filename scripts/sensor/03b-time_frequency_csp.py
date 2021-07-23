""".

====================================================================
Decoding in time-frequency space using Common Spatial Patterns (CSP)
====================================================================

This file contains two main steps:
- 1. Decoding
The time-frequency decomposition is estimated by iterating over raw data that
has been band-passed at different frequencies. This is used to compute a
covariance matrix over each epoch or a rolling time-window and extract the CSP
filtered signals. A linear discriminant classifier is then applied to these
signals. More detail are available here:
https://mne.tools/stable/auto_tutorials/machine-learning/50_decoding.html#common-spatial-pattern
Warning: This step, especially the double loop on the time-frequency bins
is very computationally expensive.

- 2. Permutation statistics
We try to answer the following question: is the difference between
the two conditions statistically significant? We use the classic permutations
cluster tests on the time-frequency roc-auc map.
More details are available here:
https://mne.tools/stable/auto_tutorials/stats-sensor-space/10_background_stats.html

The user has only to specify the list of frequency and the list of timings.


Iterations levels:
    - contrasts
        - We iterate through subjects and sessions
    -       - If there are multiple runs, runs are concatenated into one big session.
"""
# License: BSD (3-clause)

import itertools
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import logging
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

from mne.stats.cluster_level import permutation_cluster_1samp_test
from mne.epochs import BaseEpochs
from mne import create_info, read_epochs, set_log_level, compute_rank
from mne.decoding import UnsupervisedSpatialFilter, CSP
from mne.time_frequency import AverageTFR
from mne.parallel import parallel_func
from mne.utils import BunchConst, ProgressBar
from mne.report import Report

from mne_bids import BIDSPath

from config import (N_JOBS, gen_log_message, on_error,
                    failsafe_run)
import config

logger = logging.getLogger('mne-bids-pipeline')
set_log_level(verbose="WARNING")  # mne logger


# ROC-AUC chance score level
chance = 0.5
csp_plot_patterns = False

# One PCA is fitted for each frequency bin.
# The usage of the pca is highly recommended for two reasons
# 1. The execution of the code is faster.
# 2. There will be much less numerical instabilities.
csp_use_pca = True

# typing
Session = Union[None, str]
Contrast = Tuple[str, str]


class Pth:
    """Util class containing useful Paths info."""

    def __init__(self, cfg) -> None:
        """Initialize directory. Initialize the base path."""
        # We initialize to the average subject
        # and to the default None session
        self.bids_basename = BIDSPath(
            subject="average",
            task=cfg.task,
            acquisition=cfg.acq,
            run=None,
            recording=cfg.rec,
            space=cfg.space,
            suffix='epo',
            extension='.fif',
            datatype=cfg.datatype,
            root=cfg.deriv_root,
            processing='clean',
            check=False)

    def file(
        self,
        *,
        subject: str,
        session: Session,
        cfg
    ) -> BIDSPath:
        """Return the path of the file."""
        return self.bids_basename.copy().update(
            subject=subject,
            session=session)

    def report(self, subject: str, session: Session, contrast: Contrast) -> BIDSPath:
        """Path to array containing the report."""
        return self.bids_basename.copy().update(
            processing='tf+csp' + self.contrast_suffix(contrast),
            subject=subject,
            session=session,
            suffix='report',
            extension='.html')

    def freq_scores(self, subject: str, session: Session, contrast: Contrast) -> BIDSPath:
        """Path to array containing the histograms."""
        return self.bids_basename.copy().update(
            processing='csp+freq' + self.contrast_suffix(contrast),
            subject=subject,
            session=session,
            suffix='scores',
            extension='.npy')

    def freq_scores_std(self, subject: str, session: Session, contrast: Contrast) -> BIDSPath:
        """Path to array containing the std of the histograms."""
        return self.bids_basename.copy().update(
            processing='csp+freq' + self.contrast_suffix(contrast),
            subject=subject,
            session=session,
            suffix='scores+std',
            extension='.npy')

    def tf_scores(self, subject: str, session: Session, contrast: Contrast) -> BIDSPath:
        """Path to time-frequency scores."""
        return self.bids_basename.copy().update(
            processing='csp+tf' + self.contrast_suffix(contrast),
            subject=subject,
            session=session,
            suffix='scores+std',
            extension='.npy')

    def contrast_suffix(self, contrast: Contrast):
        """Contrast suffix conform with Bids format."""
        con0 = config.sanitize_cond_name(contrast[0])
        con1 = config.sanitize_cond_name(contrast[1])
        return f'+contr+{con0}+{con1}'

    def prefix(self, subject: str, session: Session, contrast: Contrast) -> str:
        """Usefull when logging messages."""
        res = f'subject-{subject}-' + self.contrast_suffix(contrast)
        if session:
            return res + f'-session-{session}'
        else:
            return res


class Tf:
    """Util class containing useful info about the time frequency windows."""

    def __init__(self, cfg, freqs, times):
        """Calculate the required time and frequency size."""
        freqs = np.array(freqs)
        times = np.array(times)

        freq_ranges = list(zip(freqs[:-1], freqs[1:]))
        time_ranges = list(zip(times[:-1], times[1:]))

        n_freq_windows = len(freq_ranges)
        n_time_windows = len(time_ranges)

        # For band passed periodic signal,
        # according to the Nyquist theorem,
        # we can reconstruct the signal if f_s > 2 * band_freq
        min_band_freq = np.min(freqs[1:] - freqs[:-1])
        min_band_time = np.min(times[1:] - times[:-1])
        recommended_w_min_time = 1 / (2 * min_band_freq)

        if recommended_w_min_time > min_band_time:
            msg = ("We recommend increasing the duration of "
                   "your time intervals "
                   f"to at least {round(recommended_w_min_time, 2)}s.")
            logger.warning(gen_log_message(msg, step=8))

        centered_w_times = (times[1:] + times[:-1])/2
        centered_w_freqs = (freqs[1:] + freqs[:-1])/2

        self.freqs = freqs
        self.freq_ranges = freq_ranges
        self.times = times
        self.time_ranges = time_ranges
        self.centered_w_times = centered_w_times
        self.centered_w_freqs = centered_w_freqs
        self.n_time_windows = n_time_windows
        self.n_freq_windows = n_freq_windows

    def check_csp_times(self, epochs: BaseEpochs):
        """Check if csp_times is contained in the epoch interval."""
        # This test can only be performed after having read the Epochs file
        # So it cannot be performed in the check section of the config file.
        # 0.01 because epochs tmain end tmax are a bit approximate, even before decimation.
        if min(self.times) < epochs.tmin - 0.01 or max(self.times) > epochs.tmax + 0.01:
            wrn = ('csp_times should be contained in the epoch interval.'
                   f'But we do not have {epochs.tmin} < {self.times} < {epochs.tmax}')
            logger.warning(wrn)


def prepare_labels(*, epochs: BaseEpochs, contrast: Contrast, cfg) -> np.ndarray:
    """Return the projection of the events_id on a boolean vector.

    This projection is useful in the case of hierarchical events:
    we project the different events contained in one condition into
    just one label.

    PARAMETERS:
    -----------
    epochs:
        Should only contain events contained in contrast.

    Returns:
    --------
    A boolean numpy array containing the labels.
    """
    epochs_cond_0 = epochs[contrast[0]]
    event_id_condition_0 = set(epochs_cond_0.events[:, 2])  # type: ignore
    epochs_cond_1 = epochs[contrast[1]]
    event_id_condition_1 = set(epochs_cond_1.events[:, 2])  # type: ignore

    y = epochs.events[:, 2].copy()
    for i in range(len(y)):
        if y[i] in event_id_condition_0 and y[i] in event_id_condition_1:
            msg = (f"Event_id {y[i]} is contained both in "
                   f"{contrast[0]}'s set {event_id_condition_0} and in "
                   f"{contrast[1]}'s set {event_id_condition_1}."
                   f"{contrast} does not constitute a valid partition.")
            logger.critical(msg)
        elif y[i] in event_id_condition_0:
            y[i] = 0
        elif y[i] in event_id_condition_1:
            y[i] = 1
        else:
            # This should not happen because epochs should already by filtered
            msg = (f"Event_id {y[i]} is not contained in "
                   f"{contrast[0]}'s set {event_id_condition_0}  nor in "
                   f"{contrast[1]}'s set {event_id_condition_1}.")
            logger.critical(msg)
    return y


# @profile
def prepare_epochs_and_y(
    *,
    epochs: BaseEpochs,
    contrast: Contrast,
    cfg,
    fmin: float,
    fmax: float
) -> Tuple[BaseEpochs, np.ndarray]:
    """Band-pass and clean the epochs and prepare labels.

    Returns:
    --------
    epochs_filter, y
    """
    # Prepare epoch_filter
    epochs_filter = epochs.copy()

    epochs_filter.pick_types(
        meg=True, eeg=True, stim=False, eog=False,
        exclude='bads')

    # We only take gradiometers to speed up computation
    #  because the information is redundant between grad and mag
    if cfg.datatype == "meg":
        epochs_filter.pick_types(meg="mag")

    # filtering out the conditions we are not interested in.
    epochs_filter = epochs_filter[contrast]

    # We frequency filter after droping channel,
    # because freq filtering is costly
    epochs_filter = epochs_filter.filter(fmin, fmax, n_jobs=1)  # type: ignore

    # prepare labels
    y = prepare_labels(epochs=epochs_filter, contrast=contrast, cfg=cfg)

    return epochs_filter, y


def plot_frequency_decoding(
    *,
    freqs: np.ndarray,
    freq_scores: np.ndarray,
    conf_int: np.ndarray,
    subject: str,
    session: Session,
    pth: Pth
) -> Figure:
    """Plot and save the frequencies results.

    Show and save the roc-auc score in a 1D histogram for
    each frequency bin.

    Keep in mind that the confidence intervals indicated on the individual
    plot use only the std of the cross-validation scores.

    Parameters:
    -----------
    freqs
        The frequencies bins.
    freq_scores
        The roc-auc scores for each frequency bin.
    freq_scores_std
        The std of the cross-validation roc-auc scores for each frequency bin.
    subject
        name of the subject or "average" subject
    pth
        Pth class.

    Returns:
    -------
    Histogram with frequency bins.
    For the average subject, we also plot the std.
    """
    plt.close()
    fig, ax = plt.subplots()

    yerr = conf_int if len(subject) > 1 else None

    ax.bar(x=freqs[:-1], height=freq_scores, yerr=yerr,
           width=np.diff(freqs)[0],
           align='edge', edgecolor='black')

    # fixing the overlapping labels
    round_label = np.around(freqs)
    round_label = round_label.astype(int)
    ax.set_xticks(ticks=freqs)
    ax.set_xticklabels(round_label)

    ax.set_ylim([0, 1])

    ax.axhline(chance,
               color='k', linestyle='--',
               label='chance level')
    ax.legend()
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Decoding Roc-Auc Scores')
    CI_msg = "95% CI" if subject == "average" else "CV std score"
    ax.set_title(f'Frequency Decoding Roc-Auc Scores - {CI_msg}')

    return fig


def plot_time_frequency_decoding(
    *,
    tf_scores: np.ndarray,
    tf: Tf,
    pth: Pth,
    subject: str,
    session: Session
) -> Figure:
    """Plot and save the time-frequencies results.

    Parameters:
    -----------
    tf_scores
        The roc-auc scores for each time-frequency bin.
    sfreq
        Sampling frequency
    tf
        Util object for time frequency information.
    pth:
        Util object for paths.
    subject
        name of the subject.
    session
        name of the subject.

    Returns:
    -------
    The roc-auc score in a 2D map for each time-frequency bin.
    """
    plt.close()
    if np.isnan(tf_scores).any():
        msg = ("There is at least one nan value in one of "
               "the time-frequencies bins.")
        logger.info(gen_log_message(message=msg, step=8,
                                    subject=subject))
    tf_scores_ = np.nan_to_num(tf_scores, nan=chance)

    # Here we just use an sfreq random number just
    # to create a basic info structure.
    av_tfr = AverageTFR(create_info(['freq'], sfreq=1.),
                        # newaxis linked with the [0] in plot
                        tf_scores_[np.newaxis, :],
                        tf.centered_w_times, tf.centered_w_freqs, 1)

    # Centered color map around the chance level.
    max_abs_v = np.max(np.abs(tf_scores_ - chance))
    figs = av_tfr.plot(
        [0],  # [0] We do not have multiple channels here.
        vmax=chance + max_abs_v,
        vmin=chance - max_abs_v,
        title="Time-Frequency Decoding ROC-AUC Scores",
    )
    return figs[0]


def plot_patterns(csp, epochs_filter: BaseEpochs, report: Report, section: str, title: str):
    """Plot csp topographic patterns and save them in the reports.

    PARAMETERS
    ----------
    csp
        csp fitted estimator
    epochs_filter
        Epochs which have been band passed filtered and maybe time cropped.
    report
        Where to save the topographic plot.
    section
        choose the section of the report.
    title
        Title of the figure in the report.

    RETURNS
    -------
    None. Just save the figure in the report.
    """
    fig = csp.plot_patterns(epochs_filter.info)
    report.add_figs_to_section(
        fig,
        section=section,
        captions=section + title)


@failsafe_run(on_error=on_error)
# @profile
def one_subject_decoding(
    *,
    cfg,
    tf: Tf,
    pth: Pth,
    subject: str,
    session: Session,
    contrast: Contrast
) -> None:
    """Run one subject.

    There are two steps in this function:
    1. The frequency analysis.
    2. The time-frequency analysis.

    For each bin of those plot, we train a classifier to discriminate
    the two conditions.
    Then, we plot the roc-auc of the classifier.

    Returns
    -------
    None. We just save the plots in the report
    and the numpy results in memory.
    """
    msg = f"Running decoding for {pth.prefix(subject, session, contrast)}..."
    logger.info(gen_log_message(msg, step=8, subject=subject, session=session))

    report = Report(title=f"csp-permutations-sub-{subject}")

    # Extract information from the raw file
    epochs = read_epochs(pth.file(subject=subject,
                                  session=session,
                                  cfg=cfg))
    tf.check_csp_times(epochs)

    # compute maximal decimation possible
    # 3 is to take a bit of margin wrt Nyquist
    decimation_needed = epochs.info["sfreq"] / (3*tf.freqs[-1])
    print(decimation_needed)
    if decimation_needed > 2 and cfg.csp_quick:
        epochs.decimate(int(decimation_needed))
    print(epochs)

    # Assemble the classifier using scikit-learn pipeline
    csp = CSP(n_components=cfg.csp_n_components,
              reg=cfg.csp_reg,
              log=True, norm_trace=False)

    rank_dic = compute_rank(epochs, rank="info")
    print("rank_dic", rank_dic)
    print("cfg.datatype", cfg.datatype)
    rank = rank_dic[cfg.datatype]

    pca = UnsupervisedSpatialFilter(PCA(rank), average=False)
    clf = make_pipeline(csp, LinearDiscriminantAnalysis())
    cv = StratifiedKFold(n_splits=cfg.decoding_n_splits,
                         shuffle=True, random_state=cfg.random_state)

    freq_scores = np.zeros((tf.n_freq_windows,))
    freq_scores_std = np.zeros((tf.n_freq_windows,))
    tf_scores = np.zeros((tf.n_freq_windows, tf.n_time_windows))

    # Loop through each frequency range of interest
    for freq, (fmin, fmax) in ProgressBar(
            enumerate(tf.freq_ranges),
            max_value=tf.n_freq_windows,
            mesg=pth.prefix(subject, session, contrast) + '- frequency loop'):

        epochs_filter, y = prepare_epochs_and_y(
            epochs=epochs, contrast=contrast, fmin=fmin, fmax=fmax, cfg=cfg)

        ##########################################################################
        # 1. Loop through frequencies, apply classifier and save scores

        X = epochs_filter.get_data()
        X_pca = pca.fit_transform(X) if csp_use_pca else X
        # print(rank, X.shape) # 72 (53, 102, 961)
        # that mean we go from dim 102 to 72
        # the pca is way less usefull than the mag channel selection

        # Save mean scores over folds
        cv_scores = cross_val_score(estimator=clf, X=X_pca, y=y,
                                    scoring='roc_auc', cv=cv,
                                    n_jobs=1)

        freq_scores_std[freq] = np.std(cv_scores, axis=0)
        freq_scores[freq] = np.mean(cv_scores, axis=0)

        if csp_plot_patterns:
            X_inv = pca.inverse_transform(X_pca) if csp_use_pca else X
            csp.fit(X_inv, y)
            plot_patterns(
                csp, epochs_filter, report,
                section="CSP Patterns - frequency",
                title=f'{pth.prefix(subject, session, contrast)}-{(fmin, fmax)}Hz - all epoch')

        ######################################################################
        # 2. Loop through frequencies and time

        # Roll covariance, csp and lda over time
        for t, (w_tmin, w_tmax) in enumerate(tf.time_ranges):

            # Originally the window size varied accross frequencies...
            # But this means also that there is some mutual information between
            # 2 different pixels in the map.
            # So the simple way to deal with this is just to fix
            # the windows size for all frequencies.

            # Crop data into time-window of interest
            # Those two conditions enable to deal with numerical imprecisions
            # Without totally silencing the crop warning if needed
            # This occurs because, when we resample the data,
            # epochs tmin and tmax change because of the padding
            # This is a Patch which unfortunately makes the first and the last
            # time bin slightly smaller than the central bins.
            # So the central bins will have very slightly smaller rocauc scores than the
            # central ones.
            # But this slightly undesirable effect won't appear when
            # setting csp_quick to false.
            if epochs_filter.tmin - 0.1 < w_tmin < epochs_filter.tmin:
                w_tmin = epochs_filter.tmin
            if epochs_filter.tmax < w_tmax < epochs_filter.tmax + 0.1:
                w_tmax = epochs_filter.tmax
            X = epochs_filter.copy().crop(w_tmin, w_tmax).get_data()

            # transform or fit_transform ?
            X_pca = pca.transform(X) if csp_use_pca else X

            # Save mean scores over folds for each frequency and time window
            cv_scores = cross_val_score(estimator=clf,
                                        X=X_pca, y=y,
                                        scoring='roc_auc',
                                        cv=cv,
                                        n_jobs=1)
            tf_scores[freq, t] = np.mean(cv_scores, axis=0)

            # We plot the patterns using all the epochs
            # without splitting the epochs
            if csp_plot_patterns:
                X_inv = pca.inverse_transform(X_pca) if csp_use_pca else X
                csp.fit(X_inv, y)
                plot_patterns(
                    csp, epochs_filter, report,
                    section="CSP Patterns - time-frequency",
                    title=f'{pth.prefix(subject, session, contrast)}-{(fmin, fmax)}Hz-{(w_tmin, w_tmax)}s')

    # Frequency savings
    np.save(file=pth.freq_scores(subject, session, contrast), arr=freq_scores)
    np.save(file=pth.freq_scores_std(
        subject, session, contrast), arr=freq_scores_std)
    fig = plot_frequency_decoding(
        freqs=tf.freqs,
        freq_scores=freq_scores,
        conf_int=freq_scores_std,
        pth=pth,
        subject=subject,
        session=session)
    section = "Frequency roc-auc decoding"
    report.add_figs_to_section(
        fig,
        section=section,
        captions=section + pth.prefix(subject, session, contrast))

    # Time frequency savings
    np.save(file=pth.tf_scores(subject, session, contrast), arr=tf_scores)
    fig = plot_time_frequency_decoding(
        tf=tf, tf_scores=tf_scores, pth=pth,
        subject=subject, session=session)
    section = "Time-frequency decoding"
    report.add_figs_to_section(
        fig,
        section=section,
        captions=section + pth.prefix(subject, session, contrast))
    report.save(pth.report(subject, session, contrast), overwrite=True,
                open_browser=config.interactive)

    msg = f"Decoding for {pth.prefix(subject, session, contrast)} finished successfully."
    logger.info(gen_log_message(message=msg, subject=subject, session=session,
                                step=8))


def load_and_average(
    path: Callable[[str, Session, Contrast], BIDSPath],
    subjects: List[str],
    contrast: Contrast,
    shape: List[int],
    average: bool = True
) -> np.ndarray:
    """Load and average a np.array.

    We average between all subjects and all sessions.

    Parameters:
    -----------
    path
        function of the subject, returning the path of the numpy array.
    average
        if True, returns average along the subject dimension.
    shape
        The shape of the results.
        Either (freq) or (freq, times)

    Returns:
    --------
    The loaded array.

    Warning:
    --------
    Gives the list of files containing NaN values.
    """
    sessions = config.get_sessions()
    len_session = len(sessions) if type(sessions) == list else 1
    shape_all = [len(subjects) * len_session] + list(shape)
    res = np.zeros(shape_all)
    iterator = itertools.product(subjects, sessions)
    for i, (sub, ses) in enumerate(iterator):
        try:
            arr = np.load(path(sub, ses, contrast))
            # Checking for previous iteration, previous shapes
            if list(arr.shape) != shape:
                msg = f"Shape mismatch for {path(sub, ses, contrast)}"
                logger.warning(gen_log_message(
                    message=msg, subject=sub, step=8))
                raise FileNotFoundError
        except FileNotFoundError:
            print(FileNotFoundError)
            arr = np.empty(shape=shape)
            arr.fill(np.NaN)
        if np.isnan(arr).any():
            msg = f"NaN values were found in {path(sub, ses, contrast)}"
            logger.warning(gen_log_message(
                message=msg, subject=sub, session=ses, step=8))
        res[i] = arr
    if average:
        return np.nanmean(res, axis=0)
    else:
        return res


def plot_axis_time_frequency_statistics(
    *,
    ax: plt.Axes,
    array: np.ndarray,
    value_type: Literal['p', 't'],
    subjects: List[str],
    cfg,
    tf: Tf
) -> None:
    """Plot one 2D axis containing decoding statistics.

    Parameters:
    -----------
    ax: pyplot axis.
    array : np.ndarray
    value_type : either "p" or "t"
    cfg : BunchConst

    Returns:
    --------
    None. inplace modification of the axis.
    """
    ax.set_title(f"{value_type}-value")
    array = np.maximum(array, 1e-7) if value_type == "p" else array
    array = np.reshape(array, (tf.n_freq_windows, tf.n_time_windows))
    array = -np.log10(array) if value_type == "p" else array

    # Adaptive color
    lims = np.array([np.min(array), np.max(array)])

    img = ax.imshow(array, cmap='Reds', origin='lower',
                    vmin=lims[0], vmax=lims[1], aspect='auto',
                    extent=[np.min(tf.times), np.max(tf.times),
                            np.min(tf.freqs), np.max(tf.freqs)])

    ax.set_xlabel('time')
    ax.set_ylabel('frequencies')
    cbar = plt.colorbar(ax=ax, shrink=0.75, orientation='horizontal',
                        mappable=img, )
    cbar.set_ticks(lims)

    cbar.set_ticklabels([f'10$^{{{-round(lim, 1)}}}$' for lim in lims])
    cbar.ax.get_xaxis().set_label_coords(0.5, -0.3)
    if value_type == "p":
        cbar.set_label(r'p-value')
    if value_type == "t":
        cbar.set_label(r't-value')


def plot_t_and_p_values(
    t_values: np.ndarray,
    p_values: np.ndarray,
    title: str,
    subjects: List[str],
    cfg,
    tf: Tf
) -> Figure:
    """Plot t-values and either (p-values or clusters).

    Returns
    -------
    A figure with two subplot: t-values and p-values.
    """
    plt.close()
    fig = plt.figure(figsize=(10, 5))
    axes = [fig.add_subplot(121), fig.add_subplot(122)]

    plot_axis_time_frequency_statistics(
        ax=axes[0], array=t_values, cfg=cfg,
        subjects=subjects, value_type="t", tf=tf)
    plot_axis_time_frequency_statistics(
        ax=axes[1], array=p_values, cfg=cfg,
        subjects=subjects,  value_type="p", tf=tf)
    plt.tight_layout()
    fig.suptitle(title)
    return fig


def compute_conf_inter(
    mean_scores: np.ndarray,
    subjects: List[str],
    contrast: Contrast,
    cfg,
    tf: Tf
) -> Dict[str, Any]:
    """Compute the 95% confidence interval through bootstrapping.

    For the moment only serves for frequency histogram.

    TODO : copy pasted from https://github.com/mne-tools/mne-bids-pipeline/blob/main/scripts/sensor/04-group_average.py#L158
    Maybe we could create a common function in mne?

    PARAMETERS:
    -----------
    mean_scores: np.array((len(subjects)*nb_session, len(times)))

    RETURNS:
    --------
    Dictionnary of meta data. 
    """
    contrast_score_stats = {
        'cond_1': contrast[0],
        'cond_2': contrast[1],
        'times': tf.times,
        'N': len(subjects),
        'mean': np.empty(tf.n_freq_windows),
        'mean_min': np.empty(tf.n_freq_windows),
        'mean_max': np.empty(tf.n_freq_windows),
        'mean_se': np.empty(tf.n_freq_windows),
        'mean_ci_lower': np.empty(tf.n_freq_windows),
        'mean_ci_upper': np.empty(tf.n_freq_windows)}

    # Now we can calculate some descriptive statistics on the mean scores.
    # We use the [:] here as a safeguard to ensure we don't mess up the
    # dimensions.
    contrast_score_stats['mean'][:] = np.nanmean(mean_scores, axis=0)
    contrast_score_stats['mean_min'][:] = mean_scores.min(axis=0)
    contrast_score_stats['mean_max'][:] = mean_scores.max(axis=0)

    # Finally, for each time point, bootstrap the mean, and calculate the
    # SD of the bootstrapped distribution: this is the standard error of
    # the mean. We also derive 95% confidence intervals.
    rng = np.random.default_rng(seed=cfg.random_state)

    for time_idx in range(tf.n_freq_windows):
        scores_resampled = rng.choice(mean_scores[:, time_idx],
                                      size=(cfg.n_boot, len(subjects)),
                                      replace=True)
        bootstrapped_means = scores_resampled.mean(axis=1)

        # SD of the bootstrapped distribution == SE of the metric.
        se = bootstrapped_means.std(ddof=1)
        ci_lower = np.quantile(bootstrapped_means, q=0.025)
        ci_upper = np.quantile(bootstrapped_means, q=0.975)

        contrast_score_stats['mean_se'][time_idx] = se
        contrast_score_stats['mean_ci_lower'][time_idx] = ci_lower
        contrast_score_stats['mean_ci_upper'][time_idx] = ci_upper

        del bootstrapped_means, se, ci_lower, ci_upper

    # We cannot use the logger here
    print("Confidence intervals results:", mean_scores)

    return contrast_score_stats


@failsafe_run(on_error=on_error)
def group_analysis(
    subjects: List[str],
    contrast: Contrast,
    cfg,
    pth: Pth,
    tf: Tf
) -> None:
    """Group analysis.

    1. Average roc-auc scores:
        - frequency 1D histogram
        - time-frequency 2D color-map
    2. Perform statistical tests
        - plot t-values and p-values
        - performs classic cluster permutation test

    Returns
    -------
    None. Plots are saved in memory.
    """
    msg = "Running group analysis..."
    logger.info(gen_log_message(msg, step=8))

    if len(subjects) < 2:
        msg = "We cannot run a group analysis with just one subject."
        logger.warning(gen_log_message(message=msg, step=8))
        return None

    report = Report(title=f"csp-permutations-sub-average")

    ######################################################################
    # 1. Average roc-auc scores across subjects

    # Average Frequency analysis
    all_freq_scores = load_and_average(
        pth.freq_scores, subjects=subjects, contrast=contrast,
        average=False, shape=[tf.n_freq_windows])
    freq_scores = np.nanmean(all_freq_scores, axis=0)

    # Calculating the 95% confidence intervals
    contrast_score_stats = compute_conf_inter(
        mean_scores=all_freq_scores, contrast=contrast,
        subjects=subjects, cfg=cfg, tf=tf)

    fig = plot_frequency_decoding(
        freqs=tf.freqs, freq_scores=freq_scores, pth=pth,
        conf_int=contrast_score_stats["mean_se"],
        subject="average", session=None)
    section = "Frequency decoding"
    report.add_figs_to_section(
        fig,
        section=section,
        captions=section + ' sub-average')

    # Average time-Frequency analysis
    all_tf_scores = load_and_average(
        pth.tf_scores, subjects=subjects, contrast=contrast,
        shape=[tf.n_freq_windows, tf.n_time_windows])

    fig = plot_time_frequency_decoding(
        tf=tf, tf_scores=all_tf_scores, pth=pth,
        subject="average", session=None)
    section = "Time - frequency decoding"
    report.add_figs_to_section(
        fig,
        section=section,
        captions=section + ' sub-average')

    ######################################################################
    # 2. Statistical tests

    # Reshape data to what is equivalent to (n_samples, n_space, n_time)
    X = load_and_average(
        pth.tf_scores, subjects=subjects, average=False, contrast=contrast,
        shape=[tf.n_freq_windows, tf.n_time_windows])
    X = X - chance

    # Analyse with cluster permutation statistics
    titles = ['Without clustering']
    out = stats.ttest_1samp(X, 0, axis=0)
    ts: List[np.ndarray] = [np.array(out[0])]  # statistics
    ps: List[np.ndarray] = [np.array(out[1])]  # pvalues

    mccs = [False]  # these are not multiple-comparisons corrected

    titles.append('Clustering')
    # Compute threshold from t distribution (this is also the default)
    threshold = stats.distributions.t.ppf(  # type: ignore
        1 - cfg.cluster_stats_alpha_t_test, len(subjects) - 1)
    t_clust, clusters, p_values, H0 = permutation_cluster_1samp_test(
        X, n_jobs=1,
        threshold=threshold,
        adjacency=None,  # a regular lattice adjacency is assumed
        n_permutations=cfg.n_permutations, out_type='mask')

    msg = "Permutations performed successfully"
    logger.info(gen_log_message(msg, step=8))
    # Put the cluster data in a viewable format
    p_clust = np.ones((tf.n_freq_windows, tf.n_time_windows))
    for cl, p in zip(clusters, p_values):
        p_clust[cl] = p
    msg = (f"We found {len(p_values)} clusters "
           f"each one with a p-value of {p_values}.")
    logger.info(gen_log_message(msg, step=8))

    if len(p_values) == 0 or np.min(p_values) > cfg.cluster_stats_alpha:
        msg = ("The results are not significant. "
               "Try increasing the number of subjects.")
        logger.info(gen_log_message(msg, step=8))
    else:
        msg = (f"Congrats, the results seem significant. At least one of "
               f"your cluster has a significant p-value "
               f"at the level {cfg.cluster_stats_alpha}. "
               "This means that there is probably a meaningful difference "
               "between the two states, highlighted by the difference in "
               "cluster size.")
        logger.info(gen_log_message(msg, step=8))

    ts.append(t_clust)
    ps.append(p_clust)
    mccs.append(True)
    for i in range(2):
        fig = plot_t_and_p_values(
            t_values=ts[i], p_values=ps[i], title=titles[i],
            subjects=subjects, cfg=cfg, tf=tf)

        cluster = "with" if i else "without"
        section = f"Time - frequency statistics - {cluster} cluster"
        report.add_figs_to_section(
            fig,
            section=section,
            captions=section + ' sub-average')

    pth_report = pth.report("average", session=None, contrast=contrast)
    report.save(pth_report, overwrite=True,
                open_browser=config.interactive)
    msg = f"Report {pth_report} saved in the average subject folder"
    logger.info(gen_log_message(message=msg, step=8))

    msg = "Group statistic analysis finished."
    logger.info(gen_log_message(msg, step=8))


def get_config(
    subject: Optional[str] = None,
    session: Optional[str] = None
) -> BunchConst:
    cfg = BunchConst(
        datatype=config.get_datatype(),
        deriv_root=config.get_deriv_root(),
        decoding_n_splits=config.decoding_n_splits,
        task=config.get_task(),
        acq=config.acq,
        rec=config.rec,
        space=config.space,
        csp_quick=config.csp_quick,
        csp_freqs=config.csp_freqs,
        csp_times=config.csp_times,
        csp_n_components=config.csp_n_components,
        csp_reg=config.csp_reg,
        n_boot=config.n_boot,
        cluster_stats_alpha=config.cluster_stats_alpha,
        cluster_stats_alpha_t_test=config.cluster_stats_alpha_t_test,
        n_permutations=config.n_permutations,
        random_state=config.random_state
    )
    return cfg


def main():
    """Run all subjects decoding in parallel."""
    msg = 'Running Step 8: Time-frequency decoding'
    logger.info(gen_log_message(message=msg, step=8))

    cfg = get_config()

    if not config.contrasts:
        msg = ('contrasts was not specified. '
               'Skipping step Time-frequency decoding...')
        logger.info(gen_log_message(message=msg, step=8))
        return None

    # Calculate the appropriate time and frequency windows size
    tf = Tf(cfg, freqs=config.csp_freqs, times=config.csp_times)

    # Compute the paths
    pth = Pth(cfg=cfg)

    subjects = config.get_subjects()
    sessions = config.get_sessions()

    for contrast in config.contrasts:

        # Useful for debugging:
        # [one_subject_decoding(
        #     cfg=cfg, tf=tf, pth=pth,
        #     subject=subject, session=session, contrast=contrast)
        #     for subject, session in
        #  itertools.product(subjects, sessions)]

        parallel, run_func, _ = parallel_func(
            one_subject_decoding, n_jobs=N_JOBS)
        parallel(run_func(cfg=cfg, tf=tf, pth=pth,
                          subject=subject,
                          session=session,
                          contrast=contrast)
                 for subject, session in
                 itertools.product(subjects, sessions))

        # Once every subject has been calculated,
        # the group_analysis is very fast to compute.
        group_analysis(subjects=config.get_subjects(),
                       contrast=contrast,
                       cfg=cfg, pth=pth, tf=tf)

    msg = 'Completed Step 8: Time-frequency decoding'
    logger.info(gen_log_message(message=msg, step=8))


if __name__ == '__main__':
    main()