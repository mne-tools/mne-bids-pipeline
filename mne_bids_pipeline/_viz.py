from typing import List
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def plot_auto_scores(auto_scores, *, ch_types) -> List[Figure]:
    # Plot scores of automated bad channel detection.
    import matplotlib.pyplot as plt
    import seaborn as sns

    ch_types_ = list(ch_types)
    if 'meg' in ch_types_:  # split it
        idx = ch_types_.index('meg')
        ch_types_[idx] = 'grad'
        ch_types_.insert(idx + 1, 'mag')

    figs: List[Figure] = []
    for ch_type in ch_types_:
        # Only select the data for mag or grad channels.
        ch_subset = auto_scores['ch_types'] == ch_type
        if not ch_subset.any():
            continue  # e.g., MEG+EEG data with finding bads with MF enabled
        ch_names = auto_scores['ch_names'][ch_subset]
        scores = auto_scores['scores_noisy'][ch_subset]
        limits = auto_scores['limits_noisy'][ch_subset]
        bins = auto_scores['bins']  # The the windows that were evaluated.

        # We will label each segment by its start and stop time, with up to 3
        # digits before and 3 digits after the decimal place (1 ms precision).
        bin_labels = [f'{start:3.3f} – {stop:3.3f}'
                      for start, stop in bins]

        # We store the data in a Pandas DataFrame. The seaborn heatmap function
        # we will call below will then be able to automatically assign the
        # correct labels to all axes.
        data_to_plot = pd.DataFrame(data=scores,
                                    columns=pd.Index(bin_labels,
                                                     name='Time (s)'),
                                    index=pd.Index(ch_names, name='Channel'))

        # First, plot the "raw" scores.
        fig, ax = plt.subplots(1, 2, figsize=(12, 8))
        fig.suptitle(f'Automated noisy channel detection: {ch_type}',
                     fontsize=16, fontweight='bold')
        sns.heatmap(data=data_to_plot, cmap='Reds',
                    cbar_kws=dict(label='Score'), ax=ax[0])
        [ax[0].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
            for x in range(1, len(bins))]
        ax[0].set_title('All Scores', fontweight='bold')

        # Now, adjust the color range to highlight segments that exceeded the
        # limit.
        sns.heatmap(data=data_to_plot,
                    vmin=np.nanmin(limits),  # input data may contain NaNs
                    cmap='Reds', cbar_kws=dict(label='Score'), ax=ax[1])
        [ax[1].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
            for x in range(1, len(bins))]
        ax[1].set_title('Scores > Limit', fontweight='bold')

        # The figure title should not overlap with the subplots.
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        figs.append(fig)
    assert figs

    return figs
