"""
Set of functions which support the functions from wucss1_functions.py

written by Simon Gross Nov-2023
"""

import pandas as pd
import numpy as np
from scipy import signal
from math import floor
from scipy.signal import periodogram, welch
from scipy.ndimage import gaussian_filter1d
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples
from matplotlib.colors import ListedColormap
import seaborn as sns

from functions import nexfile_custom as nc


def get_timestamps_for_epochs(segments, epoch_length, window_length=None, sliding_window=False):
    """
    function to get the timestamps of each epoch

    :param segments: list: of continous data for each segment with time and value
    :param epoch_length: float: time in seconds each epoch has
    :param window_length: float: time in seconds each window has
    :param sliding_window: boolean: if sliding windows are used

    :return: df_timestamps dataframe of timestamps for each start of epoch,
        epoch_numbers list of numbers each segment has,
        index_numbers list with a list of each index number in a segment
    """
    if not sliding_window:
        window_length = epoch_length

    # create empty dataframe for storing features
    df_timestamps = pd.DataFrame(data=None,columns=["timestamps"])

    # create empty list for epoch and index numbers
    epoch_numbers = []
    index_numbers = []

    # loop through the interval pieces
    for data in segments:
        timestamps = data[:, 0] - data[0, 0]

        epoch_number = int((floor(timestamps[-1] - timestamps[0]) - epoch_length) / window_length + 1)  # total number of epochs
        epoch_numbers.append(epoch_number)

        # temp storage of indexes to save indexes separated for each interval piece
        temp_index_numbers = []

        # loop through the different epochs in this piece of interval
        for i in range(epoch_number):
            # indexes of acc data in the selected epoch
            index = np.where((timestamps >= i * window_length) & (timestamps < (i * window_length + epoch_length)))
            temp_index_numbers.append(index)
            timestamp = data[index][0, 0]  # take the start time of the epoch
            df_timestamps = df_timestamps.append({"timestamps":timestamp},ignore_index=True)

        # append indexes to final index list
        index_numbers.append(temp_index_numbers)

    return df_timestamps, epoch_numbers, index_numbers


def extract_rem_ratio_feature(segments, epoch_numbers, index_numbers, df_timestamps, dict_feature, feature_name, fs=512., feature_smoothing=True):
    """
    function to extract features, specific to REM sleep

    :param segments: list: of continous data for each segment with time and value
    :param epoch_numbers: list: of numbers each segment has
    :param index_numbers: list: with a list of each index number in a segment
    :param df_timestamps: dataframe: of timestamps for each start of epoch
    :param dict_feature: dict: with key of feature name and as value the parameter e.g. low and high limit of bandpass filter
    :param feature_name: str: of name of feature
    :param fs: float: sampling rate
    :param feature_smoothing: boolean: if a smoothing of the feature should be applied

    :return: df_feature dataframe with timestamps for each epoch and the corresponding feature (theta/delta ratio) value,
        df_all_rem_features dataframe with timestamps for each epoch and all features (theta/detla ratio, theta power, delta power) values
    """
    # create empty dataframe for storing features
    df_feature = pd.DataFrame(data=None,columns=["timestamps",feature_name])
    df_features_temp = pd.DataFrame(data=None, columns=["timestamps"])  # temporary dataframe to store powers
    df_features_temp['timestamps'] = df_timestamps['timestamps']

    for band_name, boundary in dict_feature.items():
        df_feature_bandpass = extract_power_band_feature(segments, epoch_numbers, index_numbers, band_name, boundary['low'], boundary['high'],
                                                         channel_idx=2, fs=fs, feature_smoothing=feature_smoothing)
        df_features_temp = pd.merge(df_features_temp, df_feature_bandpass, how='left', on=['timestamps'])

    # Calculate power ratio
    feature_names = list(dict_feature.keys())   # assumes power bands are written in increasing order
    power_ratio_feature = np.log10(df_features_temp[feature_names[1]] / df_features_temp[feature_names[0]])

    # Fill the dataframe
    df_feature['timestamps'] = df_features_temp['timestamps']
    df_feature[feature_name] = power_ratio_feature

    df_all_rem_features = pd.DataFrame(data=None,columns=["timestamps",feature_name, feature_names[0],feature_names[1]])
    df_all_rem_features['timestamps'] = df_features_temp['timestamps']
    df_all_rem_features[feature_name] = power_ratio_feature
    df_all_rem_features[feature_names[0]] = df_features_temp[feature_names[0]]
    df_all_rem_features[feature_names[1]] = df_features_temp[feature_names[1]]

    return df_feature, df_all_rem_features


def extract_power_band_feature(segments, epoch_numbers, index_numbers, feature_name, bp_low, bp_high, channel_idx=1, fs=512., feature_smoothing=True):
    """
    function to extract the power in a specific band

    :param segments: list: of continous data for each segment with time and value
    :param epoch_numbers: list: of numbers each segment has
    :param index_numbers: list: with a list of each index number in a segment
    :param feature_name: str: of name of feature
    :param bp_low: float: low limit for filter
    :param bp_high: float: high limit for filter
    :param channel_idx: index number of channel that will be used
    :param fs: float: sampling rate
    :param feature_smoothing: boolean: if a smoothing of the feature should be applied

    :return: df_feature dataframe with timestamps for each epoch and the corresponding feature value
    """
    # create empty dataframe for storing features
    df_feature = pd.DataFrame(data=None,columns=["timestamps",feature_name])

    # loop through the interval pieces
    for data, epoch_number, index_number in zip(segments, epoch_numbers, index_numbers):
        timestamps = data[:, 0] - data[0, 0]
        signal = data[:, channel_idx]

        # Adaptly select the length of the filter based on the length of the interval piece
        if len(timestamps) / fs <= 12.5:
            n_sec_filter = 3.
        if (len(timestamps) / fs > 12.5) & (len(timestamps) / fs <= 30.):
            n_sec_filter = 4.
        if len(timestamps) / fs > 30.:
            n_sec_filter = 5.

        # get the bandpassed analytic signal
        analytic_signal = extract_power_band(signal, bp_low, bp_high, n_sec_filter, fs=fs)

        # create empty dataframe to store temporary the feature results
        df_feature_raw = pd.DataFrame(data=None, columns=["timestamps", feature_name])

        # loop through the different epochs in this piece of and extract power and timestamps of epoch
        for i, index in zip(range(epoch_number), index_number):
            feature = np.sum(np.abs(analytic_signal[index]))
            timestamp = data[index][0, 0]
            df_feature_raw = df_feature_raw.append({"timestamps":timestamp,feature_name:feature},ignore_index=True)

        # Gaussian smoothing to be applied before storing the features
        if feature_smoothing:
            feature_smooth = gaussian_smoothing(df_feature_raw[feature_name], sigma=.8)
            df_feature = df_feature.append(pd.DataFrame({"timestamps":df_feature_raw["timestamps"], feature_name:feature_smooth}),ignore_index=True)
        else:
            df_feature = df_feature.append(df_feature_raw)

    return df_feature


def extract_amplitude_feature(segments, epoch_numbers, index_numbers, feature_name, bp_low, bp_high, channel_idx=1, fs=512., feature_smoothing=True):
    """
    function to extract the amplitude in each epoch of the signal

    :param segments: list: of continous data for each segment with time and value
    :param epoch_numbers: list: of numbers each segment has
    :param index_numbers: list: with a list of each index number in a segment
    :param feature_name: str: of name of feature
    :param bp_low: float: low limit for filter
    :param bp_high: float: high limit for filter
    :param channel_idx: index number of channel that will be used
    :param fs: float: sampling rate
    :param feature_smoothing: boolean: if a smoothing of the feature should be applied

    :return: df_feature dataframe with timestamps for each epoch and the corresponding feature value
    """
    # create empty dataframe for storing features
    df_feature = pd.DataFrame(data=None,columns=["timestamps", feature_name])

    # loop through the interval pieces
    for data, epoch_number, index_number in zip(segments, epoch_numbers, index_numbers):
        timestamps = data[:, 0] - data[0, 0]
        signal = data[:, channel_idx]

        # Adaptly select the length of the filter based on the length of the interval piece
        if len(timestamps) / fs <= 12.5:
            n_sec_filter = 3.
        if (len(timestamps) / fs > 12.5) & (len(timestamps) / fs <= 30.):
            n_sec_filter = 4.
        if len(timestamps) / fs > 30.:
            n_sec_filter = 5.

        # get the bandpassed analytic signal
        bandpass_signal = extract_power_band(signal, bp_low, bp_high, n_sec_filter, return_bp=True, fs=fs)

        # create empty dataframe to store temporary the feature results
        df_feature_raw = pd.DataFrame(data=None, columns=["timestamps", feature_name])

        # loop through the different epochs in this piece of and extract min/max amplitude and timestamps of each epoch
        for i, index in zip(range(epoch_number), index_number):
            feature = np.max(bandpass_signal[index]) - np.min(bandpass_signal[index])
            timestamp = data[index][0, 0]
            df_feature_raw = df_feature_raw.append({"timestamps":timestamp,feature_name:feature},ignore_index=True)

        # Gaussian smoothing to be applied before storing the features
        if feature_smoothing:
            feature_smooth = gaussian_smoothing(df_feature_raw[feature_name], sigma=.8)
            df_feature = df_feature.append(pd.DataFrame({"timestamps":df_feature_raw["timestamps"], feature_name:feature_smooth}),ignore_index=True)
        else:
            df_feature = df_feature.append(df_feature_raw)

    return df_feature


def extract_entropy_feature(segments, epoch_numbers, index_numbers, feature_name, channel_idx=1, fs=512., feature_smoothing=True):
    """
    function to extract the entrophy in each epoch of the signal

    :param segments: list: of continous data for each segment with time and value
    :param epoch_numbers: list: of numbers each segment has
    :param index_numbers: list: with a list of each index number in a segment
    :param feature_name: str: of name of feature
    :param channel_idx: index number of channel that will be used
    :param fs: float: sampling rate
    :param feature_smoothing: boolean: if a smoothing of the feature should be applied

    :return: df_feature dataframe with timestamps for each epoch and the corresponding feature value
    """
    # create empty dataframe for storing features
    df_feature = pd.DataFrame(data=None,columns=["timestamps",feature_name])

    # loop through the interval pieces
    for data, epoch_number, index_number in zip(segments, epoch_numbers, index_numbers):
        signal = data[:, channel_idx]

        # create empty dataframe to store temporary the feature results
        df_feature_raw = pd.DataFrame(data=None, columns=["timestamps", feature_name])

        # loop through the different epochs in this piece of and extract min/max amplitude and timestamps of each epoch
        for i, index in zip(range(epoch_number), index_number):
            feature = SpecEntropy(signal[index], fs=fs, method='welch', normalize=True)
            timestamp = data[index][0, 0]
            df_feature_raw = df_feature_raw.append({"timestamps":timestamp,feature_name:feature},ignore_index=True)

        # Gaussian smoothing to be applied before storing the features
        if feature_smoothing:
            feature_smooth = gaussian_smoothing(df_feature_raw[feature_name], sigma=.8)
            df_feature = df_feature.append(pd.DataFrame({"timestamps":df_feature_raw["timestamps"], feature_name:feature_smooth}),ignore_index=True)
        else:
            df_feature = df_feature.append(df_feature_raw)

    return df_feature


def GaussianMixture_sleep_classification(n_clusters, X_feat, epsilon=0.1, max_iter=1e3):
    """
    function of algorithm to compute the probability densities of the clusters

    :param n_clusters: int: final number of clusters
    :param X_feat: numpy matrix: dataset to cluster, shape n_samples x n_features
    :param epsilon: float: minimum change to stop the algorithm; the smaller, the higher precision and computational time
    :param max_iter: int: maximum number of iterations to do. (lower number would also decrease computational time)

    :return: P with array of probability values for each epoch and cluster,
        gmm with parameters of model,
        indexes with list of low and high cluster index
    """

    np.random.seed(0)

    # Set up the GMM model
    gmm = GaussianMixture(n_components=n_clusters, n_init=10, tol=epsilon * 1e-2, max_iter=int(max_iter), random_state=0)
    # Fit the model
    gmm.fit(X_feat)
    # Centers
    V = gmm.means_
    # Get indexes in correct order
    indexes = get_indexes(V)
    # Predict probability
    P = gmm.predict_proba(X_feat)[:, indexes]

    return P, gmm, indexes


def get_indexes(centroids):
    """
    functions computes the indexes to put every stage in the correct order

    :param centroids: array: centroids found by the partition

    :return: indexes, list of which cluster is low and high in delta/theta
    """

    n_clust = centroids.shape[0]

    # Decide the indexes based on the delta and theta power
    if n_clust == 2:
        if centroids.shape[1] < 2:
            index_low = np.argmin(centroids[:, 0])
            index_high = np.argmax(centroids[:, 0])
        else:
            index_low = np.argmin(centroids[:, 1])
            index_high = np.argmax(centroids[:, 1])
        indexes = [index_low, index_high]

    if n_clust == 3:
        index_rem = np.argmax(centroids[:, 0])
        index_high = np.argmax(centroids[:, 2])
        assert index_rem != index_high, "Clusters are not well identified"
        index_low = [i for i in range(n_clust) if (i != index_rem) and (i != index_high)][0]  #np.argmin(centroids[[i for i in range(centroids.shape[0]) if i != index_rem], 2])
        indexes = [index_low, index_high, index_rem]

    return indexes


def compute_silhouettes_rem(n_clusters, features, thresholds, probability_matrix, dict_labels, colors, ax):
    """
    function to plot probabilities for epochs and calculate silhouette score

    :param n_clusters: int: of number of clusters
    :param features: array: of value of feature of each epoch
    :param thresholds: array: of thresholds to test
    :param probability_matrix: array: of probability values for each epoch and cluster
    :param dict_labels: dict: with label and name of each state
    :param colors: list: of color code for clusters
    :param ax: figure to plot subplot in

    :return: optimal_threshold_rem with float of threshold used,
        optimal_silhouette_rem with dict of cluster and average sihouette score,
        rem_labels with array of label for each epoch, ax with figure
    """
    silhouettes_rem = np.empty((len(thresholds), n_clusters + 1))  # Variable to store performance
    rem_label = int(probability_matrix.shape[1] - 1)  # value of the REM label
    for i, threshold in enumerate(thresholds):
        # First find optimal REM threshold
        rem_labels = np.array([rem_label if (p[rem_label] >= threshold) else np.argmax(p[:rem_label]) for p in probability_matrix])
        try:
            dict_silhouettes = silhouette_analysis(n_clusters, features, rem_labels, dict_labels)
            silhouettes_rem[i] = list(dict_silhouettes.values())
        except ValueError:
            silhouettes_rem[i] = np.array([-1] * (n_clusters + 1))
    # Check whether the final result contains NaNs (i.e. the number of clusters is reduced)
    for i, row in enumerate(np.isnan(silhouettes_rem)):
        if row.any():  # if one NaN is present, put the lowest score
            silhouettes_rem[i, -1] = -1
    optimal_threshold_rem = thresholds[np.argmax(silhouettes_rem[:, -1])]
    optimal_silhouette_rem = {dict_labels[0]: silhouettes_rem[:, 0][np.argmax(silhouettes_rem[:, -1])],
                              dict_labels[1]: silhouettes_rem[:, 1][np.argmax(silhouettes_rem[:, -1])]}

    rem_labels = np.array([rem_label if (p[rem_label] >= optimal_threshold_rem) else np.argmax(p[:rem_label]) for p in probability_matrix])

    # Plot results
    labels = ['NREM cluster', 'REM cluster']
    for i in range(silhouettes_rem.shape[1] - 1):
        ax.plot(thresholds, silhouettes_rem[:, i], '-o', color=colors[i], label=labels[i])
    ax.plot(thresholds, silhouettes_rem[:, -1], '-o', color=colors[-1], label='average')
    ax.plot(optimal_threshold_rem, np.max(silhouettes_rem[:, -1]), c='r', marker='*', markersize=20, label='Max Silhouette')
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_xlabel('threshold')
    ax.set_ylabel('Silhouette coefficients')
    ax.set_title('REM threshold optimization')
    ax.legend()
    ax.grid()

    return optimal_threshold_rem, optimal_silhouette_rem, rem_labels, ax


def compute_silhouettes_nrem(n_clusters, features, thresholds, probability_matrix, dict_labels, colors, ax, labels_rem=None):
    """
    function to plot probabilities for epochs and calculate silhouette score

    :param n_clusters: int: of number of clusters
    :param features: array: of value of feature of each epoch
    :param thresholds: array: of thresholds to test
    :param probability_matrix: array: of probability values for each epoch and cluster
    :param dict_labels: dict: with label and name of each state
    :param colors: list: of color code for clusters
    :param ax: figure to plot subplot in
    :param labels_rem: array: should be None in case no REM epochs are provided

    :return: optimal_threshold with float of threshold used,
        optimal_silhouette_nrem dict of cluster and average sihouette score,
        nrem_labels with array of label for each epoch,
        ax with figure
    """
    # Variables to store performance
    silhouettes_nrem = np.empty((len(thresholds), n_clusters + 1))
    if probability_matrix.shape[1] == 2:  # Only LS & DS
        for i, threshold in enumerate(thresholds):
            nrem_labels = np.array([0 if (p[0] >= threshold) else 1 for p in probability_matrix])
            try:
                dict_silhouettes = silhouette_analysis(2, features, nrem_labels, dict_labels)
                silhouettes_nrem[i] = list(dict_silhouettes.values())
            except ValueError:
                silhouettes_nrem[i] = np.array([-1] * (2 + 1))
    elif probability_matrix.shape[1] == 3:  # LS, DS and REM
        for i, threshold in enumerate(thresholds):
            nrem_labels = np.array([0 if p[0] >= threshold else 1 for p in probability_matrix[labels_rem != 2]])
            labels = labels_rem.copy()
            labels[labels_rem != 2] = nrem_labels
            try:
                dict_silhouettes = silhouette_analysis(3, features, labels)
                silhouettes_nrem[i] = list(dict_silhouettes.values())
            except ValueError:
                silhouettes_nrem[i] = np.array([-1] * (3 + 1))
    # Check whether the final result contains NaNs (i.e. the number of clusters is reduced)
    for i, row in enumerate(np.isnan(silhouettes_nrem)):
        if row.any():  # if one NaN is present, put the lowest score
            silhouettes_nrem[i, -1] = -1
    optimal_threshold = thresholds[np.argmax(silhouettes_nrem[:, -1])]
    optimal_silhouette_nrem = {dict_labels[0]: silhouettes_nrem[:, 0][np.argmax(silhouettes_nrem[:, -1])],
                               dict_labels[1]: silhouettes_nrem[:, 1][np.argmax(silhouettes_nrem[:, -1])]}

    if probability_matrix.shape[1] == 2:
        nrem_labels = np.array([0 if p[0] >= optimal_threshold else 1 for p in probability_matrix])
    elif probability_matrix.shape[1] == 3:
        nrem_labels = np.array([0 if p[0] >= optimal_threshold else 1 for p in probability_matrix[labels_rem != 2]])

    # Plot results
    labels = ['Light Sleep cluster', 'Deep Sleep cluster']
    for i in range(silhouettes_nrem.shape[1] - 1):
        ax.plot(thresholds, silhouettes_nrem[:, i], '-o', color=colors[i], label=labels[i])
    ax.plot(thresholds, silhouettes_nrem[:, -1], '-o', color=colors[-1], label='average')
    ax.plot(optimal_threshold, np.max(silhouettes_nrem[:, -1]), c='r', marker='*', markersize=20,
            label='Max Silhouette')
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_xlabel('threshold')
    ax.set_ylabel('Silhouette coefficients')
    ax.set_title('NREM threshold optimization')
    ax.legend()
    ax.grid()

    return optimal_threshold, optimal_silhouette_nrem, nrem_labels, ax


def silhouette_analysis(n_clusters, X, labels, dict_labels, ax=None, return_figure=False, colors=None, cluster_names=None):
    """
    function to calculate the silhouette scores for each epoch

    :param n_clusters: int: of number of clusters
    :param X: array: feature value
    :param labels: array: index of cluster for each epoch
    :param dict_labels: dict: with label and name of each state
    :param ax: figure: could be None
    :param return_figure: boolean: if true, figure will be plotted and returned
    :param colors: list: of color code, can be None
    :param cluster_names: list: of names for cluster, can be None

    :return: dict_silhouettes with dictionary of silhouette scores for each cluster and the average of all clusters combined
    """

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, labels)
    silhouette_avg = np.mean(sample_silhouette_values)
    # empty dictionary of silhouettes
    dict_silhouettes = {}

    silhouettes = []
    for i in range(n_clusters):
        sil = sample_silhouette_values[labels == i]
        silhouettes.append(sil)
        dict_silhouettes[dict_labels[i]] = np.mean(sil)
    dict_silhouettes["avg"] = silhouette_avg

    if return_figure:
        ax.set_xlim([-0.3, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])
        y_lower = 10

        for i, ith_cluster_silhouette_values in enumerate(silhouettes):
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = colors[i]
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                             0, ith_cluster_silhouette_values,
                             facecolor=color, edgecolor=color, alpha=0.7)
            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.2, y_lower + 0.5 * size_cluster_i, cluster_names[i])
            ax.text(0.7, y_lower + 0.5 * size_cluster_i, '<S> = %.3f' % (np.mean(ith_cluster_silhouette_values)))
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax.set_xlabel("Silhouette coefficient", fontsize=15)
        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_avg, color="k", linestyle="--")
        ax.set_yticks([])  # Clear the yaxis labels / ticks
        ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

        return dict_silhouettes, ax

    else:
        return dict_silhouettes


def plot_feature_space(nrem_features, indexes, features_names, colors, view_anlge, ax):
    """
    function to plot 3D cluster plot

    :param nrem_features: array: with value of each feature and epoch
    :param indexes: list: of nrem_features array matching the features_names
    :param features_names: list: of names of features provided
    :param colors: list: of color for each epoch
    :param view_anlge: list: of angle to look on 3D plot
    :param ax: figure: to that we add subplot

    :return: ax with figure and the subplot added
    """

    ax.scatter(nrem_features[:, indexes[0]], nrem_features[:, indexes[1]], nrem_features[:, indexes[2]], c=colors, marker='^', alpha=0.3)

    ax.set_xlim([-3., 5.])
    ax.set_ylim([-3., 5.])
    ax.set_zlim([-3., 5.])

    ax.set_xlabel(features_names[0])
    ax.set_ylabel(features_names[1])
    ax.set_zlabel(features_names[2])

    ax.view_init(view_anlge[0], view_anlge[1])

    return ax


###
# Helper functions
###
def extract_power_band(data, lim_low, lim_up, n_sec_filter, return_bp=False, fs=512.):
    """
    function extracts the power in a certain frequency band

    :param data: array: EEG data
    :param lim_low: float: lower limit of the band
    :param lim_up: float: upper limit of the band
    :param n_sec_filter: int: length of the filter in seconds
    :param return_bp: boolean: if analytical or the bandpassed signal should be returned
    :param fs: float: sampling frequency

    return: array of bandpass-filtered EEG data
    """

    # Some parameters for the powers extraction
    order = int(n_sec_filter * fs)  # order of the filter

    # Create bandpassed series
    bp_data = bandpass_filter(data, lim_low, lim_up, order, fs=fs)

    if return_bp:
        return bp_data
    else:
        # Create an analytic sygnal from the bandpassed series
        analytic_bp_data = signal.hilbert(bp_data, axis=0)
        return analytic_bp_data


def gaussian_smoothing(x, sigma):
    """
    function to perform gaussian smoothing with kernel sigma

    :param x: matrix: data with variables on diffrent columns
    :param sigma: float: gaussian kernel

    :return: y an array of the filtered signal
    """

    y = gaussian_filter1d(x, sigma, axis=0, mode='mirror')

    return y


def bandpass_filter(data, low_freq, high_freq, order, fs=512.0):
    """
    function to compute a FIR bandpass filter to apply to data.

    :param data: 2D array: n_timepoints x n_channels.
    :param low_freq: float: lower frequency of the bandpass.
    :param high_freq: float: higher frequency of the bandpass.
    :param order: int: length of filter kernel (in points) - 1; order must be even!
    :param fs: float: sampling frequency of the signal.

    :returns: data_filt with array of the band-passed signal
    """

    # Construct the filter to get the coefficients
    h = signal.firwin(order+1, [low_freq, high_freq], window='hamming', pass_zero=False, fs=fs)
    # Create an empy matrix to store the filterd signal
    data_filt = np.empty_like(data)
    data_filt = signal.filtfilt(h, 1., data, axis=0)

    return data_filt


def SpecEntropy(x, fs=512., method='fft', nperseg=None, normalize=False, axis=-1):
    """
    function to compute the spectral entropy

    :param x: array: of signal of one epoch
    :param fs: float: frequency of sampling in Hz
    :param method: str: 'fft' for Fourier Transform, 'welch' for Welch method
    :param nperseg: int or None: Length of each FFT segment for Welch method. If None (default), uses 256 samples
    :param normalize: boolean: If True, divide by log2(psd.size) to normalize the spectral entropy between 0 and 1
    :param axis: int: The axis along which the entropy is calculated. Default is -1 (last).

    :return: se with float of Spectral Entropy value for one epoch
    """

    x = np.asarray(x)
    # Compute and normalize power spectrum
    if method == 'fft':
        _, psd = periodogram(x, fs, axis=axis)
    elif method == 'welch':
        _, psd = welch(x, fs, nperseg=nperseg, axis=axis)
    psd_norm = psd / psd.sum(axis=axis, keepdims=True)
    se = -(psd_norm * np.log2(psd_norm)).sum(axis=axis)
    if normalize:
        se /= np.log2(psd_norm.shape[axis])
    return se


def create_single_itv(labels_pred, val, epoch_length):
    """
    function to get streches out of splitted segments

    :param labels_pred: array: with float of labels of each epoch for behavior state and epoch start
    :param val: int: with label of state to merge
    :param epoch_length: float: time in seconds each epoch has

    :return: array of starts and ends for merged interval
    """

    beg_times = labels_pred[labels_pred[:, 0] == val][:, 1]

    diff = [e - s for e, s in zip(beg_times[1:], beg_times[:-1])]
    starts = []
    ends = []
    starts.append(beg_times[0])
    for i, d in enumerate(diff):
        if round(d, 3) != round(epoch_length, 3):
            ends.append(beg_times[i] + epoch_length)
            starts.append(beg_times[i + 1])
    ends.append(beg_times[-1] + epoch_length)

    return np.array(starts), np.array(ends)


def fill_and_check_intervals(df):
    """
    function that corrects the timestamps of the intervals dataframe by checking that no equal consecutive intervals are present and filling the end times.

    :param df: dataframe with interval timestamps of each behavior state and level in hypnogram

    :return: df_cleaned with dataframe and cleaned intervals
    """

    df_cleaned = df.copy()
    # Correct for consecutive intervals present
    df_condition = df_cleaned['name'] == df_cleaned.shift()['name']
    idx = df_cleaned.loc[df_condition].index
    df_cleaned.drop(idx, inplace=True)
    # Additionally, reset also row index
    df_cleaned.reset_index(inplace=True, drop=True)
    # "fill the gaps" between intervals
    df_cleaned.iloc[:-1, 1] = df_cleaned.iloc[1:, 0].to_numpy() - 0.001

    return df_cleaned


def delete_short_intervals(df, itv_name, threshold):
    """
    function that deletes segments that are shorter than a certain threshold.

    :param df: dataframe: of all segments and states with start and end
    :param itv_name: str: with name of state that should be checked
    :param threshold: float: if shorter, segment will be deleted

    :return: cleaned_df with dataframe after short segments where deleted
    """

    # Create new dataframe to contain cleaned intervals
    df_condition = ((df.loc[df['name'] == itv_name, 'itv_end'] - df.loc[df['name'] == itv_name, 'itv_start']) <= threshold)
    indexes_to_delete = df.loc[df['name'] == itv_name][df_condition].index
    cleaned_df = df.drop(indexes_to_delete)

    cleaned_df = fill_and_check_intervals(cleaned_df)

    return cleaned_df


def merge_nearby_intervals(df, itv_name, threshold):
    """
    function to merge periods of two segments of the same state are close by

    :param df: dataframe: of all segments and states with start and end
    :param itv_name: str: with name of state that should be checked
    :param threshold: float: maximal separation, if shorter intervals will be merged

    :return: cleaned_df with dataframe after close by segments where merged
    """

    # Create new dataframe to contain cleaned intervals
    df_condition = ((df.loc[df['name'] == itv_name, 'itv_start'].shift(-1) - df.loc[df['name'] == itv_name, 'itv_end']) <= threshold)
    all_indexes = df_condition.index.to_numpy()
    merging_indexes = df.loc[df['name'] == itv_name][df_condition].index.to_numpy()
    indexes_to_delete = []
    for i in merging_indexes:
        idx_end = int(np.where(all_indexes == i)[0]) + 1
        indexes_to_delete += [j for j in range(i + 1, all_indexes[idx_end] + 1)]

    cleaned_df = df.drop(indexes_to_delete)

    cleaned_df = fill_and_check_intervals(cleaned_df)

    return cleaned_df


###
# functions for tansition matrices
###

def fill_labels_consecutive(starts, ends, epoch_length, labels, value):
    """
    function to fill a matrix of two columns with the epochs' stages on the first column and the timestamps on the second

    :param starts: list: of all starts of segments from one state
    :param ends: list: of all ends of segments from one state
    :param epoch_length: float: time in seconds each epoch has
    :param labels: array: with only epochs from one label of state and all epoch starts
    :param value: int: number to replace former label

    :return: labels with array of label of state and epoch starts
    """

    new_starts, new_ends = [], []
    for (s, e) in zip(starts, ends):
        if (s % epoch_length) < (epoch_length / 2):
            new_start = epoch_length * (s // epoch_length)
        else:
            new_start = epoch_length * (s // epoch_length + 1)
        new_starts.append(new_start)
        if (e % epoch_length) < (epoch_length / 2):
            new_end = epoch_length * (e // epoch_length) - 0.001
        else:
            new_end = epoch_length * (e // epoch_length + 1) - 0.001
        new_ends.append(new_end)

    for s, e in zip(new_starts, new_ends):
        if s != e:
            idx = np.where(((labels[:, 1] >= s) & (labels[:, 1] < e)))
            labels[idx, 0] = value

    return labels


def extract_all_predictions(doc, interval_list, df_intervals, epoch_length = 4.):
    """
    function that checks all intervals of different stages with detected artifacts and removes in case of overlap

    :param doc: str: with path to nex5 file
    :param interval_list: list: of interval names
    :param df_intervals: dataframe: of all segments and states with start and end
    :param epoch_length: float: time in seconds each epoch has

    :return: labels with array of label for each epoch in the first column (0->Act, 1->LS, 2->DS, 3->REM) and the timestamps on the second column
    """

    for itv in interval_list:

        if "_include" in itv:
            interval_variable = nc.read_interval_variables(doc, [itv])
            starts, ends = interval_variable[itv][0], interval_variable[itv][1]
            starts_incl, ends_incl = starts, ends
        elif "_rem_sleep" in itv:
            df_filtered = df_intervals[(df_intervals['name'] == 'ml_eeg_rem_sleep')]
            starts_rem, ends_rem = df_filtered['itv_start'].tolist(), df_filtered['itv_end'].tolist()
        elif "_light_sleep" in itv:
            df_filtered = df_intervals[(df_intervals['name'] == 'ml_eeg_light_sleep')]
            starts_ls, ends_ls = df_filtered['itv_start'].tolist(), df_filtered['itv_end'].tolist()
        elif "_deep_sleep" in itv:
            df_filtered = df_intervals[(df_intervals['name'] == 'ml_eeg_deep_sleep')]
            starts_ds, ends_ds = df_filtered['itv_start'].tolist(), df_filtered['itv_end'].tolist()
        elif "_active" in itv:
            df_filtered = df_intervals[(df_intervals['name'] == 'ml_eeg_active')]
            starts_act, ends_act = df_filtered['itv_start'].tolist(), df_filtered['itv_end'].tolist()
        else:
            msg = "Warning, interval %s could not be matched to any behavior state"
            print(msg)


    # Extract epoch times
    epoch_times = []
    for start, end in zip(starts_incl, ends_incl):
        if (start % epoch_length) < (epoch_length / 2):
            new_start = epoch_length * (start // epoch_length)
        else:
            new_start = epoch_length * (start // epoch_length + 1)
        if (end % epoch_length) < (epoch_length / 2):
            new_end = epoch_length * (end // epoch_length) + 0.001
        else:
            new_end = epoch_length * (end // epoch_length + 1) + 0.001

        epoch_number = int((floor(new_end - new_start) - epoch_length) / epoch_length + 1)  # total number of epochs
        epoch_times += [new_start + epoch_length * i for i in range(epoch_number)]
    epoch_times = np.array(epoch_times)

    labels = np.zeros((len(epoch_times), 2))
    labels[:, 1] = epoch_times

    labels = fill_labels_consecutive(starts_rem, ends_rem, epoch_length, labels, value=3)
    labels = fill_labels_consecutive(starts_ds, ends_ds, epoch_length, labels, value=2)
    labels = fill_labels_consecutive(starts_ls, ends_ls, epoch_length, labels, value=1)

    return labels


def compute_transition_matrices(labels, classes_names, start_time=None, stop_time=None, normalization=True):
    """
    This function computes the percentage proportions of different stages throughout the recordings

    :param labels: array: first column contains all the epochs of a session written as integers, the second column the associated timestamps
    :param classes_names: list: of strings with the stages' names corresponding in codes
    :param start_time: float: time of the recording from which to start considering the epochs, optional
    :param stop_time: float: time of the recording from which to stop considering the epochs, optional
    :param normalization: if values should be normalized and expressed in percentage rather than the number of transitions

    :return: df_transitions with dataframe of indexes and columns as stages and values as normalized transitions
    """

    if start_time is not None:
        labels1 = labels[labels[:, 1] >= start_time]
    else:
        labels1 = labels.copy()
    if stop_time is not None:
        labels2 = labels1[labels1[:, 1] < stop_time]
    else:
        labels2 = labels1.copy()

    # Loop through labels and store transitions in a DataFrame
    df_transitions = pd.DataFrame(0, index=classes_names, columns=classes_names)
    for t1, t2 in zip(labels2[:-1, 0], labels2[1:, 0]):
        if t1 != t2:
            df_transitions.loc[classes_names[int(t1)], classes_names[int(t2)]] += 1

    if normalization:
        # Normalize by rows
        df_transitions /= np.sum(df_transitions, axis=1).to_numpy().reshape(-1, 1)
        # Make percentage
        df_transitions *= 100

    return df_transitions


def plot_transition_matrix_one_dataset(df_data, ax, title, cmap="YlGnBu"):
    """
    function to plot the transition matrix of a single dataset

    :param df_data: dataframe: containing the transitions
    :param ax: figure: to that we add the subplot
    :param title: str: name to use as plot title
    :param cmap: str: colormap for figure

    :return: ax with figure containing the transition matrix
    """

    sns.heatmap(df_data, annot=True, fmt='.1f', annot_kws={'size': 20}, ax=ax, cbar=False, cmap=cmap)
    cmap1 = ListedColormap(['w'])
    sns.heatmap(df_data, mask=(df_data != 0), cmap=cmap1, cbar=False, ax=ax)
    ax.tick_params(axis='both', which='both', labelsize=14)
    ax.tick_params(axis='both', which='both', length=0)  # do not show ticks
    ax.set_title(title, fontsize=20)

    return ax