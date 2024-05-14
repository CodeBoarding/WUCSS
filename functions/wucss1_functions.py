"""
Set of functions for the sleep classification, called during the main task

written by Simon Gross Nov-2023
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from functools import reduce

from functions import nexfile_custom as nc
from functions import wucss2_functions as fun2


def get_continous_during_interval(doc, channel, itv, processing=True, min_length=10, time_threshold=0):
    """
    function to retrieve the signal of a specific channel (channel) during a specific period (itv)

    :param doc: string: of path to nex5 file
    :param channel: list: channel (as list) to get continous data from
    :param itv: array: list of starts and list of ends for segments the signal should be used
    :param min_length: int: smoothing parameter in seconds of minimum duration of one segment
    :param time_threshold: int: time in seconds, everything below threshold is not used
    :param processing: boolean: if processing of intervals should be applied or not,
                                      i.e. removing short intervals or the ones starting before a certain time_threshold

    :return: itv_segments list of continous data for each segment with time and value,
        var_freq float of the sampling frequency
    """

    # get continous data (of good frontal channel) from nex file
    datamat, tstamps, var_freq, fragments_tstamps = nc.read_continuous_variables(doc, channel)
    # Transpose the array
    data_transposed = np.transpose(datamat)
    columns_names = ['time'] + ['value' + str(i+1) for i in range(data_transposed.shape[1])]
    df_channel = pd.DataFrame(np.concatenate((tstamps.reshape(-1, 1), data_transposed), axis=1), columns=columns_names)

    # Extract signal relative to itv
    itv_segments = []
    for (s, e) in zip(itv[0], itv[1]):
        t_temp, d_temp = (df_channel.loc[(df_channel['time'] >= s) & (df_channel['time'] < e)]['time'].to_numpy(),
                          df_channel.loc[(df_channel['time'] >= s) & (df_channel['time'] < e)].iloc[:,1:].to_numpy())
        itv_segments.append(np.concatenate((t_temp.reshape(-1, 1), d_temp), axis=1))

    if processing:
        itv_segments_processed = []
        for seg in itv_segments:
            if (seg.shape[0] > min_length * var_freq) & (seg[0, 0] > time_threshold):
                itv_segments_processed.append(seg)

        msg = '%s segments removed' % str(len(itv_segments) - len(itv_segments_processed))
        print(msg)
        itv_segments = itv_segments_processed

    return itv_segments, var_freq


def extract_features(segments, epoch_length, window_length, fs):
    """
    function to extract the different features for each epoch

    :param segments: list: of continous data for each segment with time and value
    :param epoch_length: float: time in seconds each epoch has
    :param window_length: float: time in seconds each window has
    :param fs: int: sampling frequency

    :return: nrem_features/rem_features dataframe with number_epochs x number_features containing the value of each epoch and feature,
        epoch_times array with start of each epoch,
        mean_features/mean_rem_features array with the mean of each feature,
        std_features/std_rem_features array with standard error of each feature
    """

    # hard coded variables, specific for NREM classification
    dict_features_bandpass = {"slow_oscillation": {"low": 1., "high": 3.},
                              "delta_oscillation": {"low": 3., "high": 6.},
                              "spindle_oscillation": {"low": 6., "high": 18.}}

    dict_features_amplitude = {"min_max_amplitude_1-20Hz": {"low": 1., "high": 20.}}

    # specific for REM sleep
    dict_features_rem_sleep = {"delta_oscillation": {"low": 1., "high": 4.},
                               "theta_oscillation": {"low": 5., "high": 9.}}

    #create empty dataframe for storing features
    df_all_features = pd.DataFrame(data=None, columns=["timestamps"])

    #get the timestamps of the epochs
    df_timestamps, epoch_numbers, index_numbers = fun2.get_timestamps_for_epochs(segments, epoch_length, window_length=window_length, sliding_window=False)
    df_all_features = df_all_features.append(df_timestamps)

    # get parietal power for REM sleep
    df_feature_rem, df_all_rem_features = fun2.extract_rem_ratio_feature(segments, epoch_numbers, index_numbers, df_timestamps, dict_features_rem_sleep, 'theta_delta_ratio',  fs=fs)
    df_all_features = pd.merge(df_all_features, df_feature_rem, how='left', on=['timestamps'])
    msg = 'calculated %s feature' % df_feature_rem.columns[1]
    print(msg)

    # get the power band related features
    for feature_name,boundary in dict_features_bandpass.items():
        df_feature_bandpass = fun2.extract_power_band_feature(segments, epoch_numbers, index_numbers, feature_name, boundary['low'], boundary['high'], fs=fs)
        df_all_features = pd.merge(df_all_features, df_feature_bandpass, how='left', on=['timestamps'])
        msg = 'calculated %s feature' % feature_name
        print(msg)

    # get the amplitude related features
    for feature_name,boundary in dict_features_amplitude.items():
        df_feature_amplitude = fun2.extract_amplitude_feature(segments, epoch_numbers, index_numbers, feature_name, boundary['low'], boundary['high'], fs=fs)
        df_all_features = pd.merge(df_all_features, df_feature_amplitude, how='left', on=['timestamps'])
        msg = 'calculated %s feature' % feature_name
        print(msg)

    #get the entropy feature
    df_feature_entropy = fun2.extract_entropy_feature(segments, epoch_numbers, index_numbers, "entropy", fs=fs)
    df_all_features = pd.merge(df_all_features, df_feature_entropy, how='left', on=['timestamps'])
    msg = 'calculated entropy feature'
    print(msg)

    #split timestamps and features and transform to array
    epoch_times = df_all_features["timestamps"].to_numpy()
    nrem_features = df_all_features.loc[:, df_all_features.columns != "timestamps"].to_numpy()
    rem_features = df_all_rem_features.loc[:, df_all_rem_features.columns != "timestamps"].to_numpy()

    scaler = StandardScaler()
    nrem_features = scaler.fit_transform(nrem_features)
    msg = 'Features are standardized'
    print(msg)

    # Save also mean and std of the features
    mean_features = scaler.mean_
    std_features  = np.sqrt(scaler.var_)

    rem_features = scaler.fit_transform(rem_features)
    mean_rem_features = scaler.mean_
    std_rem_features = np.sqrt(scaler.var_)

    return nrem_features, epoch_times, mean_features, std_features, rem_features, mean_rem_features, std_rem_features


def cluster_and_label_segments(n_clusters, features, return_prob):
    """
    function to find cluster and provide probability for each epoch to cluster

    :param n_cluster: int: number of clusters to find
    :param features: array: with number_epochs x number_features containing the value of each epoch and feature
    :param return_prob: boolean: if return of probability matrix alone or not

    :return: prob_matrix with array of probability values for each epoch and cluster,
        GMM with parameters of model,
        indexes with list of low and high cluster index
    """

    # Predict probability
    prob_matrix, GMM, indexes = fun2.GaussianMixture_sleep_classification(n_clusters, features)

    if return_prob:
        return prob_matrix
    else:
        return prob_matrix, GMM, indexes


def compute_and_optimize_rem_cluster(features, dict_labels, features_mean, features_std):
    """
    function to find optimal threshold for nrem/rem cluster

    :param features: array: with number_epochs x number_features containing the value of each epoch and feature
    :param dict_labels: dict: with label and name of each state
    :param features_mean: array: with the mean of each feature
    :param features_std: array: with standard error of each feature

    :return: labels_rem with array of cluster index for each epoch,
        nrem_probability_matrix with array of probability values for each epoch and cluster,
        fig with figure that contains added subplot
        silhouettes_rem with dictionary of silhouette score of clusters,
        optimizing_threshold_rem with float of threshold value used,
        x_threshold with float of non-normalized coordinate
    """

    # Hard-coded threshold values to check
    first_threshold_values = np.array([1e-3, 1e-2, 0.05])
    mid_threshold_values = np.arange(0.1, 1, 0.1)
    last_threshold_values = np.array([0.95, 1 - 1e-2, 1 - 1e-3])
    thresholds_values = np.concatenate((first_threshold_values, mid_threshold_values, last_threshold_values))

    fig = plt.figure(figsize=[12, 24])
    ax1 = fig.add_subplot(4, 2, 1)
    colors = ['b', 'g', 'k']
    # compute probability matrix from the clustering
    prob_matrix, GMM, idx = cluster_and_label_segments(2, features[:, 0:1], return_prob=False)  # use only theta / delta feature
    rem_itv_name = dict_labels[2]
    nrem_itv_name = rem_itv_name.replace('rem', 'nrem')
    dict_labels_rem = {0: nrem_itv_name, 1: rem_itv_name}
    optimizing_threshold_rem, silhouettes_rem, labels_rem, ax1 = fun2.compute_silhouettes_rem(2, features[:, 0:1], thresholds_values,
                                                                                                   prob_matrix, dict_labels_rem, colors, ax1)

    ax3 = fig.add_subplot(4, 2, 3)
    _, ax3 = fun2.silhouette_analysis(2, features[:, 0:1], labels_rem, dict_labels_rem, ax=ax3, return_figure=True, colors=colors, cluster_names=['NREM\nSleep',
                                                                                                                                                       'REM\nSleep'])
    labels_rem[labels_rem == 1] = 2  # change value to REM epochs
    nrem_features = features[labels_rem != 2]  # features to use for NREM classification
    nrem_probability_matrix = cluster_and_label_segments(2, nrem_features[:, 1:], return_prob=True)

    # Compute the value of the REM threshold
    # Construct grid of values
    x = np.arange(np.min(features[:, 0:1]), np.max(features[:, 0:1]), 0.01)
    posterior_prob = GMM.predict_proba(x.reshape(-1,1))[:, idx]
    x_threshold = x[np.argmin(np.abs(posterior_prob[:, -1] - optimizing_threshold_rem))]
    # transform it into non-normalized coordinates
    x_threshold = x_threshold * features_std[0] + features_mean[0]

    return labels_rem, nrem_probability_matrix, fig, silhouettes_rem, optimizing_threshold_rem, x_threshold


def compute_and_optimize_nrem_cluster(features, dict_labels, labels_rem, nrem_prob_matrix, fig):
    """
    function to find optimal threshold for nrem/rem cluster

    :param features: array: with number_epochs x number_features containing the value of each epoch and feature
    :param dict_labels: dict: with label and name of each state
    :param labels_rem: array: with label of each epoch for nrem or rem
    :param nrem_prob_matrix: array: with probability of cluster for each epoch of nrem
    :param fig: figure to that we add subplots

    :return: labels_final array with labels for all clusters of each epoch,
        silhouettes_nrem with dictionary of silhouette score of clusters,
        optimizing_threshold with float of threshold value used,
        fig with figure that contains added subplot
    """

    # Hard-coded threshold values to check
    first_threshold_values = np.array([1e-3, 1e-2, 0.05])
    mid_threshold_values = np.arange(0.1, 1, 0.1)
    last_threshold_values = np.array([0.95, 1 - 1e-2, 1 - 1e-3])
    thresholds_values = np.concatenate((first_threshold_values, mid_threshold_values, last_threshold_values))

    colors = ['dodgerblue', 'm', 'k']
    ax2 = fig.add_subplot(4, 2, 2)

    nrem_features = features[labels_rem != 2]
    dict_labels_nrem = dict([(key, dict_labels[key]) for key in [0, 1]])
    optimizing_threshold, silhouettes_nrem, labels_nrem, ax2 = fun2.compute_silhouettes_nrem(2, nrem_features[:,1:], thresholds_values,
                                                                                                  nrem_prob_matrix, dict_labels_nrem, colors, ax2)
    ax4 = fig.add_subplot(4, 2, 4)
    _, ax4 = fun2.silhouette_analysis(2, nrem_features[:,1:], labels_nrem, dict_labels_nrem, ax=ax4,
                                           return_figure=True, colors=colors, cluster_names=['Light\nSleep','Deep\nSleep'])
    labels_final = labels_rem.copy()
    labels_final[labels_rem != 2] = labels_nrem

    return labels_final, silhouettes_nrem, optimizing_threshold, fig


def postprocessing_labels(doc, raw_labels, epoch_times, dict_labels, dict_postprocessing, active_itv_name,
                          ml_active_name, ml_rem_name, ml_nrem_name, epoch_length, ml_inactive_name, bouts_itv_name=None):
    """
    function to clean up intervals, meaning discarding short bouts and merge similar states that are close by

    :param doc: string: path to nex5 file
    :param raw_labels: array: of labels after the clustering for each epoch
    :param epoch_times: array: of timestamps for each epoch
    :param dict_labels: dict: of cluster label and name for interval
    :param dict_postprocessing: dict: of interval name and postprocessing parameters
    :param active_itv_name: str: of the name of the active interval
    :param ml_active_name: str: of the name of the new active interval
    :param ml_rem_name: str: of the name of the new rem interval
    :param ml_nrem_name: str: of the name of the new nrem interval
    :param epoch_length: float: of time in seconds of each epoch
    :param ml_inactive_name: str: of the name of the new inactive interval
    :param bouts_itv_name: str: of the name of the bouts interval, can be None

    :return: df_intervals with dataframe of all segments and states with start and end
    """

    # Create complete labels with timestamps
    complete_labels = np.c_[raw_labels, epoch_times]

    # Extract intervals from nex5 file
    interval_variable = nc.read_interval_variables(doc, [active_itv_name, bouts_itv_name])
    starts_active, ends_active = interval_variable[active_itv_name][0], interval_variable[active_itv_name][1]
    if bouts_itv_name is not None:
        try:  # if interval exists
            starts_bouts, ends_bouts = interval_variable[bouts_itv_name][0], interval_variable[bouts_itv_name][1]
        except:
            starts_bouts, ends_bouts = [], []
        # for this task, consider active and bouts together as activity
        starts_ml_active, ends_ml_active = list(starts_active) + list(starts_bouts), list(ends_active) + list(ends_bouts)
        starts_ml_active.sort()
        ends_ml_active.sort()
    else:
        starts_ml_active, ends_ml_active = list(starts_active), list(ends_active)

    label_LS = list(dict_labels.keys())[0]
    label_DS = list(dict_labels.keys())[1]
    label_REM = list(dict_labels.keys())[2]
    try:
        starts_LS, ends_LS = fun2.create_single_itv(complete_labels, val=label_LS, epoch_length=epoch_length)
    except:
        starts_LS, ends_LS = [], []
    try:
        starts_DS, ends_DS = fun2.create_single_itv(complete_labels, val=label_DS, epoch_length=epoch_length)
    except:
        starts_DS, ends_DS = [], []
    try:
        starts_REM, ends_REM = fun2.create_single_itv(complete_labels, val=label_REM, epoch_length=epoch_length)
    except:
        starts_REM, ends_REM = [], []
    # Put together intervals and process them
    df_intervals = pd.DataFrame(columns=['itv_start', 'itv_end', 'name', 'itv_level'])
    df_intervals.itv_start = np.concatenate((starts_ml_active, starts_REM, starts_LS, starts_DS), axis=0)
    df_intervals.itv_end = np.concatenate((ends_ml_active, ends_REM, ends_LS, ends_DS), axis=0)
    df_intervals.name = ([ml_active_name] * len(starts_ml_active) + [dict_labels[label_REM]] * len(starts_REM) +
                         [dict_labels[label_LS]] * len(starts_LS) + [dict_labels[label_DS]] * len(starts_DS))
    df_intervals.itv_level = ([0] * len(starts_ml_active) + [-1] * len(starts_REM) +
                         [-2] * len(starts_LS) + [-3] * len(starts_DS))
    df_intervals.sort_values(by='itv_start', inplace=True, ignore_index=True)  # order intervals in chronological order
    df_intervals = fun2.fill_and_check_intervals(df_intervals)
    # Loop through the intervals to process them
    for key, value in dict_postprocessing.items():
        threshold_merge, threshold_min = value
        df_intervals = fun2.merge_nearby_intervals(df_intervals, key, threshold_merge)
        df_intervals = fun2.delete_short_intervals(df_intervals, key, threshold_min)

    return df_intervals


###
# function for output
###

def save_clustering_pictures(folder, all_features, rem_features, labels, fig):
    """
    function to plot and save the 3D cluster image

    :param folder: str: containing the path of the analysis folder where to save the pictures.
    :param all_features: array: with values of every nrem feature and epoch
    :param rem_features: array: with values of every rem feature and epoch
    :param labels: array: containing all the classified epochs and the corresponding times
    :param fig: figure: to that we add the subplot

    :return: fig, with added subplot of 3D cluster image
    """
    #create feature and label arrays without rem epochs
    df1 = pd.DataFrame(all_features, columns=["1","2","3","4","5","6"])
    df2 = pd.DataFrame(rem_features, columns=["7","8","9"])
    df3 = pd.DataFrame(labels, columns=["label"])
    df_concat = pd.concat([df1, df2, df3], axis=1)
    df_wo_rem = df_concat[df_concat.label != 2]
    features_wo_rem = df_wo_rem.loc[:,["1","2","3","4","5","6"]].to_numpy()
    labels_wo_rem = np.transpose(df_wo_rem.loc[:,["label"]].to_numpy())

    # Create NREM/REM cluster
    colors = ['b' if l == 0 else ('b' if l == 1 else 'g') for l in labels]
    ax1 = fig.add_subplot(4,2,5, projection='3d')
    features_names = [r'$\theta$ power', r'$\theta / \delta$ ratio', r'$\delta$ power']
    ax1 = fun2.plot_feature_space(rem_features, [2, 0, 1], features_names, colors, [50, 10], ax1)

    # Create DS/LS cluster
    colors = ['dodgerblue' if l == 0 else ('m' if l == 1 else 'b') for l in labels_wo_rem[0]]
    ax2 = fig.add_subplot(4,2,6, projection='3d')
    features_names = [r'$\alpha$ power', r'Amplitude', r'Spectral Entropy']
    ax2 = fun2.plot_feature_space(features_wo_rem, [3, 4, 5], features_names, colors, [50, 10], ax2)

    return fig


def create_hypnogram(folder, df_intervals):
    """
    function to plot and save the hypnogram

    :param folder: str: name of the analysis folder where to save the picture.
    :param df_intervals: dataframe: of all segments and states with start and end
    """

    df_intervals.sort_values(by='itv_start', inplace=True, ignore_index=True)

    fig, ax = plt.subplots(1, 1, figsize=[150, 5])
    for index, row in df_intervals.iterrows():
        plt.plot([row['itv_start'], row['itv_end']], [row['itv_level'], row['itv_level']], 'b', linewidth=0.5)
        if index != (df_intervals.shape[0] - 1):
            plt.plot([row['itv_end'], row['itv_end']], [row['itv_level'], df_intervals.loc[index+1,'itv_level']], 'b', linewidth=0.5)

    plt.xlabel('t (s)', fontsize=20)
    plt.yticks([0, -1, -2, -3], ['Wake', 'REM', 'Light\nSleep', 'Deep\nSleep'])
    xmin, xmax = plt.xlim()
    plt.xticks(np.arange(0, xmax, 2000))
    plt.title('Hypnogram', fontsize=35)
    saving_path = os.path.join(folder, 'Hypnogram.png')
    fig.savefig(saving_path, format='png')
    plt.close()


def calculate_transition_matrices(doc, res_folder, interval_list, fig, df_intervals, rec_time = [0., 24.]):
    """
    function to calculate the transition matrices
    1. retrieve start and ends of all intervals
    2. calculate based on recording time and normalization type the matrix
    3. create figure of transition matrix and create a subplot for both, with and without normalization

    :param doc: string: nex5 path
    :param res_folder: string: path to folder in which figure should be saved
    :param interval_list: list: intervals used for transition matrix
    :param fig: figure: of previous wucss outputs, at that point 6 subplots
    :param df_intervals: dataframe: of all segments and states with start and end
    :param rec_time: list: of time span the transitions will be calculated for (in hours)

    :return: results with list of rows that contain numbers of transitions between two states
    """
    # get start and end of intervals
    labels = fun2.extract_all_predictions(doc, interval_list, df_intervals, epoch_length = 4.)

    normalization_type = [True, False]
    i = 7
    results = list()
    for norm in normalization_type:
        if norm:
            add_string = "norm"
        else:
            add_string = "w/o norm"
        all_tm_raw = []
        df_tm_raw = fun2.compute_transition_matrices(labels=labels, classes_names=['Active', 'LS', 'DS', 'REM'],
                                            start_time=rec_time[0] * 3600, stop_time=rec_time[1] * 3600, normalization=norm)
        all_tm_raw.append(df_tm_raw)
        df_tm = reduce(lambda x, y: x.add(y, fill_value=0), all_tm_raw) / 1

        plot_title_name = "%s: %sh - %sh" %(add_string, rec_time[0], rec_time[1])

        ax = fig.add_subplot(4, 2, i)
        ax = fun2.plot_transition_matrix_one_dataset(df_tm, ax, title=plot_title_name)
        i += 1

        # prepare results for database of raw transition numbers
        if not norm:
            # rearange dataframe
            df_transitions_raw = pd.DataFrame(columns=['From', 'To', 'Value'])
            for idx in df_tm.index:
                for col in df_tm.columns:
                    row = [idx, col, float(df_tm.loc[idx, col])]
                    df_transitions_raw = df_transitions_raw.append(
                        pd.DataFrame(np.array(row).reshape(1, -1), columns=['From', 'To', 'Value']), ignore_index=True)

            # prepare raw results
            for idx, row in df_transitions_raw.iterrows():
                result_type = "tm_" + row["From"] + "To" + row["To"]
                numeric_value = float(row["Value"])
                row = (result_type, np.round(numeric_value, 4), 'None', None)
                results.append(row)

    pathsplt = os.path.split(doc)
    nex_fname = pathsplt[1]
    figure_name = nex_fname.replace(".nex5", "_wucss_qc.png")
    saving_path = os.path.join(res_folder, figure_name)
    fig.savefig(saving_path, format='png')

    #return results