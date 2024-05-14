"""
The task classifies sleep into light, deep and REM sleep. The module has been adapted to allow use without interaction with databases and NeuroExplorer.

written by Simon Gross Nov-2023
"""

import os
from functions import wucss1_functions as fun1
from functions import nexfile_custom as nc

# file names the classifier runs for
recordings_to_analyze = ["Cntnap001_200504.nex5"]

# root path and folder where the recording file can be found and the results should be saved in
root_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = 'example_dataset'
results_folder = 'results'

def main_task(file_name):
    """
    This function executes the main task of WUCSS
    1. Gets the signal of the channels from the nex5 file during a specific interval (has to be included in the file)
    2. Splits the signal into epochs and extracts features for each epoch
    3. Runs cluster detection for NREM/REM sleep
    4. Runs cluster detection for Light/Deep sleep
    5. Postprocessing of detected states
    6. Creates and saves a hypnogram
    7. Creates and saves a figure output that contains an image for threhold optimization, silhouette scores, 3D cluster of states and a transition matrix
    8. Saves a file with the start and end of all segments for each state, chronological sorted

    :param file_name: list: with name of the recording file that should be analyzed
    """
    # path to file and where results should go
    nexfile = os.path.join(root_dir, data_folder, file_name)
    res_folder = os.path.join(root_dir, results_folder)

    # channel for feature extraction
    channel_name_eeg_frontal = 'FP02'
    channel_name_eeg_parietal = 'FP04'
    feature_channel_names = [channel_name_eeg_frontal, channel_name_eeg_parietal]

    # interval names in file
    inactive_itv_name = 'eeg_inactive' #name of interval in the original file in which the animal was asleep
    active_itv_name = 'eeg_active' #name of interval in the original file in which the animal was awake
    include_itv_name = 'eeg_include' #name of interval that contains only good data (artifacts and noise excluded)
    wucss_min_inactive_duration = 1 # minimum that the inactive interval should have (in seconds)

    # create interval names for classified states
    new_active_itv_name = 'ml_eeg_active'
    new_inactive_interval = 'ml_eeg_inactive'
    new_rem_itv_name = 'ml_eeg_rem_sleep'
    new_nrem_itv_name = 'ml_eeg_nrem_sleep'
    light_sleep_itv_name = 'ml_eeg_light_sleep'
    deep_sleep_itv_name = 'ml_eeg_deep_sleep'

    # hard coded variables
    epoch_length = 4. # Parameters for epoch creation
    window_length = 4. # Parameters for epoch creation
    dict_labels = {0: light_sleep_itv_name, 1: deep_sleep_itv_name, 2: new_rem_itv_name} # label and name for the different states
    dict_postprocessing = {new_rem_itv_name: (12., 0.), light_sleep_itv_name: (4., 0.), deep_sleep_itv_name: (12., 0.)} # post-processing parameters, merging window (in seconds)
    rec_time = [0., 24.] #time period with start and end in hours that should be used for the transition matrix
    tm_interval_list = [include_itv_name, new_rem_itv_name, light_sleep_itv_name, deep_sleep_itv_name] #interval names that will be used for the transition matrix


    ###
    # Go!
    ###
    msg = "working with file %s" % nexfile
    print(msg)

    #check for inactive interval
    duration = nc.calculate_interval_durations(nexfile, [inactive_itv_name])
    if duration[inactive_itv_name] < wucss_min_inactive_duration:
        msg = "duration of inactive interval is %s seconds and therefore below the required duration, skipping recording" %(duration)
        print(msg)
        return

    # check for activity bout interval and exclude it from inactive interval
    activity_bout_name = None
    for label in ["emg_", "accel_"]:
        if nc.exists_variable(nexfile, label + "activity_bouts"):
            activity_bout_name = label + "activity_bouts"
    conservative_inactive_interval = nc.exclude_interval_from_other_interval(nexfile, inactive_itv_name, activity_bout_name)

    # get continuous data limited to time of interval
    inactive_segments_processed, fs = fun1.get_continous_during_interval(nexfile, feature_channel_names,
                                                                  conservative_inactive_interval, processing=True, min_length=10,
                                                                  time_threshold=0)
    msg = 'extracted continous file of %s for %s' % (str(feature_channel_names), inactive_itv_name)
    print(msg)

    # extract features from segments
    all_features, epoch_times, all_features_mean, all_features_std, rem_features, rem_features_mean, rem_feature_std = fun1.extract_features(inactive_segments_processed, epoch_length, window_length, fs)
    msg = 'extracted the features for the segments, shape of the dataset is %s' % str(all_features.shape)
    print(msg)

    # run REM sleep detection
    labels_rem, nrem_probability_matrix, optimization_fig, \
    silhouettes_rem, optimizing_threshold_rem, rem_feature_threshold = fun1.compute_and_optimize_rem_cluster(all_features, dict_labels, all_features_mean, all_features_std)
    msg = 'labeled NREM and REM epochs.'
    print(msg)

    # run NREM sleep detection
    segment_labels, silhouettes_nrem, optimizing_threshold_nrem, optimization_silhouette_fig = fun1.compute_and_optimize_nrem_cluster(all_features, dict_labels, labels_rem, nrem_probability_matrix, optimization_fig)
    msg = 'labeled DS and LS epochs.'
    print(msg)

    # postprocessing steps for intervals
    df_intervals = fun1.postprocessing_labels(nexfile, segment_labels, epoch_times, dict_labels, dict_postprocessing,
                                active_itv_name, new_active_itv_name, new_rem_itv_name, new_nrem_itv_name, epoch_length, new_inactive_interval, activity_bout_name)
    msg = 'segments post-processed'
    print(msg)

    # create hypnogram
    fun1.create_hypnogram(res_folder, df_intervals)
    msg = '\t --> saved figure of hypnogram'
    print(msg)

    # create transition matrices
    optimization_silhouette_cluster_fig = fun1.save_clustering_pictures(res_folder, all_features, rem_features,
                                                                         segment_labels, optimization_silhouette_fig)
    fun1.calculate_transition_matrices(nexfile, res_folder, tm_interval_list, optimization_silhouette_cluster_fig, df_intervals, rec_time)
    msg = '\t --> saved figure with silhouette score, cluster and transition matrix'
    print(msg)

    # save start and ends of intervals
    fname = os.path.join(res_folder, 'start_and_ends_intervals.csv')
    df_intervals.to_csv(fname)
    msg = '\t --> saved interval start and ends'
    print(msg)


    ###
    # Done!
    ###
    msg = "done with file %s" % nexfile
    print(msg)


if __name__ == "__main__":
    for file_name in recordings_to_analyze:
        main_task(file_name)

