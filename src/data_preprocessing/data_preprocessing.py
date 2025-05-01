import os
import actipy
import numpy as np
import pandas as pd

from src.data_download import make_1_to_1_corresp, unpack_labelled_dataset, unpack_unlabelled_dataset

# Main method for preprocessing dataset
def make_dataset(
        dataset_cfg,
        paths_cfg,
        target_sampling_rate,
        low_pass_filter_freq,
        try_cached=True,
        normalize_data=False,
        win_len_s=30,
        dataset_path=None,
):
    dataset_name = dataset_cfg.name
    dataset_sample_rate = dataset_cfg.sampling_rate
    cache_path = paths_cfg.processed_data
    label_dict = dataset_cfg.labels

    if dataset_path is None:
        dataset_path = os.path.join(paths_cfg.datasets, dataset_cfg.unpacked_path)

    loaded_from_cache = False
    if try_cached:
        (
            loaded_from_cache,
            motion_data_list,
            labels_list,
            subject_ids,
        ) = load_preprocessed_dataset(
            dataset_name,
            dataset_cfg.has_labels,
            low_pass_filter_freq,
            target_sampling_rate,
            win_len_s,
            normalize_data,
            cache_path,
        )

    if loaded_from_cache and dataset_cfg.has_labels:
        return motion_data_list, labels_list, subject_ids
    elif loaded_from_cache and not dataset_cfg.has_labels:
        return motion_data_list, subject_ids

    if dataset_cfg.has_labels:
        # Load raw dataset
        motion_data_list, labels_list, subject_ids = unpack_labelled_dataset(
            dataset_name,
            dataset_path,
            label_dict,
        )
    else:
        # Load raw dataset
        motion_data_list, subject_ids = unpack_unlabelled_dataset(
            dataset_name, dataset_path)

    for i in range(len(motion_data_list)):
        motion_data_sample, _ = actipy.process(
            data=motion_data_list[i],
            sample_rate=dataset_sample_rate,
            lowpass_hz=low_pass_filter_freq,
            calibrate_gravity=True,
            detect_nonwear=True,
            resample_hz=target_sampling_rate,
            verbose=True,
        )
        win_len_samples = int(win_len_s * target_sampling_rate)

        if dataset_cfg.has_labels is True:
            motion_data_sample = process_labelled_data(labels_list[i], motion_data_sample, win_len_samples)

        motion_data_sample = process_motion_data(motion_data_sample, normalize_data, target_sampling_rate, value_clip=2,
                                                 win_len_s=win_len_s,
                                                 win_len_samples=win_len_samples)
        motion_data_list[i] = motion_data_sample

    # Save preprocessed dataset to disk
    save_preprocessed_dataset(
        dataset_name,
        motion_data_list,
        labels_list,
        subject_ids,
        low_pass_filter_freq,
        target_sampling_rate,
        win_len_s,
        normalize_data,
        cache_path,
    )

    if dataset_cfg.has_labels:
        return motion_data_list, labels_list, subject_ids
    else:
        return motion_data_list, subject_ids

def process_motion_data(motion_data_sample, normalize_data, sampling_rate, value_clip, win_len_s, win_len_samples):
    for i in range(0, len(motion_data_sample), win_len_samples):
        w = motion_data_sample.iloc[i: i + win_len_samples]
        if not is_good_quality(w, sampling_rate, win_len_s):
            # Make all data in w NaN
            motion_data_sample.iloc[i: i + win_len_samples] = np.nan

        # Clip data by replacing all values with absolute value > value_clip with value_clip
        motion_data_sample.iloc[i: i + win_len_samples] = motion_data_sample.iloc[
                                                          i: i + win_len_samples
                                                          ].clip(-value_clip, value_clip)
        # Normalize data window-wise
        if normalize_data:
            print("Normalizing data...")
            # Set lower limit for standard deviation to avoid division by zero
            stds = w.std()
            stds[stds < 1e-6] = 1e-6
            motion_data_sample.iloc[i: i + win_len_samples] = (w - w.mean()) / stds
    # Remove all motion data rows with NaNs
    motion_data_sample = motion_data_sample.dropna()

    return motion_data_sample


def process_labelled_data(labels_sample, motion_data_sample, win_len_samples):
    # Drop rows without labels
    motion_data_sample = drop_rows_without_label(motion_data_sample, labels_sample)
    # Make sure all windows have a good label
    for i in range(0, len(motion_data_sample), win_len_samples):
        w = motion_data_sample.iloc[i: i + win_len_samples]
        if not has_good_label(w, labels_sample):
            # Make all data in w NaN
            motion_data_sample.iloc[i: i + win_len_samples] = np.nan

    return motion_data_sample



def is_good_quality(window, sampling_rate, window_sec, window_tol=0.01):
    """Window quality check"""
    window_len = int(window_sec * sampling_rate)
    if window.isna().any().any():
        return False

    if len(window) != window_len:
        return False

    w_start, w_end = window.index[0], window.index[-1]
    w_duration = w_end - w_start
    target_duration = pd.Timedelta(window_sec, "s")
    if np.abs(w_duration - target_duration) > window_tol * target_duration:
        return False

    return True

def has_good_label(window, labels):
    w_start, w_end = window.index[0], window.index[-1]
    # Check if window could have label -1
    label_start = labels.index.get_indexer([w_start], method="nearest")[0]
    label_start = labels.iloc[label_start]["label"]
    label_end = labels.index.get_indexer([w_end], method="nearest")[0]
    label_end = labels.iloc[label_end]["label"]
    if label_start == -1 or label_end == -1:
        print("Dropped window with label -1")
        return False

def load_preprocessed_dataset(
        dataset_name, has_labels, low_pass_filter_freq, model_sampling_rate, win_len_s, normalize_data, cache_path
):
    motion_data_list = []
    labels_list = []
    loaded_from_cache = False
    for file in os.scandir(cache_path):
        if file.name.endswith("_preprocessed.csv"):
            try:
                # Extract parameters from filename
                parameters = file.name.split("_")
                loaded_dataset_name = parameters[0]
                loaded_sampling_rate = int(parameters[3])
                loaded_low_pass_filter_freq = int(parameters[5])
                loaded_id = int(parameters[7])
                win_len_s_pre = int(parameters[9])
                normalize_data_pre = int(parameters[11])
                normalize_data_pre = True if normalize_data_pre == 1 else False
            except:
                loaded_dataset_name = None
                loaded_sampling_rate = None
                loaded_low_pass_filter_freq = None
                win_len_s_pre = None
                normalize_data_pre = None
                loaded_id = None
            if (
                loaded_dataset_name == dataset_name
                and loaded_sampling_rate == model_sampling_rate
                and loaded_low_pass_filter_freq == low_pass_filter_freq
                and win_len_s_pre == win_len_s
                and normalize_data_pre == normalize_data
            ):
                loaded_from_cache = True
                if file.name.split("_")[1] == "motion":
                    data = pd.read_csv(
                        file.path,
                        parse_dates=["timestamp"],
                        dtype={"x": np.float32, "y": np.float32, "z": np.float32},
                    )
                    data["subject"] = loaded_id
                    data.set_index("timestamp", inplace=True)
                    motion_data_list.append(data)

                if file.name.split("_")[1] == "labels":
                    data = pd.read_csv(
                        file.path,
                        parse_dates=["timestamp"],
                        dtype={
                            "label": int,
                        },
                    )
                    data["subject"] = loaded_id
                    data.set_index("timestamp", inplace=True)
                    labels_list.append(data)
    # Sort data according to subject id
    try:
        subject_ids = [motion_data_list[i]["subject"].iloc[0] for i in range(len(motion_data_list))]
        if has_labels:
            make_1_to_1_corresp(motion_data_list, labels_list)
        for i in range(len(motion_data_list)):
            motion_data_list[i] = motion_data_list[i].drop(columns=["subject"])
            if has_labels:
                labels_list[i] = labels_list[i].drop(columns=["subject"])
    except:
        return False, None, None, None
    return loaded_from_cache, motion_data_list, labels_list, subject_ids

def save_preprocessed_dataset(
        dataset_name,
        motion_data_list,
        labels_list,
        subject_ids,
        low_pass_filter_freq,
        model_sampling_rate,
        win_len_s,
        normalize,
        cache_path,
):
    for i in range(len(motion_data_list)):
        id_i = subject_ids[i]

        motion_data_list[i]["timestamp"] = motion_data_list[i].index
        motion_data_list[i].to_csv(
            os.path.join(
                cache_path,
                f"{dataset_name}_motion_samplingrate_{str(model_sampling_rate)}_lowpass_{str(low_pass_filter_freq)}"
                f"_id_{str(id_i)}_wls_{str(win_len_s)}_norm_{str(int(normalize))}_preprocessed.csv",
            ),
            index=False,
        )
        motion_data_list[i] = motion_data_list[i].drop(columns=["timestamp"])

        if labels_list is not None and len(labels_list) > 0:
            labels_list[i]["timestamp"] = labels_list[i].index
            labels_list[i].to_csv(
                os.path.join(
                    cache_path,
                    f"{dataset_name}_labels_samplingrate_{str(model_sampling_rate)}_lowpass_{str(low_pass_filter_freq)}"
                    f"_id_{str(id_i)}_wls_{str(win_len_s)}_norm_{str(int(normalize))}_preprocessed.csv",
                ),
                index=False,
            )
            labels_list[i] = labels_list[i].drop(columns=["timestamp"])

def drop_rows_without_label(data_df: pd.DataFrame, label_df: pd.DataFrame):
    """
    Drops rows from data_df with timestamps outside the range of label_df
    :param data_df: dataframe to drop rows from, with index containing datetime objects
    :param label_df: dataframe containing labels, with index containing datetime objects
    :return: dataframe with rows dropped
    """
    timestamp_col = "timestamp"
    data_df[timestamp_col] = data_df.index
    label_df[timestamp_col] = label_df.index

    data_df = data_df[data_df[timestamp_col] >= label_df[timestamp_col].min()]
    data_df = data_df[data_df[timestamp_col] <= label_df[timestamp_col].max()]
    # Drop timestamp column again
    data_df.drop(columns=[timestamp_col], inplace=True)
    label_df.drop(columns=[timestamp_col], inplace=True)

    return data_df
