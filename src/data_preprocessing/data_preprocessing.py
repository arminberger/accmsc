import os

import actipy
import numpy as np
import pandas as pd

from src.data_download import make_1_to_1_corresp
from src.data_download import unpack_dataset


# Main method for preprocessing dataset
def make_dataset(
        dataset_name,
        target_sampling_rate,
        dataset_sample_rate,
        low_pass_filter_freq,
        try_cached=False,
        cache_path=None,
        normalize_data=False,
        win_len_s=30,
        dataset_path=None,
        label_dict=None,
):

    loaded_from_cache = False
    if try_cached:
        (
            loaded_from_cache,
            motion_data_list,
            labels_list,
            subject_ids,
        ) = load_preprocessed_dataset(
            low_pass_filter_freq,
            target_sampling_rate,
            win_len_s,
            normalize_data,
            cache_path,
        )

    if loaded_from_cache:
        return motion_data_list, labels_list, subject_ids

    # Load raw dataset
    motion_data_list, labels_list, subject_ids = unpack_dataset(
        dataset_name,
        dataset_path,
        label_dict,
    )

    # Process dataset using actipy

    for i in range(len(motion_data_list)):
        motion_data_list[i] = process_data(
            motion_data_list[i],
            labels_list[i],
            dataset_sample_rate,
            target_sampling_rate,
            low_pass_filter_freq,
            win_len_s=win_len_s,
            normalize_data=normalize_data,
        )

    # Save preprocessed dataset to disk
    save_preprocessed_dataset(
        motion_data_list,
        labels_list,
        subject_ids,
        low_pass_filter_freq,
        target_sampling_rate,
        win_len_s,
        normalize_data,
        cache_path,
    )

    return motion_data_list, labels_list, subject_ids


def process_data(
        motion_data_sample,
        labels_sample,
        dataset_sample_rate,
        sampling_rate,
        low_pass_filter_freq,
        win_len_s=30,
        value_clip=2,
        normalize_data=False,
):
    motion_data_sample, _ = actipy.process(
        data=motion_data_sample,
        sample_rate=dataset_sample_rate,
        lowpass_hz=low_pass_filter_freq,
        calibrate_gravity=True,
        detect_nonwear=True,
        resample_hz=sampling_rate,
        verbose=True,
    )
    # Drop rows without labels
    motion_data_sample = drop_rows_without_label(motion_data_sample, labels_sample)
    # Process data window-wise
    win_len_samples = int(win_len_s * sampling_rate)
    for i in range(0, len(motion_data_sample), win_len_samples):
        w = motion_data_sample.iloc[i: i + win_len_samples]
        if not is_good_quality(w, sampling_rate, win_len_s, labels_sample):
            # Make all data in w NaN
            motion_data_sample.iloc[i: i + win_len_samples] = np.nan
        # Clip data by replacing all values with absolute value > value_clip with value_clip
        motion_data_sample.iloc[i: i + win_len_samples] = motion_data_sample.iloc[
                                                          i: i + win_len_samples
                                                          ].clip(-value_clip, value_clip)
        # Normalize data window-wise
        if normalize_data:
            print("Normalizing data...")
            w = motion_data_sample.iloc[i: i + win_len_samples]
            # Set lower limit for standard deviation to avoid division by zero
            stds = w.std()
            stds[stds < 1e-6] = 1e-6
            motion_data_sample.iloc[i: i + win_len_samples] = (w - w.mean()) / stds
    # Remove all motion data rows with NaNs
    motion_data_sample = motion_data_sample.dropna()

    return motion_data_sample


def is_good_quality(window, sampling_rate, window_sec, labels, window_tol=0.01):
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

    # Check if window could have label -1
    label_start = labels.index.get_indexer([w_start], method="nearest")[0]
    label_start = labels.iloc[label_start]["label"]
    label_end = labels.index.get_indexer([w_end], method="nearest")[0]
    label_end = labels.iloc[label_end]["label"]
    if label_start == -1 or label_end == -1:
        print("Dropped window with label -1")
        return False

    return True


def load_preprocessed_dataset(
        low_pass_filter_freq, model_sampling_rate, win_len_s, normalize_data, cache_path
):
    motion_data_list = []
    labels_list = []
    loaded_from_cache = False
    for file in os.scandir(cache_path):
        if file.name.endswith("_preprocessed.csv"):
            try:
                # Extract parameters from filename
                parameters = file.name.split("_")
                loaded_sampling_rate = int(parameters[2])
                loaded_low_pass_filter_freq = int(parameters[4])
                loaded_id = int(parameters[6])
                win_len_s_pre = int(parameters[8])
                normalize_data_pre = int(parameters[10])
                normalize_data_pre = True if normalize_data_pre == 1 else False
            except:
                loaded_sampling_rate = None
                loaded_low_pass_filter_freq = None
                win_len_s_pre = None
                normalize_data_pre = None
                loaded_id = None
            if (
                    loaded_sampling_rate == model_sampling_rate
                    and loaded_low_pass_filter_freq == low_pass_filter_freq
                    and win_len_s_pre == win_len_s
                    and normalize_data_pre == normalize_data
            ):
                loaded_from_cache = True
                if file.name.startswith("motion"):
                    data = pd.read_csv(
                        file.path,
                        parse_dates=["timestamp"],
                        dtype={"x": np.float32, "y": np.float32, "z": np.float32},
                    )
                    data["subject"] = loaded_id
                    data.set_index("timestamp", inplace=True)
                    motion_data_list.append(data)

                if file.name.startswith("labels"):
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
        make_1_to_1_corresp(motion_data_list, labels_list)
        # Extract subject ids
        subject_ids = []
        for i in range(len(motion_data_list)):
            subject_ids.append(motion_data_list[i]["subject"].iloc[0])
            # Drop subject column
            motion_data_list[i] = motion_data_list[i].drop(columns=["subject"])
            labels_list[i] = labels_list[i].drop(columns=["subject"])
    except:
        return False, None, None, None
    return loaded_from_cache, motion_data_list, labels_list, subject_ids


def save_preprocessed_dataset(
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
        labels_list[i]["timestamp"] = labels_list[i].index
        motion_data_list[i].to_csv(
            os.path.join(
                cache_path,
                f"motion_samplingrate_{str(model_sampling_rate)}_lowpass_{str(low_pass_filter_freq)}"
                f"_id_{str(id_i)}_wls_{str(win_len_s)}_norm_{str(int(normalize))}_preprocessed.csv",
            ),
            index=False,
        )
        labels_list[i].to_csv(
            os.path.join(
                cache_path,
                f"labels_samplingrate_{str(model_sampling_rate)}_lowpass_{str(low_pass_filter_freq)}"
                f"_id_{str(id_i)}_wls_{str(win_len_s)}_norm_{str(int(normalize))}_preprocessed.csv",
            ),
            index=False,
        )
        # Drop timestamp column
        motion_data_list[i] = motion_data_list[i].drop(columns=["timestamp"])
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
