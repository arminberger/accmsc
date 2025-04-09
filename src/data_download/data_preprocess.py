import numpy as np
import os
import pandas as pd
from scipy import signal

import actipy
import multiprocessing
import re


# Main method for preprocessing dataset
def make_dataset(
    dataset_name,
    sampling_rate,
    low_pass_filter_freq,
    try_preprocessed=False,
    normalize_data=False,
    win_len_s=30,
):
    """
    Loads raw dataset, processes it into a form ready for a model and saves preprocessed dataset to disk at the path
    by calling project_constants.GET_PREPROCESSED_DATASET_PATH(dataset_name). If preprocessing has already been done
    and try_preprocessed is True, loads the preprocessed dataset from disk instead of preprocessing it again.
    Args:
        dataset_name:
        sampling_rate:
        low_pass_filter_freq:
        try_preprocessed:

    Returns:

    """
    loaded_from_cache = False  # TODO: Check if this is correct
    if try_preprocessed:
        (
            loaded_from_cache,
            motion_data_list,
            labels_list,
            subject_ids,
        ) = load_preprocessed_dataset(
            dataset_name,
            low_pass_filter_freq,
            sampling_rate,
            win_len_s,
            normalize_data,
        )

    if not loaded_from_cache:
        # Load raw dataset
        motion_data_list, labels_list, subject_ids = load_raw_dataset(
            dataset_name,
        )

        # Process dataset using actipy

        for i in range(len(motion_data_list)):
            motion_data_list[i] = _process_data(
                motion_data_list[i],
                labels_list[i],
                subject_ids[i],
                dataset_name,
                sampling_rate,
                low_pass_filter_freq,
                win_len_s=win_len_s,
                normalize_data=normalize_data,
            )

        # Save preprocessed dataset to disk
        save_preprocessed_dataset(
            motion_data_list,
            labels_list,
            subject_ids,
            dataset_name,
            low_pass_filter_freq,
            sampling_rate,
            win_len_s,
            normalize_data,
        )

    return motion_data_list, labels_list, subject_ids


def _process_data(
    motion_data_sample,
    labels_sample,
    subject,
    dataset_name,
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
        w = motion_data_sample.iloc[i : i + win_len_samples]
        if not is_good_quality(w, sampling_rate, win_len_s, labels_sample):
            # Make all data in w NaN
            motion_data_sample.iloc[i : i + win_len_samples] = np.nan
        # Clip data by replacing all values with absolute value > value_clip with value_clip
        motion_data_sample.iloc[i : i + win_len_samples] = motion_data_sample.iloc[
            i : i + win_len_samples
        ].clip(-value_clip, value_clip)
        # Normalize data window-wise
        if normalize_data:
            print("Normalizing data...")
            w = motion_data_sample.iloc[i : i + win_len_samples]
            # Set lower limit for standard deviation to avoid division by zero
            stds = w.std()
            stds[stds < 1e-6] = 1e-6
            motion_data_sample.iloc[i : i + win_len_samples] = (w - w.mean()) / stds
    # Remove all motion data rows with NaNs
    motion_data_sample = motion_data_sample.dropna()

    return motion_data_sample


def is_good_quality(window, sampling_rate, WINDOW_SEC, labels, WINDOW_TOL=0.01):
    """Window quality check"""
    WINDOW_LEN = int(WINDOW_SEC * sampling_rate)
    if window.isna().any().any():
        return False

    if len(window) != WINDOW_LEN:
        return False

    w_start, w_end = window.index[0], window.index[-1]
    w_duration = w_end - w_start
    target_duration = pd.Timedelta(WINDOW_SEC, "s")
    if np.abs(w_duration - target_duration) > WINDOW_TOL * target_duration:
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
    dataset_name, low_pass_filter_freq, model_sampling_rate, win_len_s, normalize_data
):
    motion_data_list = []
    labels_list = []
    loaded_from_cache = False
    for file in os.scandir(
        project_constants.GET_PREPROCESSED_DATASET_PATH(dataset_name)
    ):
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
    dataset_name,
    low_pass_filter_freq,
    model_sampling_rate,
    win_len_s,
    normalize,
):
    for i in range(len(motion_data_list)):
        id_i = subject_ids[i]
        motion_data_list[i]["timestamp"] = motion_data_list[i].index
        labels_list[i]["timestamp"] = labels_list[i].index
        motion_data_list[i].to_csv(
            os.path.join(
                project_constants.GET_PREPROCESSED_DATASET_PATH(dataset_name),
                f"motion_samplingrate_{str(model_sampling_rate)}_lowpass_{str(low_pass_filter_freq)}"
                f"_id_{str(id_i)}_wls_{str(win_len_s)}_norm_{str(int(normalize))}_preprocessed.csv",
            ),
            index=False,
        )
        labels_list[i].to_csv(
            os.path.join(
                project_constants.GET_PREPROCESSED_DATASET_PATH(dataset_name),
                f"labels_samplingrate_{str(model_sampling_rate)}_lowpass_{str(low_pass_filter_freq)}"
                f"_id_{str(id_i)}_wls_{str(win_len_s)}_norm_{str(int(normalize))}_preprocessed.csv",
            ),
            index=False,
        )
        # Drop timestamp column
        motion_data_list[i] = motion_data_list[i].drop(columns=["timestamp"])
        labels_list[i] = labels_list[i].drop(columns=["timestamp"])


def load_raw_dataset(
    dataset_name,
    dataset_path,
    label_dict,
):
    """
    Reads in raw dataset
    and returns two lists of dataframes (motion data, labels), one dataframe per subject.
    Ensures that there is a 1:1 correspondence between the motion data and labels dataframes
    and that lists are sorted by subject ID.
    :param dataset_name: Name of the dataset, must be one of the constants in dataset_constants
    :return: lists of dataframes (motion data, labels) with index containing datetime objects,
    one dataframe of format ['x', 'y', 'z', 'subject'],
    ['label', 'subject'], respectively, per subject and the name of the timestamp column
    """
    timestamp_col = "timestamp"
    motion_data = []
    labels_data = []

    if dataset_name == "applewatch":
        """
        Apple Watch dataset (wget -r -N -c -np https://physionet.org/files/sleep-accel/1.0.0/), path should
        be the path wget was run in.
        """
        motion_path = os.path.join(dataset_path, "motion")
        labels_path = os.path.join(dataset_path, "labels")
        print(motion_path)
        motion_data = []
        for file in os.scandir(motion_path):
            # use read_csv with whitespace delimiter
            # Only r
            if file.name.endswith("acceleration.txt"):
                subject_data = pd.read_csv(
                    file.path, delim_whitespace=True, header=None
                )
                # get subject name from filename
                subject_name = file.name.split("_")[0]
                # add subject name as column
                subject_data["subject"] = int(subject_name)
                # rename columns
                subject_data.columns = [timestamp_col, "x", "y", "z", "subject"]
                # parse timestamp column as datetime
                subject_data[timestamp_col] = pd.to_datetime(
                    subject_data[timestamp_col], unit="s"
                )
                subject_data.index = subject_data[timestamp_col]
                subject_data.drop(columns=[timestamp_col], inplace=True)
                # add to list
                motion_data.append(subject_data)

        # create list to store labels dataframes
        labels_data = []
        for file in os.scandir(labels_path):
            print(file.path)
            if file.name.endswith("labeled_sleep.txt"):
                # get subject name from filename
                subject_data = pd.read_csv(file.path, delim_whitespace=True)
                # get subject name from filename
                subject_name = file.name.split("_")[0]
                # add subject name as column
                subject_data["subject"] = int(subject_name)
                # rename columns
                subject_data.columns = [timestamp_col, "label", "subject"]
                # parse timestamp column as datetime
                subject_data[timestamp_col] = pd.to_datetime(
                    subject_data[timestamp_col], unit="s"
                )
                subject_data.set_index(timestamp_col, inplace=True)
                # add to list
                labels_data.append(subject_data)
            # Sort all data according to datetime index
        for i in range(len(motion_data)):
            motion_data[i].sort_index(inplace=True)
        for i in range(len(labels_data)):
            labels_data[i].sort_index(inplace=True)
        # We drop the samples with ids 5383425 and 8258170 due to too low sampling rate
        motion_data = [
            data
            for data in motion_data
            if data["subject"].iloc[0] not in [5383425, 8258170]
        ]
        labels_data = [
            data
            for data in labels_data
            if data["subject"].iloc[0] not in [5383425, 8258170]
        ]

    elif dataset_name == "geneactiv":
        motion_path = os.path.join(dataset_path, "Axivity files")
        labels_path = os.path.join(dataset_path, "Sleep Staging data")

        # Load labels data
        labels_data = []
        labels_ids = []

        for file in os.scandir(labels_path):
            if file.name.endswith(".txt"):
                # Extract ID from filename
                subject_id = file.name.split("_")[0]
                subject_id = subject_id.replace("TWIN", "")
                subject_id = int(subject_id)

                if subject_id == 1001:
                    # This probably was some test study. Different format is used and no axivity data is available, so it won't be handled
                    continue
                labels_ids.append(subject_id)
                # Read in txt file line by line

                with open(file.path, "r") as f:
                    start_timestamp = None
                    label_df = pd.DataFrame(
                        columns=["timestamp", "Label (str)", "label", "subject"]
                    )
                    reached_data = False

                    lines = f.read().splitlines()
                    for line in lines:
                        if not reached_data:
                            elems = line.split()
                            if {
                                "Epoch",
                                "Event",
                                "Start Time",
                                "Duration",
                                "Upper",
                                "Lower",
                                "Difference",
                            }.issubset(set(line.split("\t"))):
                                # Reached data
                                reached_data = True
                            elif (
                                len(elems) >= 2
                                and elems[0] == "Study"
                                and elems[1] == "Date:"
                            ):
                                # Extract start date and time
                                start_date = elems[2]
                                start_time = elems[3]
                                ampm = elems[4]
                                start_timestamp = pd.to_datetime(
                                    start_date + " " + start_time + " " + ampm,
                                    format="%m/%d/%Y %I:%M:%S %p",
                                )
                        else:
                            elems = line.split("\t")
                            if elems[1] in label_dict.keys():
                                # Extract label and timestamp
                                label_str = elems[1]
                                label_int = label_dict[label_str]
                                date = start_timestamp.date()
                                if (
                                    elems[2].split()[1] == "AM"
                                    and start_timestamp.hour > 18
                                ):
                                    # If the label is from the next day, add one day to the timestamp
                                    date = date + pd.Timedelta(days=1)
                                # convert date to string
                                date = date.strftime("%m/%d/%Y")
                                timestamp = pd.to_datetime(
                                    date + " " + elems[2],
                                    format="%m/%d/%Y %I:%M:%S %p",
                                )
                                concat_df = pd.DataFrame(
                                    {
                                        "timestamp": timestamp,
                                        "Label (str)": label_str,
                                        "label": label_int,
                                        "subject": subject_id,
                                    },
                                    index=[0],
                                )
                                label_df = pd.concat(
                                    [label_df, concat_df], ignore_index=True
                                )
                    label_df.drop(columns=["Label (str)"], inplace=True)
                    label_df.set_index("timestamp", inplace=True)
                    labels_data.append(label_df)

        # Load motion data
        motion_data = []
        motion_info = []
        motion_ids = []

        for file in os.scandir(motion_path):
            print(file)
            if file.name.endswith(".cwa"):
                # Extract ID from filename
                subject_id = file.name.split("_")[1]
                subject_id = subject_id.split(".")[0]
                subject_id = int(subject_id)
                if subject_id not in labels_ids:
                    # Skip motion data if there are no labels for it
                    continue
                motion_ids.append(subject_id)
                # Just load the data, don't do any preprocessing
                data, info = actipy.read_device(
                    input_file=os.path.join(motion_path, file),
                    lowpass_hz=None,
                    calibrate_gravity=False,
                    detect_nonwear=False,
                    resample_hz=None,
                )
                data = data[["x", "y", "z"]]
                data["subject"] = subject_id
                motion_data.append(data)

        # Drop labels for which there is no motion data
        labels_data_new = []
        for i in range(len(labels_data)):
            if labels_data[i]["subject"].iloc[0] in motion_ids:
                labels_data_new.append(labels_data[i])
        labels_data = labels_data_new

    elif dataset_name == "newcastle":
        motion_path = os.path.join(dataset_path, "acc")
        labels_path = os.path.join(dataset_path, "psg")

        # Load labels data
        labels_data = []
        labels_ids = []

        for file in os.scandir(labels_path):
            if file.name.endswith(".txt"):
                # Extract ID from filename
                subject_id = file.name.split("_")[0]
                subject_id = subject_id.replace("mecsleep", "")
                subject_id = int(subject_id)

                if subject_id == 29:
                    # Here the labels time does not match the motion time, thus we drop the frame
                    continue

                labels_ids.append(subject_id)
                # Read in txt file line by line

                with open(file.path, "r") as f:
                    start_timestamp = None

                    file = f.read()
                    recording_date_match = re.search(
                        r"Recording Date:\s*(\d{2}/\d{2}/\d{4})", file
                    )
                    recording_date = (
                        recording_date_match.group(1) if recording_date_match else None
                    )
                    day, month, year = recording_date.split("/")

                    # Extracting the table content
                    table_content = file.split("\n\n")[-1]
                    table_rows = table_content.split("\n")

                    # Check if table rows are empty and issue warning
                    if len(table_rows) <= 1:
                        print(f"Warning: No labels found for subject {subject_id}")
                        continue

                    """columns = ['Sleep Stage', 'Position', 'timestamp', 'Event', 'Duration[s]', 'Sleep Stage (int)']"""
                    columns = table_rows[0].split("\t")
                    # Get index of time and sleep stage columns
                    time_index = columns.index("Time [hh:mm:ss]")
                    sleep_stage_index = columns.index("Sleep Stage")
                    n_past_midnight = 0
                    last_hour = None
                    rows = []
                    for row in table_rows[1:]:
                        if row == "":
                            break

                        row_split = row.split("\t")
                        time = row_split[time_index]
                        hours, minutes, seconds = time.split(":")

                        # Check if we went past midnight
                        if last_hour is not None and int(hours) < last_hour:
                            n_past_midnight += 1
                        last_hour = int(hours)
                        timestamp = pd.Timestamp(
                            year=int(year),
                            month=int(month),
                            day=int(day) + n_past_midnight,
                            hour=int(hours),
                            minute=int(minutes),
                            second=int(seconds),
                        )

                        sleep_stage = row_split[sleep_stage_index]

                        row = [timestamp, sleep_stage, label_dict[sleep_stage]]
                        row_df = pd.DataFrame(
                            [row],
                            columns=["timestamp", "Sleep Stage", "Sleep Stage (int)"],
                        )
                        rows.append(row_df)
                    label_df = pd.concat(rows, ignore_index=True)
                    label_df = label_df[["timestamp", "Sleep Stage (int)"]]
                    label_df.columns = ["timestamp", "label"]
                    label_df["subject"] = subject_id
                    label_df.set_index("timestamp", inplace=True)
                    labels_data.append(label_df)

        # Load motion data

        motion_data_left = []
        motion_ids_left = []
        motion_data_right = []
        motion_ids_right = []

        for file in os.scandir(motion_path):
            if file.name.endswith(".bin"):
                id = file.name.split("_")[0]
                id = int(id.replace("MECSLEEP", ""))
                wrist = file.name.split("_")[1].split(" ")[0]

                if id not in labels_ids:
                    # Skip motion data if there are no labels for it
                    continue

                # Just load the data, don't do any preprocessing
                data, info = actipy.read_device(
                    input_file=os.path.join(motion_path, file),
                    lowpass_hz=None,
                    calibrate_gravity=False,
                    detect_nonwear=False,
                    resample_hz=None,
                )

                data = data[["x", "y", "z"]]
                data["subject"] = id

                if wrist == "left":
                    motion_ids_left.append(id)
                    motion_data_left.append(data)
                elif wrist == "right":
                    motion_ids_right.append(id)
                    motion_data_right.append(data)
                else:
                    raise ValueError(f"Unknown wrist {wrist}")

                motion_data = motion_data_left

    elif dataset_name == "this is the ichi dataset which we dont use for now":
        datapath = dataset_path

        def rle2raw(dta):
            # extract RLE encoded acceleration data
            print(dta["d"])
            rle = np.array(
                (
                    dta["d"],
                    dta["t"],
                    dta["x"],
                    dta["y"],
                    dta["z"],
                )
            ).transpose()

            labels = np.array(
                (
                    dta["d"],
                    dta["t"],
                    dta["l"],
                )
            ).transpose()

            # convert from RLE to pure RAW data
            # v[0] is the number of repetitions
            # v[1:] is the data
            # v[0] has to be int!!!
            return np.vstack([int(v[0]) * [v[1:]] for v in rle]), np.vstack(
                [int(v[0]) * [v[1:]] for v in labels]
            )

        userlist = [
            "002",
            "003",
            "005",
            "007",
            # "08a",
            # "08b",
            # "09a",
            # "09b",
            "10a",
            "011",
            "013",
            "014",
            # "15a",
            # "15b",
            "016",
            "017",
            "018",
            "019",
            "020",
            "021",
            "022",
            "023",
            "025",
            "026",
            "027",
            "028",
            "029",
            "030",
            "031",
            "032",
            "033",
            "034",
            "035",
            "036",
            "037",
            "038",
            "040",
            "042",
            "043",
            "044",
            "045",
            "047",
            "048",
            "049",
            # "051_hl",
            # "051_hr",
        ]

        def import_ichi14_participant(p):
            dta = np.load(os.path.join(datapath, f"p{p}.npy"))

            data, labels = rle2raw(dta)

            print(
                f"Part {p} has {np.round(data.shape[0] / 100 / 3600, 2)}h of recording"
            )

            return data, labels

        data_dfs = []
        label_dfs = []
        for p in userlist:
            data_np, labels_np = import_ichi14_participant(p)
            # convert data to pandas and add timestamps with 100Hz
            print("Converting to pandas")
            start_time = pd.Timestamp(data_np[0, 0], unit="s")
            end_time = pd.Timestamp(data_np[-1, 0], unit="s")
            data_index = pd.date_range(start=start_time, end=end_time, freq="10ms")
            data_df = pd.DataFrame(
                data_np[:, 1:], columns=["x", "y", "z"], index=data_index
            )
            data_df["subject"] = p
            data_dfs.append(data_df)
            start_time = pd.Timestamp(labels_np[0, 0], unit="s")
            end_time = pd.Timestamp(labels_np[-1, 0], unit="s")
            label_index = pd.date_range(start=start_time, end=end_time, freq="10ms")
            label_df = pd.DataFrame(
                labels_np[:, 1:], columns=["label"], index=label_index
            )
            label_df["subject"] = p
            label_dfs.append(label_df)

    make_1_to_1_corresp(labels_data, motion_data)
    # Remove subject column
    subject_ids = []
    for i in range(len(motion_data)):
        subject_ids.append(motion_data[i]["subject"].iloc[0])
        motion_data[i].drop(columns=["subject"], inplace=True)
        labels_data[i].drop(columns=["subject"], inplace=True)
    return motion_data, labels_data, subject_ids


def make_1_to_1_corresp(labels_data, motion_data):
    """
    Ensures that the labels and motion data are in the same order and that there is a 1 to 1 correspondence between
    them.
    Args:
        labels_data:
        motion_data:

    Returns:

    """
    # Sort motion data and labels by ID
    motion_data.sort(key=lambda x: x["subject"].iloc[0])
    labels_data.sort(key=lambda x: x["subject"].iloc[0])
    # Ensure that motion data and labels have the same length and that the IDs match
    assert len(motion_data) == len(labels_data)
    for i in range(len(motion_data)):
        assert motion_data[i]["subject"].iloc[0] == labels_data[i]["subject"].iloc[0]
        print(motion_data[i]["subject"].iloc[0])



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



