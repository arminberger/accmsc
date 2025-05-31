import os
import re
import actipy
import pandas as pd
from tqdm import tqdm
import numpy as np

def unpack_unlabelled_dataset(dataset_name, dataset_path):
    if dataset_name=='capture24':
        motion_data, subject_ids = unpack_capture24(dataset_path)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return motion_data, subject_ids

def unpack_capture24(dataset_path):
    motion_data = []
    subject_ids = []
    for file in tqdm(os.scandir(dataset_path)):
        filepath = file.path
        print(filepath)
        filename = os.path.basename(filepath)
        # Load subject
        if filename.endswith(".csv.gz"):
            print(filename)
            # Extract ID from filename
            subject_id = filename.split(".")[0]
            subject_id = subject_id.replace("P", "")
            subject_id = int(subject_id)
            print("Loading csv of subject ", subject_id)
            data_pd = pd.read_csv(
                filepath,
                parse_dates=["time"],
                index_col='time',
                dtype={
                    "x": np.float32,
                    "y": np.float32,
                    "z": np.float32,
                    "annotation": str,
                },
            )
            print("Cleaning up format of subject ", subject_id)
            data_pd.sort_index(inplace=True, ascending=True, kind="mergesort")
            data = data_pd
            data = data.drop(columns=["annotation"])
            data["subject"] = subject_id
            print("Done loading subject ", subject_id)
            motion_data.append(data)
            subject_ids.append(subject_id)
    motion_data.sort(key=lambda x: x["subject"].iloc[0])
    # Drop subject column
    for i in range(len(motion_data)):
        motion_data[i].drop(columns=["subject"], inplace=True)

    subject_ids.sort()
    return motion_data, subject_ids

def unpack_labelled_dataset(
        dataset_name,
        dataset_path,
        label_dict=None,
):
    """
    Reads in raw dataset
    and returns two lists of dataframes (motion data, labels), one dataframe per subject.
    Ensures that there is a 1:1 correspondence between the motion data and labels dataframes
    and that lists are sorted by subject ID.
    :param dataset_name: Name of the dataset, must be one of the constants in dataset_constants
    :param dataset_path: Path to the dataset
    :param label_dict: Dictionary mapping original sleep stage labels to integers according to our convention
    
    :return: lists of dataframes (motion data, labels) with index containing datetime objects,
    one dataframe of format ['x', 'y', 'z', 'subject'],
    ['label', 'subject'], respectively, per subject
    """


    if dataset_name == "applewatch":
        labels_data, motion_data = unpack_applewatch(dataset_path)

    elif dataset_name == "geneactiv":
        labels_data, motion_data = unpack_geneactiv(dataset_path, label_dict)

    elif dataset_name == "newcastle":
        labels_data, motion_data = unpack_newcastle(dataset_path, label_dict)

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    make_1_to_1_corresp(labels_data, motion_data)
    # Remove subject column
    subject_ids = []
    for i in range(len(motion_data)):
        subject_ids.append(motion_data[i]["subject"].iloc[0])
        motion_data[i].drop(columns=["subject"], inplace=True)
        labels_data[i].drop(columns=["subject"], inplace=True)
    return motion_data, labels_data, subject_ids


def unpack_newcastle(dataset_path, label_dict):
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
    return labels_data, motion_data


def unpack_geneactiv(dataset_path, label_dict):
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
    return labels_data, motion_data


def unpack_applewatch(dataset_path):
    """
        Apple Watch dataset (wget -r -N -c -np https://physionet.org/files/sleep-accel/1.0.0/), path should
        be the path wget was run in.
        """
    timestamp_col = "timestamp"
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
    return labels_data, motion_data


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
