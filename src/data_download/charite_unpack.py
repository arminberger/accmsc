import pandas as pd
from pyedflib import highlevel
import numpy as np
import glob as glob
import os
import io
import heartpy as hp
import zipfile
from scipy.signal import resample
from datetime import timedelta

def unpack_charite(path):
    # Here we load the EDF files and compute heart rates
    motion_datas = []
    bpm_datas = []
    patient_codes = []
    start_recordings = []
    end_recordings = []
    start_dates = []
    for f in glob.glob(path + '/*.edf'):
        motion_data, bpm_data, patient_code, start_recording, end_recording, start_date = get_edffile(f)
        motion_datas.append(motion_data)
        bpm_datas.append(bpm_data)
        patient_codes.append(patient_code)
        start_recordings.append(start_recording)
        end_recordings.append(end_recording)
        start_dates.append(start_date)
    # Here we work on getting the PSG files into normal format
    slp_path = os.path.join(path, "PSG")
    with zipfile.ZipFile(os.path.join(path, "PSG.zip"), "r") as zip_ref:
        zip_ref.extractall(slp_path)
    sleep = []
    for i, pc in enumerate(patient_codes):
        path = os.path.join(slp_path, pc + '.slp')
        if not os.path.exists(path):
            sleep.append(None)
            continue
        ss = pd.read_table(path)
        header = ss.columns.values[0]
        header_io = io.StringIO(header)
        header = pd.read_table(header_io, header=None, sep='\s+',
                               names=['psg_start', 'error_code', 'offset', 'slope', 'lights_off', 'lights_on'])
        footer = ss.iloc[-1].to_numpy()[0]
        header['psg_stop'] = footer
        header_io.close()
        # Drop last element
        ss = ss.iloc[:-1]
        ss.columns = ['sleep_phase']
        ss['subject'] = pc
        # Check that times are not too far off
        lights_off = start_recordings[i]
        lights_on = end_recordings[i]
        start_date = start_dates[i]
        # Only keep date, remove all hours etc.
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        # Get only hour and smaller units
        lights_on_psg = header['lights_on'].to_numpy()[0]
        lights_off_psg = header['lights_off'].to_numpy()[0]
        # Convert to timedeltas
        lights_on_psg = timedelta(hours=lights_on_psg) + start_date
        lights_off_psg = timedelta(hours=lights_off_psg) + start_date
        # Check difference
        diff_on = (lights_on - lights_on_psg).total_seconds()
        diff_off = (lights_off - lights_off_psg).total_seconds()
        print(diff_on, diff_off)
        if not (abs(diff_on) < 1 and abs(diff_off) < 1):
            sleep.append(None)
            continue
        # Add timestamp column to the ss
        psg_start = timedelta(hours=header['psg_start'].to_numpy()[0]) + start_date
        psg_stop = timedelta(hours=header['psg_stop'].to_numpy()[0]) + start_date
        idx = pd.date_range(start=psg_start, end=psg_stop, periods=ss.size)
        # Check if periods are roughly 30s long
        if (idx[1] - idx[0]).total_seconds() < 29 or (idx[1] - idx[0]).total_seconds() > 31:
            print(f'{pc} has sleep staging periods that are not 30s')
            sleep.append(None)
            continue
        ss['timestamp'] = idx
        if header['error_code'].to_numpy()[0] == 0:
            sleep.append(ss)
        else:
            sleep.append(None)
        ### Now we have to update the timestamp column with the offset read
        print(header)
        psg_offset = timedelta(hours=header['offset'].to_numpy()[0])
        print(psg_offset)
        motion_datas[i]['timestamp'] = motion_datas[i]['timestamp'] + psg_offset
        bpm_datas[i]['timestamp'] = bpm_datas[i]['timestamp'] + psg_offset

    # Finally, we have to remove all the data for which sleep is none
    for i, e in enumerate(sleep):
        if e is None:
            motion_datas[i] = None
            bpm_datas[i] = None
            patient_codes[i] = None

    # Now we create new lists that dont contain the None values
    motion_datas = [x for x in motion_datas if x is not None]
    bpm_datas = [x for x in bpm_datas if x is not None]
    sleep = [x for x in sleep if x is not None]
    patient_codes = [x for x in patient_codes if x is not None]
    # Finally we make all  the 'timestamp' columns the index and remove them from the columns
    for i in range(len(motion_datas)):
        motion_datas[i] = motion_datas[i].set_index('timestamp')
        bpm_datas[i] = bpm_datas[i].set_index('timestamp')
        sleep[i] = sleep[i].set_index('timestamp')

    return motion_datas, bpm_datas, sleep, patient_codes

def get_channels_from_somno_edf(somno):
    for i in range(len(somno[0])):
        if somno[1][i]['label'] == 'ECG':
            ecg = somno[0][i]
            sr_ecg = int(somno[1][i]['sample_frequency'])
        elif somno[1][i]['label'] == 'X':
            x = somno[0][i]
            sr_x = int(somno[1][i]['sample_frequency'])
        elif somno[1][i]['label'] == 'Y':
            y = somno[0][i]
            sr_y = int(somno[1][i]['sample_frequency'])
        elif somno[1][i]['label'] == 'Z':
            z = somno[0][i]
            sr_z = int(somno[1][i]['sample_frequency'])
    return np.array(ecg), np.array(x), np.array(y), np.array(z), sr_ecg, sr_x, sr_y, sr_z

def get_bpm_from_ecg(ecg, sr_ecg, segment_width):
    # My own heartpy calculation
    hp_upsampling_factor = 4
    fs=sr_ecg

    resampled_signal = resample(ecg, len(ecg)*hp_upsampling_factor)
    filtered = hp.remove_baseline_wander(hp.scale_data(resampled_signal), fs*hp_upsampling_factor)
    ecg_proc = hp.scale_data(filtered)

    wd, m = hp.process_segmentwise(ecg_proc, sample_rate=fs*hp_upsampling_factor, segment_width=segment_width, segment_overlap=0, mode='full', replace_outliers=True)
    samples_per_segment = hp_upsampling_factor*fs*segment_width
    si = list(wd['segment_indices'])
    indices = [int(x/samples_per_segment) for (x,y) in si]
    # si are the indices that have been successfully computed by hp
    # Now we compute how many indices there should be in total
    num_windows = int(len(ecg_proc) / samples_per_segment)
    # Create an array with that length
    bpm_array = np.zeros(num_windows)
    bpm_array[:] = np.nan
    for idx, k in enumerate(indices):
        wbpm = m['bpm'][idx]
        bpm_array[k] = wbpm
    # Now we want to interpolate bpm_array
    bpm_array = pd.DataFrame(data=bpm_array).ffill().to_numpy().flatten()
    return bpm_array


def get_edffile(edf_file):
    segment_width_bpm = 20
    raw = highlevel.read_edf(edf_file)
    patient_code = raw[2]['patientcode']
    # Getting to EDF time
    startdate = raw[2]['startdate']
    start_offset = raw[2]['annotations'][0]
    end_offset = raw[2]['annotations'][1]
    # Verify we have the record we want
    if start_offset[2] == 'Licht aus' and end_offset[2] == 'Licht an' and start_offset[1] == -1:
        print('Offsets seem fine')
    else:
        raise AssertionError
    start_offset = timedelta(seconds=start_offset[0])
    end_offset = timedelta(seconds=end_offset[0])
    end_recording = startdate + end_offset
    start_recording = startdate + start_offset

    #
    ecg, x, y, z, sr_ecg, sr_x, sr_y, sr_z = get_channels_from_somno_edf(raw)
    bpm_array = get_bpm_from_ecg(ecg, sr_ecg, segment_width_bpm)
    # Now we want to put times to all the arrays

    freq = int(1/sr_x*(10**9))
    idx = pd.date_range(start=startdate, periods=len(x), freq=f'{freq}ns')
    assert len(x) == len(y) == len(z) == len(idx)
    acc_df = pd.DataFrame(data={'x': x, 'y':y, 'z':z, 'timestamp': idx})
    acc_df['subject'] = patient_code
    idx = pd.date_range(start=startdate, periods=len(bpm_array), freq=f'{segment_width_bpm}s')
    bpm_df = pd.DataFrame(data={'timestamp':idx, 'bpm':bpm_array})
    bpm_df['subject'] = patient_code
    return acc_df, bpm_df, patient_code, start_recording, end_recording, startdate