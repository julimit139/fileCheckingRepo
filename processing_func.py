import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import signal
import math
from datetime import datetime

def abnormality_pred(modelPath, signals, sampling_rate, timepoints_to_skip, set_to_zero_threshold, window_length, n_windows):

    # load model
    model = tf.keras.models.load_model(modelPath)

    # perform_average_reference
    signals = signals - signals.mean(axis=0)

    # get window for analysis (time x channels)
    n_channels = signals.shape[0]

    all_windows = signals[:, timepoints_to_skip:timepoints_to_skip + n_windows * window_length].T
    all_windows = np.reshape(all_windows, (n_windows, -1, n_channels))

    # Set artifacts to zero and normalize windows
    artifact_mask = ((-set_to_zero_threshold < all_windows) & (all_windows < set_to_zero_threshold)) * 1
    all_windows *= artifact_mask
    all_windows -= all_windows.mean(axis=(1, 2), keepdims=True)
    all_windows /= all_windows.std(axis=(1, 2), keepdims=True)
    all_windows *= artifact_mask

    # run prediction
    y_est = model.predict(all_windows)

    pred_data = np.array([elem for singleList in y_est for elem in singleList])
    pred_data = np.append(pred_data, y_est.mean())

    # print("Predictions per window:", y_est)
    # print("Mean prediction: p(abnormal) =", y_est.mean())

    return pred_data


def get_time_tags(fileName, tag_name, descPath):
    # def get time for description

    tag_time = {'begin': [], 'start': [], 'stop': []}
    is_found = False
    run_first = True

    # read file
    f = open(descPath + fileName, "r", encoding="utf-16")
    for line in f:
        #print(line)
        #print(is_found)

        # find the first line with recording data
        if run_first:
            if 'd1' in line:
                temp = line.split(maxsplit=2)
                tag_time['begin'].append(temp[1])
                run_first = False
            else:
                continue

        # find next event - stop
        if is_found:
            temp = line.split(maxsplit=2)
            tag_time['stop'].append(temp[1])
            #tag_time.append(temp[1])
            is_found = False

        # find event description lines - start
        if tag_name in line:
            temp = line.split(maxsplit=2)
            tag_time['start'].append(temp[1])
            #tag_time.append(temp[1])
            is_found = True

    f.close()
    return tag_time

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, abs(m/sd))


def butter_filter(signals_raw, filter_degree, filter_freq):
    sos = signal.butter(filter_degree, filter_freq, 'hp', fs=1000, output='sos')
    filtered = signal.sosfilt(sos, signals_raw)
    # ax2.plot(t, filtered)
    # ax2.set_title('After 15 Hz high-pass filter')
    # ax2.axis([0, 1, -2, 2])
    # ax2.set_xlabel('Time [seconds]')
    # plt.tight_layout()
    # plt.show()
    return filtered


def get_tag_frames(name, signals, sampling_rate, epoch_size, overlap, tag_time):

    epochs = []

    # windows settings
    frame_numbers = len(tag_time['start'])
    print(frame_numbers)

    temp_results = []
    tag_begin = datetime.strptime(tag_time['begin'][0], '%H:%M:%S')

    for f_start, f_stop in zip(tag_time['start'], tag_time['stop']):

        print(f_start, f_stop)

        # time to sample numbers conversion
        # start - begin > s * sampling rate
        tag_start = datetime.strptime(f_start, '%H:%M:%S')
        start_sample = ((tag_start.hour - tag_begin.hour) * 60 * 60 + (
                    tag_start.minute - tag_begin.minute) * 60 + tag_start.second - tag_begin.second) * sampling_rate
        # stop - begin > s * sampling rate
        tag_stop = datetime.strptime(f_stop, '%H:%M:%S')
        stop_sample = ((tag_stop.hour - tag_begin.hour) * 60 * 60 + (tag_stop.minute - tag_begin.minute) * 60 + (
                    tag_stop.second - tag_begin.second)) * sampling_rate
        # select samples from start to stop
        data_to_analyze = signals[:, start_sample:stop_sample]

        # verify length
        if data_to_analyze.shape[1] > epoch_size * sampling_rate:
            print(data_to_analyze.shape[1])

            # get window for analysis (time x channels)
            n_channels = data_to_analyze.shape[0]
            # create time stamps for epochs
            t_stamp = np.arange(0, data_to_analyze.shape[1] + 1, round(overlap) * sampling_rate)
            epoch_start = t_stamp[:-2]
            epoch_stop = t_stamp[2:]

            # for each epoch
            for e_start, e_stop in zip(epoch_start, epoch_stop):
                print(e_start, e_stop)

                eeg_raw = data_to_analyze[:, e_start:e_stop]

                epochs.append(eeg_raw)

        # if not enough samples in frame, skip to next
        else:
            print('too short frame')
            continue

    #patients_data = dict()  # contain json with eeg frames
    patients_data = np.array(epochs).tolist()

    return patients_data


def get_peak_results(name, data, epoch_size, sampling_rate, channelsNames, brain_waves, butter_degree):

    result_col_names = ['frame_number', 'channel', "gamma", "beta", "alpha", "theta", "delta"]
    peak_results = pd.DataFrame(columns=result_col_names, dtype=float)
    #time = np.arange(epoch_size * sampling_rate) / sampling_rate

    # for each frame in patients data
    for frame, frame_nb in zip(data, np.arange(len(data))):

        eeg_raw = {key: value for key, value in zip(channelsNames, frame)}

        # for each channel in frame
        for channel_eeg, channel_name in zip(eeg_raw, channelsNames):
            # print(channel_name)

            channel_data = np.array(eeg_raw[channel_eeg])

            # compute frequencies vector until half the sampling rate
            Nyquist = sampling_rate / 2
            Nsamples = int(math.floor(channel_data.size / 2))
            hz = np.linspace(0, Nyquist, num=Nsamples)

            # df for all patients
            # peak_results = peak_results.append({'patient_ID': name, 'frame_number': frame_nb, 'channel_name': channel_name}, ignore_index=True)
            # df for one patient
            peak_results = peak_results.append({'frame_number': frame_nb, 'channel': channel_name}, ignore_index=True)

            # for each wave filters
            for wave in brain_waves:
                # print(wave)

                # highpass filter
                sos = signal.butter(butter_degree, brain_waves[wave]['start'], 'highpass', fs=1000, output='sos')
                filtered_low = signal.sosfilt(sos, channel_data)

                # lowpass filter
                sos = signal.butter(butter_degree, brain_waves[wave]['stop'], 'lowpass', fs=1000, output='sos')
                filtered = signal.sosfilt(sos, filtered_low)

                # Fourier transform
                FourierCoeff = np.fft.fft(filtered) / filtered.size
                DC = [np.abs(FourierCoeff[0])]
                amp = np.concatenate((DC, 2 * np.abs(FourierCoeff[1:])))
                amp = amp[:len(hz)]

                # find frequency peak
                wave_hz = hz[(hz > brain_waves[wave]['start']) & (hz < brain_waves[wave]['stop'])]
                wave_amp = amp[(hz > brain_waves[wave]['start']) & (hz < brain_waves[wave]['stop'])]
                max_peak = wave_hz[np.argmax(wave_amp)]
                # print(max_peak)
                peak_results.at[peak_results.index[-1], wave] = max_peak

    return peak_results