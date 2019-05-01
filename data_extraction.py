import os
import numpy as np
from scipy import fftpack
from scipy import stats
import random
from matplotlib import pyplot

# the path should lead to the same folder structure as 
# A2_Data as it was uploaded to piazza for the assignment or
# as it is drawn in readme.txt
def step_2_feature_extraction(path):
    activity_list = create_activity_list(
        get_folder_data('./A2_Data/pocket/dshen/'))
    pocket_entries = os.listdir(path)
    data_x = []
    data_y = []
    # window length and shift in seconds
    window_length = 10
    shift_length = 2

    for entry in pocket_entries:
        print(entry)
        folder_data = get_folder_data(path + entry + '/' )
        
        # figure out the number of rows per window/shift
        window_size, window_shift = calculate_window_sizes(
            folder_data, 'accel', window_length, shift_length
        )
        # arrange the data in the folder
        accel_data_x, accel_data_y = get_all_sensor_data(
            folder_data, 'accel', activity_list
        )
        accel_data_x = np.linalg.norm(accel_data_x, axis=1)
        # get the activity label matrix for the windows.
        activity_labels = get_activity_labels(
            accel_data_y, window_size, window_shift
        )
        # get the time domain features for accelerometer
        # mean, variance, max, min, range, rate of change
        time_domain_features = get_time_domain_features(
            accel_data_x, window_size, window_shift
        )
        # get the frequency domain features for accelerometer
        # spectral centroid, spectral slope, spectral rolloff 80%
        freq_domain_features = get_frequency_domain_features(
            accel_data_x, window_size, window_shift
        )
        accel_features = np.concatenate(
            (time_domain_features, freq_domain_features,), axis=1
        )
        

        # do the same for pressure
        window_size, window_shift = calculate_window_sizes(
            folder_data, 'pressure', window_length, shift_length
        )
        # press_data_y is uneeded
        press_data_x, press_data_y = get_all_sensor_data(
            folder_data, 'pressure', activity_list
        )
        time_domain_features = get_time_domain_features(
            press_data_x, window_size, window_shift
        )
        freq_domain_features = get_frequency_domain_features(
            press_data_x, window_size, window_shift
        )
        press_features = np.concatenate(
            (time_domain_features, freq_domain_features), axis=1
        )
        

        # combine features into one matrix
        shortest_len = min(accel_features.shape[0], press_features.shape[0])
        all_features = np.concatenate(
            (accel_features[:shortest_len], press_features[:shortest_len]),
            axis=1
        )
        activity_labels = activity_labels[:shortest_len]
        data_x.append(all_features)
        data_y.extend(activity_labels)
    data_x = np.concatenate(data_x)
    data_y = np.array(data_y)
    return data_x, data_y, activity_list

###############################################################################

# returns an array of folder data, the first column is the path to a file
# the second column is the the sensor type
# the third column is the activity type
def get_folder_data(folder_path):
    directory_entries = os.listdir(folder_path)
    #random.shuffle(directory_entries)
    folder_data = []
    for entry in directory_entries:
        # Only take entries which have a corresponding
        # pressure, gyrometer and accelerometer data
        sensor_type_start = entry.rfind('-')+1
        recording_str = entry[:sensor_type_start]
        pressure_entry = recording_str + 'pressure.txt'
        gyro_entry = recording_str + 'gyro.txt'
        accel_entry = recording_str + 'accel.txt'
        if (pressure_entry in directory_entries and
            gyro_entry in directory_entries and
            accel_entry in directory_entries and
            os.stat(folder_path + pressure_entry).st_size != 0 and
            os.stat(folder_path + gyro_entry).st_size != 0 and
            os.stat(folder_path + accel_entry).st_size != 0
            ):

            row = []
            row.append(folder_path+entry)
            # sensor type
            row.append(entry[sensor_type_start:entry.rfind('.')])
            #activity type
            row.append(entry[36:entry.rfind('-', 36, sensor_type_start-1)])
            folder_data.append(row)
    folder_data = np.array(folder_data)
    return folder_data

def create_activity_list(folder_data):
    activity_list = []
    for file in folder_data:
        if file[1] == 'pressure':
            if file[2] not in activity_list:
                activity_list.append(file[2])
    return activity_list

# returns number of samples per second
def get_sample_rate(file):
    sensor_data = np.genfromtxt(file, delimiter=',')
    total_time = sensor_data[-1,0] - sensor_data[0,0]

    return int(1000/total_time * sensor_data.shape[0])

def get_all_sensor_data(folder_data, sensor_type, activity_list):
    data_x = []
    data_y = []

    for file in folder_data:
        if file[1] == sensor_type and file[2] in activity_list:
            file_data = np.genfromtxt(file[0], delimiter=',')
            if len(file_data.shape) == 2:
                activity_code = activity_list.index(file[2])
                data_x.append(file_data)
                data_y.extend(np.full(file_data.shape[0], activity_code))
    # we won't need the time stamps
    data_x = np.concatenate(data_x)[:,1:]
    data_y = np.array(data_y)
    return data_x, data_y

# returns the number of samples per window and number of samples per shift
# depending on the number of seconds the window size and shift are supposed
# to be 
def calculate_window_sizes(
    folder_data, sensor_type, window_length, shift_length
):
    # figure out the number of rows per window/shift
    idx = 0
    while folder_data[idx, 1] != sensor_type and idx < folder_data.shape[0]:
        idx = idx + 1    
    sample_rate = get_sample_rate(folder_data[idx,0])
    window_size = sample_rate * window_length
    window_shift = sample_rate * shift_length
    return window_size, window_shift

def window_function(function, nparray, window_size, window_shift):
    if(window_size > nparray.shape[0]):
        print("window size too big")
        return
    n_windows = int((nparray.shape[0]-window_size)/window_shift)+1
    out_array = np.empty(n_windows)
    window_start = 0
    window_end = window_size
    idx = 0
    while idx < n_windows:
        out_array[idx] = function(idx, nparray[window_start:window_end])
        idx = idx + 1
        window_start = window_start + window_shift
        window_end = window_end + window_shift
    return out_array

# determines what the activity label for a window is based on the most
# common label
def get_activity_labels(data_y, window_size, window_shift):
    y = lambda i, window : np.argmax(np.bincount(window))
    window_y = window_function(y, data_y, window_size, window_shift)
    return window_y


# get the time domain features 
# mean, variance, max, min, range, rate of change
def get_time_domain_features(data_x, window_size, window_shift):
        means, variances = get_variances(data_x, window_size, window_shift)
        maxes, mins, ranges = get_ranges(data_x, window_size, window_shift)
        rocs = get_rates_of_change(data_x, window_size, window_shift)
        n_rows = means.shape[0]
        time_domain_features = np.concatenate(
            (means, variances, maxes, mins, ranges, rocs)
        )
        time_domain_features = np.reshape(time_domain_features, (6, n_rows))
        time_domain_features = np.transpose(time_domain_features)
        return time_domain_features

# get the frequency domain features
# spectral centroid, spectral slope, 80% rolloff point
def get_frequency_domain_features(data_x, window_size, window_shift):
    freq_domain = get_freq_domain(window_size)
    freq_domain_vals = get_freq_domain_vals(data_x, window_size, window_shift)
    spectral_centroids = get_spectral_centroids(freq_domain_vals, freq_domain)
    spectral_slopes = get_spectral_slopes(freq_domain_vals, freq_domain)
    spectral_rolloff = get_spectral_rolloff(freq_domain_vals, freq_domain, 0.8)
    n_rows = spectral_centroids.shape[0]
    frequency_domain_features = np.concatenate(
        (spectral_centroids, spectral_slopes, spectral_rolloff)
    )
    frequency_domain_features = np.reshape(frequency_domain_features, (
        3, n_rows)
    )
    frequency_domain_features = np.transpose(frequency_domain_features)
    return frequency_domain_features

# gets means from windows
def get_means(data_x, window_size, window_shift):
    mean = lambda i, window:np.mean(window)
    means = window_function(mean, data_x, window_size, window_shift)
    return means

# returns both the mean and variance
def get_variances(data_x, window_size, window_shift):
    means = get_means(data_x, window_size, window_shift)
    variance = lambda i, window : np.mean(np.square(window - means[i]))
    variances = window_function(variance, data_x, window_size, window_shift)
    return means, variances

# returns the max, the min and the range
def get_ranges(data_x, window_size, window_shift):
    max_f = lambda i, window : np.amax(window)
    min_f = lambda i, window: np.amin(window)
    maxes = window_function(max_f, data_x, window_size, window_shift)
    mins = window_function(min_f, data_x, window_size, window_shift)
    ranges = maxes - mins
    return maxes, mins, ranges

def get_rates_of_change(data_x, window_size, window_shift):
    rate_of_change = lambda i, window:(window[-1] - window[0])/window_size
    rates_of_change = window_function(
        rate_of_change, data_x, window_size, window_shift)
    return rates_of_change

def get_freq_domain_vals(data_x, window_size, window_shift):
    if(window_size > data_x.shape[0]):
        print("window size too big")
        return
    freq_domain_vals = []
    window_start = 0
    window_end = window_size
    while window_end < data_x.shape[0]:
        window = data_x[window_start:window_end]
        # window_mean = np.mean(window)
        transform = fftpack.fft(window)[:window_size//2+1]
        freq_domain_vals.append(transform.flatten())
        window_start = window_start + window_shift
        window_end = window_end + window_shift
    freq_domain_vals = np.array(freq_domain_vals)
    return freq_domain_vals

def get_freq_domain(window_size):
    return (np.arange(window_size)/window_size)[:window_size//2+1]

def get_spectral_centroids(freq_domain_vals, freq_domain):
    spec_cent = np.empty(freq_domain_vals.shape[0])
    for i, transform in enumerate(freq_domain_vals):       
        magnitudes = np.abs(transform)
        spec_cent[i] = np.sum(magnitudes * freq_domain)/np.sum(magnitudes)
    return spec_cent

def get_spectral_slopes(freq_domain_vals, freq_domain):
    spec_slopes = np.empty(freq_domain_vals.shape[0])
    for i, transform in enumerate(freq_domain_vals):
        magnitudes = np.abs(transform)
        # a,b,c,d aren't needed
        spec_slopes[i],a,b,c,d = stats.linregress(magnitudes, freq_domain)
    return spec_slopes

# threshold is a value between 0 and 1 which represents the % rolloff point
def get_spectral_rolloff(freq_domain_vals, freq_domain, threshold):
    rolloff_pts = np.empty(freq_domain_vals.shape[0])
    for i, transform in enumerate(freq_domain_vals):
        magnitudes = np.abs(transform)
        total_mag = np.sum(magnitudes)
        accum = 0
        idx = 0
        while accum/total_mag < threshold:
            accum = accum + magnitudes[idx]
            idx = idx + 1
        rolloff_pts[i] =  freq_domain[idx]
    return rolloff_pts

###############################################################################

