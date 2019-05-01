import numpy as np
import matplotlib.pyplot as plt
import data_extraction

# plots all the acitivity sessions as if they happened right after each other
def plot_activity_vs_time(folder_data):
    
    data = {}
    time = 0

    for file in folder_data:

        # only need one file per recording session
        # the specific sensor doesn't really matter
        if file[1] == 'pressure':

            activity_name = file[2]

            # get the activity run time
            f = open(file[0])
            line_list = f.readlines()
            f.close()

            first_line = line_list[0]
            start = int(first_line[:13])
            last_line = line_list[-1]
            end = int(last_line[:13])

            # add data to plot
            data[time] = activity_name
            time = time + end - start
            data[time] = activity_name
            time = time + 1

    times = data.keys()
    activities = data.values()   
    plt.plot(times, activities)
    plt.ylabel = 'time(seconds)'  

def plot_accel_vs_time(folder_data):
    accel_magnitudes = []
    times = []

    time = 0

    for file in folder_data:
        if file[1] == 'accel':
            file_data = np.genfromtxt(file[0], delimiter=',')
            file_times = file_data[:,0]
            start_time = file_times[0]
            end_time = file_times[-1]
            
            file_times = file_times - start_time + time
            times.extend(file_times)

            file_accel_values = file_data[:, 1:]
            file_accel_magnitudes = np.linalg.norm(file_accel_values, axis=1)
            accel_magnitudes.extend(file_accel_magnitudes)

            time = time + end_time - start_time + 1000
    plt.plot(times, accel_magnitudes)
    
def plot_pressure_vs_time(folder_data):
    pressure_values = []
    times = []

    time = 0
    
    for file in folder_data:
        if file[1] == 'pressure':
            file_data = np.genfromtxt(file[0], delimiter=',')
            file_times = file_data[:,0]
            start_time = file_times[0]
            end_time = file_times[-1]
            
            file_times = file_times - start_time + time
            times.extend(file_times)

            pressure_values.extend(file_data[:, 1:])

            time = time + end_time - start_time + 1000
    plt.plot(times, pressure_values)

###############################################################################

# location of my data, but you can change it to someone else
folder_path = './A2_Data/pocket/dshen/'
folder_data = data_extraction.get_folder_data(folder_path)

n_rows = 3
plt.subplot(n_rows, 1, 1)
plot_activity_vs_time(folder_data)
plt.subplot(n_rows, 1, 2)
plot_accel_vs_time(folder_data)
plt.subplot(n_rows, 1, 3)
plot_pressure_vs_time(folder_data)
plt.show()
