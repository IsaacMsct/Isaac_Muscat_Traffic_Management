import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def map_traffic(traffic):
    if 'no traffic' in traffic:
        return 0
    elif 'moderate' in traffic:
        return 5
    elif 'heavy' in traffic:
        return 10
    return None

def read_and_process_data(file_path):
    # Read the data, assuming each row is a traffic report with the location name embedded
    data = pd.read_csv(file_path, header=None)
    columns = ['datetime']

    # Extracting location names from the first data row (assuming the first row is representative)
    first_row = pd.read_csv(file_path, header=None, nrows=1)
    for idx in range(1, len(first_row.columns)):
        # Split the string to extract the location part before the colon
        location_name = first_row.iloc[0, idx].split(':')[0].strip()
        columns.append(location_name)

    data.columns = columns
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data[~data['datetime'].dt.dayofweek.isin([5, 6])]

    # Mapping traffic descriptions to numerical values
    for column in data.columns[1:]:
        data[column] = data[column].apply(lambda x: x.split(':')[-1].strip()).apply(map_traffic)

    hourly_avg = data.groupby(data['datetime'].dt.hour).mean()
    hourly_avg = hourly_avg.reindex(range(24), fill_value=0)
    hourly_avg = hourly_avg.reset_index()
    hourly_avg.columns = ['Hour'] + list(hourly_avg.columns[1:])

    return hourly_avg

def plot_traffic(hourly_avg, ax, title):
    for location in hourly_avg.columns[1:]:
        ax.plot(hourly_avg['Hour'], hourly_avg[location], label=location)
    ax.set_title(title)
    ax.set_xlabel('Hour of the Day')
    ax.set_ylabel('Average Traffic Level')
    ax.grid(True)
    ax.legend()
    ax.set_xticks(range(0, 24))

# Reading and processing data for each webcam
webcam1_data = read_and_process_data('msida_traffic_report.txt')
webcam2_data = read_and_process_data('marsahamrun_traffic_report.txt')
webcam3_data = read_and_process_data('qormicanonroad_traffic_report.txt')

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(18, 10))  # Adjusted for better fit in landscape mode
plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1, hspace=0.4)  # Increase spacing for subtitle
plot_traffic(webcam1_data, axs[0], 'Webcam 1 (Msida) Traffic')
plot_traffic(webcam2_data, axs[1], 'Webcam 2 (Marsa Hamrun) Traffic')
plot_traffic(webcam3_data, axs[2], 'Webcam 3 (Qormi Canon Road) Traffic')
fig.suptitle('Traffic Data Analysis for Weekdays', fontsize=16)  # Subtitle for the figure
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to make room for the suptitle
plt.show()
