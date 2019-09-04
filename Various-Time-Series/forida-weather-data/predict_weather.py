import matplotlib.pyplot as plt
import numpy as np

# Read weather data
weather_filename = 'ft-lauderdale-beach-1952-2017.csv'
weather_file = open(weather_filename)
weather_data = weather_file.read()
weather_file.close()

# print(len(weather_data))
# print(weather_data[:200])

# Break the weather records into lines
lines = weather_data.split('\n')

# print(len(lines))
# for i in range(5):
#     print(lines[:i])

# Separating the data into labels and values
labels = lines[0]
values = lines[1:]
n_values = len(values)

# print(labels)
# for i in range(10):
#     print(values[i])

# Break teh list of comma-separated value strings
# into lists of values

year = [];      j_year = 1
month = [];     j_month = 2
day = [];       j_day = 3
max_temp = [];  j_max_temp = 5

for i_row in range(n_values):
    split_values = values[i_row].split(',')
    # print(split_values)
    if len(split_values) >= j_max_temp:
        year.append(int(split_values[j_year]))
        month.append(int(split_values[j_month]))
        day.append(int(split_values[j_day]))
        max_temp.append(float(split_values[j_max_temp]))

# for i_day in range(100):
#     print(max_temp[i_day])

# plt.plot(max_temp)
# plt.show()

# Isolate the recent data
i_mid = len(max_temp) // 2
temps = np.array(max_temp[i_mid:])
temps[np.where(temps == -99.9)] = np.nan

# plt.plot(temps, color='black', marker='.', linestyle='none')
# plt.show()

# Remove all the nans.
# Trim both ends and fill nans in the middle
# Find the first non-nan
# print(np.where(np.isnan(temps))[0])
# print(np.where(np.logical_not(np.isnan(temps)))[0])
i_start = np.where(np.logical_not(np.isnan(temps)))[0][0]
temps = temps[i_start:]
#print(np.where(np.isnan(temps))[0])
i_nans = np.where(np.isnan(temps))[0]
#print(np.diff(i_nans))
# Replace all nans with the most recent non-nan.
for i in range(temps.size):
    if np.isnan(temps[i]):
        temps[i] = temps[i-1]

plt.plot(temps)
plt.show()