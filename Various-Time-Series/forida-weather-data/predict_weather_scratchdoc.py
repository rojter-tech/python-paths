# Metaquestion: Should I buy plane ticket or not to Fourt Lauderdale?
# Question to answer: Will the high temperature in Fort Lauderdale be
# above 85 degrees Fahrenheit three days from now?
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

# Deciding, at this stage, to work with timestamp and
# maximum temperature for our use case.
# Break the list of comma-separated value strings
# and append into lists of values.

year = [];      j_year = 1
month = [];     j_month = 2
day = [];       j_day = 3
max_temp = [];  j_max_temp = 5

for i_row in range(n_values):
    row_split_values = values[i_row].split(',')
    # print(split_values)
    if len(row_split_values) >= j_max_temp:
        year.append(int(row_split_values[j_year]))
        month.append(int(row_split_values[j_month]))
        day.append(int(row_split_values[j_day]))
        max_temp.append(float(row_split_values[j_max_temp]))

# for i_day in range(100):
#     print(max_temp[i_day])

# plt.plot(max_temp)
# plt.show()

# Isolating the recent data chunk, choosing to neglect the old chunk of
# temperature data and make the assumtion that the recent data have better
# predictive power for our use case, thus answer the question.
# Inserting data into a numerical python array is convinuent at this stage.
i_mid = len(max_temp) // 2
temps = np.array(max_temp[i_mid:])
year = np.array(year[i_mid:])
month = np.array(month[i_mid:])
day = np.array(day[i_mid:])
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
year = year[i_start:]
month = month[i_start:]
day = day[i_start:]
#print(np.where(np.isnan(temps))[0])
# Exploring however there is a systematic leftout of temperature data
i_nans = np.where(np.isnan(temps))[0]
#print(np.diff(i_nans))

# Replace all nans with the most recent non-nan,
# as the method for handling missing temperature data
# noting that this could have been done in a couple of diffrent ways
for i in range(temps.size):
    if np.isnan(temps[i]):
        temps[i] = temps[i-1]

#plt.plot(temps)
#plt.show()

# We can now regard the dataset cleaned up
# Determine whether the previous day's temperature
# is related to that of the following day. Lets check
# by plotting temperature against temperature shifted one day
#    leavs last  leaves first
# plt.plot(temps[:-1], temps[1:], color='black', marker='.', markersize=1, linestyle='none')
# plt.xlabel("Temperature at day d")
# plt.ylabel("Temperature at day d+1")
# plt.show()

# Show the relationship between two variables.
# adding some jitter (random noise) to pertubate
# potentially on-top stacked datapoints

def jscatter(x,y,std=.5,xlabel="x values",ylabel="y values", title="Chart"):
    """
    x,y -- array-like objects
    Make a scatter plot with jitter.
    jscatter(x,y,std=.5,xlabel='x',ylabel='y')
    """
    x_jitter = x + np.random.normal(size=x.size, scale=std)
    y_jitter = y + np.random.normal(size=y.size, scale=std)
    plt.plot(x_jitter,y_jitter,
             color='black',
             marker='.',
             markersize=3,
             linestyle='none',
             alpha=.1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

# shift = 1
# jscatter(temps[:-shift], temps[shift:])
# print(np.corrcoef(temps[:-shift], temps[shift:]))

# correlation as we increase the shift
# first 5 gives, for example, cv value of:
# 0.8502081607485872
# 0.7591285235024923
# 0.7187302160565565
# 0.6986113377990819
# 0.688032826185699

n_days = 365*9
autocorr = []
for shift in range(1,n_days):
    cv = np.corrcoef(temps[:-shift], temps[shift:])[0][1]
    autocorr.append(cv)

# plt.plot(autocorr)
# Add a sinusoidal approximation curve
d = np.arange(n_days)
fit = .6 * np.cos(2 * np.pi * d / 365)
# plt.plot(d, fit, color="green")
# plt.show()

# Autocorrelation patterns for temperature shows a strongly annual
# trend, which can be subracted away to quantify what is left, however,
# to catch this pattern in a more general way we could calculate the
# median temperature for every datapoints around some range around the days
# for every year that theese day of the year have records. For example the median for
# Juli 15 can be calculated from all datapoints from Juli 10 to Juli 20 that
# exist in the dataset, and in 'temps' portion of this dataset there is 19 years of 
# such data which gives 10*19 = 190 datapoints for each estimation to be made.

def find_day_of_year(year, month, day):
    """
    Convert year, month, date to day of the year
    Januari 1 = 0

    Parameters
    -----------
    year: int
    month: int
    day: int

    Returns
    -------
    day_of_year: int
    """
    days_each_month = np.array([
        31,  # January [0]
        28,  # February [1]
        31,  # Mars [2]
        30,  # April [3]
        31,  # May [4]
        30,  # June [5]
        31,  # July [6]
        31,  # August [7]
        30,  # September [8]
        31,  # October [9]
        30,  # November [10]
        31   # December [11]
    ])
    # For leap years
    if year%4 == 0:
        days_each_month[1] += 1
    
    months_days_passed = np.array(days_each_month[:(month-1)])
    total_days_passed = sum(months_days_passed) + day
    day_of_year = total_days_passed - 1
    return day_of_year


day_of_year = np.zeros(temps.size)
for i_row in range(temps.size):
    day_of_year[i_row] = find_day_of_year(year[i_row],
                          month[i_row],
                          day[i_row]
                          )

# jscatter(day_of_year, temps)

## Create 10-day medians for each day of the year
median_temp_calendar = np.zeros(366)
ten_day_medians = np.zeros(temps.size)
for i_day in range(0, 365):
    lower_bound_day = i_day - 5
    higher_bound_day = i_day + 5
    if lower_bound_day < 0:
        lower_bound_day += 365
    if higher_bound_day > 365:
        higher_bound_day += -365
    if lower_bound_day < higher_bound_day:
        i_window_days = np.where(
            np.logical_and(day_of_year >= lower_bound_day,
            day_of_year <= higher_bound_day))
    else:
        i_window_days = np.where(
            np.logical_or(day_of_year >=lower_bound_day,
                      day_of_year <= higher_bound_day))

    ten_day_median = np.median(temps[i_window_days])
    median_temp_calendar[i_day] = ten_day_median
    ten_day_medians[np.where(day_of_year == i_day)] = ten_day_median

    if i_day == 364:
        ten_day_medians[np.where(day_of_year == 365)] = ten_day_median
        median_temp_calendar[365] = ten_day_median
## Gaps to be filled
# 1. Calculate 'day_of_year' for each temp [x]
# 2. Handle beginning and end of year      [x]
# 3. Handle leap year              [x]

# print(ten_day_medians.size, np.unique(ten_day_medians), ten_day_medians)
# jscatter(ten_day_medians,temps)
# jscatter(ten_day_medians, temps - ten_day_medians)
# print(temps[np.where(temps<50)],
#       day_of_year[np.where(temps<50)],
#       np.where(temps<50)[0] / 365.25 )
# plt.plot(temps)
# plt.plot(ten_day_medians, color='black')
# plt.show()

def predict(year, month, day, temperature_calendar):
    """
    For a given day, month, and year, predict the 
    high temperature for Fort Lauderdale Beach.

    Parameters
    ----------
    year, month, day: int
    The date of interest
    temperature_calendar: arraf of floats
    The typical temperature for each day of the year.
    Jan 1 = 0, etc.
    
    Returns
    -------
    prediction: float
    """
    i_day = find_day_of_year(year,month,day)
    prediction = temperature_calendar[i_day]
    return prediction

if __name__ == '__main__':
    for test_day in range(1,30):
        test_year = 2016
        test_month = 6
        prediction = predict(test_year, test_month, test_day, median_temp_calendar)
        print(test_year, test_month, test_day, prediction)

