import csv
from datetime import datetime
from collections import Counter
# Load data from CSV
data = []
with open("tempLog-1.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Convert numeric timestamp to datetime
        time_stamp_num = float(row['Timestamp'])
        time_stamp = datetime.fromtimestamp(time_stamp_num)  # human-readable datetime
        
        air_temp = float(row['Air'])   # air temperature
        cpu_temp = float(row['CPU'])   # cpu temperature
        
        data.append({'Timestamp': time_stamp, 'Air': air_temp, 'CPU': cpu_temp})

        
# # print first 5 rows to see results
# for d in data[:5]:
#     print(d)

# basic stats
air_vals = sorted([d['Air'] for d in data])
cpu_vals = sorted([d['CPU'] for d in data])

# find basic stats. Here I used functions to make it clearer.
def mean(vals):
    return sum(vals)/len(vals)

def median(vals):
    n = len(vals)
    mid = n // 2
    if n % 2 == 0:
        return (vals[mid-1] + vals[mid]) / 2
    else:
        return vals[mid]

def mode(vals):
    counts = Counter(vals)
    max_count = max(counts.values())
    modes = [k for k,v in counts.items() if v == max_count]
    return modes[0]  

def data_range(vals):
    return max(vals) - min(vals)

# # print results (temporarily, I will remove after building up the app)
# print("Air Temperature Statistics:")
# print("Mean:", mean(air_vals))
# print("Median:", median(air_vals))
# print("Mode:", mode(air_vals))
# print("Min:", min(air_vals))
# print("Max:", max(air_vals))
# print("Range:", data_range(air_vals))
# print() # leave space

# print("CPU Temperature Statistics:")
# print("Mean:", mean(cpu_vals))
# print("Median:", median(cpu_vals))
# print("Mode:", mode(cpu_vals))
# print("Min:", min(cpu_vals))
# print("Max:", max(cpu_vals))
# print("Range:", data_range(cpu_vals))

# import matplotlib.pyplot as plt

# timestamps = [d['Timestamp'] for d in data]
# air_vals = [d['Air'] for d in data]
# cpu_vals = [d['CPU'] for d in data]

# plt.figure(figsize=(10,5))
# plt.plot(timestamps, air_vals, label="Air Temp", color="blue")
# plt.plot(timestamps, cpu_vals, label="CPU Temp", color="red")
# plt.xlabel("Timestamp")
# plt.ylabel("Temperature (°C)")
# plt.title("Temperature Over Time")
# plt.legend()
# plt.show()

# for interactive plot, we need plotly and pandas.
import pandas as pd
import plotly.express as px
import plotly.offline as pyo

df = pd.DataFrame(data)  # open data in pandas dataframe
fig_plot = px.line(df, x='Timestamp', y=['Air','CPU'],
              title="Temperature Over Time",
              labels={'value':'Temperature (°C)', 'variable':'Type'}) # plot the figure using pyplot express
pyo.plot(fig_plot, filename="plot.html", auto_open=True) # to view the data in HTML plot, we used pyo which is an offline plotly.



# Convert to from wide from int long form which is suitable for histogram so that we plot them on one figure
df_long = df.melt(id_vars='Timestamp', value_vars=['Air','CPU'], 
                  var_name='Type', value_name='Temperature')

# plot histograms together in an interactive way
fig = px.histogram(df_long, x='Temperature', color='Type', barmode='overlay',
                   nbins=30, title='Air and CPU Temperature Distribution')

fig.update_traces(opacity=0.6)  # make bars semi transparent
pyo.plot(fig, filename="histogram_air_cpu.html", auto_open=True)

# basic filtering functionality:

start_date = input("Please enter start date between 2021-11-12 and 2022-07-19 (YYYY-MM-DD): ")
end_date = input("Please enter end date between 2021-11-12 and 2022-07-19 (YYYY-MM-DD): ")

df_filtered = df[(df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)]
# calculate statistics over the specified range of date:
air_avg = df_filtered['Air'].mean()
air_max = df_filtered['Air'].max()
air_min = df_filtered['Air'].min()

cpu_avg = df_filtered['CPU'].mean()
cpu_max = df_filtered['CPU'].max()
cpu_min = df_filtered['CPU'].min()

# Print results including date range
print(f"\nStatistics for Air and CPU temperatures from {start_date} to {end_date}:")
print(f"Air Temperature -> Avg: {air_avg:.2f}, Max: {air_max}, Min: {air_min}")
print(f"CPU Temperature -> Avg: {cpu_avg:.2f}, Max: {cpu_max}, Min: {cpu_min}")



