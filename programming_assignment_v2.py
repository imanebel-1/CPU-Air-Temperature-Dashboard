import csv
from datetime import datetime
from collections import Counter
import pandas as pd
import plotly.express as px
import plotly.offline as pyo
import matplotlib.pyplot as plt
import numpy as np

# load data from CSV
def load_data(file_path):
    data = []
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric timestamp to datetime
            time_stamp_num = float(row['Timestamp'])
            time_stamp = datetime.fromtimestamp(time_stamp_num)  # human-readable datetime
            
            air_temp = float(row['Air'])   # air temperature
            cpu_temp = float(row['CPU'])   # cpu temperature
            
            data.append({'Timestamp': time_stamp, 'Air': air_temp, 'CPU': cpu_temp})
    return data

# basic stats
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

# plot temperature over time (interactive)
def plot_temperature(df):
    fig_plot = px.line(df, x='Timestamp', y=['Air','CPU'],
                  title="Temperature Over Time",
                  labels={'value':'Temperature (°C)', 'variable':'Type'})
    pyo.plot(fig_plot, filename="plot.html", auto_open=True)

# plot histogram
def plot_histogram(df):
    df_long = df.melt(id_vars='Timestamp', value_vars=['Air','CPU'], 
                      var_name='Type', value_name='Temperature')
    fig = px.histogram(df_long, x='Temperature', color='Type', barmode='overlay',
                       nbins=30, title='Air and CPU Temperature Distribution')
    fig.update_traces(opacity=0.6)
    pyo.plot(fig, filename="histogram_air_cpu.html", auto_open=True)

# basic filtering functionality:
def filter_by_date(df, start_date, end_date):
    df_filtered = df[(df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)]
    air_avg = df_filtered['Air'].mean()
    air_max = df_filtered['Air'].max()
    air_min = df_filtered['Air'].min()

    cpu_avg = df_filtered['CPU'].mean()
    cpu_max = df_filtered['CPU'].max()
    cpu_min = df_filtered['CPU'].min()
    count = df_filtered.shape[0]

    return air_avg, air_max, air_min, cpu_avg, cpu_max, cpu_min, count

# transformation and plot
def transform_and_plot(df, feature, transform_type, start_date, end_date):
    df_filtered = df[(df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)].copy()
    
    if transform_type == 'log':
        df_filtered[feature + '_transformed'] = np.log(df_filtered[feature])
    elif transform_type == 'standardisation':
        df_filtered[feature + '_transformed'] = (df_filtered[feature] - df_filtered[feature].mean()) / df_filtered[feature].std()
    else:
        df_filtered[feature + '_transformed'] = df_filtered[feature]
    
    plt.figure(figsize=(10,5))
    plt.plot(df_filtered['Timestamp'], df_filtered[feature + '_transformed'], marker='o')
    plt.title(f"{feature} ({transform_type}) from {start_date} to {end_date}")
    plt.xlabel("Timestamp")
    plt.ylabel(feature + f" ({transform_type})")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# anomaly detection
def detect_anomalies(df, threshold):
    cpu_mean = df['CPU'].mean()
    cpu_std = df['CPU'].std()
    df['CPU_anomaly'] = (df['CPU'] > cpu_mean + threshold*cpu_std)
    anomalies = df[df['CPU_anomaly']]
    return df, anomalies

## added correlation between cpu and air temperatures:
def analyse_correlation(df):
    correlation = df['CPU'].corr(df['Air'])
    return correlation

def plot_correlation(df):
    fig_correlation = px.scatter(
        df,
        x='Air',
        y='CPU',
        trendline='ols',
        labels={'Air': 'Air Temperature (°C)', 'CPU': 'CPU Temperature (°C)'},
        title='Correlation between CPU and Air Temperature'
    )
    fig_correlation.update_layout(title_x=0.5)
    pyo.plot(fig_correlation, filename="correlation.html", auto_open=True)

# main function added
if __name__ == "__main__":
    data = load_data("tempLog-1.csv")
    df = pd.DataFrame(data)

    # test correlation
    correlation_value = analyse_correlation(df)
    print(f"Correlation between CPU and Air temperature: {correlation_value:.2f}")
    plot_correlation(df)

    # test filtering
    air_avg, air_max, air_min, cpu_avg, cpu_max, cpu_min, count = filter_by_date(df, "2021-11-12", "2022-07-19")
    print(f"\nStatistics for Air and CPU temperatures:")
    print(f"Air Temperature -> Avg: {air_avg:.2f}, Max: {air_max}, Min: {air_min}")
    print(f"CPU Temperature -> Avg: {cpu_avg:.2f}, Max: {cpu_max}, Min: {cpu_min}")
    print(f"Number of readings: {count}")

    # test anomaly detection
    threshold = 2
    df, anomalies = detect_anomalies(df, threshold)
    print(f"Number of anomalies detected in CPU (threshold={threshold}):", len(anomalies))
    print(anomalies[['Timestamp','CPU']])

# added this to plot nicely the anomalies on the a dashboard
def plot_cpu_with_anomalies(df, anomalies):
    """
    Returns a Plotly figure of CPU temperature with anomalies highlighted.
    """
    fig = px.line(df, x='Timestamp', y='CPU', title="CPU Temperature with Anomalies")
    fig.add_scatter(x=anomalies['Timestamp'], y=anomalies['CPU'],
                    mode='markers', marker=dict(color='red', size=8),
                    name='Anomalies')
    return fig