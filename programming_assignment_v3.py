from collections import Counter
import pandas as pd
import plotly.express as px
import plotly.offline as pyo
import matplotlib.pyplot as plt
import numpy as np
import os


# load data from CSV using pandas
def load_data(file_path):
    if not os.path.isfile(file_path): # if the file is not found, raise error
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    df = pd.read_csv(file_path, parse_dates=['Timestamp'],
                     date_parser=lambda x: pd.to_datetime(float(x), unit='s')) # load the file and convert timestamp
    
    # column checks if all required columns are present
    required_cols = ['Timestamp', 'Air', 'CPU']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # convert Air and CPU to float, handle errors
    for col in ['Air','CPU']:
        try: # check if the column can be converted to float
            df[col] = df[col].astype(float)
        except Exception: # raise error otherwise
            raise ValueError(f"Column {col} contains non-numeric data")
    
    # fill missing values with column mean
    if df.isna().any().any(): 
        df.fillna(df.mean(), inplace=True)
        print("Warning: Missing values detected and filled with column mean.")
    
    return df.to_dict('records')  # return as list of dicts


# basic stats
def mean(vals):
    return np.mean(vals)

def median(vals):
    return np.median(vals)

def mode(vals):
    counts = Counter(vals)
    max_count = max(counts.values())
    modes = [k for k,v in counts.items() if v == max_count]
    return modes[0]  

def data_range(vals):
    return np.max(vals) - np.min(vals)


# plot temperature over time (interactive)
def plot_temperature(df):
    if not df: # check if dataframe is empty
        print("Warning: Empty data for plotting temperature.")
        return
    fig_plot = px.line(df, x='Timestamp', y=['Air','CPU'],
                  title="Temperature Over Time",
                  labels={'value':'Temperature (°C)', 'variable':'Type'})
    try: # check if plotting works
        pyo.plot(fig_plot, filename="plot.html", auto_open=True)
    except Exception as e: # if plotting doesnt work raise error
        print(f"Error plotting temperature: {e}")


# plot histogram
def plot_histogram(df):
    if not df: # check if dataframe is empty
        print("Warning: Empty data for plotting histogram.")
        return
    df_long = df.melt(id_vars='Timestamp', value_vars=['Air','CPU'],  # convert to long format to plot histogram
                      var_name='Type', value_name='Temperature')
    fig = px.histogram(df_long, x='Temperature', color='Type', barmode='overlay',
                       nbins=30, title='Air and CPU Temperature Distribution') # histogram plot using plotly express
    fig.update_traces(opacity=0.6)
    try: # check if plotting works
        pyo.plot(fig, filename="histogram_air_cpu.html", auto_open=True)
    except Exception as e:
        print(f"Error plotting histogram: {e}") # print error if plotting fails, so code doesnt crash.


# basic filtering functionality:
def filter_by_date(df, start_date, end_date): # filter by date range
    if pd.to_datetime(start_date) > pd.to_datetime(end_date): # check if start date is earlier than end date
        raise ValueError("start_date must be earlier than end_date") # raise error 
    
    df_filtered = df[(df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)]
    if df_filtered.empty:
        print("Warning: No data in the selected date range.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0

    air_avg = df_filtered['Air'].mean() # find mean, max, min and count of the filtered data for Air column
    air_max = df_filtered['Air'].max()
    air_min = df_filtered['Air'].min()

    cpu_avg = df_filtered['CPU'].mean() # find mean, max, min and count of the filtered data for CPU column
    cpu_max = df_filtered['CPU'].max()
    cpu_min = df_filtered['CPU'].min()
    count = df_filtered.shape[0] # count of rows in the filtered data

    return air_avg, air_max, air_min, cpu_avg, cpu_max, cpu_min, count # return the calculated values


# transformation and plot
def transform_and_plot(df, feature, transform_type, start_date, end_date): # transform filtered data into log or standardisation
    df_filtered = df[(df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)].copy()
    if df_filtered.empty: # check if filtered data is empty
        print("Warning: No data in the selected date range for plotting.")
        return
    if feature not in df_filtered.columns: # check if the feature column exists
        raise ValueError(f"Column {feature} not found for transformation.")
    
    if transform_type == 'log': # check what type of transformation user wants
        if (df_filtered[feature] <= 0).any(): # ensure no non-positive values for log transformation
            print("Warning: Non-positive values detected. Using log1p instead of log.")
            df_filtered[feature + '_transformed'] = np.log1p(df_filtered[feature])
        else: # apply log transformation if all positive values
            df_filtered[feature + '_transformed'] = np.log(df_filtered[feature])
    elif transform_type == 'standardisation': # otherwise check if standardisation is requested
        std_val = df_filtered[feature].std() # calculate standard deviation
        if std_val == 0: # if std is zero, avoid dividng by zero and raise an error.
            df_filtered[feature + '_transformed'] = 0 # set all transformed values to zero
            print("Warning: Zero variance, transformed values set to 0.") # warn user about it
        else: #otherwise apply standardisation
            df_filtered[feature + '_transformed'] = (df_filtered[feature] - df_filtered[feature].mean()) / std_val
    else: # if transformation is unknown, use the raw data (non transformed)
        df_filtered[feature + '_transformed'] = df_filtered[feature]
    
    plt.figure(figsize=(10,5)) # plot the transformed data using matplotlib
    plt.plot(df_filtered['Timestamp'], df_filtered[feature + '_transformed'], marker='o')
    plt.title(f"{feature} ({transform_type}) from {start_date} to {end_date}")
    plt.xlabel("Timestamp") # labels
    plt.ylabel(feature + f" ({transform_type})")
    plt.xticks(rotation=45) # rotate x ticks for better visibility
    plt.grid(True)
    plt.tight_layout() # tight layout for better padding
    plt.show()


# anomaly detection
def detect_anomalies(df, threshold): # detect anomalies in CPU temperature based on user defined threshold
    if threshold <= 0: # if threshold is not positive, raise error
        raise ValueError("Threshold must be positive.")
    if df['CPU'].std() == 0: # if std is zero (no variance), no anomalies can be detected
        print("Warning: CPU variance is zero. No anomalies will be detected.")
        df['CPU_anomaly'] = False
        return df, df[df['CPU_anomaly']] # return empty anomalies dataframe
    
    cpu_mean = df['CPU'].mean() # otherwise calculate mean and std
    cpu_std = df['CPU'].std()
    df['CPU_anomaly'] = (df['CPU'] > cpu_mean + threshold*cpu_std)
    anomalies = df[df['CPU_anomaly']]
    return df, anomalies # return df and anomalies dataframe


## added correlation between cpu and air temperatures:
def analyse_correlation(df):
    if df['CPU'].std() == 0 or df['Air'].std() == 0: # check for zero standard deviation
        print("Warning: Zero variance detected. Correlation may be undefined.")
    correlation = df['CPU'].corr(df['Air']) # calculate correlation
    return correlation # return correlation value

def plot_correlation(df): # plot correlation using plotly
    if df.empty:
        print("Warning: Cannot plot correlation. Data is empty.")
        return
    fig_correlation = px.scatter(  
        df,
        x='Air',
        y='CPU',
        trendline='ols',
        labels={'Air': 'Air Temperature (°C)', 'CPU': 'CPU Temperature (°C)'},
        title='Correlation between CPU and Air Temperature'
    )  # scatter plot with trendline
    fig_correlation.update_layout(title_x=0.5)
    try: # check if plotting works
        pyo.plot(fig_correlation, filename="correlation.html", auto_open=True)
    except Exception as e: # raise exception if plotting fails instead of crashing
        print(f"Error plotting correlation: {e}")


# added this to plot nicely the anomalies on the a dashboard
def plot_cpu_with_anomalies(df, anomalies): # plot cpu temperature with anomalies highlighted
    """
    Returns a Plotly figure of CPU temperature with anomalies highlighted.
    """
    if df.empty: # if dataframe is empty, raise warning and return
        print("Warning: Cannot plot CPU with anomalies. Data is empty.") # print warning
        return # return nothing to stop the function
    anomalies = anomalies if anomalies else pd.DataFrame(columns=['Timestamp','CPU']) # handle empty anomalies
    fig = px.line(df, x='Timestamp', y='CPU', title="CPU Temperature with Anomalies") # line plot of cpu temperature
    fig.add_scatter(x=anomalies['Timestamp'], y=anomalies['CPU'],
                    mode='markers', marker=dict(color='red', size=8),
                    name='Anomalies') # add anomalies as red markers
    try: # check if plotting works
        pyo.plot(fig, filename="cpu_with_anomalies.html", auto_open=True)
    except Exception as e: # raise exception if plotting fails instead of crashing
        print(f"Error plotting CPU with anomalies: {e}")
    return fig
