import streamlit as st
import pandas as pd
from programming_assignment_v3 import *  # import your helper functions
import plotly.express as px
import datetime
import plotly.graph_objects as go
import requests

# ------------------------------
# page settings and title
# ------------------------------
st.set_page_config(page_title="temperature dashboard", layout="wide")
st.title("🌡 temperature dashboard")
st.markdown("welcome to imane's dashboard for cpu temperature insights.")
st.sidebar.image("check-cpu-temperature.webp", width=150)  # small sidebar image

# ------------------------------
# load data
# ------------------------------
data = load_data("tempLog-1.csv")  # load csv data
df = pd.DataFrame(data)  # convert to pandas dataframe

# ------------------------------
# sidebar filters
# ------------------------------
st.sidebar.header("filters")

# dynamic min/max dates from the data
min_date = df['Timestamp'].min().date()
max_date = df['Timestamp'].max().date()

# date input widgets
start_date = st.sidebar.date_input("start date", min_value=min_date, max_value=max_date, value=min_date)
end_date = st.sidebar.date_input("end date", min_value=start_date, max_value=max_date, value=max_date)

# slider for cpu anomaly threshold
threshold = st.sidebar.slider("cpu anomaly threshold (std dev)", 1.0, 5.0, 2.0)

# sidebar display options
st.sidebar.header("display options")
show_all_data = st.sidebar.checkbox("display all data", value=True)
show_statistics = st.sidebar.checkbox("show filtered statistics")
show_anomalies = st.sidebar.checkbox("show cpu anomaly table")
plot_anomalies = st.sidebar.checkbox("plot cpu with anomalies")
show_correlation = st.sidebar.checkbox("show correlation plot")
show_time_plot = st.sidebar.checkbox("show temperature over time")
show_histogram = st.sidebar.checkbox("show temperature histogram")

# convert start/end dates to full datetime
start_date_dt = datetime.datetime.combine(start_date, datetime.time(0,0,0))
end_date_dt = datetime.datetime.combine(end_date, datetime.time(23,59,59))

# filter dataframe based on selected dates
df_filtered = df[(df['Timestamp'] >= start_date_dt) & (df['Timestamp'] <= end_date_dt)]

# ------------------------------
# gauge for latest cpu temperature
# ------------------------------
current_cpu_temp = df_filtered['CPU'].iloc[-1]  # last cpu value
current_air_temp = df_filtered['Air'].iloc[-1]  # last air value

fig_cpu_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=current_cpu_temp,
    title={'text': "cpu end's date temperature (°c)"},
    gauge={
        'axis': {'range': [0, 60]},
        'bar': {'color': "red"},
        'steps': [
            {'range': [0, 40], 'color': "lightgreen"},
            {'range': [40, 50], 'color': "yellow"},
            {'range': [50, 60], 'color': "red"}
        ]
    }
))
fig_cpu_gauge.update_layout(height=250, width=250)
st.plotly_chart(fig_cpu_gauge, use_container_width=10)

# ------------------------------
# main dashboard content
# ------------------------------
st.subheader("selected date range")
st.write(f"from {start_date_dt.strftime('%b %d, %Y')} to {end_date_dt.strftime('%b %d, %Y')}")

# show filtered data
if show_all_data:
    st.write(df_filtered)

# show basic statistics
if show_statistics:
    air_avg, air_max, air_min, cpu_avg, cpu_max, cpu_min, count = filter_by_date(df_filtered, start_date_dt, end_date_dt)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("air temperature avg", f"{air_avg:.2f} °c", delta=f"max: {air_max}, min: {air_min}")
    with col2:
        st.metric("cpu temperature avg", f"{cpu_avg:.2f} °c", delta=f"max: {cpu_max}, min: {cpu_min}")
    st.write(f"number of readings: {count}")

# show cpu anomalies table
if show_anomalies:
    df_anomaly, anomalies = detect_anomalies(df_filtered, threshold)
    st.subheader("cpu anomalies")
    st.write(f"number of anomalies detected: {len(anomalies)}")
    st.dataframe(anomalies[['Timestamp','CPU']])

# plot cpu anomalies
if plot_anomalies:
    df_anomaly, anomalies = detect_anomalies(df_filtered, threshold)
    st.subheader("cpu temperature with anomalies")
    fig_cpu_anomaly = plot_cpu_with_anomalies(df_filtered, anomalies)
    st.plotly_chart(fig_cpu_anomaly, use_container_width=True)

# correlation plot
if show_correlation:
    correlation_value = analyse_correlation(df_filtered)
    st.write(f"correlation between cpu and air temperature: {correlation_value:.2f}")
    fig_corr = px.scatter(
        df_filtered,
        x='Air',
        y='CPU',
        trendline='ols',
        labels={'Air':'air temperature','CPU':'cpu temperature'},
        title="cpu vs air temperature"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# time series plot
if show_time_plot:
    fig_time = px.line(df_filtered, x='Timestamp', y=['Air','CPU'], labels={'value':'temperature','variable':'type'},
                       title="temperature over time")
    st.plotly_chart(fig_time, use_container_width=True)

# histogram plot
if show_histogram:
    df_long = df_filtered.melt(id_vars='Timestamp', value_vars=['Air','CPU'], var_name='Type', value_name='Temperature')
    fig_hist = px.histogram(df_long, x='Temperature', color='Type', barmode='overlay', nbins=30, opacity=0.6,
                            title="air and cpu temperature distribution")
    st.plotly_chart(fig_hist, use_container_width=True)

# ------------------------------
# live weather info from openweathermap
# ------------------------------
st.sidebar.header("live weather")

api_key = "15d0d6eec4d0b8c7732f1e0cf16b79d9"
city = "London"
url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

response = requests.get(url)
data = response.json()

if response.status_code == 200:
    temp = data['main']['temp']
    st.sidebar.metric(label=f"current temp in {city}", value=f"{temp} °c")
    st.sidebar.write(data['weather'][0]['description'].title())
else:
    st.sidebar.error("could not fetch temperature data")
