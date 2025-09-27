import csv
#load data into CSV file:
data = []
with open("tempLog-1.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        time_stamp = float(row['Timestamp'])
        air_temp = float(row['Air'])   # air temperature
        cpu_temp = float(row['CPU'])  # cpu temperature
        data.append({'Timestamp': time_stamp, 'Air': air_temp, 'CPU': cpu_temp})
