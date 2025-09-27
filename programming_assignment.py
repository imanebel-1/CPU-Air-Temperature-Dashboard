import csv
from datetime import datetime
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

        
# print first 5 rows to see results
for d in data[:5]:
    print(d)