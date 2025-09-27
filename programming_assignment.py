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

# print results (temporarily, I will remove after building up the app)
print("Air Temperature Statistics:")
print("Mean:", mean(air_vals))
print("Median:", median(air_vals))
print("Mode:", mode(air_vals))
print("Min:", min(air_vals))
print("Max:", max(air_vals))
print("Range:", data_range(air_vals))
print() # leave space

print("CPU Temperature Statistics:")
print("Mean:", mean(cpu_vals))
print("Median:", median(cpu_vals))
print("Mode:", mode(cpu_vals))
print("Min:", min(cpu_vals))
print("Max:", max(cpu_vals))
print("Range:", data_range(cpu_vals))