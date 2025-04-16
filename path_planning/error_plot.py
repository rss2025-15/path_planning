import csv
import matplotlib.pyplot as plt

csv_file = '/home/racecar/racecar_ws/src/path_planning/ctrl_error.csv'

time = []
error = []

with open(csv_file, 'r') as file:
    reader = list(csv.reader(file))
    start_time = float(reader[0][0])
    for row in reader:
        time.append(float(row[0]) - start_time)
        error.append(float(row[1]))

plt.figure(figsize=(10, 5))
plt.plot(time, error, label='dist_to_segments', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Error')
plt.grid(True)
plt.legend()
plt.savefig('/home/racecar/racecar_ws/src/path_planning/error_plot.png')
plt.close()
