#!/Users/SandeepGangundi/anaconda/envs/carnd-term1/bin/python
import os
import csv
import numpy as np

cwd = os.getcwd()
driving_log = cwd + '/myDrivingData/driving_log.csv'
print('Reading csv file', driving_log)
data = csv.reader(open(driving_log), delimiter=",",quotechar='|')
steering = []
print('Looping CSV')
for row in data:
    steering.append(row[3])
steering = np.asarray(steering, dtype=np.float32)

print("Neutral Steering Samples: ", steering[steering==0].size)
print("Negative(left) Steering Samples: ", steering[steering<0].size)
print("Positive(right) Steering Samples: ", steering[steering>0].size)
print("Total Steering Samples: ", steering.size)
tmp = os.system("find ./myDrivingData/IMG/ -name \"*\" | wc")
print(tmp)
