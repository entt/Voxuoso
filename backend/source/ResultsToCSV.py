import csv
from os.path import (
    join,
    abspath,
    dirname,
)

# Where the module is
module_directory = abspath(dirname(__file__))

# Directory of Praat AVQI results
results_dir = join(module_directory, "./data/AVQI Results.txt")

# Directory of output file for training
csv_dir = join(module_directory, "./data/Outputs.csv")

# Headers for CSV
fields = ['file_name', 'avqi_result']

# Open text file from Praat and write to CSV
with open(results_dir, 'r') as read, open(csv_dir, 'w') as write:
    writer = csv.DictWriter(write, fieldnames=fields)
    writer.writeheader()

    for line in read:
        file = line.split()[0]
        result = line.split()[1]
        print "Current file: " + file.split("/")[-1]
        writer.writerow({'file_name': file, 'avqi_result': result})
