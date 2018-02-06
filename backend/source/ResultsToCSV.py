import csv
from os.path import (
    join,
    abspath,
    dirname,
)

results_dir = join(abspath(dirname(__file__)), 'data/AVQI Results.txt')
csv_dir = join(abspath(dirname(__file__)), 'data/Outputs.csv')

fields = ['file_name', 'avqi_result']

with open(results_dir, 'r') as read, open(csv_dir, 'w') as write:
    writer = csv.DictWriter(write, fieldnames=fields)
    writer.writeheader()

    for line in read:
        file = line.split()[0]
        result = line.split()[1]
        print 'Current file: ' + file.split('/')[-1]
        writer.writerow({'file_name': file, 'avqi_result': result})
