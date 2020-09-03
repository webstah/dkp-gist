import os
import time
import csv

class CSVLogger:
    def __init__(self, fieldnames, args):
        datetime = time.strftime('%y%m%d_%H%M%S')
        self.fieldnames = fieldnames

        if not os.path.exists(os.path.relpath(os.path.join('results', args.train_mode))):
            os.makedirs(os.path.relpath(os.path.join('results', args.train_mode)))
        self.filename = os.path.relpath(os.path.join("results", args.train_mode, "{}.csv".format(datetime)))

        with open(self.filename, 'a', newline='') as csvfile:
            csvfile.write(str(args) + '\n')
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()

    def save_values(self, *values):
        assert len(values) == len(self.fieldnames), 'The number of values should match the number of field names.'
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            row = {}
            for i, val in enumerate(values):
                row[self.fieldnames[i]] = val

            writer.writerow(row)
