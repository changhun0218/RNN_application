import numpy as np
import os
from date_to_day import date_to_day as dd

cwd = os.getcwd()
data = np.load(cwd + "/RNN_data/processed_stock_data.npy")
date = dd(data[:,1])
data[:,1] = date

np.save(cwd + "/RNN_data/processed_stock_data_date", data)

output=data[data[:,1]>6570]

input_=data[data[:,1]<=6570]
np.save(cwd + "/RNN_data/input", input_)
np.save(cwd + "/RNN_data/output",output)
