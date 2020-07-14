import numpy as np
import pandas as pd
from optparse import OptionParser

parser = OptionParser()

parser.add_option("-f", "--f_path", dest = "f_path", help = "Path to data csv file with data.")
parser.add_option("-c", "--skip_cols", dest = "skip_cols", type = int, default = 0, help = "Number of columns (from the left) to skip in csv file.")
parser.add_option("-d", "--dest_path", dest = "dest_path", default = "trainval_set", help = "File name to store the created sets.")
parser.add_option("-m", "--missing_rate", dest = "missing_rate", type = float, default = 0.15, help = "Ratio of missing data to use in trainval sets.")
parser.add_option("-v", "--val_rate", dest = "val_rate", type = float, default = 0.20, help = "Proportion of validation data to use.")
parser.add_option("-t", "--time_window", dest = "time_window", type = int, default = 0, help = "Time window on which to make the imputation.")

(options, args) = parser.parse_args()

data = pd.read_csv(options.f_path)
data = data.iloc[:, options.skip_cols:].replace(np.nan, 0).values.astype("float32")

# Creating artificial missing values
y = np.zeros(data.shape, dtype = "float32")

idx_non_zero = np.where(data != 0)
n_non_zeros = idx_non_zero[0].shape[0]

sel_size = np.ceil(n_non_zeros * options.missing_rate).astype("int")
sel_t_idx = np.random.choice(range(n_non_zeros), sel_size, replace = False)

# Separate into train and validation
val_size = int(options.val_rate * sel_size)
if val_size > 0:
    y_val = np.zeros(data.shape, dtype = "float32")
    sel_v_idx = np.random.choice(sel_t_idx, val_size, replace = False)
    sel_t_idx = np.setdiff1d(sel_t_idx, sel_v_idx)

    sel_x_idx = idx_non_zero[0][sel_v_idx]
    sel_y_idx = idx_non_zero[1][sel_v_idx]
    y_val[sel_x_idx, sel_y_idx] = data[sel_x_idx, sel_y_idx]
    data[sel_x_idx, sel_y_idx] = 0
else:
    y_val = []

sel_x_idx = idx_non_zero[0][sel_t_idx]
sel_y_idx = idx_non_zero[1][sel_t_idx]
y[sel_x_idx, sel_y_idx] = data[sel_x_idx, sel_y_idx]
data[sel_x_idx, sel_y_idx] = 0

n_time_steps = data.shape[0]
t_window = options.time_window if options.time_window > 0 else n_time_steps

x = np.array([data[i:(i + t_window), :] for i in range(n_time_steps - t_window + 1)])
y_train = np.array([y[i:(i + t_window), :] for i in range(n_time_steps - t_window + 1)])

np.savez(options.dest_path + ".npz", x = x, y_train = y_train, y_val = y_val)
print("Finished created dataset npz file")