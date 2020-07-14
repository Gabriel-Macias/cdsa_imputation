from models import *
import pandas as pd
import numpy as np
from optparse import OptionParser

parser = OptionParser()

parser.add_option("-f", "--csv_path", dest = "csv_path", help = "Path to data csv file with original data.")
parser.add_option("-c", "--skip_cols", dest = "skip_cols", type = int, default = 0, help = "Number of columns (from the left) to skip in csv file.")
parser.add_option("-o", "--out_path", dest = "out_path", default = "imputed_data.csv", help = "Name of output file with imputed data.")

parser.add_option("-s", "--set_path", dest = "set_path", default = "trainval_set.npz", help = "Path to npz file with created datasets.")
parser.add_option("-d", "--model_dim", dest = "model_dim", type = int, default = 128, help = "Embedding dimension for attention layers.")
parser.add_option("-l", "--n_layers", dest = "n_layers", type = int, default = 8, help = "Number of stacked encoder layers.")

(options, args) = parser.parse_args()

imp_set_file = options.set_path

# Recover training missing values
x = np.load(imp_set_file)["x"] + np.load(imp_set_file)["y_train"]
y_val = np.load(imp_set_file)["y_val"]

# Build the imputation model
model_dim = options.model_dim
n_layers = options.n_layers
n_heads = model_dim // 8 if model_dim > 8 else 1
dff = 4 * model_dim
time_dim = x.shape[1]
n_signals = x.shape[2]

model_weights = "imp_model_weights_D{}_L{}_H{}_DFF{}.hdf5".format(model_dim, n_layers, n_heads, dff)
imp_model = get_imp_model(model_dim, n_layers, n_heads, dff, time_dim, n_signals)
imp_model.load_weights(model_weights)

def impute_dataset(data, model, n_timesteps):

    t_window = data.shape[1]
    imp_data = np.zeros((n_timesteps, data.shape[2]), dtype = "float32")
    orig_data = np.zeros((n_timesteps, data.shape[2]), dtype = "float32")
    avg_arr = np.zeros((n_timesteps, 1))

    # Batch-wise loop
    for i in range(data.shape[0]):
        orig_data[i:(i + t_window), :] += data[i, :, :]
        imp_data[i:(i + t_window), :] += model.predict(data[i:(i + 1), :, :])[0, :, :, 0]
        avg_arr[i:(i + t_window)] += np.ones((t_window, 1))

    # Average over repeated predictions
    imp_data = imp_data / avg_arr
    orig_data = orig_data / avg_arr
    imp_data[orig_data != 0] = 0
    imp_data += orig_data

    return imp_data

# Estimation of reconstruction error
imp_data = impute_dataset(x, imp_model, n_timesteps = y_val.shape[0])
rmse_arr = (imp_data - y_val) ** 2
print("\nThe Estimated Imputation RMSE: {:2.4f}".format(np.sqrt(rmse_arr[y_val != 0].mean())))

# Final imputation (adding y_val to have the most information possible)
t_window = x.shape[1]
if t_window == y_val.shape[0]:
    x += np.expand_dims(y_val, axis = 0)
else:
    x += np.array([y_val[i:(i + t_window), :] for i in range(y_val.shape[0] - t_window + 1)])

imp_data = impute_dataset(x, imp_model, n_timesteps = y_val.shape[0])

# Impute the original file and save it
original_df = pd.read_csv(options.csv_path)
original_df.iloc[:, options.skip_cols:] = imp_data
original_df.replace(0, np.nan).to_csv(options.out_path, index = False)

print("\nFinished creating the final imputed file")