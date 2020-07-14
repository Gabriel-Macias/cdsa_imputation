from models import *
import numpy as np
from optparse import OptionParser
import time

parser = OptionParser()

parser.add_option("-f", "--f_path", dest = "f_path", default = "trainval_set.npz", help = "Path to npz file with created datasets.")
parser.add_option("-d", "--model_dim", dest = "model_dim", type = int, default = 128, help = "Embedding dimension for attention layers.")
parser.add_option("-l", "--n_layers", dest = "n_layers", type = int, default = 2, help = "Number of stacked encoder layers.")
parser.add_option("-b", "--batch_size", dest = "batch_size", type = int, default = 64, help = "Training batch size.")
parser.add_option("-e", "--n_epochs", dest = "n_epochs", type = int, default = 5, help = "Number of training epochs.")

(options, args) = parser.parse_args()

imp_set_file = options.f_path

x = np.load(imp_set_file)["x"]
y_train = np.load(imp_set_file)["y_train"]

# Build the imputation model
model_dim = options.model_dim
n_layers = options.n_layers
n_heads = model_dim // 8 if model_dim > 8 else 1
dff = 4 * model_dim
time_dim = x.shape[1]
n_signals = x.shape[2]

imp_model = get_imp_model(model_dim, n_layers, n_heads, dff, time_dim, n_signals)

mc = ModelCheckpoint("imp_model_weights_D{}_L{}_H{}_DFF{}.hdf5".format(model_dim, n_layers, n_heads, dff),
                     monitor = "val_loss", mode = "min", verbose = 1, save_best_only = True, save_weights_only = True)
#cbks = [mc, lr_scheduler(initial_lr = 1e-2)]
cbks = [mc]

b_size = np.min([options.batch_size, x.shape[0]])

if x.shape[0] == 1:
    y_val = np.expand_dims(np.load(imp_set_file)["y_val"], axis = 0)
    v_data = [x, y_val]
    val_prop = None
else:
    val_prop = 0.20 if int(0.20 * x.shape[0]) > 0 else 0
    v_data = None

s0 = time.time()
hist = imp_model.fit(x = x, y = y_train, validation_split = val_prop, validation_data = v_data,
                     callbacks = cbks, shuffle = True, verbose = 1, batch_size = b_size, epochs = options.n_epochs)

print("Training finished in {:2.2f} min.".format((time.time() - s0) / 60))