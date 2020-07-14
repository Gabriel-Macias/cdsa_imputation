# Multivariate Time Series Imputation via Cross-Dimensional Self-Attention (CDSA)

Implementation in `Tensorflow 2.1.0` of the multivariate time-series imputation method described in Cross-Dimensional Self-Attention for Multivariate, Geo-tagged Time Series Imputation (https://arxiv.org/abs/1905.09904).

**NOTE:** This particular method is designed for continous time series where zero values are usually associated with missing values. However, if zero values are actually meaningful, as in the case of item prediction, this method might not be optimal.

The expected format of the input files are of `.csv` format, where each column corresponds to a time series to be processed. Additionally, it is possible to have additional informative columns at the beginning of the data files, i.e. timestamp id, notes, etc. 
These non-utilizable columns can be removed from the data processing with the `--skip_cols` argument when running the pre-processing (phase 1) and imputation steps (final phase).  

The first step to follow is to run the `make_train_sets.py` script by executing the following command:

```
python make_train_sets.py --f_path "<path_to_csv_data_file>" --skip_cols 1 --dest_path "<set_filename>" --missing_rate 0.15 \ 
                          --val_rate 0.20 --time_window 30
``` 

Where `--f_path` and `--dest_path` are the paths to the input csv file with the data and the output `.npz` file that will be generated after the pre-processing.
The script creates two `(-1, time_window, n_signals)` tensors from the input data of shape `(n_samples, n_signals)`. The first tensor is the `x` training data and the second one is a tensor of the same dimensions 
that has the ground truth for artificial missing values extracted from `x`. A third tensor of shape `(n_samples, n_signals)` is also generated in order to validate/test final imputation performance of the model during training and during the imputation phase.

The `--time_window` argument defines how many time steps of all time series we will use at the same time. From personal experience I recommend using, whenever possible around 50% of the available timestamps in order to allow the model to learn longer time-dependent patterns. 

Keep in mind that at some point your GPU should be able to allocate at least the `(batch, time_window, time_window)` and `(batch, n_signals, n_signals)` tensors in memory. These are the most memory-intensive parts of the computations described in the paper.

The percentage of missing values can be controlled with the `--missing_rate` argument, and the percentage of data to save for validation/testing can be controlled with the `--val_rate` argument.
Note that the percentage of data reserved for testing is given by `missing_rate * val_rate`.

To train the model run:

```
python run_training.py --f_path "<path_to_npz_dataset_file> (output of previous script)" --model_dim 32 --n_layers 8 --batch_size 64 --n_epochs 5 
``` 

As a recommendation on the parameters, based on previous experiments I recommend giving preference to the number of layers `--n_layers` over the embedding dimension. 
The model weights are automatically saved to the working directory with the details of the hyperparameters used in the configuration.

Finally, to create a `.csv` file with the imputed missing values run:

```
python run_imputation.py --csv_path "<path_to_csv_data_file> (original input)" --skip_cols 1 --out_path "<name_of_output_imputed_csv_file>" \
                         --set_path "<previos_npz_set_file>" --model_dim 128 --n_layers 8
``` 

Note that the different arguments in this command execution must match the previous steps for the script to run smoothly.

A quick dummy dataset can be found in the `example_dataset` directory. The whole script can be executed by using the `example_dataset/dummy_set.csv` file which contains the search volume for 5 types of food obtained from Google Trends over 199 months.
You can then corroborate your results with the `example_dataset/dummy_set_complete.csv` which contains no missing values. 
Note that performance might not be so good since there might not be a very strong dependence between the 5 signals in order to fully exploit the advantages of this approach.

Thanks for reading! 