# VAE-control

Collect data using OSCAR simulator.

## Download the OSCAR Source Code

```
$ git clone https://github.com/awskhalil/oscar.git --recursive
```

Once you have the drive data in a CSV file:

You can test datasets creation using the file `test_create_datasets.py`.

- The `test_create_datasets.py` uses `drive_data_new_oscar.py` to read your csv file. 
- Then it used `process_data_vae.py` to preprocess the data based on your `config.yaml` file.
- after that it will use the `build_datasets()` function to create general train, valid, and test datasets, that are based on the whole available data in the CSV file.
- From these dataets you then create filtered datsets using the function `build_filtered_dataset()` with certain structure based on your application
  - Choose which columns to use from the dataset and which drop in your `config.yaml` file.
  - define inputs and outputs to your network.


