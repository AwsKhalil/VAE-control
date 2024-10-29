#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 19:06:41 2022

@author: aws
"""
import tensorflow as tf
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from drive_data_new_oscar import DriveData
from image_process import ImageProcess
import gpu_options
import yaml
from tqdm import tqdm
import pandas as pd


config_path = '/home/aws/latency_mitigaion_vae/2024/config/'
with open(config_path+'config.yaml') as file:
# with open('config.yaml') as file:
    global config
    config = yaml.load(file, Loader=yaml.FullLoader)

gpu_options.set()

class ProcessData:
    
    ###########################################################################
    # data_path = 'path_to_drive_data'  e.g. ../data/2017-09-22-10-12-34-56/'
    def __init__(self, data_path, balance_data=False):
                    
        if data_path[-1] == '/':
            data_path = data_path[:-1]

        self.data_path = data_path

        loc_slash = data_path.rfind('/')
        if loc_slash != -1: # there is '/' in the data path
            model_name = self.data_path[loc_slash + 1:] # get folder name
            #model_name = model_name.strip('/')
        else:
            model_name = self.data_path

        self.csv_path = self.data_path + '/' + model_name + '.csv'  # use it for csv file name 
        # print(self.csv_path)
        # csv_path = data_path + '/2022-05-18-11-05-24_img_shift.csv'

        self.data = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.max_velocity = None
        
        self.data = DriveData(self.csv_path)
        # self.image_process = ImageProcess()  
        
        self._prepare_data(balance_data=balance_data)
        # self._build_datasets()              
        
    ###########################################################################
    #
    def _prepare_data(self, balance_data=False):
        # Load your data and split it into train, valid, and test sets
        self.data.read(normalize=False)

        # Convert to NumPy array for easy slicing
        self.data.yaw_rates_and_headings = np.array(self.data.yaw_rates_and_headings)

        # Compute the maximum velocity from the dataset
        self.max_velocity = max(map(float, self.data.velocities))
        # Normalize velocities by dividing by the maximum velocity
        normalized_velocities = list(map(lambda v: float(v) / self.max_velocity, self.data.velocities))

        samples = list(zip(self.data.image_names, 
                    normalized_velocities, 
                    map(float, self.data.yaw_rates_and_headings[:, 0]),  # yaw rates
                    map(float, self.data.yaw_rates_and_headings[:, 1]),  # headings
                    map(float, np.array(self.data.actions)[:, 0]),  # steering angle
                    map(float, np.array(self.data.actions)[:, 3])))  # throttle/brake
        
        if config['in_prev_steer']:
            samples = self.add_previous_steering(samples)

        if balance_data:
            print("before balancing ...\n")
            print("length:\n", len(samples))
            print("type of all samples:\n", type(samples))
            print("type of one sample in samples:\n", type(samples[0]))
            print("1st 5 rows:\n", samples[:5])
            samples = self.steering_histogram_based_data_balancing(samples)
            print("After balancing ...\n")
            print("length:\n", len(samples))
            print("type of all samples:\n", type(samples))
            print("type of one sample in samples:\n", type(samples[0]))
            print("1st 5 rows:\n", samples[:5])        
        
        self.train_data, self.test_data = train_test_split(samples, test_size=config['test_rate'])
        self.train_data, self.valid_data = train_test_split(self.train_data, test_size=config['validation_rate'])
        
        print('Samples shape', np.shape(np.array(samples)))
        print('Train samples:', len(self.train_data))
        print('Valid samples:', len(self.valid_data))
        print('Test samples:', len(self.test_data))

    ###########################################################################
    # add a new colum of previous steering angle so we can use it as input
    def add_previous_steering(self, samples):
        """
        Adds the previous steering angle to each data row.
        Sets the previous steering for the first row to zero.
        example of np.roll
        x = np.array([1,2,3,4])
        x2= np.roll(x,shift=1) --> output: [4 1 2 3]
        x2[0] = 0 --> x2 = [0 1 2 3]
        """
        # Convert samples to a DataFrame for easier manipulation
        df_samples = pd.DataFrame(samples, 
                columns=['img', 'velocity', 'yaw_rate', 'heading', 'steering', 'thr_brk'])

        # Generate the previous steering array
        # Efficiently create an array where each element is shifted one index forward
        prev_steering = np.roll(df_samples['steering'].to_numpy(), shift=1)
        prev_steering[0] = 0  # Set the first element to zero, as there is no "previous" for it

        # Add the `prev_steering` as a new column before 'steering'
        df_samples.insert(df_samples.columns.get_loc('steering'), 
                          'prev_steering', prev_steering)

        # Convert DataFrame back to list of tuples
        samples_w_prev_steer = [tuple(row) for row in df_samples.to_records(index=False)]

        return samples_w_prev_steer
        

    ###########################################################################
    # Balancing function based on steering angle histogram
    def steering_histogram_based_data_balancing(self, samples, num_bins=60, samples_per_bin=300):
        """
        Balance dataset based on steering angle histogram with progress display and visualization.

        Parameters:
        - samples: List of tuples containing (image_path, velocity, yaw_rate, heading, steering, thr_brk)
        - bins: Number of histogram bins for steering angle
        - max_per_bin: Maximum number of samples allowed per bin

        Returns:
        - balanced_samples: List of balanced samples with limited data per bin
        """
        # Convert samples to DataFrame for easier manipulation and create steering angles array
        if config['in_prev_steer']:
            df_samples = pd.DataFrame(samples, 
                columns=['img', 'velocity', 'yaw_rate', 'heading', 
                     'prev_steering', 'steering', 'thr_brk'])
        else:
            df_samples = pd.DataFrame(samples, 
                columns=['img', 'velocity', 'yaw_rate', 'heading', 
                     'steering', 'thr_brk'])
        
        steering_angles = df_samples['steering'].to_numpy()
        
        # Generate initial histogram
        hist, bins = np.histogram(steering_angles, num_bins)
        center = (bins[:-1] + bins[1:]) * 0.5

        # Plot original histogram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.bar(center, hist, width=0.05, color='blue', alpha=0.7)
        ax1.set_title('Original Steering Distribution')

        # Balancing process
        remove_indices = []  # To track indices to remove

        print("Balancing data based on steering angle bins...")
        for j in tqdm(range(num_bins), desc="Bins Processed", unit="bin"):
            # Select indices for samples within current bin
            bin_mask = (steering_angles >= bins[j]) & (steering_angles < bins[j + 1])
            bin_indices = np.where(bin_mask)[0]
            
            # Shuffle and prune samples in this bin if exceeding samples_per_bin
            if len(bin_indices) > samples_per_bin:
                np.random.shuffle(bin_indices)
                remove_indices.extend(bin_indices[samples_per_bin:])
        
        print("Total samples removed:", len(remove_indices))
        
        # Remove overrepresented samples
        df_samples.drop(index=remove_indices, inplace=True)
        df_samples.reset_index(drop=True, inplace=True)

        # Generate new balanced histogram
        balanced_steering_angles = df_samples['steering'].to_numpy()
        balanced_hist, _ = np.histogram(balanced_steering_angles, bins)
        ax2.bar(center, balanced_hist, width=0.05, color='green', alpha=0.7)
        ax2.plot([np.min(steering_angles), np.max(steering_angles)], [samples_per_bin, samples_per_bin], 'r--')
        ax2.set_title('Balanced Steering Distribution')
        
        plt.tight_layout()
        # plt.savefig(self.get_data_path() + '_balanced.png', dpi=150)
        # plt.savefig(self.get_data_path() + '_balanced.pdf', dpi=150)
        plt.show()

        # Convert balanced DataFrame back to list of tuples for further processing
        balanced_samples = df_samples.to_records(index=False)
        
        # After balancing, convert each numpy.record to a tuple
        balanced_samples = [tuple(sample) for sample in balanced_samples]

        return list(balanced_samples)
    

    ###########################################################################
    #
    def _preprocess_image(self, img_path):
        """Preprocess the image: resize, crop, normalize."""
        img = cv2.imread(self.data_path + '/' + img_path.decode())
        # Convert to grayscale if input_image_depth is 1, otherwise keep it as RGB
        if config['input_image_depth'] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif config['input_image_depth'] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format
        
        # Crop, resize, and normalize the image
        img = img[config['image_crop_y1']:config['image_crop_y2'], config['image_crop_x1']:config['image_crop_x2']]
        img = cv2.resize(img, (config['input_image_width'], config['input_image_height']))
        img = img / 255.0  # Normalize to [0, 1]

        # If grayscale, add an extra dimension for consistency with input shape requirements
        if config['input_image_depth'] == 1:
            img = np.expand_dims(img, axis=-1)

        return img.astype(np.float32)  # Cast to float32
    ###########################################################################
    #
    # def _parse_function(self, sample):
    #     """Convert a sample (image path, velocity, etc.) into tensors."""
    #     img_path, velocity, yaw_rate, heading, steering_ang, thr_brk = sample
    #     img = tf.numpy_function(self._preprocess_image, [img_path], tf.float32)
    #     img = tf.reshape(img, [config['input_image_height'], config['input_image_width'], config['input_image_depth']])
    #     return (img, velocity, yaw_rate, heading), (steering_ang, thr_brk)

    def _parse_function(self, image_path, numerical_data):
        """Preprocess the image and return all available data."""
        img = tf.numpy_function(self._preprocess_image, [image_path], tf.float32)
        img = tf.reshape(img, [config['input_image_height'], config['input_image_width'], config['input_image_depth']])

        # Cast numerical data to float32 if necessary
        numerical_data = tf.cast(numerical_data, tf.float32)
        
        # prev_steer_avail = config['in_prev_steer']
        # if prev_steer_avail:
        #     velocity, yaw_rate, heading, prev_steering, steering, thr_brk = numerical_data
        #     # Always return all available data and assume we are reconstructing the input image.
        #     return (img, velocity, yaw_rate, heading, prev_steering), (img, steering, thr_brk)
        # else:
        #     velocity, yaw_rate, heading, steering, thr_brk = numerical_data
        #     # Always return all available data and assume we are reconstructing the input image.
        #     return (img, velocity, yaw_rate, heading), (img, steering, thr_brk)
        
        # Access numerical values by indexing to avoid unpacking (unpacking causes an error)
        if config['in_prev_steer']:
            velocity = numerical_data[0]
            yaw_rate = numerical_data[1]
            heading = numerical_data[2]
            prev_steering = numerical_data[3]
            steering = numerical_data[4]
            thr_brk = numerical_data[5]

            # Return structured data with previous steering angle included
            return (img, velocity, yaw_rate, heading, prev_steering), (img, steering, thr_brk)
        else:
            velocity = numerical_data[0]
            yaw_rate = numerical_data[1]
            heading = numerical_data[2]
            steering = numerical_data[3]
            thr_brk = numerical_data[4]

            # Return structured data without previous steering angle
            return (img, velocity, yaw_rate, heading), (img, steering, thr_brk)
    
    ###########################################################################
    # Data augmentation function
    def _augment_function(self, inputs, outputs):
        """
        Apply data augmentation: 
        flip image, reverse steering, yaw rate, and heading.
        """
        prev_steer_avail = config['in_prev_steer']
        if prev_steer_avail:
            (img, velocity, yaw_rate, heading, prev_steering), (img_target, steering, thr_brk) = inputs, outputs
            prev_steering = -prev_steering
        else:
            (img, velocity, yaw_rate, heading), (img_target, steering, thr_brk) = inputs, outputs

        # Flip image horizontally
        img = tf.image.flip_left_right(img)
        img_target = tf.image.flip_left_right(img_target)

        # Reverse the sign of steering, yaw rate, and heading
        steering = -steering
        yaw_rate = -yaw_rate
        heading = -heading
        if prev_steer_avail:
            return (img, velocity, yaw_rate, heading, prev_steering), (img_target, steering, thr_brk)
        else:
            return (img, velocity, yaw_rate, heading), (img_target, steering, thr_brk)
    ###########################################################################    
    #
    def _build_dataset(self, samples, data_balance=False, augment=False):

         # Confirm samples structure is consistent
        print("Sample structure preview:", samples[:5])
        for idx, sample in enumerate(samples[:5]):
            if not isinstance(sample, (list, tuple)) or len(sample) < 2:
                raise ValueError(f"Sample at index {idx} is not structured as expected: {sample}")
        

        image_paths = [sample[0] for sample in samples]
        numerical_data = [sample[1:] for sample in samples]

        print(np.shape(np.array(image_paths)))
        print(np.shape(np.array(numerical_data)))

        image_ds = tf.data.Dataset.from_tensor_slices(image_paths)
        numerical_ds = tf.data.Dataset.from_tensor_slices(numerical_data)
        
        dataset = tf.data.Dataset.zip((image_ds, numerical_ds))

        # Apply preprocessing
        dataset = dataset.map(self._parse_function, num_parallel_calls=tf.data.AUTOTUNE)

        # Apply augmentation if specified
        if augment:
            augmented_ds = dataset.map(self._augment_function, num_parallel_calls=tf.data.AUTOTUNE)
            # Concatenate original and augmented datasets
            dataset = dataset.concatenate(augmented_ds)
        
        dataset = dataset.batch(config['batch_size']).prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset
    ###########################################################################
    #
    def build_datasets(self, augment=False):
        print("Please wait while creating datasets ... ")

        # Build train dataset with optional augmentation
        self.train_dataset = self._build_dataset(self.train_data, augment=augment)
        print('*** Train Dataset is created ***')
        if augment:
            print('Train Dataset length is {}'.format(len(self.train_data)*2))
        else:
            print('Train Dataset length is {}'.format(len(self.train_data)))
        # # you can do it on the train_dataset but it takes a lot of time:
        # train_dataset_size = sum(1 for _ in self.train_dataset) * config['batch_size']  # Calculate total samples
        # print('Train Dataset length (total samples): {}'.format(train_dataset_size))

        # Build validation dataset
        self.valid_dataset = self._build_dataset(self.valid_data)
        print('*** Validation Dataset is created ***')
        print('Validation Dataset shape is {}'.format(len(self.valid_data)))
        # # you can do it on the train_dataset but it takes a lot of time:
        # valid_dataset_size = sum(1 for _ in self.valid_dataset) * config['batch_size']  # Calculate total samples
        # print('Validation Dataset length (total samples): {}'.format(valid_dataset_size))
        
        # Build test dataset
        self.test_dataset = self._build_dataset(self.test_data)
        print('*** Test Dataset is created ***')
        print('Test Dataset shape is {}'.format(len(self.test_data)))
        # # you can do it on the train_dataset but it takes a lot of time:
        # test_dataset_size = sum(1 for _ in self.test_dataset) * config['batch_size']  # Calculate total samples
        # print('Test Dataset length (total samples): {}'.format(test_dataset_size))
        
            
            
def check_data(processed_data):
    """
    Check data is used to plot some images from
    the training and testing datasets.
    """
    train_data = processed_data.train_dataset
    test_data = processed_data.test_dataset
    
    # for 1 input only (img_input)
    x_train = train_data
    # y_train = train_data[-1]
    x_test = test_data
    # y_test = test_data[-1]
    
    image_process = ImageProcess()

    # Accessing a batch from train and test dataset
    for x_train, y_train in train_data.take(1):
        img_train = x_train[0].numpy()  # Separate image data (first element in x_train tuple)
        # Numerical data: x_train[1] contains the numerical values (e.g., velocity, yaw_rate, etc.)
        train_batch_size = img_train.shape[0]
    
    for x_test, y_test in train_data.take(1):
        img_test = x_test[0].numpy()  # Separate image data (first element in x_test tuple)
        # Numerical data: x_test[1] contains the numerical values
        test_batch_size = img_test.shape[0]

    # Normalize images using ImageProcess class
    img_train = image_process.norm_0_1(img_train)
    img_test = image_process.norm_0_1(img_test)

    # Randomly select indices from the batch
    train_indices = np.random.randint(0, train_batch_size, 4)
    test_indices = np.random.randint(0, test_batch_size, 4)

    # View a few random images from the training dataset
    plt.figure(1)
    for i, idx in enumerate(train_indices):
        plt.subplot(2, 2, i+1)
        plt.imshow(img_train[idx][:,:,0], cmap='gray')
    plt.show()

    # View a few random images from the testing dataset
    plt.figure(2)
    for i, idx in enumerate(test_indices):
        plt.subplot(2, 2, i+1)
        plt.imshow(img_test[idx][:,:,0], cmap='gray')
    plt.show()

def collect_dataset(dataset, name="Collecting data"):
    """
    collect_dataset is used to save each value separately
    mainly for plotting purposed, such as:
    plotting velocity, yaw, heading, steering against time
    """
    prev_steer_avail = config['in_prev_steer']
    images = []
    velocities = []
    yaw_rates = []
    headings = []
    if prev_steer_avail:
        prev_steerings = []
    steerings = []
    throttle_brakes = []

    # Iterate through the dataset to collect all batches
    # Use tqdm for progress bar during iteration
    for input_data, output_data in tqdm(dataset, desc=name, unit="batch"):
    # for input_data, output_data in dataset:
        # Unpack input data (img, velocity, yaw rate, heading)
        if prev_steer_avail:
            img, velocity, yaw_rate, heading, prev_steer = input_data
        else:
            img, velocity, yaw_rate, heading = input_data

        # Unpack output data (steering, throttle/brake)
        # we do not care about returning the image because the collect_dataset
        # function return each value separately and not used for training
        _ , steering, thr_brk = output_data 

        # Append all data to respective lists
        images.append(img.numpy())
        velocities.append(velocity.numpy())
        yaw_rates.append(yaw_rate.numpy())
        headings.append(heading.numpy())
        if prev_steer_avail:
            prev_steerings.append(prev_steer.numpy())
        steerings.append(steering.numpy())
        throttle_brakes.append(thr_brk.numpy())

    # Concatenate the list of NumPy arrays into a single array for each type of data
    images = np.concatenate(images, axis=0)
    velocities = np.concatenate(velocities, axis=0)
    yaw_rates = np.concatenate(yaw_rates, axis=0)
    headings = np.concatenate(headings, axis=0)
    steerings = np.concatenate(steerings, axis=0)
    throttle_brakes = np.concatenate(throttle_brakes, axis=0)
    if prev_steer_avail:
        prev_steerings = np.concatenate(prev_steerings, axis=0)
        return images, velocities, yaw_rates, headings, prev_steerings, steerings, throttle_brakes
    else:
        return images, velocities, yaw_rates, headings, steerings, throttle_brakes


def build_filtered_dataset(dataset):
    """
    build_filtered_dataset is pick and choose what are the inputs
    and outputs of your model when training.
    Unlike the original dataset, this will filter the inputs 
    and outputs based on your config file.
    """
    def filter_data(input_data, output_data):
        # Unpack input data based on config
        if config['in_prev_steer']:
            img, velocity, yaw_rate, heading, prev_steer = input_data
            # Reshape prev_steer to (None, 1) if it’s being used as input
            prev_steer = tf.reshape(prev_steer, (-1, 1))
        else:
            img, velocity, yaw_rate, heading = input_data
            
        reconstructed_img, steering, thr_brk = output_data

        # Reshape velocity to (None, 1) if required by the model
        velocity = tf.reshape(velocity, (-1, 1))

        # Apply input filtering based on config
        filtered_input = []
        if config['in_img']:
            filtered_input.append(img)
        if config['in_vel']:
            filtered_input.append(velocity)
        if config['in_yaw']:
            filtered_input.append(yaw_rate)
        if config['in_head']:
            filtered_input.append(heading)
        if config['in_prev_steer']:
            filtered_input.append(prev_steer)

        # Apply output filtering based on config
        filtered_output = []
        if config['out_img']:
            filtered_output.append(reconstructed_img)
        if config['out_steer']:
            filtered_output.append(steering)
        if config['out_thr_brk']:
            filtered_output.append(thr_brk)

        return tuple(filtered_input), tuple(filtered_output)

    # Use map to apply the filtering within the TF data pipeline
    filtered_dataset = dataset.map(filter_data, num_parallel_calls=tf.data.AUTOTUNE)
    return filtered_dataset


if __name__ == '__main__':
    
    data_path = config['data_path']
    try:
        processed_data = ProcessData(data_path)
        if config['check_datasets'] == True:
            check_data(processed_data)
    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
