#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 
12/17/2023: Modified by Aws

"""


import pandas as pd
from progressbar import ProgressBar
import matplotlib.pyplot as plt
import numpy as np
import random

from config_oscar import Config


class DriveData:

    if Config.data_collection['brake'] is True:
        
        if Config.data_collection['heading_avail'] is True:
        
            csv_header = ['image_fname', 
                        'steering_angle', 'throttle', 'brake', 
                        'linux_time',
                        'vel', 'vel_x', 'vel_y', 'vel_z', 
                        'pos_x', 'pos_y', 'pos_z',
                        'imu_accel_x', 'imu_accel_y', 'imu_accel_z',
                        'yaw_rate_deg', 'heading_current',
                        'accel_calc',  'timestamp']
        else:
            
            csv_header = ['image_fname', 
                        'steering_angle', 'throttle', 'brake',
                        'linux_time',
                        'vel', 'vel_x', 'vel_y', 'vel_z', 
                        'pos_x', 'pos_y', 'pos_z',
                        'imu_accel_x', 'imu_accel_y', 'imu_accel_z',
                        'accel_calc',  'timestamp']

    else:
        
        if Config.data_collection['heading_avail'] is True:
        
            csv_header = ['image_fname', 
                        'steering_angle', 'throttle',
                        'linux_time',
                        'vel', 'vel_x', 'vel_y', 'vel_z', 
                        'pos_x', 'pos_y', 'pos_z',
                        'imu_accel_x', 'imu_accel_y', 'imu_accel_z',
                        'yaw_rate_deg', 'heading_current',
                        'accel_calc',  'timestamp']
        else:
            
            csv_header = ['image_fname', 
                        'steering_angle', 'throttle',
                        'linux_time',
                        'vel', 'vel_x', 'vel_y', 'vel_z', 
                        'pos_x', 'pos_y', 'pos_z',
                        'imu_accel_x', 'imu_accel_y', 'imu_accel_z',
                        'accel_calc',  'timestamp']
        

    def __init__(self, csv_fname):
        self.csv_fname = csv_fname
        self.df = None
        self.image_names = []
        self.actions = []
        self.linux_times = []
        self.velocities = []
        self.velocities_xyz = []
        self.positions_xyz = []
        self.imu_accelerations_xyz = []
        self.yaw_rates_and_headings = []
        self.calculated_accelerations = []
        self.time_stamps = []

    def read(self, read = True, show_statistics = True, normalize = True):
        
        print('*********************************************************************************************')
        print(self.csv_fname)
        print(self.csv_header)
        print('*********************************************************************************************')
        
        # Reduce Data Type Overhead: 
        # Explicitly specify data types for columns while reading the CSV. 
        # This reduces memory usage and improves processing speed.
        dtypes = {
        'image_fname': 'string',
        'steering_angle': 'float32',
        'throttle': 'float32',
        'brake': 'float32',
        'linux_time': 'float64',
        'vel': 'float32',
        'vel_x': 'float32',
        'vel_y': 'float32',
        'vel_z': 'float32',
        'pos_x': 'float32',
        'pos_y': 'float32',
        'pos_z': 'float32',
        'imu_accel_x': 'float32',
        'imu_accel_y': 'float32',
        'imu_accel_z': 'float32',
        'yaw_rate_deg': 'float32',
        'heading_current': 'float32',
        'accel_calc': 'float32',
        'timestamp': 'float64'}

        print(f"Reading CSV file: {self.csv_fname}")

        # Use chunksize in pandas.read_csv: 
        # Instead of loading the entire dataset into memory, load it 
        # in chunks and process each chunk. 
        # This reduces memory usage and speeds up the reading process.
        chunksize = 10000  # Adjust this based on available memory
        chunks = []

        # self.df = pd.read_csv(self.csv_fname, names=self.csv_header, index_col=False)
        # if Config.neural_net['accel']:
        #     self.df['vel'] = (self.df['vel']-self.df['vel'].min())/(self.df['vel'].max()-self.df['vel'].min())
        #     self.df['accel'] = (self.df['accel']-self.df['accel'].min())/(self.df['accel'].max()-self.df['accel'].min())
        #self.fname = fname
        
        # add a new colum which would have both throttle and brake values 
        # if Config.data_collection['brake'] is True:
        #     self.df['throttle_brake'] = self.df['throttle'] - self.df['brake']

        # Read the CSV file in chunks
        for chunk in pd.read_csv(self.csv_fname, names=self.csv_header, 
                                 dtype=dtypes, chunksize=chunksize, index_col=False):
            # Process throttle-brake column directly on chunk
            if Config.data_collection['brake']:
                chunk['throttle_brake'] = chunk['throttle'] - chunk['brake']
            chunks.append(chunk)
        
        # Concatenate all chunks into a single DataFrame
        self.df = pd.concat(chunks, ignore_index=True)


        ############################################
        # Show statistics
        if show_statistics:
            print("\n### Data Statistics ###")
            for col in ['steering_angle', 'throttle', 'brake', 'vel', 'accel_calc']:
                if col in self.df.columns:
                    print(f"{col} Statistics:")
                    print(self.df[col].describe())


        ############################################
        # normalize data
        # 'normalize' arg is for overriding 'normalize_data' config.
        if (Config.neural_net['normalize_data'] and normalize):
            print('\nhistogram-based normalizing... wait for a moment')
            num_bins = 50 # it was 50
            samples_per_bin = 200 # it was 200
            remove_list = []

            if Config.neural_net['num_outputs'] == 3:
                print('\nnum_outputs = 3')
                fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
                hist_steer, bins_steer = np.histogram(self.df['steering_angle'], num_bins)
                hist_throttle, bins_throttle = np.histogram(self.df['throttle'], num_bins)
                hist_brake, bins_brake = np.histogram(self.df['brake'], num_bins)
                hist_throttle_brake, bins_throttle_brake = np.histogram(self.df['throttle_brake'], num_bins)

                center_steer = (bins_steer[:-1] + bins_steer[1:]) * 0.5
                center_throttle = (bins_throttle[:-1] + bins_throttle[1:]) * 0.5
                center_brake = (bins_brake[:-1] + bins_brake[1:]) * 0.5
                center_throttle_brake = (bins_throttle_brake[:-1] + bins_throttle_brake[1:]) * 0.5

                ax1.bar(center_steer, hist_steer, width=0.05)
                ax1.set(title='original - steering')

                ax2.bar(center_throttle, hist_throttle, width=0.05)
                ax2.set(title='original - throttle')

                ax3.bar(center_brake, hist_brake, width=0.05)
                ax3.set(title='original - brake')

                ax4.bar(center_throttle_brake, hist_throttle_brake, width=0.05)
                ax4.set(title='original - throttle_brake')

                # for hist, bins, action_col in zip([hist_steer, hist_throttle, hist_brake],
                #                                 [bins_steer, bins_throttle, bins_brake],
                #                                 ['steering_angle', 'throttle', 'brake']):
                #     for j in range(num_bins):
                #         list_ = []
                #         for i in range(len(self.df[action_col])):
                #             # Check if all three actions are in the zero bin
                #             if all(bins[j] <= self.df.loc[i, col] <= bins[j + 1] for col in ['steering_angle', 'throttle', 'brake']):
                #                 list_.append(i)
                #             # Check if brake is zero and either steering or throttle is zero
                #             # elif bins[j] <= self.df.loc[i, 'brake'] <= bins[j + 1] and any(
                #             #         bins[j] <= self.df.loc[i, col] <= bins[j + 1] for col in ['steering_angle', 'throttle']):
                #             #     list_.append(i)
                #         random.shuffle(list_)
                #         list_ = list_[samples_per_bin:]
                #         remove_list.extend(list_)

                for hist, bins, action_col in zip([hist_steer, hist_throttle_brake],
                                                [bins_steer, bins_throttle_brake],
                                                ['steering_angle', 'throttle_brake']):
                    for j in range(num_bins):
                        list_ = []
                        for i in range(len(self.df[action_col])):
                            # Check if all three actions are in the zero bin
                            if all(bins[j] <= self.df.loc[i, col] <= bins[j + 1] for col in ['steering_angle', 'throttle_brake']):
                                list_.append(i)
                            # Check if brake is zero and either steering or throttle is zero
                            # elif bins[j] <= self.df.loc[i, 'brake'] <= bins[j + 1] and any(
                            #         bins[j] <= self.df.loc[i, col] <= bins[j + 1] for col in ['steering_angle', 'throttle']):
                            #     list_.append(i)
                        random.shuffle(list_)
                        list_ = list_[samples_per_bin:]
                        remove_list.extend(list_)

                print('\r####### Histogram-based data normalization #########')
                print('\n####### if the data point values fall in the zero bin it will be removed #########')
                print('removed:', len(remove_list))
                self.df.drop(self.df.index[remove_list], inplace=True)
                self.df.reset_index(inplace=True)
                self.df.drop(['index'], axis=1, inplace=True)
                print('remaining:', len(self.df))

                hist_steer, _ = np.histogram(self.df['steering_angle'], num_bins)
                # hist_throttle, _ = np.histogram(self.df['throttle'], num_bins)
                # hist_brake, _ = np.histogram(self.df['brake'], num_bins)
                hist_throttle_brake, _ = np.histogram(self.df['throttle_brake'], num_bins)

                ax5.bar(center_steer, hist_steer, width=0.05)
                # ax5.bar(center_throttle, hist_throttle, width=0.05, alpha=0.5)
                # ax5.bar(center_brake, hist_brake, width=0.05, alpha=0.5)
                ax5.bar(center_throttle_brake, hist_throttle_brake, width=0.05, alpha=0.5)

                ax5.plot(label='steering')
                # ax5.plot((np.min(self.df['steering_angle']), np.max(self.df['steering_angle'])),
                #         (samples_per_bin, samples_per_bin), label='steering')
                # ax5.plot((np.min(self.df['throttle']), np.max(self.df['throttle'])),
                #         (samples_per_bin, samples_per_bin), label='throttle')
                # ax5.plot((np.min(self.df['brake']), np.max(self.df['brake'])),
                #         (samples_per_bin, samples_per_bin), label='brake')
                # ax5.plot((np.min(self.df['throttle_brake']), np.max(self.df['throttle_brake'])),
                #         (samples_per_bin, samples_per_bin), label='throttle_brake')
                ax5.plot(label='throttle_brake')
                ax5.set(title='normalized')
                ax5.legend()

                # plt.tight_layout()
                plt.savefig(self.get_data_path() + '_normalized.png', dpi=150)
                plt.savefig(self.get_data_path() + '_normalized.pdf', dpi=150)

            else:
                fig, (ax1, ax2) = plt.subplots(1, 2)
                #fig.suptitle('Data Normalization')
                hist, bins = np.histogram(self.df['steering_angle'], (num_bins))
                center = (bins[:-1] + bins[1:])*0.5
                ax1.bar(center, hist, width=0.05)
                ax1.set(title = 'original')

                for j in range(num_bins):
                    list_ = []
                    for i in range(len(self.df['steering_angle'])):
                        if self.df.loc[i,'steering_angle'] >= bins[j] and self.df.loc[i,'steering_angle'] <= bins[j+1]:
                            list_.append(i)
                    random.shuffle(list_)
                    list_ = list_[samples_per_bin:]
                    remove_list.extend(list_)
                
                print('\r####### data normalization based on steering_angle #########')
                print('removed:', len(remove_list))
                self.df.drop(self.df.index[remove_list], inplace = True)
                self.df.reset_index(inplace = True)
                self.df.drop(['index'], axis = 1, inplace = True)
                print('remaining:', len(self.df))
                
                hist, _ = np.histogram(self.df['steering_angle'], (num_bins))
                ax2.bar(center, hist, width=0.05)
                ax2.plot((np.min(self.df['steering_angle']), np.max(self.df['steering_angle'])), 
                            (samples_per_bin, samples_per_bin))  
                ax2.set(title = 'normalized')          

                plt.tight_layout()
                plt.savefig(self.get_data_path() + '_normalized.png', dpi=150)
                plt.savefig(self.get_data_path() + '_normalized.pdf', dpi=150)
                #plt.show()


        ############################################ 
        # read out
        # if (read): 
        #     num_data = len(self.df)
            
        #     bar = ProgressBar()
            
        #     for i in bar(range(num_data)): # we don't have a title
        #     # for i in bar(range(10)):
        #         self.image_names.append(self.df.loc[i]['image_fname'])
        #         if Config.data_collection['brake'] is True:
        #             self.actions.append((float(self.df.loc[i]['steering_angle']),
        #                                     float(self.df.loc[i]['throttle']), 
        #                                     float(self.df.loc[i]['brake']),
        #                                     float(self.df.loc[i]['throttle_brake'])))
        #         else:
        #             self.actions.append((float(self.df.loc[i]['steering_angle']),
        #                                     float(self.df.loc[i]['throttle']), 
        #                                     0.0)) # dummy value for old data
        #         self.linux_times.append(float(self.df.loc[i]['linux_time']))
        #         self.velocities.append(float(self.df.loc[i]['vel']))
        #         self.velocities_xyz.append((float(self.df.loc[i]['vel_x']), 
        #                                     float(self.df.loc[i]['vel_y']), 
        #                                     float(self.df.loc[i]['vel_z'])))
        #         self.positions_xyz.append((float(self.df.loc[i]['pos_x']), 
        #                                     float(self.df.loc[i]['pos_y']), 
        #                                     float(self.df.loc[i]['pos_z'])))
        #         self.imu_accelerations_xyz.append((float(self.df.loc[i]['imu_accel_x']), 
        #                                     float(self.df.loc[i]['imu_accel_y']), 
        #                                     float(self.df.loc[i]['imu_accel_z'])))
        #         if Config.data_collection['heading_avail'] is True:
        #             self.yaw_rates_and_headings.append((float(self.df.loc[i]['yaw_rate_deg']), 
        #                                         float(self.df.loc[i]['heading_current'])))
        #         self.calculated_accelerations.append(float(self.df.loc[i]['accel_calc']))
        #         self.time_stamps.append(float(self.df.loc[i]['timestamp']))
        
        # Vectorized Operations:
        # Replace loops with vectorized operations where 
        # possible (e.g., column operations using pandas).
        # Optimize Progress Tracking: 
        # Instead of updating the progress bar in a loop (which is slow), 
        # use batch progress updates.
        # Efficient Data Assignment: 
        # Use tolist() or values.tolist() to quickly extract 
        # data into lists for attributes.

        # Populate class attributes if `read` is True
        if read:
            print("\nPopulating data attributes...")
            self.image_names = self.df['image_fname'].tolist()
            if Config.data_collection['brake'] is True:
                self.actions = self.df[['steering_angle', 'throttle', 'brake', 
                                        'throttle_brake']].values.tolist() 
            else:
                self.actions = self.df[['steering_angle', 'throttle']].values.tolist()
            self.linux_times = self.df['linux_time'].tolist()
            self.velocities = self.df['vel'].tolist()
            self.velocities_xyz = self.df[['vel_x', 'vel_y', 'vel_z']].values.tolist()
            self.positions_xyz = self.df[['pos_x', 'pos_y', 'pos_z']].values.tolist()
            self.imu_accelerations_xyz = self.df[['imu_accel_x', 'imu_accel_y', 'imu_accel_z']].values.tolist()
            if Config.data_collection['heading_avail']:
                self.yaw_rates_and_headings = self.df[['yaw_rate_deg', 'heading_current']].values.tolist()
            self.calculated_accelerations = self.df['accel_calc'].tolist()
            self.time_stamps = self.df['timestamp'].tolist()
    
    def get_data_path(self):
        loc_slash = self.csv_fname.rfind('/')
        
        if loc_slash != -1: # there is '/' in the data path
            data_path = self.csv_fname[:loc_slash] # get folder name
            return data_path
        else:
            exit('ERROR: csv file path must have a separator.')




###############################################################################
#  for testing DriveData class only
def main(data_path):
    import const

    if data_path[-1] == '/':
        data_path = data_path[:-1]

    loc_slash = data_path.rfind('/')
    if loc_slash != -1: # there is '/' in the data path
        model_name = data_path[loc_slash + 1:] # get folder name
        #model_name = model_name.strip('/')
    else:
        model_name = data_path
    
    csv_path = data_path + '/' + model_name + const.DATA_EXT   
    data = DriveData(csv_path)
    data.read(read = False)


###############################################################################
#       
if __name__ == '__main__':
    import sys

    try:
        if (len(sys.argv) != 2):
            exit('Usage:\n$ python {} data_path'.format(sys.argv[0]))

        main(sys.argv[1])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
