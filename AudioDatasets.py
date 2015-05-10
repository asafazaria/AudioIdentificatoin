'''
Created on May 1, 2015

@author: Butzik
'''


import os
import cPickle
import numpy as np
import logging

from scipy.fftpack import rfft
from sklearn import decomposition
from sklearn import preprocessing

from AudioFilesPreprocessor import AudioFilesPreprocessor
from sklearn.decomposition.incremental_pca import IncrementalPCA

import scipy.signal as scisig


# TODO list:
# 1. Figure out if we want to strip silence also from the end of the files
# 2. Change the silence measurement to dB rather than percentage
# 3. Normalize the sound file by taking the Z-score? (In Preprocessor)
# 4. Implement the extraction of the class labels
# 5. Look at the results from the PCA with None reduction to pick the dimensionality to reduce to
# 6. Try to use magnitude and phase before doing PCA rather than real and imaginary parts
# 7. Make it more efficient so it will not have to do the DFT several times.


# Metal - 44100
# C - half and half
# Keys - 96000  

class Datasets_Manager(object):
    def __init__(self):
        pass
                    
    def load_learning_dataset_stepwise(self, base_path, standardize=True):    
        self.x, self.y_loc, self.y_obj = self.stepwise_load_and_reduce_dataset(base_path)
        if standardize: 
            self.scaler = self.standardize_dataset()
        
    def load_learning_dataset_at_once(self, base_path, standardize_before_reduction=False, standardize=True):
        self.x, self.y_loc, self.y_obj = self.load_signals_dataset(base_path)
        if standardize_before_reduction: self.standardize_dataset()
        reduced_dataset = self.transform_and_reduce_dataset(self.x)
        del self.x
        self.x = reduced_dataset
        if standardize: 
            self.scaler = self.standardize_dataset()
            
    def load_test_set_at_once(self, base_path, standardize=True):
        self.x_test, self.y_loc_test, self.y_obj_test = self.load_signals_dataset(base_path)
        self.x_test = self.transform_and_reduce_dataset(self.x_test)
        if standardize:
            self.x_test = self.scaler.transform(self.x_test) 
        
    def standardize_dataset(self):
        scaler = preprocessing.StandardScaler().fit(self.x)
        self.x = scaler.transform(self.x)
        return scaler
        
    def stepwise_load_signal(self, file_path):
        signal = np.load(file_path)
        padded_signal = np.pad(signal,(0,self.maximal_signal_length-len(signal)),mode="constant",constant_values=(0,))
        return padded_signal
    
    def learn_reduction_for_dataset(self, base_path):
        logging.error("Datasets_Manager: learning reduction from %s" % base_path)
        data_file_names = [file_name for file_name in os.listdir(base_path)
                           if (os.path.isfile(os.path.join(base_path,file_name))) and ("DS" not in file_name)]
        for file_name in data_file_names:
            loaded_signal = self.stepwise_load_signal(os.path.join(base_path,file_name))
            self.stepwise_learn_reduction(loaded_signal)
            del loaded_signal
    
    def get_maximal_signal_length(self, base_path):    
        data_file_names = [file_name for file_name in os.listdir(base_path)
                           if (os.path.isfile(os.path.join(base_path,file_name))) and ("DS" not in file_name)]
        self.maximal_signal_length = 0
        for file_name in data_file_names:
            loaded_signal = self.stepwise_load_signal(os.path.join(base_path,file_name))
            self.maximal_signal_length = max(self.maximal_signal_length, len(loaded_signal))
            del loaded_signal

    def transform_dataset_according_to_learnt_reduction(self, base_path):
        logging.error("Datasets_Manager: transforming %s according to reduction" % base_path)
        data_file_names = [file_name for file_name in os.listdir(base_path)
                           if (os.path.isfile(os.path.join(base_path,file_name))) and ("DS" not in file_name)]
        x = np.empty((0,self.reduced_dimensionality), dtype=np.float32)
        y_loc = np.empty((0,1), dtype=int)
        y_obj = np.empty((0,1), dtype=int)
        for file_name in data_file_names:
            loaded_signal = self.stepwise_load_signal(os.path.join(base_path,file_name))
            reduced_signal = self.stepwise_reduce_signal(loaded_signal)
            x = np.append(x,reduced_signal, axis=0)
            y_loc = np.append(y_loc, self.get_location_label_from_filename(file_name), axis=0)
            y_obj = np.append(y_obj, self.get_object_label_from_filename(file_name), axis=0)
            del loaded_signal
            del reduced_signal
        return x, y_loc, y_obj

    def stepwise_load_and_reduce_dataset(self, base_path):
        logging.error("Datasets_Manager: loading dataset step by step from %s" % base_path)
        self.learn_reduction_for_dataset(base_path)
        return self.transform_dataset_according_to_learnt_reduction(base_path)
        
            
    def load_signals_dataset(self, base_path):
        logging.error("Datasets_Manager: loading dataset from %s" % base_path)
        data_file_names = [file_name for file_name in os.listdir(base_path)
                           if (os.path.isfile(os.path.join(base_path,file_name))) and ("DS" not in file_name)]
        np_arrays_dataset = [np.load(os.path.join(base_path,file_name)) for file_name in data_file_names]
        signal_lengths = [len(np_array) for np_array in np_arrays_dataset]
        self.maximal_signal_length = max(signal_lengths)
        x = np.array([np.pad(signal,(0,self.maximal_signal_length-len(signal)),mode="constant",constant_values=(0,)).tolist() for signal in np_arrays_dataset])
        
        y_loc = np.array([self.get_location_label_from_filename(file_name) for file_name in data_file_names])
        y_obj = np.array([self.get_object_label_from_filename(file_name) for file_name in data_file_names])
        logging.info("Datasets_Manager: dataset loaded")
        return x, y_loc, y_obj
    
    def get_object_label_from_filename(self, file_name):
        objects_mapping = {
                           "C":0,
                           "K":1,
                           "M":2,
                           "S":3,
                           }
        return objects_mapping[file_name[0]]
    
    def get_location_label_from_filename(self, file_name):
        rows = ["A","B","C","D","E","F"]        
        label = rows.index(file_name[1])*6 + int(file_name[2])
        return label
         
    def random_select(self, x, y_loc, y_obj, selection_size):
        number_of_selection_samples = int(selection_size * len(x))  
        selected_indices = np.random.choice(len(x), number_of_selection_samples, replace=False)
        rest_indices = np.array([index for index in range(0,len(x)) if index not in selected_indices])
        
        rest_x = x[rest_indices]
        rest_y_loc = y_loc[rest_indices]
        rest_y_obj = y_obj[rest_indices]
        selected_x = x[selected_indices]
        selected_y_loc = y_loc[selected_indices]
        selected_y_obj = y_obj[selected_indices]
        
        return rest_x, rest_y_loc, rest_y_obj, selected_x, selected_y_loc, selected_y_obj
                
    def genereate_train_and_validate_from_learning_dataset(self, validation_sample_size):
        self.validation_sample_size = validation_sample_size
        self.x_train, self.y_loc_train, self.y_obj_train, self.x_validate, self.y_loc_validate, self.y_obj_validate = self.random_select(self.x, self.y_loc, self.y_obj, self.validation_sample_size)
        
    def save(self, to_file):
        cPickle.dump(self, file(to_file, "wb"))
        
    @classmethod
    def loader(cls, from_file):
        return cPickle.load(file(from_file,"rb"))

    

class Envelope_DimReduction(Datasets_Manager):
    cutoff_freq = 500
    fs = 44100
    filter_order = 5
    
    def __init__(self, reduced_dimensionality, recording_configuration):
        self.reduced_dimentionality = reduced_dimensionality
        self.recording_conf = recording_configuration
    
    def compute_cutoff_and_downsampling_from_target_dim(self):
        self.sampling_frequency = self.recording_conf["sample_rate"]
        self.downsampling_factor = self.maximal_signal_length/self.reduced_dimentionality
        self.cutoff_freq = (self.sampling_frequency/self.downsampling_factor) * 0.5 
        # This would be the maximal frequency we can capture well with the sampling rate. 
        # As we are using the ABS value of the signal, the result would be following the envelope of the signal
        
    def load_learning_dataset(self, base_path, **kw):
        self.load_learning_dataset_at_once(base_path, **kw)

    def generate_butter_lowpass_params(self, cutoff_freq, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff_freq / nyq
        b, a = scisig.butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def filter_signal_to_get_envelope(self, signal, cutoff_freq, fs, order):
        b, a = self.generate_butter_lowpass_params(cutoff_freq, fs, order=order)
        envelope = scisig.lfilter(b, a, signal)
        return envelope
    
    def transform_and_reduce_signal(self, signal, standardize=True):
        # Note that this function assumes that all signals are padded to the maximal length
        envelope = self.filter_signal_to_get_envelope(np.abs(signal), self.cutoff_freq, self.sampling_frequency, self.filter_order)
        downsampled_envelope = scisig.resample(envelope, (len(envelope)/self.downsampling_factor))
        if standardize:
            downsampled_envelope = self.scaler.transform(downsampled_envelope)
        
        np.save("./temp",downsampled_envelope)
        return downsampled_envelope
    
    def transform_and_reduce_dataset(self, time_domain_dataset):
        # Note that this function assumes it is called by load_signals_dataset. Therefore all signals are padded already to the maximal size.
        self.compute_cutoff_and_downsampling_from_target_dim()
        reduced_dataset = np.array([self.transform_and_reduce_signal(signal, standardize=False) for signal in time_domain_dataset])
        return reduced_dataset
        
    
class DSManager_DFTDimReduction(Datasets_Manager):
        
    def  __init__(self, reduced_dimensionality, recording_configuration):
        # Note: this function assumes the time_domain_dataset to be a 2D np.array where each row is a signal. See DatasetManager as an example to how it is done.
        self.reduced_dimentionality = reduced_dimensionality

    def load_learning_dataset(self, base_path, **kw):
        self.load_learning_dataset_at_once(base_path, **kw)
    
    def seperate_real_and_imaginary_parts(self, signal):
        expanded_signal = np.array(np.append(np.real(signal), np.imag(signal)))
        return expanded_signal

    def transform_and_reduce_dataset(self, time_domain_dataset):
        # This is only to make sure that all signals are of the same length, otherwise we will not have the same granularity in the frequency domain.
        signal_lengths = [len(signal) for signal in time_domain_dataset]
        assert min(signal_lengths) == max(signal_lengths)
        self.signals_length = max(signal_lengths)
                
        freq_domain_dataset = np.array([self.seperate_real_and_imaginary_parts(rfft(signal)) for signal in time_domain_dataset])
        self.pca = decomposition.PCA(n_components=self.reduced_dimentionality)
        logging.info("DSManager_DFTDimReduction: fitting PCA to frequency domain dataset to reduce dimensionality")
        self.pca.fit(freq_domain_dataset)
        reduced_freq_domain_dataset = self.pca.transform(freq_domain_dataset)
        logging.info("DFTDimReduction: PCA was fitted and the dimensionality reduced")
        return reduced_freq_domain_dataset
        
    def transform_and_reduce_signal(self, time_domain_signal, standardize=True):
        freq_domain_signal = self.seperate_real_and_imaginary_parts(rfft(time_domain_signal, self.signals_length))
        reduced_signal = self.pca.transform(freq_domain_signal)
        if standardize:
            reduced_signal=self.scaler.transform(reduced_signal)
        return reduced_signal
    
    
class DSManager_Stepwise_DFTDimReduction(DSManager_DFTDimReduction):
    '''
    This class does the same as the DFTDimReductoin only stepwise - using IncrementalPCA.
    It is very limited however as you can only learn (partial fit) with batches which are larger than the 
    size of the dimensionality you would ultimately want to get.
    
    It serves as a good example though for reducing the dataset stepwise.
    '''
    
    def  __init__(self, reduced_dimensionality, recording_configuration):
        # Note: this function assumes the time_domain_dataset to be a 2D np.array where each row is a signal. See DatasetManager as an example to how it is done.
        self.reduced_dimentionality = reduced_dimensionality
        self.pca = decomposition.IncrementalPCA(n_components=self.reduced_dimentionality)

    def load_learning_dataset(self, base_path, **kw):
        self.get_maximal_signal_length(base_path)
        self.load_learning_dataset_stepwise(base_path, **kw)
    
    def stepwise_learn_reduction(self, signal):
        assert len(signal) == self.maximal_signal_length
        freq_domain_signal = self.seperate_real_and_imaginary_parts(rfft(signal))
        self.pca.partial_fit(freq_domain_signal)
        
    def stepwise_reduce_signal(self, signal):
        assert len(signal) == self.maximal_signal_length
        freq_domain_signal = self.seperate_real_and_imaginary_parts(rfft(signal))
        return self.pca.transform(freq_domain_signal)

        

