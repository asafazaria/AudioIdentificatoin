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
        
    def load_learning_dataset (self, base_path, standardize=False):
        self.x, self.y_loc, self.y_obj = self.load_signals_dataset(os.path.join(base_path, AudioFilesPreprocessor.original_signal_substracted_path))
        if standardize: self.standardize_dataset()
        self.x = self.transform_and_reduce_dataset(self.x)
        
    def standardize_dataset(self):
        self.scaler = preprocessing.StandardScaler().fit(self.x)
        self.x = self.scaler.transform(self.x)
        
    def load_signals_dataset(self, base_path):
        logging.info("Datasets_Manager: loading dataset from %s" % base_path)
        data_file_names = [file_name for file_name in os.listdir(base_path)
                           if (os.path.isfile(os.path.join(base_path,file_name))) and ("DS" not in file_name)]
        np_arrays_dataset = [np.load(os.path.join(base_path,file_name)) for file_name in data_file_names]
        signal_lengths = [len(np_array) for np_array in np_arrays_dataset]
        maximal_signal_length = max(signal_lengths)
        x = np.array([np.pad(signal,(0,maximal_signal_length-len(signal)),mode="constant",constant_values=(0,)).tolist() for signal in np_arrays_dataset])
        
        y_loc = np.array([self.get_location_label_from_filename(file_name) for file_name in data_file_names])
        y_obj = np.array([self.get_object_label_from_filename(file_name) for file_name in data_file_names])
        logging.info("Datasets_Manager: dataset loaded")
        return x, y_loc, y_obj
    
    def get_object_label_from_filename(self, file_name):
        objects_mapping = {
                           "C":0,
                           "K":1,
                           "M":2,
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

    

        
class DSManager_DFTDimReduction(Datasets_Manager):
    
    def  __init__(self, reduced_dimensionality):
        # Note: this function assumes the time_domain_dataset to be a 2D np.array where each row is a signal. See DatasetManager as an example to how it is done.
        self.reduced_dimentionality = reduced_dimensionality
    
    def seperate_real_and_imaginary_parts(self, signal):
        expanded_signal = np.array(np.append(np.real(signal), np.imag(signal)))
        return expanded_signal
        
    def transform_and_reduce_dataset(self, time_domain_dataset):
        # This is only to make sure that all signals are of the same length, otherwise we will not have the same granularity in the frequency domain.
        signal_lengths = [len(signal) for signal in time_domain_dataset]
        assert min(signal_lengths) == max(signal_lengths)
        self.signals_length = max(signal_lengths)
                
        freq_domain_dataset = np.array([self.seperate_real_and_imaginary_parts(rfft(signal)) for signal in time_domain_dataset])
        self.pca = decomposition.PCA(self.reduced_dimentionality)
        logging.info("DSManager_DFTDimReduction: fitting PCA to frequency domain dataset to reduce dimensionality")
        self.pca.fit(freq_domain_dataset)
        reduced_freq_domain_dataset = self.pca.transform(freq_domain_dataset)
        logging.info("DFTDimReduction: PCA was fitted and the dimensionality reduced")
        return reduced_freq_domain_dataset
        
    def reduce_dimentionality_of_signal(self, time_domain_signal):
        freq_domain_signal = self.seperate_real_and_imaginary_parts(rfft(time_domain_signal, self.signals_length))
        return self.pca.transform(freq_domain_signal)
   
        



'''   
class

    def load_dataset():
        self.dataset_time_domain = [self.get_sound_file_as_np_array(self.base_path, file_name) for file_name in os.listdir(self.base_path) 
                                    if ("DS" not in file_name) and os.path.isfile(file_name))]
    
    def get_maximal_and_minimal_signals_length():
        signal_lengths = [len(np_array) for np_array in self.dataset_time_domain]
        self.maximal_signal_lenght = max(signal_lengths)
        self.minimal_signal_length = min(signal_lengths)
    
    def transform_to_frequency_domain_and_fit_PCA(n_components = None):
        self.dataset_freq_domain = np.array([fft(signal, self.maximal_signal_lenght) for signal in self.dataset_time_domain])
        self.pca = decomposition.PCA(n_components)
        self.pca.fit(self.dataset_freq_domain)
        self.dataset_freq_domain_pca_reduced = pca.transform(self.dataset_freq_domain)
        # TODO: plot a histogram of the components fitted from the PCA to know how many components to pick
    
    def generate_dataset_from_raw_files(self):
    

        
        
        
        
        
        
        
    
    

        
        

# get the length of the maximal signal
# FFT the signal
# Do PCA on the data_set 
# pickle and save it

def reduce_using_FFT(signal, reduced_dimention):
    

# TODO List:
# 1. Align the signals according to dB rather than percentage.
# 2. Remove the original signal from the picked signal
# 3. Reduce functionality - encode FFT
# 4. Follow just the envelope of the signal?



    
    
    
class WaveletDimReduction(object):
    def __init__(self, base_path, file_name, audio_configuration):
        pass
    
    def encode_time_series(time_seriesno
    


BASE_PATH = "../../Sensor_Final/"
FILE_NAME_1 = "TB3_PF0_empty_1.raw"
FILE_NAME_2 = "TB3_PF0_empty_2.raw"

#remove_silent_from_file(BASE_PATH, FILE_NAME, recording_configuration, silence_configuration)

sound1 = load_sound_file_as_np_array(BASE_PATH,FILE_NAME_1, recording_configuration)
sound2 = load_sound_file_as_np_array(BASE_PATH,FILE_NAME_2, recording_configuration)
plt.plot(sound2, color="red")


#plt.plot(sound1, color="blue")

print(len(sound1))
print(len(sound2))
                         
           '''             



        