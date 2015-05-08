'''
Created on May 1, 2015

@author: Butzik
'''


import numpy as np
import array
import os
from scipy.fftpack import rfft, irfft
from sklearn import decomposition
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import cPickle
import datetime
from IPython.utils.tests.test_module_paths import TEST_FILE_PATH



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


class AudioFilesPreprocessor(object):
    
    '''
    This class pre-process the audio files. 
    It removes silence from the beginning and the end of each file using SoX. (should be installed on your system)
    It also removes the original signal from the picked-up signal by subtracting them in the frequency domain.
    The results are saved in an np.array format in the folder specified in original_signal_substracted_path.
    '''
    silence_stripped_path = "Silence_stripped"
    original_signal_substracted_path = "Original_substracted"
    silence_configuration = {"below_period" : 1,
                             "override_duration": 0,
                             "threshold" : 1}

    
    def __init__(self, base_path, audio_files_configuration, original_sound_base_path, original_sound_file_name):
        self.base_path = base_path
        self.audio_configuration = audio_files_configuration
        self.original_sound_base_path = original_sound_base_path
        self.original_sound_file_name = original_sound_file_name
        
    def strip_silence_from_file(self, base_path, file_name):
        file_absolute_path = os.path.join(base_path, file_name).replace(" ","\ ").replace("(", "\(").replace(")","\)")
        output_file = os.path.join(base_path, self.silence_stripped_path, file_name).replace(" ","\ ").replace("(", "\(").replace(")","\)")
        sox_command = "/usr/local/bin/sox -e %s -b%d -L -r%d -c1 %s %s " % (self.audio_configuration["encoding"],
                                                             self.audio_configuration["encoding_size"],
                                                             self.audio_configuration["sample_rate"],
                                                             file_absolute_path,
                                                             output_file)   
        silence_filter = "silence %d %d %s reverse silence %d %d %s reverse" % (self.silence_configuration["below_period"],
                                                                                self.silence_configuration["override_duration"],
                                                                                str(self.silence_configuration["threshold"]) + "%",
                                                                                self.silence_configuration["below_period"],
                                                                                self.silence_configuration["override_duration"],
                                                                                str(self.silence_configuration["threshold"]) + "%")
        print "AudioFilesPreprocessor removing silence: %s" % (sox_command + silence_filter)
        print os.popen(sox_command + silence_filter).read()
        
    def strip_silence_from_entire_dataset(self):
        files = [self.strip_silence_from_file(self.base_path,file_name) for file_name in os.listdir(self.base_path) if (("DS" not in file_name) and (os.path.isfile(os.path.join(self.base_path,file_name))))]
        
        
    def get_signal_array_from_file(self, base_path, file_name):
        sound_file = open(os.path.join(base_path, file_name),"rb")
        sound_raw_buffer = sound_file.read()
        signal_array = np.array(array.array(self.audio_configuration["encoding_for_array"],sound_raw_buffer).tolist(), self.audio_configuration["encoding_dtype"])
        return signal_array
    
    def subtract_original_signal_from_picked_signal(self, original_signal, picked_signal):
        # Note this function assumes that the signals are aligned for the starting point!
        fft_length = max(len(original_signal), len(picked_signal))
        original_f_domain = rfft(original_signal, n= fft_length)
        picked_f_domain = rfft(picked_signal, n= fft_length)
        assert len(original_f_domain) == len(picked_f_domain)
        difference_signal = picked_f_domain - original_f_domain
        return irfft(difference_signal)
    
    def subtract_original_signal_from_dataset(self, original_signal_base_path, original_signal_file_name):
        self.original_signal = self.get_signal_array_from_file(original_signal_base_path, original_signal_file_name)
        files_list = [file_name for file_name in os.listdir(os.path.join(self.base_path,self.silence_stripped_path)) 
                      if (os.path.isfile(os.path.join(self.base_path,self.silence_stripped_path,file_name)) and ("DS" not in file_name))]    
        for file_name in files_list:
            print "AudioFilesPreprocessor removing original sound from: %s" % (file_name)
            signal = self.get_signal_array_from_file(os.path.join(self.base_path,self.silence_stripped_path), file_name)
            substracted_signal = self.subtract_original_signal_from_picked_signal(self.original_signal,signal)
            np.save(os.path.join(self.base_path,self.original_signal_substracted_path,file_name),substracted_signal)
            
    def preprocess_dataset(self, strip_silence=True, subtract_original=True, normalize=False):
        if strip_silence: self.strip_silence_from_entire_dataset()
        if subtract_original: self.subtract_original_signal_from_dataset(self.original_sound_base_path, self.original_sound_file_name)
        
    def preprocess_file(self, base_path, file_name, strip_silence=True, subtract_original=True, normalize=False):
        print "AudioFilesPreprocessor: preprocessing file: %s" % file_name
        signal = self.get_signal_array_from_file(base_path, file_name)
        if strip_silence: self.strip_silence_from_file(base_path, file_name)
        if subtract_original: self.subtract_original_signal_from_picked_signal(self.original_signal,signal)
            
        
              

class Datasets_Manager(object):
    def __init__(self):
        pass
        
    def load_learning_dataset (self, base_path):
        self.x, self.y_loc, self.y_obj = self.load_signals_dataset(os.path.join(base_path, AudioFilesPreprocessor.original_signal_substracted_path))
        self.x = self.transform_and_reduce_dataset(self.x)
        
    def load_signals_dataset(self, base_path):
        print "Datasets_Manager: loading dataset from %s" % base_path
        data_file_names = [file_name for file_name in os.listdir(base_path)
                           if (os.path.isfile(os.path.join(base_path,file_name))) and ("DS" not in file_name)]
        np_arrays_dataset = [np.load(os.path.join(base_path,file_name)) for file_name in data_file_names]
        signal_lengths = [len(np_array) for np_array in np_arrays_dataset]
        maximal_signal_length = max(signal_lengths)
        x = np.array([np.pad(signal,(0,maximal_signal_length-len(signal)),mode="constant",constant_values=(0,)).tolist() for signal in np_arrays_dataset])
        
        y_loc = np.array([self.get_location_label_from_filename(file_name) for file_name in data_file_names])
        y_obj = np.array([self.get_object_label_from_filename(file_name) for file_name in data_file_names])
        print "Datasets_Manager: dataset loaded"
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
        print "DSManager_DFTDimReduction: fitting PCA to frequency domain dataset"
        self.pca.fit(freq_domain_dataset)
        print "PCA fitted"
        reduced_freq_domain_dataset = self.pca.transform(freq_domain_dataset)
        print "DFTDimReduction: PCA was fitted and the dimensionality reduced"
        return reduced_freq_domain_dataset
        
    def reduce_dimentionality_of_signal(self, time_domain_signal):
        freq_domain_signal = self.seperate_real_and_imaginary_parts(rfft(time_domain_signal, self.signals_length))
        return self.pca.transform(freq_domain_signal)
   
class Classifier(object):
        
    def __init__(self):
        pass
        
    def load_raw_dataset_from_folder(self, base_path, reduction_class, target_dimentionality):
        self.datasets = reduction_class(target_dimentionality)
        self.datasets.load_learning_dataset(base_path)
        
    def load_pickled_dataset(self, base_path, file_name):
        self.datasets = Datasets_Manager.loader(os.path.join(base_path,file_name))
        
    def train_using_single_set(self, validation_set_size):
        self.datasets.genereate_train_and_validate_from_learning_dataset(validation_set_size)            
        self.loc_classifier, self.loc_training_score = self.train_and_choose_parameters(self.datasets.y_loc_train, self.datasets.y_loc_validate)
        self.obj_classifier, self.obj_training_score = self.train_and_choose_parameters(self.datasets.y_obj_train, self.datasets.y_obj_validate)
        
    def predict_object_label(self, signal):
        reduced_signal = self.datasets.reduce_dimentionality_of_signal(signal)
        return self.obj_classifier.predict(reduced_signal)  

    def predict_location_label(self, signal):
        reduced_signal = self.datasets.reduce_dimentionality_of_signal(signal)
        return self.loc_classifier.predict(reduced_signal)  
    
    def similarity_of_signals(self, signal_x, signal_y, distance_order=None):
        return np.linalg.norm((signal_x-signal_y), ord=distance_order)
    
    def direct_search_of_closest_signal(self, signal):
        reduced_signal = self.datasets.reduce_dimentionality_of_signal(signal)
        minimal_distance = np.Inf
        closest_signal = None
        for signal in self.datasets.x:
            distance = self.similarity_of_signals(reduced_signal, signal)
            if distance < minimal_distance:
                minimal_distance = distance
                closest_signal = signal
        return closest_signal, minimal_distance
    
    def save(self, to_file):
        cPickle.dump(self, file(to_file, "wb"))
        
    @classmethod
    def loader(cls, from_file):
        return cPickle.load(file(from_file,"rb"))



class RForests_classifier(Classifier):
    
    criterion = ["gini", "entropy"]
    n_estimators = [1, 5, 10, 15, 20, 100, 200]
       
    def train_with_parameters(self, n_estimators, criterion_string, training_labels, validation_labels):
        classifier = RandomForestClassifier(n_estimators = n_estimators, criterion=criterion_string)
        classifier.fit(self.datasets.x_train, training_labels)
        score = classifier.score(self.datasets.x_validate, validation_labels)
        return classifier, score
    
    def train_and_choose_parameters(self, training_labels, validation_labels):
        best_score = 0
        best_classifier = None
        for n_estimator in self.n_estimators:
            for criterion_string in self.criterion:
                classifier, score = self.train_with_parameters(n_estimator, criterion_string, training_labels, validation_labels)
                if score >= best_score:
                    best_score = score
                    best_classifier = classifier
        print "RF: Classifier trained with accuracy %s" % best_score
        print "RF: Classifiers parameters are: %s" % best_classifier.get_params()
        return best_classifier, best_score

      
        
        
if __name__ == "__main__":
    BASE_PATH = "/Users/Butzik/Dropbox (MIT)/Sensor_Final/TrainingData_sample"

    ORIGINAL_FILE_PATH = "/Users/Butzik/Dropbox (MIT)/Sensor_Final/Table_Readings/Reference_Files"
    ORIGINAL_FILE_NAME = 'No_Input.raw'
    
    REDUCED_DATASET_MANAGER_FILE_NAME = "DSManager_DFTDimReduction_pickled"
    CLASSIFIER_FILE_NAME = "RForests_classifier_pickled"
    
    TEST_SIGNAL = ""
    TEST_BASE_PATH = "/Users/Butzik/Dropbox (MIT)/Sensor_Final/TestData_sample"
    
    NO_OF_DIMENTIONS = 50 # Note the number of dimensions should be less than the order of magnitude of the number of training samples.
    recording_configuration = {"encoding_dtype" : np.float32,
                               "encoding_for_array": 'f',
                               "encoding" : "floating-point",
                               "encoding_size" : 32,
                               "sample_rate" : 96000}
    
    
    # STEP 1: first we do some pre-processing
    afp = AudioFilesPreprocessor(BASE_PATH,recording_configuration,os.path.join(BASE_PATH,ORIGINAL_FILE_PATH), ORIGINAL_FILE_NAME)
    #afp.preprocess_dataset()
    
    # STEP 2: initialize a dataset manager instance from the reduction method of your choice and perform the dimensionality reduction on the dataset.
    dsm = DSManager_DFTDimReduction(NO_OF_DIMENTIONS)
    dsm.load_learning_dataset(BASE_PATH)
    
    current_time = datetime.datetime.now().strftime("%Y%m%d_%I%M%S")
    dsm.save(os.path.join(BASE_PATH, REDUCED_DATASET_MANAGER_FILE_NAME + "_" + current_time))
    
    # STEP 3: initialize a classifier of your choice and provide the reduced dataset.
    rfc = RForests_classifier()
    rfc.load_pickled_dataset(BASE_PATH, REDUCED_DATASET_MANAGER_FILE_NAME + "_" + current_time)
       
    # Alternatively, this can be done with step 2 implicitly by using:
    # rfc.load_raw_dataset_from_folder(BASE_PATH, DSManager_DFTDimReduction, NO_OF_DIMENTIONS)

    rfc.train_using_single_set(0.2)
    rfc.save(os.path.join(BASE_PATH,CLASSIFIER_FILE_NAME + "_" + current_time))
    
    loaded_rfc = RForests_classifier.loader(os.path.join(BASE_PATH,CLASSIFIER_FILE_NAME + "_" + current_time))
    
    afp.preprocess_file(TEST_BASE_PATH, TEST_FILE_PATH)
    print loaded_rfc.predict_object_label(np.load(os.path.join(TEST_BASE_PATH, AudioFilesPreprocessor.original_signal_substracted_path, TEST_FILE_PATH)))
    print loaded_rfc.predict_location_label(np.load(os.path.join(TEST_BASE_PATH, AudioFilesPreprocessor.original_signal_substracted_path, TEST_FILE_PATH)))



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



        