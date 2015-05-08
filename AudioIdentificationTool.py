'''
Created on May 7, 2015

@author: Butzik
'''


import os
import datetime
import logging
import argparse
import numpy as np



from AudioFilesPreprocessor import AudioFilesPreprocessor
from Classifiers import *
from AudioDatasets import DSManager_DFTDimReduction

class AudioIdentificationCommandline(object):

    DEFAULT_LOG_PATH = "./"
    DEFAULT_LOG_NAME = "AudioIdentificationToolLog.log"
    
    VALIDATION_SET_SIZE = 0.2
    NO_OF_DIMENTIONS = 128
    
    RECORDING_CONF = {"encoding_dtype" : np.float32,
                               "encoding_for_array": 'f',
                               "encoding" : "floating-point",
                               "encoding_size" : 32,
                               "sample_rate" : 96000}
    ORIGINAL_FILE_PATH = "/Users/Butzik/Dropbox (MIT)/Sensor_Final/Table_Readings/Reference_Files"
    ORIGINAL_FILE_NAME = 'No_Input.raw'

        
    # State here the class for each reducer command
    REDUCER_OF_CLASS = {"DFT_PCA": DSManager_DFTDimReduction, 
                        "WAVELETS": None,
                        "PAA": None}
    
    # State here the class for each classifier
    CLASSIFIERS_OF_CLASS = {"SVM":SVM_classifier,
                            "RForests":RForests_classifier,
                            "AdaBoost":Adaboost_classifier, 
                            "KNN":KNN_classifier, 
                            "LRegression":LogisticRegression_classifier, 
                            "ALL": None}

    def __init__(self):
        
        # State here the possible stages and their handlers
        self.STAGE_OF_HANDLER = {"preprocess": self.handle_preprocess, 
                                 "reduce_dataset": self.handle_reduce_dataset, 
                                 "train_classifier": self.handle_train_classifier, 
                                 "classify": self.handle_classify}

        self.parse_commandline_arguments()
        logging.basicConfig(self.command_args.log_path, level=logging.INFO)
        self.call_stage_method()

    def parse_commandline_arguments(self):
        parser = argparse.ArgumentParser(description='AudioIdentificationTool - MAS836')
        
        parser.add_argument("stage", 
                            help = "Specify the stage to execute",
                            choices=self.STAGE_OF_HANDLER.keys())
        parser.add_argument("input_path", 
                            help = "The input path (either file or folder according to the required action")
        parser.add_argument("output_path",
                            help = "The output path (either file or folder according to the required action")
        
        #Reduce Dataset arguments
        parser.add_argument("--normalize", dest="normalize", 
                            help="Normalize data before reducing its dimensionality", action="store_true")
        parser.add_argument("-r", "--reducer_type", dest="reducer_type", 
                            choices= self.REDUCER_OF_CLASS.keys(),
                            default="DFT_PCA",
                            help="Specifies what kind of reducer Class to apply in order to reduce the data's dimensionality")
        parser.add_argument("-d", "--dimension", dest="target_dimension",
                            type=int,                        
                            default=self.NO_OF_DIMENTIONS,
                            help="Specifies the target dimension to reduce the data to")
        
        #Train Classifier arguments
        parser.add_argument("-c", "--classifier_type", dest="classifier_type", 
                            choices=[ ],
                            default="SVM",
                            help="Specifies what kind of classifier to train on the dataset. ALL will run all types subsequently")
        
        #General features
        parser.add_argument("-l","--log_path", dest="log_path", default = os.path.join(self.DEFAULT_LOG_PATH, self.DEFAULT_LOG_NAME),
                            help = "Specify path where to log the execution results")
        
        self.command_args = parser.parse_args()

    def call_stage_method(self):
        self.STAGE_OF_HANDLER[self.command_args.stage](self.command_args)

    def handle_preprocess(self):
        afp = AudioFilesPreprocessor(self.command_args.input_path, self.RECORDING_CONF, self.ORIGINAL_FILE_PATH, self.ORIGINAL_FILE_NAME)
        afp.preprocess_dataset()
        os.symlink(os.path.join(self.command_args.input_path,AudioFilesPreprocessor.original_signal_substracted_path), self.command_args.output_path)
        logging.info("AudioIdentificationCommandline: preprocessing done")
        
    def handle_reduce_dataset(self):    
        dsm = self.reducers_dict[self.command_args.reducer_type](self.command_args.target_dimension)
        dsm.load_learning_dataset(self.command_args.input_file, standardize=self.command_args.normalize)
        current_time = datetime.datetime.now().strftime("%Y%m%d_%I%M%S")
        dsm.save(self.command_args.output_path + "_" + current_time+ ".reduced")
        logging.info("AudioIdentificationCommandline: data reduced to: %s" % self.command_args.output_path + "_" + current_time+ ".reduced")
        
    def handle_train_classifier(self):
        classifier = self.CLASSIFIERS_OF_CLASS[self.command_args.classifier_type]()
        classifier.load_pickled_dataset(self.command_args.input_path)
        classifier.train_using_single_set(self.VALIDATION_SET_SIZE)
        current_time = datetime.datetime.now().strftime("%Y%m%d_%I%M%S")
        classifier.save(self.command_args.output_path + "_" + current_time + ".classifier")
        logging.info("AudioIdentificationCommandline: classifier trained: %s" % self.command_args.output_path + "_" + current_time + ".classifier")

    def handle_classify(self, command_args):
        pass

    


if __name__ == "__main__":
    AudioIdentificationCommandline()
    

    
    
'''        
    # STEP 1: first we do some pre-processing
    afp = AudioFilesPreprocessor(BASE_PATH,recording_configuration, ORIGINAL_FILE_PATH, ORIGINAL_FILE_NAME)
    afp.preprocess_dataset()

    
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
    
    afp.preprocess_file(TEST_BASE_PATH, TEST_SIGNAL)
    print loaded_rfc.predict_object_label(np.load(os.path.join(TEST_BASE_PATH, AudioFilesPreprocessor.original_signal_substracted_path, TEST_SIGNAL)))
    print loaded_rfc.predict_location_label(np.load(os.path.join(TEST_BASE_PATH, AudioFilesPreprocessor.original_signal_substracted_path, TEST_SIGNAL)))
'''