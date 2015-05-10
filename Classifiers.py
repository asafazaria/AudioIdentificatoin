'''
Created on May 7, 2015

@author: Butzik
'''

import os
import cPickle
import numpy as np
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report

from AudioDatasets import Datasets_Manager
from AudioFilesPreprocessor import AudioFilesPreprocessor


class Classifier(object):
        
    def __init__(self):
        pass
        
    def load_raw_dataset_from_folder(self, base_path, reduction_class, target_dimentionality):
        self.datasets = reduction_class(target_dimentionality)
        self.datasets.load_learning_dataset(base_path)
        
    def load_test_set_from_folder(self, base_path):
        # Note that the datasets object takes care of scaling, reduction etc.. but not Preprocessing! (This should be done once separately)
        self.datasets.load_test_set_at_once(base_path)
        
    def load_pickled_dataset(self, file_path):
        self.datasets = Datasets_Manager.loader(file_path)
        
    def evaluate_accuracy_on_test_set(self, test_set_path):
        self.load_test_set_from_folder(test_set_path)
        predicted_y = self.obj_classifier.predict(self.datasets.x_test)
        # Note this is tightly coupled with the definitions in AudioDatasets - should be fixed and updated with the right names
        target_names = ["C","K", "M", "S"]
        print classification_report(self.datasets.y_obj_test, predicted_y, target_names=target_names)
        
    def train_using_single_set(self, validation_set_size):
        self.datasets.genereate_train_and_validate_from_learning_dataset(validation_set_size)
        #self.loc_classifier, self.loc_training_score = self.train_and_choose_parameters(self.datasets.y_loc_train, self.datasets.y_loc_validate)
        self.obj_classifier, self.obj_training_score = self.train_and_choose_parameters(self.datasets.y_obj_train, self.datasets.y_obj_validate)
        
                
    def predict_object_label_for_file(self, base_path, file_name, recording_configuration, original_base, original_name):
        afp = AudioFilesPreprocessor(base_path, recording_configuration, original_base, original_name)
        afp.preprocess_file(base_path, file_name)
        signal = self.datasets.stepwise_load_signal(os.path.join(base_path,AudioFilesPreprocessor.original_signal_substracted_path,file_name+".npy"))
        reduced_signal = self.datasets.transform_and_reduce_signal(signal)
        return self.predict_object_label(reduced_signal)
        
    def predict_object_label(self, reduced_signal):
        return self.obj_classifier.predict(reduced_signal)  

    #def predict_location_label(self, reduced_signal):
    #    return self.loc_classifier.predict(reduced_signal)  
    
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
        logging.info("RF: Classifier trained with accuracy %s" % best_score)
        logging.info("RF: Classifiers parameters are: %s" % best_classifier.get_params())
        return best_classifier, best_score


class LogisticRegression_classifier(Classifier):
    C=[0.0001, 0.001, 0.1, 1, 10, 20, 100]
    penalty = ["l1","l2"]
        
    
    def train_with_parameters(self, c_value, penalty, training_labels, validation_labels):
        classifier = linear_model.LogisticRegression(C=c_value, penalty=penalty)
        classifier.fit(self.datasets.x_train, training_labels)
        score = classifier.score(self.datasets.x_validate,validation_labels)
        return classifier, score
    
    def train_and_choose_parameters(self, training_labels, validation_labels):
        best_score = 0
        best_classifier = None
        for c_value in self.C:
            for pen in self.penalty:
                classifier, score = self.train_with_parameters(c_value, pen,training_labels, validation_labels)
                if score >= best_score:
                    best_score = score
                    best_classifier = classifier
        logging.info("LOGISTICREG: Classifier trained with accuracy %s" % best_score)
        logging.info("LOGISTICREG: Classifiers parameters are: %s" % best_classifier.get_params())
        return best_classifier, best_score
    
    
class SVM_classifier(Classifier):
    
    beta_range = [0.001, 0.01, 0.1, 1, 10]
    c_range = [0.001, 0.01, 0.1, 1, 10, 20]        
                
    def train_linear_classifier(self, c_const, training_labels, validation_labels):
        classifier = SVC(C = c_const, kernel="linear")
        classifier.fit(self.datasets.x_train, training_labels)
        score = classifier.score(self.datasets.x_validate,validation_labels)
        return classifier, score

    def train_rbf_calssifier(self, c_const, beta_const, training_labels, validation_labels):
        classifier = SVC(C = c_const, kernel="rbf", gamma= beta_const)
        classifier.fit(self.datasets.x_train, training_labels)
        score = classifier.score(self.datasets.x_validate,validation_labels)
        return classifier, score
    
    def best_linear_classifier(self, training_labels, validation_labels):
        best_score = 0
        best_classifier = None
        for c_const in self.c_range:
            classifier, score = self.train_linear_classifier(c_const, training_labels, validation_labels)
            if score >= best_score:
                best_score = score
                best_classifier = classifier
        return best_classifier, best_score
    
    def best_rbf_classifier(self, training_labels, validation_labels):
        best_score = 0
        best_classifier = None
        for c_const in self.c_range:
            for beta_const in self.beta_range:
                classifier, score = self.train_rbf_calssifier(c_const, beta_const, training_labels, validation_labels)
                if score >= best_score:
                    best_score = score
                    best_classifier = classifier
        return best_classifier, best_score
    
    def train_and_choose_parameters(self, training_labels, validation_labels):
        linear_classifier, linear_score = self.best_linear_classifier(training_labels, validation_labels)
        rfb_classifier, rbf_score = self.best_rbf_classifier(training_labels, validation_labels)
        if linear_score >= rbf_score:
            best_classifier = linear_classifier
            best_score = linear_score
        else:
            best_classifier = rfb_classifier
            best_score = rbf_score
        
        logging.info("SVM: Classifier trained with accuracy %s" % best_score)
        logging.info("SVM: Classifiers parameters are: %s" % best_classifier.get_params())
        
        return best_classifier, best_score
      
class KNN_classifier(Classifier):
    k_neighbors = [1, 5, 10, 20]
    weights = ["uniform", "distance"]
    leaf_size = [2,5,10,20,30,100]
    p_value = [1,2,3]
            
    def train_with_parameters(self, k, w, l, p, training_labels, validation_labels):
        classifier = neighbors.KNeighborsClassifier(n_neighbors=k,weights=w,leaf_size=l,p=p)
        classifier.fit(self.datasets.x_train, training_labels)
        score = classifier.score(self.datasets.x_validate,validation_labels)
        return classifier, score
    
    def train_and_choose_parameters(self, training_labels, validation_labels):
        best_score = 0
        best_classifier = None
        for k in self.k_neighbors:
            for w in self.weights:
                for l in self.leaf_size:
                    for p in self.p_value:
                        classifier, score = self.train_with_parameters(k,w,l,p, training_labels, validation_labels)
                        if score >= best_score:
                            best_score = score
                            best_classifier = classifier
        logging.info("KNN: Classifier trained with accuracy %s" % best_score)
        logging.info("KNN: Classifiers parameters are: %s" % best_classifier.get_params())
        return best_classifier, best_score


class Adaboost_classifier(Classifier):
    n_estimators = [1, 10, 50, 100, 200, 500, 700]
    
    
    def train_with_parameters(self, n_estimators, training_labels, validation_labels):
        classifier = AdaBoostClassifier(n_estimators = n_estimators)
        classifier.fit(self.datasets.x_train, training_labels)
        score = classifier.score(self.datasets.x_validate,validation_labels)
        return classifier, score
    
    def train_and_choose_parameters(self, training_labels, validation_labels):
        best_score = 0
        best_classifier = None
        for n_estimator in self.n_estimators:
            classifier, score = self.train_with_parameters(n_estimator, training_labels, validation_labels)
            if score >= best_score:
                best_score = score
                best_classifier = classifier
        logging.info("ADABOOST: Classifier trained with accuracy %s" % best_score)
        logging.info("ADABOOST: Classifiers parameters are: %s" % best_classifier.get_params())
        return best_classifier, best_score
        