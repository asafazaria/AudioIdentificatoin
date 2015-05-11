'''
Created on May 10, 2015

@author: Butzik
'''


import logging

class Logger_object(object):    
    def __init__(self):
        pass
        
    def set_log_path(self, log_path):
        logging.basicConfig(filename=log_path, level=logging.INFO)
        
    def log(self, message):
        print message
        logging.info(message)
        
Logger = Logger_object()



        