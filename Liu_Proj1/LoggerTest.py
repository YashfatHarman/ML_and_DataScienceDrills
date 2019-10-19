#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 16:02:42 2019

@author: osboxes
"""

'''
A quick test to see if logging can be used for output redirection.
'''

import sys
from contextlib import contextmanager
import datetime

class Logger(object):
    def __init__(self, filename = "default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        pass
    
    def __getattr__(self, attr):
        return getattr(self.terminal, attr)
    
        

@contextmanager
def stdout_redirected(new_stdout):
    save_stdout = sys.stdout
    sys.stdout = new_stdout
    try:
        yield None
    finally:
        sys.stdout = save_stdout
        
def get_filename():
    #get the date and time and return as a string
    
    dt = datetime.datetime.now()
    
    dt_string = str(dt.year) + "_" + str(dt.month) + "_" + str(dt.day) + "_" + str(dt.hour) + "_" + str(dt.minute) + "_" + str(dt.second)
    
    return dt_string
     
    pass


import pickle
def pickle_write():
    myData = [5,7,8,1,4]
    f = open("myData.pkl", "wb")
#    pickler = pickle.Pickler(f)
#    for e in myData:
#        pickler.dump(e)
#    f.close()
#     
    for e in myData:
        pickle.dump(e,f)
    f.close()
    
def pickle_read():
    f = open("myData.pkl", "rb")
#    unpickler = pickle.Unpickler(f)
    #data = unpickler.load()
#    while True:
#        try:
#            data = unpickler.load()
#            print(data)
#        except EOFError:
#            break
    while True:
        try:
            data = pickle.load(f)
            print(data)
        except EOFError:
            break
    f.close()
     
    
if __name__ == "__main__":
    #    sys.stdout = Logger("test_logfile.log")
    #    print("Hello World")
    #    for ii in range(10,15):
    #        print("writing {}".format(ii))
    #    ResetLogger()
    #    for ii in range(15,20):
    #        print("writing {}".format(ii))
    #
    #    print("Oka Bye!")

#    with(open("test_output_redirection.txt","w")) as f:
#        with stdout_redirected(f):
#            print("Hello World!")
#            print("Oka bye!")
#    print(get_filename())
    #pickle_write()
    pickle_read()