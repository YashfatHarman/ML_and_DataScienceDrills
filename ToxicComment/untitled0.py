#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 04:22:50 2018

@author: osboxes
"""

'''
Just a test to pass values to functions using a dict
'''

def test_func(first = 0, second = 0, third = 0, fourth = 0):
    print("first:", first)
    print("second:", second)
    print("third:", third)
    print("fourth:", fourth)
    pass

if __name__ == "__main__":
    print("Hello World!")
    dic = {"first" : 99, "second" : 88}
    test_func(**dic)
    print("The world never says hello back!")