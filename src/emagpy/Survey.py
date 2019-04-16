#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 20:27:12 2019

@author: jkl
"""
import pandas as pd

class Survey(object):
    ''' Create a Survey object containing the raw EMI data.
    '''
    def __init__(self, fname):
        self.df = pd.read_csv(fname)
        self.freq = None # how to handle different frequencies ?
        
    def show(self):
        ''' Show the data.
        '''
    
    def showMap(self):
        ''' Display a map of the measurements.
        '''
    
    def gridData(self):
        ''' TODO
        '''
        