#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 20:29:19 2019

@author: jkl
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from invertHelper import fCS, fMaxwellECa, fMaxwell, getQs
from Survey import Survey
'''
API structure:
- Survey class (coil specific)
    - df: main dataframe (one column per coil, one row per location)
    - read()
    - interpolate() (kriging or gridding)
    - crossOverError() using cross-over points
- CalibrationData class (EC, ECa)
    - apply(Survey)
    - show()
- Model class (depth specific)
    - df: main dataframe (one column per layer/depths, one row per location)
    - show()
    - setDepths()
    - setEC()
- Problem class (master class)
    - surveys: list of Survey object
    - models: list of Model object
    - invert(forwardModel='CS',
             method='SCEUA/TNC/',
             constrain='quasi2D', 'quasi3D', 'none')
    - forwardEM()
    - forwardCS()
    
'''

class Problem(object):
    ''' Class defining an inversion problem.
    '''
    def __init__(self):
        self.depths = np.array([1, 2]) # depths of the bottom of each layer (last one is -inf)
        self.conds = np.array([20, 20, 20]) # initial conductivity for each layer
#        self.fixedConds = []
#        self.fixedDepths = []
        self.surveys = []
        self.models = []
        
        
    def createSurvey(self, fname):
        ''' Create a survey object.
        '''
        self.surveys.append(Survey(fname))
        
        
    def createTimeLapseSurvey(self, dirname):
        ''' Create a list of surveys object.
        '''
        pass
    
        
    def setDepths(self, depths):
        ''' Set the depths of the bottom of each layer. Last layer goes to -inf.
        Depths should be positive going down.
        '''
        if len(depths) == 0:
            raise ValueError('No depths specified.')
        if all(np.diff(depths) > 0):
            raise ValueError('Depths should be ordered and increasing.')
        self.depths = np.array(depths)
        
        
    def invert(self, method='solve', **kwargs):
        self.invertMinimize(**kwargs)
    
    
    def invertMinimize(self):
        ''' Invert using the minimize function of scipy and an objective
        function (by default the RMS).
        '''
        
        
    def forwardCS(self):
        ''' Forward modelling using the cumulative sensitivity function.
        '''
        return
        fCS(sigma, depths, cspacing, cpos, hx=1)
        
        sigma = np.array([30, 30, 30, 30]) # layers conductivity [mS/m]
depths = np.array([0.3, 0.7, 2]) # thickness of each layer (last is infinite)
f = 30000 # Hz frequency of the coil
cpos = np.array(['hcp','hcp','hcp','vcp','vcp','vcp']) # coil orientation
#cpos = np.array(['hcp','hcp','hcp','prp','prp','prp']) # coil orientation
#cspacing = np.array([0.32, 0.71, 1.18, 0.32, 0.71, 1.18])
cspacing = np.array([1.48, 2.82, 4.49, 1.48, 2.82, 4.49]) # explorer

print('fCS:', fCS(sigma, depths, cspacing, cpos))
print('fMaxwellECa:', fMaxwellECa(sigma, depths, cspacing, cpos))
print('fMaxwellQ:', fMaxwellQ(sigma, depths, cspacing, cpos))
print('fCSandrade:', fCSandrade(sigma, depths, cspacing, cpos, hx=0))

print('fCS:', fCS(sigma, depths, cspacing, cpos, hx=1))
print('fCS:', fCS(sigma, depths, cspacing, cpos, hx=1, rescaled=True))
print('fCSandrade:', fCSandrade(sigma, depths, cspacing, cpos, hx=1))


    
    def fowardFS(self):
        ''' Forward modelling using the full Maxwell-based solution.
        '''
        
        
    
    