#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 20:36:28 2019

@author: jkl
"""

from emagpy import Problem
testdir = './emagpy/test/'

#%% importing from GF instrument
k = Problem()
k.importGF(testdir + 'coverCropLo.dat', testdir + 'coverCropHi.dat')
k.show()

#%% runnning mean
k = Problem()
k.createSurvey(testdir + 'coverCrop.csv')
k.showMap(contour=True)
k.rollingMean()
k.showMap(contour=True) # TODO segmentation fault

#%% mapping the regolith of North Wyke
k = Problem()
k.createSurvey(testdir + 'regolith.csv')
k.convertFromNMEA()
k.showMap(contour=True, pts=True)
k.show()
k.gridData(method='cubic')
k.surveys[0].df = k.surveys[0].dfg
k.showMap(coil = k.coils[1])

#%% inversion
k = Problem()
k.createSurvey(testdir + 'coverCrop.csv')
k.surveys[0].df = k.surveys[0].df[:10] # only keep 10 first measurements to make it faster
k.invert(forwardModel='CS fast')
k.showResults()
k.showMisfit()
k.showOne2one()

k.invert(forwardModel='CS')
k.showResults()
k.showMisfit()
k.showOne2one()

k.invert(forwardModel='FS')
k.showResults()
k.showMisfit()
k.showOne2one()

k.invert(forwardModel='FSandrade')
k.showResults()
k.showMisfit()
k.showOne2one()

k.invert(forwardModel='Q')
k.showResults()
k.showMisfit()
k.showOne2one()

#%%
k = Problem()
k.createSurvey(testdir + 'coverCrop.csv')
k.invert(forwardModel='CS', alpha=0.1, beta=1)
k.showResults()

#%% mapping
k = Problem()
k.createSurvey('emagpy/test/regolith.csv')
k.convertFromNMEA()
k.invertGN()
k.showSlice()
k.saveMap(fname='emagpy/test/map.tiff', method='idw')
k.saveSlice(fname='emagpy/test/slice.tiff')

#%% inverting iwith SCUEA
k = Problem()
k.createSurvey(testdir + 'coverCrop.csv')
k.surveys[0].df = k.surveys[0].df[:5]
#k.invertMCMC()
k.invertMCMCq(rep=500)
k.showMisfit()
#k.showOne2one()
#k.showResults()


