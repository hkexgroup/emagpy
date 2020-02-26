#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 20:36:28 2019

@author: jkl
"""
import time
import matplotlib.pyplot as plt
from emagpy import Problem
datadir = 'examples/'

#%% importing from GF instrument
k = Problem()
# k.createSurvey(datadir + 'timelapse-wheat/170316.csv')
# k.createTimeLapseSurvey(datadir + 'timelapse-wheat')
# k.surveys = k.surveys[:2]
k.createSurvey(datadir + 'cover-crop/coverCrop.csv')
# k.importGF(datadir + 'cover-crop/coverCropLo.dat', datadir + 'cover-crop/coverCropHi.dat')

# t0 = time.time()
# k.invert(parallel=False, method='ROPE')
# print('elapsed {:.2f}s'.format(time.time() - t0))

t0 = time.time()
k.invert(njobs=-1)
print('elapsed {:.2f}s'.format(time.time() - t0))



#%% runnning mean
k = Problem()
k.createSurvey(datadir + 'cover-crop/coverCrop.csv')
k.showMap(contour=True)
k.rollingMean()
k.showMap(contour=False)
k.surveys[0].df = k.surveys[0].df[:20]
k.setInit(depths0=[0.7], 
          fixedConds=[False, False],
          fixedDepths=[False])
k.invert(forwardModel='CS', method='ROPE')
k.showResults()


#%% mapping the regolith of North Wyke
k = Problem()
k.createSurvey(datadir + 'saprolite/regolith.csv')
k.convertFromNMEA()
k.showMap(contour=True, pts=True)
k.show()
k.gridData(method='cubic')
k.surveys[0].df = k.surveys[0].dfg
k.showMap(coil = k.coils[1])
k.surveys[0].df = k.surveys[0].df[:10]


#%% inversion
k = Problem()
k.createSurvey(datadir + 'cover-crop/coverCrop.csv')
k.surveys[0].df = k.surveys[0].df[:5] # only keep 10 first measurements to make it faster

titles = []
for m in ['L-BFGS-B', 'ROPE']:
    for fm in ['CS','FS','FSandrade']:
        t0 = time.time()
        fig, axs = plt.subplots(1, 3, figsize=(10,3))
        k.invert(forwardModel=fm, method=m)
        k.showResults(ax=axs[2])
        k.showMisfit(ax=axs[1])
        k.showOne2one(ax=axs[0])
        title = '{:s} {:s} ({:.2f}s)'.format(fm, m, time.time()-t0)
        titles.append(title)
        fig.suptitle(title)
print('\n'.join(titles))


#%% test lateral smoothing
k = Problem()
k.createSurvey(datadir + 'cover-crop/coverCrop.csv')
k.invert(forwardModel='CS', alpha=0.1, beta=1)
k.showResults()


#%% mapping and save georeferenced slice
k = Problem()
k.createSurvey(datadir + 'saprolite/regolith.csv')
k.convertFromNMEA()
k.invertGN()
k.showSlice()
k.saveMap(fname=datadir + 'saprolite/map.tiff', method='idw')
k.saveSlice(fname=datadir + 'saprolite/slice.tiff')



