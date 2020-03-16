#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 20:36:28 2019

@author: jkl
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from emagpy import Problem
datadir = 'examples/'


#%% importing from GF instrument and filtering
k = Problem()
k.importGF(datadir + 'cover-crop/coverCropLo.dat', datadir + 'cover-crop/coverCropHi.dat')

# filtering
k.filterRange(vmin=0, vmax=25)
k.filterPercentile(qmin=2, qmax=95)
k.rollingMean()
k.filterDiff()


#%% mapping potatoes field
k = Problem()
k.createSurvey(datadir + 'potatoes/potatoesLo.csv')
k.convertFromNMEA()
k.crossOverPoints()
k.plotCrossOverMap()
k.showMap(contour=True, pts=True)
k.saveMap(datadir + 'potatoes/map.tiff')
k.show()
k.gridData(method='cubic')
k.surveys[0].df = k.surveys[0].dfg


#%% inversion with uncertainty
k = Problem()
k.createSurvey(datadir + 'cover-crop/coverCrop.csv')
k.surveys[0].df = k.surveys[0].df[:20]
k.setInit(depths0=[0.3, 0.7], fixedDepths=[True, False], fixedConds=[False, True, False])
k.invert(method='ROPE', rep=500, njobs=-1)
k.showResults(errorbar=True, overlay=True)
k.showProfile(errorbar=True)


#%% inversion
k = Problem()
k.createSurvey(datadir + 'cover-crop/coverCrop.csv')
k.surveys[0].df = k.surveys[0].df[:5] # only keep 10 first measurements to make it faster

titles = []
for m in ['L-BFGS-B', 'ROPE']:
    for fm in ['CS','FSlin','FSeq']:
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

k.showMisfit()
k.showOne2one()


#%% test lateral smoothing
k = Problem()
k.createSurvey(datadir + 'cover-crop/coverCrop.csv')
k.setInit(depths0=[0.5], fixedDepths=[False])
k.invert(forwardModel='CS', method='ROPE', alpha=0.07, beta=0.1, rep=300)
k.showResults(errorbar=True)


#%% from background survey (time-lapse)
k = Problem()
k.createTimeLapseSurvey(datadir + 'timelapse-wheat')
k.setInit(depths0=[0.5], fixedDepths=[False])
ss = []
for s in k.surveys[:2]:
    s.df = s.df[:20]
    ss.append(s)
k.surveys = ss
k.trimSurveys()
k.invert(method='ROPE', alpha=0.1, gamma=0.5)
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8,3))
k.showResults(index=0, rmse=True, ax=axs[0])
k.showResults(index=1, rmse=True, ax=axs[1])

# we need fixed depth to compute change
k.setInit(depths0=[0.5], fixedDepths=[True])
k.invert(alpha=0.1, gamma=0.5)
fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,3))
k.showResults(index=0, rmse=True, ax=axs[0])
k.showResults(index=1, rmse=True, ax=axs[1])
k.computeChange() # compute change in inverted EC
k.showResults(index=1, cmap='bwr', ax=axs[2])


#%%  parallel and sequential inversion
k = Problem()
k.createSurvey(datadir + 'cover-crop/coverCrop.csv')
# k.createSurvey(datadir + 'timelapse-wheat/170316.csv')
# k.calibrate(datadir + 'calib/dfeca.csv', datadir + 'calib/dfec.csv', apply=True)

t0 = time.time()
# k.invert(method='ROPE', njobs=-1)
k.invert(njobs=-1)
t1 = time.time()
k.showResults()


t2 = time.time()
# k.invert(method='ROPE', njobs=1)
k.invert(njobs=1)
print('SEQ elapsed {:.2f}s'.format(time.time() - t2))
print('PAR elapsed {:.2f}s'.format(time.time() - t0))
k.showResults()


#%% invert change in ECa
k = Problem()
k.createTimeLapseSurvey(datadir + 'timelapse-wheat')
k.surveys = k.surveys[:2]
k.computeApparentChange()
k.setInit(depths0=np.linspace(0.1, 2, 10))
k.invertGN()
k.showResults(index=1, cmap='bwr')


#%% mapping and save georeferenced slice
k = Problem()
k.createSurvey(datadir + 'saprolite/regolith.csv')
k.convertFromNMEA()
k.invertGN()
k.showSlice()
k.saveMap(fname=datadir + 'saprolite/map.tiff', method='idw')
k.saveSlice(fname=datadir + 'saprolite/slice.tiff')
k.saveInvData(datadir + 'saprolite/')
k.showSlice()
k.showDepths()


#%% forward modelling
nlayer = 2
npos = 20
conds = np.ones((npos, nlayer))*[20, 100]
x = np.linspace(0.1, 2, npos)[:,None]
depths = 0 + 2/(1+np.exp(-4*(x-1)))
# fig, ax = plt.subplots()
# ax.plot(depths, 'r-')
# fig.show()

coils0 = ['VCP1.48f10000h0', 'VCP2.82f10000h0', 'VCP4.49f10000h0',
          'HCP1.48f10000h0', 'HCP2.82f10000h0', 'HCP4.49f10000h0']

k1 = Problem()
k1.setModels([depths], [conds])
dfs = k1.forward(forwardModel='FSlin', coils=coils0, noise=0.05)
k1.show()
fig, axs = plt.subplots(1, 2, figsize=(8,3), sharex=True, sharey=True)
k1.showResults(ax=axs[0])
k1.setInit(depths0=[0.5], fixedDepths=[False])
k1.invert(method='ROPE')
k1.showResults(ax=axs[1], rmse=True)



#%% ANN inversion
# k = Problem()
# k.createSurvey(datadir + 'cover-crop/coverCrop.csv')
# k.setInit(depths0=[0.7], fixedDepths=[False])
# k.invert(method='ANN', nsample=100, noise=0.02, annplot=True)
# k.showResults(rmse=True)



