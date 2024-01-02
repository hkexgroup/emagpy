#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 20:36:28 2019

@author: jkl
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from emagpy import Problem
datadir = 'examples/'


#%% importing from GF instrument and filtering
k = Problem()
k.importGF(datadir + 'cover-crop/coverCropLo.dat',
           datadir + 'cover-crop/coverCropHi.dat')
k.importGF(datadir + 'cover-crop/coverCropLo.dat')
k.importGF(datadir + 'cover-crop/coverCropHi.dat')
k.importGF(datadir + 'cover-crop/coverCropLo.dat',
           datadir + 'cover-crop/coverCropHi.dat', device='CMD Explorer', calib='F-0m')
k.gfCorrection(calib='F-0m')

# filtering
k.filterRange(vmin=0, vmax=25)
k.filterPercentile(qmin=2, qmax=95)
k.rollingMean()
k.filterDiff()

k = Problem()
k.importGF(fnameLo=datadir + 'potatoes/potatoesLo.dat',
           fnameHi=datadir + 'potatoes/potatoesHi.dat',
           targetProjection='EPSG:27700')
           
k.showDist()
k.computeStat()
k.showMap()
k.filterBearing(phiMin=0, phiMax=25)
k.showMap()
k.filterRepeated(tolerance=1.5)
k.showMap()


#%% mapping potatoes field
k = Problem()
k.createMergedSurvey([datadir + 'potatoes/potatoesLo.csv',
                      datadir + 'potatoes/potatoesHi.csv'],
                     targetProjection='EPSG:27700', how='first')

k = Problem()
k.createMergedSurvey([datadir + 'potatoes/potatoesLo.csv',
                      datadir + 'potatoes/potatoesHi.csv'],
                     targetProjection='EPSG:27700', how='add')
df0 = k.surveys[0].df.copy()
k.surveys[0].driftCorrection(xStation=338379, yStation=405420, radius=5, fit='all')
k.surveys[0].driftCorrection(radius=5, fit='each')
k.surveys[0].driftCorrection(radius=5, fit='each', apply=True)
k.surveys[0].driftCorrection(xStation=338379, yStation=405420, radius=5, fit='all', apply=True)
k.surveys[0].driftCorrection(xStation=338379, yStation=405420, radius=5, fit='all')
# k.surveys[0].crossOverPointsDrift()
k.crossOverPointsError()
k.plotCrossOverMap()
k.showMap(contour=True, pts=True)
k.show()
k.gridData(method='cubic')
k.gridData(method='linear')
k.gridData(method='idw')
# k.gridData(method='kriging')
k.showMap()


#%% inversion with uncertainty
k = Problem()
k.createSurvey(datadir + 'cover-crop/coverCrop.csv')
k.surveys[0].df = k.surveys[0].df[:20]
k.setInit(depths0=[0.3, 0.7], fixedDepths=[True, False], fixedConds=[False, True, False])
k.invert(method='DREAM', rep=500, njobs=-1, bnds=[(0.1, 0.5),(0,50,),(0,50)])
k.showResults(errorbar=True, overlay=True)
k.showResults(errorbar=True, overlay=False, contour=True)
k.showProfile(errorbar=True)


#%% inversion
k = Problem()
k.createSurvey(datadir + 'cover-crop/coverCrop.csv')
k.surveys[0].df = k.surveys[0].df[:10] # only keep first measurements to make it faster
k.lcurve()

infos = []
for m in ['L-BFGS-B', 'ROPE']:
    for fm in ['CS','FSlin','FSeq','Q']:
        t0 = time.time()
        fig, axs = plt.subplots(1, 3, figsize=(10,3))
        alpha = 0 if fm == 'Q' else 0.07
        k.invert(forwardModel=fm, method=m, alpha=alpha, njobs=-1)
        elapsed = time.time() - t0
        k.showResults(ax=axs[2])
        k.showMisfit(ax=axs[1])
        k.showOne2one(ax=axs[0])
        fig.suptitle('{:s} {:s} ({:.2f}s)'.format(fm, m, elapsed))
        infos.append('{:8s} {:8s} ({:5.2f}s) RMSPE: {:6.2f} %'.format(
            fm, m, elapsed, k.getRMSE()['all'].values[0]))
        # fig.show()
print('\n'.join(infos))
# Q is quite sensitive to alpha

#%% testing l1
k = Problem()
k.createSurvey(datadir + 'cover-crop/coverCrop.csv')
k.surveys[0].df = k.surveys[0].df[:10] # only keep first measurements to make it faster
k.removeCoil(icoil=0)
k.removeCoil(coilName=k.coils[1])
k.invert(forwardModel='CS', method='DREAM', regularization='l1')

#%% testing grid parameter search (GPS) -- TODO finish the implementation
# k = Problem()
# k.createSurvey(datadir + 'cover-crop/coverCrop.csv')
# k.setInit(depths0=[0.4, 0.7, 1], conds0=[20, 20, 20, 20])
# k.invert(method='GPS', bnds=[(1, 100), (1, 100), (1, 100), (1, 100)])


#%% testing algorithms on synthetic data
nlayer = 2
npos = 10
conds = np.ones((npos, nlayer))*[20, 30]
x = np.linspace(0.1, 1.5, npos)[:,None]
depths = 0 + 2/(1+np.exp(-4*(x-1)))
# fig, ax = plt.subplots()
# ax.plot(depths, 'r-')
# fig.show()

coils0 = ['VCP1.48f10000h0', 'VCP2.82f10000h0', 'VCP4.49f10000h0',
          'HCP1.48f10000h0', 'HCP2.82f10000h0', 'HCP4.49f10000h0']

infos = []
for m in ['L-BFGS-B', 'ROPE']:
    for fm in ['CS', 'FSlin', 'FSeq','Q']:
        t0 = time.time()
        k = Problem()
        k.setModels([depths], [conds])
        dfs = k.forward(forwardModel='FSeq', coils=coils0, noise=0.0)
        fig, axs = plt.subplots(1, 2, figsize=(8,3), sharex=True, sharey=True)
        k.showResults(ax=axs[0])
        k.setInit(depths0=[0.5], fixedDepths=[False], conds0=[20, 30])
        k.invert(forwardModel=fm, method=m, alpha=0.01, njobs=-1)
        k.showResults(ax=axs[1], rmse=True)
        fig.suptitle('{:s} {:s} ({:.2f}s)'.format(fm, m, elapsed))
        infos.append('{:8s} {:8s} ({:5.2f}s) RMSPE: {:6.2f} %'.format(
            fm, m, elapsed, k.getRMSE()['all'].values[0]))
        # fig.show()
print('\n'.join(infos))



#%% testing gauss-Newton
k = Problem()
k.createSurvey(datadir + 'cover-crop/coverCropTransect.csv')
k.filterRange(vmax=50)
fig, axs = plt.subplots(3, 1, sharex=True)
ax = axs[0]
k.invert(forwardModel='CS', method='Gauss-Newton')
k.computeDOI()
k.showResults(ax=ax, rmse=True, vmin=10, vmax=30, doi=True)
ax.set_title('(a) CS with Gauss-Newton')
ax.set_xlabel('')

ax = axs[1]
k.invert(forwardModel='FSlin', method='Gauss-Newton')
k.showResults(ax=ax, rmse=True, vmin=10, vmax=30)
ax.set_title('(b) FSlin with Gauss-Newton')
ax.set_xlabel('')

ax = axs[2]
k.invert(forwardModel='FSeq', method='Gauss-Newton')
k.showResults(ax=ax, rmse=True, vmin=10, vmax=30)
ax.set_title('(c) FSeq with Gauss-Newton')


#%% test lateral smoothing
k = Problem()
k.createSurvey(datadir + 'cover-crop/coverCrop.csv')
k.surveys[0].df = k.surveys[0].df[:10]
k.setInit(depths0=[0.5], fixedDepths=[False])
k.invert(forwardModel='CS', method='SCEUA', alpha=0.07, beta=0.1, rep=300)
k.showResults(errorbar=True)


#%% from background survey (time-lapse)
# try a few variants
k = Problem()
try:
    k.createTimeLapseSurvey([
        datadir + 'timelapse-wheat/eca2017-03-16.csv'])
except Exception as e:
    print(e)

k = Problem()
k.createTimeLapseSurvey([
    datadir + 'timelapse-wheat/eca2017-03-16.csv',
    datadir + 'timelapse-wheat/eca2017-04-03.csv'])

k = Problem()
k.createTimeLapseSurvey(datadir + 'timelapse-wheat')
ss = []
for s in k.surveys[:2]:
    s.df = s.df[:20]
    ss.append(s)
k.surveys = ss
k.trimSurveys()
k.setInit(depths0=[0.5], fixedDepths=[False])
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
k.calibrate(datadir + 'calib/dfeca.csv', datadir + 'calib/dfec.csv',
            forwardModel='CS', apply=False)
k.calibrate(datadir + 'calib/dfeca.csv', datadir + 'calib/dfec.csv',
            forwardModel='FSlin', apply=False)
k.calibrate(datadir + 'calib/dfeca.csv', datadir + 'calib/dfec.csv',
            forwardModel='FSeq', apply=True)

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


#%% calibration with Boxford dataset
k = Problem()
k.createSurvey(datadir + 'boxford-calib/eca_calibration.csv')
k.calibrate(fnameECa=datadir + 'boxford-calib/eca_calibration.csv',
            fnameEC=datadir + 'boxford-calib/eri_ec.csv')
k.calibrate(fnameECa=datadir + 'boxford-calib/eca_calibration.csv', meshType='quad',
            fnameResMod=datadir + 'boxford-calib/quadmesh.dat', binInt=4)
# k.calibrate(fnameECa=datadir + 'boxford-calib/eca_calibration.csv',
#             fnameResMod=datadir + 'boxford-calib/trimesh.dat', binInt=4)


#%% invert change in ECa
k = Problem()
k.createTimeLapseSurvey(datadir + 'timelapse-wheat')
k.surveys = k.surveys[:2] # we reading the save models as well by default :/
k.interpData()
k.computeApparentChange()
k.setInit(depths0=np.linspace(0.1, 2, 10))
k.invert(forwardModel='CSgn')
k.getRMSE()
k.showResults(index=1, cmap='bwr', dist=True)

#%% check if project already set
k = Problem()
k.setProjection('EPSG:27700')
k.createSurvey(datadir + 'saprolite/regolith.csv')


#%% mapping and save georeferenced slice
k = Problem()
try:
    k.setInit(depths0=[2, 1]) # exception
    k.showResults() # exception
except:
    pass
k.createSurvey(datadir + 'saprolite/regolith.csv')
k.convertFromCoord()
k.invert(forwardModel='CSgn')
k.showSlice()
k.saveMap(fname=datadir + 'saprolite/map-idw.tiff', coils=[k.coils[0]], method='idw', color=True)
k.saveMap(fname=datadir + 'saprolite/map-idw.tiff', coils=[k.coils[0]], method='idw', color=False)
# k.saveMap(fname=datadir + 'saprolite/map-krig.tiff', method='kriging', color=True)
k.saveSlice(fname=datadir + 'saprolite/slice.tiff', color=True)
k.saveSlice(fname=datadir + 'saprolite/slice.tiff', color=False)
k.saveInvData(datadir + 'saprolite/')
k.saveData(datadir + 'saprolite/')
k.importModel(datadir + 'saprolite/inv_regolith.csv')
k.showSlice()
k.showSlice(contour=True, pts=True)

# remove artifacts
os.remove(datadir + 'saprolite/map-idw_' + k.coils[0] + '.tiff')
# os.remove(datadir + 'saprolite/map-krig.tiff')
os.remove(datadir + 'saprolite/slice.tiff')
os.remove(datadir + 'saprolite/inv_regolith.csv')
os.remove(datadir + 'saprolite/data_regolith.csv')


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
k1.setInit(depths0=[0.5], fixedDepths=[False], conds0=[20, 30])
k1.invert(method='ROPE')
k1.showResults(ax=axs[1], rmse=True)


#%% quasi3D inversion
k = Problem()
k.createSurvey(datadir + 'cover-crop/coverCrop.csv')
k.setInit(depths0=[0.5], fixedDepths=[False])
k.invert(threed=True, beta=0.1)
k.saveVTK(datadir + 'cover-crop/inv.vtk')
os.remove(datadir + 'cover-crop/inv.vtk')
try:
    import pyvista as pv
    pl = pv.BackgroundPlotter()
    k.show3D(pl=pl, edges=True, pvgrid=True)
    pl.clear()
    k.show3D(pl=pl, pvcontour=[20, 30, 40], pvgrid=True)
    pl.clear()
    k.show3D(pl=pl, pvslices=([5,10,15,20,25],[1.5],[0.5]), pvgrid=True)
except:
    pass
k.showDepths()
k.showDepths(contour=True, pts=True)


#%% ANN inversion
# k = Problem()
# k.createSurvey(datadir + 'cover-crop/coverCrop.csv')
# k.setInit(depths0=[0.7], fixedDepths=[False])
# k.invert(method='ANN', nsample=100, noise=0.02, annplot=True)
# k.showResults(rmse=True)


#%% uncertainty
nlayer = 2
npos = 5
conds = np.ones((npos, nlayer))*[20, 40]
depths = np.ones((npos, nlayer-1))*0.5
coils = ['HCP0.32','HCP0.71','HCP1.18','VCP0.32','VCP0.71','VCP1.18']
k = Problem()
k.setModels([depths], [conds])
dfs = k.forward(forwardModel='CS', coils=coils, noise=0.0)
k.setInit(depths0=[0.7], fixedDepths=[False], conds0=np.ones(conds.shape)*25)
bnds = [(0.01, 1), (5, 50), (5, 50)]
k.invert(forwardModel='CS', method='MCMC', rep=300, bnds=bnds, alpha=0, 
         regularization='l2', njobs=1)
k.showResults(rmse=True, errorbar=True)


