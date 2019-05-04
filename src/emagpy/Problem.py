#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 20:29:19 2019

@author: jkl
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from invertHelper import fCS, fMaxwellECa, fMaxwellQ, buildSecondDiff, buildJacobian#, getQs
from scipy.optimize import minimize
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
        self.depths0 = np.array([1, 2]) # initial depths of the bottom of each layer (last one is -inf)
        self.conds0 = np.array([20, 20, 20]) # initial conductivity for each layer
#        self.fixedConds = []
#        self.fixedDepths = []
        self.surveys = []
        self.models = []
        self.rmses = []
        
        
        
    def createSurvey(self, fname, freq=None, hx=None):
        ''' Create a survey object.
        
        Parameters
        ----------
        fname : str
            Path to the csv file with the data.
        freq : float, optional
            Frequency for all the coils (can also be specified for each coil in the file).
        hx : float, optional
            Height of the instrument above the ground (can also be specified for each coil in the file).
        '''
        survey = Survey(fname)
        if len(self.surveys) == 0:
            self.coils = survey.coils
            self.freqs = survey.freqs
            self.cspacing = survey.cspacing
            self.cpos = survey.cpos
            self.hx = survey.hx
            self.surveys.append(survey)
        else: # check we have the same configuration
            check = [a == b for a,b, in zip(self.coils, survey.coils)]
            if all(check) is True:
                self.surveys.append(survey)
        
        
    def createTimeLapseSurvey(self, dirname):
        ''' Create a list of surveys object.
        '''
        files = sorted(os.listdir(dirname))
        for f in files:
            self.createSurvey(os.path.join(dirname, f))
        
    
        
    def setDepths(self, depths):
        ''' Set the depths of the bottom of each layer. Last layer goes to -inf.
        Depths should be positive going down.
        '''
        if len(depths) == 0:
            raise ValueError('No depths specified.')
        if all(np.diff(depths) > 0):
            raise ValueError('Depths should be ordered and increasing.')
        self.depths = np.array(depths)
        
        
        
    def invert(self, forwardModel='CS', regularization='l2', smoothing='1d',
               alpha=0.07, dump=None, method='L-BFGS-B', **kwargs):
        '''Invert the apparent conductivity measurements.
        
        Parameters
        ----------
        forwardModel : str, optional
            Type of forward model:
                - CS : Cumulative sensitivity (default)
                - FS : Full Maxwell solution with low-induction number (LIN) approximation
                - FSandrade : Full Maxwell solution without LIN approximation (see Andrade 2016)
                - CSgn : Cumulative sensitivity with jacobian matrix (using Gauss-Newton)
                - CSgndiff : Cumulative sensitivty for difference inversion - NOT IMPLEMENTED YET
        regularization : str, optional
            Type of regularization, either l1 (blocky model) or l2 (smooth model)
        smoothing : str, optional
            Smoothing used (either 1d, 2d, or 3d).
        alpha : float, optional
            Smoothing factor for the inversion.
        dump : function, optional
            Function to print the progression. Default is `print`.
        method : str, optional
            Name of the optimization method for `scipy.optimize.minimize`.
        **kwargs : optional
            Additional keyword arguments will be passed to `scipy.optimize.minimize()`.
        '''
        if dump is None:
            dump = print
        
        if forwardModel in ['CS','FS','FSandrade']:
            # define the forward model
            if forwardModel == 'CS':
                def fmodel(p):
                    return fCS(p, self.depths0, self.cspacing, self.cpos, hx=self.hx[0])
            elif forwardModel == 'FS':
                def fmodel(p):
                    return fMaxwellECa(p, self.depths0, self.cspacing, self.cpos, f=self.freqs[0], hx=self.hx[0])
            elif forwardModel == 'FSandrade':
                def fmodel(p):
                    return fMaxwellQ(p, self.depths0, self.cspacing, self.cpos, f=self.freqs[0], hx=self.hx[0])
            
            # build objective function (RMSE based)
            L = buildSecondDiff(len(self.conds0)) # L is used inside the smooth objective fct

            def dataMisfit(p, app):
                return fmodel(p) - app
            def modelMisfit(p):
                return np.dot(L, p)
            
            if regularization  == 'l1':
                if smoothing == '1d':
                    def objfunc(p, app):
                        return np.sqrt(np.sum(np.abs(dataMisfit(p, app)))/len(app)
                                       + alpha*np.sum(np.abs(modelMisfit(p)))/len(p))
            elif regularization == 'l2':
                if smoothing == '1d':
                    def objfunc(p, app):
                        return np.sqrt(np.sum(dataMisfit(p, app)**2)/len(app)
                                       + alpha*np.sum(modelMisfit(p)**2)/len(p))
            # not sure about the division by len(app) for modelMisfit
                    
            # inversion row by row
            for i, survey in enumerate(self.surveys):
                apps = survey.df[self.coils].values
                rmse = np.zeros(apps.shape[0])*np.nan
                model = np.zeros((apps.shape[0], len(self.conds0)))*np.nan
                dump('Survey', i+1, '/', len(self.surveys))
                for j in range(survey.df.shape[0]):
                    app = apps[j,:]
#                    try:
                    res = minimize(objfunc, self.conds0, args=(app),
                                   method=method, **kwargs)
                    out = res.x
#                    except:
#                        out = np.ones(len(self.conds0))*np.nan
                    model[j,:] = out
                    rmse[j] = np.sqrt(np.sum(dataMisfit(out, app)**2)/len(app))
                    dump(j+1, '/', apps.shape[0], 'inverted')
                self.models.append(model)
                self.rmses.append(rmse)
                    
                    # TODO can add bounds
                    
    
    def invertGN(self, alpha=0.07, alpha_ref=None):
        '''Fast inversion usign Gauss-Newton and cumulative sensitivity.
        
        Parameters
        ----------
        alpha : float, optional
            Smoothing factor.
        alpha_ref : float, optional
            Only used for difference inversion to contrain the bottom of
            the profile to not changing (see Annex in Whalley et al., 2017).
        '''
        J = buildJacobian(self.depths0, self.cspacing, self.cpos)
        L = buildSecondDiff(J.shape[1])
        def fmodel(p):
            return fCS(p, self.depths0, self.cspacing, self.cpos, hx=self.hx[0])
        
        def dataMisfit(p, app):
            return fmodel(p) - app
        def modelMisfit(p):
            return np.dot(L, p)
        
        for i, survey in enumerate(self.surveys):
            apps = survey.df[self.coils].values
            rmse = np.zeros(apps.shape[0])*np.nan
            model = np.zeros((apps.shape[0], len(self.conds0)))*np.nan
            print('Survey', i+1, '/', len(self.surveys))
            for j in range(survey.df.shape[0]):
                app = apps[j,:]
                cond = np.ones((len(self.conds0),1))*np.nanmean(app) # initial EC is the mean of the apparent
                for l in range(3): # FIXME this is diverging with time ..;
                    d = dataMisfit(cond, app)
                    LHS = np.dot(J.T, J) + alpha*L
                    RHS = np.dot(J.T, d[:,None]) + alpha*np.dot(L, cond)
                    if alpha_ref is not None: # constraint the change of the last element of the profile
                        LHS[-1:,-1:] = alpha_ref
                        RHS[-1:] = alpha_ref*cond[i,-1]
                    solution = np.linalg.solve(LHS, RHS)
                    cond = cond + solution # it's an iterative process but it converges in one iteration as it's linear
                out = cond.flatten()
                model[j,:] = out
                rmse[j] = np.sqrt(np.sum(dataMisfit(out, app)**2)/len(app))
                print(j+1, '/', apps.shape[0], 'inverted')
            self.models.append(model)
            self.rmses.append(rmse)

            
                    
    def forward(self, forwardModel='CS'):
        '''Forward model.
        
        Parameters
        ----------
        forwardModel : str, optional
            Type of forward model:
                - CS : Cumulative sensitivity (default)
                - FS : Full Maxwell solution with low-induction number (LIN) approximation
                - FSandrade : Full Maxwell solution without LIN approximation (see Andrade 2016)
                - CSfast : Cumulative sensitivity with jacobian matrix (not minimize) - NOT IMPLEMENTED YET
                - CSdiff : Cumulative sensitivty for difference inversion - NOT IMPLEMENTED YET
        
        Returns
        -------
        df : pandas.DataFrame
            With the apparent ECa in the same format as input for the Survey class.
        '''
        if forwardModel in ['CS','FS','FSandrade']:
            # define the forward model
            if forwardModel == 'CS':
                def fmodel(p):
                    return fCS(p, self.depths0, self.cspacing, self.cpos, hx=self.hx[0])
            elif forwardModel == 'FS':
                def fmodel(p):
                    return fMaxwellECa(p, self.depths0, self.cspacing, self.cpos, f=self.freqs[0], hx=self.hx[0])
            elif forwardModel == 'FSandrade':
                def fmodel(p):
                    return fMaxwellQ(p, self.depths0, self.cspacing, self.cpos, f=self.freqs[0], hx=self.hx[0])
        
        dfs = []
        for i, model in enumerate(self.models):
            apps = np.zeros((model.shape[0], len(self.coils)))*np.nan
            for j in range(model.shape[0]):
                conds = model[j,:]
                apps[j,:] = fmodel(conds)
        
            df = pd.DataFrame(apps, columns=self.coils)
            dfs.append(df)
        
        return dfs
    
    
    
    def show(self, index=0, **kwargs):
        '''Show the raw data of the survey.
        
        Parameters
        ----------
        index : int, optional
            Survey number, by default, the first survey is chosen.
        '''
        self.surveys[index].show(**kwargs)
    
    
    
    def showMap(self, index=0, **kwargs):
        '''Show spatial map of the selected survey.
        
        Parameters
        ----------
        index : int, optional
            Survey number, by default, the first survey is chosen.
        '''
        self.surveys[index].showMap(**kwargs)
        
    
    
    def gridData(self, nx=100, ny=100, method='nearest'):
        ''' Grid data (for 3D).
        
        Parameters
        ----------
        nx : int, optional
            Number of points in x direction.
        ny : int, optional
            Number of points in y direction.
        method : str, optional
            Interpolation method (nearest, cubic or linear see
            `scipy.interpolate.griddata`). Default is `nearest`.
        '''
        
        for survey in self.surveys:
            survey.gridData(nx=nx, ny=ny, method=method)
        
        
    def convertFromNMEA(self,  targetProjection='EPSG:27700'): # British Grid 1936
        ''' Convert NMEA string to selected CRS projection.
        
        Parameters
        ----------
        targetProjection : str, optional
            Target CRS, in EPSG number: e.g. `targetProjection='EPSG:27700'`
            for the British Grid.
        '''
        for survey in self.surveys:
            survey.convertFromNMEA(targetProjection=targetProjection)
    
    
    def showResults(self, index=0, ax=None, vmin=None, vmax=None,
                    maxDepth=None, padding=0, cm='viridis'):
        '''Show invertd model.
        
        Parameters
        ----------
        index : int, optional
            Index of the survey to plot.
        '''            
        sig = self.models[index]
        x = np.arange(sig.shape[0])
        depths = np.repeat(self.depths0[:,None], sig.shape[0], axis=1).T
                
        if depths[0,0] != 0:
            depths = np.c_[np.zeros(depths.shape[0]), depths]
        if vmin is None:
            vmin = np.nanpercentile(sig, 5)
        if vmax is None:
            vmax = np.nanpercentile(sig, 95)
        cmap = plt.get_cmap(cm)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        if maxDepth is None:
            maxDepth = np.max(depths) + padding
        depths = np.c_[depths, np.ones(depths.shape[0])*maxDepth]
        h = np.diff(depths, axis=1)
        h = np.c_[np.zeros(h.shape[0]), h]
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        for i in range(1, h.shape[1]):
            ax.bar(x, -h[:,i], bottom=-np.sum(h[:,:i], axis=1),
                   color=cmap(norm(sig[:,i-1])), edgecolor='none', width=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, label='Conductivity [mS/m]')
        ax.set_xlabel('X position')
        ax.set_ylabel('Depth [m]')
        ax.set_title(self.surveys[index].name)
        ax.set_ylim([-maxDepth, 0])
#        ax.set_aspect('equal')
        def format_coord(i,j):
            col=int(np.floor(i))+1
            if col < sig.shape[0]:
                row = int(np.where(-depths[col,:] < j)[0].min())-1
                return 'x={0:.4f}, y={1:.4f}, value={2:.4f}'.format(col, row, sig[col, row])
            else:
                return ''
        ax.format_coord = format_coord
        fig.tight_layout()


    
    def showMisfit(self, index=0, coil='all', ax=None):
        '''Show Misfit after inversion.
            
        Parameters
        ----------
        index : int, optional
            Index of the survey to plot.
        coil : str, optional
            Which coil to plot. Default is all.
        ax : matplotlib.Axes, optional
            If specified the graph will be plotted on this axis.
        '''
        dfsForward = self.forward()
        survey = self.surveys[index]
        cols = survey.coils
        obsECa = survey.df[cols].values
        simECa = dfsForward[index][cols].values
        if ax is None:
            fig, ax = plt.subplots()
        xx = np.arange(survey.df.shape[0])
        ax.plot(xx, obsECa, '.')
        ax.set_prop_cycle(None)
        ax.plot(xx, simECa, '-')
        ax.legend(cols)
        
        
    def showOne2one(self, index=0, coil='all', ax=None, vmin=None, vmax=None):
        '''Show one to one plot with inversion results.
            
        Parameters
        ----------
        index : int, optional
            Index of the survey to plot.
        coil : str, optional
            Which coil to plot. Default is all.
        ax : matplotlib.Axes, optional
            If specified the graph will be plotted on this axis.
        vmin : float, optional
            Minimum ECa on the graph.
        vmax : float, optional
            Maximum ECa on the graph.
        '''
        dfsForward = self.forward()
        survey = self.surveys[index]
        cols = survey.coils
        obsECa = survey.df[cols].values
        simECa = dfsForward[index][cols].values
        if vmin is None:
            vmin = np.nanpercentile(obsECa.flatten(), 5)
        if vmax is None:
            vmax = np.nanpercentile(obsECa.flatten(), 95)
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(obsECa, simECa, '.')
        ax.plot([vmin, vmax], [vmin, vmax], 'k-', label='1:1')
        ax.set_xlim([vmin, vmax])
        ax.set_ylim([vmin, vmax])
        ax.set_xlabel('Observed ECa [mS/m]')
        ax.set_ylabel('Simulated ECa [mS/m]')
        ax.legend(cols)
    
    
    
    def lcurve(self, isurvey=0, irow=0, alphas=None, ax=None):
        '''Compute an L-curve given different values of alphas.
        
        Parameters
        ----------
        isurvey : int, optional
            Index of survey to be used, by default the first one.
        irow : int, optional
            Index of measurements to be used inside the survey. First by default.
        alpha : list or array-like, optional
            List or array of values of alphas to build the L-curve.
        ax : matplotlib.Axes, optional
            If specified, the graph will be plotted agains this axis.
        '''
        app = self.surveys[isurvey].df[self.coils].values[irow,:]
        if alphas is None:
            alphas = np.logspace(-3,2,20)
        def fmodel(p):
            return fCS(p, self.depths0, self.cspacing, self.cpos)
        L = buildSecondDiff(len(self.conds0))
        def dataMisfit(p, app):
            return fmodel(p) - app
        def modelMisfit(p):
            return np.dot(L, p)
        def objfunc(p, app, alpha):
            return np.sqrt(np.sum(dataMisfit(p, app)**2)/len(app)
                           + alpha*np.sum(modelMisfit(p)**2)/len(p))
        phiData = np.zeros(len(alphas))
        phiModel = np.zeros(len(alphas))
        for i, alpha in enumerate(alphas):
            res = minimize(objfunc, self.conds0, args=(app, alpha))
            phiData[i] = np.sum(dataMisfit(res.x, app)**2)
            phiModel[i] = np.sum(modelMisfit(res.x)**2)
            
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_title('L curve')
        ax.plot(phiModel, phiData, '.-')
        for a, ix, iy in zip(alphas, phiModel, phiData):
            ax.text(ix, iy, '{:.2f}'.format(a))
        ax.set_xlabel('Model Misfit ||L$\sigma$||$^2$')
        ax.set_ylabel('Data Misfit ||$\sigma_a - f(\sigma)$||$^2$')



    def calibrate(self, fnameECa, fnameEC, forwardModel='CS'):
        '''Calibrate ECa with given EC profile.
        
        Parameters
        ----------
        fnameECa : str
            Path of the .csv file with the ECa data collected on the calibration points.
        fnameEC : str
            Path of the .csv file with the EC profile data. One row per location
            corresponding to the rows of fnameECa. The header should be the
            corresponding depths in meters positive downards.
        forwardModel : str, optional
            Forward model to use. Either CS (default), FS or FSandrade.
        '''
        survey = Survey(fnameECa)
        dfec = pd.read_csv(fnameEC)
        if survey.df.shape[0] != dfec.shape[0]:
            raise ValueError('input ECa and inputEC should have the same number of rows so the measurements can be paired.')
        depths = dfec.columns.values.astype(float)
        
        # define the forward model
        if forwardModel == 'CS':
            def fmodel(p):
                return fCS(p, depths, survey.cspacing, survey.cpos, hx=survey.hx[0])
        elif forwardModel == 'FS':
            def fmodel(p):
                return fMaxwellECa(p, depths, survey.cspacing, survey.cpos, f=survey.freqs[0], hx=survey.hx[0])
        elif forwardModel == 'FSandrade':
            def fmodel(p):
                return fMaxwellQ(p, depths, survey.cspacing, survey.cpos, f=survey.freqs[0], hx=survey.hx[0])
    
        # compute the forward response
        simECa = np.zeros((dfec.shape[0], len(survey.coils)))
        for i in range(dfec.shape[0]):
            simECa[i,:] = fmodel(dfec.values[i,:])
        
        #TODO plot graph with subplots
        # plot equation, apply it or not directly
                    
        
        

if __name__ == '__main__':
    # cover crop example
    k = Problem()
    k.depths0 = np.linspace(0.5, 2, 10) # not starting at 0 !
    k.conds0 = np.ones(len(k.depths0)+1)*20
    k.createSurvey('test/coverCrop.csv')
    k.surveys[0].df 
    #k.show()
#    k.lcurve()
    k.invertGN()
#    k.invert()
#    k.showMisfit()
#    k.showResults()
    k.showOne2one()
#    k.forward()
    
    
    # mapping example (potatoes)
#    k = Problem()
#    k.createSurvey('test/potatoesLo.csv')
#    k.convertFromNMEA()
#    k.show()
#    k.gridData()
#    k.surveys[0].df = k.surveys[0].dfg
#    k.showMap(coil = k.coils[1])
    

