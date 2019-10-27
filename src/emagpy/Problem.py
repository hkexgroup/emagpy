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
from matplotlib.collections import PolyCollection
from scipy.optimize import minimize
from scipy.stats import linregress

from emagpy.invertHelper import (fCS, fMaxwellECa, fMaxwellQ, buildSecondDiff,
                                 buildJacobian, getQs, eca2Q, Q2eca2, Q2eca)
from emagpy.Survey import Survey


class Problem(object):
    """Class defining an inversion problem.
    
    """
    def __init__(self):
        self.depths0 = np.array([1, 2]) # initial depths of the bottom of each layer (last one is -inf)
        self.conds0 = np.array([20, 20, 20]) # initial conductivity for each layer
#        self.fixedConds = []
#        self.fixedDepths = []
        self.surveys = []
        self.models = [] # contains conds TODO rename to conds ?
        self.rmses = []
        self.freqs = []
        self.depths = [] # contains inverted depths or just depths0 if fixed
        self.ikill = False # if True, the inversion is killed
        
        
    def createSurvey(self, fname, freq=None, hx=None):
        """Create a survey object.
        
        Parameters
        ----------
        fname : str
            Path to the csv file with the data.
        freq : float, optional
            Frequency for all the coils (can also be specified for each coil in the file).
        hx : float, optional
            Height of the instrument above the ground (can also be specified for each coil in the file).
        """
        survey = Survey(fname, freq=freq, hx=hx)
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
        """ Create a list of surveys object.
        """
        files = sorted(os.listdir(dirname))
        for f in files:
            self.createSurvey(os.path.join(dirname, f))
        
    
    def importGF(self, fnameLo=None, fnameHi=None, device='CMD Mini-Explorer', hx=0):
        """Import GF instrument data with Lo and Hi file mode. If spatial data
        a regridding will be performed to match the data.
        
        Parameters
        ----------
        fnameLo : str
            Name of the file with the Lo settings.
        fnameHi : str
            Name of the file with the Hi settings.
        device : str, optional
            Type of device. Default is Mini-Explorer.
        hx : float, optional
            Height of the device above the ground in meters according to the
            calibration use (e.g. `F-Ground` -> 0 m, `F-1m` -> 1 m).
        """
        survey = Survey()
        survey.importGF(fnameLo, fnameHi, device, hx)
        self.coils = survey.coils
        self.freqs = survey.freqs
        self.cspacing = survey.cspacing
        self.cpos = survey.cpos
        self.hx = survey.hx
        self.surveys.append(survey)
        
        
    def setDepths(self, depths):
        """ Set the depths of the bottom of each layer. Last layer goes to -inf.
        Depths should be positive going down.
        """
        if len(depths) == 0:
            raise ValueError('No depths specified.')
        if all(np.diff(depths) > 0):
            raise ValueError('Depths should be ordered and increasing.')
        self.depths = np.array(depths)
        
    
        
    def invert(self, forwardModel='CS', regularization='l2', alpha=0.07,
               beta=0.0, dump=None, method='L-BFGS-B', bnds=None,
               fixedDepths=True, options={}):
        """Invert the apparent conductivity measurements.
        
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
        beta : float, optional
            Smoothing factor for neightbouring profile.
        dump : function, optional
            Function to print the progression. Default is `print`.
        method : str, optional
            Name of the optimization method for `scipy.optimize.minimize`.
        bnds : list of float, optional
            If specified, will create bounds for the inversion. Doesn't work with
            Nelder-Mead solver.
        fixedDepth : bool, optional
            If False, the depth will be considered as a parameter to be 
            optimized.
        options : optional
            Additional dictionary arguments will be passed to `scipy.optimize.minimize()`.
        """
        self.models = []
        self.depths = []
        self.rmses = []
        self.ikill = False
        
        if dump is None:
            dump = print
        
        if 'maxiter' not in options.keys():
            options = {'maxiter':15}
            
        if bnds is not None:
            top = np.ones(len(self.conds0))*bnds[1]
            bot = np.ones(len(self.conds0))*bnds[0]
            bounds = list(tuple(zip(bot, top)))
        else:
            bounds = None
        
        nc = len(self.conds0)
        
        if fixedDepths is False:
            mdepths = self.depths0[:-1] + np.diff(self.depths0)/2
            bot = np.r_[np.ones(nc)*2, np.r_[0.2, mdepths]]
            top = np.r_[np.ones(nc)*100, np.r_[mdepths, self.depths0[-1] + 0.2]]
            bounds = list(tuple(zip(bot, top)))
            print(bounds)
#            beta = 0 # not lateral smoothing with changing depths
        
        if forwardModel in ['CS','FS','FSandrade']:
            # define the forward model
            if forwardModel == 'CS':
                if fixedDepths is True:
                    def fmodel(p):
                        return fCS(p, self.depths0, self.cspacing, self.cpos, hx=self.hx[0])
                else:
                    def fmodel(p):
                        return fCS(p[:nc], p[nc:], self.cspacing, self.cpos, hx=self.hx[0])
            elif forwardModel == 'FS':
                if fixedDepths is True:
                    def fmodel(p):
                        return fMaxwellECa(p, self.depths0, self.cspacing, self.cpos, f=self.freqs[0], hx=self.hx[0])
                else:
                    def fmodel(p):
                        return fMaxwellECa(p[:nc], p[nc:], self.cspacing, self.cpos, f=self.freqs[0], hx=self.hx[0])
            elif forwardModel == 'FSandrade':
                if fixedDepths is True:
                    def fmodel(p):
                        return fMaxwellQ(p, self.depths0, self.cspacing, self.cpos, f=self.freqs[0], hx=self.hx[0])
                else:
                    def fmodel(p):
                        return fMaxwellQ(p[:nc], p[nc:], self.cspacing, self.cpos, f=self.freqs[0], hx=self.hx[0])

            # build objective function (RMSE based)
            L = buildSecondDiff(len(self.conds0)) # L is used inside the smooth objective fct

            def dataMisfit(p, app):
                return fmodel(p) - app
            if fixedDepths is True:
                def modelMisfit(p):
                    return np.dot(L, p)
            else:
                def modelMisfit(p): # model misfit only for conductivities
                    return np.dot(L, p[:nc])

            
            if regularization  == 'l1':
                def objfunc(p, app, pn):
                    return np.sqrt(np.sum(np.abs(dataMisfit(p, app)))/len(app)
                                   + alpha*np.sum(np.abs(modelMisfit(p)))/nc
                                   + beta*np.sum(np.abs(p[:nc] - pn))/len(p))
            elif regularization == 'l2':
                def objfunc(p, app, pn):
                    return np.sqrt(np.sum(dataMisfit(p, app)**2)/len(app)
                                   + alpha*np.sum(modelMisfit(p)**2)/nc
                                   + beta*np.sum((p[:nc] - pn)**2)/len(p))
                    
            # inversion row by row
            if fixedDepths is True:
                x0 = self.conds0
            else:
                x0 = np.r_[self.conds0, self.depths0]
            for i, survey in enumerate(self.surveys):
                if self.ikill:
                    break
                apps = survey.df[self.coils].values
                rmse = np.zeros(apps.shape[0])*np.nan
                model = np.zeros((apps.shape[0], len(self.conds0)))*np.nan
                depth = np.zeros((apps.shape[0], len(self.depths0)))*np.nan
                dump('Survey {:d}/{:d}'.format(i+1, len(self.surveys)))
                for j in range(survey.df.shape[0]): # this could be done in //
                    if self.ikill:
                        break
                    app = apps[j,:]
                    if j == 0:
                        pn = np.zeros(nc)
                    else:
                        pn = model[j-1,:]
#                    try:
                    res = minimize(objfunc, x0, args=(app, pn), tol=0.1,
                                   method=method, bounds=bounds, options=options)
                    out = res.x
#                    except ValueError as e:
#                        pass
#                        out = np.ones(len(self.conds0))*np.nan
                    if fixedDepths is False:
                        depth[j,:] = out[-len(self.depths0):]
                    else:
                        depth[j,:] = self.depths0
                    model[j,:] = out[:nc]
                    rmse[j] = np.sqrt(np.sum(dataMisfit(out, app)**2)/len(app))
                    status = 'converged' if res.success else 'not converged'
                    dump('{:d}/{:d} inverted ({:s})'.format(j+1, apps.shape[0], status))
                self.models.append(model)
                self.depths.append(depth)
                self.rmses.append(rmse)
        else:
            self.invertGN(alpha=alpha, alpha_ref=None, dump=dump)
                    
    # TODO add smoothing 3D: maybe invert all profiles once with GN and then
    # invert them again with a constrain on the 5 nearest profiles by distance
                    
    
    def invertGN(self, alpha=0.07, alpha_ref=None, dump=print):
        """Fast inversion usign Gauss-Newton and cumulative sensitivity.
        
        Parameters
        ----------
        alpha : float, optional
            Smoothing factor.
        alpha_ref : float, optional
            Only used for difference inversion to contrain the bottom of
            the profile to not changing (see Annex in Whalley et al., 2017).
        dump : function, optional
            Function to output the running inversion.
        """
        self.models = []
        self.rmses = []
        self.depths = []
        J = buildJacobian(self.depths0, self.cspacing, self.cpos)
        L = buildSecondDiff(J.shape[1])
        def fmodel(p):
            return fCS(p, self.depths0, self.cspacing, self.cpos, hx=self.hx[0])
        
        # fCS is automatically adding a leading 0 but not buildJacobian
        def dataMisfit(p, app):
            return app - fmodel(p)
        def modelMisfit(p):
            return np.dot(L, p)
        
        for i, survey in enumerate(self.surveys):
            apps = survey.df[self.coils].values
            rmse = np.zeros(apps.shape[0])*np.nan
            model = np.zeros((apps.shape[0], len(self.conds0)))*np.nan
            dump('Survey {:d}/{:d}'.format(i+1, len(self.surveys)))
            for j in range(survey.df.shape[0]):
                app = apps[j,:]
                cond = np.ones((len(self.conds0),1))*np.nanmean(app) # initial EC is the mean of the apparent (doesn't matter)
                # OR search for best starting model here
                for l in range(1): # FIXME this is diverging with time ..;
                    d = dataMisfit(cond, app)
                    LHS = np.dot(J.T, J) + alpha*L
                    RHS = np.dot(J.T, d[:,None]) - alpha*np.dot(L, cond) # minux or plus doesn't matter here ?!
                    if alpha_ref is not None: # constraint the change of the last element of the profile
                        LHS[-1:,-1:] = alpha_ref
                        RHS[-1:] = alpha_ref*cond[i,-1]
                    solution = np.linalg.solve(LHS, RHS)
                    cond = cond + solution # it's an iterative process but it converges in one iteration as it's linear
                out = cond.flatten()
                model[j,:] = out
                rmse[j] = np.sqrt(np.sum(dataMisfit(out, app)**2)/len(app))
                dump('{:d}/{:d} inverted'.format(j+1, apps.shape[0]))
            self.models.append(model)
            self.rmses.append(rmse)
            depth = np.repeat(self.depths0[None,:], apps.shape[0], axis=0)
            self.depths.append(depth)
            

    def invertQ(self, regularization='l2', alpha=1e-9,
               beta=0.0, dump=None, method='L-BFGS-B', bnds=None,
               fixedDepths=True, options={}):
        """Invert the apparent conductivity measurements by minimizing the
        quadrature Q.
        
        Parameters
        ----------
        regularization : str, optional
            Type of regularization, either l1 (blocky model) or l2 (smooth model)
        smoothing : str, optional
            Smoothing used (either 1d, 2d, or 3d).
        alpha : float, optional
            Smoothing factor for the inversion.
        beta : float, optional
            Smoothing factor for neightbouring profile.
        dump : function, optional
            Function to print the progression. Default is `print`.
        method : str, optional
            Name of the optimization method for `scipy.optimize.minimize`.
        bnds : list of float, optional
            If specified, will create bounds for the inversion. Doesn't work with
            Nelder-Mead solver.
        fixedDepth : bool, optional
            If False, the depth will be considered as a parameter to be 
            optimized.
        options : optional
            Additional dictionary arguments will be passed to `scipy.optimize.minimize()`.
        """
        self.models = []
        self.depths = []
        self.rmses = []
        self.ikill = False
        
        if dump is None:
            dump = print
        
        if 'maxiter' not in options.keys():
            options = {'maxiter':100}
            
        if bnds is not None:
            top = np.ones(len(self.conds0))*bnds[1]
            bot = np.ones(len(self.conds0))*bnds[0]
            bounds = list(tuple(zip(bot, top)))
        else:
            bounds = None
        
        nc = len(self.conds0)
        
        if fixedDepths is False:
            mdepths = self.depths0[:-1] + np.diff(self.depths0)/2
            bot = np.r_[np.ones(nc)*2, np.r_[0.2, mdepths]]
            top = np.r_[np.ones(nc)*100, np.r_[mdepths, self.depths0[-1] + 0.2]]
            bounds = list(tuple(zip(bot, top)))
        
        # define the forward model
        #WARNING assumption is made all coils got the same frequency
        if fixedDepths is True:
            def fmodel(p):
                return getQs(p, self.depths0, self.cspacing, self.cpos, f=self.freqs[0], hx=self.hx[0])
        else:
            def fmodel(p):
                return getQs(p[:nc], p[nc:], self.cspacing, self.cpos, f=self.freqs[0], hx=self.hx[0])

        # build objective function (RMSE based)
        L = buildSecondDiff(len(self.conds0)) # L is used inside the smooth objective fct

        def dataMisfit(p, q):
            return np.imag(fmodel(p)) - q
        if fixedDepths is True:
            def modelMisfit(p):
                return np.dot(L, p)
        else:
            def modelMisfit(p): # model misfit only for conductivities
                return np.dot(L, p[:nc])

        
        if regularization  == 'l1':
            def objfunc(p, app, pn):
                return np.sqrt(np.sum(np.abs(dataMisfit(p, app)))/len(app)
                               + alpha*np.sum(np.abs(modelMisfit(p)))/nc
                               + beta*np.sum(np.abs(p[:nc] - pn))/len(p))
        elif regularization == 'l2':
            def objfunc(p, app, pn):
                return np.sqrt(np.sum(dataMisfit(p, app)**2)/len(app)
                               + alpha*np.sum(modelMisfit(p)**2)/nc
                               + beta*np.sum((p[:nc] - pn)**2)/len(p))
                
        # inversion row by row
        if fixedDepths is True:
            x0 = self.conds0
        else:
            x0 = np.r_[self.conds0, self.depths0]
        for i, survey in enumerate(self.surveys):
            if self.ikill:
                break
            apps = survey.df[self.coils].values
            rmse = np.zeros(apps.shape[0])*np.nan
            model = np.zeros((apps.shape[0], len(self.conds0)))*np.nan
            depth = np.zeros((apps.shape[0], len(self.depths0)))*np.nan
            dump('Survey {:d}/{:d}'.format(i+1, len(self.surveys)))
            for j in range(survey.df.shape[0]):
                if self.ikill:
                    break
                app = apps[j,:]
                Qs = np.array([eca2Q(a*1e-3, s) for a, s in zip(app, self.cspacing)])
                if j == 0:
                    pn = np.zeros(nc)
                else:
                    pn = model[j-1,:]
#                    try:
                res = minimize(objfunc, x0, args=(Qs, pn),
                               method=method, bounds=bounds, options=options)
                out = res.x
#                    except ValueError as e:
#                        pass
#                        out = np.ones(len(self.conds0))*np.nan
                if fixedDepths is False:
                    depth[j,:] = out[-len(self.depths0):]
                else:
                    depth[j,:] = self.depths0
                model[j,:] = out[:nc]
                rmse[j] = np.sqrt(np.sum(dataMisfit(out, Qs)**2)/len(Qs))
                status = 'converged' if res.success else 'not converged'
                dump('{:d}/{:d} inverted ({:s})'.format(j+1, apps.shape[0], status))
            self.models.append(model)
            self.depths.append(depth)
            self.rmses.append(rmse)
                    


    def tcorrECa(self, tdepths, tprofile):
        """Temperature correction using XXXX formula.
        
        Parameters
        ----------
        tdepths : list of arrays
            Depths in meters of the temperature sensors (negative downards).
        tprofile : list of arrays
            Temperature values corresponding in degree Celsius.
        """
        for i, s in enumerate(self.surveys):
            s.tcorr(tdepths[i], tprofile[i])


            
    def tcorrEC(self, tdepths, tprofile):
        """Temperature correction for inverted models using XXXX formula.
        
        Parameters
        ----------
        tdepths : array-like
            Depths in meters of the temperature sensors (negative downards).
        tprofile : array-like
            Temperature values corresponding in degree Celsius.
        """
        for i, model in enumerate(self.models):
            pass
        #TODO correct ECa or inverted EC ? maybe let this to the user


    def write2vtk(self):
        """Write .vtk cloud points with the inverted models.
        """
        for i, m in enumerate(self.models):
            
            pass


    def rollingMean(self, window=3):
        """Perform a rolling mean on the data.
        
        Parameters
        ----------
        window : int, optional
            Size of the windows for rolling mean.
        """
        for survey in self.surveys:
            survey.rollingMean(window=window)
            
            
        
    def forward(self, forwardModel='CS'):
        """Forward model.
        
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
        """
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
        """Show the raw data of the survey.
        
        Parameters
        ----------
        index : int, optional
            Survey number, by default, the first survey is chosen.
        """
        coil = kwargs['coil'] if 'coil' in kwargs else 'all'
        vmin = kwargs['vmin'] if 'vmin' in kwargs else None
        vmax = kwargs['vmax'] if 'vmax' in kwargs else None
        ax = kwargs['ax'] if 'ax' in kwargs else None
        self.surveys[index].show(coil=coil, vmin=vmin, vmax=vmax, ax=ax)
    
    
    def showMap(self, index=0, **kwargs):
        """Show spatial map of the selected survey.
        
        Parameters
        ----------
        index : int, optional
            Survey number, by default, the first survey is chosen.
        """
        coil = kwargs['coil'] if 'coil' in kwargs else 'all'
        vmin = kwargs['vmin'] if 'vmin' in kwargs else None
        vmax = kwargs['vmax'] if 'vmax' in kwargs else None
        contour = kwargs['contour'] if 'contour' in kwargs else False
        pts = kwargs['pts'] if 'pts' in kwargs else False
        cmap = kwargs['cmap'] if 'cmap' in kwargs else 'viridis_r'
        ax = kwargs['ax'] if 'ax' in kwargs else None
        self.surveys[index].showMap(coil=coil, vmin=vmin, vmax=vmax,
                    contour=contour, pts=pts, cmap=cmap, ax=ax)
        
    
    def saveMap(self, index=0, **kwargs):
        """Save georefenced .tiff.
        
        Parameters
        ----------
        index : int, optional
            Survey number, by default, the first survey is chosen.
        """
        self.surveys[index].saveMap(**kwargs)
    
    
    def gridData(self, nx=100, ny=100, method='nearest'):
        """ Grid data (for 3D).
        
        Parameters
        ----------
        nx : int, optional
            Number of points in x direction.
        ny : int, optional
            Number of points in y direction.
        method : str, optional
            Interpolation method (nearest, cubic or linear see
            `scipy.interpolate.griddata`). Default is `nearest`.
        """
        
        for survey in self.surveys:
            survey.gridData(nx=nx, ny=ny, method=method)
        
        
    def convertFromNMEA(self,  targetProjection='EPSG:27700'): # British Grid 1936
        """ Convert NMEA string to selected CRS projection.
        
        Parameters
        ----------
        targetProjection : str, optional
            Target CRS, in EPSG number: e.g. `targetProjection='EPSG:27700'`
            for the British Grid.
        """
        for survey in self.surveys:
            survey.convertFromNMEA(targetProjection=targetProjection)
    
    
    
    def showResults_old(self, index=0, ax=None, vmin=None, vmax=None,
                    maxDepth=None, padding=1, cmap='viridis_r'):
        """Show invertd model.
        
        Parameters
        ----------
        index : int, optional
            Index of the survey to plot.
        ax : Matplotlib.Axes, optional
            If specified, the graph will be plotted against this axis.
        vmin : float, optional
            Minimum value of the colorbar.
        vmax : float, optional
            Maximum value of the colorbar.
        maxDepth : float, optional
            Maximum negative depths of the graph.
        padding : float, optional
            Thickness of the bottom infinite layer in [m].
        cmap : str, optional
            Name of the Matplotlib colormap to use.
        """            
        sig = self.models[index]
        x = np.arange(sig.shape[0])
#        depths = np.repeat(self.depths0[:,None], sig.shape[0], axis=1).T
        depths = self.depths[0]      
        
        if depths[0,0] != 0:
            depths = np.c_[np.zeros(depths.shape[0]), depths]
        if vmin is None:
            vmin = np.nanpercentile(sig, 5)
        if vmax is None:
            vmax = np.nanpercentile(sig, 95)
        cmap = plt.get_cmap(cmap)
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
         # optimization doesn't seem to work maybe it needs only increasing arrays...   
#        ax.bar(np.tile(x, h.shape[1]-1), -h[:,1:].flatten('F'), bottom=-np.cumsum(h, axis=1)[:,1:].flatten('F'),
#               color=cmap(norm(sig.flatten('F'))), edgecolor='none', width=1)
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


    
    def showResults(self, index=0, ax=None, vmin=None, vmax=None,
                    maxDepth=None, padding=1, cmap='viridis_r',
                    contour=False):
        """Show inverted model.
        
        Parameters
        ----------
        index : int, optional
            Index of the survey to plot.
        ax : Matplotlib.Axes, optional
            If specified, the graph will be plotted against this axis.
        vmin : float, optional
            Minimum value of the colorbar.
        vmax : float, optional
            Maximum value of the colorbar.
        maxDepth : float, optional
            Maximum negative depths of the graph.
        padding : float, optional
            DONT'T KNOW
        cmap : str, optional
            Name of the Matplotlib colormap to use.
        contour : bool, optional
            If `True` a contour plot will be plotted.
        """            
        sig = self.models[index]
        x = np.arange(sig.shape[0])
#        x = np.sqrt(np.diff(self.surveys[index].df[['x', 'y']].values, axis=1)**2)
#        depths = np.repeat(self.depths0[:,None], sig.shape[0], axis=1).T
        depths = self.depths[0]
        
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        if depths[0,0] != 0:
            depths = np.c_[np.zeros(depths.shape[0]), depths]
        if vmin is None:
            vmin = np.nanpercentile(sig, 5)
        if vmax is None:
            vmax = np.nanpercentile(sig, 95)
        cmap = plt.get_cmap(cmap)
        if maxDepth is None:
            maxDepth = np.max(depths) + padding
        depths = np.c_[depths, np.ones(depths.shape[0])*maxDepth]
        
        # vertices
        nlayer = sig.shape[1]
        nsample = sig.shape[0]
        x2 = np.arange(nsample+1)
#        dist = np.sqrt((self.surveys[index].df['x'].values - self.surveys[index].df['y'].values)**2)
        xs = np.tile(np.repeat(x2, 2)[1:-1][:,None], nlayer+1)
        ys = -np.repeat(depths, 2, axis=0)
        vertices = np.c_[xs.flatten('F'), ys.flatten('F')]
        
        # connection matrix
        n = vertices.shape[0]
        connection = np.c_[np.arange(n).reshape(-1,2),
                           2*nsample + np.arange(n).reshape(-1,2)[:,::-1]]
#        ie1 = connection[:,0] % (2*len(x)-2) == 0
        ie2 = (connection >= len(vertices)).any(1)
        ie = ie2
        connection = connection[~ie, :]
        coordinates = vertices[connection]
        
        # plotting
        if contour is True:
            centroid = np.mean(coordinates, axis=1)
            x = np.r_[centroid[:nsample,0], centroid[:,0], centroid[:nsample,0]]
            y = np.r_[np.zeros(nsample), centroid[:,1], -np.ones(nsample)*maxDepth]
            z = np.c_[sig[:,0], sig, sig[:,-1]]
            if vmax > vmin:
                levels = np.linspace(vmin, vmax, 7)
            else:
                levels = None
            cax = ax.tricontourf(x, y, z.flatten('F'),
                                 cmap=cmap, levels=levels, extend='both')
#            ax.plot(x, y, 'k+')
            fig.colorbar(cax, ax=ax, label='Conductivity [mS/m]')
        else:
    #        ax.plot(vertices[:,0], vertices[:,1], 'k.')
            coll = PolyCollection(coordinates, array=sig.flatten('F'), cmap=cmap)
            coll.set_clim(vmin=vmin, vmax=vmax)
            ax.add_collection(coll)
            fig.colorbar(coll, label='Conductivity [mS/m]')

        ax.set_xlabel('X position')
        ax.set_ylabel('Depth [m]')
        ax.set_title(self.surveys[index].name)
        ax.set_ylim([-maxDepth, 0])
        ax.set_xlim([x[0], x[-1]+1])
#        ax.set_aspect('equal')
        def format_coord(i,j):
            col=int(np.floor(i))
            if col < sig.shape[0]:
                row = int(np.where(-depths[col,:] < j)[0].min())-1
                return 'x={0:.4f}, y={1:.4f}, value={2:.4f}'.format(col, row, sig[col, row])
            else:
                return ''
        ax.format_coord = format_coord
        fig.tight_layout()


    
    def getRMSE(self):
        """Returns RMSE for all coils (columns) and all surveys (row).
        """
        dfsForward = self.forward()
        def rmse(x, y):
            return np.sqrt(np.sum((x - y)**2)/len(x))
        
        dfrmse = pd.DataFrame(columns=np.r_[self.coils, ['all']])
        for i in range(len(self.surveys)):
            survey = self.surveys[i]
            for coil in self.coils:
                obsECa = survey.df[coil].values
                simECa = dfsForward[i][coil].values
                dfrmse.loc[i, coil] = rmse(obsECa, simECa)
            obsECa = survey.df[self.coils].values.flatten()
            simECa = dfsForward[i][self.coils].values.flatten()
            dfrmse.loc[i, 'all'] = rmse(obsECa, simECa)
        
        return dfrmse
        
        
        
    def showMisfit(self, index=0, coil='all', ax=None):
        """Show Misfit after inversion.
            
        Parameters
        ----------
        index : int, optional
            Index of the survey to plot.
        coil : str, optional
            Which coil to plot. Default is all.
        ax : matplotlib.Axes, optional
            If specified the graph will be plotted on this axis.
        """
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
        ax.plot(xx, simECa, '^-')
        ax.legend(cols)
        ax.set_xlabel('Measurements')
        ax.set_ylabel('Conductivity [mS/m]')
        ax.set_title('Dots (observed) vs triangles (modelled)')
        
        
        
    def showOne2one(self, index=0, coil='all', ax=None, vmin=None, vmax=None):
        """Show one to one plot with inversion results.
            
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
        """
        dfsForward = self.forward()
        survey = self.surveys[index]
        cols = survey.coils
        obsECa = survey.df[cols].values
        simECa = dfsForward[index][cols].values
        rmse = np.sqrt(np.sum((obsECa.flatten() - simECa.flatten())**2)/len(obsECa.flatten()))
        rmses = np.sqrt(np.sum((obsECa - simECa)**2, axis=0)/obsECa.shape[0])
        if vmin is None:
            vmin = np.nanpercentile(obsECa.flatten(), 5)
        if vmax is None:
            vmax = np.nanpercentile(obsECa.flatten(), 95)
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_title('RMSE: {:.3f}'.format(rmse))
        ax.plot(obsECa, simECa, '.')
        ax.plot([vmin, vmax], [vmin, vmax], 'k-', label='1:1')
        ax.set_xlim([vmin, vmax])
        ax.set_ylim([vmin, vmax])
        ax.set_xlabel('Observed ECa [mS/m]')
        ax.set_ylabel('Simulated ECa [mS/m]')
        ax.legend(['{:s} ({:.2f})'.format(c, r) for c, r in zip(cols, rmses)])
    
    
    def keepBetween(self, vmin=None, vmax=None):
        """Filter out measurements that are not between vmin and vmax.
        
        Parameters
        ----------
        vmin : float, optional
            Minimal ECa value, default is minimum observed.
        vmax : float, optional
            Maximum ECa value, default is maximum observed.
        """
        for s in self.surveys:
            s.keepBetween(vmin=vmin, vmax=vmax)
    
    
    
    def lcurve(self, isurvey=0, irow=0, alphas=None, ax=None):
        """Compute an L-curve given different values of alphas.
        
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
        """
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



    def calibrate(self, fnameECa, fnameEC, forwardModel='CS', ax=None):
        """Calibrate ECa with given EC profile.
        
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
        ax : matplotlib.Axes
            If specified the graph will be plotted against this axis.
        """
        survey = Survey(fnameECa)
        if survey.freqs[0] is None: # fallback in case the use doesn't specify the frequency in the headers
            try:
                survey.freqs = np.ones(len(survey.freqs))*self.freqs[0]
                print('EMI frequency not specified in headers, will use the one from the main data:' + str(self.freqs[0]) + 'Hz')
            except:
                print('Frequency not found, revert to CS')
                forwardModel = 'CS' # doesn't need frequency
        dfec = pd.read_csv(fnameEC)
        if survey.df.shape[0] != dfec.shape[0]:
            raise ValueError('input ECa and inputEC should have the same number of rows so the measurements can be paired.')
        depths = np.abs(dfec.columns.values.astype(float)) # those are the depths of at mid layer
        depths = depths[:-1] + np.diff(depths) # those are depths of the bottom of the layer
        
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
        
        # graph
        obsECa = survey.df[survey.coils].values
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(obsECa, simECa, '.')
        x = np.r_[obsECa.flatten(), simECa.flatten()]
        x = x[~np.isnan(x)]
        vmin, vmax = np.nanmin(x), np.nanmax(x)
        ax.plot([vmin, vmax], [vmin, vmax], 'k-', label='1:1')
        ax.set_xlim([vmin, vmax])
        ax.set_ylim([vmin, vmax])
        ax.set_xlabel('ECa(EM) [mS/m]')
        ax.set_ylabel('ECa(ER) [mS/m]')
        ax.legend(survey.coils)
        
        # plot equation, apply it or not directly
        predECa = np.zeros(obsECa.shape)
        for i, coil in enumerate(survey.coils):
            x, y = obsECa[:,i], simECa[:,i]
            inan = ~np.isnan(x)& ~np.isnan(y)
            slope, intercept, r_value, p_value, std_err = linregress(x[inan], y[inan])
            print(coil, '{:.2f} * x + {:.2f} (R={:.2f})'.format(slope, intercept, r_value))
            predECa[:,i] = obsECa[:,i]*slope + intercept
        ax.set_prop_cycle(None)
        ax.plot(obsECa, predECa, '-')
        
        
    def crossOverPoints(self, index=0, coil=None, ax=None, dump=print):
        """ Build an error model based on the cross-over points.
        
        Parameters
        ----------
        index : int, optional
            Survey index to fit the model on. Default is the first.
        coil : str, optional
            Name of the coil.
        ax : Matplotlib.Axes, optional
            Matplotlib axis on which the plot is plotted against if specified.
        dump : function, optional
            Function to print the output.
        """
        survey = self.surveys[index]
        survey.crossOverPoints(coil=coil, ax=ax, dump=dump)
    
    
    
    def plotCrossOverMap(self, index=0, coil=None, ax=None):
        """Plot the map of the cross-over points for error model.
        
        Parameters
        ----------
        index : int, optional
            Survey index to fit the model on. Default is the first.
        coil : str, optional
            Name of the coil.
        ax : Matplotlib.Axes, optional
            Matplotlib axis on which the plot is plotted against if specified.
        """
        survey = self.surveys[index]
        survey.plotCrossOverMap(coil=coil, ax=ax)
        
    
    def showSlice(self, index=0, islice=0, contour=False, vmin=None, vmax=None,
                  cmap='viridis_r', ax=None, pts=True):
        """Show depth slice.
        
        Parameters
        ----------
        index : int, optional
            Survey index. Default is first.
        islice : int, optional
            Depth index. Default is first depth.
        contour : bool, optional
            If `True` then there will be contouring.
        vmin : float, optional
            Minimum value for colorscale.
        vmax : float, optional
            Maximum value for colorscale.
        cmap : str, optional
            Name of colormap. Default is viridis_r.
        ax : Matplotlib.Axes, optional
            If specified, the graph will be plotted against it.
        pts : boolean, optional
            If `True` (default) the data points will be plotted over the contour.
        """
        z = self.models[index][:,islice]
        x = self.surveys[index].df['x'].values
        y = self.surveys[index].df['y'].values
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        if vmin is None:
            vmin = np.nanmin(z)
        if vmax is None:
            vmax = np.nanmax(z)
        if contour is False or (y == y[0]).all() or (x == x[0]).all(): # can not contour if they are on a line
            if contour is True:
                print('All points on a line, can not contour this.')
            cax = ax.scatter(x, y, c=z, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            levels = np.linspace(vmin, vmax, 7)
            cax = ax.tricontourf(x, y, z, levels=levels, cmap=cmap, extend='both')
            if pts:
                ax.plot(x, y, 'k+')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        fig.colorbar(cax, ax=ax, label='Conductivity [mS/m]')
        depths = np.r_[[0], self.depths0, [-np.inf]]
        ax.set_title('{:.2f}m - {:.2f}m'.format(depths[islice], depths[islice+1]))
        
        
#%%
if __name__ == '__main__':
    # cover crop example
    k = Problem()
    k.depths0 = np.array([0.2, 1]) # not starting at 0 !
    k.conds0 = np.ones(len(k.depths0)+1)*20
    k.createSurvey('test/coverCrop.csv', freq=30000)
#    k.convertFromNMEA()
#    k.createSurvey('test/warren170316.csv', freq=30000)
    k.surveys[0].df = k.surveys[0].df[:10]
#    k.show()
#    k.lcurve()
#    k.invertGN(alpha=0.07)
    k.invert(method='CG')
#    k.invert(forwardModel='CS', alpha=0.07, method='L-BFGS-B', options={'maxiter':100}, beta=0, fixedDepths=False) # this doesn't work well
#    k.invertGN() # similar as CG with nit=2
#    k.invertQ()
#    k.showMisfit()
#    k.showResults(vmin=30, vmax=50)
    k.showOne2one()
#    k.showMisfit()
#    k.models[0] = np.ones(k.models[0].shape)*20
#    k.forward(forwardModel='FSandrade')
#    k.calibrate('test/dfeca.csv', 'test/dfec.csv', forwardModel='FS') # TODO
    
#    k.showSlice(contour=False, cmap='jet', vmin=10, vmax=50)
    
    #%% test for inversion with FSandrade
    import time
    cond = np.array([10, 20, 30, 30])
#    app = fMaxwellQ(cond, k.depths0, k.cspacing, k.cpos, hx=k.hx[0], f=k.freqs[0])
#    app = fMaxwellECa(cond, k.depths0, k.cspacing, k.cpos, hx=k.hx[0], f=k.freqs[0])
    app = k.surveys[0].df[k.coils].values[0,:]
    L = buildSecondDiff(len(cond))
    def objfunc(p, app):
        return np.sqrt((np.sum((app - fMaxwellECa(p, k.depths0, k.cspacing, k.cpos, hx=k.hx[0], f=k.freqs[0]))**2)
                              + 0.07*np.sum(np.dot(L, p[:,None])**2))/len(app))
    t0 = time.time()
    res = minimize(objfunc, k.conds0, args=(app,), method='Nelder-Mead', options={'maxiter':10})
    print(res)
    print('{:.3f}s'.format(time.time() - t0))
    
    #%%
    solvers = ['Nelder-Mead', 'Powell', 'CG', 'BFGS',
               'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP']
    tt = []
    for solver in solvers:
        print(solver)
        t0 = time.time()
        res = minimize(objfunc, k.conds0, args=(app,), method=solver)
        tt.append([time.time() - t0, res.nfev, res.fun])
    
    tt = np.vstack(tt)
    xx = np.arange(len(solvers))
    fig, ax = plt.subplots()
    ax.plot(xx, tt)
    ax.set_xticks(xx)
    ax.set_xticklabels(solvers, rotation=90)
    fig.tight_layout()
    fig.show()
    
#%%   see if we can speed-up minimize by decreasing iteration number
    nits = [0, 1, 2, 3, 10, 20, 50, 100, 1000]
    for nit in nits:
        t0 = time.time()
        res = minimize(objfunc, k.conds0, args=(app,), method='Nelder-Mead',
                       options={'maxiter':nit})
        elapsed = time.time() - t0
        print(res.x)
        rmseEC = np.sqrt(np.sum((res.x - cond)**2)/len(cond))
        rmseECa = np.sqrt(np.sum((fMaxwellECa(res.x, k.depths0, k.cspacing, k.cpos, hx=k.hx[0], f=k.freqs[0]) - app)**2)/len(app))
        print('nit={:d} in {:.3f}s with RMSE={:.2f}'.format(nit, elapsed, rmseECa))
    
    
    # mapping example (potatoes)
#    k = Problem()
#    k.createSurvey('test/regolith.csv')
#    k.convertFromNMEA()
#    k.showMap(contour=True, pts=True)
#    k.show()
#    k.gridData(method='cubic')
#    k.surveys[0].df = k.surveys[0].dfg
#    k.showMap(coil = k.coils[1])
#    

#%% GF direct import
#    k = Problem()
#    k.importGF('emagpy/test/coverCropLo.dat', 'emagpy/test/coverCropHi.dat')
#    k.invertGN()
#    k.showSlice(contour=True)
