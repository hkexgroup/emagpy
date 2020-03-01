#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 20:29:19 2019

@author: jkl
"""
import os
import sys
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.optimize import minimize
from scipy.stats import linregress
from joblib import Parallel, delayed
from collections import defaultdict # for joblib monkey patching
import joblib # for joblib monkey patching

from emagpy.invertHelper import (fCS, fMaxwellECa, fMaxwellQ, buildSecondDiff,
                                 buildJacobian, getQs, eca2Q)
from emagpy.Survey import Survey, idw, clipConvexHull, griddata

class HiddenPrints:
    # https://stackoverflow.com/questions/8391411/suppress-calls-to-print-python
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        

class Problem(object):
    """Class defining an inversion problem.
    
    """
    def __init__(self):
        self.depths0 = np.array([0.5, 1.5]) # initial depths of the bottom of each layer (last one is -inf)
        self.conds0 = np.array([20.0, 20.0, 20.0]) # initial conductivity for each layer
        self.fixedConds = np.array([False, False, False])
        self.fixedDepths = np.array([True, True])
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
        # create Survey object
        survey = Survey(fname, freq=freq, hx=hx)
        
        # remove NaN from survey
        inan = np.zeros(survey.df.shape[0], dtype=bool)
        for c in survey.coils:
            inan = inan | (survey.df[c].isna())
        if np.sum(inan) > 0:
            print('Removing {:d} NaN from survey'.format(np.sum(inan)))
            survey.df = survey.df[~inan]
            
        # set attribut according to the first survey
        if len(self.surveys) == 0:
            self.coils = survey.coils
            self.freqs = survey.freqs
            self.cspacing = survey.cspacing
            self.cpos = survey.cpos
            self.hx = survey.hx
            self.surveys.append(survey)
        else: # check we have the same configuration than other survey
            check = [a == b for a,b, in zip(self.coils, survey.coils)]
            if all(check) is True:
                self.surveys.append(survey)
        
        
    def createTimeLapseSurvey(self, dirname):
        """ Create a list of surveys object.
        
        Parameters
        ----------
        dirname : str
            Directory with files to be parsed or list of file to be parsed.
        """
        files = dirname
        if isinstance(dirname, list): # it's a list of filename
            if len(dirname) < 2:
                raise ValueError('at least two files needed for timelapse inversion')
            files = dirname
        else: # it's a directory and we import all the files inside
            if os.path.isdir(dirname):
                files = [os.path.join(dirname, f) for f in np.sort(os.listdir(dirname)) if f[0] != '.']
                # this filter out hidden file as well
            else:
                raise ValueError('dirname should be a directory path or a list of filenames')
        for f in files:
            self.createSurvey(f)
        
        
    
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
        
        
    def _matchSurveys(self):
        """Return a list of indices for common measurements between surveys.
        """
        print('Matching positions between surveys for time-lapse inversion...', end='')
        t0 = time.time()
        dfs = [s.df for s in self.surveys]

        # sort all dataframe (should already be the case)
        dfs2 = []
        for df in dfs:
            dfs2.append(df)#.sort_values(by=['a','b','m','n']).reset_index(drop=True))

        # concatenate columns of string
        def cols2str(cols):
            cols = cols.astype(str)
            x = cols[:,0]
            for i in range(1, cols.shape[1]):
                x = np.core.defchararray.add(x, cols[:,i])
            return x

        # get measurements common to all surveys
        df0 = dfs2[0]
        x0 = cols2str(df0[['x','y']].values.astype(int))
        icommon = np.ones(len(x0), dtype=bool)
        for df in dfs2[1:]:
            x = cols2str(df[['x','y']].values.astype(int))
            ie = np.in1d(x0, x)
            icommon = icommon & ie
        print(np.sum(icommon), 'in common...', end='')

        # create boolean index to match those measurements
        indexes = []
        xcommon = x0[icommon]
        for df in dfs2:
            x = cols2str(df[['x','y']].values)
            indexes.append(np.in1d(x, xcommon))

        print('done in {:.3}s'.format(time.time()-t0))

        return indexes
    
    
    
    def trimSurveys(self):
        """Will trim all surveys to get them ready for difference inversion
        where all datasets must have the same number of measurements.
        """
        indexes = self._matchSurveys()
        for s, index in zip(self.surveys, indexes):
            s.df = s.df[index]
        
        
        
    def setDepths(self, depths):
        """ Set the depths of the bottom of each layer. Last layer goes to -inf.
        Depths should be positive going down.
        """
        if len(depths) == 0:
            raise ValueError('No depths specified.')
        if all(np.diff(depths) > 0):
            raise ValueError('Depths should be ordered and increasing.')
        self.depths = np.array(depths)
        
    
        
    def invert(self, forwardModel='CS', method='L-BFGS-B', regularization='l2',
               alpha=0.07, beta=0.0, gamma=0.0, dump=None, bnds=None,
               options={}, Lscaling=False, rep=100, njobs=1):
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
        method : str, optional
            Name of the optimization method either L-BFGS-B, TNC, CG or Nelder-Mead
            to be passed to `scipy.optimize.minmize()` or ROPE, SCEUA, DREAM for
            a MCMC-based solver based on the `spotpy` Python package.
        regularization : str, optional
            Type of regularization, either l1 (blocky model) or l2 (smooth model)
        alpha : float, optional
            Smoothing factor for the inversion.
        beta : float, optional
            Smoothing factor for neightbouring profile.
        gamma : float, optional
            Smoothing factor between surveys (for time-lapse only).
        dump : function, optional
            Function to print the progression. Default is `print`.
        bnds : list of float, optional
            If specified, will create bounds for the inversion. Doesn't work with
            Nelder-Mead solver.
        options : dict, optional
            Additional dictionary arguments will be passed to `scipy.optimize.minimize()`.
        Lscaling : bool, optional
            If True the regularization matrix will be weighted based on 
            centroids of layers differences.
        rep : int, optional
            Number of sample for the MCMC-based methods.
        njobs : int, optional
            If -1 all CPUs are used. If 1 is given, no parallel computing code
            is used at all, which is useful for debugging. For n_jobs below -1,
            (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs
            but one are used.
        """
        # switch in case Gauss-Newton routine is selected
        if forwardModel in ['CSgn', 'CSgndiff']:
            self.invertGN(alpha=alpha, alpha_ref=None, dump=dump)
            return
        
        self.models = []
        self.depths = []
        self.rmses = []
        self.ikill = False
        mMinimize = ['L-BFGS-B','TNC','CG','Nelder-Mead']
        mMCMC = ['ROPE','SCEUA','DREAM']
        
        if dump is None:
            def dump(x):
                print('\r' + x, end='')

        nc = len(self.conds0)
        vd = ~self.fixedDepths # variable depths
        vc = ~self.fixedConds # variable conductivity

        # define bounds
        if bnds is not None:
            if len(bnds) == 2 and (isinstance(bnds[0], int) or isinstance(bnds[0], float)):
                # we just have min/max of EC
                top = np.ones(nc)*bnds[1]
                bot = np.ones(nc)*bnds[0]
                bounds = list(tuple(zip(bot[vc], top[vc])))
            else:
                bounds = bnds
        else:
            bounds = None
        if ((np.sum(vd) > 0) or (method in mMCMC)) and (bounds is None):
            mdepths = self.depths0[:-1] + np.diff(self.depths0)/2
            bot = np.r_[np.r_[0.2, mdepths], np.ones(nc)*2]
            top = np.r_[np.r_[mdepths, self.depths0[-1] + 0.2], np.ones(nc)*100]
            bounds = list(tuple(zip(bot[np.r_[vd, vc]], top[np.r_[vd, vc]])))
        dump('bounds = ' + str(bounds) + '\n')

        # time-lapse constrain
        if gamma != 0:
            n = self.surveys[0].df.shape[0]
            for s in self.surveys[1:]:
                if s.df.shape[0] != n:
                    raise ValueError('For time-lapse constrain (gamma > 0), all surveys need to have the same length.')
                    gamma = 0
        
        # define the forward model
        def fmodel(p): # p contains first the depths then the conductivities
            depth = self.depths0
            if np.sum(vd) > 0:
                depth[vd] = p[:np.sum(vd)]
            cond = self.conds0.copy()
            if np.sum(vc) > 0:
                cond[vc] = p[np.sum(vd):]
            if forwardModel == 'CS':
                return fCS(cond, depth, self.cspacing, self.cpos, hx=self.hx[0])
            elif forwardModel == 'FS':
                return fMaxwellECa(cond, depth, self.cspacing, self.cpos, f=self.freqs[0], hx=self.hx[0])
            elif forwardModel == 'FSandrade':
                return fMaxwellQ(cond, depth, self.cspacing, self.cpos, f=self.freqs[0], hx=self.hx[0])
            elif forwardModel == 'Q':
                return np.imag(getQs(cond, depth, self.cspacing, self.cpos, f=self.freqs[0], hx=self.hx[0]))


        # build roughness matrix
        L = buildSecondDiff(len(self.conds0)) # L is used inside the smooth objective fct
        # each constrain is proportional to the distance between the centroid of the two layers
        if len(self.depths0) > 1:
            centroids = np.r_[self.depths0[0]/2, self.depths0[:-1] + np.diff(self.depths0)/2]
            if len(self.depths0) > 2:
                distCentroids = np.r_[centroids[1] - centroids[0],
                                     centroids[2:] - centroids[:-2],
                                     centroids[-1] - centroids[-2],
                                     1] # last layer is infinite so we don't apply any weights
            else:
                distCentroids = np.r_[centroids[1] - centroids[0],
                                      centroids[-1] - centroids[-2],
                                      1]
            if Lscaling is True:
                L = L/distCentroids[:,None]
        # TODO what for 3 layers ? sum of those distances ?

        # data misfit
        def dataMisfit(p, obs):
            misfit = fmodel(p) - obs
            if forwardModel == 'Q':
                misfit = misfit * 1e5 # to help the solver with small Q
            return misfit 
        
        # model misfit only for conductivities not depths
        def modelMisfit(p):
            cond = self.conds0.copy()
            if np.sum(vc) > 0:
                cond[vc] = p[:np.sum(vc)]
            return np.dot(L, cond)
        
        # set up regularisation
        # p : parameter, app : ECa,
        # pn : consecutive previous profile (for lateral smoothing)
        # spn : profile from other survey (for time-lapse)
        if regularization  == 'l1':
            def objfunc(p, app, pn, spn):
                return np.sqrt(np.sum(np.abs(dataMisfit(p, app)))/len(app)
                               + alpha*np.sum(np.abs(modelMisfit(p)))/nc
                               + beta*np.sum(np.abs(p - pn))/len(p)
                               + gamma*np.sum(np.abs(p - spn))/len(p))
        elif regularization == 'l2':
            def objfunc(p, app, pn, spn):
                return np.sqrt(np.sum(dataMisfit(p, app)**2)/len(app)
                               + alpha*np.sum(modelMisfit(p)**2)/nc
                               + beta*np.sum((p - pn)**2)/len(p)
                               + gamma*np.sum((p - spn)**2)/len(p))
            
        # define spotpy class if MCMC-based methods
        if method in mMCMC:
            try:
                import spotpy
            except ImportError:
                print('Please install spotpy to use MCMC-based methods.')
                return
            
            class spotpy_setup(object):
                def __init__(self, obsVals, fmodel, bounds, pn, spn):
                    self.params = []
                    for i, bnd in enumerate(bounds):
                        self.params.append(
                                spotpy.parameter.Uniform(
                                        'x{:d}'.format(i), bnd[0], bnd[1], 10, 10, bnd[0], bnd[1]))
                    self.obsVals = obsVals
                    self.fmodel = fmodel
                    self.pn = pn
                    self.spn = spn
                    
                def parameters(self):
                    return spotpy.parameter.generate(self.params)
            
                def simulation(self, vector):
                    x = np.array(vector)
                    #simulations = self.fmodel(x).flatten()
                    return x # trick return  parameters as the simulation is done in the
                    # objective function
                
                def evaluation(self): # what the function return when called with the optimal values
                    observations = self.obsVals.flatten()
                    return observations
                
                def objectivefunction(self, simulation, evaluation, params=None):
                    #val = -spotpy.objectivefunctions.rmse(evaluation, simulation)
                    # simulation is actually parameters, the simulation (forward model)
                    # is done inside the objective function itself
                    val = -objfunc(simulation, evaluation, self.pn, self.spn)
                    return val
        
        # check parallel
        if njobs != 1 and beta != 0:
            dump('WARNING: No parallel is possible with lateral smoothing (beta > 0).\n')
            njobs = 1
        
        # define optimization function
        x0 = np.r_[self.depths0[vd], self.conds0[vc]]
        def solve(obs, pn, spn):
            if self.ikill is True:
                raise ValueError('killed')
            if method in mMinimize: # minimize
                res = minimize(objfunc, x0, args=(obs, pn, spn),
                               method=method, bounds=bounds, options=options)
                out = res.x
                # status = 'converged' if res.success else 'not converged'
                # if res.success:
                #     c += 1      
            elif method in mMCMC: # MCMC based methods
                cols = ['parx{:d}'.format(a) for a in range(len(bounds))]
                spotpySetup = spotpy_setup(obs, fmodel, bounds, pn, spn)
                if method == 'ROPE':
                    sampler = spotpy.algorithms.rope(spotpySetup)
                elif method == 'DREAM':
                    sampler = spotpy.algorithms.dream(spotpySetup)
                elif method == 'SCEUA':
                    sampler = spotpy.algorithms.sceua(spotpySetup)
                else:
                    raise ValueError('Method {:s} unkown'.format(method))
                    return
                with HiddenPrints():
                    sampler.sample(rep) # this output a lot of stuff
                results = np.array(sampler.getdata())
                ibest = np.argmin(np.abs(results['like1']))
                out = np.array(list(results[ibest][cols]))
                # status = 'ok'
            return out

        # monkey patch joblib progress output (https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution)
        # patch joblib progress callback
        nrows = 0
        class BatchCompletionCallBack(object):
            completed = defaultdict(int)
            global nrows
            def __init__(self, time, index, parallel):
                self.index = index
                self.parallel = parallel
            def __call__(self, index):
                BatchCompletionCallBack.completed[self.parallel] += 1
                if BatchCompletionCallBack.completed[self.parallel] == nrows:
                    add = '\n'
                else:
                    add = ''
                dump('{:d}/{:d} inverted'.format(BatchCompletionCallBack.completed[self.parallel], nrows) + add)
                if self.parallel._original_iterator is not None:
                    self.parallel.dispatch_next()    
        joblib.parallel.BatchCompletionCallBack = BatchCompletionCallBack
        
        
        # inversion row by row
        c = 0 # number of inversion that converged
        for i, survey in enumerate(self.surveys):
            if self.ikill:
                break
            c = 0
            apps = survey.df[self.coils].values
            rmse = np.zeros(apps.shape[0])*np.nan
            model = np.ones((apps.shape[0], len(self.conds0)))*self.conds0
            depth = np.ones((apps.shape[0], len(self.depths0)))*self.depths0
            dump('Survey {:d}/{:d}\n'.format(i+1, len(self.surveys)))
            params = []
            for j in range(survey.df.shape[0]):
                # if self.ikill:
                #     break
                
                # define observations and convert to Q if needed
                obs = apps[j,:]
                if forwardModel == 'Q':
                    obs = np.array([eca2Q(a*1e-3, s) for a, s in zip(obs, self.cspacing)])
                
                # define previous profile in case we want to lateral constrain
                if j == 0:
                    pn = np.zeros(np.sum(np.r_[vd, vc]))
                else:
                    pn = np.r_[depth[j-1,:][vd], model[j-1,:][vc]]
                    
                # define profile from previous survey for time-lapse constrain
                if i == 0 or gamma == 0:
                    spn = np.zeros(np.sum(np.r_[vd, vc]))
                else: # constrain to the first inverted survey
                    spn = np.r_[self.depths[0][j,:][vd], self.models[0][j,:][vc]]

                params.append((obs, pn, spn))
            
            nrows = survey.df.shape[0]
            try: # if self.ikill is True, an error is raised inside solve that is catched here
                self.parallel = Parallel(n_jobs=njobs, verbose=50)
                outs = self.parallel(delayed(solve)(*a) for a in params)
                # backend multiprocessing inmpossible because local object
                # can not be pickled (only global can) however default locky works
            except ValueError:
                return
                
            for j, out in enumerate(outs):
                # store results from optimization
                depth[j,vd] = out[:np.sum(vd)]
                model[j,vc] = out[np.sum(vd):]
                rmse[j] = np.sqrt(np.sum(dataMisfit(out, obs)**2)/len(obs))
                # dump('{:d}/{:d} inverted ({:s})'.format(j+1, apps.shape[0], status))
            self.models.append(model)
            self.depths.append(depth)
            self.rmses.append(rmse)
            dump('{:d} measurements inverted\n'.format(apps.shape[0]))
                    
    # TODO add smoothing 3D: maybe invert all profiles once with GN and then
    # invert them again with a constrain on the 5 nearest profiles by distance

    
    
    def invertGN(self, alpha=0.07, alpha_ref=None, dump=None):
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
        
        if dump is None:
            def dump(x):
                print('\r' + x, end='')
                
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
            dump('Survey {:d}/{:d}\n'.format(i+1, len(self.surveys)))
            for j in range(survey.df.shape[0]):
                app = apps[j,:]
                cond = np.ones((len(self.conds0),1))*np.nanmean(app) # initial EC is the mean of the apparent (doesn't matter)
                # OR search for best starting model here
                for l in range(1): # FIXME this is diverging with time ..;
                    d = dataMisfit(cond, app)
                    LHS = np.dot(J.T, J) + alpha*L
                    RHS = np.dot(J.T, d[:,None]) - alpha*np.dot(L, cond) # minus or plus doesn't matter here ?!
                    if alpha_ref is not None: # constrain the change of the last element of the profile
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
            dump('{:d} measurements inverted\n'.format(apps.shape[0]))
           


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
            
    
    
    def filterDiff(self, coil=None, thresh=5):
        """Filter out consecutive measurements when the difference between them
        is larger than val.
        
        Parameters
        ----------
        thresh : float, optional
            Value of absolute consecutive difference above which the second 
            data point will be discarded.
        coil : str, optional
            Coil on which to apply the processing.
        """
        for s in self.surveys:
            s.filterDiff(coil=coil, thresh=thresh)
            
            
        
    def forward(self, forwardModel='CS', coils=None, noise=0.0):
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
        coils : list of str, optional
            If `None`, then the default attribute of the object will be used (foward
            mode on inverted solution).
            If specified, the coil spacing, orientation and height above the ground
            will be set. In this case you need to assign at models and depths (full forward mode).
            The ECa values generated will be incorporated as a new Survey object.
        noise : float, optional
            Percentage of noise to add on the generated apparent conductivities.
        
        Returns
        -------
        df : pandas.DataFrame
            With the apparent ECa in the same format as input for the Survey class.
        If `coils` argument is specified a new Survey object will be added as well.
        """
        if coils is None: # forward mode on inverted solution
            cspacing = self.cspacing
            cpos = self.cpos
            hxs = self.hx
            freqs = self.freqs
            iForward = False
        else: # full forward mode
            print('Forward modelling')
            iForward = True
            self.coils = coils
            self.depths0 = self.depths[0][0,:]
            self.conds0 = self.models[0][0,:]
            cspacing = []
            cpos = []
            hxs = []
            freqs = []
            for arg in coils:
                cpos.append(arg[:3].lower())
                b = arg[3:].split('f')
                cspacing.append(float(b[0]))
                if len(b) > 1:
                    c = b[1].split('h')
                    freqs.append(float(c[0]))
                    if len(c) > 1:
                        hxs.append(float(c[1]))
                    else:
                        hxs.append(0)
                else:
                    freqs.append(30000) # Hz default is not specified !!
                    hxs.append(0)
            self.cspacing = cspacing
            self.cpos = cpos
            self.hx = hxs
            self.freqs = freqs
            
        
        if forwardModel in ['CS','FS','FSandrade']:
            # define the forward model
            if forwardModel == 'CS':
                def fmodel(p, depth):
                    return fCS(p, depth, cspacing, cpos, hx=hxs[0])
            elif forwardModel == 'FS':
                def fmodel(p, depth):
                    return fMaxwellECa(p, depth, cspacing, cpos, f=freqs[0], hx=hxs[0])
            elif forwardModel == 'FSandrade':
                def fmodel(p, depth):
                    return fMaxwellQ(p, depth, cspacing, cpos, f=freqs[0], hx=hxs[0])
        
        def addnoise(x, level=0.05):
            return x + np.random.randn(len(x))*x*level
        
        dfs = []
        for i, model in enumerate(self.models):
            depths = self.depths[i]
            apps = np.zeros((model.shape[0], len(self.coils)))*np.nan
            for j in range(model.shape[0]):
                conds = model[j,:]
                depth = depths[j,:]
                apps[j,:] = addnoise(fmodel(conds, depth), level=noise)
        
            df = pd.DataFrame(apps, columns=self.coils)
            dfs.append(df)
        
        if iForward:
            self.surveys = []
            for df in dfs:
                s = Survey()
                s.readDF(df)
                self.surveys.append(s)
        
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
        coil = kwargs['coil'] if 'coil' in kwargs else None
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
    
    
    
    def saveSlice(self, fname, index=0, islice=0, nx=100, ny=100, method='nearest',
                xmin=None, xmax=None, ymin=None, ymax=None, color=False,
                cmap='viridis', vmin=None, vmax=None):
        """Save a georeferenced raster TIFF file for the specified inverted depths.
        
        Parameters
        ----------
        fname : str
            Path of where to save the .tiff file.
        index : int, optional
            Survey index. Default is first.
        islice : int, optional
            Depth index. Default is first depth.
        nx : int, optional
            Number of points in x direction.
        ny : int, optional
            Number of points in y direction.
        method : str, optional
            Interpolation method (nearest, cubic or linear see
            `scipy.interpolate.griddata`) or IDW (default).
        xmin : float, optional
            Mininum X value.
        xmax : float, optional
            Maximum X value.
        ymin : float, optional
            Minimium Y value.
        ymax : float, optional
            Maximum Y value
        color : bool, optional
            If True a colormap will be applied.
        cmap : str, optional
            If `color == True`, name of the colormap. Default is viridis.
        vmin : float, optional
            Minimum value for colomap.
        vmax : float, optional
            Maximum value for colormap.
        """
        try:
            import rasterio
            from rasterio.transform import from_origin
        except:
            raise ImportError('Rasterio is needed to save georeferenced .tif file. Install it using \'pip install rasterio\'')

        values = self.models[index][:,islice]
        xknown = self.surveys[index].df['x'].values
        yknown = self.surveys[index].df['y'].values

        if xmin is None:
            xmin = np.min(xknown)
        if xmax is None:
            xmax = np.max(xknown)
        if ymin is None:
            ymin = np.min(yknown)
        if ymax is None:
            ymax = np.max(yknown)
        X, Y = np.meshgrid(np.linspace(xmin, xmax, nx),
                           np.linspace(ymin, ymax, ny))
        x, y = X.flatten(), Y.flatten()
        if method == 'idw':
            z = idw(x, y, xknown, yknown, values)
            z = z.reshape(X.shape)
        else:
            z = griddata(np.c_[xknown, yknown], values, (X, Y), method=method)
        inside = np.ones(nx*ny)
        inside2 = clipConvexHull(xknown, yknown, x, y, inside)
        ie = np.isnan(inside2).reshape(z.shape)
        z[ie] = np.nan
        Z = np.flipud(z.T)
        
        # distance between corners
        dist0 = np.abs(xmax - xmin)
        dist1 = np.abs(ymax - ymin)        
    
        Z = np.fliplr(np.flipud(Z.T))
        yscale = dist1/Z.shape[0]
        xscale = dist0/Z.shape[1]
        
        tOffsetScaling = from_origin(xmin - xscale/2, ymax - yscale/2, xscale, yscale)
        tt = tOffsetScaling
        
        if color == True:
            if vmin is None:
                vmin = np.nanpercentile(Z.flatten(), 2)
            if vmax is None:
                vmax = np.nanpercentile(Z.flatten(), 98)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            Z = plt.get_cmap(cmap)(norm(Z))
            Z = 255*Z
            Z = Z.astype('uint8')
            for i in range(4):
                Z[np.fliplr(ie.T).T, i] = 0
        
            with rasterio.open(fname, 'w',
                           driver='GTiff',
                           height=Z.shape[0],
                           width=Z.shape[1], count=4, dtype=Z.dtype,
                           crs='epsg:27700', transform=tt) as dst:
                for i in range(4):
                    dst.write(Z[:,:,i], i+1)
        else:
            with rasterio.open(fname, 'w',
                               driver='GTiff',
                               height=Z.shape[0],
                               width=Z.shape[1], count=1, dtype=Z.dtype,
                               crs='epsg:27700', transform=tt) as dst:
                dst.write(Z, 1)
        
    
    def saveInvData(self, outputdir):
        """Save inverted data as one .csv per file with columns for
        layer conductivity (in mS/m) and columns for depth (in meters).
        The filename will be the same as the survey name prefixed with 'inv_'.
        
        Parameters
        ----------
        outputdir : str
            Path where the .csv files will be saved.
        """
        for i, survey in enumerate(self.surveys):
            fname = os.path.join(outputdir, 'inv_' + survey.name + '.csv')
            lcol = ['layer{:d}'.format(a+1) for a in range(len(self.conds0))]
            dcol = ['depth{:d}'.format(a+1) for a in range(len(self.depths0))]
            data = np.c_[survey.df[['x','y']].values, self.models[i], self.depths[i]]
            df = pd.DataFrame(data, columns=[['x','y'] + lcol + dcol])
            df.to_csv(fname, index=False)
    
    
    
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
        
    
    def setInit(self, depths0, conds0=None, fixedDepths=None, fixedConds=None):
        """Set the initial depths and conductivity for the inversion.
        
        Parameters
        ----------
        depths0 : list or array
            Depth as positive number of the bottom of each layer.
            There is N-1 depths for N layers as the last layer is infinite.
        conds0 : list or array, optional
            Starting conductivity in mS/m of each layer.
            By default a homogeneous conductivity of 20 mS/m is defined.
        fixedDepths : array of bool, optional
            Boolean array of same length as `depths0`. True if depth is fixed.
            False if it's a parameter. By default all depths are fixed.
        fixedConds : array of bool, optional
            Boolean array of same length as `conds0`. True if conductivity if fixed.
            False if it's a parameter. By default all conductivity are variable.'
        """
        depths0 = np.array(depths0)
        if np.sum(depths0 < 0) > 0:
            raise ValueError('All depth should be specified as positive number.')
        if np.sum(depths0 == 0) > 0:
            raise ValueError('No depth should be equals to 0 (infinitely thin layer)')
        if conds0 is None:
            conds0 = np.ones(len(depths0)+1)*20
        if fixedDepths is None:
            fixedDepths = np.ones(len(depths0), dtype=bool)
        if fixedConds is None:
            fixedConds = np.zeros(len(conds0), dtype=bool)
        if len(fixedConds) != len(conds0):
            raise ValueError('len(fixedConds) should be equal to len(conds0).')
        if len(fixedDepths) != len(depths0):
            raise ValueError('len(fixedDepths) should be equal to len(depths0)')    
        if len(depths0) + 1 != len(conds0):
            raise ValueError('length of conds0 should be equals to length of depths0 + 1')
        else:
            self.depths0 = depths0
            self.conds0 = np.array(conds0, dtype=float)
            self.fixedDepths = np.array(fixedDepths, dtype=bool)
            self.fixedConds = np.array(fixedConds, dtype=bool)


    
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
                    contour=False, rmse=False):
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
        rmse : bool, optional
            If `True`, the RMSE for each transect will be plotted on a second axis.
            Note that misfit can also be shown with `showMisfit()`.
        """            
        sig = self.models[index]
        x = np.arange(sig.shape[0])
#        x = np.sqrt(np.diff(self.surveys[index].df[['x', 'y']].values, axis=1)**2)
#        depths = np.repeat(self.depths0[:,None], sig.shape[0], axis=1).T
        depths = self.depths[index]
        
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
            fig.colorbar(coll, label='Conductivity [mS/m]', ax=ax)
        
        if rmse:
            ax2 = ax.twinx()
            xx = np.arange(len(self.rmses[index])) + 0.5
            ax2.plot(xx, self.rmses[index], 'kx-')
            ax2.set_ylabel('RMSE')

        ax.set_xlabel('Samples')
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


    def setModels(self, depths, models):
        """Set the models (depth-specific EC) and the depths. Use
        for forward modelling.
        
        Parameters
        ----------
        depths : list of array
            List of array. Each array is npos x ndepths. Each depth
            is a positive number representing the bottom of each layer.
            There is not depth 0.
        models : list of array
            List of array of size npos x nlayers. nlayer should be equals
            exactly to ndepths + 1. All specified in mS/m.
        """
        if len(depths) != len(models):
            raise ValueError('depths and models list should have the same length.')
        for m, d in zip(models, depths):
            if m.shape[1] != d.shape[1] + 1:
                raise ValueError('Number of layer should be equal to number of depths + 1 exactly.')
            if m.shape[0] != d.shape[0]:
                raise ValueError('Number of position in layers and depths array should match.')
        self.depths = depths
        self.models = models

    
    def computeApparentChange(self, ref=0):
        """Subtract the apparent conductivities of the reference survey
        to all other surveys. By default the reference survey is the 
        first survey. The survey can then be inverted with `invertGN()`.
        
        Parameters
        ----------
        ref : int, optional
            Index of the reference survey. By defaut the first survey
            is used as reference.
        """
        print('Trimming surveys and only keep common positions')
        self.trimSurveys()
        print('Computing relative ECa compared to background (1st survey).')
        background = self.surveys[ref].df[self.coils].values
        for i, s in enumerate(self.surveys):
            if i != ref: # the background survey stays the same
                s.df.loc[:,self.coils] = s.df[self.coils].values - background
        
        
    
    def computeChange(self, ref=0):
        """Compute the difference between inverted models compared
        to the ref model.
        
        Parameters
        ----------
        ref : int, optional
            Index of the reference model. By defaut the first model
            (corresponding to the first survey) is used as reference.
        """
        m0 = self.models[0]
        for m in self.models:
            m = m - m0
        
    
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
        #print('number of nan', np.sum(np.isnan(obsECa)), np.sum(np.isnan(simECa)))
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
    
    
    def filterRange(self, vmin=None, vmax=None):
        """Filter out measurements that are not between vmin and vmax.
        
        Parameters
        ----------
        vmin : float, optional
            Minimal ECa value, default is minimum observed.
        vmax : float, optional
            Maximum ECa value, default is maximum observed.
        """
        for s in self.surveys:
            s.filterRange(vmin=vmin, vmax=vmax)
    
    
    def filterPercentile(self, coil=None, qmin=None, qmax=None):
        """Filter out measurements based on percentile.
        
        Parameters
        ----------
        coil : str, optional
            Coil on which apply the filtering.
        qmin : float, optional
            Minimum percentile value below-which measurements are discarded.
        qmax : float, optional
            Maximum percentila value above-which measurements are discarded.
        """
        for s in self.surveys:
            s.filterPercentile(coil=coil, qmin=qmin, qmax=qmax)
    
    
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
        # TODO what about doing that for beta and gamma as well ?
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
            ax.set_xlim([np.nanmin(x), np.nanmax(x)])
            ax.set_ylim([np.nanmin(y), np.nanmax(y)])
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
        
        
        
    def showDepths(self, index=0, idepth=0, contour=False, vmin=None, vmax=None,
                  cmap='viridis_r', ax=None, pts=True):
        """Show depth slice.
        
        Parameters
        ----------
        index : int, optional
            Survey index. Default is first.
        idepth : int, optional
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
        z = self.depths[index][:,idepth]
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
            ax.set_xlim([np.nanmin(x), np.nanmax(x)])
            ax.set_ylim([np.nanmin(y), np.nanmax(y)])
        else:
            levels = np.linspace(vmin, vmax, 7)
            cax = ax.tricontourf(x, y, z, levels=levels, cmap=cmap, extend='both')
            if pts:
                ax.plot(x, y, 'k+')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        fig.colorbar(cax, ax=ax, label='Depth [m]')
        ax.set_title('Depths[{:d}]'.format(idepth))

        
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
