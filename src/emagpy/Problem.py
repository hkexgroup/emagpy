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
from matplotlib.colors import ListedColormap
from scipy.optimize import minimize
from scipy.stats import linregress

# for parallel computing
from joblib import Parallel, delayed
from tqdm import tqdm

# emagpy custom import
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
        self.conds0 = [] # initial conductivity for each layer
        self.fixedConds = []
        self.fixedDepths = []
        self.surveys = []
        self.models = [] # TODO rename to conds?
        self.pstds = [] # parameter standard deviation for MCMC inversion
        self.rmses = []
        self.freqs = []
        self.depths = [] # contains inverted depths or just depths0 if fixed
        self.ikill = False # if True, the inversion is killed
        self.c = 0 # counter
        self.calibrated = False # flag for ERT calibration
        self.annReplaced = 0 # number of measurement outliers by ANN
        self.runningUI = False # True if run in UI, just change output of parallel stuff
        
        
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
            
        # set attribute according to the first survey
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
        for i, survey in enumerate(self.surveys):
            survey.df = survey.df[indexes[i]]
        
        
    
    def importModel(self, fname):
        """Import a save model from previous inversion.
        
        Parameters
        ----------
        fname : str
            Path of the .csv file.
        """
        df = pd.read_csv(fname)
        ccols = [c for c in df.columns if c[:5] == 'layer']
        dcols = [c for c in df.columns if c[:5] == 'depth']
        if len(ccols) != len(dcols) + 1:
            print('Number of depths should be number of layer - 1')
        conds = df[ccols].values
        depths = df[dcols].values
        self.setModels([depths], [conds])
    
    
        
    def invert(self, forwardModel='CS', method='L-BFGS-B', regularization='l1',
               alpha=0.07, beta=0.0, gamma=0.0, dump=None, bnds=None,
               options={}, Lscaling=False, rep=100, noise=0.05, nsample=100, 
               annplot=False, njobs=1):
        """Invert the apparent conductivity measurements.
        
        Parameters
        ----------
        forwardModel : str, optional
            Type of forward model:
                - CS : Cumulative sensitivity (default)
                - FSlin : Full Maxwell solution with low-induction number (LIN) approximation
                - FSeq : Full Maxwell solution without LIN approximation (see Andrade et al., 2016)
                - CSgn : Cumulative sensitivity with jacobian matrix (using Gauss-Newton)
                - CSgndiff : Cumulative sensitivty for difference inversion - NOT IMPLEMENTED YET
        method : str, optional
            Name of the optimization method either L-BFGS-B, TNC, CG or Nelder-Mead
            to be passed to `scipy.optimize.minmize()` or ROPE, SCEUA, DREAM, MCMC for
            a MCMC-based solver based on the `spotpy` Python package.
            Alternatively 'ANN' can be used (requires tensorflow), it will train
            an artificial neural network on synthetic data and use it for inversion.
            Note that smoothing (alpha, beta, gamma) is not supported by ANN.
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
            **Experimental feature** If True the regularization matrix will be weighted based on 
            centroids of layers differences.
        rep : int, optional
            Number of sample for the MCMC-based methods.
        noise : float, optional
            If ANN method is used, describe the noise applied to synthetic data
            in training phase. Values between 0 and 1 (100% noise).
        nsample : int, optional
            If ANN method is used, describe the size of the synthetic data
            generated in trainig phase.
        annplot : bool, optional
            If True, the validation plot will be plotted.
        njobs : int, optional
            If -1 all CPUs are used. If 1 is given, no parallel computing code
            is used at all, which is useful for debugging. For n_jobs below -1,
            (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs
            but one are used.
        """
        # check if we have initial model
        if len(self.conds0) == 0:
            self.setInit(self.depths0)
            
        # switch in case Gauss-Newton routine is selected
        if forwardModel in ['CSgn', 'CSgndiff']:
            self.invertGN(alpha=alpha, alpha_ref=None, dump=dump)
            return
        
        self.models = []
        self.depths = []
        self.rmses = []
        self.pstds = []
        self.annReplaced = 0
        self.ikill = False
        mMinimize = ['L-BFGS-B','TNC','CG','Nelder-Mead']
        mMCMC = ['ROPE','SCEUA','DREAM', 'MCMC']
        nc = len(self.conds0) # number of layers
        vd = ~self.fixedDepths # variable depths
        vc = ~self.fixedConds # variable conductivity
        
        if dump is None:
            def dump(x):
                print(x, end='')

        # if (njobs != 1) and (method in mMCMC):
        #     dump('WARNING: parallel execution is currently not supported for {:s}.'
        #          'Reverting to sequential execution.'.format(method))
        #     njobs = 1

        nc = self.conds0[0].shape[1] # number of layers
        nd = self.depths0[0].shape[1] # number of depths
        vd = ~self.fixedDepths # variable depths
        vc = ~self.fixedConds # variable conductivity
        depths0 = self.depths0[0][0,:] # approximation

        # time-lapse constrain
        if gamma != 0:
            n = self.surveys[0].df.shape[0]
            for s in self.surveys[1:]:
                if s.df.shape[0] != n:
                    raise ValueError('For time-lapse constrain (gamma > 0), all surveys need to have the same length.')
                    gamma = 0
        
        # define the forward model
        def fmodel(p, ini0): # p contains first the depths then the conductivities
            depth = ini0[0].copy()
            cond = ini0[1].copy()
            if np.sum(vd) > 0:
                depth[vd] = p[:np.sum(vd)]
            if np.sum(vc) > 0:
                cond[vc] = p[np.sum(vd):]
            if forwardModel == 'CS':
                return fCS(cond, depth, self.cspacing, self.cpos, hx=self.hx)
            elif forwardModel == 'FSlin':
                return fMaxwellECa(cond, depth, self.cspacing, self.cpos, f=self.freqs, hx=self.hx)
            elif forwardModel == 'FSeq':
                return fMaxwellQ(cond, depth, self.cspacing, self.cpos, f=self.freqs, hx=self.hx)
            elif forwardModel == 'Q':
                return np.imag(getQs(cond, depth, self.cspacing, self.cpos, f=self.freqs, hx=self.hx))

        # define bounds
        if bnds is not None:
            if len(bnds) == 2 and (isinstance(bnds[0], int) or isinstance(bnds[0], float)):
                # we just have min/max of EC
                top = np.ones(nc)*bnds[1]
                bot = np.ones(nc)*bnds[0]
                bounds = list(tuple(zip(bot[vc], top[vc])))
            else:
                bounds = bnds
            # check initial values lies within bounds
            # d0 = self.depths0[vd]
            # s0 = self.conds0[vc]
            # p0 = np.r_[d0, s0]
            # for p, bnd in zip(p0, bounds):
                # if (p < bnd[0]) or (p > bnd[1]):
                    # dump('ERROR: initial parameters values {:s} are out of the given bounds.'
                          # 'Please use Problem.setInit() to define initial values or modify the boudns.'.format(str(p0)))
                    # return
        else:
            bounds = None
        if ((np.sum(vd) > 0) or (method in mMCMC)) and (bounds is None):
            # for MCMC method or fixed depths, we need bounds
            mdepths = depths0[:-1] + np.diff(depths0)/2
            bot = np.r_[np.r_[0.2, mdepths], np.ones(nc)*2]
            top = np.r_[np.r_[mdepths, depths0[-1] + 0.2], np.ones(nc)*100]
            bounds = list(tuple(zip(bot[np.r_[vd, vc]], top[np.r_[vd, vc]])))
        # if bounds is not None:
            # dump('bounds = ' + str(bounds) + '\n')

        # build ANN network
        if method == 'ANN':
            if gamma != 0 or beta != 0:
                dump('ANN does not accept any smoothing parameters.')
            dump('Building and training ANN network\n')
            t0 = time.time()
            if bounds is None: # happen where all depths are fixed
                vmin = np.nanpercentile(self.surveys[0].df[self.coils].values, 2)
                vmax = np.nanpercentile(self.surveys[0].df[self.coils].values, 98)
                bounds = list(tuple(zip(np.ones(nc)*vmin, np.ones(nc)*vmax)))
                dump('bounds = ' + str(bounds) + '\n')
            self.buildANN(fmodel, bounds, noise=noise, nsample=nsample, dump=dump, iplot=annplot)
            dump('Finish training the network ({:.2f}s)\n'.format(time.time() - t0))
            
        # build roughness matrix
        L = buildSecondDiff(nc) # L is used inside the smooth objective fct
        # each constrain is proportional to the distance between the centroid of the two layers
        if nd > 1:
            centroids = np.r_[depths0[0]/2, depths0[:-1] + np.diff(depths0)/2]
            if nd > 2:
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
        def dataMisfit(p, obs, ini0):
            misfit = fmodel(p, ini0) - obs
            if forwardModel == 'Q':
                misfit = misfit * 1e5 # to help the solver with small Q
            return misfit 
        
        # model misfit only for conductivities not depths
        def modelMisfit(p):
            cond = self.conds0[0][0,:].copy()
            if np.sum(vc) > 0:
                cond[vc] = p[:np.sum(vc)]
            return np.dot(L, cond)
        
        # set up regularisation
        # p : parameter, app : ECa,
        # pn : consecutive previous profile (for lateral smoothing)
        # spn : profile from other survey (for time-lapse)
        if regularization  == 'l1':
            def objfunc(p, app, pn, spn, alpha, beta, gamma, ini0):
                return np.sqrt(np.sum(np.abs(dataMisfit(p, app, ini0)))/len(app)
                               + alpha*np.sum(np.abs(modelMisfit(p)))/nc
                               + beta*np.sum(np.abs(p - pn))/len(p)
                               + gamma*np.sum(np.abs(p - spn))/len(p))
        elif regularization == 'l2':
            def objfunc(p, app, pn, spn, alpha, beta, gamma, ini0):
                return np.sqrt(np.sum(dataMisfit(p, app, ini0)**2)/len(app)
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
                def __init__(self, obsVals, bounds, pn, spn, alpha, beta, gamma, ini0):
                    self.params = []
                    for i, bnd in enumerate(bounds):
                        self.params.append(
                                spotpy.parameter.Uniform(
                                        'x{:d}'.format(i), bnd[0], bnd[1], 10, 10, bnd[0], bnd[1]))
                    self.obsVals = obsVals
                    self.pn = pn
                    self.spn = spn
                    self.alpha = alpha
                    self.beta = beta
                    self.gamma = gamma
                    self.ini0 = ini0
                    
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
                    val = -objfunc(simulation, evaluation, self.pn, self.spn,
                                   self.alpha, self.beta, self.gamma, self.ini0)
                    return val
        
        # check parallel
        if njobs != 1 and beta != 0:
            dump('WARNING: No parallel is possible with lateral smoothing (beta > 0).\n')
            njobs = 1
        
        # define optimization function
        def solve(obs, pn, spn, alpha, beta, gamma, ini0):
            if self.ikill is True:
                raise ValueError('killed') # https://github.com/joblib/joblib/issues/356
            if method in mMinimize: # minimize
                x0 = np.r_[ini0[0][vd], ini0[1][vc]]
                res = minimize(objfunc, x0, args=(obs, pn, spn, alpha, beta, gamma, ini0),
                               method=method, bounds=bounds, options=options)
                out = res.x
                # status = 'converged' if res.success else 'not converged'
                # if res.success:
                #     c += 1      
            elif method in mMCMC: # MCMC based methods
                cols = ['parx{:d}'.format(a) for a in range(len(bounds))]
                spotpySetup = spotpy_setup(obs, bounds, pn, spn, alpha, beta, gamma, ini0)
                if method == 'ROPE':
                    sampler = spotpy.algorithms.rope(spotpySetup)
                elif method == 'DREAM':
                    sampler = spotpy.algorithms.dream(spotpySetup)
                elif method == 'SCEUA':
                    sampler = spotpy.algorithms.sceua(spotpySetup)
                elif method == 'MCMC':
                    sampler = spotpy.algorithms.mcmc(spotpySetup)
                else:
                    raise ValueError('Method {:s} unkown'.format(method))
                    return
                # using hiddenPrints() here cause issue for // computing
                # because the save file (stdout) is one and close in //
                sampler.sample(rep) # this is outputing too much so we use hiddenprints() context
                results = np.array(sampler.getdata())
                ibest = np.argmin(np.abs(results['like1']))
                outval = np.array([results[col][ibest] for col in cols])
                outstd = np.array([np.nanstd(results[col]) for col in cols])
                out = (outval, outstd)
                # status = 'ok'
                
            return out

        # inversion row by row
        for i, survey in enumerate(self.surveys):
            if self.ikill:
                break
            self.c = 0
            apps = survey.df[self.coils].values
            rmse = np.zeros(apps.shape[0])*np.nan
            model = self.conds0[i].copy()
            depth = self.depths0[i].copy()
            dump('Survey {:d}/{:d}\n'.format(i+1, len(self.surveys)))
            params = []
            nrows = survey.df.shape[0]
            for j in range(nrows):
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
                    g = 0
                else: # constrain to the first inverted survey
                    spn = np.r_[self.depths[0][j,:][vd], self.models[0][j,:][vc]]
                    g = gamma
                    
                # initial values
                ini0 = (self.depths0[i][j,:], self.conds0[i][j,:])
                
                params.append((obs, pn, spn, alpha, beta, g, ini0))                
            
            outs = []
            # sequential
            if (method != 'ANN') & (njobs == 1): # sequential inversion (default)
                for j in range(nrows):
                    with HiddenPrints():
                        outs.append(solve(*params[j]))
                    dump('\r{:d}/{:d} inverted'.format(j+1, nrows))
            
            # parallel computing with locky backend
            elif (method != 'ANN') & (njobs != 1):
                try:
                    with HiddenPrints():
                        outs = Parallel(n_jobs=njobs, verbose=0)(delayed(solve)(*a) for a in tqdm(params))
                except Exception: # might be when we kill it using UI
                    return
                
           # artifical neural network inversion 
            elif method == 'ANN':
                obss = np.vstack([a[0] for a in params])
                normobss = self.norm(obss) # normalize data
                outs = self.model.predict(normobss)
                # detecting negative depths
                ibad1 = np.array([outs[:,l] < bounds[l][0] for l in range(len(bounds))]).any(0)
                ibad2 = np.array([outs[:,l] > bounds[l][1] for l in range(len(bounds))]).any(0)
                ibad = ibad1 | ibad2
                self.annReplaced = np.sum(ibad)
                if np.sum(ibad) > 0:
                    dump('WARNING: ANN: {:d} values out of bounds replaced by L-BFGS-G values.'
                         ' Try to increase the noise level and/or the number of samples.\n.'.format(
                             np.sum(ibad)))
                    for l in np.where(ibad)[0]:
                        if self.ikill:
                            return
                        x0 = np.r_[self.depths0[i][l,vd], self.conds0[i][l,vc]]
                        res = minimize(objfunc, x0, args=params[l],
                                       method='L-BFGS-B', bounds=bounds, 
                                       options=options)
                        outs[l,:] = res.x
                outs = list(outs)

            # store results from optimization                
            for j, outt in enumerate(outs):
                obs = params[j][0]
                if method in mMCMC:
                    std = outt[1]
                    stds[j,:] = std
                    out = outt[0]
                else:
                    out = outt
                depth[j,vd] = out[:np.sum(vd)]
                model[j,vc] = out[np.sum(vd):]
                rmse[j] = np.sqrt(np.sum(dataMisfit(out, obs, params[j][-1])**2)/len(obs))
            dump('\n')
            self.models.append(model)
            self.depths.append(depth)
            self.rmses.append(rmse)
            self.pstds.append(stds)
            # dump('{:d} measurements inverted\n'.format(apps.shape[0]))
                    
    # TODO add smoothing 3D: maybe invert all profiles once with GN and then
    # invert them again with a constrain on the 5 nearest profiles by distance

    
    def buildANN(self, fmodel, bounds, noise=0.05, iplot=False, nsample=100,
                 dump=None, epochs=500):
        """Build and train the artificial neural network on synthetic values
        derived from observed ECa values.
        
        Parameters
        ----------
        fmodel : function
            Function use to generated the synthetic data.
        bounds : list of tuple
            List of tuple with the bounds of each parameters to pass to the
            `fmodel` function.
        noise : float, optional
            Noise level to apply on the synthetic data generated.
        iplot : bool, optional
            If True, the validation graph will be plotted.
        nsample : int, optional
            Number of samples to be synthetically generated.
        dump : function, optional
            Function to dump information.
        epochs : int, optional
            Number of epochs.
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
        except:
            raise ImportError('Tensorflow is needed for NN inversion.')
            return
        
        if dump is None:
            def dump(x):
                print(x, end='')
            
        def addnoise(x):
            return x + np.random.randn(len(x))*x*noise
        
        # build a set of synthetic data based on bounds
        nc = len(bounds)
        ini0 = (self.depths0[0][0,:].copy(), self.conds0[0][0,:].copy())
        param = np.zeros((nsample, nc))
        for i, bnd in enumerate(bounds):
            param[:,i] = np.random.uniform(bnd[0], bnd[1], nsample)
            
        eca = np.zeros((nsample, len(self.coils)))
        for i in range(nsample):
            eca[i,:] = addnoise(fmodel(param[i,:], ini0))
        
        vcols = list(self.coils.copy())
        pcols = ['p{:d}'.format(i+1) for i in range(nc)]
        df = pd.DataFrame(np.c_[eca, param], columns=vcols+pcols)

        # split dataset in training and testing
        df_train = df.sample(frac=0.8, random_state=0)
        df_test = df.drop(df_train.index)
        train_labels = df_train[pcols].values
        test_labels = df_test[pcols].values
        train_dataset = df_train[vcols].values
        test_dataset = df_test[vcols].values
        
        # normalize dataset using statistics from train data
        # NOTE: this need to be done when feeding other data to it
        normMean = np.mean(train_dataset, axis=0)
        normStd = np.std(train_dataset, axis=0)
        def norm(x):
            return (x - normMean) / normStd
        self.norm = norm # store function to be used in invert()
        normed_train_data = self.norm(train_dataset)
        normed_test_data = self.norm(test_dataset)
        
        # build the model
        def build_model():
            model = keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=[train_dataset.shape[1]]),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(train_labels.shape[1])
            ])
            optimizer = tf.keras.optimizers.RMSprop(0.001)
            model.compile(loss='mse',
                          optimizer=optimizer,
                          metrics=['mae', 'mse'])
            return model
        
        self.model = build_model()
        # self.model.summary()

        # display training progress by printing a single dot for each completed epoch
        class PrintDot(keras.callbacks.Callback):
          def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0:
                dump('\n')
            dump('.')
        
        # the patience = 50 is the number of epochs to check for improvement
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
        
        # train the model
        history = self.model.fit(normed_train_data, train_labels, epochs=epochs,
                                 validation_split = 0.2, verbose=0, 
                                 callbacks=[PrintDot(), early_stop])  
        
        loss, mae, mse = self.model.evaluate(normed_test_data, test_labels, verbose=0)
        dump('\nANN: Testing set Mean Abs Error: {:5.2f}\n'.format(mae))
           
        # produce evaluationg graph
        if iplot:
            hist = pd.DataFrame(history.history)
            hist['epoch'] = history.epoch
            
            def plot_history(history):
                hist = pd.DataFrame(history.history)
                hist['epoch'] = history.epoch
                fig, axs = plt.subplots(1, 2, figsize=(8,3))
                ax = axs[0]
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Mean Abs Error')
                ax.plot(hist['epoch'], hist['mean_absolute_error'],
                       label='Training')
                ax.plot(hist['epoch'], hist['val_mean_absolute_error'],
                       label = 'Validation')
                ax.legend()
                
                ax = axs[1]
                bins = np.arange(0, 100 + 10, 5)
                mbins = bins[:-1] + np.diff(bins)
                ax.plot([],[],'k-', label='Observed')
                ax.plot([],[],'k:', label='Synthetic')
                for i, c in enumerate(self.coils):
                    freq1, _ = np.histogram(self.surveys[0].df[c].values, bins=bins)
                    freq2, _ = np.histogram(df[c].values, bins=bins)
                    cax = ax.step(mbins, freq1, linestyle='-', label=c)
                    ax.step(mbins, freq2, linestyle=':', color=cax[0].get_color())
                ax.legend()
                ax.set_xlabel('ECa')
                ax.set_ylabel('Frequency')
                    
            
                # plt.figure()
                # plt.xlabel('Epoch')
                # plt.ylabel('Mean Square Error')
                # plt.plot(hist['epoch'], hist['mean_squared_error'],
                #        label='Train Error')
                # plt.plot(hist['epoch'], hist['val_mean_squared_error'],
                #        label = 'Val Error')
                # plt.legend()
            
            plot_history(history)
            
    
    
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
        self.pstds = []
        
        if len(self.conds0) == 0:
            self.setInit(self.depths0)
            
        depths0 = self.depths0[0][0,:].copy()
        conds0 = self.conds0[0][0,:].copy()
        
        if dump is None:
            def dump(x):
                print(x, end='')
                
        J = buildJacobian(depths0, self.cspacing, self.cpos)
        L = buildSecondDiff(J.shape[1])
        def fmodel(p):
            return fCS(p, depths0, self.cspacing, self.cpos, hx=self.hx)
        
        # fCS is automatically adding a leading 0 but not buildJacobian
        def dataMisfit(p, app):
            return app - fmodel(p)
        def modelMisfit(p):
            return np.dot(L, p)
        
        for i, survey in enumerate(self.surveys):
            apps = survey.df[self.coils].values
            rmse = np.zeros(apps.shape[0])*np.nan
            model = np.zeros((apps.shape[0], len(conds0)))*np.nan
            dump('Survey {:d}/{:d}\n'.format(i+1, len(self.surveys)))
            for j in range(survey.df.shape[0]):
                app = apps[j,:]
                cond = np.ones((len(conds0),1))*np.nanmean(app) # initial EC is the mean of the apparent (doesn't matter)
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
                dump('\r{:d}/{:d} inverted'.format(j+1, apps.shape[0]))
            self.models.append(model)
            self.rmses.append(rmse)
            depth = np.repeat(depths0[None,:], apps.shape[0], axis=0)
            self.depths.append(depth)
            self.pstds.append(np.zeros(model.shape))
            dump('\n')
           


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
                - FSeq : Full Maxwell solution without LIN approximation (see Andrade 2016)
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
            self.depths0 = self.depths.copy()
            self.conds0 = self.models.copy()
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
            
        # define the forward model        
        if forwardModel in ['CS','FSlin','FSeq']:
            if forwardModel == 'CS':
                def fmodel(p, depth):
                    return fCS(p, depth, cspacing, cpos, hx=hxs)
            elif forwardModel == 'FSlin':
                def fmodel(p, depth):
                    return fMaxwellECa(p, depth, cspacing, cpos, f=freqs, hx=hxs)
            elif forwardModel == 'FSeq':
                def fmodel(p, depth):
                    return fMaxwellQ(p, depth, cspacing, cpos, f=freqs, hx=hxs)
        
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
                s.name = 'True model'
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
            lcol = ['layer{:d}'.format(a+1) for a in range(self.depths[0].shape[1])]
            dcol = ['depth{:d}'.format(a+1) for a in range(self.models[0].shape[1])]
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
        """Set the initial depths and conductivity for the inversion. Must
        be set after all filtering and just before the inversion.
        
        Parameters
        ----------
        depths0 : list or array
            Depth as positive number of the bottom of each layer.
            There is N-1 depths for N layers as the last layer is infinite.
        conds0 : list or array, optional
            Starting conductivity in mS/m of each layer.
            By default a homogeneous conductivity of 20 mS/m is defined.
        fixedDepths : list of type bool, optional
            Boolean array of same length as `depths0`. True if depth is fixed.
            False if it's a parameter. By default all depths are fixed.
        fixedConds : list of type bool, optional
            Boolean array of same length as `conds0`. True if conductivity if fixed.
            False if it's a parameter. By default all conductivity are variable.'
        """
        depths0 = np.array(depths0, dtype=float)
        if np.sum(depths0 < 0) > 0:
            raise ValueError('All depth should be specified as positive number.')
            return
        if np.sum(depths0 == 0) > 0:
            raise ValueError('No depth should be equals to 0 (infinitely thin layer)')
            return
        if len(self.surveys) == 0:
            raise Exception('First import surveys and then set initial conditions')
            return

        ddepths0 = []
        if len(depths0.shape) == 2:
            # it's an array so let's make sure it matches each survey
            ndepth = depths0.shape[1]
            for s in self.surveys:
                ddepths0.append(depths0)
                if s.df.shape[0] != depths0.shape[0]:
                    raise ValueError('The shape of depths0 does not match all samples from all surveys.')
                    return
        else: # it's a vector
            ndepth = len(depths0)
            for s in self.surveys:
                ddepths0.append(np.ones((s.df.shape[0], ndepth))*depths0[None,:])
        
        cconds0 = []
        if conds0 is None:
            for s in self.surveys:
                cconds0.append(np.ones((s.df.shape[0], ndepth+1), dtype=float)*20)
        else:
            conds0 = np.array(conds0)
            if len(conds0.shape) == 2: # matrix
                if conds0.shape[1] != ndepth + 1:
                    raise ValueError('Number of layers shoud be exactly equal to number of depths + 1.')
                    return
                for s in self.surveys:
                    cconds0.append(conds0.astype(float))
                    if s.df.shape[0] != conds0.shape[0]:
                        raise ValueError('The shape of depths0 does not match all samples from all surveys.')
                        return
            elif len(conds0.shape) == 1: # vector
                if len(conds0) != ndepth + 1:
                    raise ValueError('Number of layers shoud be exactly equal to number of depths + 1.')
                for s in self.surveys:
                    cconds0.append(np.ones((s.df.shape[0], ndepth+1), dtype=float)*conds0)
        
        # ffixedDepths = []
        # if len(fixedDepths.shape) == 2: # matrix
        #     if fixedDepths.shape[1] != ddepths0[0].shape[1]:
        #         raise ValueError('Shape of depths0 and fixedDepths should match.')
        #         return
        #     for s in self.surveys:
        #         ffixedDepths.append(fixedDepths)
        # elif len(fixedDepths.shape) == 1: # vector
        #     if len(fixedDepths) != ndepth:
        #         raise ValueError('Shape of depths0 and fixedDepths should match.')
        #     for s in self.surveys:
        #         ffixedDepths.append(np.ones((s.df.shape[0], ndepth), dtype=bool)*fixedDepths)
        # else:
        #     for s in self.surveys:
        #         ffixedDepths.append(np.ones((s.df.shape[0], ndepth), dtype=bool))
                
        # ffixedConds = []
        # if len(fixedConds.shape) == 2: # matrix
        #     if fixedConds.shape[1] != cconds0[0].shape[1]:
        #         raise ValueError('Shape of conds0 and fixedConds should match.')
        #         return
        #     for s in self.surveys:
        #         ffixedConds.append(conds0)
        # elif len(fixedConds.shape) == 1: # vector
        #     if len(fixedConds) != cconds0[0].shape[1]:
        #         raise ValueError('Shape of conds0 and fixedConds should match.')
        #     for s in self.surveys:
        #         ffixedConds.append(np.ones((s.df.shape[0], ndepth+1), dtype=bool)*fixedConds)
        # else:
        #     for s in self.surveys:
        #         ffixedConds.append(np.zeros((s.df.shape[0], ndepth+1), dtype=bool))

        if fixedDepths is None:
            fixedDepths = np.ones(ndepth, dtype=bool)
        if len(fixedDepths) != ndepth:
            raise ValueError('len(fixedDepths) should match len(depths0).')
            return
        if fixedConds is None:
            fixedConds = np.zeros(ndepth + 1, dtype=bool)
        if len(fixedConds) != ndepth + 1:
            raise ValueError('len(fixedConds) should match len(conds0).')
            return

        self.depths0 = ddepths0
        self.conds0 = cconds0
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
    
    

    def showProfile(self, index=0, ipos=0, ax=None, vmin=None, vmax=None,
                    maxDepth=None, errorbar=False):
        """Show specific inverted profile.
        
        Parameters
        ----------
        index : int, optional
            Index of the survey to plot.
        ipos : int, optional
            Index of the sample in the survey.
        ax : Matplotlib.Axes, optional
            If specified, the graph will be plotted against this axis.
        rmse : bool, optional
            If `True`, the RMSE for each transect will be plotted on a second axis.
            Note that misfit can also be shown with `showMisfit()`.
        errorbar : bool, optional
            If `True` and inversion is MCMC-based, standard deviation bar are
            drawn for the predicted depths and conductivities.
        """
        conds = self.models[index][ipos,:]
        depths = self.depths[index][ipos, :]
        x = np.r_[conds[0], conds]
        y = np.r_[[0], depths, np.max(depths) + 1]
        vd = ~self.fixedDepths
        vc = ~self.fixedConds
        if ax is None:
            fig, ax = plt.subplots()
        ax.step(x, -y, where='post')
        if errorbar:
            mid = conds[:-1] + np.diff(conds)/2
            mic = y[:-1] + np.diff(y)/2
            for i, ii in enumerate(np.where(vd)[0]):
                xd = mid[ii]
                yd = depths[ii]
                yerr = self.pstds[index][ipos,ii]
                ax.errorbar(xd, -yd, yerr=yerr, color='k', linestyle='none', capsize=2)
            for i, ii, in enumerate(np.where(vc)[0]):
                xc = conds[ii]
                yc = mic[ii]
                xerr = self.pstds[index][ipos,np.sum(vd) + i]
                ax.errorbar(xc, -yc, xerr=xerr, color='k', linestyle='none', capsize=2)
        ax.set_xlabel('EC [mS/m]')
        ax.set_ylabel('Depth [m]')
        ax.set_title('{:s}, sample {:d} (RMSE={:.2f})'.format(
            self.surveys[index].name, ipos, self.rmses[index][ipos]))
        

    
    def showResults(self, index=0, ax=None, vmin=None, vmax=None,
                    maxDepth=None, padding=1, cmap='viridis_r',
                    contour=False, rmse=False, errorbar=False, overlay=False):
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
        errorbar : bool, optional
            If `True` and inversion is MCMC-based, standard deviation bar are
            drawn for the predicted depths.
        overlay : bool, optional
            If `True`, a white transparent overlay is applied depending on the
            conductivity standard deviation from MCMC-based inversion
        """
        try:
            sig = self.models[index]
            depths = self.depths[index]
        except Exception:
            raise ValueError('No inverted model to plot')
            return

        # set up default arguments        
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure        
        if depths[0,0] != 0: # add depth 0
            depths = np.c_[np.zeros(depths.shape[0]), depths]
        if vmin is None:
            vmin = np.nanpercentile(sig, 5)
        if vmax is None:
            vmax = np.nanpercentile(sig, 95)
        cmap = plt.get_cmap(cmap)
        if maxDepth is None:
            maxDepth = np.nanpercentile(depths, 98) + padding
        depths = np.c_[depths, np.ones(depths.shape[0])*maxDepth]
        
        # vertices
        nlayer = sig.shape[1]
        nsample = sig.shape[0]
        x = np.arange(nsample+1) # number of samples + 1
        # x = np.sqrt(np.diff(self.surveys[index].df[['x', 'y']].values, axis=1)**2)
        xs = np.tile(np.repeat(x, 2)[1:-1][:,None], nlayer+1)
        ys = -np.repeat(depths, 2, axis=0)
        vertices = np.c_[xs.flatten('F'), ys.flatten('F')]
        
        # connection matrix
        n = vertices.shape[0]
        connection = np.c_[np.arange(n).reshape(-1,2),
                           2*nsample + np.arange(n).reshape(-1,2)[:,::-1]]
        ie = (connection >= len(vertices)).any(1)
        connection = connection[~ie, :]
        coordinates = vertices[connection]
        
        # plotting
        if contour is True:
            centroid = np.mean(coordinates, axis=1)
            xc = np.r_[centroid[:nsample,0], centroid[:,0], centroid[:nsample,0]]
            yc = np.r_[np.zeros(nsample), centroid[:,1], -np.ones(nsample)*maxDepth]
            zc = np.c_[sig[:,0], sig, sig[:,-1]]
            if vmax > vmin:
                levels = np.linspace(vmin, vmax, 7)
            else:
                levels = None
            cax = ax.tricontourf(xc, yc, zc.flatten('F'),
                                 cmap=cmap, levels=levels, extend='both')
            fig.colorbar(cax, ax=ax, label='EC [mS/m]')
        else:
            coll = PolyCollection(coordinates, array=sig.flatten('F'), cmap=cmap)
            coll.set_clim(vmin=vmin, vmax=vmax)
            ax.add_collection(coll)
            pad = 0.1 if rmse else 0.05
            fig.colorbar(coll, label='EC [mS/m]', ax=ax, pad=pad)
        
        if rmse:
            ax2 = ax.twinx()
            xx = np.arange(len(self.rmses[index])) + 0.5
            ax2.plot(xx, self.rmses[index], 'kx-')
            ax2.set_ylabel('RMSE')
            
        vc = ~self.fixedConds
        vd = ~self.fixedDepths
        if errorbar:
            for i, ii in enumerate(np.where(vd)[0]):
                xm = x[:-1] + np.diff(x)/2
                ym = depths[:,ii+1] # +1 because of zero depths
                yerr = self.pstds[index][:,i]
                ax.errorbar(xm, -ym, yerr=yerr, color='k', linestyle='none', capsize=2)
            
        if overlay: # uncertainty overlay
            zu = np.zeros(sig.shape)
            zu[:,vc] = self.pstds[index][:,np.sum(vd):]
            acmap = np.ones((10, 4), dtype=float)
            acmap[:, -1] = np.linspace(0, 0.7, 10) # alpha
            acmap = ListedColormap(acmap)
            if contour is True:
                centroid = np.mean(coordinates, axis=1)
                xc = np.r_[centroid[:nsample,0], centroid[:,0], centroid[:nsample,0]]
                yc = np.r_[np.zeros(nsample), centroid[:,1], -np.ones(nsample)*maxDepth]
                zc = np.c_[zu[:,0], zu, zu[:,-1]]
                cax = ax.tricontourf(xc, yc, zc.flatten('F'),
                                     cmap=acmap, extend='both')
            else:
                coll = PolyCollection(coordinates, array=zu.flatten('F'), cmap=acmap)
                ax.add_collection(coll)
                
        ax.set_xlabel('Samples')
        ax.set_ylabel('Depth [m]')
        if len(self.surveys) > 0:
            ax.set_title(self.surveys[index].name)
        ax.set_ylim([-maxDepth, 0])
        ax.set_xlim([x[0], x[-2]])
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
        for i in range(len(self.surveys)):
            if i != ref:
                self.models[i] = self.models[i] - self.models[ref]
        
    
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
        ax.set_ylabel('EC [mS/m]')
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
        if len(self.conds0) == 0: # not set
            depths0 = np.array(self.depths0)
            conds0 = np.ones(len(depths0) + 1)*20
        else:
            depths0 = self.depths0[isurvey][irow,:].copy()
            conds0 = self.conds0[isurvey][irow,:].copy()
        if alphas is None:
            alphas = np.logspace(-3,2,20)
        def fmodel(p):
            return fCS(p, depths0, self.cspacing, self.cpos)
        L = buildSecondDiff(len(conds0))
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
            res = minimize(objfunc, conds0, args=(app, alpha))
            phiData[i] = np.sum(dataMisfit(res.x, app)**2)
            phiModel[i] = np.sum(modelMisfit(res.x)**2)
            
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_title('L curve')
        ax.plot(phiModel, phiData, '.-')
        for a, ix, iy in zip(alphas, phiModel, phiData):
            ax.text(ix, iy, '{:.2f}'.format(a))
        ax.set_xlabel(r'Model Misfit ||L$\sigma$||$^2$')
        ax.set_ylabel(r'Data Misfit ||$\sigma_a - f(\sigma)$||$^2$')



    def calibrate(self, fnameECa, fnameEC, forwardModel='CS', ax=None, apply=False, dump=None):
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
            Forward model to use. Either CS (default), FS or FSeq.
        ax : matplotlib.Axes
            If specified the graph will be plotted against this axis.
        apply : bool, optional
            If `True` the ECa values will be calibrated. If `False`, the relationship
            will just be plotted.
        dump : function, optional
            Display different parts of the calibration.
        """
        if dump is None:
            def dump(x):
                print(x)
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
        elif forwardModel == 'FSlin':
            def fmodel(p):
                return fMaxwellECa(p, depths, survey.cspacing, survey.cpos, f=survey.freqs[0], hx=survey.hx[0])
        elif forwardModel == 'FSeq':
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
        
        # plot equation, apply it or not directly
        predECa = np.zeros(obsECa.shape)
        slopes = np.zeros(len(self.coils))
        offsets = np.zeros(len(self.coils))
        ax.set_prop_cycle(None)
        for i, coil in enumerate(survey.coils):
            x, y = obsECa[:,i], simECa[:,i]
            inan = ~np.isnan(x)& ~np.isnan(y)
            slope, intercept, r_value, p_value, std_err = linregress(x[inan], y[inan])
            slopes[i] = slope
            offsets[i] = intercept
            dump('{:s}: ECa(ERT) = {:.2f} * ECa(EMI) + {:.2f} (R^2={:.2f})'.format(coil, slope, intercept, r_value**2))
            predECa[:,i] = obsECa[:,i]*slope + intercept
            ax.plot(obsECa[:,i], predECa[:,i], '-', label='{:s} (R$^2$={:.2f})'.format(coil, r_value**2))
        ax.legend()

        
        # apply it to all ECa values
        if apply:
            if self.calibrated: # already calibrated
                dump('Data can only be calibrated once!'
                     ' Please reimport the survey before recalibrating your data.')
                return
            self.calibrated = True
            
            # replot the same graph but with corrected EC
            ax.clear()
            ax.plot([vmin, vmax], [vmin, vmax], 'k-', label='1:1')
            for i, s in enumerate(self.coils):
                # obsECaCorr = (obsECa[:,i] - offsets[i])/slopes[i]
                obsECaCorr = obsECa[:,i] + offsets[i] - (1-slopes[i]) * obsECa[:,i]
                x, y = obsECaCorr, simECa[:,i]
                cax = ax.plot(x, y, '.')
                inan = ~np.isnan(x) & ~np.isnan(y)
                slope, intercept, r_value, p_value, std_err = linregress(x[inan], y[inan])
                # dump('{:s} corrected: ECa(ERT) = {:.2f} * ECa(EMI) + {:.2f} (R^2={:.2f})'.format(
                    # coil, slope, intercept, r_value**2))
                predECaCorr = x * slope + intercept
                ax.plot(x, predECaCorr, '-', color=cax[0].get_color(),
                        label='{:s} (R$^2$={:.2f})'.format(coil, r_value**2))
            ax.legend()
            ax.set_xlim([vmin, vmax])
            ax.set_ylim([vmin, vmax])

            # apply correction on all datasets
            for s in self.surveys:
                for i, c in enumerate(self.coils):
                    # s.df.loc[:, c] = (s.df[c].values - offsets[i])/slopes[i]
                    s.df.loc[:, c] = s.df[c].values + offsets[i] - (1-slopes[i]) * s.df[c].values
            dump('Correction is applied.')
        
        
        
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
        fig.colorbar(cax, ax=ax, label='EC [mS/m]')
        # depths = np.r_[[0], self.depths0, [-np.inf]]
        # ax.set_title('{:.2f}m - {:.2f}m'.format(depths[islice], depths[islice+1]))
        ax.set_title('Layer {:d}'.format(islice+1))
        
        
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

