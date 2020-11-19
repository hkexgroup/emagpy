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
import tempfile
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap
import matplotlib.tri as mtri
# import matplotlib.patches as mpatches
# import matplotlib.path as mpath
from scipy.optimize import minimize
from scipy.stats import linregress, gaussian_kde
from scipy import interpolate

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
        # data attributes
        self.surveys = []
        self.freqs = []
        self.coils = []
        self.coilsInph = []
        self.coilsErr = []
        self.coilsQuad = []
        self.cpos = []
        self.cspacing = []
        self.hx = []
        self.projection = None # string of projection as EPSG
        
        # initiatial inversion parameters
        self.depths0 = np.array([0.5, 1.5]) # initial depths of the bottom of each layer (last one is -inf)
        self.conds0 = [] # initial conductivity for each layer
        self.fixedConds = []
        self.fixedDepths = []
        self.forwardModel = None # store the forward model choosen for showMisfit and showOne2One
        
        # processing outputs
        self.models = []
        self.depths = [] # contains inverted depths or just depths0 if fixed
        self.pstds = [] # parameter standard deviation for MCMC inversion
        self.rmses = []
        self.dois = [] # list of array of DOI computed from ('computeDOI()')

        # flags
        self.ikill = False # if True, the inversion is killed
        self.c = 0 # counter
        self.calibrated = False # flag for ERT calibration
        self.runningUI = False # True if run in UI, just change output of parallel stuff
        
        # others
        self.annReplaced = 0 # number of measurement outliers by ANN
        
        
    def createSurvey(self, fname, freq=None, hx=None, targetProjection=None):
        """Create a survey object.
        
        Parameters
        ----------
        fname : str
            Path to the csv file with the data.
        freq : float, optional
            Frequency for all the coils (can also be specified for each coil in the file).
        hx : float, optional
            Height of the instrument above the ground (can also be specified for each coil in the file).
        targetProjection : str, optional
            If specified, a conversion from NMEA string in 'Latitude' and 'Longitude'
            columns will be performed according to EPSG code: e.g. 'EPSG:27700'.
        """
        # create Survey object
        if self.projection is not None:
            targetProjection = self.projection
            
        survey = Survey(fname, freq=freq, hx=hx, targetProjection=targetProjection)
        
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
            self.coilsInph = survey.coilsInph
            self.coilsQuad = survey.coilsQuad
            self.coilsErr = survey.coilsErr
            self.surveys.append(survey)
        else: # check we have the same configuration than other survey
            check = [a == b for a,b, in zip(self.coils, survey.coils)]
            if all(check) is True:
                self.surveys.append(survey)
        
            
        
    def createTimeLapseSurvey(self, fnames, targetProjection=None):
        """Import multiple surveys.
        
        Parameters
        ----------
        fnames : list of str
            List of file to be parsed or directory where the files are.
        targetProjection : str, optional
            If specified, a conversion from NMEA string in 'Latitude' and 'Longitude'
            columns will be performed according to EPSG code: e.g. 'EPSG:27700'.
        """
        if isinstance(fnames, list): # it's a list of filename
            if len(fnames) < 2:
                raise ValueError('at least two files needed for timelapse inversion')
        else: # it's a directory and we import all the files inside
            if os.path.isdir(fnames):
                fnames = [os.path.join(fnames, f) for f in np.sort(os.listdir(fnames)) if f[0] != '.']
                # this filter out hidden file as well
            else:
                raise ValueError('dirname should be a directory path or a list of filenames')
        if self.projection is not None:
            targetProjection = self.projection
        for fname in fnames:
            self.createSurvey(fname, targetProjection=targetProjection)
            
    
    
    def createMergedSurvey(self, fnames, method='nearest', how='add', targetProjection=None):
        """Create a unique survey from different files by spatially interpolating the values.
        This can be useful when two surveys (Hi and Lo mode, vertical and horizontal) were
        taken on the same site successively. The method adds the 'x' and 'y' positions of all
        surveys and use interpolation to compute the missing values.
        
        Paramaters
        ----------
        fnames : list of str
            Paths of the .csv file with the data. Note each file need an 'x' and 'y' column for
            the spatial merge.
        method : str, optional
            Interpolation method to use. Either 'nearest', 'linear' or 'cubic'.
        how : str, optional
            How to merge the data from the different surveys:
                - 'add': add all positions from all surveys and interpolate
                - 'first': use positions from first survey only and interpolate on other surveys
        targetProjection : str, optional
            If specified, a conversion from NMEA string in 'Latitude' and 'Longitude'
            columns will be performed according to EPSG code: e.g. 'EPSG:27700'.
        """
        # import all surveys
        if self.projection is not None:
            targetProjection = self.projection
        surveys = []
        for fname in fnames:
            surveys.append(Survey(fname, targetProjection=targetProjection))
        
        # check all surveys have different coil configurations
        coils = np.hstack([survey.coils for survey in surveys])
        if len(np.unique(coils)) < len(coils):
            raise Exception('Each survey to be merged should have different coil configurations.')
            return
        
        # spatially merge the survey
        if how == 'first':
            xy = surveys[0].df[['x','y']].values
        else: # 'add'
            xy = np.vstack([survey.df[['x','y']].values for survey in surveys])
        df = pd.DataFrame(xy, columns=['x','y'])
        for survey in surveys:
            points = survey.df[['x','y']].values
            for coil in survey.coils:
                values = survey.df[coil].values
                df[coil] = griddata(points, values, xy, method=method)
            values = survey.df['elevation'].values
            if np.sum(values) != 0:
                df['elevation'] = griddata(points, values, xy, method=method)
            else:
                df['elevation'] = 0
                
        # append the newly merged survey
        mergedSurvey = Survey()
        mergedSurvey.readDF(df)
        self.surveys.append(mergedSurvey)
        self.coils = mergedSurvey.coils
        self.coilsInph = mergedSurvey.coilsInph
        self.coilsQuad = mergedSurvey.coilsQuad
        self.coilsErr = mergedSurvey.coilsErr
        self.freqs = mergedSurvey.freqs
        self.cspacing = mergedSurvey.cspacing
        self.cpos = mergedSurvey.cpos
        self.hx = mergedSurvey.hx
            
        
    
    def importGF(self, fnameLo=None, fnameHi=None, device='CMD Mini-Explorer',
                 hx=0, calib=None, targetProjection=None):
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
            Height of the device above the ground in meters. Note that this is
            different from the 'calib' used. Data can be collected at 1 m (hx=1)
            but using the 'F-0m' calibration.
        calib : str, optional
            Calibration used. Either 'F-0m' or 'F-1m'. If specified, the 
            `gfCorrection()` function will be called and ECa values will be
            converted to LIN ECa (this is recommended for inversion).
        targetProjection : str, optional
            If both Lo and Hi dataframe contains 'Latitude' with NMEA values
            a conversion first is done using `self.convertFromNMEA()` before
            being regrid using nearest neightbours.
        """
        if self.projection is not None:
            targetProjection = self.projection
        survey = Survey()
        survey.importGF(fnameLo, fnameHi, device, hx, calib, targetProjection)
        self.coils = survey.coils
        self.coilsInph = survey.coilsInph
        self.coilsQuad = survey.coilsQuad
        self.coilsErr = survey.coilsErr
        self.freqs = survey.freqs
        self.cspacing = survey.cspacing
        self.cpos = survey.cpos
        self.hx = survey.hx
        self.surveys.append(survey)
        
    
    def gfCorrection(self, calib):
        """Apply a correction to convert the calibrated ECa taking using F-0m or
        F-1m on CMD Explorer and Mini-Explorer to LIN ECa.
        
        GF instruments directly map the quadrature values measured to ECa using
        a linear calibration. This allows to have ECa values representative of
        the ground EC even when the device is operated at 1 m above the ground
        for instance. However, this calibration gets in the way when modelling
        the EM response based on physical equations for the inversion. Hence,
        we recommend to apply a correction and convert back the 'calibrated ECa' 
        to LIN ECa. This function contains the retro-engineered coefficients
        of the GF calibration. The ECa values are first uncalibrated back to 
        quadrature values and then converted back to ECa using the LIN approximation.
        
        Parameters
        ----------
        calib : str
            Name of the calibration used. Either 'F-0m' of 'F-1m'.
        """
        for s in self.surveys:
            s.gfCorrection(calib)
        
        
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
        x0 = cols2str(df0[['x','y']].values.astype(float))
        icommon = np.ones(len(x0), dtype=bool)
        for df in dfs2[1:]:
            x = cols2str(df[['x','y']].values.astype(float))
            ie = np.in1d(x0, x)
            icommon = icommon & ie
        print(np.sum(icommon), 'in common...', end='')

        # create boolean index to match those measurements
        indexes = []
        xcommon = x0[icommon]
        for df in dfs2:
            x = cols2str(df[['x','y']].values.astype(float))
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
        
        
    
    def importModel(self, fnames):
        """Import a save model from previous inversion.
        
        Parameters
        ----------
        fname : str or list of str
            Path(s) of the .csv file.
        """
        if isinstance(fnames, str):
            fnames = [fnames]
        conds = []
        depths = []
        for fname in fnames:
            df = pd.read_csv(fname)
            ccols = [c for c in df.columns if c[:5] == 'layer']
            dcols = [c for c in df.columns if c[:5] == 'depth']
            if len(ccols) != len(dcols) + 1:
                print('Number of depths should be number of layer - 1')
            conds.append(df[ccols].values)
            depths.append(df[dcols].values)
        self.setModels(depths, conds)
    
    
        
    def invert(self, forwardModel='CS', method='L-BFGS-B', regularization='l2',
               alpha=0.07, beta=0.0, gamma=0.0, dump=None, bnds=None,
               options={}, Lscaling=False, rep=100, noise=0.05, nsample=100, 
               annplot=False, threed=False, njobs=1):
        """Invert the apparent conductivity measurements.
        
        Parameters
        ----------
        forwardModel : str, optional
            Type of forward model:
                - CS : Cumulative sensitivity (default)
                - FSlin : Full Maxwell solution with low-induction number (LIN) approximation
                - FSeq : Full Maxwell solution without LIN approximation (see Andrade et al., 2016)
        method : str, optional
            Name of the optimization method either L-BFGS-B, TNC, CG or Nelder-Mead
            to be passed to `scipy.optimize.minmize()` or ROPE, SCEUA, DREAM, MCMC for
            a MCMC-based solver based on the `spotpy` Python package.
            Alternatively 'ANN' can be used (requires tensorflow), it will train
            an artificial neural network on synthetic data and use it for inversion.
            Note that smoothing (alpha, beta, gamma) and regularization are not
            supported by ANN, MCMC, DREAM and SCUEA (but well by ROPE).
            Another option is the use of the Gauss-Newton algorithm which can be 
            faster in some situation. Maximum number of GN iteration can be specified as
            options={maxiter:3}. Default is 1.
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
        threed : bool, optional
            If `True`, the beta parameters will serve to do a quasi3D inversion.
        njobs : int, optional
            If -1 all CPUs are used. If 1 is given, no parallel computing code
            is used at all, which is useful for debugging. For n_jobs below -1,
            (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs
            but one are used.
        """
        mMinimize = ['L-BFGS-B','TNC','CG','Nelder-Mead']
        mMCMC = ['ROPE','SCEUA','DREAM', 'MCMC']
        mOther = ['ANN','Gauss-Newton','GPS']
        if (method not in mMinimize) and (method not in mMCMC) and (method not in mOther):
            raise ValueError('Unknown method {:s}'.format(method))
            return
        
        # check if we have initial model
        if len(self.conds0) == 0:
            self.setInit([0.5, 1.5]) # default depths
            
        # switch in case Gauss-Newton routine is selected
        if (forwardModel == 'CSgn') | ((forwardModel == 'CS') & (method == 'Gauss-Newton')):
            self.invertGN(alpha=alpha, alpha_ref=None, dump=dump)
            self.forwardModel = 'CS'
            return
        
        if method == 'Gauss-Newton':
            if forwardModel == 'Q':
                raise ValueError('Forward model Q can not be used with Gauss-Newton at the moment.'
                                 'Choose CS, FSlin or FSeq instead.')
                return
            if regularization == 'l1':
                print('Regularization autommatically set to L2 for Gauss-Newton')
            if beta != 0:
                print('No lateral smoothing possible with Gauss-Newton for now.')
        
        self.forwardModel = forwardModel # for future RMSE or misfit computation
        self.models = []
        self.depths = []
        self.rmses = []
        self.pstds = []
        self.annReplaced = 0
        self.ikill = False
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
            
        # check time-lapse constrain
        if gamma != 0:
            n = self.surveys[0].df.shape[0]
            for s in self.surveys[1:]:
                if s.df.shape[0] != n:
                    raise ValueError('For time-lapse constrain (gamma > 0), all surveys need to have the same length.')
                    gamma = 0
        
        # check parallel
        if njobs != 1 and beta != 0:
            dump('WARNING: No parallel is possible with lateral smoothing (beta > 0).\n')
            njobs = 1
            
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
            elif forwardModel == 'QP':
                return getQs(cond, depth, self.cspacing, self.cpos, f=self.freqs, hx=self.hx)

        # define bounds
        if bnds is not None:
            if len(bnds) == 2 and (isinstance(bnds[0], int) or isinstance(bnds[0], float)):
                # we just have min/max of EC
                top = np.ones(nc)*bnds[1]
                bot = np.ones(nc)*bnds[0]
                bounds = list(tuple(zip(bot[vc], top[vc])))
            else:
                bounds = bnds
            # check len(bounds) match len(param)
            nparam = np.sum(vd) + np.sum(vc)
            if len(bnds) != nparam:
                raise ValueError('len(bounds) ({:d}) should be the same as '
                                 'the number of parameters ({:d})'.format(len(bounds), nparam))
            
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


        # gridded parameter search as an inversion method
        if method == 'GPS':
            if alpha !=0 or gamma != 0 or beta != 0:
                dump('WARNING: Gridded Parameter Search does not accept any smoothing parameters.\n')
            
            dump('Inverting data using Gridded Parameter Search\n')
            if len(self.conds0) > 0:
                fixedParamsD = []
                for i in range(0, len(self.depths0[0][0])):
                    if self.fixedDepths[i] == True:
                        fixedParamsD.append(self.depths0[0][0][i])
                    else:
                        fixedParamsD.append(None)        
                fixedParamsC = []
                for i in range(0, len(self.conds0[0][0])):
                    if self.fixedConds[i] == True:
                        fixedParamsC.append(self.conds0[0][0][i])
                    else:
                        fixedParamsC.append(None)
                fixedParams = fixedParamsD + fixedParamsC
                nlayers = len(fixedParamsC)            
            else:
                fixedParams = None
                nlayers = 2
            bestConds, bestDepths, bestMisfits, paramSd, paramMin, paramMax = self.gridParamSearch(forwardModel=fmodel, bnds=bnds, regularization=regularization, fixedParams=fixedParams)
            self.models.append(bestConds)
            self.depths.append(bestDepths)
            self.rmses.append(bestMisfits)
            self.pstds.append(paramSd)

        # build ANN network
        if method == 'ANN': # pragma: no cover
            if gamma != 0 or beta != 0:
                dump('WARNING: ANN does not accept any smoothing parameters.\n')
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
                #misfit = misfit*1e6 # to help the solver with small Q
                misfit = np.abs(misfit/obs)
            if forwardModel == 'QP':
                misfit[len(misfit)//2:] *= 1e5 # TODO to be tested out
            return misfit 
        
        # model misfit only for conductivities not depths
        def modelMisfit(p):
            cond = p[:np.sum(vc)] # smoothing only for parameters elements (depth or cond)
            return cond[:-1] - cond[1:]
        
        # set up regularisation
        # p : parameter, app : ECa,
        # pn : consecutive previous profile (for lateral smoothing)
        # spn : profile from other survey (for time-lapse)
        if regularization  == 'l1':
            def objfunc(p, app, pn, spn, alpha, beta, gamma, ini0):
                return np.sqrt(np.sum(np.abs(dataMisfit(p, app, ini0)))/len(app)
                               + alpha*np.sum(np.abs(modelMisfit(p)))/np.sum(vc)
                               + beta*np.sum(np.abs(p - pn))/len(p)
                               + gamma*np.sum(np.abs(p - spn))/len(p))
        elif regularization == 'l2':
            def objfunc(p, app, pn, spn, alpha, beta, gamma, ini0):
                return np.sqrt(np.sum(dataMisfit(p, app, ini0)**2)/len(app)
                               + alpha*np.sum(modelMisfit(p)**2)/np.sum(vc)
                               + beta*np.sum((p - pn)**2)/len(p)
                               + gamma*np.sum((p - spn)**2)/len(p))
            
        # define spotpy class if McMC-based methods
        if method in mMCMC:
            try:
                import spotpy
            except ImportError:
                print('Please install spotpy to use MCMC-based methods.')
                return
            
            class spotpy_setup(object):
                def __init__(self, obsVals, bounds, pn, spn, alpha, beta, 
                             gamma, ini0, fmodel):
                    self.params = []
                    for i, bnd in enumerate(bounds):
                        self.params.append(
                                spotpy.parameter.Uniform(
                                        'x{:d}'.format(i), low=bnd[0], high=bnd[1], optguess=np.mean(bnd)))
                    self.obsVals = obsVals
                    self.pn = pn
                    self.spn = spn
                    self.alpha = alpha
                    self.beta = beta
                    self.gamma = gamma
                    self.ini0 = ini0
                    self.fmodel = fmodel
                    
                def parameters(self):
                    return spotpy.parameter.generate(self.params)
            
                def simulation(self, vector):
                    x = np.array(vector) # this are actually the parameters
                    return x 
                
                def evaluation(self): # what the function return when called with the optimal values
                    observations = self.obsVals.flatten()
                    return observations
                
                def objectivefunction(self, simulation, evaluation, params=None):
                    # simulation is actually parameters, the simulation (forward model)
                    # is done inside the objective function itself
                    # val = -spotpy.objectivefunctions.rmse(evaluation, fmodel(simulation, self.ini0))
                    val = -objfunc(simulation, evaluation, self.pn, self.spn,
                                    self.alpha, self.beta, self.gamma, self.ini0)
                    # NOTE the objfunc is negative as spotpy algorithm will search
                    # at maximizing the misfit
                    return val
        
        
        # define optimization function
        def solve(obs, pn, spn, alpha, beta, gamma, ini0):
            if self.ikill is True:
                raise ValueError('killed') # https://github.com/joblib/joblib/issues/356
            if method in mMinimize: # minimize
                x0 = np.r_[ini0[0][vd], ini0[1][vc]]
                res = minimize(objfunc, x0, args=(obs, pn, spn, alpha, beta, gamma, ini0),
                               method=method, bounds=bounds, options=options)
                out = res.x     
            elif method in mMCMC: # MCMC based methods
                spotpySetup = spotpy_setup(obs, bounds, pn, spn, alpha, beta, 
                                           gamma, ini0, fmodel)
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
                cols = ['parx{:d}'.format(a) for a in range(len(bounds))]
                ibest = np.argmin(np.abs(results['like1'])) # lowest misfit sampled
                outval = np.array([results[col][ibest] for col in cols])
                # actually the mode is not a such a good estimate
                # samples = [results[col] for col in cols]
                # outval = np.array([a[np.argmax(gaussian_kde(a)(a))] for a in samples])
                ie = np.abs(results['like1']) < np.nanpercentile(np.abs(results['like1']), 10)
                outstd = np.array([np.nanstd(results[col][ie]) for col in cols])
                out = (outval, outstd)
                
                # below is for paper figure on MCMC inversion
                # vals = np.array([results[col] for col in cols]).T
                # ibest = np.argmin(np.abs(results['like1']))
                # bmisfit = np.abs(results[ibest]['like1'])
                # print('lowest misfit is: {:.2f} with param'.format(
                #     bmisfit), results[ibest][cols])
                # samples = [results[col] for col in cols]
                # pmode = np.array([a[np.argmax(gaussian_kde(a)(a))] for a in samples])
                # pmisfit = spotpy.objectivefunctions.rmse(obs, fmodel(pmode, ini0))
                # print('mode misfit is: {:.2f} with param'.format(pmisfit), pmode)
                # stds[i,:] = np.std(vals, axis=0)
                # tmisfit = spotpy.objectivefunctions.rmse(obs, fmodel(np.array([0.5, 20, 40]), ini0))
                
                # import seaborn as sns
                # fig, axs = plt.subplots(1, 3, figsize=(8,2.5))
                # ax = axs[0]
                # ax.set_title('(a) depth m={:.1f} std={:.1f}'.format(
                # np.mean(vals[:,0]), np.std(vals[:,0])))
                # # ax.hist(vals[:,0], bins=20)
                # sns.kdeplot(vals[:,0], ax=ax)
                # # ax.axvline(pmode[0], color='m', linestyle='-')
                # ax.axvline(0.5, color='r', linestyle='--')
                # ax.axvline(vals[ibest,0], color='lime', linestyle='--')
                # ax.set_ylabel('KDE')
                # ax = axs[1]
                # ax.set_title('(b) layer1 m={:.1f} std={:.1f}'.format(
                # np.mean(vals[:,1]), np.std(vals[:,1])))
                # # ax.hist(vals[:,1], bins=20)
                # sns.kdeplot(vals[:,1], ax=ax)
                # # ax.axvline(pmode[1], color='m', linestyle='-')
                # ax.axvline(20, color='r', linestyle='--')
                # ax.axvline(vals[ibest,1], color='lime', linestyle='--')
                # ax.set_ylabel('KDE')
                # ax = axs[2]
                # ax.set_title('(c) layer2 m={:.1f} std={:.1f}'.format(
                # np.mean(vals[:,2]), np.std(vals[:,2])))
                # # ax.hist(vals[:,2], bins=20)
                # sns.kdeplot(vals[:,2], ax=ax)
                # # ax.axvline(pmode[2], color='m', linestyle='-', label='{:.2f}'.format(pmisfit))
                # ax.axvline(40, color='r', linestyle='--', label='True (RMSE={:.2f})'.format(tmisfit))
                # ax.axvline(vals[ibest,2], color='lime', linestyle='--', label='Best est. (RMSE={:.2f})'.format(bmisfit))
                # ax.legend(fontsize=8)
                # ax.set_ylabel('KDE')
                # fig.tight_layout()
                # fig.savefig('figures/mcmc.jpg', dpi=500)
                # fig.show()
                                
            return out
        
        # inversion row by row
        for i, survey in enumerate(self.surveys):
            if self.ikill:
                break
            self.c = 0
            apps = survey.df[self.coils].values # ECa in mS/m
            inph = survey.df[self.coilsInph].values # inphase in ppt
            rmse = np.zeros(apps.shape[0])*np.nan
            model = self.conds0[i].copy()
            depth = self.depths0[i].copy()
            stds = np.zeros((apps.shape[0], np.sum(vd) + np.sum(vc)))
            xy = self.surveys[i].df[['x','y']].values
            iinverted = np.zeros(apps.shape[0], dtype=bool)
            dist0 = np.sqrt(np.sum((xy - xy[0,:])**2, axis=1))
            iorder = np.argsort(dist0)
            if len(self.surveys) > 1: # if only one survey, this isn't needed
                dump('Survey {:d}/{:d}\n'.format(i+1, len(self.surveys)))
            params = []
            outs = []
            nrows = survey.df.shape[0]
            for j in range(nrows):
                # define observations and convert to Q if needed
                obs = apps[j,:]
                if forwardModel == 'Q':
                    obs = np.array([eca2Q(a*1e-3, s) for a, s in zip(obs, self.cspacing)])
                if forwardModel == 'QP':
                    obs = np.array([eca2Q(a*1e-3, s) for a, s in zip(obs, self.cspacing)])
                    
                # define previous profile in case we want to lateral constrain
                if j == 0:
                    pn = np.zeros(np.sum(np.r_[vd, vc]))
                    b = 0
                    iinverted[j] = True
                else:
                    if threed: # mean of 3 closest inverted survey
                        ipt = iorder[j]
                        dist = np.sqrt(np.sum((xy-xy[ipt,:])**2, axis=1))
                        icloseInverted = np.argsort(dist[iinverted])[:3]
                        val = np.c_[depth[icloseInverted,:][:,vd],
                                    model[icloseInverted,:][:,vc]]
                        pn = np.mean(val, axis=0)
                        iinverted[ipt] = True
                    else: # previous survey
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
                
                params.append((obs, pn, spn, alpha, b, g, ini0))                
            
                # sequential inversion (default)
                if (method not in mOther) & (njobs == 1):
                    try:
                        with HiddenPrints():
                            outt = solve(*params[j])
                        dump('\r{:d}/{:d} inverted'.format(j+1, nrows))
                        obs = params[j][0]
                        ini0 = params[j][-1]
                        if method in mMCMC:
                            std = outt[1]
                            stds[j,:] = std
                            out = outt[0]
                        else:
                            out = outt
                        depth[j,vd] = out[:np.sum(vd)]
                        model[j,vc] = out[np.sum(vd):]
                        rmse[j] = np.sqrt(np.sum(dataMisfit(out, obs, ini0)**2)/np.sum(obs**2)/len(obs))*100
                    except Exception as e:
                        print('Killed')
                        return
                    
                if method == 'Gauss-Newton':
                    try:
                        dump('\r{:d}/{:d} inverted'.format(j+1, nrows))
                        # compute Jacobian
                        sens = self.computeSens(forwardModel=forwardModel,
                                                coils=None, # this trigger normal forward modelling
                                                models=[ini0[1][None,:]], depths=[ini0[0][None,:]])
                        sens = sens[0][:,:,0]
                        J = sens/np.sum(sens, axis=0) # so that sum == 1
                        J = J.T
                        
                        # Gauss-Newton algorithm
                        cond = np.copy(ini0[1])[:,None]
                        app = obs.copy()
                        # rrmse = np.sqrt(1/len(app)*np.sum(dataMisfit(cond[:,0], app, ini0)**2)/np.sum(app**2))
                        # print('ini: RMSE: {:.5f}%'.format(rrmse), ' '.join(
                                # ['{:.2f}'.format(a) for a in cond[:,0]]))
                        maxiter = options['maxiter'] if 'maxiter' in options else 1
                        for l in range(maxiter): # only one iteration as the jacobian doesn't depend on the cond
                            d = -dataMisfit(cond.flatten(), app, ini0) # NOTE we need minus to get the right direction
                            LHS = np.dot(J.T, J) + alpha*L
                            RHS = np.dot(J.T, d[:,None]) - alpha*np.dot(L, cond)
                            solution = np.linalg.solve(LHS, RHS)
                            cond = cond + solution
                            out = cond.flatten()
                            # rrmse = np.sqrt(1/len(app)*np.sum(dataMisfit(out, app, ini0)**2)/np.sum(app**2))
                            # print('{:d}: RMSE: {:.5f}%'.format(l, rrmse), ' '.join(
                                # ['{:.2f}'.format(a) for a in cond[:,0]]))
                        depth[j,vd] = out[:np.sum(vd)]
                        model[j,vc] = out[np.sum(vd):]
                        rmse[j] = np.sqrt(np.sum(dataMisfit(out, obs, ini0)**2)/np.sum(obs**2)/len(obs))*100
                    except Exception as e:
                        print('Killed')
                        return
            
            # parallel computing with loky backend
            if (method != 'ANN') & (njobs != 1):
                try:
                    with HiddenPrints():
                        outs = Parallel(n_jobs=njobs, verbose=0, backend='loky')(delayed(solve)(*a) for a in tqdm(params))
                except Exception as e: # might be when we kill it using UI
                    print('Error in // inversion:', e)
                    return
                
           # artifical neural network inversion 
            elif method == 'ANN': # pragma: no cover
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
            if (method == 'ANN') or (njobs != 1):
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
                    rmse[j] = np.sqrt(np.sum(dataMisfit(out, obs, params[j][-1])**2)/np.sum(obs**2)/len(obs))*100
            dump('\n')
            self.models.append(model)
            self.depths.append(depth)
            self.rmses.append(rmse)
            self.pstds.append(stds)
            # dump('{:d} measurements inverted\n'.format(apps.shape[0]))
                    
            
    
    def buildANN(self, fmodel, bounds, noise=0.05, iplot=False, nsample=100,
                 dump=None, epochs=500): # pragma: no cover
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
            raise ImportError('Tensorflow is needed for ANN inversion.')
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
                for l in range(1): # only one iteration as the jacobian doesn't depend on the cond
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
                rmse[j] = np.sqrt(np.sum(dataMisfit(out, app)**2)/np.sum(app**2)/len(app))*100
                dump('\r{:d}/{:d} inverted'.format(j+1, apps.shape[0]))
            self.models.append(model)
            self.rmses.append(rmse)
            depth = np.repeat(depths0[None,:], apps.shape[0], axis=0)
            self.depths.append(depth)
            self.pstds.append(np.zeros(model.shape))
            dump('\n')
           


    def tcorrECa(self, tdepths, tprofile):
        """Temperature correction based on temperature profile.
        An 'apparent' temperature is computed for each coil configuration
        using the CS function and the observed ECa is corrected according
        to a 2% increase in EC per degC.
        
        Parameters
        ----------
        tdepths : list of arrays
            Depths in meters of the temperature sensors (negative downards).
        tprofile : list of arrays
            Temperature values corresponding in degree Celsius.
        """
        for i, s in enumerate(self.surveys):
            s.tcorr(tdepths[i], tprofile[i])


            
    def tcorrEC(self, tdepths, tprofile, a=0.02):
        """Temperature correction for inverted models using a 2% increase
        of EC per degC.
        
        EC_t = EC * (1 - a * (t - 25))
        
        where a == 0.02 (2% by default)
        
        Parameters
        ----------
        tdepths : array-like
            Depths in meters of the temperature sensors (negative downards).
        tprofile : array-like
            Temperature values corresponding in degree Celsius.
        a : float, optional
            Correction coefficient. By default a 2% (a=0.02) of EC per degC
            is assumed.
        """
        for i, model in enumerate(self.models):
            # interpolate depth
            
            # apply correction
            pass


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
        """Keep consecutive measurements when the difference between them
        is smaller than `thresh`.
        
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
            

    def filterBearing(self, phiMin, phiMax):
        """Keep measurements in a certain bearing range between phiMin and phiMax. 
        
        Parameters
        ----------
        phiMin : float, optional
            Minimum angle, in degrees. 
        phiMax : float, optional
            Maximum angle, in degrees.
        """
        for s in self.surveys:
            s.fitlerBearing(phiMin=phiMin, phiMax=phiMax)
        
        
    def computeStat(self, timef=None):
        """Compute geometrical statistics of consective points: azimuth and 
        bearing of walking direction, time between consective measurements and
        distance between consecutive measurements. Results added to the main
        dataframe.
        
        Parameters
        ----------
        timef : str, optional
            Time format of the 'time' column of the dataframe (if available).
            To be passed to `pd.to_datetime()`. If `None`, it will be inferred.
            e.g. '%Y-%m-%d %H:%M:%S'
            
        Notes
        -----
        Requires projected spatial data (using convertFromNMEA() if needed).
        """
        for s in self.surveys:
            s.computeStat(timef=timef)
            
            
    def filterRepeated(self, tolerance=0.2):
        """Remove consecutive points when the distance between them is
        below `tolerance`.
        
        Parameters
        ----------
        tolerance : float, optional
            Minimum distance away previous point in order to be retained.
        """
        for s in self.surveys:
            s.filterRepeated(tolerance=tolerance)
            
        
        
    def forward(self, forwardModel='CS', coils=None, noise=0.0,
                models=[], depths=[]):
        """Compute the forward response.
        
        Parameters
        ----------
        forwardModel : str, optional
            Type of forward model:
                - CS : Cumulative sensitivity (default)
                - FS : Full Maxwell solution with low-induction number (LIN) approximation
                - FSeq : Full Maxwell solution without LIN approximation (see Andrade 2016)
        coils : list of str, optional
            If `None`, then the default attribute of the object will be used (foward
            mode on inverted solution).
            If specified, the coil spacing, orientation and height above the ground
            will be set. In this case you need to assign at models and depths (full forward mode).
            The ECa values generated will be incorporated as a new Survey object.
        noise : float, optional
            Percentage of noise to add on the generated apparent conductivities.
        models : list of numpy.array of float
            List of array of shape Nsample x Nlayer with conductiivty in mS/m. If empty,
            `self.models` will be used.
        depths : list of numpy.array of float
            List of array of shape Nsample x (Nlayer - 1) with the depth (positive number)
            of the bottom of the layer in meters relative to the surface.
            If empty `self.depths` will be used.
            
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
        if forwardModel == 'Q':
            print('For the Q forward model, the ECa values will be computed using FSeq')
            forwardModel = 'FSeq'
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
        else:
            raise ValueError('Forward model {:s} is not available.'
                             'Choose between CS, FSlin or FSeq'.format(forwardModel))
        
        def addnoise(x, level=0.05):
            return x + np.random.randn(len(x))*x*level
        
        if len(models) == 0:
            models = self.models
        if len(depths) == 0:
            depths = self.depths
        dfs = []
        for model, depth in zip(models, depths):
            apps = np.zeros((model.shape[0], len(self.coils)))*np.nan
            for j in range(model.shape[0]):
                conds = model[j,:]
                sdepth = depth[j,:]
                apps[j,:] = addnoise(fmodel(conds, sdepth), level=noise)
        
            df = pd.DataFrame(apps, columns=self.coils)
            dfs.append(df)
        
        if iForward:
            self.surveys = []
            for i, df in enumerate(dfs):
                s = Survey()
                s.readDF(df)
                s.name = 'Model {:d}'.format(i+1)
                self.surveys.append(s)
        
        return dfs
    
    
    def computeSens(self, forwardModel='CS', coils=None, models=[], depths=[]):
        """Compute normalised local sensitivity using perturbation method.
        
        Parameters
        ----------
        forwardModel : str, optional
            Type of forward model:
                - CS : Cumulative sensitivity (default)
                - FS : Full Maxwell solution with low-induction number (LIN) approximation
                - FSeq : Full Maxwell solution without LIN approximation (see Andrade 2016)
        coils : list of str, optional
            If `None`, then the default attribute of the object will be used (foward
            mode on inverted solution).
            If specified, the coil spacing, orientation and height above the ground
            will be set. In this case you need to assign at models and depths (full forward mode).
            The ECa values generated will be incorporated as a new Survey object.
        models : list of numpy.array of float
            List of array of shape Nsample x Nlayer with conductiivty in mS/m. If empty,
            `self.models` will be used.
        depths : list of numpy.array of float
            List of array of shape Nsample x (Nlayer - 1) with the depth (positive number)
            of the bottom of the layer in meters relative to the surface.
            If empty `self.depths` will be used.
        
        Returns
        -------
        senss : list of numpy.array of float
            List of matrix of size Nsample x Ncoils x Nlayers containing the normalised
            local sensitivity.
        """
        if len(models) == 0:
            models = self.models
        if len(depths) == 0:
            depths = self.depths
            
        senss = []
        for model, depth in zip(models, depths): # for each model
            npos = model.shape[0]  
            nlayer = depth.shape[1] + 1 # number of layer (last layer is infinite)
            nprofile = nlayer + 1 # number of 1D profile (last profile is reference)
            smodels = np.dstack([model]*nprofile) # Nsample x Nlayer x Nprofile
            ix = np.arange(nlayer)
            smodels[:,ix,ix] = smodels[:,ix,ix] + 1 # perturbation
            sdepths = np.dstack([depth]*nprofile)
            lmodels = [smodels[i,:,:].T for i in range(npos)]
            ldepths = [sdepths[i,:,:].T for i in range(npos)]
            dfs = self.forward(forwardModel=forwardModel, coils=coils,
                               models=lmodels, depths=ldepths,
                               noise=0.0)
            eca = np.dstack([df.values for df in dfs]) # Nsample x Ncoils x Nprofiles
            # sens = eca[:-1,:,:] / eca[-1,:,:][None,:,:] - 1 # dividing by ref (undisturbed ECa)
            sens = eca[:-1,:,:] - eca[-1,:,:][None,:,:] # subtracting the ref ECa (slighly better up to 1e-16)
            sens = sens/np.max(sens, axis=0)[None,:,:] # normalising
            senss.append(sens)
        return senss
    

    
    def show(self, index=0, coil='all', ax=None, vmin=None, vmax=None, 
             dist=True):
        """Show the raw data of the survey.
        
        Parameters
        ----------
        index : int, optional
            Survey number, by default, the first survey is chosen.
        coil : str, optional
            Specify which coil to plot. Default is all coils available.
        ax : matplotlib.Axes, optional
            If supplied, the graph will be plotted against `ax`.
        vmin : float, optional
            Minimal Y value.
        vmax : float, optional
            Maximial Y value.
        dist : bool, optional
            If `True` the true distance between points will be computed else
            the sample index is used as X index.
        """
        self.surveys[index].show(coil=coil, vmin=vmin, vmax=vmax, ax=ax, dist=dist)
    
    
    
    def showMap(self, index=0, coil=None, contour=False, ax=None, vmin=None, vmax=None,
                pts=False, cmap='viridis_r', xlab='x', ylab='y', nlevel=7):
        """ Display a map of the measurements.
        
        Parameters
        ----------
        index : int, optional
            Survey number, by default, the first survey is chosen.
        coil : str, optional
            Name of the coil to plot. By default, the first coil is plotted.
        contour : bool, optional
            If `True` filled contour will be plotted using `tricontourf`.
        ax : Matplotlib.Axes, optional
            If specified, the graph will be plotted against the axis.
        vmin : float, optional
            Minimum of the colorscale.
        vmax : float, optional
            Maximum of the colorscale.
        pts : bool, optional
            If `True` the measurements location will be plotted on the graph.
        xlab : str, optional
            X label.
        ylab : str, optional
            Y label.
        nlevel : int, optional
            Number of levels for the contourmap. Default is 7.
        """
        self.surveys[index].showMap(coil=coil, vmin=vmin, vmax=vmax,
                    contour=contour, pts=pts, cmap=cmap, ax=ax, xlab=xlab,
                    ylab=ylab, nlevel=nlevel)
        
    
    def showPseudo(self, index=0, coil='all', ax=None, vmin=None, vmax=None, 
             dist=True, cmap='viridis_r'):
        """Show the raw data of the survey.
        
        Parameters
        ----------
        index : int, optional
            Survey number, by default, the first survey is chosen.
        coil : str, optional
            Specify which coil to plot. Default is all coils available.
        ax : matplotlib.Axes, optional
            If supplied, the graph will be plotted against `ax`.
        vmin : float, optional
            Minimal Y value.
        vmax : float, optional
            Maximial Y value.
        dist : bool, optional
            If `True` the true distance between points will be computed else
            the sample index is used as X index.
        cmap : str, optional
            Name of the colormap.
        """
        # compute local sensitivity for a 1D profile given coil configurations of the survey
        ksens = Problem()
        cond = np.ones((1, 100))
        depth = np.linspace(0.05, 3*np.max(self.cspacing), cond.shape[1]-1)
        out = ksens.computeSens(forwardModel='CS', coils=self.coils, models=[cond], depths=[depth[None,:]])
        sens = out[0].squeeze(-1) # depth x coils
        mdepths = np.r_[depth[0]/2, depth[:-1] + np.diff(depth)/2, depth[-1] + (depth[-1]-depth[-2])/2]
        #print(mdepths)
        
        # look at 70% cumulate signal
        cs = np.cumsum(sens, axis=0)
        cs = cs/np.max(cs, axis=0)
        idoe = np.argmin(np.abs(cs - 0.7), axis=0)
        #print([cs[j,i] for i,j in enumerate(idoe)])
        doe = mdepths[idoe]
        #print(doe)
        
        # figure
        eca = self.surveys[index].df[self.coils].values.flatten()
        x = self.surveys[index].df['x'].values
        y = self.surveys[index].df['y'].values
        if dist:
            dist = np.r_[0, np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))]
            xlab = 'Distance [m]'
        else:
            dist = np.arange(len(x))
            xlab = 'Samples'
        dd = np.repeat(dist, len(self.coils))
        does = np.tile(doe, self.surveys[index].df.shape[0])
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12,3))
        else:
            fig = ax.figure
            
        if vmin is None:
            vmin = np.nanmin(eca)
        if vmax is None:
            vmax = np.nanmax(eca)
        levels = np.linspace(vmin, vmax, 7)
        # cax = ax.scatter(dd, -does, s=15, c=eca)
        cax = ax.tricontourf(dd, does, eca, cmap=cmap, levels=levels, extend='both')
        ax.plot(dd, does, 'k+')
        ax.invert_yaxis()
        ax.set_ylabel('Pseudo depth [m]')
        ax.set_xlabel(xlab)
        fig.colorbar(cax, ax=ax, label='ECa [mS/m]')

    
    
    def saveMap(self, fname, index=0, coil=None, nx=100, ny=100, method='linear',
                xmin=None, xmax=None, ymin=None, ymax=None, color=True,
                cmap='viridis_r', vmin=None, vmax=None, nlevel=14):
        """Save a georeferenced raster TIFF file.
        
        Parameters
        ----------
        fname : str
            Path of where to save the .tiff file.
        index : int, optional
            Survey number, by default, the first survey is chosen.
        coil : str, optional
            Name of the coil to plot. By default, the first coil is plotted.
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
        nlevel : int, optional
            Number of level in the colormap. Default 7.
        """
        self.surveys[index].saveMap(fname, coil=coil, nx=nx, ny=ny, method=method,
                xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, color=color,
                cmap=cmap, vmin=vmin, vmax=vmax, nlevel=nlevel)
    
    
    
    def saveSlice(self, fname, index=0, islice=0, nx=100, ny=100, method='linear',
                xmin=None, xmax=None, ymin=None, ymax=None, color=True,
                cmap='viridis', vmin=None, vmax=None, nlevel=14):
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
            Z = plt.get_cmap(cmap, nlevel)(norm(Z))
            Z = 255*Z
            Z = Z.astype('uint8')
            for i in range(4):
                Z[np.fliplr(ie.T).T, i] = 0
        
            with rasterio.open(fname, 'w',
                           driver='GTiff',
                           height=Z.shape[0],
                           width=Z.shape[1], count=4, dtype=Z.dtype,
                           crs=self.projection, transform=tt) as dst:
                for i in range(4):
                    dst.write(Z[:,:,i], i+1)
        else:
            with rasterio.open(fname, 'w',
                               driver='GTiff',
                               height=Z.shape[0],
                               width=Z.shape[1], count=1, dtype=Z.dtype,
                               crs=self.projection, transform=tt) as dst:
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
            lcol = ['layer{:d}'.format(a+1) for a in range(self.models[0].shape[1])]
            dcol = ['depth{:d}'.format(a+1) for a in range(self.depths[0].shape[1])]
            data = np.c_[survey.df[['x','y']].values, self.models[i], self.depths[i]]
            df = pd.DataFrame(data, columns=[['x','y'] + lcol + dcol])
            df.to_csv(fname, index=False)
    
    
    
    def gridData(self, nx=100, ny=100, method='nearest', 
                 xmin=None, xmax=None, ymin=None, ymax=None):
        """ Grid data on the same grid for all surveys.
        
        Parameters
        ----------
        nx : int, optional
            Number of points in x direction.
        ny : int, optional
            Number of points in y direction.
        method : str, optional
            Interpolation method (nearest, cubic or linear see
            `scipy.interpolate.griddata`). Default is `nearest`.
        xmin : float, optional
            Mininum X value.
        xmax : float, optional
            Maximum X value.
        ymin : float, optional
            Minimium Y value.
        ymax : float, optional
            Maximum Y value
        """
        if xmin is None:
            xmin = np.min([s.df['x'].min() for s in self.surveys])
        if xmax is None:
            xmax = np.max([s.df['x'].max() for s in self.surveys])
        if ymin is None:
            ymin = np.min([s.df['y'].min() for s in self.surveys])
        if ymax is None:
            ymax = np.max([s.df['y'].max() for s in self.surveys])
            
        for survey in self.surveys:
            survey.gridData(nx=nx, ny=ny, method=method, xmin=xmin, xmax=xmax,
                           ymin=ymin, ymax=ymax)
        
    

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
        if self.projection is not None:
            targetProjection = self.projection
        for survey in self.surveys:
            survey.convertFromNMEA(targetProjection=targetProjection)
    
    def setProjection(self, targetProjection='EPSG:27700'):
        """Set surveys projection to the targetProjection.
        
        Parameters
        ----------
        targetProjection : str, optional
            Target CRS, in EPSG number: e.g. `targetProjection='EPSG:27700'`
            for the British Grid.
        """
        self.projection = targetProjection
        for survey in self.surveys:
            survey.projection = targetProjection
        

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
        ax.set_ylabel('Elevation [m]')
        ax.set_title('{:s}, sample {:d} (RMSE={:.2f})'.format(
            self.surveys[index].name, ipos, self.rmses[index][ipos]))
        

    
    def showResults(self, index=0, ax=None, vmin=None, vmax=None,
                    maxDepth=None, padding=1, cmap='viridis_r', dist=True,
                    contour=False, rmse=False, errorbar=False, overlay=False,
                    elev=False, doi=False):
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
        dist : bool, optional
            If `True`, true distances are used for X. Otherwise measurement
            index is used.
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
        elev : bool, optional
            If `True`, each inverted profile will be adjusted according to
            elevation.
        doi : bool, optional
            If `True` and `computeDOI()` was called, the estimated DOI from 
            above which 70% of the deeper coil configuration is coming from will
            be plotted on top of the graph as a red dotted line.
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
        depths = -np.c_[depths, np.ones(depths.shape[0])*maxDepth]
        if elev:
            depths = depths + self.surveys[index].df['elevation'].values[:,None]
        if dist:
            if len(self.surveys) == 0:
                dist = False # no survey to take position from
        
        # vertices
        nlayer = sig.shape[1]
        nsample = sig.shape[0]
        x = np.arange(nsample+1) # number of samples + 1
        if nsample == 1:
            dist = False
        if dist:
            xy = self.surveys[index].df[['x','y']].values
            distance = np.sqrt(np.sum(np.diff(xy, axis=0)**2, axis=1))
            distance = np.r_[[0], distance, distance[[-1]]]
            x = np.cumsum(distance)

        xs = np.tile(np.repeat(x, 2)[1:-1][:,None], nlayer+1)
        ys = np.repeat(depths, 2, axis=0)
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
                levels = np.linspace(vmin, vmax, 14)
            else:
                levels = None
            # cax = ax.tricontourf(xc, yc, zc.flatten('F'),
                                 # cmap=cmap, levels=levels, extend='both')
            xc = centroid[:,0].reshape((-1, nsample)).T
            yc = centroid[:,1].reshape((-1, nsample)).T
            zc = sig
            cax = ax.contourf(xc, yc, zc, cmap=cmap, levels=levels, extend='both')
            # set clip path
            # pathvert = np.c_[np.r_[xs[:,0], xs[::-1,0]], np.r_[ys[:,-1], ys[::-1,0]]]
            # path = mpath.Path(pathvert)
            # patch = mpatches.PathPatch(path, facecolor='none', edgecolor='k')
            # ax.add_patch(patch)
            # for col in cax.collections:
            #     col.set_clip_path(patch)
    
            fig.colorbar(cax, ax=ax, label='EC [mS/m]')
        else:
            coll = PolyCollection(coordinates, array=sig.flatten('F'), cmap=cmap)
            coll.set_clim(vmin=vmin, vmax=vmax)
            ax.add_collection(coll)
            pad = 0.15 if rmse else 0.05
            fig.colorbar(coll, label='EC [mS/m]', ax=ax, pad=pad)
        
        if rmse:
            ax2 = ax.twinx()
            ax2.plot(x[:-1] + np.diff(x)/2, self.rmses[index], 'kx-')
            ax2.set_ylabel('RRMSE [%]')
            
        if errorbar or overlay:
            vc = ~self.fixedConds
            vd = ~self.fixedDepths
            
        if errorbar:
            for i, ii in enumerate(np.where(vd)[0]):
                xm = x[:-1] + np.diff(x)/2
                ym = depths[:,ii+1] # +1 because of zero depths
                yerr = self.pstds[index][:,i]
                ax.errorbar(xm, ym, yerr=yerr, color='k', linestyle='none', capsize=2)
            
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
        if doi:
            if len(self.dois) > 0:
                dois = -self.dois[index]
                ax.step(x, np.r_[dois, dois[-1]], 'r:', where='post')
                
        if dist:
            ax.set_xlabel('Distance [m]')
        else:
            ax.set_xlabel('Samples')
        ax.set_ylabel('Elevation [m]')
        if len(self.surveys) > 0:
            ax.set_title(self.surveys[index].name)
        ax.set_ylim([np.min(depths), np.max(depths)])
        ax.set_xlim([np.min(x), np.max(x)])
        def format_coord(i,j):
            col = int(np.floor(i))
            if col < sig.shape[0]:
                row = int(np.where(-depths[col,:] < j)[0].min())-1
                return 'x={0:.4f}, y={1:.4f}, value={2:.4f}'.format(col, row, sig[col, row])
            else:
                return ''
        ax.format_coord = format_coord
        fig.tight_layout()



    def show3D(self, index=0, pl=None, vmin=None, vmax=None,
                maxDepth=None, cmap='viridis_r', elev=False, edges=False,
                background_color=(0.8,0.8,0.8), pvslices=([],[],[]),
                pvthreshold=None, pvgrid=False, pvcontour=[]): # pragma: no cover
        """Show inverted model in 3D with pyvista (pip install pyvista).
        
        Parameters
        ----------
        index : int, optional
            Index of the survey to plot.
        pl : pyvista.Plotter, optional
            If specified, the graph will be plotted against it.
        vmin : float, optional
            Minimum value of the colorbar.
        vmax : float, optional
            Maximum value of the colorbar.
        maxDepth : float, optional
            Maximum negative depths of the graph.
        cmap : str, optional
            Name of the Matplotlib colormap to use.
        elev : bool, optional
            If `True`, each inverted profile will be adjusted according to
            elevation.
        edges : bool, optional
            If `True`, edges will be displayed.
        background_color : tuple, optional 
            Background color assigned to pyvista plotter object when created.
        pvslices : tuple of list of float, optional
            Determine the X, Y, Z slices. e.g.: ([3], [], [-3, -4]) will add
            a slice normal to X in 3 and two slices normal to Z in -3 and -4.
        pvthreshold : list of two floats, optional
            Keep values between pvthreshold[0] and pvthreshold[1].
        pvgrid : bool, optional
            Show grid or not.
        pvcontour : list of float, optional
            Values of the isosurface to be plotted.
        """
        try:
            import pyvista as pv
        except:
            raise ImportError('Please install pyvista: pip install pyvista.')
            return
        fpl = True if pl is not None else False # flag if pl is supplied
        
        # save vtk to temporary folder
        folder = tempfile.TemporaryDirectory()
        fname = os.path.join(folder.name, 'emagpyMesh.vtk')
        self.saveVTK(fname, index=index, elev=elev)
        
        # vmin/vmax values
        if vmin is None:
            vmin = np.nanmin(self.models[index])
        if vmax is None:
            vmax = np.nanmax(self.models[index])

        # create plotter is none specified
        if pl is None:
            pl = pv.Plotter()
        if type(pl) != type(pv.PlotterITK()): # PlotterITK doesn't have .set_background()
            pl.set_background(background_color)
        self.pvmesh = pv.read(fname)
        
        # thresholding between pvthreshold[0] and pvthreshold[1]
        if pvthreshold is not None:
            if isinstance(pvthreshold, list):
                ec = self.models[index].flatten()
                if pvthreshold[0] is None:
                    pvthreshold[0] = np.nanmin(ec)
                if pvthreshold[1] is None:
                    pvthreshold[1] = np.nanmax(ec)
            self.pvmesh = self.pvmesh.threshold(value=pvthreshold)
        
        # if contour, then convert to point data
        if len(pvcontour) > 0:
            self.pvmesh = self.pvmesh.cell_data_to_point_data()
            self.pvmesh = self.pvmesh.contour(isosurfaces=pvcontour)
        
        # show grid
        if pvgrid:
            pl.show_grid(color='k')
        
        # create slices
        if np.sum([len(a) for a in pvslices]) > 0:
            pl.add_mesh(self.pvmesh.outline(), color='k')
            for i, ss in enumerate(pvslices):
                normal = np.zeros(3)
                normal[i] = 1
                for s in ss:
                    origin = np.zeros(3)
                    origin[i] = s
                    mesh_slice = self.pvmesh.slice(normal=normal, origin=origin)
                    pl.add_mesh(mesh_slice,
                                cmap=cmap,
                                clim=[vmin, vmax],
                                show_edges=edges,
                                scalar_bar_args={'color':'k'})
        else:
            if type(pl) != type(pv.PlotterITK()): # PlotterITK is strange
                pl.add_mesh(self.pvmesh,
                            cmap=cmap,
                            clim=[vmin,vmax],
                            show_edges=edges,
                            scalar_bar_args={'color':'k',
                                             'vertical':False,
                                             'title_font_size':16,
                                             'label_font_size':14})
            else:
                pl.add_mesh(self.pvmesh)
        
        # show mesh and clean temporary folder
        if fpl is False:
            pl.show()
        folder.cleanup()
        
        
    
    def saveVTK(self, fname, index=0, maxDepth=None, elev=False):
        """ Writes a vtk file.
        
        Parameters
        ------------
        fname : str
            Path where to save the file.
        index : int, optional
            Index of the survey to save.
        maxDepth : float, optional
            Maximum positively defined depth of the bottom infinite layer.
        elev : bool, optional
            If `True`, topography will be added.
        """
        if fname[-4:] != '.vtk':
            fname = fname + '.vtk'
            
        depths = self.depths[index]
        sig = self.models[index]
        xy = self.surveys[index].df[['x','y']].values
        nlayer = sig.shape[1]
        nsample = xy.shape[0]
        padding = 1
        if maxDepth is None:
            maxDepth = np.nanpercentile(depths, 98) + padding
        depths = -np.c_[np.zeros(nsample), depths, np.ones(depths.shape[0])*maxDepth]
        if elev:
            depths = depths + self.surveys[index].df['elevation'].values[:,None]

        triang = mtri.Triangulation(xy[:,0], xy[:,1])
        triangles = triang.triangles # connection matrix for 2D
        ntri = triangles.shape[0]
        
        # build node and connection matrix for prism
        z = depths.flatten('F')
        nodes = np.c_[np.tile(xy.T, nlayer+1).T, z]
        nnodes = nodes.shape[0]
        ec = sig.flatten('F')
        conMatrix = []
        vals = []
        for i in range(nlayer):
            conUp = triangles + i*nsample
            conDown = triangles + (i+1)*nsample
            vals.append(np.mean(ec[conUp], axis=-1))
            conMatrix.append(np.c_[conUp, conDown])
        conMatrix = np.vstack(conMatrix)
        vals = np.hstack(vals)
        nelem = conMatrix.shape[0]
        
        # check the node to cell interpolation
        # fig, axs = plt.subplots(2, 1, figsize=(14,3))
        # ax = axs[0]
        # cax = ax.scatter(xy[:,0], xy[:,1], s=35, c=ec[:nsample], cmap='viridis_r')
        # fig.colorbar(cax, ax=ax)
        # ax = axs[1]
        # xyc = np.mean(xy[triangles], axis=1)
        # ax.scatter(xyc[:,0], xyc[:,1], s=35, c=vals[:xyc.shape[0]], cmap='viridis_r')
        # ax.triplot(triang, lw=0.5, color='k')
        # fig.colorbar(cax, ax=ax)
        # ax.set_aspect('equal')
        # fig.show()
        
        # writing vtk file
        with open(fname, 'w') as f:
            f.write('# vtk DataFile Version 3.0\n')
            f.write('{:s}\n'.format(self.surveys[index].name))
            f.write('ASCII\nDATASET UNSTRUCTURED_GRID\n')
            
            # node coordinates
            f.write('POINTS {:d} double\n'.format(nnodes))
            for i in range(nnodes):
                f.write('{:8.6f} {:8.6f} {:8.6f}\n'.format(
                    nodes[i,0], nodes[i,1], nodes[i,2]))
            
            # connection matrix   
            f.write('CELLS {:d} {:d}\n'.format(nelem, nelem*(6 + 1)))
            # second argument is nelem*(number of nodes per elem + 1)
            for i in range(nelem):
                f.write('{:d}\t{:s}\n'.format(
                    6, # number of vertices
                    ' '.join(conMatrix[i,:].astype(str))))
            
            # cell types (prism = vtkWedge = code 13)
            # https://vtk.org/doc/nightly/html/classvtkWedge.html#details
            f.write('CELL_TYPES {:d}\n'.format(nelem))
            f.write(' '.join((np.ones(nelem)*13).astype(int).astype(str)))
            f.write('\n')
            
            # cell data
            f.write('CELL_DATA {:d}\n'.format(nelem))
            f.write('SCALARS {:s} double 1\n'.format('EC'))
            f.write('LOOKUP_TABLE default\n')
            [f.write('{:8.6f} '.format(a)) for a in vals[:nelem]]
            f.write('\n')
            # TODO add pstds
            
            # point data
            f.write('POINT_DATA {:d}\n'.format(nnodes))        
        


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
        for i in range(len(self.models)):
            if i != ref:
                self.models[i] = self.models[i] - self.models[ref]
        
    
    def getRMSE(self, forwardModel=None):
        """Returns RMSE for all coils (columns) and all surveys (row).
        
        Parameters
        ----------
        forwardModel : str, optional
            Type of forward model:
                - CS : Cumulative sensitivity (default)
                - FS : Full Maxwell solution with low-induction number (LIN) approximation
                - FSeq : Full Maxwell solution without LIN approximation (see Andrade 2016)
            If `None` (default), the forward model used for the inversion is used.
        """
        if forwardModel is None:
            forwardModel = self.forwardModel
        dfsForward = self.forward(forwardModel=forwardModel)
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
        
        
        
    def showMisfit(self, index=0, coil='all', forwardModel=None, ax=None):
        """Show Misfit after inversion.
            
        Parameters
        ----------
        index : int, optional
            Index of the survey to plot.
        coil : str, optional
            Which coil to plot. Default is all.
        forwardModel : str, optional
            Type of forward model:
                - CS : Cumulative sensitivity (default)
                - FS : Full Maxwell solution with low-induction number (LIN) approximation
                - FSeq : Full Maxwell solution without LIN approximation (see Andrade 2016)
            If `None` (default), the forward model used for the inversion is used.
        ax : matplotlib.Axes, optional
            If specified the graph will be plotted on this axis.
        """
        if forwardModel is None:
            forwardModel = self.forwardModel
        dfsForward = self.forward(forwardModel=forwardModel)
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
        
        
        
    def showOne2one(self, index=0, coil='all', forwardModel=None, ax=None,
                    vmin=None, vmax=None):
        """Show one to one plot with inversion results.
            
        Parameters
        ----------
        index : int, optional
            Index of the survey to plot.
        coil : str, optional
            Which coil to plot. Default is all.
        forwardModel : str, optional
            Type of forward model:
                - CS : Cumulative sensitivity (default)
                - FS : Full Maxwell solution with low-induction number (LIN) approximation
                - FSeq : Full Maxwell solution without LIN approximation (see Andrade 2016)
            If `None` (default), the forward model used for the inversion is used.
        ax : matplotlib.Axes, optional
            If specified the graph will be plotted on this axis.
        vmin : float, optional
            Minimum ECa on the graph.
        vmax : float, optional
            Maximum ECa on the graph.
        """
        if forwardModel is None:
            forwardModel = self.forwardModel
        dfsForward = self.forward(forwardModel=forwardModel)
        survey = self.surveys[index]
        cols = survey.coils
        obsECa = survey.df[cols].values
        simECa = dfsForward[index][cols].values
        #print('number of nan', np.sum(np.isnan(obsECa)), np.sum(np.isnan(simECa)))
        rmses = np.sqrt(np.sum((obsECa - simECa)**2, axis=0)/np.sum(obsECa**2, axis=0)/obsECa.shape[0])*100
        rmse = np.sum(rmses)/len(self.coils)
        if vmin is None:
            vmin = np.nanpercentile(obsECa.flatten(), 5)
        if vmax is None:
            vmax = np.nanpercentile(obsECa.flatten(), 95)
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_title('RRMSE: {:.3f} %'.format(rmse))
        ax.plot(obsECa, simECa, '.')
        ax.plot([vmin, vmax], [vmin, vmax], 'k-', label='1:1')
        ax.set_xlim([vmin, vmax])
        ax.set_ylim([vmin, vmax])
        ax.set_xlabel('Observed ECa [mS/m]')
        ax.set_ylabel('Simulated ECa [mS/m]')
        ax.legend(['{:s} ({:.2f} %)'.format(c, r) for c, r in zip(cols, rmses)])
    
    
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



    def resMod2EC(self, fnameECa, fnameResMod, binInt=None, nbins=None, calib=None):
        """Convert mesh data to dfec array to be used in calibrate.
        
        Parameters
        ----------
        fnameECa : str
            Path of the .csv file with the ECa data collected on the calibration points.
        fnameResMod : str
            Path of the .dat file with the restivity model.
        binInt : int, optional
            Bin interval in metres, over which to average resistivity model.
        nbins : int, optional
            Number of bins to average the resistivity model over.
        calib str, optional
            If specified, will apply a GF correction. Note that the main dataset
            needs to be corrected as well.
        """
        # import data
        s = Survey(fnameECa)
        if calib is not None:
            s.gfCorrection(calib=calib)
        eca = s.df[['x'] + s.coils].values
        resmod = pd.read_table(fnameResMod, sep='\s+', header=None).values
        min_xpos = np.min(eca[:,0])
        max_xpos = np.max(eca[:,0])
        
        if nbins is None:
            if binInt is None:
                nbins = int(eca.shape[0]-1)
            else:
                nbins = int((max_xpos - min_xpos)//binInt)
                
        if nbins > eca.shape[0]:
            raise ValueError('You have specified {:d} bins but you only have {:d} data points.'.format(
                nbins, eca.shape[0]))

        # check if mesh is triangular or quadrilateral, meshes with topography may be read wrong
        # if the product of the number of unique X and Z values equals the number of rows, it's a quad mesh
        if len(np.unique(resmod[:,0]))*len(np.unique(resmod[:,1])) == len(resmod[:,1]):
            meshType = 'quad'
            print('Mesh is quadrilateral.')
        else:
            meshType = 'tri'
            print('Mesh is triangular, it will be regridded.')

        # if triangular we will grid the data
        if meshType == 'tri':
            x = resmod[:,0]
            z = resmod[:,1]
            res = resmod[:,2]
            xi = np.arange(np.min(x), np.max(x), 0.25)
            zi = np.linspace(np.min(z), np.max(z), 15) # 15 layers in Y
            xi, zi = np.meshgrid(xi, zi)
            resi = griddata((x,z), res, (xi, zi), method='linear') # linear interpolation
            x = np.unique(xi)
            z = np.unique(zi)
            resmodxz = np.array(np.meshgrid(x,z)).T.reshape(-1,2)            
            res = resi.T.flatten()
            resmod = np.concatenate((resmodxz, res[:,None]), axis=1)
            resmod = resmod[~np.isnan(resmod[:,2]),:]
    
        resmod = resmod[(resmod[:,0] >= min_xpos) & (resmod[:,0] <= max_xpos),:]    
        mid_depths = -np.unique(resmod[:,1])
        resmod = resmod[np.where((resmod[:,0] >= min_xpos) & (resmod[:,0] <= max_xpos)),:][0]
        
        # computes mean EC for each bin
        bins = np.linspace(min_xpos, max_xpos, nbins+1)
        bin_id = np.digitize(np.unique(resmod[:,0]), bins+1)
        ec = np.ones((nbins, len(mid_depths)))
        mid_depths_r = mid_depths[::-1]
        for i in range(0, nbins):
            for j in range(0, len(mid_depths)):
                idepth = resmod[:,1] == -mid_depths_r[j]
                ibins = bin_id == i+1
                ec[i,j] = 1000/np.mean(resmod[idepth,:][ibins, 2])
                
        # compute mean ECa for each bin
        bin_id = np.digitize(np.unique(eca[:,0]), bins)
        depths = (mid_depths_r[1:] + mid_depths_r[:-1])/2
        eca2 = np.zeros((nbins, eca.shape[1]))
        for i in range(0, nbins):
            ie = bin_id == i+1
            if np.sum(ie) > 0:
                eca2[i,:] = np.mean(eca[ie,:], axis=0)
        eca2 = np.array(eca2)
        eca2 = eca2[~np.isnan(eca2).any(axis=1)]
        
        return ec, depths, eca2[:, 1:]



    def calibrate(self, fnameECa, fnameEC=None, fnameResMod=None, 
                  forwardModel='CS', ax=None, apply=False, dump=None,
                  nbins=None, binInt=None, calib=None):
        """Calibrate ECa with given EC profile.
        
        Parameters
        ----------
        fnameECa : str
            Path of the .csv file with the ECa data collected on the calibration points.
        fnameEC : str
            Path of the .csv file with the EC profile data. One row per location
            corresponding to the rows of fnameECa. The header should be the
            corresponding depths in meters positive downards.
        fnameResMod : str
            File name of resmod, R2 format e.g. f001_mod.dat.
        forwardModel : str, optional
            Forward model to use. Either CS (default), FSlin or FSeq.
        ax : matplotlib.Axes
            If specified the graph will be plotted against this axis.
        apply : bool, optional
            If `True` the ECa values will be calibrated. If `False`, the relationship
            will just be plotted.
        dump : function, optional
            Display different parts of the calibration.
        binInt : int, optional
            Bin interval in metres, over which to average resistivity model.
        nbins : int, optional
            Number of bins to average the resistivity model over.
        calib : str, optional
            If specified, the corresponding GF correction will be applied prior
            to ERT calibration. This is needed if you apply the correction to
            your main dataset as well! See `gfCorrection()`.
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
            
        if fnameResMod is not None:
            ec, depths, eca = self.resMod2EC(fnameECa=fnameECa,
                                             fnameResMod=fnameResMod,
                                             nbins=nbins, binInt=binInt,
                                             calib=calib)
            depths = depths
            dfec = pd.DataFrame(ec)
            dfeca = pd.DataFrame(eca, columns=survey.coils)
            survey.df = dfeca

        if fnameEC is not None:
            dfec = pd.read_csv(fnameEC)
            depths = np.abs([float(a[1:]) for a in dfec.columns if a[0] == 'd']) # those are the depths of at mid layer
            depths = depths[:-1] + np.diff(depths) # those are depths of the bottom of the layer
            if calib is not None:
                survey.gfCorrection(calib=calib)
            
        if survey.df.shape[0] != dfec.shape[0]:
            raise ValueError('input ECa and inputEC should have the same number of rows so the measurements can be paired.')
        
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
            dump('{:s}: ECa(ERT) = {:.2f} * ECa(EMI) {:+.2f} (R^2={:.2f})'.format(coil, slope, intercept, r_value**2))
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
        
        
        
    def crossOverPointsError(self, index=0, coil=None, ax=None, dump=print):
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
        survey.crossOverPointsError(coil=coil, ax=ax, dump=dump)
    
    
    
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
                  cmap='viridis_r', ax=None, pts=False):
        """Show depth slice of EC (if islice > 0) and depth (if islice < 0).
        
        Parameters
        ----------
        index : int, optional
            Survey index. Default is first.
        islice : int, optional
            Layer index (if islice > 0). Default is first layer. If islice < 0,
            the depths will be display instead (e.g. islice = -1 will display
            the depth of the bottom of the first layer).
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
        if islice >= 0:
            z = self.models[index][:,islice]
            label = 'EC [mS/m]'
            title = 'Layer {:d}'
        else:
            islice = np.abs(islice) - 1
            z = self.depth[index][:,islice]
            label = 'Depth [m]'
            title = 'Bottom depth of layer {:d}'
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
            # ax.set_xlim([np.nanmin(x), np.nanmax(x)])
            # ax.set_ylim([np.nanmin(y), np.nanmax(y)])
        else:
            levels = np.linspace(vmin, vmax, 14)
            cax = ax.tricontourf(x, y, z, levels=levels, cmap=cmap, extend='both')
            if pts:
                ax.plot(x, y, 'k+')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        fig.colorbar(cax, ax=ax, label=label)
        # depths = np.r_[[0], self.depths0, [-np.inf]]
        # ax.set_title('{:.2f}m - {:.2f}m'.format(depths[islice], depths[islice+1]))
        ax.set_title(title.format(islice+1))
        
        
    def showDepths(self, index=0, idepth=0, contour=False, vmin=None, vmax=None,
                  cmap='viridis_r', ax=None, pts=False):
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

    def computeDOI(self, conds=None, depths=None, nlayers=50):
        """Compute a depth of investigation (DOI) for each 1D EC model.
        Sensitivity cutoff at 0.3, i.e. 70% of signal comes from above the DOI.
        
        Parameters
        ----------
        forwardModel : str, 
            forward model
        nlayers : int, 
            number of layers for model. Depth index. Default is first depth.
        step : int, 
            number of steps to use for each range
        fixedParam : list of int and None,
            array of whether parameters out to be fixed in grid parameter search or not
        bnds : list of float, optional
            If specified, will create bounds for the inversion parameters
        topPer : int, 
            Top X percentage of models to be used for model boundaries
        """
        # initial argument check
        ilocal = False
        if conds is None:
            conds = self.models.copy()
            ilocal = True
        else:
            conds = [conds]
        if depths is None:
            depths = self.depths.copy()
        else:
            depths = [depths]
        
        dois = []
        d = np.linspace(0, np.max(depths[0]), nlayers)[1:]
        ddiff = np.diff(d, axis=0)/2
        dm = mdepths = np.r_[d[0]/2, d[1:] + ddiff, d[-1] + ddiff[-1]]
        print('\rComputing DOI {:d}/{:d} done'.format(0, len(depths)), end='')
        for i in range(len(depths)): # for each survey
            # discretize
            n = depths[i].shape[0]
            depth2 = np.ones((n, len(d)))*d[None,:] # depth of bottom of each layer
            mdepths = np.c_[depths[i][:,0][:,None]/2,
                            depths[i][:,1:] + np.diff(depths[i], axis=1)/2,
                            np.ones((n, 1))*(1+np.max(depths[0]))]
            # interpolate
            conds2 = np.array([np.interp(dm, mdepths[j,:], conds[i][j,:]) for j in range(n)])
        
            sens = self.computeSens(forwardModel=self.forwardModel, coils=None,
                                    models=[conds2], depths=[depth2])
            # NOTE coils need to be None to tell self.forward() that we
            # are not creating a new survey
            S = sens[0] # Nsample x Ncoils x Nprofiles
            S2 = S[::-1,:,:]
            cumS = np.cumsum(S2, axis=0)[::-1,:,:]
            cumS = cumS/np.max(cumS, axis=0) # normalize so that top is 1
            imin = np.argmin(np.abs(cumS - 0.3), axis=0) # depth closes to 70% cumulative sensitivity
            imax = np.max(imin, axis=0) # deeper depth amongst coil config
            doi = dm[imax]
            dois.append(doi)
            print('\rComputing DOI {:d}/{:d} done'.format(i+1, len(depths)), end='')
        print('')

        self.dois = dois



    def gridParamSearch(self, forwardModel, nlayers=2, step=25, misfitMax=0.1,
                        regularization='l1', fixedParams=None, bnds=None):
        """Using a grid based parameter search method this returns a list of
        best models for a specified number of layers, the minimum and maximum
        parameter bounds for the the top x percentage of models is also returned.
        This method can be used to 'invert' data or provide initial model
        parameter and parameter bounds for McMC methods.
        
        Parameters
        ----------
        forwardModel : str
            Forward model name. Either 'CS','FSlin' of 'FSeq'.
        nlayers : int, optional
            Number of layers for model. Depth index. Default is first depth.
        step : int, optional
            Number of steps to use for each range.
        fixedParam : list of int, optional
            Array of whether parameters out to be fixed in grid parameter search or not.
            TODO not explicit what this does
        bnds : list of float, optional
            If specified, will create bounds for the inversion parameters
        maxMisfit : int, optional
            Maximum allowable misfit.
            TODO then what? happens if it's over?'
        TODO regularization is not in the docstring
        TODO the docstring parameters are not ordered the same way as the argument of the method
        """
        dfeca = self.surveys[0].df.loc[:, self.coils] # TODO what if multiple survey?
        eca = np.asarray(dfeca)
        nparams = 2 * nlayers - 1   
        ndepths = nparams - nlayers

        # check arguments
        if type(fixedParams) == list and len(fixedParams) < nlayers + ndepths:
            print('Number of fixed params should match number of parameters')
        
        if type(bnds) == list and len(bnds) < nlayers + ndepths:
            print('Length of bnds should match number of parameters')

        # build bounds if not specified
        if bnds is None:
            bnds = []
            for i in range(0, ndepths):
                bnds.append(((0.1 + i, 1 + i)))
            for i in range(0, nlayers):
                bnds.append(((1, 100)))

        # combine bounds ??? TODO
        paramsRange = []
        if type(fixedParams) == list:
            for i in range(0, len(fixedParams)):
                if type(fixedParams[i]) == float:
                    paramsRange.append(((fixedParams[i], fixedParams[i])))
                else:
                    paramsRange.append(((bnds[i][0], bnds[i][1])))
        else:
            paramsRange = bnds

        # create a grid for each parameter
        params=[]
        for i in range(0, nparams):
            if paramsRange[i][0]==paramsRange[i][1]:
                params.append(paramsRange[i][0])
            else:
                params.append((np.linspace(paramsRange[i][0], paramsRange[i][1], step)))
 
        # create the N-dimensional meshgrid of parameter values
        modParams = np.array(np.meshgrid(*params)).T.reshape(-1,nparams)
        print('Will compute {:d} forward models'.format(len(modParams)))      
        
        # compute forward response
        depths = modParams[:,0:ndepths].reshape((len(modParams[:,0]),ndepths))
        conds = modParams[:,ndepths:ndepths+nlayers]
        eca_m = np.asarray(self.forward(depths=[depths], models=[conds])[0])

        bestMod = [] # TODO better to already set the size of these arrays using np.zeros((m,n))*np.nan, and then just assign items. This will make the code faster
        modList = []
        paramMin= []
        paramMax = []
        paramSd = []
        for i in range(0, eca.shape[0]): # TODO this will be slow, but not sure we can speed it up and might still be faster than other solvers
            eca_d = eca[i,:]
            if regularization == 'l1':
                coilMisfits = np.abs(eca_m - eca_d[None,:]) / eca_d[None,:]
            if regularization == 'l2':
                coilMisfits = ((eca_m - eca_d[None,:]) / eca_d[None,:])**2
            totalMisfit = np.sum(coilMisfits, axis=1) / len(self.coils)
            # converged = np.where(totalMisfit < misfitMax) # TODO this can be a boolean array
            converged = totalMisfit < misfitMax # TODO I think this can be a boolean array (boolean are faster than int arrays)
            convergedModels = np.hstack((modParams[converged,:][0],totalMisfit[converged][:,None]))
            
            if len(convergedModels) > 0:
                modList.append(convergedModels)
                paramMin.append(np.amin(convergedModels[:,:-1], axis=0))                 
                paramMax.append(np.amax(convergedModels[:,:-1], axis=0))
                paramSd.append(np.std(convergedModels[:,:-1], axis=0))
            else: 
                paramMin.append(np.amin(paramsRange, axis=1))                 
                paramMax.append(np.amax(paramsRange, axis=1))
                paramSd.append(np.std(paramsRange, axis=0))

            bestMod.append(np.append(modParams[np.where(totalMisfit == np.min(totalMisfit))[0][0],:],np.min(totalMisfit)))
            bestDepths=np.asarray(bestMod)[:,0:ndepths]
            bestConds=np.asarray(bestMod)[:,ndepths:nparams]
            bestMisfits=np.asarray(bestMod)[:,-1]
    
        return bestConds, bestDepths, bestMisfits, paramSd, paramMin, paramMax    
        
