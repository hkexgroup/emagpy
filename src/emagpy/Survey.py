#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 20:27:12 2019

@author: jkl
"""
import os
import warnings
from datetime import time, datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata, NearestNDInterpolator
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull

from emagpy.invertHelper import Q2eca



def clipConvexHull(xdata,ydata,x,y,z):
    """ set to nan data outside the convex hull of xdata, yxdata
    
    Parameters:
        xdata, ydata (arrays) : x and y position of the data collected
        x, y (arrays) : x and y position of the computed interpolation points
        z (arrays) : value of the interpolation
    
    Return:
        znew (array) : a copy of z with nan when outside convexHull
    """
    knownPoints = np.array([xdata, ydata]).T
    newPoints = np.array([x,y]).T
    _, idx = np.unique(xdata+ydata, return_index=1)
    q = ConvexHull(knownPoints[idx,:])
    hull = Delaunay(q.points[q.vertices, :])
#    qq = q.points[q.vertices,:]
        
    znew = np.copy(z)
    for i in range(0, len(z)):
        if hull.find_simplex(newPoints[i,:])<0:
            znew[i] = np.nan
    return znew


def idw(xnew, ynew, xknown, yknown, zknown, n=1):
    znew = np.zeros(len(xnew))
    for i,(x,y) in enumerate(zip(xnew, ynew)):
#        dist = pdist(x, y, xknown, yknown)
        dist = np.sqrt((x-xknown)**2+(y-yknown)**2)
        # if we consider all the points (might be not good)
        w = (1/dist)**n # exponent to be chosen
        znew[i] = np.sum(zknown*w)/np.sum(w)
    return znew


def convertFromCoord(df, targetProjection='EPSG:27700'):
    """Convert coordinates string (NMEA or DMS) to selected CRS projection.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with a 'Latitude' and 'Longitude' columns that contains NMEA
        string.
    targetProjection : str, optional
        Target CRS, in EPSG number: e.g. `targetProjection='EPSG:27700'`
        for the British Grid.
    """
    import pyproj
    
    def NMEA(arg):
        """ Convert NMEA string to WGS84 (GPS) decimal degree.
        """
        letter = arg[-1]
        if (letter == 'W') | (letter == 'S'):
            sign = -1
        else:
            sign = 1
        arg = arg[:-1]
        x = arg.index('.')
        a = float(arg[:x-2]) # degree
        b = float(arg[x-2:]) # minutes
        return (a + b/60)*sign
    
    def DMS(arg):
        """Convert convert degrees, minutes, seconds to decimal degrees
        """
        letter = arg[-1]
        if (letter == 'W') | (letter == 'S'):
            sign = -1
        else:
            sign = 1
        #extract degrees from string
        deg_idx = arg.find('°')
        deg = int(arg[:deg_idx])
        #extract minutes from string
        min_idx = arg.find("'")
        mins = int(arg[(deg_idx+1):min_idx])
        #extract seconds from string
        sec_idx = arg.find('"')
        secs = float(arg[(min_idx+1):sec_idx])
        DD = deg + (mins/60) + (secs/3600) # decimal degree calculation
        return sign*DD # return with sign
    
    check = df['Latitude'][0]
    if check.find('°') !=-1 and check.find("'") != -1:
        print("Coordinates appear to be given as Degrees, minutes, seconds ... adjusting conversion scheme")
        gps2deg = np.vectorize(DMS)
    else:
        gps2deg = np.vectorize(NMEA)
    
    
    df['lat'] = gps2deg(df['Latitude'].values)
    df['lon'] = gps2deg(df['Longitude'].values)
    
    wgs84 = pyproj.Proj("EPSG:4326") # LatLon with WGS84 datum used by GPS units and Google Earth
    osgb36 = pyproj.Proj(targetProjection) # UK Ordnance Survey, 1936 datum
    
    df['x'], df['y'] = pyproj.transform(wgs84, osgb36, 
                          df['lat'].values, df['lon'].values)

    return df
    
    

    
class Survey(object):
    """ Create a Survey object containing the raw EMI data.
    
    Parameters
    ----------
    fname : str
        Path of the .csv file to import.
    freq : float, optional
        The instrument frequency in Hz. Can be specified per coil in the .csv.
    hx : float, optional
        The height of the instrument above the ground. Can be specified per coil
        in the .csv.
    targetProjection : str, optional
        If specified, the 'Latitude' and 'Longitude' NMEA string will be
        converted to the targeted grid e.g. : 'EPSG:27700'.
    """
    def __init__(self, fname=None, freq=None, hx=None, targetProjection=None):
        self.df = None # main dataframe
        self.drift_df = None # drift station dataframe
        self.fil_df = None # filtered dataframe 
        self.freqs = [] # frequency of each coil [Hz]
        self.errorModel = None # a function that returns an error
        self.sensor = None # sensor name (+ company)
        self.coils = [] # columns with the coils names and configuration
        self.cpos = [] # orientation of the coils
        self.cspacing = [] # spacing between Tx and Rx [m]
        self.coilsInph = [] # name of the coils with inphase value in [ppt]
        self.hx = [] # height of the instrument above the ground [m]
        self.name = ''
        self.iselect = []
        self.projection = None # store the project
        if fname is not None:
            self.readFile(fname, targetProjection=targetProjection)
            if freq is not None:
                self.freqs = np.ones(len(self.coils))*freq
            if hx is not None:
                self.hx = np.ones(len(self.coils))*hx
        
    
    def readFile(self, fname, sensor=None, targetProjection=None):
        """Read a .csv file.
        
        Parameters
        ----------
        fname : str
            Filename.
        sensor : str, optional
            Name of the sensor (for metadata only).
        targetProjection : str, optional
            EPSG string describing the projection of a 'Latitude' and 'Longitude'
            column is found in the dataframe. e.g. 'EPSG:27700' for the British grid.
        """
        name = os.path.basename(fname)[:-4]
        delimiter=','
        if fname.find('.DAT')!=-1:
            delimiter = '\t'
        df = pd.read_csv(fname, delimiter=delimiter)
        self.readDF(df, name, sensor, targetProjection)
        
        
    def readDF(self, df, name=None, sensor=None, targetProjection=None):
        """Parse dataframe.
        
        Parameters
        ----------
        df : pandas.DataFrame
            A pandas dataframe where each column contains ECa per coil and
            each row a different positions.
        name : str, optional
            Name of the survey.
        sensor : str, optional
            A string describing the sensor (for metadata only).
        targetProjection : str, optional
            EPSG string describing the projection of a 'Latitude' and 'Longitude'
            column is found in the dataframe. e.g. 'EPSG:27700' for the British grid.
        """
        self.name = 'Survey 1' if name is None else name
        for c in df.columns:
            orientation = c[:3].upper()
            if ((orientation == 'VCP') | (orientation == 'VMD') | (orientation == 'PRP') |
                    (orientation == 'HCP') | (orientation == 'HMD')):
                # replace all orientation in HCP/VCP/PRP mode
                if orientation == 'HMD':
                    df = df.rename(columns={c:c.replace('HMD','VCP')})
                if orientation == 'VMD':
                    df = df.rename(columns={c:c.replace('VMD','HCP')})
                if c[-5:] == '_inph':
                    self.coilsInph.append(c)
                else:
                    self.coils.append(c)
        df = df.rename(columns={'X':'x','Y':'y','ELEVATION':'elevation'})
        if 'x' not in df.columns:
            df['x'] = np.arange(df.shape[0])
        if 'y' not in df.columns:
            df['y'] = 0
        if 'elevation' not in df.columns:
            df['elevation'] = 0
        coilInfo = [self.getCoilInfo(c) for c in self.coils]
        self.freqs = [a['freq'] for a in coilInfo]
        self.hx = [a['height'] for a in coilInfo]
        self.cspacing = [a['coilSeparation'] for a in coilInfo]
        self.cpos = [a['orientation'] for a in coilInfo]
        self.df = df
        self.sensor = sensor
        if targetProjection is not None:
            self.convertFromNMEA(targetProjection=targetProjection)

        
    def getCoilInfo(self, arg):
        arg = arg.lower()
        orientation = arg[:3]
        b = arg[3:].split('f')
        coilSeparation = float(b[0])
        if len(b) > 1:
            c = b[1].split('h')
            freq = float(c[0])
            if len(c) > 1:
                height = float(c[1])
            else:
                height = 0
        else:
            freq = 30000 # Hz default is not specified !!
            height = 0
        return {'orientation': orientation,
                'coilSeparation': coilSeparation,
                'freq': freq,
                'height': height}
        
    
    def filterRange(self, vmin=None, vmax=None):
        """Filter out measurements that are not between vmin and vmax.
        
        Parameters
        ----------
        vmin : float, optional
            Minimal ECa value, default is minimum observed.
        vmax : float, optional
            Maximum ECa value, default is maximum observed.
        """
        if vmin is not None:
            ie1 = (self.df[self.coils].values > vmin).all(1)
        else:
            ie1 = np.ones(self.df.shape[0], dtype=bool)
        if vmax is not None:
            ie2 = (self.df[self.coils].values < vmax).all(1)
        else:
            ie2 = np.ones(self.df.shape[0], dtype=bool)
        i2keep = ie1 & ie2
        print('{:d}/{:d} data removed (filterRange).'.format(np.sum(~i2keep), len(i2keep)))
        self.df = self.df[i2keep].reset_index(drop=True)
        
        
    def rollingMean(self, window=3):
        """Perform a rolling mean on the data.
        
        Parameters
        ----------
        window : int, optional
            Size of the windows for rolling mean.
        """
        cols = ['x','y'] + self.coils + self.coilsInph
        self.df[cols] = self.df[cols].rolling(window).mean()
        i2discard = self.df[self.coils].isna().any(1)
        self.df = self.df[~i2discard]
        print('dataset shrunk of {:d} measurements'.format(np.sum(i2discard)))        
    
    
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
        if coil is None:
            coil = self.coils[0]
        val = self.df[coil].values
        qmin = 0 if qmin is None else qmin
        qmax = 100 if qmax is None else qmax
        vmin = np.nanpercentile(val, qmin)
        vmax = np.nanpercentile(val, qmax)
        i2keep = (val > vmin) & (val < vmax)
        print('{:d}/{:d} data removed (fitlerPercentile).'.format(np.sum(~i2keep), len(i2keep)))
        self.df = self.df[i2keep]
    
    
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
        if coil is None:
            coil = self.coils[0]
        val = self.df[coil].values
        i2keep = np.r_[0, np.abs(np.diff(val))] < thresh
        print('{:d}/{:d} data removed (filterDiff).'.format(np.sum(~i2keep), len(i2keep)))
        self.df = self.df[i2keep]
    
    
    def show(self, coil='all', ax=None, vmin=None, 
             vmax=None, dist=False):
        """ Show the data.
        
        Parameters
        ----------
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
        if coil == 'all':
            cols = self.coils
        else:
            cols = coil
        
        if ax is None:
            fig, ax = plt.subplots()
     
        # interactive point selection
        self.iselect = np.zeros(self.df.shape[0], dtype=bool)
        xpos = np.arange(self.df.shape[0]) # number of sample, not true distance
        if dist:
            xy = self.df[['x','y']].values
            distance = np.sqrt(np.sum(np.diff(xy, axis=0)**2, axis=1))
            xpos = np.r_[[0], np.cumsum(distance)]
        
        def setSelect(ie, boolVal): # pragma: no cover
            ipoints[ie] = boolVal
            self.iselect = ipoints
    
        def onpick(event): # pragma: no cover
            xid = xpos[event.ind[0]]
            isame = xpos == xid
            if (ipoints[isame] == True).all():
                setSelect(isame, False)
            else:
                setSelect(isame, True)
            if len(y.shape) == 1:
                killed.set_xdata(x[ipoints])
                killed.set_ydata(y[ipoints])
            else:
                xtmp = []
                ytmp = []
                for i in range(y.shape[1]):
                    xtmp.append(x[ipoints])
                    ytmp.append(y[ipoints, i])
                killed.set_xdata(xtmp)
                killed.set_ydata(ytmp)
            killed.figure.canvas.draw()
    
        ax.set_title(coil)
        caxs = ax.plot(xpos, self.df[cols].values, '.-', picker=5)
        ax.legend(cols)
        ax.set_ylim([vmin, vmax])
        ax.set_xlabel('Measurements')
        if dist:
            ax.set_xlabel('Distance [m]')
        if coil[-5:] == '_inph':
            ax.set_ylabel('Inphase [ppt]')
        else:
            ax.set_ylabel('ECa [mS/m]')
        for cax in caxs:
            cax.figure.canvas.mpl_connect('pick_event', onpick)        
        killed, = ax.plot([],[],'rx')
        x = xpos
        y = self.df[cols].values
        ipoints = np.zeros(len(x), dtype=bool)


    def dropSelected(self): # pragma: no cover
        i2keep = ~self.iselect
        print('{:d}/{:d} data removed (filterDiff).'.format(np.sum(~i2keep), len(i2keep)))
        self.df = self.df[i2keep].reset_index(drop=True)


    def tcorrECa(self, tdepths, tprofile):
        """Temperature correction using XXXX formula.
        
        Parameters
        ----------
        tdepths : array-like
            Depths in meters of the temperature sensors (negative downards).
        tprofile : array-like
            Temperature values corresponding in degree Celsius.
        """
        # TODO if ECa -> compute an 'apparent' temperature
        # TODO what about per survey ?
        pass


    def convertFromNMEA(self, targetProjection='EPSG:27700'): # British Grid 1936
        """Convert coordinates string (NMEA or DMS) to selected CRS projection.
    
        Parameters
        ----------
        targetProjection : str, optional
            Target CRS, in EPSG number: e.g. `targetProjection='EPSG:27700'`
            for the British Grid.
        """
        self.projection = targetProjection
        self.df = convertFromCoord(self.df, targetProjection)


    
    def showMap(self, coil=None, contour=False, ax=None, vmin=None, vmax=None,
                pts=False, cmap='viridis_r', xlab='x', ylab='y', nlevel=7):
        """ Display a map of the measurements.
        
        Parameters
        ----------
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
        if coil is None:
            coil = self.coils[0]
#        if coil == 'all': # trick for ui
#            coil = self.coils[-1]
        x = self.df['x'].values
        y = self.df['y'].values
        val = self.df[coil].values
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        if vmin is None:
            vmin = np.nanpercentile(val, 5)
        if vmax is None:
            vmax = np.nanpercentile(val, 95)
        ax.set_title(coil)
        if contour is True:
            levels = np.linspace(vmin, vmax, nlevel)
            cax = ax.tricontourf(x, y, val, levels=levels, extend='both', cmap=cmap)
            if pts is True:
                ax.plot(x, y, 'k+')
        else:
            cax = ax.scatter(x, y, s=15, c=val, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_xlim([np.nanmin(x), np.nanmax(x)])
        ax.set_ylim([np.nanmin(y), np.nanmax(y)])
        if coil[-5:] == '_inph':
            fig.colorbar(cax, ax=ax, label='Inphase [ppt]')
        else:
            fig.colorbar(cax, ax=ax, label='ECa [mS/m]')
        

    def saveMap(self, fname, coil=None, nx=100, ny=100, method='linear',
                xmin=None, xmax=None, ymin=None, ymax=None, color=True,
                cmap='viridis_r', vmin=None, vmax=None, nlevel=14):
        """Save a georeferenced raster TIFF file.
        
        Parameters
        ----------
        fname : str
            Path of where to save the .tiff file.
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
        try:
            import rasterio
            from rasterio.transform import from_origin
        except:
            raise ImportError('rasterio is needed to save georeferenced .tif file. Install it with "pip install rasterio"')
        
        if coil is None:
            coil = self.coils[0]
        xknown = self.df['x'].values
        yknown = self.df['y'].values
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
        values = self.df[coil].values
        if method == 'idw':
            z = idw(x, y, xknown, yknown, values)
            z = z.reshape(X.shape)
        elif method == 'kriging':
            from pykrige.ok import OrdinaryKriging
            gridx = np.linspace(xmin, xmax, nx)
            gridy = np.linspace(ymin, ymax, ny)
            OK = OrdinaryKriging(xknown, yknown, values, variogram_model='linear',
                                 verbose=True, enable_plotting=False, nlags=25)
            z, ss = OK.execute('grid', gridx, gridy)
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
                        
    
    
    def gridData(self, nx=100, ny=100, method='nearest', xmin=None, xmax=None,
                 ymin=None, ymax=None):
        """ Grid data.
        
        Parameters
        ----------
        nx : int, optional
            Number of points in x direction.
        ny : int, optional
            Number of points in y direction.
        xmin : float, optional
            Mininum X value.
        xmax : float, optional
            Maximum X value.
        ymin : float, optional
            Minimium Y value.
        ymax : float, optional
            Maximum Y value
        method : str, optional
            Interpolation method (nearest, cubic or linear see
            `scipy.interpolate.griddata`) or IDW (default).
        """
        xknown = self.df['x'].values
        yknown = self.df['y'].values
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
        inside = np.ones(nx*ny)
        inside2 = clipConvexHull(self.df['x'].values,
                                 self.df['y'].values,
                                 X.flatten(), Y.flatten(), inside)
        ie = ~np.isnan(inside2)
        df = pd.DataFrame()
        df['x'] = X.flatten()
        df['y'] = Y.flatten()
        for col in np.r_[self.coils, ['elevation']]:
            values = self.df[col].values
            if method == 'idw':
                z = idw(X.flatten(), Y.flatten(), xknown, yknown, values)
            elif method == 'kriging':
                from pykrige.ok import OrdinaryKriging
                gridx = np.linspace(xmin, xmax, nx)
                gridy = np.linspace(ymin, ymax, ny)
                OK = OrdinaryKriging(xknown, yknown, values, variogram_model='linear',
                                     verbose=True, enable_plotting=False, nlags=25)
                z, ss = OK.execute('grid', gridx, gridy)
            else:
                z = griddata(np.c_[xknown, yknown], values, (X, Y), method=method)
            df[col] = z.flatten()
        self.df = df[ie]
        
    
    def crossOverPointsError(self, coil=None, ax=None, dump=print, minDist=1):
        """ Build an error model based on the cross-over points.
        
        Parameters
        ----------
        coil : str, optional
            Name of the coil.
        ax : Matplotlib.Axes, optional
            Matplotlib axis on which the plot is plotted against if specified.
        dump : function, optional
            Output function for information.
        minDist : float, optional
            Point at less than `minDist` from each other are considered
            identical (cross-over). Default is 1 meter.
        """
        if coil is None:
            coil = self.coils[0]
        df = self.df
        dist = cdist(df[['x', 'y']].values,
                     df[['x', 'y']].values)
        ix, iy = np.where(((dist < minDist) & (dist > 0))) # 0 == same point
        ifar = (ix - iy) > 200 # they should be at least 200 measuremens apart
        ix, iy = ix[ifar], iy[ifar]
        print('found', len(ix), '/', df.shape[0], 'crossing points')
        
        if len(ix) < 10:
            dump('None or too few colocated measurements found for error model.')
            return
        
        val = df[coil].values
        x = val[ix]
        y = val[iy]
        means = np.mean(np.c_[x,y], axis=1)
        error = np.abs(x - y)
        
        # bin data (constant number)
        nbins = 30 # number of data per bin
        end = int(np.floor(len(means)/nbins)*nbins)
        errorBinned = error[:end].reshape((-1, nbins)).mean(axis=1)
        meansBinned = means[:end].reshape((-1, nbins)).mean(axis=1)
        
        # bin data (constant width)
        # errorBinned, binEdges, _ = binned_statistic(
        #         means, error, 'mean', bins=20)
        # meansBinned = binEdges[:-1] + np.diff(binEdges)
        

        # compute model
        inan = ~np.isnan(meansBinned) & ~np.isnan(errorBinned)
        inan = inan & (meansBinned > 0) & (errorBinned > 0)
        slope, intercept, r_value, p_value, std_err = linregress(
                np.log10(meansBinned[inan]), np.log10(errorBinned[inan]))
        
        self.df[coil + '_err'] = intercept + slope * self.df[coil]
            
        # plot
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_title(coil)
        ax.loglog(means, error, '.')
        ax.loglog(meansBinned, errorBinned, 'o')
        predError = 10**(intercept + slope * np.log10(means))
        eq = r'$\epsilon = {:.2f} \times \sigma^{{{:.2f}}}$'.format(10**intercept, slope)
        isort = np.argsort(means)
        ax.loglog(means[isort], predError[isort], 'k-', label=eq)
        ax.legend()
        ax.set_xlabel(r'Mean $\sigma_a$ [mS/m]')
        ax.set_ylabel(r'Error $\epsilon$ [mS/m]')
     
    
    
    def plotCrossOverMap(self, coil=None, ax=None, minDist=1):
        """Plot the map of the cross-over points for error model.
        
        Parameters
        ----------
        coil : str, optional
            Name of the coil.
        ax : Matplotlib.Axes, optional
            Matplotlib axis on which the plot is plotted against if specified.
        minDist : float, optional
            Point at less than `minDist` from each other are considered
            identical (cross-over). Default is 1 meter.
        """
        if coil is None:
            coil = self.coils[0]
        df = self.df
        dist = cdist(df[['x', 'y']].values,
                     df[['x', 'y']].values)
        ix, iy = np.where(((dist < minDist) & (dist > 0))) # 0 == same point
        ifar = (ix - iy) > 200 # they should be at least 200 measuremens apart
        ix, iy = ix[ifar], iy[ifar]
        print('found', len(ix), '/', df.shape[0], 'crossing points')
        
        # plot cross-over points
        xcoord = df['x'].values
        ycoord = df['y'].values
        icross = np.unique(np.r_[ix, iy])
        
        if ax is None:
            fig1, ax = plt.subplots()
        ax.set_title(coil)
        ax.plot(xcoord, ycoord, '.')
        ax.plot(xcoord[icross], ycoord[icross], 'ro', label='crossing points')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')


    
    def gfCorrection(self):
        """Converting GF calibrated ECa to LIN ECa.
        """
        df = self.df.copy()
        hx = self.hx[0]
        coils = ['{:s}{:.2f}'.format(a.upper(),b) for a,b in zip(self.cpos, self.cspacing)]
        print('Transformation to LIN ECa at F-{:.0f}m calibration'.format(hx))
        if hx == 0:
            gfcoefs = {'HCP1.48': 24.87076856,
                       'HCP2.82': 7.34836983,
                       'HCP4.49': 3.18322873,
                       'VCP1.48': 23.96851467,
                       'VCP2.82': 6.82559412,
                       'VCP4.49': 2.81033124,
                       'HCP0.32': 169.35849385,
                       'HCP0.71': 35.57031603,
                       'HCP1.18': 13.42529324,
                       'VCP0.32': 167.10523718,
                       'VCP0.71': 34.50404729,
                       'VCP1.18': 12.744378}
            for i, coil in enumerate(coils):
                qvalues = 0+df[self.coils[i]].values/gfcoefs[coil]*1e-3j
                df.loc[:, self.coils[i]] = Q2eca(qvalues, self.cspacing[i], f=self.freqs[i])*1000 # mS/m
        if hx == 1:
            gfcoefs = {'HCP1.48': 43.714823,
                       'HCP2.82': 9.22334343,
                       'HCP4.49': 3.51201955,
                       'VCP1.48': 77.90907085,
                       'VCP2.82': 14.02757873,
                       'VCP4.49': 4.57001088}
            ''' from mS/m to Q in ppt using GF calibration
            from Q (not in ppt) to ECa LIN using Q2eca
            '''
            for i, coil in enumerate(coils):
                qvalues = 0+df[self.coils[i]].values/gfcoefs[coil]*1e-3j # in part per thousand
                df.loc[:, self.coils[i]] = Q2eca(qvalues, self.cspacing[i], f=self.freqs[i])*1000
        self.df = df
    
    
    
    def importGF(self, fnameLo=None, fnameHi=None, device='CMD Mini-Explorer',
                 hx=0, targetProjection='EPSG:27700'):
        """Import GF instrument data with Lo and Hi file mode. If spatial data
        a regridding will be performed to match the data.
        
        Parameters
        ----------
        fnameLo : str
            Name of the file with the Lo settings.
        fnameHi : str, optional
            Name of the file with the Hi settings.
        device : str, optional
            Type of device. Default is Mini-Explorer.
        hx : float, optional
            Height of the device above the ground in meters according to the
            calibration use (e.g. `F-Ground` -> 0 m, `F-1m` -> 1 m).
        targetProjection : str, optional
            If both Lo and Hi dataframe contains 'Latitude' with NMEA values
            a conversion first is done using `self.convertFromNMEA()` before
            being regrid using nearest neightbours.
        """
        if fnameLo is None and fnameHi is None:
            raise ValueError('You must specify at least one of fnameLo or fnameHi.')
        if fnameLo is not None:
            self.name = os.path.basename(fnameLo)[:-4] # to remove .dat
        else:
            self.name = os.path.basename(fnameHi)[:-4]
            
        if device == 'CMD Mini-Explorer':
            freq = 30000
            csep = [0.32, 0.71, 1.18]
        elif device == 'CMD Explorer':
            freq = 10000
            csep = [1.48, 2.82, 4.49]
        else:
            raise ValueError('Device ' + device + ' unknown.')

        loCols = ['VCP{:.2f}'.format(a) for a in csep]
        loCols += [a + '_inph' for a in loCols]
        hiCols = ['HCP{:.2f}'.format(a) for a in csep]
        hiCols += [a + '_inph' for a in hiCols]
        cols = ['Cond.1[mS/m]', 'Cond.2[mS/m]', 'Cond.3[mS/m]',
                'Inph.1[ppt]', 'Inph.2[ppt]', 'Inph.3[ppt]']
        
        def harmonizeHeaders(df):
            x = df.columns.values
            tmp = []
            for a in x:
                tmp.append(a.replace(' [','[') # when dowloaded from usb
                            .replace('Cond1.','Cond.1') # when downloaded from usb and manual
                            .replace('Cond2.','Cond.2')
                            .replace('Cond3.','Cond.3'))
            df = df.rename(columns=dict(zip(x, tmp)))
            return df
        
        if fnameLo is not None:
            loFile = pd.read_csv(fnameLo, sep='\t')
            loFile = harmonizeHeaders(loFile)
            loFile = loFile.rename(columns=dict(zip(cols, loCols)))
        if fnameHi is not None:
            hiFile = pd.read_csv(fnameHi, sep='\t')
            hiFile = harmonizeHeaders(hiFile)
            hiFile = hiFile.rename(columns=dict(zip(cols, hiCols)))

        if fnameLo is not None and fnameHi is not None:
            if 'Latitude' not in loFile.columns and 'Latitude' not in hiFile.columns:
                if loFile.shape[0] == hiFile.shape[0]:
                    print('importGF: joining on rows.')
                    df = loFile[loCols].join(hiFile[hiCols])
                    df['x'] = np.arange(df.shape[0])
                    df['y'] = 0
                else:
                    df = None
                    raise ValueError('Can not join the dataframe as they have different lengths: {:d} and {:d}'.format(loFile.shape[0], hiFile.shape[0]))
            else:
                print('Using nearest neighbours to assign values to all merged positions.')
                # transformation of NMEA to x, y coordinates
                loFile = convertFromCoord(loFile, targetProjection)
                hiFile = convertFromCoord(hiFile, targetProjection)
                
                # regridding using nearest neighbour
                df = pd.concat([loFile, hiFile], sort=False).reset_index(drop=True)
                ie = np.zeros(df.shape[0], dtype=bool)
                ie[:loFile.shape[0]] = True
                pointsLo = loFile[['x','y']].values
                pointsHi = hiFile[['x','y']].values
                for col in loCols:
                    values = loFile[col].values
                    interpolator = NearestNDInterpolator(pointsLo, values)
                    df.loc[~ie, col] = interpolator(pointsHi)
                for col in hiCols:
                    values = hiFile[col].values
                    interpolator = NearestNDInterpolator(pointsHi, values)
                    df.loc[ie, col] = interpolator(pointsLo)
                    
            coils = loCols[:3] + hiCols[:3]
            coilsInph = loCols[3:] + hiCols[3:]
        elif fnameLo is not None:
            df = loFile
            df['x'] = np.arange(df.shape[0])
            df['y'] = 0
            coils = loCols[:3]
            coilsInph = loCols[3:]
        elif fnameHi is not None:
            df = hiFile
            df['x'] = np.arange(df.shape[0])
            df['y'] = 0
            coils = hiCols[:3]
            coilsInph = hiCols[3:]

        if df is not None:
            self.coils = coils
            self.coilsInph = coilsInph
            coilInfo = [self.getCoilInfo(c) for c in self.coils]
            self.freqs = np.repeat([freq], len(self.coils))
            self.cspacing = [a['coilSeparation'] for a in coilInfo]
            self.cpos = [a['orientation'] for a in coilInfo]
            self.hx = np.repeat([hx], len(self.coils))*0 # as we corrected it before
            self.df = df
            self.sensor = device
            self.gfCorrection() # convert calibrated ECa to LIN ECa
            
            
            
    ### jamyd91 contribution edited by jkl ### 
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
        df = self.df
        x = df['x'].values
        y = df['y'].values
        interdist = np.sqrt(np.sum(np.diff(np.c_[x,y], axis=0)**2, axis=1))
        bearing = np.zeros(len(x)-1) # first point doesn't have bearing
        azimuth = np.zeros(len(x)-1) # or azimuth

        # time handling
        if 'time' in df.columns:
            tcol = 'time'
        elif 'Time' in df.columns:
            tcol = 'Time'
        elif 'date' in df.columns:
            tcol = 'date'
        elif 'Date' in df.columns:
            tcol = 'Date'
        else:
            tcol = 'none'
        if tcol != 'none':
            times = pd.to_datetime(df[tcol], format=timef)
            
        def quadrantCheck(dx,dy):
            quad = 0
            edge = False
            if dx>0 and dy>0:#both positive
                quad = 1 # 'NE'
            elif dx>0 and dy<0:
                quad = 2 # 'SE'
            elif dx<0 and dy<0:#both negative
                quad = 3 # 'SW'
            elif dx<0 and dy>0:
                quad = 4 # 'NE'
            else:#edge case
                edge = True
                if dx==0 and dy==0:
                    quad = 0 #'0'
                elif dx==0 and dy>0:
                    quad = 0
                elif dx>0 and dy==0:
                    quad = 90
                elif dx==0 and dy<0:
                    quad = 180
                elif dx<0 and dy==0:
                    quad = 270
            return quad, edge
        quadrantCheck = np.vectorize(quadrantCheck)
                    
        dx = np.diff(x) # delta x 
        dy = np.diff(y) # delta y 
        h = np.sqrt(dx**2 + dy**2) # length of hypothenus 
        
        # computing quadrant and if point is on edge case
        quad, edge = quadrantCheck(dx,dy)
        angle = np.rad2deg(np.arcsin(dx/h)) # angle of measurement direction relative to north 

        # computing azimuth
        azimuth[edge] = quad[edge]
        
        ie = (edge == False) & (quad == 1)
        azimuth[ie] = angle[ie]
        
        ie = (edge == False) & (quad == 2)
        azimuth[ie] = 180 - angle[ie]
        
        ie = (edge == False) & (quad == 3)
        azimuth[ie] = 180 + abs(angle[ie])
        
        ie = (edge == False) & (quad == 4)
        azimuth[ie] = 360 - abs(angle[ie])
        
        # computing bearing
        bearing[azimuth > 180] = azimuth[azimuth > 180] - 180 # add 180 in order to get a postive bearing or strike 
        bearing[azimuth <= 180] = azimuth[azimuth <= 180]

        elapsed = times - times[0]

        df['interdist'] = np.r_[0, interdist] # distance between consecutive points 
        df['azimuth'] = np.r_[0, azimuth] # walking direction in terms of azimuth relative to local coordinate system
        df['bearing'] = np.r_[0, bearing] # walking direction in terms of bearing relative to local coordinate system
        df['surveyTime'] = times # times in python datetime format 
        df['elapsed(sec)'] = [a.seconds for a in elapsed]# number of seconds elasped 
            
        self.df = df
    
    
    
    def filterRepeated(self, tolerance=0.2):
        """Remove consecutive points when the distance between them is
        below `tolerance`.
        
        Parameters
        ----------
        tolerance : float, optional
            Minimum distance away previous point in order to be retained.
        """
        # error checking
        if not isinstance(tolerance,int) and not isinstance(tolerance,float):
            raise ValueError("tolerance instance should be int or float")
        if 'interdist' not in self.df.columns:
            self.computeStat()
        i2keep = self.df['interdist'].values > tolerance
        print('{:d}/{:d} data removed (filterRepeated).'.format(np.sum(~i2keep), len(i2keep)))
        self.df = self.df[i2keep].reset_index(drop=True)
        
    
    
    def fitlerBearing(self, phiMin, phiMax):
        """Keep measurements in a certain bearing range between phiMin and phiMax. 
        
        Parameters
        ----------
        phiMin : float, optional
            Minimum angle, in degrees. 
        phiMax : float, optional
            Maximum angle, in degrees.
        """
        # error checking
        if not isinstance(phiMin,int) and not isinstance(phiMin,float):
            raise ValueError("phiMin instance should be int or float")
        if not isinstance(phiMax,int) and not isinstance(phiMax,float):
            raise ValueError("phiMax instance should be int or float")
        if phiMin >= phiMax:
            raise ValueError("Min and max bearings cannot be the same, and min must be smaller!")
        if 'bearing' not in self.df.columns:
            self.computeStat()
        bearing = self.df['bearing'].values
        i2keep = (bearing > phiMin) & (bearing < phiMax)
        print('{:d}/{:d} data removed (filterBearing).'.format(np.sum(~i2keep), len(i2keep)))
        self.df = self.df[i2keep].reset_index(drop=True)
        
        
        
    def driftCorrection(self, xStation=None, yStation=None, coils='all', 
                        radius=1, fit='all', ax=None, apply=False):
        """Compute drift correction from EMI given a station point and a radius.

        Parameters
        ----------
        xStation : float, optional
            X position of the drift station. Default from first point.
        yStation : float, optional
            Y position of the drift station. Default from first point.
        coil : str or list of str, optional
            Name of coil for the analysis. Default is 'all'.
        radius : float, optional
            Radius around the station point inside which data will be averaged.
            The default is 1.
        fit : str, optional
            Type of fit. Either 'all' if one drift correction is applied on all
            data (default) or 'each' if one fit is done between each time
            the user came back to the drift point.
        ax : matplotlib.Axes, optional
            If specified, the drift graph will be plotted against. The default is None.
        apply : bool, optional
            If `True` the drift correction will be applied. The default is False.
        """
        x = self.df['x'].values
        y = self.df['y'].values
        if xStation is None:
            xStation = x[0]
        if yStation is None:
            yStation = y[0]
        if coils == 'all':
            coils = self.coils
        if isinstance(coils, str):
            coils = [coils]
        val = self.df[coils].values
        dist = np.sqrt((x-xStation)**2 + (y-yStation)**2)
        idrift = dist < radius
        igroup = np.where(np.diff(idrift) != 0)[0]
        igroup = np.r_[0, igroup, val.shape[0]]
        a = 0 if idrift[0] == True else 1
        
        # compute group mean and std
        groups = [val[igroup[i]:igroup[i+1],:] for i in np.arange(len(igroup)-1)[a::2]]
        print('{:d} drift points detected.'.format(len(groups)))
        vm = np.array([np.mean(g, axis=0) for g in groups])
        vsem = np.array([np.std(g, axis=0)/np.sqrt(len(g)) for g in groups])
        if fit == 'all':
            xs = np.linspace(0, 1, vm.shape[0])
            vpred = np.zeros(vm.shape)
            xpred = np.arange(vm.shape[0])
            for i, coil in enumerate(coils):
                slope, offset = np.polyfit(xs, vm[:,i], 1)
                print('{:s}: ECa = {:.2f} * x {:+.2f}'.format(coil, slope, offset))
                vpred[:,i] = xs * slope + offset
                if apply:
                    vm[:,i] = vm[:,i] - xs * slope - offset + np.mean(vm[:,i])
                    corr = -np.linspace(0, 1, self.df.shape[0]) * slope - offset + np.mean(vm[:,i])
                    self.df.loc[:,coil] = self.df[coil].values + corr
        elif fit == 'each':
            xs = np.array([0,1])
            vpred = np.zeros((vm.shape[0]*2-2, vm.shape[1]))
            xpred = np.repeat(np.arange(vm.shape[0]),2)[1:-1]
            for i, coil in enumerate(coils):
                for j in range(vm.shape[0]-1):
                    slope, offset = np.polyfit(xs, vm[j:j+2,i], 1)
                    vpred[j*2:j*2+2,i] = xs * slope + offset
                    if apply:
                        # correct part between two drift points
                        ie = np.zeros(self.df.shape[0], dtype=bool)
                        ie[igroup[a+j*2+1]:igroup[a+j*2+2]] = True
                        corr = -(np.linspace(0, 1, np.sum(ie)) * slope + offset) + np.mean(vm[:,i])
                        self.df.loc[ie, coil] = self.df[ie][coil].values + corr
                if apply:
                    # correct drift points
                    for j in range(vm.shape[0]):
                        ie = np.zeros(self.df.shape[0], dtype=bool)
                        ie[igroup[a+j*2]:igroup[a+j*2+1]] = True
                        corr = -vm[j,i] + np.mean(vm[:,i])
                        self.df.loc[ie, coil] = self.df[ie][coil].values + corr
                    vm[:,i] = np.mean(vm[:,i]) # for graph

        # graph
        if ax is None:
            fig, ax = plt.subplots()
        xx = np.arange(vm.shape[0])
        for i, coil in enumerate(coils):
            cax = ax.errorbar(xx, vm[:,i], yerr=vsem[:,i],
                        marker='.', label=coil, linestyle='none')
            ax.plot(xpred, vpred[:,i], '-', color=cax[0].get_color())
        ax.set_ylabel('ECa at drift station [mS/m]')
        ax.set_xlabel('Drift points')
        ax.legend()
        if apply is True:
            ax.set_title('Drift fitted and applied')
        else:
            ax.set_title('Drift fitted but not applied')
        
    
    def crossOverPointsDrift(self, coil=None, ax=None, dump=print, minDist=1,
                             apply=False): # pragma: no cover
        """ Build an error model based on the cross-over points.
        
        Parameters
        ----------
        coil : str, optional
            Name of the coil.
        ax : Matplotlib.Axes, optional
            Matplotlib axis on which the plot is plotted against if specified.
        dump : function, optional
            Output function for information.
        minDist : float, optional
            Point at less than `minDist` from each other are considered
            identical (cross-over). Default is 1 meter.
        apply : bool, optional
            If `True`, the drift correction will be applied.
        """
        if coil is None:
            coils = self.coils
        if isinstance(coil, str):
            coils = [coil]
        df = self.df
        dist = cdist(df[['x', 'y']].values,
                     df[['x', 'y']].values)
        ix, iy = np.where(((dist < minDist) & (dist > 0))) # 0 == same point
        ifar = (ix - iy) > 200 # they should be at least 200 measuremens apart
        ix, iy = ix[ifar], iy[ifar]
        print('found', len(ix), '/', df.shape[0], 'crossing points')
        
        print(ix.shape)
        print(iy.shape)
        
        val = df[coils].values
        x = val[ix,:]
        y = val[iy,:]
        misfit = np.abs(x - y)
        xsample = np.abs(ix - iy) # number of sample taken between
        # two pass
        
        if ax is None:
            fig, ax = plt.subplots()
        ax.semilogy(xsample, misfit, '.')
        ax.set_xlabel('Number of samples between two pass at same location')
        ax.set_ylabel('Misfit between the two pass [mS/m]')
        #TODO not sure about this
        
        

        
