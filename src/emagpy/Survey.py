#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 20:27:12 2019

@author: jkl
"""
import os
from datetime import time, datetime
#NB: using python datetime due to compatbility issues between pandas and matplotlib for plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from pykrige.ok import OrdinaryKriging
from scipy.stats import linregress, binned_statistic
from scipy.spatial.distance import cdist, pdist
from scipy.interpolate import griddata, NearestNDInterpolator
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from emagpy.invertHelper import emSens, Q2eca
#if __name__ == '__main__':
#    import isinpolygon as iip
#else:
#    import emagpy.isinpolygon as iip


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


def idw(xnew, ynew, xknown, yknown, zknown):
    znew = np.zeros(len(xnew))
    for i,(x,y) in enumerate(zip(xnew, ynew)):
#        dist = pdist(x, y, xknown, yknown)
        dist = np.sqrt((x-xknown)**2+(y-yknown)**2)
        # if we consider all the points (might be not good)
        w = (1/dist)**2 # exponent to be chosen
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
    
    wgs84 = pyproj.Proj("+init=EPSG:4326") # LatLon with WGS84 datum used by GPS units and Google Earth
    osgb36 = pyproj.Proj("+init=" + targetProjection) # UK Ordnance Survey, 1936 datum
    
    df['x'], df['y'] = pyproj.transform(wgs84, osgb36, 
                          df['lon'].values, df['lat'].values)

    return df
    
    
class Survey(object):
    """ Create a Survey object containing the raw EMI data.
    
    """
    def __init__(self, fname=None, freq=None, hx=None):
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
        if fname is not None:
            self.readFile(fname)
            if freq is not None:
                self.freqs = np.ones(len(self.coils))*freq
            if hx is not None:
                self.hx = np.ones(len(self.coils))*hx
        
    
    def readFile(self, fname, sensor=None):
        """Read a .csv file.
        
        Parameters
        ----------
        fname : str
            Filename.
        sensor : str, optional
            Type of sensor.
        """
        self.name = os.path.basename(fname)[:-4]
        delimiter=','
        if fname.find('.DAT')!=-1:
            delimiter = '\t'
        df = pd.read_csv(fname,delimiter=delimiter)
        for c in df.columns:
            orientation = c[:3]
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
        if 'x' not in df.columns:
            df['x'] = np.arange(df.shape[0])
        if 'y' not in df.columns:
            df['y'] = 0 # maybe not needed
        coilInfo = [self.getCoilInfo(c) for c in self.coils]
        self.freqs = [a['freq'] for a in coilInfo]
        self.hx = [a['height'] for a in coilInfo]
        self.cspacing = [a['coilSeparation'] for a in coilInfo]
        self.cpos = [a['orientation'] for a in coilInfo]
        self.df = df
        self.sensor = sensor

        
    def getCoilInfo(self, arg):
        orientation = arg[:3].lower()
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
        
    
    def keepBetween(self, vmin=None, vmax=None):
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
        ie = ie1 & ie2
        print('Deleted {:d}/{:d} measurements'.format(np.sum(~ie), len(ie)))
        self.df = self.df[ie]
        
        
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
        print('dataset shrink of {:d} measurements'.format(np.sum(i2discard)))        
    
    
    def show(self, coil='all', attr='ECa', ax=None, vmin=None, 
             vmax=None):
        """ Show the data.
        """
        if coil == 'all':
            cols = self.coils
        else:
            cols = coil
        
        if ax is None:
            fig, ax = plt.subplots()
     
        # interactive point selection
        self.iselect = np.zeros(self.df.shape[0], dtype=bool)
        xpos = np.arange(self.df.shape[0])
        
        def setSelect(ie, boolVal):
            ipoints[ie] = boolVal
            self.iselect = ipoints
    
        def onpick(event):
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
        caxs = ax.plot(self.df[cols].values, 'o-', picker=5)
        ax.legend(cols)
        ax.set_ylim([vmin, vmax])
        ax.set_xlabel('Measurements')
        if coil[-5:] == '_inph':
            ax.set_ylabel('Inphase [ppt]')
        else:
            ax.set_ylabel('Apparent Conductivity [mS/m]')
        for cax in caxs:
            cax.figure.canvas.mpl_connect('pick_event', onpick)        
        killed, = ax.plot([],[],'rx')
        x = xpos
        y = self.df[cols].values
        ipoints = np.zeros(len(x), dtype=bool)


    def dropSelected(self):
        self.df = self.df[~self.iselect]
        print('{:d} points removed'.format(np.sum(self.iselect)))        


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
#            nx = 100
#            ny = 100
#            X, Y = np.meshgrid(np.linspace(np.min(x), np.max(x), nx),
#                           np.linspace(np.min(y), np.max(y), ny))
#            inside = np.ones(nx*ny)
#            inside2 = clipConvexHull(x, y, X.flatten(), Y.flatten(), inside)
#            ie = ~np.isnan(inside2)
#            Z = idw(X.flatten(), Y.flatten(), x, y, val)
##            z = griddata(np.c_[x, y], values, (X, Y), method=method)
#            Z[~ie] = np.nan
#            Z = Z.reshape(X.shape)
#            cax = ax.contourf(X, Y, Z, levels=levels)
#            
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
            fig.colorbar(cax, ax=ax, label='Apparent Conductivity [mS/m]')
        

    def saveMap(self, fname, coil=None, nx=100, ny=100, method='nearest',
                xmin=None, xmax=None, ymin=None, ymax=None, color=False,
                cmap='viridis', vmin=None, vmax=None):
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
        """
        import rasterio
        from rasterio.transform import from_origin
        
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
            for i in range(4):
                Z[np.fliplr(ie.T).T, i] = np.nan
        
            with rasterio.open(fname, 'w',
                           driver='GTiff',
                           height=Z.shape[0],
                           width=Z.shape[1], count=4, dtype=Z.dtype,
                           crs='+init=epsg:27700', transform=tt) as dst:
                for i in range(4):
                    dst.write(Z[:,:,i], i+1)
        else:
            with rasterio.open(fname, 'w',
                               driver='GTiff',
                               height=Z.shape[0],
                               width=Z.shape[1], count=1, dtype=Z.dtype,
                               crs='+init=epsg:27700', transform=tt) as dst:
                dst.write(Z, 1)
                

    def pointsKiller(self):
        """Interactively kill points. Then save df after that.
        """
        pass
        
        geom = True
        resist = self.df['resist'].values
        clabel = 'Apparent Resistivity [$\Omega.m$]'
        if label == '':
            label = clabel
        inan = np.isnan(resist)
        resist = resist.copy()[~inan]
        array = array.copy()[~inan]
        self.iselect = np.zeros(len(inan), dtype=bool)
        
        def setSelect(ie, boolVal):
            ipoints[ie] = boolVal
            self.iselect[~inan] = ipoints
        spacing = np.mean(np.diff(self.elec[:,0]))
        nelec = np.max(array)
        elecpos = np.arange(0, spacing*nelec, spacing)
        
        self.eselect = np.zeros(len(elecpos), dtype=bool)
        
        if geom: # compute and applied geometric factor
            apos = elecpos[array[:,0]-1]
            bpos = elecpos[array[:,1]-1]
            mpos = elecpos[array[:,2]-1]
            npos = elecpos[array[:,3]-1]
            AM = np.abs(apos-mpos)
            BM = np.abs(bpos-mpos)
            AN = np.abs(apos-npos)
            BN = np.abs(bpos-npos)
            K = 2*np.pi/((1/AM)-(1/BM)-(1/AN)+(1/BN)) # geometric factor
            resist = resist*K
            
        if log:
            resist = np.sign(resist)*np.log10(np.abs(resist))
        
        array = np.sort(array, axis=1) # need to sort the array to make good wenner pseudo section
        cmiddle = np.min([elecpos[array[:,0]-1], elecpos[array[:,1]-1]], axis=0) \
            + np.abs(elecpos[array[:,0]-1]-elecpos[array[:,1]-1])/2
        pmiddle = np.min([elecpos[array[:,2]-1], elecpos[array[:,3]-1]], axis=0) \
            + np.abs(elecpos[array[:,2]-1]-elecpos[array[:,3]-1])/2
        xpos = np.min([cmiddle, pmiddle], axis=0) + np.abs(cmiddle-pmiddle)/2
        ypos = - np.sqrt(2)/2*np.abs(cmiddle-pmiddle)
        
        
        def onpick(event):
            if lines[event.artist] == 'data':
                xid, yid = xpos[event.ind[0]], ypos[event.ind[0]]
                isame = (xpos == xid) & (ypos == yid)
                if (ipoints[isame] == True).all():
                    setSelect(isame, False)
                else:
                    setSelect(isame, True)
            
            if lines[event.artist] == 'elec':
                ie = (array == (event.ind[0]+1)).any(-1)
                if all(ipoints[ie] == True):
                    setSelect(ie, False)
                else:
                    setSelect(ie, True)
                if self.eselect[event.ind[0]] == True:
                    self.eselect[event.ind[0]] = False
                else:
                    self.eselect[event.ind[0]] = True
                elecKilled.set_xdata(elecpos[self.eselect])
                elecKilled.set_ydata(np.zeros(len(elecpos))[self.eselect])
            killed.set_xdata(x[ipoints])
            killed.set_ydata(y[ipoints])
            killed.figure.canvas.draw()                
                
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        caxElec, = ax.plot(elecpos, np.zeros(len(elecpos)), 'ko', picker=5)
        cax = ax.scatter(xpos, ypos, c=resist, marker='o', picker=5, vmin=vmin,
                         vmax=vmax)
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label(label)
        cax.figure.canvas.mpl_connect('pick_event', onpick)
        
        killed, = cax.axes.plot([],[],'rx')
        elecKilled, = cax.axes.plot([],[],'rx')
        x = cax.get_offsets()[:,0]
        y = cax.get_offsets()[:,1]
        
        ipoints = np.zeros(len(y),dtype=bool)

        lines = {cax:'data',caxElec:'elec',killed:'killed'}
          
    
    
    
    
    def gridData(self, nx=100, ny=100, method='nearest', xmin=None, xmax=None,
                 ymin=None, ymax=None):
        """ Grid data (for 3D).
        
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
        x = self.df['x'].values
        y = self.df['y'].values
        if xmin is None:
            xmin = np.min(x)
        if xmax is None:
            xmax = np.max(x)
        if ymin is None:
            ymin = np.min(y)
        if ymax is None:
            ymax = np.max(y)
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
        for coil in self.coils:
            values = self.df[coil].values
            if method == 'idw':
                z = idw(X.flatten(), Y.flatten(), x, y, values)
            else:
                z = griddata(np.c_[x, y], values, (X, Y), method=method)
            df[coil] = z.flatten()
        self.dfg = df[ie]
        # TODO add OK kriging ?
        
    
    def crossOverPoints(self, coil=None, ax=None, dump=print):
        """ Build an error model based on the cross-over points.
        
        Parameters
        ----------
        coil : str, optional
            Name of the coil.
        ax : Matplotlib.Axes, optional
            Matplotlib axis on which the plot is plotted against if specified.
        dump : function, optional
            Output function for information.
        """
        if coil is None:
            coil = self.coils[0]
        df = self.df
        dist = cdist(df[['x', 'y']].values,
                     df[['x', 'y']].values)
        minDist = 1 # points at less than 1 m from each other are identical
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
        errorBinned, binEdges, _ = binned_statistic(
                means, error, 'mean', bins=20)
        meansBinned = binEdges[:-1] + np.diff(binEdges)

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
     
    
    
    def plotCrossOverMap(self, coil=None, ax=None):
        """Plot the map of the cross-over points for error model.
        
        Parameters
        ----------
        coil : str, optional
            Name of the coil.
        ax : Matplotlib.Axes, optional
            Matplotlib axis on which the plot is plotted against if specified.
        """
        if coil is None:
            coil = self.coils[0]
        df = self.df
        dist = cdist(df[['x', 'y']].values,
                     df[['x', 'y']].values)
        minDist = 1 # points at less than 1 m from each other are identical
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
        """ Apply the correction due to the 1m calibration.
        """
        pass
        # TODO
        
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
            # applying correction for GF instruments
            print('Transformation to LIN ECa for {:s} at F-{:.0f}m calibration'.format(device, hx))
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
                    qvalues = 0+df[coil].values/gfcoefs[coil]*1e-3j
                    df.loc[:, coil] = Q2eca(qvalues, self.cspacing[i], f=self.freqs[i])*1000 # mS/m
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
                    qvalues = 0+df[coil].values/gfcoefs[coil]*1e-3j # in part per thousand
                    df.loc[:, coil] = Q2eca(qvalues, self.cspacing[i], f=self.freqs[i])*1000
#                coefs = np.zeros(len(coils))
#                for i in range(len(coils)):
#                    coefs[i] = emSens(np.array([1]), self.cspacing[i], self.cpos[i], hx=0)[0]
#                df.loc[:,coils] = df.loc[:, coils]*coefs
            self.df = df
            self.sensor = device
        
        
    ### jamyd91 contribution ### 
    def consPtStat(self):# work out distance between consective points 
        """compute geometrical statistics of consective points, azimuth and 
        bearing of walking direction, time between consective measurements and
        distance between consecutive measurements. Results appended to self.df. 
        
        Notes
        ---------
        Requires local coordinates system to be assigned first! 
        """
        df = self.df
        try:
            x = df.x.values # x values of data frame
            y = df.y.values # y values of data frame 
        except AttributeError:
            raise KeyError(" %s \n ... It looks like no local coordinate system has been assigned, try running self.convertFromNMEA() first")
        dist = np.zeros_like(x,dtype=float) # allocate array to store distance between points 
        bearing = np.zeros_like(x,dtype=float) # allocate array to store survey bearing 
        azimuth = np.zeros_like(x,dtype=float) # allocate array to store survey bearing 
        dist[0] = 999 # first value cant have a distance 
        #time handling 
        timeStr = df.Time.values # string arguments describing time 
        timeLs = timeStr[0].split(':') # parse the time string
        second = float(timeLs[2]) # seconds
        micro = (second - int(second))*100000 # parse the micro second argument 
        times = [time(hour = int(timeLs[0]), #make python datetime.time class
                           minute = int(timeLs[1]), 
                           second = int(second),
                           microsecond=int(micro)
                           )]*len(x)
        current_date = datetime.now()
        current_year = current_date.year
        current_month = current_date.month
        current_day = current_date.day
        dttimes = [datetime(year=current_year,
                          month= current_month,
                          day=current_day,
                          hour = int(timeLs[0]), 
                          minute = int(timeLs[1]), 
                          second = int(second), 
                          microsecond=int(micro))]*len(x)
            
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
                    
        #go through each datframe entry and work out relevant stats 
        for i in range(1,len(x)):
            dx = x[i] - x[i-1] # delta x 
            dy = y[i] - y[i-1] # delta y 
            h = np.sqrt(dx**2 + dy**2) # hypoternues length 
            dist[i] = h
            
            #bearing stat
            quad, edge = quadrantCheck(dx,dy)
            if edge:
                az = quad
            else:
                angle = np.rad2deg(np.arcsin(dx/h)) # angle of measurement direction relative to north 
                if quad == 1:
                    az = angle
                elif quad == 2:
                    az = 180 - angle 
                elif quad == 3:
                    az = 180 + abs(angle)
                elif quad == 4:
                    az = 360 - abs(angle) 
            
            azimuth[i] = az
            if az > 180: #if over 180
                bearing[i] = az - 180 # add 180 in order to get a postive bearing or strike 
            else:
                bearing[i] = az
            
            #time related stats
            timeLs = timeStr[i].split(':') # parse the time
            second = float(timeLs[2])
            micro = (second - int(second))*100000
            times[i] = time(hour = int(timeLs[0]), 
                             minute = int(timeLs[1]), 
                             second = int(second), 
                             microsecond=int(micro))
            dttimes[i] = datetime(year=current_year,#datetimes needed to get time delta objects 
                                  month= current_month,
                                  day=current_day,
                                  hour = int(timeLs[0]), 
                                  minute = int(timeLs[1]), 
                                  second = int(second), 
                                  microsecond=int(micro))
        df['conDist'] = dist # distance between consecutive points 
        df['azimuth'] = azimuth # walking direction in terms of azimuth relative to local coordinate system
        df['bearing'] = bearing # walking direction in terms of bearing relative to local coordinate system
        df['PythonTime'] = times # times in  python datetime format 
        df['elasped(sec)'] = [(dttimes[i] - dttimes[0]).seconds for i in range(len(x))]# number of seconds elasped 
            
        #return df 
        self.df = df
    
    def rmRepeatPt(self, tolerance=0.2):
        """Remove points taken too close together consecutively.
        
        Parameters
        ----------
        tolerance : float, optional
            Minimum distance away previous point in order to be retained. 
        
        Returns 
        -------
        self.fil_df : pandas dataframe
            Truncated dataframe leaving only the measurements which are spaced 
            more than [tolerance value] apart. 
        """
        #error checking
        if not isinstance(tolerance,int) and not isinstance(tolerance,float):
            raise ValueError("tolerance instance should be int or float")
        df = self.df
        try:
            vals = df.conDist
        except AttributeError as e:
            raise KeyError("Error: %s \n... Try running self.consPtStat first"%e)
        ioi = vals>tolerance
        out = df[ioi].copy()
        #return out.reset_index() 
        self.fil_df = out.reset_index()
        
    
    def rmBearing(self, phiMin, phiMax):
        """Remove measurments recorded in a certian bearing range. Where phiMax -
        phiMin is the bearing range to remove. 
        
        Parameters
        ----------
        phiMin : float, optional
            Minimum angle, in degrees. 
        phiMax : float, optional
            Maximum angle, in degrees.
        
        Returns 
        -------
        self.fil_df : pandas dataframe
            Truncated dataframe leaving only the measurements from outside the 
            given bearing range. 
        """
        #error checking
        if not isinstance(phiMin,int) and not isinstance(phiMin,float):
            raise ValueError("phiMin instance should be int or float")
        if not isinstance(phiMax,int) and not isinstance(phiMax,float):
            raise ValueError("phiMax instance should be int or float")
        if phiMin >= phiMax:
            raise ValueError("Min and max bearings cannot be the same, and min must be smaller!")
        df = self.df
        try:
            vals = df.bearing
        except AttributeError as e:
            raise KeyError("Error: %s \n... Try running self.consPtStat first"%e)
        
        ioi= [True]*len(vals)
        for i in range(len(vals)):
            if vals[i] > phiMin and vals[i] < phiMax:
                ioi[i] = False
        
        out = df[ioi].copy()
        #return out.reset_index() 
        self.fil_df = out.reset_index()# its necassary to reset the indexes for other filtering techniques 
        
        
    def driftStn(self, xStn=None, yStn=None, tolerance=0.5):
        """Extract values taken at a given x y point. By default the drift 
        station is taken at the location where the survey starts.
        
        Parameters
        ----------
        xStn : float, optional
            X coordinate of drift station (using local coordinate system, 
            not long and lat)
        yStn : float, optional
            Y coordinate of drift station (using local coordinate system, 
            not long and lat)
        tolerance : float, optional
            Maximum distance away from the drift station using local units. 
        
        Returns
        -------
        self.drift_df : pandas.DataFrame
            Truncated dataframe leaving only the measurements from the drift
            station. 
        """
        df = self.df
        x = df.x.values # x values of data frame
        y = df.y.values # y values of data frame 
        if xStn is None or yStn is None:
            xStn = x[0] # take first xy measurement as drift station
            yStn = y[0]
        
        dx = x - xStn # vector calculation
        dy = y - yStn
        dist = np.sqrt(dx**2 + dy**2) 
        ioi = dist < tolerance # index of interest 

        self.drift_df = df[ioi].copy() #return df[ioi].copy()
        
        
    def plotDrift(self, coil=None, ax=None, fit=True):
        """ Plot drift through time.
        
        Parameters
        ----------
        coil : str, optional
            Coil for which to plot the drift.
        ax : matplotlib.Axes, optional
            If specified, the graph will be plotted against.
        fit : boolean, optional
            If `True` a relatinship will be fitted.
        """
        if ax is None:
            fig, ax = plt.subplots()
        try:
            df = self.drift_df.copy()
        except AttributeError:
            self.driftStn()
            df = self.drift_df.copy()
        if coil is None:
            coil = 'Cond.1 [mS/m]'
        vals = df[coil].values
        ax.scatter(df['PythonTime'].values,vals)
        ax.set_xlabel('Time')
        ax.set_ylabel('Conductivity (mS/m)')
        if fit:
            self.fitDrift(coil=coil)
            seconds = self.drift_df['elasped(sec)'].values
            times = self.drift_df['PythonTime'].values
            cond_mdl = np.polyval(self.drift_mdl,seconds)
            ax.plot(times,cond_mdl)
        
        
    def fitDrift(self, coil=None, order=1):
        """Fit a polynomial model to the drift.
        
        Parameters
        ----------
        coil : str, optional
            Coil for which to plot the drift.
        order : int, optional
            Order of the polyfit. Default is 1.
        """
        seconds = self.drift_df['elasped(sec)'].values
        cond = self.drift_df[coil].values
        mdl = np.polyfit(seconds,cond,order)
        self.drift_mdl = mdl
        
        
    def applyDriftCorrection(self, coil=None):
        """Apply a drift correction to the coil values.
        
        Parameters
        ----------
        coil : str, optional
            Coil for which to plot the drift.
        """
        try:
            mdl = self.drift_mdl
        except AttributeError:
            self.fitDrift(coil=coil)
            mdl = self.drift_mdl
        mdl[-1] = 0 # in this case the c value should be 0 so that the model is normalised to zero
        df = self.df
        vals = df[coil].values
        correction = np.polyval(mdl,df['elasped(sec)'].values)
        self.df[coil] = vals - correction
        
        
#    def aoi(self, polyX, polyY):
#        """Identify area of interest inside a polygon.
#        
#        Parameters
#        ----------
#        polyX : ???
#        
#        polyY : ???
#        """
#        df = self.df
#        try:
#            x = df.x.values # x values of data frame
#            y = df.y.values # y values of data frame
#        except AttributeError:
#            raise KeyError(" %s \n ... It looks like no local coordinate system has been assigned, try running self.convertFromNMEA() first")        
#        inside = iip.isinpolygon(x,y,[polyX,polyY]) # dependencies not satisfied
#        self.df['AOI'] = inside


#%% test
        
if __name__ == '__main__':
#    s = Survey('test/coverCrop.csv')
    #s.show(coils='HCP0.32')
#    s.showMap(contour=True, vmax=40, pts=True)
    
#    s = Survey('test/trimpLo.csv')
#    s.convertFromNMEA()
#    s.show(s.coils[1])
#    s.keepBetween(-5,11)
#    s.rollingMean(10)
#    s.show()
    
#    s.convertFromNMEA()
#    s.showMap(contour=True, pts=True)
#    s.crossOverPoints(s.coils[1])
#    s.gridData(method='idw')
#    s.showMap(s.coils[1])
#    s.df = s.dfg
#    s.showMap(s.coils[1], contour=True)



#%%
    s = Survey('emagpy/test/coverCrop.csv')
#    s.importGF('test/potatoesLo.dat', 'test/potatoesHi.dat', hx=0)
#    s.importGF('test/trimpLo.dat', 'test/trimpHi.dat', hx=1, device='CMD Explorer')
#    s.readFile('test/potatoesLo.csv')
#    s.convertFromNMEA()
#    s.consPtStat() # bearing
#    s.rmRepeatPt() # 
#    print(s.df.head())
#    s.showMap()
#    s.saveMap('test/potatoes.tiff', nx=100, ny=300)
    s.show(s.coils[0])