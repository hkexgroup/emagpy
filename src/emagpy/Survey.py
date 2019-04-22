#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 20:27:12 2019

@author: jkl
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
from scipy.spatial.distance import cdist

#%%

class Survey(object):
    ''' Create a Survey object containing the raw EMI data.
    Raw EMI file is
    '''
    def __init__(self, fname=None):
        self.df = None # main dataframe
        self.freq = None # how to handle different frequencies ?
        self.errorModel = None # a function that returns an error
        self.sensor = None # sensor name (+ company)
        self.coils = None # columns with the coils names and configuration
        if fname is not None:
            self.readFile(fname)
    
    def readFile(self, fname, sensor=None):
        df = pd.read_csv(fname)
        coils = []
        for c in df.columns:
            orientation = c[:3]
            if ((orientation == 'VCP') | (orientation == 'VMD') | (orientation == 'PRP') |
                    (orientation == 'HCP') | (orientation == 'HMD')):
                # replace all orientation in HCP/VCP/PRP mode
                if orientation == 'HMD':
                    df = df.rename(columns={c:c.replace('HMD','VCP')})
                if orientation == 'VMD':
                    df = df.rename(columns={c:c.replace('VMD','HCP')})
                coils.append(c)
        self.coils = coils
        self.freqs = df[coils].values[0,:] # first row for frequency (Hz)
        self.hx = df[coils].values[1,:] # second for height (m)
        self.df = df[2:]
        self.sensor = sensor
        # TODO if inphase is present we can parse it _inph or _quad
        
        
        
    def show(self, coils='all', attr='ECa', ax=None):
        ''' Show the data.
        '''
        if coils == 'all':
            cols = self.coils
        else:
            cols = coils
        
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
       
        ax.plot(self.df[cols].values, 'o-')
        ax.legend(cols)
        ax.set_xlabel('Measurements')
        ax.set_ylabel('Apparent Conductivity [mS/m]')
        


s = Survey('test/testFile1.csv')
s.show()


    #%%
    def convertFromNMEA(self, targetProjection='EPSG:27700'): # British Grid 1936
        ''' Convert NMEA string to selected CRS projection.
        '''
'''
        def func(arg):
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
gps2deg = np.vectorize(func)


wgs84=pyproj.Proj("+init=EPSG:4326") # LatLon with WGS84 datum used by GPS units and Google Earth
osgb36=pyproj.Proj("+init=EPSG:27700") # UK Ordnance Survey, 1936 datum

df['easting'], df['northing'] = pyproj.transform(wgs84, osgb36, 
                                      df['lon'].values, df['lat'].values)

'''


    
    def showMap(self):
        ''' Display a map of the measurements.
        '''
'''
    # clip with convexHull of data
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
    from scipy.spatial import ConvexHull
    _, idx = np.unique(xdata+ydata, return_index=1)
    q = ConvexHull(knownPoints[idx,:])
    from scipy.spatial import Delaunay
    hull = Delaunay(q.points[q.vertices, :])
#    qq = q.points[q.vertices,:]
        
    znew = np.copy(z)
    for i in range(0, len(z)):
        if hull.find_simplex(newPoints[i,:])<0:
            znew[i] = np.nan
    return znew
'''
    
    
    def gridData(self):
        ''' TODO
        '''
    
    
    def crossOverPoints(self):
        ''' Build an error model based on the cross-over points.
        '''
'''       
        modes = ['lo', 'hi']
for i, couple in enumerate(dfs[:1]):
    for k, df in enumerate(couple[:1]):
        dist = cdist(df[['northing', 'easting']].values,
                     df[['northing', 'easting']].values)
        minDist = 1 # points at less than 1 m from each other are identical
        ix, iy = np.where(((dist < minDist) & (dist > 0))) # 0 == same point
        ifar = (ix - iy) > 200 # they should be at least 40 measuremens apart
        ix, iy = ix[ifar], iy[ifar]
        print('found', len(ix), '/', df.shape[0], 'crossing points')
        
        # plot cross-over points
        xcoord = df['easting'].values
        ycoord = df['northing'].values
        icross = np.unique(np.r_[ix, iy])
        fig, ax = plt.subplots()
        ax.set_title(dates[i] + ' ' + modes[k])
        ax.plot(xcoord, ycoord, '.')
        ax.plot(xcoord[icross], ycoord[icross], 'ro', label='crossing points')
        ax.set_xlabel('Easing [m]')
        ax.set_ylabel('Northing [m]')
        fig.tight_layout()
        fig.savefig(outputdir + 'crossOverPoints-' + dates[i] + modes[k] + '.png')
#        fig.show()

'''

    
    def gfCorrection(self):
        ''' Apply the correction due to the 1m calibration.
        '''
        


# test
'''
import a well formatted text file
import two field dataset after selecting the instrument
join lo and hi based on measurements numbers or regridding data

'''

#%%
