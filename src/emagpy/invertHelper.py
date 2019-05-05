#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 07:52:50 2019

@author: jkl
"""

import os
import numpy as np
import matplotlib.pyplot as plt
#from multiprocessing import Pool, Manager
from scipy.constants import mu_0
from hankel import HankelTransform
from scipy import special, integrate
from scipy.optimize import minimize, curve_fit, newton, brent


# useful functions for Hankel transform

# load this important variables once (to speed up) -> do not overwrite them !
dirname = os.path.basename(__file__)
hankel_w0 = np.loadtxt(os.path.join(dirname, 'j1_140.txt))
hankel_w1 = np.loadtxt('j0_120.txt')
hankel5_w0 = np.loadtxt('hankelwts0.txt')
hankel5_w1 = np.loadtxt('hankelwts1.txt')
hankel5_lamb = np.loadtxt('hankelpts.txt')


def func_hankel(typ, K, r):
    ''' Hankel transform based on Guptasarma et al.
    '''
    if typ == 0:
        a = -8.3885
        s = 0.0904
        i = np.arange(0,120).reshape((1,120))
#        w = np.loadtxt('j0_120.txt')
        w = hankel_w0
    elif typ == 1:
        a = -7.9100
        s = 0.0880
        i = np.arange(0,140).reshape((1,140))
#        w = np.loadtxt('j1_140.txt')
        w = hankel_w1
    lamb = 1/r*10**(a+(i-1)*s) # lambda here is not the frequency just your integral variable
#    K0 = np.exp(-lamb0*(z+htx)+getRTE(lamb0)*np.exp(lamb0*(z-htx)))*lamb0    
    hankel = np.sum(K(lamb)*w)/r
    return hankel


def func_hankel2(nu, f, s, N=140, h=0.03):
    """ This one is from hankel python package and allow to
    adjust N and h.
    """
    ht = HankelTransform(nu, N, h) # for HCP h=0.2 works well
    Fk = ht.transform(f, k=s)
    return Fk[0]


def func_hankel3(nu, f, s):
    """ Manual integration using scipy integrate
    """
    def integrand(x):
        return f(x)*special.jv(nu, s*x)*x
    return integrate.quad(integrand, 0, 100)


def func_hankel5(typ, K, r): # prefered one as the step is not linear >< hankel
    ''' we need this type of hankel transform because we can input r (Tx-Rx)
    coil spacing. We can't do that with the python hankel package.
    based on Anderson Fortran code 1979 Geophysic 44(7): 1287-1305
    '''
    if typ == 0:
#        w = np.loadtxt('hankelwts0.txt')
        w = hankel5_w0
    elif typ == 1:
#        w = np.loadtxt('hankelwts1.txt')
        w = hankel5_w1
#    lamb = np.loadtxt('hankelpts.txt') # lambda here is not the frequency just your integral variable
    lamb = hankel5_lamb
    hankel = np.sum(w*K(lamb/r)/r)
    return hankel

""" NOTE
func_hankel and func_hankel5 do not contains the multiplication by lambda in
in their kernel function. This has to be added after.
"""



# useful functions for full Maxwell

def getR0_1(lamb, sigg, f, d):
    """ Compute reflexion coefficient based on Deidda 2018.
    
    Parameters
    ----------
    lamb : numpy.array
        Radial wave number (integration variable of Hankel transform).
    sigg : numpy.array
        Conductivity of the n layers.
    f : float
        Frequency in Hz of the instrument.
    d : numpy.array
        Height of each layers [m].
    
    Returns
    -------
    R0 : complex
        Reflexion coefficient for layer 0 (air layer).
    """
    sigma = np.r_[0, sigg.copy()] # sigma 0 is above the ground
    d = np.r_[d.copy(), 0] # last one won't be used, just for index
    omega = 2*np.pi*f # assume single omega here as lamb is already a vector
    u = np.sqrt(lamb**2+1j*sigma*mu_0*omega)
    N = u/(1j*mu_0*omega)
    Y = np.zeros(N.shape, dtype=np.complex)*np.nan # Y is actually -1 smaller than N but easier like this for index
    Y[-1] = N[-1]
    for i in range(len(N)-2, 0, -1):
        Y[i] = N[i]*(Y[i+1]+N[i]*np.tanh(d[i]*u[i]))/ \
                    (N[i] + Y[i+1]*np.tanh(d[i]*u[i]))
    R0 = (N[0]-Y[1])/(N[0]+Y[1])
    return R0

#@jit(nopython=True) # cannot guessed the type of R (complex)
def getR0_2(lamb, sigg, f, d):
    """ Compute reflexion coefficient based on vonHebel2014, Lavoué2010.
    
    Parameters
    ----------
    lamb : numpy.array
        Radial wave number (integration variable of Hankel transform).
    sigg : numpy.array
        Conductivity of the n layers.
    f : float
        Frequency in Hz of the instrument.
    d : numpy.array
        Height of each layers [m].
    
    Returns
    -------
    R0 : complex
        Reflexion coefficient for layer 0 (air layer).
    """
    sigma = np.concatenate([[0], sigg]) # sigma 0 is above the ground
    h = np.concatenate([[0], d, [0]]) # last one won't be used, because Rn+1 = 0 and
    # first zero won't be used because we never use hn, always hn+1, just
    # for easier indexing
    gamma = np.sqrt(lamb**2 + 1j*2*np.pi*f*mu_0*sigma)
    R = np.zeros((len(sigma)), dtype=np.complex)#*np.nan
#    R[-1] = 0 # no waves from the lower half space
    for i in range(len(sigma)-2, -1, -1):
        num = (gamma[i]-gamma[i+1])/(gamma[i]+gamma[i+1]) \
            + R[i+1]*np.exp(-2*gamma[i+1]*h[i+1])
        den = 1+(gamma[i]-gamma[i+1])/(gamma[i]+gamma[i+1]) \
            * R[i+1]*np.exp(-2*gamma[i+1]*h[i+1])
        R[i] = num/den
    return R[0]

#%timeit getR0_2(1, np.array([60,70,60]), 30000, np.array([0.3, 0.5]))
# 53 us

#@jit
#def foo(sigg, d, lamb):
#    sigma = np.array([0] + sigg)
#    h = np.array([0] + d + [0])
#    gamma = np.sqrt(lamb**2 + 1j*2*np.pi*f*mu_0*sigma)
#    return gamma
#
#%timeit foo([10, 20], [2], 0.3)
##foo.inspect_types()


# test
#%timeit print(getR0_1(1, [60,70,60], 30000, [0.3, 0.5])) # very good exactly the same !
#print(getR0_2(1, np.array([60,70,60]), 30000, np.array([0.3, 0.5])))
# if homogenous, only R0 is != 0, all other are = 0 because there
# is no reflexion if the layers have the same conductivity.

#getRn = np.vectorize(getR0_1)
#getRn = np.vectorize(getR0_2)

def getRn(lamb, sig, f, h, getR0=getR0_2):
    return np.array([getR0(l, sig, f, h) for l in lamb.flatten()])

def getQhomogeneous(cpos, s, sig, f):
    """ Compute quadrature ratio based on analytical solution of McNeil1980
    for homogeneous half-space only.
    """
    omega = 2*np.pi*f
    g = np.sqrt(1j*omega*mu_0*sig) # for gamma
    if cpos == 'hcp':
        Q = 2/(g*s)**2*(9-(9+9*g*s+4*(g*s)**2+(g*s)**3)*np.exp(-g*s))
    elif cpos == 'vcp':
        Q = 2*(1-3/(g*s)**2+(3+3*g*s+(g*s)**2)*np.exp(-g*s)/(g*s)**2)
    return Q
    
    
def getQ(cpos, s, sig, f, h, typ=5):
    """ Compute the quadrature ratio.
    """
    if cpos == 'vcp':
        if typ == 1|5:
            def func(lamb):
                return getRn(lamb, sig, f, h)*lamb
        else:
            def func(lamb):
                return getRn(lamb, sig, f, h)
    elif cpos == 'hcp':
        if typ == 1|5:
            def func(lamb):
                return getRn(lamb, sig, f, h)*lamb**2
        else:
            def func(lamb):
                return getRn(lamb, sig, f, h)*lamb
    dicoHankel = {1:func_hankel,
                2:func_hankel2,
                3:func_hankel3,
                5:func_hankel5}
    dicoBessel = {'hcp':0,
                  'vcp':1}
    dicoCoefs = {'hcp': s**3,
                 'vcp': s**2}
    return 1-dicoCoefs[cpos]*dicoHankel[typ](dicoBessel[cpos], func, s)


def getLIN(sigma, s, f=30000):
    """ Get Low Induction Number. It should be << 1.
    """
    skinDepth = np.sqrt(2/(2*np.pi*f*mu_0*sigma))
    return s/skinDepth


def Q2eca(Q, s, f=30000):
    """ Returns apparent conductivity given the quadrature value (assuming we
    are in the Low Induction Number (LIN) approximation)
    """
    return 4/(2*np.pi*f*mu_0*s**2)*np.imag(Q)


# test
#sig = np.array([60,60,60])*1e-3
#f = 30000
#h = [0.3, 0.5]
#s = 0.32
#cpos = 'hcp'
#def func(lamb):
#    return getRn(lamb, sig, f, h)*lamb**2
#app = -4*s/(2*np.pi*30000*mu_0)*np.imag(func_hankel5(0, func, s))
#print(app*1e3)
#print(Q2eca(getQ(cpos, s, sig, f, h), s, f)*1e3) # pk


# test homogeneous
#sigH = 60*1e-3 # S/m
#sig = np.ones(3)*sigH
#f = 30000
#h = [0.3, 0.5]
#s = 0.71
#cpos = 'hcp'
#print(getQ(cpos, s, sig, f, h))
#print(getQhomogeneous(cpos, s, sigH, f)) # not quite...
#print(Q2eca(getQ(cpos, s, sig, f, h), s, f)*1e3)
#print(Q2eca(getQhomogeneous(cpos, s, sigH, f), s, f)*1e3) # not quite...
#



# --------------------- getQ2 faster
def getRn2(lamb, sigg, f, d): # compute reflexion coefficients
    sigma = np.array([0] + sigg.tolist()) # sigma 0 is above the ground
    h = np.array([0] + d.tolist() + [0]) # last one won't be used, because Rn+1 = 0 and
    # first zero won't be used because we never use hn, always hn+1, just
    # for easier indexing
    sigmaLen = len(sigma)
    realPart = np.repeat(lamb[:,None], sigmaLen, axis=1)**2
    imagPart = 2*np.pi*f*mu_0*np.repeat(sigma[:,None], len(lamb), axis=1)
    gamma = np.sqrt(realPart.T + 1j*imagPart)
#    gamma = np.sqrt(lamb**2 + 1j*2*np.pi*f*mu_0*sigma)
    R = np.zeros((sigmaLen, len(lamb)), dtype=np.complex)#*np.nan
    R[-1,:] = 0+0j # no waves from the lower half space
    for i in range(sigmaLen-2, -1, -1): # it's recursive so needs for loop
        R[i,:] = (gamma[i,:]-gamma[i+1,:])/(gamma[i,:]+gamma[i+1,:]) \
            + R[i+1,:]*np.exp(-2*gamma[i+1,:]*h[i+1])
        R[i,:] /= 1+(gamma[i,:]-gamma[i+1,:])/(gamma[i,:]+gamma[i+1,:]) \
            * R[i+1,:]*np.exp(-2*gamma[i+1,:]*h[i+1])

    return R[0,:]

# test code
#getR0_2(1, np.array([60,70,60]), 30000, np.array([0.3, 0.5]))
#lamb = np.array([1])
#getRn2(lamb, np.array([60,70,60]), 30000, np.array([0.3, 0.5]))


def getQ2(cpos, s, sig, f, h, typ=None):
    lamb = hankel5_lamb/s # normalized to be used in Hankel
    if cpos == 'vcp':
        w = hankel5_w1 # for Bessel function of order 1
        K = getRn2(lamb, sig, f, h)*lamb # kernel with reflexion coef
        hankel = np.sum(w*K/s) # hankel transform
        return 1-s**2*hankel
    elif cpos == 'hcp':
        w = hankel5_w0 # for Bessel function of order 0
        K = getRn2(lamb, sig, f, h)*lamb**2 # kernel with reflexion coef
        hankel = np.sum(w*K/s) # hankel transform
        return 1-s**3*hankel
    elif cpos == 'prp': # see Anderson for correct formula
        w = hankel5_w1 # for Bessel function of order 0
        K = getRn2(lamb, sig, f, h)*lamb**2 # kernel with reflexion coef
        hankel = np.sum(w*K/s) # hankel transform
        return 0-s**3*hankel
  

#sigH = 60*1e-3 # S/m
#sig = np.ones(3)*sigH
#f = 30000
#h = np.array([0.3, 0.5])
#s = 0.71
#cpos = 'hcp'
#lamb = hankel5_lamb/s
#Ka = getRn(lamb, sig, f, h) # identical
#Kb = getRn2(lamb, sig, f, h) # identical
#getQ(cpos, s, sig, f, h) # 47.5 ms
#getQ2(cpos, s, sig, f, h) # 672 us !
#getQhomogeneous('hcp',0.71, 60, 30000)
#print(getQ2('hcp', 0.71, np.array([60, 60]), 30000, np.array([1])))
#print(getQ('hcp', 0.71, np.array([60, 60]), 30000, np.array([1])))


# definition of forward model used
def fMaxwellECa(cond, depths, s, cpos, hx=None, f=30000):
    """ Compute quadrature ratio Q using Maxwell equation but then use the LIN
    to convert it to apparent conductivity.
    """
    if len(cond)-1 != len(depths):
        raise ValueError('len(cond)-1 should be equal to len(depths).')
    if len(s) != len(cpos):
        raise ValueError('len(s) should be equal to len(cpos).')
    h = np.r_[depths[0], np.diff(depths)]
    if hx is not None:
        h = np.r_[hx, h]
        cond = np.r_[0, cond]
    response = np.zeros(len(s))*np.nan
    cond = cond*1e-3 # convert to S/m
    for i in range(len(s)):
        Q = getQ2(cpos[i], s[i], cond, f, h)
        response[i] = Q2eca(Q, s[i], f)*1e3
    return response


def fMaxwellQ(cond, depths, s, cpos, hx=None, f=30000, maxiter=50):
    """ Compute the quadrature ratio using Maxwell equation and then try to
    find a value of conductivity that match the observed Q
    """
    if len(cond)-1 != len(depths):
        raise ValueError('len(cond)-1 should be equal to len(depths).')
    if len(s) != len(cpos):
        raise ValueError('len(s) should be equal to len(cpos).')
#    eca0 = fCS(cond, depths, s, cpos, hx=hx) # to kick start the solver
    h = np.r_[depths[0], np.diff(depths)]
    if hx is not None:
        h = np.r_[hx, h]
        cond = np.r_[0, cond]
    response = np.zeros(len(s))*np.nan
    cond = cond*1e-3 # convert to S/m
    for i in range(len(s)):
        Qobs = np.imag(getQ2(cpos[i], s[i], cond, f, h))
        if cpos[i] == 'prp': # we don't have analytical for PRP
            def objfunc(sig):
                sigg = np.ones(len(cond))*sig
                Qmod = np.imag(getQ2(cpos[i], s[i], sigg, f, h))
                return np.abs(Qmod - Qobs)
        else:
            def objfunc(sig): # analytical is much faster
                Qmod = np.imag(getQhomogeneous(cpos[i], s[i], sig, f)) # faster even if more rounding errors
                return np.abs(Qmod - Qobs)
        sig0 = Q2eca(getQ2(cpos[i], s[i], cond, f, h), s[i], f) # still in S/m
#        sig0 = eca0[i]
#        res = minimize(objfunc, x0=sig0, method='BFGS')
#        zero = res.x[0]*1e3
#        zero = brent(objfunc, brack=(sig0*0.8, sig0, sig0*1.2))*1e3
        zero = newton(objfunc, sig0, maxiter=maxiter)*1e3 # back to mS/m # faster !
        response[i] = zero
    return response

# test
#cpos = 'hcp'
#s = 0.32
#cond = np.array([60,60,60])*1e-3 # [S/m]
#f = 30000
#h = [0.4, 0.7]
#print(getQ(cpos, s, cond, f, h))
#print(getQhomogeneous(cpos, s, cond, f))
#Qobs = np.imag(getQ(cpos, s, cond, f, h))
#def objfunc(sig):
#    sigg = np.ones(len(cond))*sig
#    Qmod = np.imag(getQ(cpos, s, sigg, f, h))
#    return np.abs(Qmod - Qobs)
#sig0 = Q2eca(getQ(cpos, s, cond, f, h), s, f) # still in S/m
#print(sig0)
#zero = newton(objfunc, sig0)*1e3 # back to mS/m
#print(zero)



def emSens(depths,s,coilPosition, hx=0, rescaled=False):
    """return mcNeil senstivity values
    
    depths (positive and increasing with depth)
    s = distance between coils
    coilPosition is "hcp" for horizontal coplanar (HCP) and "vcp" for VCP
    x = range on which compute the sensitivity (array_like)
     """
    depths = depths + np.abs(hx)
    z=np.array(depths)/s
    # from mcNeil 1980 in Callegary2007
    if coilPosition == 'hcp':
        cs = 1/np.sqrt(4*z**2+1)
    if coilPosition == 'vcp':
        cs = np.sqrt(4*z**2+1)-2*z
    if coilPosition == 'prp':
        cs = 1-2*z/np.sqrt(4*z**2 + 1)
    
    if (hx != 0) and rescaled is True: # rescale the cumulative sensitivity so that it reaches 1
        cs = cs/cs[0]

    return cs # all but the first one as it is just use to scale everything

# test code
#depths = np.arange(0, 2, 0.1)
#cs1 = emSens(depths, 0.32, 'hcp', hx=0)
#cs2 = emSens(depths, 0.32, 'hcp', hx=1)
#
#fig, ax = plt.subplots()
#ax.plot(cs1, -depths, '-')
#ax.plot(cs2, -depths, '--')
#ax.plot(cs1, -depths+1, ':')
#fig.show()


def forward1d_full(cond,depths,s,cpos, hx=None, rescaled=False):
    if hx is None:
        hx = np.zeros(len(s))
    if len(cond.shape)>1:
#        print('WARNING : need 1 dimension matrix for forward model')
        cond = cond.flatten()
    if depths[0] != 0:
        print('ERROR: first depth should be zero, will add it for you')
        depths = np.r_[0, depths]
    if len(depths) != len(cond):
        print('ERROR: unmatching depths and conds')
        return
    if np.sum(depths < 0) > 0:
        print('ERROR: depth contains negative depth. Please input only positive depth')
        return
    out = np.zeros(len(s))*np.nan
    for i in range(0,len(out)):
        cs = emSens(depths,s[i], cpos[i], hx[i], rescaled=rescaled)
#        print('CS = ', np.sum(np.diff(cs)))
        appcond = np.sum(cond[:-1]*(cs[:-1]-cs[1:]))
        appcond = appcond + cond[-1]*(cs[-1])
        out[i] = appcond
    return out

# test code
#app1 = forward1d_full(np.array([10,30]),[1],[0.32],['hcp'], hx=[1])
#app2 = forward1d_full(np.array([0,10,30]),[1,2],[0.32],['hcp'], hx=[0])
#print(app1, app2)
    

def fCS(cond, depths, s, cpos, hx=None, rescaled=False):
    """ Use the cumulative sensitivity function of McNeil 1980.
    """
    if len(cond)-1 != len(depths):
        raise ValueError('len(cond)-1 should be equal to len(depths).')
    if len(s) != len(cpos):
        raise ValueError('len(s) should be equal to len(cpos).')
    if depths[0] != 0:
#        print('add 0 for you')
        depths = np.r_[0, depths] # otherwise forward1d_full complains
    if hx is not None:
        hx = np.ones(len(cpos))*hx
    response = forward1d_full(cond, depths, s, cpos, hx=hx, rescaled=rescaled)
    return response


# based on Andrade2018 (general case)
def emSensAndrade(depths,s,coilPosition, hx=0):
    """return RESCALED mcNeil senstivity values (Andrade 2018)
    
    depths (positive and increasing with depth)
    s = distance between coils
    coilPosition is "hcp" for horizontal coplanar (HCP) and "vcp" for VCP
    x = range on which compute the sensitivity (array_like)
     """
    depths = depths
    z=np.array(depths)/s
    if coilPosition == 'hcp':
        cs = ((4*(hx/s)**2+1)**0.5)/(4*(z+hx/s)**2+1)**0.5
    if coilPosition == 'vcp':
        cs = ((4*(z+hx/s)**2+1)**0.5 - 2*(z + hx/s))/((4*(hx/s)**2+1)**0.5 - 2*hx/s)
    return cs

def fCSandrade(cond,depths,s,cpos, hx=None):
    if hx is None:
        hx = 0
    hx = np.ones(len(s))*hx
    if len(cond.shape)>1:
#        print('WARNING : need 1 dimension matrix for forward model')
        cond = cond.flatten()
    if depths[0] != 0:
#        print('ERROR: first depth should be zero, will add it for you')
        depths = np.r_[0, depths]
    if len(depths) != len(cond):
        print('ERROR: unmatching depths and conds')
        return
    if np.sum(depths < 0) > 0:
        print('ERROR: depth contains negative depth. Please input only positive depth')
        return
    out = np.zeros(len(s))*np.nan
    for i in range(0,len(out)):
        cs = emSensAndrade(depths,s[i], cpos[i], hx[i])
        appcond = np.sum(cond[:-1]*(cs[:-1]-cs[1:]))
        appcond = appcond + cond[-1]*(cs[-1])
        out[i] = appcond
    return out


def buildSecondDiff(ndiag):
    x=np.ones(ndiag)
    a=np.diag(x[:-1]*-1,k=-1)+np.diag(x*2,k=0)+np.diag(x[:-1]*-1,k=1)
    a[0,0]=1
    a[-1,-1]=1
    return a


#%% test bench
#sigma = np.array([30, 30, 30, 30]) # layers conductivity [mS/m]
#depths = np.array([0.3, 0.7, 2]) # thickness of each layer (last is infinite)
#f = 30000 # Hz frequency of the coil
#cpos = np.array(['hcp','hcp','hcp','vcp','vcp','vcp']) # coil orientation
##cpos = np.array(['hcp','hcp','hcp','prp','prp','prp']) # coil orientation
##cspacing = np.array([0.32, 0.71, 1.18, 0.32, 0.71, 1.18])
#cspacing = np.array([1.48, 2.82, 4.49, 1.48, 2.82, 4.49]) # explorer
#
#print('fCS:', fCS(sigma, depths, cspacing, cpos))
#print('fMaxwellECa:', fMaxwellECa(sigma, depths, cspacing, cpos))
#print('fMaxwellQ:', fMaxwellQ(sigma, depths, cspacing, cpos))
#print('fCSandrade:', fCSandrade(sigma, depths, cspacing, cpos, hx=0))
#
#print('fCS:', fCS(sigma, depths, cspacing, cpos, hx=1))
#print('fCS:', fCS(sigma, depths, cspacing, cpos, hx=1, rescaled=True))
#print('fCSandrade:', fCSandrade(sigma, depths, cspacing, cpos, hx=1))


#%% usefull functions for the direct inversion
def buildJacobian(depths, s, cpos):
    '''Build Jacobian matrix for Cumulative Sensitivity based inversion.
    
    Parameters
    ----------
    cond : array
        Conductivities of the model layer.
    depths : array
        Depths of the lower layer bound.
    s : array
        Array of coil separation [m].
    cpos : array of str
        Array of coil orientation.
    
    Returns
    -------
    A matrix with each row per coil spacing/orientation and each column per
    model parameters (per layer).
    '''
    depths = np.r_[0, depths]
    jacob = np.zeros((len(s),len(depths)))*np.nan
    for i in range(0,len(s)):      
        cs = emSens(depths, s[i], cpos[i])         
        jacob[i,:-1] = cs[:-1]-cs[1:]
        jacob[i,-1] = cs[-1]
    return jacob

# test
#J = buildJacobian([0.5, 0.7],[0.32, 0.72, 1.14],['vcp','vcp','vcp'])
