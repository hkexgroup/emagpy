#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 01:31:02 2020

@author: jkl
"""
import numpy as np
import matplotlib.pyplot as plt
from emagpy import Problem
k = Problem()
k.createSurvey('examples/cover-crop/coverCropTransect.csv')
k.filterRange(vmax=50)
k.surveys[0].df = k.surveys[0].df[:5]
k.setInit(depths0=np.linspace(0.1, 2, 10))
# k.invert(alpha=100)
k.depths = k.depths0.copy()
k.models = k.conds0.copy()
sens = k.computeSens()

depth = k.depths[0][0,:]
mdepths = np.r_[depth[0]/2, depth[:-1] + np.diff(depth)/2, depth[-1]]
fig, ax = plt.subplots()
ax.plot(sens[0][:,:,0], -mdepths)
fig.show()

k.invert(method='Gauss-Newton')

