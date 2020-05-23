#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 20:25:54 2019

@author: jkl

Main UI
"""
import os
import sys
import time

print('''
===========================================================
######## ##     ##    ###     ######   ########  ##    ## 
##       ###   ###   ## ##   ##    ##  ##     ##  ##  ##  
##       #### ####  ##   ##  ##        ##     ##   ####   
######   ## ### ## ##     ## ##   #### ########     ##    
##       ##     ## ######### ##    ##  ##           ##    
##       ##     ## ##     ## ##    ##  ##           ##    
######## ##     ## ##     ##  ######   ##           ##    
===========================================================
''')

from emagpy import Problem
from emagpy import EMagPy_version
import numpy as np
import pandas as pd
from multiprocessing import freeze_support

from PyQt5.QtWidgets import (QMainWindow, QSplashScreen, QApplication, QPushButton, QWidget,
    QTabWidget, QVBoxLayout, QLabel, QLineEdit, QMessageBox, QCompleter,
    QFileDialog, QCheckBox, QComboBox, QTextEdit, QHBoxLayout,
    QTableWidget, QFormLayout, QTableWidgetItem, QHeaderView, QProgressBar,
    QStackedLayout, QGroupBox)#, QRadioButton, QAction, QListWidget, QShortcut)
from PyQt5.QtGui import QIcon, QPixmap, QIntValidator, QDoubleValidator#, QKeySequence
from PyQt5.QtCore import QThread, pyqtSignal#, QProcess, QSize
from PyQt5.QtCore import Qt
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True) # for high dpi display
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import rcParams
rcParams.update({'font.size': 12}) # CHANGE HERE for graph font size


#%% General crash ERROR
import threading
import traceback

def errorMessage(etype, value, tb):
    print('ERROR begin:')
    traceback.print_exception(etype, value, tb)
    print('ERROR end.')
    errorMsg = traceback.format_exception(etype, value, tb,limit=None, chain=True)
    finalError =''
    for errs in errorMsg:
        finalError += errs
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText("<b>Critical error:</b>")
    msg.setInformativeText('''Please see the detailed error below.<br>You can report the errors at:<p><a href='https://gitlab.com/hkex/emagpy/issues'>https://gitlab.com/hkex/emagpy/issues</a></p><br>''')
    msg.setWindowTitle("Error!")
#    msg.setWindowFlags(Qt.FramelessWindowHint)
    msg.setDetailedText('%s' % (finalError))
    msg.setStandardButtons(QMessageBox.Retry)
    msg.exec_()

def catchErrors():
    sys.excepthook = errorMessage
    threading.Thread.__init__


#%%get relative path of images
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class customThread(QThread):
    ''' class needed to run out of main thread computation and then emit
    signal to the main GUI thread to display message box.
    '''
    signal = pyqtSignal('PyQt_PyObject')

    def __init__(self, func):
        QThread.__init__(self)
        self.func = func

    def run(self):
        output = self.func()
        self.signal.emit(output) # inform the main thread of the output


#%%

# small code to see where are all the directories
frozen = 'not'
if getattr(sys, 'frozen', False):
        # we are running in a bundle
        frozen = 'ever so'
        bundle_dir = sys._MEIPASS
else:
        # we are running in a normal Python environment
        bundle_dir = os.path.dirname(os.path.abspath(__file__))
print( 'we are',frozen,'frozen')
print( 'bundle dir is', bundle_dir )
print( 'sys.argv[0] is', sys.argv[0] )
print( 'sys.executable is', sys.executable )
print( 'os.getcwd is', os.getcwd() )


#%%
class MatplotlibWidget(QWidget):
    ''' Class to put matplotlibgraph into QWidget.
    '''
    def __init__(self, parent=None, navi=True, itight=True, threed=False):
        super(MatplotlibWidget, self).__init__(parent) # we can pass a figure but we can replot on it when
        # pushing on a button (I didn't find a way to do it) while with the axes, you can still clear it and
        # plot again on them
        figure = Figure(tight_layout=itight)
        self.canvas = FigureCanvasQTAgg(figure)
        if threed is True:
            ax = figure.add_subplot(111, projection='3d')
            ax.set_aspect('auto')
        else:
            ax = figure.add_subplot(111)
            ax.set_aspect('auto')
        self.figure = figure
        self.axis = ax

        self.layoutVertical = QVBoxLayout(self)
        self.layoutVertical.addWidget(self.canvas)

        if navi is True:
            self.navi_toolbar = NavigationToolbar(self.canvas, self)
            self.navi_toolbar.setMaximumHeight(30)
            self.layoutVertical.addWidget(self.navi_toolbar)


    def setMinMax(self, vmin=None, vmax=None):
        coll = self.axis.collections[0]
#        print('->', vmin, vmax)
#        print('array ', coll.get_array())
        if vmin is None:
            vmin = np.nanmin(coll.get_array())
        else:
            vmin = float(vmin)
        if vmax is None:
            vmax = np.nanmax(coll.get_array())
        else:
            vmax = float(vmax)
#        print(vmin, vmax)
        coll.set_clim(vmin, vmax)
        self.canvas.draw()


    def plot(self, callback, **kwargs):
        ''' Plot on the canvas, given a function and it's arguments.
        
        Parameters
        ----------
        callback : function
            Function of the API that accept an 'ax' argument.
        **kwargs : keyword arguments
            Key word arguments to be passed to the API function.
        '''
        self.figure.clear() # need to clear the figure with the colorbar as well
        ax = self.figure.add_subplot(111)
        ax.set_aspect('auto')
        callback(ax=ax, **kwargs)
        self.callback = callback
        self.canvas.draw()

    def setCallback(self, callback):
        self.callback = callback

    def replot(self, threed=False, **kwargs):
        self.figure.clear()
        if threed is False:
            ax = self.figure.add_subplot(111)
        else:
            ax = self.figure.add_subplot(111, projection='3d')
        self.axis = ax
        self.callback(ax=ax, **kwargs)
        ax.set_aspect('auto')
        self.canvas.draw()

    def clear(self):
        self.axis.clear()
        self.figure.clear()
        self.canvas.draw()


#%%
class App(QMainWindow):
    ''' Main UI application for emagpy. This call the main emagpy object for 
    all the processing.
    '''
    def __init__(self, parent=None):
        super().__init__()
        
        self.setWindowTitle('EMagPy v{:s}'.format(EMagPy_version))
        self.setGeometry(100,100,1100,600)
        
        if frozen == 'not':
            self.datadir = os.path.join(bundle_dir, './examples')
        else:
            self.datadir = os.path.join(bundle_dir, 'emagpy', 'examples')
        self.fnameHi = None
        self.fnameLo = None
        self.running = False # True when inverison is running
        
        self.errorLabel = QLabel('<i style="color:black">Error messages will be displayed here</i>')
        QApplication.processEvents()

        self.table_widget = QWidget()
        self.layout = QVBoxLayout()
        self.tabs = QTabWidget()
        
        self.problem = Problem()
        self.problem.runningUI = True
        
        
        
        #%% tab 0 forward modelling
        forwardTab = QTabWidget()
        self.tabs.addTab(forwardTab, 'Forward')
        
        def fimportBtnFunc():
            self._dialog = QFileDialog()
            fname, _ = self._dialog.getOpenFileName(importTab, 'Select data file', self.datadir, '*.csv')
            if fname != '':
                try:
                    df = pd.read_csv(fname)
                    ccols = [c for c in df.columns if c[:5] == 'layer']
                    dcols = [c for c in df.columns if c[:5] == 'depth']
                    if len(ccols) != len(dcols) + 1:
                        self.errorDump('Number of depths should be number of layer - 1')
                    conds = df[ccols].values
                    depths = df[dcols].values
                    self.problem.setModels([depths], [conds])
                    self.mwf.replot()
                except Exception as e:
                    print(e)
                    self.errorDump('Error in reading file. Please check format.')
        self.fimportBtn = QPushButton('Import model')
        self.fimportBtn.clicked.connect(fimportBtnFunc)
        self.fimportBtn.setToolTip('File needs to be .csv with layer1, layer2, depth1, ... columns.\n'
                                   'layerX contains EC in mS/m and depthX contains depth of the bottom\n'
                                   'of the layer in meters (positively defined)')
        
        # toolset for generating model
        finstructions = QLabel('Create a synthetic model or import one.')
        fnlayerLabel = QLabel('Number of layers:')
        def updateTable():
            try:
                n = int(self.fnlayer.text())
                if n > 0:
                    self.paramTable.setLayer(n)
            except:
                pass
        self.fnlayer = QLineEdit('2')
        self.fnlayer.setValidator(QIntValidator())
        self.fnlayer.textChanged.connect(updateTable)

        fnsampleLabel = QLabel('Number of samples:')
        self.fnsample = QLineEdit('10')
        self.fnsample.setValidator(QIntValidator())
        
        class ParamTable(QTableWidget):
            def __init__(self, nlayer=2, headers=['Parameter', 'Start', 'End']):
                ncol = len(headers)
                nrow = nlayer + nlayer - 1
                super(ParamTable, self).__init__(nrow, ncol)
                self.nlayer = nlayer
                self.nrow = nrow
                self.ncol = ncol
                self.headers = headers
                self.setHorizontalHeaderLabels(self.headers)
                self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
                self.setLayer(nlayer)
                
            def setLayer(self, nlayer):
                self.nlayer = nlayer
                self.nrow = nlayer + nlayer - 1
                self.clear()
                self.setRowCount(self.nrow)
                for i in range(nlayer):
                    val = (i+1)*5
                    self.setItem(i, 0, QTableWidgetItem('layer{:d}'.format(i+1)))
                    self.item(i, 0).setFlags(Qt.ItemIsEnabled)
                    self.setItem(i, 1, QTableWidgetItem('{:.1f}'.format(val)))
                    self.setItem(i, 2, QTableWidgetItem('{:.1f}'.format(val*2)))
                for i in range(nlayer-1):
                    depth = (i+1)*0.5
                    self.setItem(nlayer+i, 0, QTableWidgetItem('depth{:d}'.format(i+1)))
                    self.item(nlayer+i, 0).setFlags(Qt.ItemIsEnabled)
                    self.setItem(nlayer+i, 1, QTableWidgetItem('{:.1f}'.format(depth)))
                    self.setItem(nlayer+i, 2, QTableWidgetItem('{:.1f}'.format(depth)))
                
            def getTable(self, nsample):
                conds = []
                depths = []
                for i in range(self.nlayer):
                    start = float(self.item(i,1).text())
                    end = float(self.item(i,2).text())
                    conds.append(np.linspace(start, end, nsample))
                for i in range(self.nlayer-1):
                    start = float(self.item(i+self.nlayer,1).text())
                    end = float(self.item(i+self.nlayer,2).text())
                    depth = np.linspace(start, end, nsample)
                    if i > 0:
                        if (depth > depths[i-1]).all():
                            depths.append(depth)
                        else:
                            self.errorDump('Error in depths specification.')
                    else:
                        depths.append(depth)
                        
                return np.vstack(depths).T, np.vstack(conds).T
                
        self.paramTable = ParamTable()
        
        def generateModel():
            depths, conds = self.paramTable.getTable(int(self.fnsample.text()))
            self.problem.setModels([depths], [conds])
            self.mwf.replot()
            
        self.fgenerateBtn = QPushButton('Generate Model')
        self.fgenerateBtn.clicked.connect(generateModel)
        
        
        coilLabel = QLabel('Specify coil orientation (HCP or VCP), coil separation [m],'
                           ' frequency [Hz] (only used in FS solutions) and height above'
                           ' the ground hx [m]:')
        coilLabel.setWordWrap(True)
        
        # coil selection table
        class CoilTable(QTableWidget):
            def __init__(self, nrow=10, headers=['HCP/VCP','coil sep','freq','hx']):
                ncol = len(headers)
                super(CoilTable, self).__init__(nrow, ncol)
                self.nrow = nrow
                self.ncol = ncol
                self.headers = headers
                self.setHorizontalHeaderLabels(self.headers)
                self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
                self.setTable(nrow)
                
            def setTable(self, nrow):
                self.clear()
                self.nrow = nrow
                self.setRowCount(self.nrow)
                a = ['HCP','HCP','HCP','VCP','VCP','VCP']
                b = ['0.32','0.71','1.18','0.32','0.71','1.18']
                for i in range(6):
                    self.setItem(i, 0, QTableWidgetItem(a[i]))
                    self.setItem(i, 1, QTableWidgetItem(b[i]))
                    self.setItem(i, 2, QTableWidgetItem('30000'))
                    self.setItem(i, 3, QTableWidgetItem('0'))
                
            def getTable(self):
                coils = []
                for i in range(self.nrow):
                    if self.item(i,0) is not None:
                        a = self.item(i,0).text()
                        b = self.item(i,1).text()
                        c = self.item(i,2).text()
                        d = self.item(i,3).text()
                        if a != '':
                            coils.append(a + b + 'f' + c + 'h' + d)
                return coils
        
        self.coilTable = CoilTable()
        
        # forward modelling options
        fforwardLabel = QLabel('Forward model:')
        fforwardModels = ['CS', 'FSlin', 'FSeq']
        self.fforwardCombo = QComboBox()
        for c in fforwardModels:
            self.fforwardCombo.addItem(c)
        fnoiseLabel = QLabel('Noise [%]:')
        self.fnoise = QLineEdit('0')
        self.fnoise.setValidator(QDoubleValidator())
        def fforwardBtnFunc():
            coils = self.coilTable.getTable()
            forwardModel = fforwardModels[self.fforwardCombo.currentIndex()]
            noise = float(self.fnoise.text())/100
            self.problem.forward(forwardModel, coils=coils, noise=noise)
            self.setupUI()
            self.tabs.setCurrentIndex(1)
        self.fforwardBtn = QPushButton('Compute Forward response')
        self.fforwardBtn.clicked.connect(fforwardBtnFunc)
        self.fforwardBtn.setAutoDefault(True)
        self.fforwardBtn.setStyleSheet('background-color:orange')
        
        
        # matplotlib widget
        self.mwf = MatplotlibWidget()
        self.mwf.setCallback(self.problem.showResults)
        
        
        # layout
        forwardLayout = QHBoxLayout()
        leftLayout = QVBoxLayout()
        leftLayout.addWidget(finstructions)
        leftLayout.addWidget(self.fimportBtn)
        fformLayout = QFormLayout()
        fformLayout.addRow(fnlayerLabel, self.fnlayer)
        fformLayout.addRow(fnsampleLabel, self.fnsample)
        leftLayout.addLayout(fformLayout)
        leftLayout.addWidget(self.paramTable)
        leftLayout.addWidget(self.fgenerateBtn)
        leftLayout.addWidget(coilLabel)
        leftLayout.addWidget(self.coilTable)
        
        rightLayout = QVBoxLayout()
        foptsLayout = QHBoxLayout()
        foptsLayout.addWidget(fforwardLabel)
        foptsLayout.addWidget(self.fforwardCombo)
        foptsLayout.addWidget(fnoiseLabel)
        foptsLayout.addWidget(self.fnoise)
        foptsLayout.addWidget(self.fforwardBtn)
        rightLayout.addLayout(foptsLayout)
        rightLayout.addWidget(self.mwf)
        
        forwardLayout.addLayout(leftLayout, 45)
        forwardLayout.addLayout(rightLayout, 55)
        
        forwardTab.setLayout(forwardLayout)
        
        
        #%% tab 1 importing data
        ''' STRUCTURE OF A TAB
        - first create the tab widget and it's layout
        - then create all the widgets that will interact in the tab
        - at the end do the layouting (create sub layout if needed) and
        '''

        importTab = QTabWidget()
        self.tabs.addTab(importTab, 'Importing')
        self.tabs.setCurrentIndex(1) # by default show import tab

        # select type of sensors
        def sensorComboFunc(index):
            print('sensor selected is:', sensors[index])
            if index == 1 or index == 2:
                showGF(True)
            else:
                showGF(False)
        self.sensorCombo = QComboBox()
        sensors = ['CMD Mini-Explorer',
                   'CMD Explorer'
                   ]
        sensors = sorted(sensors)
        sensors = ['All'] + sensors
        for sensor in sensors:
            self.sensorCombo.addItem(sensor)
        self.sensorCombo.activated.connect(sensorComboFunc)
        
        
        # import data
        def importBtnFunc():
            self._dialog = QFileDialog()
            fnames, _ = self._dialog.getOpenFileNames(importTab, 'Select data file(s)', self.datadir, '*.csv *.CSV')
            if len(fnames) > 0:
                self.processFname(fnames)
                    
        self.importBtn = QPushButton('Import Dataset(s)')
        self.importBtn.setAutoDefault(True)
        self.importBtn.setStyleSheet('background-color:orange')
        self.importBtn.clicked.connect(importBtnFunc)
        
        def importGFLoFunc():
            fname, _ = QFileDialog.getOpenFileName(importTab, 'Select data file', self.datadir)
            if fname != '':
                self.fnameLo = fname
                self.importGFLo.setText(os.path.basename(fname))
        self.importGFLo = QPushButton('Select Lo')
        self.importGFLo.clicked.connect(importGFLoFunc)
        def importGFHiFunc():
            fname, _ = QFileDialog.getOpenFileName(importTab, 'Select data file', self.datadir)
            if fname != '':
                self.fnameHi = fname
                self.importGFHi.setText(os.path.basename(fname))
        self.importGFHi = QPushButton('Select Hi')
        self.importGFHi.clicked.connect(importGFHiFunc)
        self.hxLabel = QLabel('Height above the ground [m]:')
        self.hxEdit = QLineEdit('0')
        self.hxEdit.setValidator(QDoubleValidator())
        def importGFApplyFunc():
            hx = float(self.hxEdit.text()) if self.hxEdit.text() != '' else 0
            device = self.sensorCombo.itemText(self.sensorCombo.currentIndex())
            self.problem.importGF(self.fnameLo, self.fnameHi, device, hx)
            self.infoDump('Surveys well imported')
            self.setupUI()
        self.importGFApply = QPushButton('Import')
        self.importGFApply.setStyleSheet('background-color: orange')
        self.importGFApply.clicked.connect(importGFApplyFunc)
        
        
        def showGF(arg):
            visibles = np.array([True, False, False, False, False, False])
            objs = [self.importBtn, self.importGFLo, self.importGFHi,
                    self.hxLabel, self.hxEdit, self.importGFApply]
            if arg is True:
                [o.setVisible(v) for o,v in zip(objs, ~visibles)]
            else:
                [o.setVisible(v) for o,v in zip(objs, visibles)]
        showGF(False)

        
        # projection (only if GPS data are available)
        self.projLabel = QLabel('EPSG:')
        self.projLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        ### Preparing the ~5000 projections:
        pcs = pd.read_csv(resource_path('emagpy/pcs.csv'))
        pcs_names = pcs['COORD_REF_SYS_NAME'].tolist()
        pcs_names.extend(pcs['COORD_REF_SYS_NAME_rev'].tolist())
        self.pcsCompleter = QCompleter(pcs_names)
        self.pcsCompleter.setCaseSensitivity(Qt.CaseInsensitive)
        
        self.projEdit = QLineEdit()
        self.projEdit.setPlaceholderText('Type projection CRS')
        self.projEdit.setToolTip('Type the CRS projection and then select from the options')
        # self.projEdit.setValidator(QDoubleValidator())
        self.projEdit.setCompleter(self.pcsCompleter)
        # self.projEdit.setEnabled(False)

        self.projBtn = QPushButton('Convert NMEA')
        self.projBtn.clicked.connect(self.projBtnFunc)
        self.projBtn.setEnabled(False)
        
        
        # filtering options
        self.filtLabel = QLabel('Filter Options |')
        self.filtLabel.setStyleSheet('font-weight:bold')

        
        # display options
        self.showParams = {'index': 0, 'coil':'all', 'contour':False, 'vmin':None,
                           'vmax':None,'pts':False, 'cmap':'viridis_r'} 

        # vmin/vmax filtering
        self.vminfLabel = QLabel('Vmin:')
        self.vminfEdit = QLineEdit()
        self.vminfEdit.setValidator(QDoubleValidator())
        self.vmaxfLabel = QLabel('Vmax:')
        self.vmaxfEdit = QLineEdit()
        self.vmaxfEdit.setValidator(QDoubleValidator())
        def keepApplyBtnFunc():
            vmin = float(self.vminfEdit.text()) if self.vminfEdit.text() != '' else None
            vmax = float(self.vmaxfEdit.text()) if self.vmaxfEdit.text() != '' else None
            self.problem.filterRange(vmin, vmax)
            self.replot()
        self.keepApplyBtn = QPushButton('Apply')
        self.keepApplyBtn.clicked.connect(keepApplyBtnFunc)
        self.keepApplyBtn.setEnabled(False)
        self.keepApplyBtn.setAutoDefault(True)
        
        # rolling mean
        self.rollingLabel = QLabel('Window size:')
        self.rollingEdit = QLineEdit('3')
        self.rollingEdit.setValidator(QIntValidator())
        def rollingBtnFunc():
            window = int(self.rollingEdit.text()) if self.rollingEdit.text() != '' else None
            self.problem.rollingMean(window=window)
            self.replot()
        self.rollingBtn = QPushButton('Rolling Mean')
        self.rollingBtn.clicked.connect(rollingBtnFunc)
        self.rollingBtn.setEnabled(False)
        self.rollingBtn.setAutoDefault(True)
        
        # manual point killer selection
        def ptsKillerBtnFunc():
            self.problem.surveys[self.showParams['index']].dropSelected()
            self.replot()
        self.ptsKillerBtn = QPushButton('Delete selected points')
        self.ptsKillerBtn.clicked.connect(ptsKillerBtnFunc)
        self.ptsKillerBtn.setEnabled(False)
        self.ptsKillerBtn.setAutoDefault(True)
        
        
        # display options
        self.displayLabel = QLabel('Display Options |')
        self.displayLabel.setStyleSheet('font-weight:bold')
        
        # survey selection (useful in case of time-lapse dataset)
        def surveyComboFunc(index):
            self.showParams['index'] = index
            self.replot()
        self.surveyCombo = QComboBox()
        self.surveyCombo.activated.connect(surveyComboFunc)
        self.surveyCombo.setEnabled(False)
        
        # coil selection
        def coilComboFunc(index):
            self.showParams['coil'] = self.coilCombo.itemText(index)
            self.replot()
        self.coilCombo = QComboBox()
        self.coilCombo.activated.connect(coilComboFunc)
        self.coilCombo.setEnabled(False)
        
        
        # switch between Raw data and map view
#        def showRadioFunc(state):
#            mwRaw.setCallback(self.problem.show)
#            mwRaw.replot(**self.showParams)
#        showRadio = QRadioButton('Raw')
#        showRadio.setChecked(True)
#        showRadio.toggled.connect(showRadioFunc)
#        showRadio.setEnabled(False)
#        def mapRadioFunc(state):
#            showMapOptions(state)
#            mwRaw.setCallback(self.problem.showMap)
#            mwRaw.replot(**self.showParams)
#        mapRadio = QRadioButton('Map')
#        mapRadio.setEnabled(False)
#        mapRadio.setChecked(False)
#        mapRadio.toggled.connect(mapRadioFunc)
#        showGroup = QGroupBox()
#        showGroupLayout = QHBoxLayout()
#        showGroupLayout.addWidget(showRadio)
#        showGroupLayout.addWidget(mapRadio)
#        showGroup.setLayout(showGroupLayout)
#        showGroup.setFlat(True)
#        showGroup.setContentsMargins(0,0,0,0)
#        showGroup.setStyleSheet('QGroupBox{border: 0px;'
#                                'border-style:inset;}')
        
        # alternative button checkable
        def showRadioFunc(state):
            if state:
                self.mapRadio.setChecked(False)
                showMapOptions(False)
                self.mwRaw.setCallback(self.problem.show)
                self.replot()
            else:
                self.mapRadio.setChecked(True)
        self.showRadio = QPushButton('Raw')
        self.showRadio.setCheckable(True)
        self.showRadio.setChecked(True)
        self.showRadio.toggled.connect(showRadioFunc)
        self.showRadio.setEnabled(False)
        self.showRadio.setAutoDefault(True)
        def mapRadioFunc(state):
            if state:
                self.showRadio.setChecked(False)
                showMapOptions(True)
                self.mwRaw.setCallback(self.problem.showMap)
                if self.showParams['coil'] == 'all':
                    self.coilCombo.setCurrentIndex(0)
                    coilComboFunc(0)
                self.replot()
            else:
                self.showRadio.setChecked(True)
        self.mapRadio = QPushButton('Map')
        self.mapRadio.setCheckable(True)
        self.mapRadio.setEnabled(False)
        self.mapRadio.setChecked(False)
        self.mapRadio.setAutoDefault(True)
        self.mapRadio.toggled.connect(mapRadioFunc)
        showGroup = QGroupBox()
        showGroupLayout = QHBoxLayout()
        showGroupLayout.addWidget(self.showRadio)
        showGroupLayout.addWidget(self.mapRadio)
        showGroup.setLayout(showGroupLayout)
        showGroup.setFlat(True)
        showGroup.setContentsMargins(0,0,0,0)
        showGroup.setStyleSheet('QGroupBox{border: 0px;'
                                'border-style:inset;}')
        
        def showMapOptions(arg):
            objs = [self.ptsLabel, self.ptsCheck, self.contourLabel,
                    self.contourCheck, self.cmapCombo, self.psMapExpBtn]
            [o.setVisible(arg) for o in objs]
            if arg is False:
                self.coilCombo.addItem('all')
            else:
                n = len(self.problem.coils)
                if self.coilCombo.currentIndex() == n:
                    self.coilCombo.setCurrentIndex(n-1)
                self.coilCombo.removeItem(n)
            print([self.coilCombo.itemText(i) for i in range(self.coilCombo.count())])

        # apply the display vmin/vmax for colorbar or y label                
        self.vminEdit = QLineEdit('')
        self.vminEdit.setValidator(QDoubleValidator())
        self.vmaxEdit = QLineEdit('')
        self.vmaxEdit.setValidator(QDoubleValidator())        
        def applyBtnFunc():
            self.showParams['vmin'] = float(self.vminEdit.text()) if self.vminEdit.text() != '' else None
            self.showParams['vmax'] = float(self.vmaxEdit.text()) if self.vmaxEdit.text() != '' else None
            self.replot()
        self.applyBtn = QPushButton('Apply')
        self.applyBtn.clicked.connect(applyBtnFunc)
        self.applyBtn.setEnabled(False)
        self.applyBtn.setAutoDefault(True)
        
        # select different colormap
        def cmapComboFunc(index):
            self.showParams['cmap'] = cmaps[index]
            self.replot()
        self.cmapCombo = QComboBox()
        cmaps = ['viridis', 'viridis_r', 'seismic', 'rainbow', 'jet','jet_r']
        for cmap in cmaps:
            self.cmapCombo.addItem(cmap)
        self.cmapCombo.activated.connect(cmapComboFunc)
        self.cmapCombo.setEnabled(False)
        self.cmapCombo.setVisible(False)

        # allow map contouring using tricontourf()
        self.contourLabel = QLabel('Contour')
        self.contourLabel.setVisible(False)
        def contourCheckFunc(state):
            self.showParams['contour'] = state
            self.replot()
        self.contourCheck = QCheckBox()
        self.contourCheck.setEnabled(False)
        self.contourCheck.clicked.connect(contourCheckFunc)
        self.contourCheck.setVisible(False)
        
        # show data points on the contoured map
        self.ptsLabel = QLabel('Points')
        self.ptsLabel.setVisible(False)
        def ptsCheckFunc(state):
            self.showParams['pts'] = state
            self.replot()
        self.ptsCheck = QCheckBox()
        self.ptsCheck.clicked.connect(ptsCheckFunc)
        self.ptsCheck.setToolTip('Show measurements points')
        self.ptsCheck.setVisible(False)
        
        # export GIS raster layer
        def setProjection():
            val = self.projEdit.text()
            if any(pcs['COORD_REF_SYS_NAME'] == val) is True or any(pcs['COORD_REF_SYS_NAME_rev'] == val) is True:
                epsg_code = pcs['COORD_REF_SYS_CODE'][pcs['COORD_REF_SYS_NAME'] == val].values
                epsgVal = 'EPSG:'+str(epsg_code)
                self.problem.projection = epsgVal
            else:
                self.errorDump('CRS projection is not correctly defined')
            
            
            
        def expPsMap():
            fname, _ = QFileDialog.getSaveFileName(importTab,'Export raster map', self.datadir, 'TIFF (*.tif)')
            print('fname:', fname)
            setProjection()
            self.problem.saveMap(fname=fname)
            
            
            
        self.psMapExpBtn = QPushButton('Exp. GIS layer')
        self.psMapExpBtn.setToolTip('Export a georeferenced TIFF file to directly be imported in GIS software.\n'
                                    'Choose the correct EPSG CRS projection!')
        self.psMapExpBtn.setVisible(False)
        self.psMapExpBtn.clicked.connect(expPsMap)
        


        # display it
        self.mwRaw = MatplotlibWidget()
        
        
        # layout
        importLayout = QVBoxLayout()
        
        topLayout = QHBoxLayout()
        topLayout.addWidget(self.sensorCombo, 15)
        topLayout.addWidget(self.importBtn, 15)
        topLayout.addWidget(self.importGFLo, 10)
        topLayout.addWidget(self.importGFHi, 10)
        topLayout.addWidget(self.hxLabel, 10)
        topLayout.addWidget(self.hxEdit, 10)
        topLayout.addWidget(self.importGFApply, 10)
        topLayout.addWidget(self.projLabel, 5)
        topLayout.addWidget(self.projEdit, 10)
        topLayout.addWidget(self.projBtn, 10)
        
        filtLayout = QHBoxLayout()
        filtLayout.addWidget(self.filtLabel)
        filtLayout.addWidget(self.vminfLabel)
        filtLayout.addWidget(self.vminfEdit)
        filtLayout.addWidget(self.vmaxfLabel)
        filtLayout.addWidget(self.vmaxfEdit)
        filtLayout.addWidget(self.keepApplyBtn)
        filtLayout.addWidget(self.rollingLabel)
        filtLayout.addWidget(self.rollingEdit)
        filtLayout.addWidget(self.rollingBtn)
        filtLayout.addWidget(self.ptsKillerBtn)
    
        midLayout = QHBoxLayout()
        midLayout.addWidget(self.displayLabel)
        midLayout.addWidget(self.surveyCombo, 7)
        midLayout.addWidget(QLabel('Select coil:'))
        midLayout.addWidget(self.coilCombo, 7)
        midLayout.addWidget(showGroup)
        midLayout.addWidget(QLabel('Vmin:'))
        midLayout.addWidget(self.vminEdit, 5)
        midLayout.addWidget(QLabel('Vmax:'))
        midLayout.addWidget(self.vmaxEdit, 5)
        midLayout.addWidget(self.applyBtn)
        midLayout.addWidget(self.cmapCombo)
        midLayout.addWidget(self.contourLabel)
        midLayout.addWidget(self.contourCheck)
        midLayout.addWidget(self.ptsLabel)
        midLayout.addWidget(self.ptsCheck)
        midLayout.addWidget(self.psMapExpBtn)
        
        importLayout.addLayout(topLayout)
        importLayout.addLayout(filtLayout)
        importLayout.addLayout(midLayout)
        importLayout.addWidget(self.mwRaw)
        
        importTab.setLayout(importLayout)
        

#        #%% filtering data
#        filterTab = QTabWidget()
#        self.tabs.addTab(filterTab, 'Filtering')
#
#        '''TODO add filtering tab ?
#        - add filtering tab with vmin/vmax filtering
#        - pointsKiller
#        - regridding of spatial data
#        - rolling mean
#        - how replace point by :
#        
#        '''        
#        
#        
#        
#        # graph
#        mwFiltered = MatplotlibWidget()
#        
#        
#        
#        # layout
#        filterLayout = QVBoxLayout()
#        
#        filterTab.setLayout(filterLayout)
#        
        
        
        #%% calibration
        calibTab = QTabWidget()
        self.tabs.addTab(calibTab, 'ERT Calibration')
        
        # import ECa csv (same format, one coil per column)
        def ecaImportBtnFunc():
            fname, _ = QFileDialog.getOpenFileName(importTab, 'Select data file', self.datadir, '*.csv')
            if fname != '':
                self.fnameECa = fname
                self.ecaImportBtn.setText(os.path.basename(fname))
        self.ecaImportBtn = QPushButton('Import ECa')
        self.ecaImportBtn.clicked.connect(ecaImportBtnFunc)
        
        # import EC depth-specific (one depth per column) -> can be from ERT
        def ecImportBtnFunc():
            fname, _ = QFileDialog.getOpenFileName(importTab, 'Select data file', self.datadir, '*.csv')
            if fname != '':
                self.fnameEC = fname
                self.fnameresmod = None
                self.ecImportBtn.setText(os.path.basename(fname))
        self.ecImportBtn = QPushButton('Import EC profiles')
        self.ecImportBtn.clicked.connect(ecImportBtnFunc)


        # import Resistivity model, format should be x, z, resistivity with space delimeter, e.g. R2 'f001_res.dat' format
        def ertImportBtnFunc():
            fname, _ = QFileDialog.getOpenFileName(importTab, 'Select data file', self.datadir, '*.dat')
            if fname != '':
                self.fnameresmod = fname
                self.fnameEC = None

                self.ertImportBtn.setText(os.path.basename(fname))
        self.ertImportBtn = QPushButton('Import ERT Model')
        self.ertImportBtn.clicked.connect(ertImportBtnFunc)
        
        # choose which forward model to use
        self.forwardCalibCombo = QComboBox()
        forwardCalibs = ['CS', 'FSlin', 'FSeq']
        for forwardCalib in forwardCalibs:
            self.forwardCalibCombo.addItem(forwardCalib)
        
        # perform the fit (equations display in the console)
        def fitCalibBtnFunc():
            forwardModel = self.forwardCalibCombo.itemText(self.forwardCalibCombo.currentIndex())
            self.mwCalib.setCallback(self.problem.calibrate)
            self.mwCalib.replot(fnameECa=self.fnameECa, fnameEC=self.fnameEC, fnameresmod=self.fnameresmod,
                           forwardModel=forwardModel)
        self.fitCalibBtn = QPushButton('Fit calibration')
        self.fitCalibBtn.clicked.connect(fitCalibBtnFunc)
        
        # apply the calibration to the ECa measurements of the survey imported
        def applyCalibBtnFunc():
            forwardModel = self.forwardCalibCombo.itemText(self.forwardCalibCombo.currentIndex())
            self.mwCalib.replot(fnameECa=self.fnameECa, fnameEC=self.fnameEC, fnameresmod=self.fnameresmod,
                           forwardModel=forwardModel, apply=True)
            self.replot()
            self.infoDump('Calibration applied')
        self.applyCalibBtn = QPushButton('Apply Calibration')
        self.applyCalibBtn.clicked.connect(applyCalibBtnFunc)
        
        
        # graph
        self.mwCalib = MatplotlibWidget()
        
        
        # layout
        calibLayout = QVBoxLayout()
        calibOptions = QHBoxLayout()
        calibOptions.addWidget(self.ecaImportBtn)
        calibOptions.addWidget(self.ecImportBtn)
        calibOptions.addWidget(self.ertImportBtn)
        calibOptions.addWidget(self.forwardCalibCombo)
        calibOptions.addWidget(self.fitCalibBtn)
        calibOptions.addWidget(self.applyCalibBtn)
        calibLayout.addLayout(calibOptions)
        calibLayout.addWidget(self.mwCalib)
        
        calibTab.setLayout(calibLayout)
        
        
        #%% error model
        errTab = QTabWidget()
        self.tabs.addTab(errTab, 'Error Modelling')
        
        self.surveyErrCombo = QComboBox()
        self.coilErrCombo = QComboBox()
        
        def fitErrBtnFunc():
            index = self.surveyErrCombo.currentIndex()
            coil = self.coilErrCombo.itemText(self.coilErrCombo.currentIndex())
            self.mwErr.setCallback(self.problem.crossOverPoints)
            self.mwErr.replot(index=index, coil=coil, dump=self.infoDump)
            self.mwErrMap.setCallback(self.problem.plotCrossOverMap)
            self.mwErrMap.replot(index=index, coil=coil)
        self.fitErrBtn = QPushButton('Fit Error Model based on colocated measurements')
        self.fitErrBtn.clicked.connect(fitErrBtnFunc)
        
        
        # graph
        self.mwErr = MatplotlibWidget()
        self.mwErrMap = MatplotlibWidget()
        
        # layout
        errLayout = QVBoxLayout()
        errOptionLayout = QHBoxLayout()
        errOptionLayout.addWidget(self.surveyErrCombo)
        errOptionLayout.addWidget(self.coilErrCombo)
        errOptionLayout.addWidget(self.fitErrBtn)
        errLayout.addLayout(errOptionLayout)
        errGraphLayout = QHBoxLayout()
        errGraphLayout.addWidget(self.mwErrMap)
        errGraphLayout.addWidget(self.mwErr)
        errLayout.addLayout(errGraphLayout)
        
        errTab.setLayout(errLayout)
        
        
        
        #%% inversion settings (starting model + lcurve)
        settingsTab = QTabWidget()
        self.tabs.addTab(settingsTab, 'Inversion Settings')
        
        class ModelTable(QTableWidget):
            def __init__(self, nrow=3, headers=['Layer bottom depth [m]', 'Fixed', 'EC [mS/m]', 'Fixed']):
                ncol = len(headers)
                super(ModelTable, self).__init__(nrow, ncol)
                self.nrow = nrow
                self.ncol = ncol
                self.headers = headers
                self.setHorizontalHeaderLabels(self.headers)
                self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
                self.horizontalHeader().sortIndicatorChanged.connect(self.setAllFixed)
                
            def setTable(self, depths0, conds0, fixedDepths=None, fixedConds=None):
                if len(conds0) < 2:
                    return
                self.clear()
                self.setHorizontalHeaderLabels(self.headers)
                self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
                self.nrow = len(conds0)
                self.setRowCount(self.nrow)
                for i, depth in enumerate(depths0):
                    self.setItem(i, 0, QTableWidgetItem('{:.3f}'.format(depth)))
                for j in range(i+1, self.nrow-1):
                    self.setItem(j, 0, QTableWidgetItem('0.0'))
                self.setItem(self.nrow - 1, 0, QTableWidgetItem('-'))
                self.item(self.nrow - 1, 0).setFlags(Qt.ItemIsEnabled)
                for i, cond in enumerate(conds0):
                    self.setItem(i, 2, QTableWidgetItem('{:.2f}'.format(cond)))
                for j in range(i+1, self.nrow):
                    self.setItem(j, 2, QTableWidgetItem('20.0'))
                self.setFixedDepths(fixedDepths)
                self.setFixedConds(fixedConds)

#            def keyPressEvent(self, e):
##                print(e.modifiers(), 'and', e.key())
#                if (e.modifiers() == Qt.ControlModifier) & (e.key() == Qt.Key_V):
#                    cell = self.selectedIndexes()[0]
#                    c0, r0 = cell.column(), cell.row()
#                    self.paste(c0, r0)
#                elif e.modifiers() != Qt.ControlModifier: # start editing
##                    print('start editing...')
#                    cell = self.selectedIndexes()[0]
#                    c0, r0 = cell.column(), cell.row()
#                    self.editItem(self.item(r0,c0))

            def getTable(self):
                depths0 = np.zeros(self.nrow - 1)
                conds0 = np.zeros(self.nrow)
                for i in range(self.nrow -1):
                    depths0[i] = float(self.item(i, 0).text())
                for i in range(self.nrow):
                    conds0[i] = float(self.item(i, 2).text())
                fixedDepths = self.getFixedDepths()
                fixedConds = self.getFixedConds()
                return depths0, conds0, fixedDepths, fixedConds
            
            def addRow(self):
                depths0, conds0, fixedDepths, fixedConds = self.getTable()
                depths0 = np.r_[depths0, depths0[-1]+1]
                conds0 = np.r_[conds0, conds0[-1]]
                fixedDepths = np.r_[fixedDepths, fixedDepths[-1]]
                fixedConds = np.r_[fixedConds, fixedConds[-1]]
                self.setTable(depths0, conds0, fixedDepths, fixedConds)
                
            def delRow(self):
                depths0, conds0, fixedDepths, fixedConds = self.getTable()
                self.setTable(depths0[:-1], conds0[:-1],
                              fixedDepths[:-1], fixedConds[:-1])
                
            def setFixedDepths(self, vals=None):
                if vals is None:
                    vals = np.ones(self.nrow-1, dtype=bool)
                for i in range(len(vals)):
                    checkBoxWidget = QWidget()
                    checkBoxLayout = QHBoxLayout()
                    checkBoxLayout.setContentsMargins(5,5,5,5)
                    checkBoxLayout.setAlignment(Qt.AlignCenter)
                    fixedCheck = QCheckBox()
                    fixedCheck.setChecked(bool(vals[i]))
                    checkBoxLayout.addWidget(fixedCheck)
                    checkBoxWidget.setLayout(checkBoxLayout)
                    self.setCellWidget(i, 1, checkBoxWidget)
                self.setItem(self.nrow - 1, 1, QTableWidgetItem('-'))
                self.item(self.nrow - 1, 1).setFlags(Qt.ItemIsEnabled)
                
                
            def setFixedConds(self, vals=None):
                if vals is None:
                    vals = np.zeros(self.nrow, dtype=bool)
                for i in range(len(vals)):
                    checkBoxWidget = QWidget()
                    checkBoxLayout = QHBoxLayout()
                    checkBoxLayout.setContentsMargins(5,5,5,5)
                    checkBoxLayout.setAlignment(Qt.AlignCenter)
                    fixedCheck = QCheckBox()
                    fixedCheck.setChecked(bool(vals[i]))
                    checkBoxLayout.addWidget(fixedCheck)
                    checkBoxWidget.setLayout(checkBoxLayout)
                    self.setCellWidget(i, 3, checkBoxWidget)

            def setAllFixed(self, colIndex):
                if self.headers[colIndex] == 'Fixed':
                    j = colIndex
                    n = self.row-1 if colIndex == 1 else self.row
                    for i in range(n):
                        fixedCheck = self.cellWidget(i, j).findChildren(QCheckBox)[0]
                        if fixedCheck.isChecked() is True:
                            fixedCheck.setChecked(False)
                        else:
                            fixedCheck.setChecked(True)
                            
            def getFixedDepths(self):
                fixedDepths = np.zeros(self.nrow - 1, dtype=bool)
                for i in range(self.nrow-1):
                    fixedCheck = self.cellWidget(i, 1).findChildren(QCheckBox)[0]
                    if fixedCheck.isChecked() is True:
                        fixedDepths[i] = True
                return fixedDepths
            
            def getFixedConds(self):
                fixedConds = np.zeros(self.nrow, dtype=bool)
                for i in range(self.nrow):
                    fixedCheck = self.cellWidget(i, 3).findChildren(QCheckBox)[0]
                    if fixedCheck.isChecked() is True:
                        fixedConds[i] = True
                return fixedConds
            
        
        self.modelLabel = QLabel('Bottom depth and starting conductivity of each layer.')
        
        def addRowBtnFunc():
            self.modelTable.addRow()
        self.addRowBtn = QPushButton('Add Row')
        self.addRowBtn.clicked.connect(addRowBtnFunc)
        
        def delRowBtnFunc():
            self.modelTable.delRow()
        self.delRowBtn = QPushButton('Remove Row')
        self.delRowBtn.clicked.connect(delRowBtnFunc)
        
        self.nLayerLabel = QLabel('Number of Layer:')
        self.nLayerEdit = QLineEdit('3')
        self.nLayerEdit.setValidator(QIntValidator())
        
        self.thicknessLabel = QLabel('Thickness:')
        self.thicknessEdit = QLineEdit('0.5')
        self.thicknessEdit.setValidator(QDoubleValidator())
        
        self.startingLabel = QLabel('Starting EC:')
        self.startingEdit = QLineEdit('20')
        self.startingEdit.setValidator(QDoubleValidator())
        
        def createModelBtnFunc():
            nlayer = int(self.nLayerEdit.text()) if self.nLayerEdit.text() != '' else 3
            thick = float(self.thicknessEdit.text()) if self.thicknessEdit.text() != '' else 0.5
            ecstart = float(self.startingEdit.text()) if self.startingEdit.text() != '' else 20
            depths = np.linspace(thick, thick*(nlayer-1), nlayer-1)
            conds0 = np.ones(len(depths)+1) * ecstart
            self.modelTable.setTable(depths, conds0)
        self.createModelBtn = QPushButton('Create Model')
        self.createModelBtn.setAutoDefault(True)
        self.createModelBtn.clicked.connect(createModelBtnFunc)
        
        
        self.modelTable = ModelTable()
        self.modelTable.setTable([0.7, 1.5], [20, 10, 50])
        
        
        def lcurveBtnFunc():
            self.mwlcurve.plot(self.problem.lcurve)
        self.lcurveBtn = QPushButton('Fit L-curve')
        self.lcurveBtn.clicked.connect(lcurveBtnFunc)
        
        # graph
        self.mwlcurve = MatplotlibWidget()
        
        
        # layout
        settingsLayout = QHBoxLayout()
        
        invStartLayout = QVBoxLayout()
        invStartLayout.addWidget(self.modelLabel)
        invBtnLayout = QHBoxLayout()
        invBtnLayout.addWidget(self.addRowBtn)
        invBtnLayout.addWidget(self.delRowBtn)
        invStartLayout.addLayout(invBtnLayout)
        invFormLayout = QFormLayout()
        invFormLayout.addRow(self.nLayerLabel, self.nLayerEdit)
        invFormLayout.addRow(self.thicknessLabel, self.thicknessEdit)
        invFormLayout.addRow(self.startingLabel, self.startingEdit)
        invStartLayout.addLayout(invFormLayout)
        invStartLayout.addWidget(self.createModelBtn)        
        invStartLayout.addWidget(self.modelTable)
        settingsLayout.addLayout(invStartLayout)
        
        invlcurveLayout = QVBoxLayout()
        invlcurveLayout.addWidget(self.lcurveBtn)
        invlcurveLayout.addWidget(self.mwlcurve)
        settingsLayout.addLayout(invlcurveLayout)
        
        
        settingsTab.setLayout(settingsLayout)
        
        
        #%% invert graph
        invTab = QTabWidget()
        self.tabs.addTab(invTab, 'Inversion')
        
        def forwardComboFunc(index):
            objs = [self.methodCombo, self.betaEdit, self.gammaEdit, self.lCombo,
                    self.nitEdit, self.parallelCheck]
            if index == 1: #GN inversion
                [o.setEnabled(False) for o in objs]
            else:
                [o.setEnabled(True) for o in objs]
        self.forwardCombo = QComboBox()
        forwardModels = ['CS', 'CS (fast)', 'FSlin', 'FSeq', 'Q']
        for forwardModel in forwardModels:
            self.forwardCombo.addItem(forwardModel)
        self.forwardCombo.currentIndexChanged.connect(forwardComboFunc)
        self.forwardCombo.setToolTip('''Choice of forward model:
        CS fast : Cumulative Sensitivity with
        Gauss-Newton solver (faster).
        CS : Cumulative Sensitivity with minimize solver
        FS : Full solution with LIN conversion
        FSandrade : Full solution without LIN conversion''')

        def methodComboFunc(index):
            objs1 = [self.alphaLabel, self.alphaEdit,
                    self.betaLabel, self.betaEdit,
                    self.gammaLabel, self.gammaEdit,
                    self.nitLabel, self.nitEdit,
                    self.parallelCheck]
            objs2 = [self.annSampleLabel, self.annSampleEdit,
                     self.annNoiseLabel, self.annNoiseEdit]
            if index == 8: #ANN
                [o.setVisible(False) for o in objs1]
                [o.setVisible(True) for o in objs2]
            else:
                [o.setVisible(True) for o in objs1]
                [o.setVisible(False) for o in objs2]
                if len(self.problem.surveys) > 1:
                    self.gammaLabel.setVisible(True)
                    self.gammaEdit.setVisible(True)
                else:
                    self.gammaLabel.setVisible(False)
                    self.gammaEdit.setVisible(False)
            
                
        self.methodCombo = QComboBox()
        self.methodCombo.setToolTip('''Choice of solver:
        L-BFGS-B : minimize, faster
        CG : Congugate Gradient, fast
        TNC : Truncated Newton, robust
        Nelder-Mead : more robust
        ROPE : McMC-based
        SCEUA : MCMC-based
        DREAM : MCMC-based
        MCMC : Markov Chain Monte Carlo
        ANN : Artificial Neural Network''')
        mMinimize = ['L-BFGS-B', 'CG', 'TNC', 'Nelder-Mead']
        mMCMC = ['ROPE', 'SCEUA', 'DREAM', 'MCMC']
        methods = mMinimize + mMCMC + ['ANN']
        for method in methods:
            self.methodCombo.addItem(method)
        self.methodCombo.currentIndexChanged.connect(methodComboFunc)
        
        self.alphaLabel = QLabel('Vertical smooth:')
        self.alphaEdit = QLineEdit('0.07')
        self.alphaEdit.setValidator(QDoubleValidator())
        self.alphaEdit.setToolTip('Vertical smoothing between layers from the same profiles.\n'
                                  'Can be determined from the L-Curve in "inversion settings" tab.')

        self.betaLabel = QLabel('Lateral smooth:')
        self.betaEdit = QLineEdit('0.0')
        self.betaEdit.setToolTip('Lateral smoothing between contiguous profiles.\n 0 means no lateral smoothing.')
        self.betaEdit.setValidator(QDoubleValidator())
        
        self.gammaLabel = QLabel('Time smooth:')
        self.gammaLabel.setVisible(False)
        self.gammaEdit = QLineEdit('0.0')
        self.gammaEdit.setToolTip('Smoothing between the first survey and other surveys.')
        self.gammaEdit.setValidator(QDoubleValidator())
        self.gammaEdit.setVisible(False)
        
        self.lLabel = QLabel('Regularization:')
        self.lCombo = QComboBox()
        self.lCombo.addItem('l2')
        self.lCombo.addItem('l1')
        self.lCombo.setToolTip('Set to l1 for sharp model and l2 for smooth model.')
        self.lCombo.setCurrentIndex(0)
        
        self.nitLabel = QLabel('Nit:')
        self.nitEdit = QLineEdit('15')
        self.nitEdit.setToolTip('Maximum Number of Iterations')
        self.nitEdit.setValidator(QIntValidator())
        
        self.parallelCheck = QCheckBox('Parallel')
        self.parallelCheck.setToolTip('If checked, inversion will be run in parallel.')
        
        self.annSampleLabel = QLabel('Number of samples:')
        self.annSampleEdit = QLineEdit('100')
        self.annSampleEdit.setValidator(QIntValidator())
        self.annSampleEdit.setToolTip('Number of synthetic samples for training the model.')
        self.annSampleLabel.setVisible(False)
        self.annSampleEdit.setVisible(False)
        
        self.annNoiseLabel = QLabel('Noise [%]:')
        self.annNoiseEdit = QLineEdit('0')
        self.annNoiseEdit.setValidator(QDoubleValidator())
        self.annNoiseEdit.setToolTip('Noise in percent to apply on synthetic data for training the network.')
        self.annNoiseLabel.setVisible(False)
        self.annNoiseEdit.setVisible(False)
        
        # opts = [self.alphaLabel, self.alphaEdit, self.betaLabel, self.betaEdit,
        #         self.gammaLabel, self.gammaEdit, self.lLabel,
        #         self.lCombo, self.nitLabel, self.nitEdit, self.parallelCheck]
        
        def logTextFunc(arg):
            text = self.logText.toPlainText()
            if arg[0] == '\r':
                text = text.split('\n')
                text.pop() # remove last element
                text = '\n'.join(text) + arg
            else:
                text = text + arg
            self.logText.setText(text)
            QApplication.processEvents()
        self.logText = QTextEdit('hello there !')
        self.logText.setReadOnly(True)
        
        def invertBtnFunc():
            if self.running == False:
                self.problem.ikill = False
                self.running = True
                self.invertBtn.setText('Kill')
                self.invertBtn.setStyleSheet('background-color:red')
            else: # button press while running => killing
                print('killing')
                self.problem.ikill = True
                return
            outputStack.setCurrentIndex(0)
            self.logText.clear()

            # collect parameters
            depths0, conds0, fixedDepths, fixedConds = self.modelTable.getTable()
            self.problem.setInit(depths0, conds0, fixedDepths, fixedConds)
            regularization = self.lCombo.itemText(self.lCombo.currentIndex())
            alpha = float(self.alphaEdit.text()) if self.alphaEdit.text() != '' else 0.07
            forwardModel = self.forwardCombo.itemText(self.forwardCombo.currentIndex())
            method = self.methodCombo.itemText(self.methodCombo.currentIndex())
            beta = float(self.betaEdit.text()) if self.betaEdit.text() != '' else 0.0
            gamma = float(self.gammaEdit.text()) if self.gammaEdit.text() != '' else 0.0
            depths = np.r_[[0], depths0, [-np.inf]]
            nit = int(self.nitEdit.text()) if self.nitEdit.text() != '' else 15
            nsample = int(self.annSampleEdit.text()) if self.annSampleEdit.text() != '' else 100
            noise = float(self.annNoiseEdit.text()) if self.annNoiseEdit.text() != '' else 0
            njobs = -1 if self.parallelCheck.isChecked() else 1
            self.sliceCombo.clear()
            for i in range(len(depths)-1):
                self.sliceCombo.addItem('{:.2f}m - {:.2f}m'.format(depths[i], depths[i+1]))
            self.sliceCombo.activated.connect(sliceComboFunc)
            
            # invert
            if forwardModel == 'CS (fast)':
                self.problem.invertGN(alpha=alpha, dump=logTextFunc)
            else:
                self.problem.invert(forwardModel=forwardModel, alpha=alpha,
                                    dump=logTextFunc, regularization=regularization,
                                    method=method, options={'maxiter':nit},
                                    beta=beta, gamma=gamma, nsample=nsample,
                                    noise=noise/100, njobs=njobs)
            
            # plot results
            if self.problem.ikill == False: # program wasn't killed
                self.mwInv.setCallback(self.problem.showResults)
                self.mwInv.replot(**showInvParams)
                self.mwInvMap.setCallback(self.problem.showSlice)
                self.mwInvMap.replot(**showInvMapParams)
                self.mwMisfit.plot(self.problem.showMisfit)
                self.mwOne2One.plot(self.problem.showOne2one)
                outputStack.setCurrentIndex(1)
            
            # reset button
            self.running = False
            self.problem.ikill = False
            self.invertBtn.setText('Invert')
            self.invertBtn.setStyleSheet('background-color:orange')
               
            
        self.invertBtn = QPushButton('Invert')
        self.invertBtn.setStyleSheet('background-color:orange')
        self.invertBtn.clicked.connect(invertBtnFunc)
        
        
        # profile display
        showInvParams = {'index':0, 'vmin':None, 'vmax':None, 
                         'cmap':'viridis_r', 'contour':False}
        
        def cmapInvComboFunc(index):
            showInvParams['cmap'] = self.cmapInvCombo.itemText(index)
            self.mwInv.replot(**showInvParams)
        self.cmapInvCombo = QComboBox()
        cmaps = ['viridis_r', 'viridis', 'seismic', 'rainbow', 'jet']
        for cmap in cmaps:
            self.cmapInvCombo.addItem(cmap)
        self.cmapInvCombo.activated.connect(cmapInvComboFunc)
        
        def surveyInvComboFunc(index):
            showInvParams['index'] = index
            self.mwInv.replot(**showInvParams)    
        self.surveyInvCombo = QComboBox()
        self.surveyInvCombo.activated.connect(surveyInvComboFunc)
        
        self.vminInvLabel = QLabel('vmin:')
        self.vminInvEdit = QLineEdit('')
        self.vminInvEdit.setValidator(QDoubleValidator())
        
        self.vmaxInvLabel = QLabel('vmax:')
        self.vmaxInvEdit = QLineEdit('')
        self.vmaxInvEdit.setValidator(QDoubleValidator())
        
        def applyInvBtnFunc():
            vmin = float(self.vminInvEdit.text()) if self.vminInvEdit.text() != '' else None
            vmax = float(self.vmaxInvEdit.text()) if self.vmaxInvEdit.text() != '' else None
            showInvParams['vmin'] = vmin
            showInvParams['vmax'] = vmax
            self.mwInv.replot(**showInvParams)
        self.applyInvBtn = QPushButton('Apply')
        self.applyInvBtn.clicked.connect(applyInvBtnFunc)
        
        self.contourInvLabel = QLabel('Contour:')
        def contourInvCheckFunc(state):
            showInvParams['contour'] = state
            self.mwInv.replot(**showInvParams)
        self.contourInvCheck = QCheckBox()
        self.contourInvCheck.clicked.connect(contourInvCheckFunc)
 
        def saveInvDataBtnFunc():
            fdir = QFileDialog.getExistingDirectory(invTab, 'Choose directory where to save the files')
            if fdir != '':
                self.problem.saveInvData(fdir)
        self.saveInvDataBtn = QPushButton('Save Results')
        self.saveInvDataBtn.clicked.connect(saveInvDataBtnFunc)

        
        
        # for the map
        showInvMapParams = {'index':0, 'islice':0, 'vmin':None, 'vmax':None, 'cmap':'viridis_r'}

        def cmapInvMapComboFunc(index):
            showInvMapParams['cmap'] = self.cmapInvMapCombo.itemText(index)
            self.mwInvMap.replot(**showInvMapParams)
        self.cmapInvMapCombo = QComboBox()
        cmaps = ['viridis_r', 'viridis', 'seismic', 'rainbow', 'jet']
        for cmap in cmaps:
            self.cmapInvMapCombo.addItem(cmap)
        self.cmapInvMapCombo.activated.connect(cmapInvMapComboFunc)
        
        def surveyInvMapComboFunc(index):
            showInvMapParams['index'] = index
            self.mwInvMap.replot(**showInvMapParams)
        self.surveyInvMapCombo = QComboBox()
        self.surveyInvMapCombo.activated.connect(surveyInvMapComboFunc)
        
        self.vminInvMapLabel = QLabel('vmin:')
        self.vminInvMapEdit = QLineEdit('')
        self.vminInvMapEdit.setValidator(QDoubleValidator())
        
        self.vmaxInvMapLabel = QLabel('vmax:')
        self.vmaxInvMapEdit = QLineEdit('')
        self.vmaxInvMapEdit.setValidator(QDoubleValidator())
        
        def applyInvMapBtnFunc():
            vmin = float(self.vminInvMapEdit.text()) if self.vminInvMapEdit.text() != '' else None
            vmax = float(self.vmaxInvMapEdit.text()) if self.vmaxInvMapEdit.text() != '' else None
            showInvMapParams['vmin'] = vmin
            showInvMapParams['vmax'] = vmax
            self.mwInvMap.replot(**showInvMapParams)
        self.applyInvMapBtn = QPushButton('Apply')
        self.applyInvMapBtn.clicked.connect(applyInvMapBtnFunc)

        self.sliceLabel = QLabel('Layer:')
        def sliceComboFunc(index):
            showInvMapParams['islice'] = index
            self.mwInvMap.replot(**showInvMapParams)
        self.sliceCombo = QComboBox()
        self.sliceCombo.activated.connect(sliceComboFunc)
        
        self.contourInvMapLabel = QLabel('Contour:')
        def contourInvMapCheckFunc(state):
            showInvMapParams['contour'] = state
            self.mwInvMap.replot(**showInvMapParams)
        self.contourInvMapCheck = QCheckBox()
        self.contourInvMapCheck.clicked.connect(contourInvMapCheckFunc)
        
        def saveInvMapDataBtnFunc():
            fdir = QFileDialog.getExistingDirectory(invTab, 'Choose directory where to save the files')
            if fdir != '':
                self.problem.saveInvData(fdir)
        self.saveInvMapDataBtn = QPushButton('Save Results')
        self.saveInvMapDataBtn.clicked.connect(saveInvMapDataBtnFunc)
        
        
        
        self.graphTabs = QTabWidget()
        self.profTab = QTabWidget()
        self.mapTab = QTabWidget()
        self.graphTabs.addTab(self.profTab, 'Profile')
        self.graphTabs.addTab(self.mapTab, 'Slice')
        
        # graph or log    
        self.mwInv = MatplotlibWidget()
        self.mwInvMap = MatplotlibWidget()

        
        # layout
        invLayout = QVBoxLayout()
        
        invOptions = QHBoxLayout()
        invOptions.addWidget(self.forwardCombo, 15)
        invOptions.addWidget(self.methodCombo, 5)
        invOptions.addWidget(self.alphaLabel)
        invOptions.addWidget(self.alphaEdit)
        invOptions.addWidget(self.betaLabel)
        invOptions.addWidget(self.betaEdit)
        invOptions.addWidget(self.gammaLabel)
        invOptions.addWidget(self.gammaEdit)
        invOptions.addWidget(self.lLabel)
        invOptions.addWidget(self.lCombo)
        invOptions.addWidget(self.nitLabel)
        invOptions.addWidget(self.nitEdit)
        invOptions.addWidget(self.parallelCheck) # disable for compilation
        invOptions.addWidget(self.annSampleLabel)
        invOptions.addWidget(self.annSampleEdit)
        invOptions.addWidget(self.annNoiseLabel)
        invOptions.addWidget(self.annNoiseEdit)
        invOptions.addWidget(self.invertBtn, 25)
        invLayout.addLayout(invOptions)
        
        outputStack = QStackedLayout()
        outputLog = QVBoxLayout()
        outputRes = QVBoxLayout()
        outputLogW = QWidget()
        outputLogW.setLayout(outputLog)
        outputResW = QWidget()
        outputResW.setLayout(outputRes)
        outputStack.addWidget(outputLogW)
        outputStack.addWidget(outputResW)
        
        outputRes.addWidget(self.graphTabs)
        
        profLayout = QVBoxLayout()
        profOptionsLayout = QHBoxLayout()
        profOptionsLayout.addWidget(self.surveyInvCombo)
        profOptionsLayout.addWidget(self.vminInvLabel)
        profOptionsLayout.addWidget(self.vminInvEdit)
        profOptionsLayout.addWidget(self.vmaxInvLabel)
        profOptionsLayout.addWidget(self.vmaxInvEdit)
        profOptionsLayout.addWidget(self.applyInvBtn)
        profOptionsLayout.addWidget(self.contourInvLabel)
        profOptionsLayout.addWidget(self.contourInvCheck)
        profOptionsLayout.addWidget(self.cmapInvCombo)
        profOptionsLayout.addWidget(self.saveInvDataBtn)
        profLayout.addLayout(profOptionsLayout)
        profLayout.addWidget(self.mwInv)
        self.profTab.setLayout(profLayout)


        mapLayout = QVBoxLayout()
        mapOptionsLayout = QHBoxLayout()
        mapOptionsLayout.addWidget(self.surveyInvMapCombo)
        mapOptionsLayout.addWidget(self.sliceLabel)
        mapOptionsLayout.addWidget(self.sliceCombo)
        mapOptionsLayout.addWidget(self.vminInvMapLabel)
        mapOptionsLayout.addWidget(self.vminInvMapEdit)
        mapOptionsLayout.addWidget(self.vmaxInvMapLabel)
        mapOptionsLayout.addWidget(self.vmaxInvMapEdit)
        mapOptionsLayout.addWidget(self.applyInvMapBtn)
        mapOptionsLayout.addWidget(self.contourInvMapLabel)
        mapOptionsLayout.addWidget(self.contourInvMapCheck)
        mapOptionsLayout.addWidget(self.cmapInvMapCombo)
        mapOptionsLayout.addWidget(self.saveInvMapDataBtn)
        mapLayout.addLayout(mapOptionsLayout)
        mapLayout.addWidget(self.mwInvMap)
        self.mapTab.setLayout(mapLayout)
        
        outputLog.addWidget(self.logText)
        
        invLayout.addLayout(outputStack)
        
        invTab.setLayout(invLayout)
        
        
        #%% goodness of fit
        postTab = QTabWidget()
        self.tabs.addTab(postTab, 'Misfit')
        
        self.misfitLabel = QLabel('Misfit after inversion')
        
        self.mwMisfit = MatplotlibWidget()
        self.mwOne2One = MatplotlibWidget()
        
        # layout
        postLayout = QVBoxLayout()
        postLayout.addWidget(self.misfitLabel)
        graphLayout = QHBoxLayout()
        graphLayout.addWidget(self.mwMisfit)
        graphLayout.addWidget(self.mwOne2One)
        postLayout.addLayout(graphLayout)
        
        postTab.setLayout(postLayout)
        
        
        #%%about tab
        tabAbout = QTabWidget()
        self.tabs.addTab(tabAbout, 'About')

        infoLayout = QVBoxLayout()
        aboutText = QLabel()
        aboutText.setText('''<h1>About EMagPy</h1>
<p><b>Version: {:s}</b></p>
<p><i>EMagPy is a free and open source software for inversion of 1D electromagnetic data</i></p>
<p>If you encouter any issues or would like to submit a feature request, please raise an issue on our gitlab repository at:</p>
<p><a href="https://gitlab.com/hkex/emagpy/issues">https://gitlab.com/hkex/emagpy/issues</a></p>
<p>EMagPy uses a few Python packages: numpy, pandas, matplotlib, scipy, spotpy, pyproj, joblib, rasterio (optional).
<ul>
<li>Travis E, Oliphant. <strong>A guide to NumPy</strong>,
USA: Trelgol Publishing, (2006).
</li>
<li>
John D. Hunter.
<strong>Matplotlib: A 2D Graphics Environment</strong>,
Computing in Science &amp; Engineering, <strong>9</strong>, 90-95 (2007),
<a class="reference external" href="https://doi.org/10.1109/MCSE.2007.55">DOI:10.1109/MCSE.2007.55</a>
</li>

<li>
Wes McKinney. <strong>Data Structures for Statistical Computing in Python</strong>,
Proceedings of the 9th Python in Science Conference, <strong>51-56</strong> (2010)
</li>
<li>
K. Jarrod Millman and Michael Aivazis.
<strong>Python for Scientists and Engineers</strong>,
Computing in Science & Engineering, <strong>13</strong>, 9-12 (2011),
<a class="reference external" href="https://doi.org/10.1109/MCSE.2011.36">DOI:10.1109/MCSE.2011.36</a>
</li>
<li>
Houska, T., Kraft, P., Chamorro-Chavez, A. and Breuer, L.
<strong>SPOTting Model Parameters Using a Ready-Made Python Package</strong>,
PLoS ONE, <strong>10</strong>,12 (2015)
<a class="reference external" href="https://doi.org/10.1371/journal.pone.0145180">DOI:10.1371/journal.pone.0145180</a>
</li>
<li><a class="reference external" href="http://pyproj4.github.io/pyproj/stable/">pyproj</a>
</li>
<li>
<a href="https://joblib.readthedocs.io/en/latest/">joblib</a>
</li>
</ul>
</p>
<p><strong>EMagPy's core developers: Guillaume Blanchy and Paul McLachlan.<strong></p>
<p>Contributors: Jimmy Boyd</p>
'''.format(EMagPy_version))
#<p><b>Citing ResIPy</b>:<br>Blanchy G., Saneiyan S., Boyd J., McLachlan P. and Binley A. 2020.<br>“ResIPy, an Intuitive Open Source Software for Complex Geoelectrical Inversion/Modeling.”<br>Computers & Geosciences, February, 104423. <a href="https://doi.org/10.1016/j.cageo.2020.104423">https://doi.org/10.1016/j.cageo.2020.104423</a>.</p>

        aboutText.setOpenExternalLinks(True)
        aboutText.setWordWrap(True)
        aboutText.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        infoLayout.addWidget(aboutText)

        tabAbout.setLayout(infoLayout)
        
        
        #%% general Ctrl+Q shortcut + general tab layout

        self.layout.addWidget(self.tabs)
        self.layout.addWidget(self.errorLabel)
        self.table_widget.setLayout(self.layout)
        self.setCentralWidget(self.table_widget)
        self.show()
    
    def projBtnFunc(self):
        val = float(self.projEdit.text()) if self.projEdit.text() != '' else None
        self.problem.convertFromNMEA(targetProjection='EPSG:{:.0f}'.format(val))
        self.replot()

    def replot(self):
        index = self.showParams['index']
        coil = self.showParams['coil']
        contour = self.showParams['contour']
        vmin = self.showParams['vmin']
        vmax = self.showParams['vmax']
        pts = self.showParams['pts']
        cmap = self.showParams['cmap']
        if self.mapRadio.isChecked():
            self.mwRaw.replot(index=index, coil=coil, contour=contour,
                              vmin=vmin, vmax=vmax, pts=pts, cmap=cmap)
        else:
            self.mwRaw.replot(index=index, coil=coil, vmin=vmin, vmax=vmax)
                
    def processFname(self, fnames):
        self.problem.surveys = [] # empty the list of current survey
        if len(fnames) == 1:
            fname = fnames[0]
            self.importBtn.setText(os.path.basename(fname))
            self.problem.createSurvey(fname)
            self.gammaEdit.setVisible(False)
            self.gammaLabel.setVisible(False)
        else:
            self.importBtn.setText(os.path.basename(fnames[0]) + ' .. '
                                   + os.path.basename(fnames[-1]))
            self.gammaEdit.setVisible(True)
            self.gammaLabel.setVisible(True)
            self.problem.createTimeLapseSurvey(fnames)
        self.infoDump('Files well imported')
        self.setupUI()
        
    def setupUI(self):
        self.mwRaw.setCallback(self.problem.show)
        self.mwRaw.replot()
        
        # fill the combobox with survey and coil names
        self.coilErrCombo.clear()
        self.coilCombo.clear()
        for coil in self.problem.coils:
            self.coilCombo.addItem(coil)
            self.coilErrCombo.addItem(coil)
        self.coilCombo.addItem('all')
        self.coilCombo.setCurrentIndex(len(self.problem.coils))
        self.surveyCombo.clear()
        self.surveyInvCombo.clear()
        self.surveyInvMapCombo.clear()
        for survey in self.problem.surveys:
            self.surveyCombo.addItem(survey.name)
            self.surveyErrCombo.addItem(survey.name)
            self.surveyInvCombo.addItem(survey.name)
            self.surveyInvMapCombo.addItem(survey.name)
        
        # set to default values
        self.showRadio.setChecked(True)
        self.contourCheck.setChecked(False)

        # enable widgets
        if 'Latitude' in survey.df.columns:
            self.projBtn.setEnabled(True)
            self.projEdit.setEnabled(True)
            self.projBtnFunc() # automatically convert NMEA string
        self.keepApplyBtn.setEnabled(True)
        self.rollingBtn.setEnabled(True)
        self.ptsKillerBtn.setEnabled(True)
        self.coilCombo.setEnabled(True)
        self.surveyCombo.setEnabled(True)
        self.showRadio.setEnabled(True)
        self.mapRadio.setEnabled(True)
        self.applyBtn.setEnabled(True)
        self.cmapCombo.setEnabled(True)
        self.contourCheck.setEnabled(True)
        self.ptsCheck.setEnabled(True)
    
            
    def errorDump(self, text, flag=1):
        text = str(text)
        timeStamp = time.strftime('%H:%M:%S')
        if flag == 1: # error in red
            col = 'red'
        else:
            col = 'black'
        self.errorLabel.setText('<i style="color:'+col+'">['+timeStamp+']: '+text+'</i>')

    def infoDump(self, text):
        self.errorDump(text, flag=0)

    def keyPressEvent(self, e):
        if (e.modifiers() == Qt.ControlModifier) & (e.key() == Qt.Key_Q):
            self.close()
        if (e.modifiers() == Qt.ControlModifier) & (e.key() == Qt.Key_W):
            self.close()

#%% updater function and wine check
    # based on https://kushaldas.in/posts/pyqt5-thread-example.html
#    def updateChecker(self): # check for new updates on gitlab
#        version = ResIPy_version
#        try:
#            versionSource = urlRequest.urlopen('https://gitlab.com/hkex/pyr2/raw/master/src/version.txt?inline=false')
#            versionCheck = versionSource.read().decode()
#            version = versionCheck.split()[1] # assuming version number is in 2nd line of version.txt
#            print('online version :', version)
#        except:
#            pass
#        return version
#    
#    def updateCheckerShow(self, version):
#        if ResIPy_version != version:
#            msg = QMessageBox()
#            msg.setIcon(QMessageBox.Information)
#            msg.setText('''<b>ResIPy version %s is available</b>''' % (version))
#            msg.setInformativeText('''Please download the latest version of ResIPy at:<p><a href='https://gitlab.com/hkex/pyr2#gui-for-r2-family-code'>https://gitlab.com/hkex/pyr2</a></p><br>''')
#            msg.setWindowTitle("New version available")
#            bttnUpY = msg.addButton(QMessageBox.Yes)
#            bttnUpY.setText('Update')
#            bttnUpN = msg.addButton(QMessageBox.No)
#            bttnUpN.setText('Ignore')
#            msg.setDefaultButton(bttnUpY)
#            msg.exec_()
#            if msg.clickedButton() == bttnUpY:
#                webbrowser.open('https://gitlab.com/hkex/pyr2#gui-for-r2-family-code') # can add download link, when we have a direct dl link
#    


#%% main function
if __name__ == '__main__':
    catchErrors() # prevent crash of the app
    freeze_support()
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setWindowIcon(QIcon(os.path.join(bundle_dir, 'logo.png'))) # that's the true app icon
    
    # initiate splash screen when loading libraries    
    splash_pix = QPixmap(os.path.join(bundle_dir, 'loadingLogo.jpg'))
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
    splash.setEnabled(False)
    progressBar = QProgressBar(splash)
    progressBar.setMaximum(10)
    progressBar.setGeometry(100, splash_pix.height() - 50, splash_pix.width() - 200, 20)
    splash.show()
    splash.showMessage("Loading libraries", Qt.AlignBottom | Qt.AlignCenter, Qt.black)
    app.processEvents()
    progressBar.setValue(1)
    app.processEvents()

    # in this section all import are made except the one for pyQt
#    print('importing matplotlib')
#    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
#    from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
#    from matplotlib.figure import Figure
#    from matplotlib import rcParams
#    rcParams.update({'font.size': 12}) # CHANGE HERE for graph font size
    progressBar.setValue(2)
    app.processEvents()

    # library needed for update checker + wine checker
    print('other needed libraries')
    import platform
    OS = platform.system()
#    from urllib import request as urlRequest
#    import webbrowser

    print('import emagpy')
#    from resipy.R2 import R2
#    from resipy.r2help import r2help
#    splash.showMessage("ResIPy is ready!", Qt.AlignBottom | Qt.AlignCenter, Qt.black)
    progressBar.setValue(10)
    app.processEvents()

    ex = App()
    splash.hide() # hiding the splash screen when finished
    
    sys.exit(app.exec_())
