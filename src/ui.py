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
print(EMagPy_version)
import numpy as np
import pandas as pd
from multiprocessing import freeze_support

from PyQt5.QtWidgets import (QMainWindow, QSplashScreen, QApplication, QPushButton, QWidget,
    QTabWidget, QVBoxLayout, QLabel, QLineEdit, QMessageBox, QCompleter,
    QFileDialog, QCheckBox, QComboBox, QTextEdit, QHBoxLayout, QTextBrowser,
    QTableWidget, QFormLayout, QTableWidgetItem, QHeaderView, QProgressBar,
    QStackedLayout, QGroupBox, QFrame, QMenu, QAction)#, QRadioButton, QListWidget, QShortcut)
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

try:
    import pyvista as pv
    try:
        from pyvistaqt import QtInteractor # newer version
    except:
        from pyvista import QtInteractor # older version
    pvfound = True
except:
    pvfound = False
    print('WARNING: pyvista not found, 3D plotting will be disabled.')


# debug options
DEBUG = True # set to false to not display message in the console
def pdebug(*args, **kwargs):
    if DEBUG:
        print('DEBUG:', *args, **kwargs)
    else:
        pass


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
        self.fnameEC = None
        self.fnameResMod = None
        self.running = False # True when inverison is running
        self.apiLog = '' # store log of API call
        self.meshType = None # 'tri' or 'quad' if specified
        
        self.errorLabel = QLabel('<i style="color:black">Error messages will be displayed here</i>')
        QApplication.processEvents()

        self.table_widget = QWidget()
        self.layout = QVBoxLayout()
        self.tabs = QTabWidget()
        
        self.problem = Problem()
        self.problem.runningUI = True
        self.writeLog('# ======= EMagPy API log ======')
        self.writeLog('# EMagPy version: ' + EMagPy_version)
        self.writeLog('from emagpy import Problem')
        self.writeLog('k = Problem()')
        
        self.optBtn = QPushButton('Options')
        self.menu = QMenu()
        self.menu.addAction('Save API log', self.saveLog)
        self.menu.addAction('Save processed data', self.saveData)
        self.optBtn.setMenu(self.menu)
        self.tabs.setCornerWidget(self.optBtn, Qt.TopRightCorner)
        
        
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
                    self.writeLog('k.importModel("{:s}")'.format(fname))
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
            if len(self.problem.models) == 0:
                self.errorDump('Generate a model first')
                return
            coils = self.coilTable.getTable()
            forwardModel = fforwardModels[self.fforwardCombo.currentIndex()]
            noise = float(self.fnoise.text())/100
            self.problem.forward(forwardModel, coils=coils, noise=noise)
            self.writeLog('k.forward("{:s}", coils={:s}, noise={:.2f}'.format(
                forwardModel, str(coils), noise))
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
            if index == 1 or index == 2 or index == 3:
                showGF(True)
            else:
                showGF(False)
        self.sensorCombo = QComboBox()
        sensors = ['CMD Mini-Explorer',
                   'CMD Explorer',
                   'CMD Mini-Explorer 6L'
                   ]
        sensors = sorted(sensors)
        sensors = ['All'] + sensors
        for sensor in sensors:
            self.sensorCombo.addItem(sensor)
        self.sensorCombo.activated.connect(sensorComboFunc)
        
        
        # import data
        self.mergedCheck = QCheckBox('Merge surveys')
        self.mergedCheck.setToolTip('Check to merge all files in a single survey.')
        
        def importBtnFunc():
            self._dialog = QFileDialog()
            fnames, _ = self._dialog.getOpenFileNames(importTab, 'Select data file(s)', self.datadir, '*.csv *.CSV')
            if len(fnames) > 0:
                self.processFname(fnames, merged=self.mergedCheck.isChecked())
        self.importBtn = QPushButton('Import Dataset(s)')
        self.importBtn.setAutoDefault(True)
        self.importBtn.setStyleSheet('background-color:orange')
        self.importBtn.clicked.connect(importBtnFunc)
        
        def importGFLoFunc():
            fname, _ = QFileDialog.getOpenFileName(importTab, 'Select data file', self.datadir, '*.dat *.DAT')
            if fname != '':
                self.fnameLo = fname
                self.importGFLo.setText(os.path.basename(fname))
        self.importGFLo = QPushButton('Select Lo')
        self.importGFLo.clicked.connect(importGFLoFunc)
        
        def importGFHiFunc():
            fname, _ = QFileDialog.getOpenFileName(importTab, 'Select data file', self.datadir, '*.dat *.DAT')
            if fname != '':
                self.fnameHi = fname
                self.importGFHi.setText(os.path.basename(fname))
        self.importGFHi = QPushButton('Select Hi')
        self.importGFHi.clicked.connect(importGFHiFunc)
        
        self.hxLabel = QLabel('Height [m]:')
        
        self.hxEdit = QLineEdit('0')
        self.hxEdit.setValidator(QDoubleValidator())
        self.hxEdit.setToolTip('Height above the ground [m]')
        
        def importGFApplyFunc():
            hx = float(self.hxEdit.text()) if self.hxEdit.text() != '' else 0
            device = self.sensorCombo.itemText(self.sensorCombo.currentIndex())
            if self.fnameLo is None and self.fnameHi is None:
                self.errorDump('Specify at least one file to import (Lo and/or Hi)')
                return
            self.problem.surveys = []  # remove all previous surveys
            self.problem.importGF(self.fnameLo, self.fnameHi, device, hx)
            if self.fnameLo is not None and self.fnameHi is not None:
                self.writeLog('k.importGF("{:s}", "{:s}", "device={:s}", hx={:.2f})'.format(
	        self.fnameLo, self.fnameHi, device, hx))
            if self.fnameLo is not None:
                self.writeLog('k.importGF("{:s}", "device={:s}", hx={:.2f})'.format(
	        self.fnameLo, device, hx))
            if self.fnameHi is not None:
                self.writeLog('k.importGF("{:s}", "device={:s}", hx={:.2f})'.format(
	        self.fnameHi, device, hx))
            self.infoDump('Surveys well imported')
            self.setupUI()
        self.importGFApply = QPushButton('Import')
        self.importGFApply.setStyleSheet('background-color: orange')
        self.importGFApply.clicked.connect(importGFApplyFunc)
        
        self.gfCalibCombo = QComboBox()
        self.gfCalibCombo.addItem('F-0m')
        self.gfCalibCombo.addItem('F-1m')
        self.gfCalibCombo.setToolTip('Select calibration used.')
        self.gfCalibCombo.setVisible(False)
        
        def gfCorrectionBtnFunc():
            self.problem.gfCorrection(calib=self.gfCalibCombo.currentText())
            self.writeLog('k.gfCorrection(calib="{:s}")'.format(
                self.gfCalibCombo.currentText()))
            self.replot()
        self.gfCorrectionBtn = QPushButton('Convert to LIN ECa')
        self.gfCorrectionBtn.clicked.connect(gfCorrectionBtnFunc)
        self.gfCorrectionBtn.setToolTip('GF Instruments output calibrated ECa values.'
                                        'However, LIN ECa values are need for inversion.'
                                        'This convert the calibrated ECa to LIN ECa, '
                                        'usually making them smaller.')
        self.gfCorrectionBtn.setVisible(False)
        
        def showGF(arg):
            visibles = np.array([True, True, False, False, False, False, False, False, False])
            objs = [self.importBtn, self.mergedCheck, self.importGFLo, self.importGFHi,
                    self.gfCalibCombo, self.gfCorrectionBtn,
                    self.hxLabel, self.hxEdit, self.importGFApply]
            if arg is True:
                [o.setVisible(v) for o,v in zip(objs, ~visibles)]
            else:
                [o.setVisible(v) for o,v in zip(objs, visibles)]
        showGF(False)

        
        # projection (only if GPS data are available)
        self.projLabel = QLabel('Map CRS:')
        self.projLabel.setToolTip('Project columns "Latitude" and "Longitude" to specified coordinate system (CRS)')
        self.projLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        ### Preparing the ~5000 projections:
        self.pcs = pd.read_csv(resource_path('emagpy/pcs.csv'))
        pcs_names = self.pcs['COORD_REF_SYS_NAME'].tolist()
        pcs_names.extend(self.pcs['COORD_REF_SYS_NAME_rev'].tolist())
        self.pcsCompleter = QCompleter(pcs_names)
        self.pcsCompleter.setCaseSensitivity(Qt.CaseInsensitive)
        
        self.projEdit = QLineEdit()
        self.projEdit.setPlaceholderText('Type projection CRS')
        self.projEdit.setToolTip('Type the CRS projection and then select from the options\nDefault is British National Grid / OSGB 1936')
        # self.projEdit.setValidator(QDoubleValidator())
        self.projEdit.setCompleter(self.pcsCompleter)
        # self.projEdit.setEnabled(False)

        self.projBtn = QPushButton('Apply CRS')
        self.projBtn.setToolTip('Convert NMEA/DMS string or decimal degree to selected coordinate system (CRS) - select a CRS first')
        self.projBtn.clicked.connect(self.projBtnFunc)
        self.projBtn.setEnabled(False)
        
        
        # filtering options
        self.filtLabel = QLabel('Filter Options |')
        self.filtLabel.setStyleSheet('font-weight:bold')

        
        # display options
        self.showParams = {'index': 0, 'coil':'all', 'contour':False, 'vmin':None,
                           'vmax':None,'pts':False, 'cmap':'viridis_r', 'dist':True} 

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
            self.writeLog('k.filterRange(vmin={:s}, vmax={:s})'.format(
                str(vmin), str(vmax)))
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
            self.writeLog('k.rollingMean(window={:d}'.format(window))
            self.replot()
        self.rollingBtn = QPushButton('Rolling Mean')
        self.rollingBtn.clicked.connect(rollingBtnFunc)
        self.rollingBtn.setEnabled(False)
        self.rollingBtn.setAutoDefault(True)
        
        # manual point killer selection
        def ptsKillerBtnFunc():
            self.problem.surveys[self.showParams['index']].dropSelected()
            self.replot()
        self.ptsKillerBtn = QPushButton('Delete points')
        self.ptsKillerBtn.setToolTip('Click on points in the figure then click to delete them.')
        self.ptsKillerBtn.clicked.connect(ptsKillerBtnFunc)
        self.ptsKillerBtn.setEnabled(False)
        self.ptsKillerBtn.setAutoDefault(True)
        
        def gridBtnFunc():
            nx = int(self.gridx.text())
            ny = int(self.gridy.text())
            self.problem.gridData(nx=nx, ny=ny)
            self.writeLog('k.gridData(nx={:d}, ny={:d}'.format(nx, ny))
            self.replot()
        self.gridBtn = QPushButton('Grid Data')
        self.gridBtn.setEnabled(False)
        self.gridBtn.clicked.connect(gridBtnFunc)
        
        self.gridxLabel = QLabel('nx:')
        self.gridx = QLineEdit('100')
        self.gridx.setValidator(QIntValidator())
        self.gridx.setToolTip('Grid size in X direction.')
        
        self.gridyLabel = QLabel('ny:')
        self.gridy = QLineEdit('100')
        self.gridy.setValidator(QIntValidator())
        self.gridy.setToolTip('Grid size in Y direction.')
        
        
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
        
        # show dots, map or pseudo
        def showComboFunc(index):
            if self.showCombo.currentText() == 'Dots':
                showMapOptions(False)
                self.mwRaw.setCallback(self.problem.show)
                self.replot()
            elif self.showCombo.currentText() == 'Pseudo':
                showMapOptions(False)
                self.mwRaw.setCallback(self.problem.showPseudo)
                self.replot()                
            elif self.showCombo.currentText() == 'Map':
                showMapOptions(True)
                self.mwRaw.setCallback(self.problem.showMap)
                if self.showParams['coil'] == 'all':
                    self.coilCombo.setCurrentIndex(0)
                    coilComboFunc(0)
                else:
                    self.replot()
        self.showCombo = QComboBox()
        self.showCombo.addItem('Dots')
        self.showCombo.addItem('Pseudo')
        self.showCombo.addItem('Map')
        self.showCombo.activated.connect(showComboFunc)
        self.showCombo.setEnabled(False)
        
        def showMapOptions(arg):
            objs = [self.ptsLabel, self.ptsCheck, self.contourLabel,
                    self.contourCheck, self.cmapCombo, self.psMapExpBtn]
            [o.setVisible(arg) for o in objs]
            if arg is False:
                self.coilCombo.addItem('all')
                self.xlabCombo.setVisible(True)
            else:
                self.xlabCombo.setVisible(False)
                n = len(self.problem.coils)
                if self.coilCombo.currentIndex() == n:
                    self.coilCombo.setCurrentIndex(n-1)
                self.coilCombo.removeItem(n)
                
            # print([self.coilCombo.itemText(i) for i in range(self.coilCombo.count())])

        def xlabComboFunc(index):
            if index == 0:
                self.showParams['dist'] = True
            else:
                self.showParams['dist'] = False
            self.replot()
        self.xlabCombo = QComboBox()
        self.xlabCombo.addItem('Distance')
        self.xlabCombo.addItem('Samples')
        self.xlabCombo.currentIndexChanged.connect(xlabComboFunc)
        self.xlabCombo.setEnabled(False)
        
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
            self.showParams['cmap'] = psMapCmaps[index]
            self.replot()
        self.cmapCombo = QComboBox()
        psMapCmaps = ['viridis', 'viridis_r', 'Greys', 'seismic', 'rainbow', 'jet','jet_r', 'turbo']

        self.cmapCombo.addItems(psMapCmaps)
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
        def expPsMap():
            if self.projEdit.text() == '':
                self.errorDump('First specify a coordinate system (CRS)')
                return
            fname, _ = QFileDialog.getSaveFileName(importTab,'Export raster map', self.datadir, 'TIFF (*.tif)')
            if fname != '':
                self.setProjection()
                self.problem.saveMap(fname=fname, cmap=self.cmapCombo.currentText())
                self.writeLog('k.saveMap(fname="{:s}", cmap="{:s}")'.format(
                    fname, self.cmapCombo.currentText()))
                self.infoDump('File saved in ' + fname)
            
        self.psMapExpBtn = QPushButton('Export GIS')
        self.psMapExpBtn.setToolTip('Export a georeferenced TIFF file to directly be imported in GIS software.\n'
                                    'Choose the correct EPSG CRS projection!')
        self.psMapExpBtn.setVisible(False)
        self.psMapExpBtn.clicked.connect(expPsMap)

        # display it
        self.mwRaw = MatplotlibWidget()
        
        
        # 3D viewing tab
        
        
        
        # layout
        importLayout = QVBoxLayout()
        
        topLayout = QHBoxLayout()
        topLayout.addWidget(self.sensorCombo, 10)
        topLayout.addWidget(self.mergedCheck, 5)
        topLayout.addWidget(self.importBtn, 10)
        topLayout.addWidget(self.importGFHi, 10)
        topLayout.addWidget(self.importGFLo, 10)
        topLayout.addWidget(self.hxLabel, 5)
        topLayout.addWidget(self.hxEdit, 5)
        topLayout.addWidget(self.importGFApply, 10)
        topLayout.addWidget(self.gfCalibCombo, 5)
        topLayout.addWidget(self.gfCorrectionBtn, 10)
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
        filtLayout.addWidget(self.gridBtn)
        filtLayout.addWidget(self.gridxLabel)
        filtLayout.addWidget(self.gridx)
        filtLayout.addWidget(self.gridyLabel)
        filtLayout.addWidget(self.gridy)
    
        midLayout = QHBoxLayout()
        midLayout.addWidget(self.displayLabel)
        midLayout.addWidget(self.surveyCombo, 7)
        midLayout.addWidget(QLabel('Select coil:'))
        midLayout.addWidget(self.coilCombo, 7)
        # midLayout.addWidget(showGroup)
        midLayout.addWidget(self.showCombo)
        midLayout.addWidget(self.xlabCombo)
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
                self.fnameResMod = None
                self.ecImportBtn.setText(os.path.basename(fname))
                self.ertImportBtn.setText('Import ERT model')
        self.ecImportBtn = QPushButton('Import EC profiles')
        self.ecImportBtn.clicked.connect(ecImportBtnFunc)


        # import Resistivity model, format should be x, z, resistivity with space delimeter, e.g. R2 'f001_res.dat' format
        def ertImportBtnFunc():
            fname, _ = QFileDialog.getOpenFileName(importTab, 'Select data file', self.datadir, '*.dat')
            if fname != '':
                self.fnameResMod = fname
                self.fnameEC = None
                self.ecImportBtn.setText('Import EC profiles')
                self.ertImportBtn.setText(os.path.basename(fname))
        self.ertImportBtn = QPushButton('Import ERT model')
        self.ertImportBtn.clicked.connect(ertImportBtnFunc)
        self.meshTypeBtn = QComboBox()
        self.meshTypeBtn.addItem('Quadrilateral ERT Mesh')
        self.meshTypeBtn.addItem('Triangular ERT Mesh')
        self.meshTypeBtn.setToolTip('Select whether ERT model used is quadrilateral or triangular')
        
        # GF calibration for the calibration dataset
        self.gfCalibCalibCombo = QComboBox()
        self.gfCalibCalibCombo.addItem('None')
        self.gfCalibCalibCombo.addItem('F-0m')
        self.gfCalibCalibCombo.addItem('F-1m')
        self.gfCalibCalibCombo.setToolTip('Select GF calibration used if any.'
                                          'If selected, it should be the same'
                                          ' as the one used for the main dataset.')
        
        # choose which forward model to use
        self.forwardCalibCombo = QComboBox()
        forwardCalibs = ['CS', 'FSlin', 'FSeq']
        for forwardCalib in forwardCalibs:
            self.forwardCalibCombo.addItem(forwardCalib)
        
        # perform the fit (equations display in the console)
        def fitCalibBtnFunc():
            if self.meshTypeBtn.currentText() == 'Quadrilateral ERT Mesh':
                self.meshType = 'quad'
            if self.meshTypeBtn.currentText() == 'Triangular ERT Mesh':
                self.meshType = 'tri'
            forwardModel = self.forwardCalibCombo.itemText(self.forwardCalibCombo.currentIndex())
            calib = self.gfCalibCalibCombo.currentText() if self.gfCalibCalibCombo.currentText() != 'None' else None 
            self.mwCalib.setCallback(self.problem.calibrate)
            self.mwCalib.replot(fnameECa=self.fnameECa, fnameEC=self.fnameEC,
                                fnameResMod=self.fnameResMod, calib=calib, meshType=self.meshType,
                           forwardModel=forwardModel)
            self.writeLog('k.calibrate(fnameECa="{:s}", fnameEC="{:s}",'
                          'fnameResMod="{:s}", calib="{:s}", meshType="{:s}",'
                          'forwardModel="{:s}")'.format(self.fnameECa, str(self.fnameEC),
                                                        str(self.fnameResMod), str(calib), str(self.meshType),
                                                        forwardModel))
        self.fitCalibBtn = QPushButton('Fit calibration')
        self.fitCalibBtn.clicked.connect(fitCalibBtnFunc)
        
        # apply the calibration to the ECa measurements of the survey imported
        def applyCalibBtnFunc():
            forwardModel = self.forwardCalibCombo.itemText(self.forwardCalibCombo.currentIndex())
            calib = self.gfCalibCalibCombo.currentText() if self.gfCalibCalibCombo.currentText() != 'None' else None 
            self.mwCalib.replot(fnameECa=self.fnameECa, fnameEC=self.fnameEC,  meshType=self.meshType,
                                fnameResMod=self.fnameResMod, calib=calib,
                           forwardModel=forwardModel, apply=True)
            self.writeLog('k.calibrate(fnameECa="{:s}", fnameEC="{:s}",'
                          'fnameResMod="{:s}", calib="{:s}", meshType="{:s}",'
                          'forwardModel="{:s}", apply=True)'.format(self.fnameECa, str(self.fnameEC),
                                                        str(self.fnameResMod), str(calib), str(self.meshType),
                                                        forwardModel))
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
        calibOptions.addWidget(self.meshTypeBtn)
        calibOptions.addWidget(self.gfCalibCalibCombo)
        calibOptions.addWidget(self.forwardCalibCombo)
        calibOptions.addWidget(self.fitCalibBtn)
        calibOptions.addWidget(self.applyCalibBtn)
        calibLayout.addLayout(calibOptions)
        calibLayout.addWidget(self.mwCalib)
        
        calibTab.setLayout(calibLayout)
        
        
        #%% error model
        errTab = QTabWidget()
        self.tabs.addTab(errTab, 'Error Modelling')
        
        self.errLabel = QLabel('EXPERIMENTAL: this tab helps to fit an '
                               'error model based on cross-over measurements. '
                               'It helps estimate the amount of error on '
                               'the data. Currently, the error model is not used by the '
                               'inversion.')
        self.errLabel.setWordWrap(True)
        
        self.surveyErrCombo = QComboBox()
        self.coilErrCombo = QComboBox()
        
        def fitErrBtnFunc():
            index = self.surveyErrCombo.currentIndex()
            coil = self.coilErrCombo.itemText(self.coilErrCombo.currentIndex())
            self.mwErr.setCallback(self.problem.crossOverPointsError)
            self.mwErr.replot(index=index, coil=coil, dump=self.infoDump)
            self.writeLog('k.crossOverPointsError(index={:d}, coil="{:s}")'.format(
                index, coil))
            self.mwErrMap.setCallback(self.problem.plotCrossOverMap)
            self.mwErrMap.replot(index=index, coil=coil)
            self.writeLog('k.plotCrossOverMap(index={:d}, coil="{:s}")'.format(
                index, coil))
        self.fitErrBtn = QPushButton('Fit Error Model based on colocated measurements')
        self.fitErrBtn.clicked.connect(fitErrBtnFunc)
        
        
        # graph
        self.mwErr = MatplotlibWidget()
        self.mwErrMap = MatplotlibWidget()
        
        # layout
        errLayout = QVBoxLayout()
        errLayout.addWidget(self.errLabel)
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
            
        self.icustomModel = False
        
        def importModelBtnFunc():
            self._dialog = QFileDialog()
            fname, _ = self._dialog.getOpenFileName(settingsTab, 'Select data file', self.datadir, '*.csv')
            if fname != '':
                # try:
                self.problem.importModel(fname)
                self.writeLog('k.importModel("{:s}"'.format(fname))
                nlayer = self.problem.models[0].shape[1]
                self.modelTable.setTable([-1]*(nlayer-1),[-1]*nlayer)
                self.createModelBtn.setEnabled(False)
                self.addRowBtn.setEnabled(False)
                self.delRowBtn.setEnabled(False)
                self.nLayerEdit.setEnabled(False)
                self.startingEdit.setEnabled(False)
                self.thicknessEdit.setEnabled(False)
                self.icustomModel = True
                self.importModelBtn.setText(os.path.basename(fname))
                self.infoDump('Model successfully set.')
                # except Exception as e:
                #     print(e)
                #     self.errorDump('Error in reading file. Please check format.')
        self.importModelBtn = QPushButton('Import Model')
        self.importModelBtn.clicked.connect(importModelBtnFunc)
        self.importModelBtn.setToolTip('File needs to be .csv with layer1, layer2, depth1, ... columns.\n'
                                   'layerX contains EC in mS/m and depthX contains depth of the bottom\n'
                                   'of the layer in meters (positively defined)')
        
        def clearModelBtnFunc():
            self.createModelBtn.setEnabled(True)
            self.addRowBtn.setEnabled(True)
            self.delRowBtn.setEnabled(True)
            self.nLayerEdit.setEnabled(True)
            self.startingEdit.setEnabled(True)
            self.thicknessEdit.setEnabled(True)
            self.icustomModel = False
            self.importModelBtn.setText('Import Model')
        self.clearModelBtn = QPushButton('Clear Model')
        self.clearModelBtn.clicked.connect(clearModelBtnFunc)
        
        self.modelLabel = QLabel('Bottom depth and starting EC of each layer.')
        
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
            self.writeLog('k.lcurve()')
        self.lcurveBtn = QPushButton('Fit L-curve')
        self.lcurveBtn.clicked.connect(lcurveBtnFunc)
        
        # graph
        self.mwlcurve = MatplotlibWidget()
        
        
        # layout
        settingsLayout = QHBoxLayout()
        
        invStartLayout = QVBoxLayout()
        importLayout = QHBoxLayout()
        importLayout.addWidget(self.importModelBtn)
        importLayout.addWidget(self.clearModelBtn)
        invStartLayout.addLayout(importLayout)
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
        
        self.forwardCombo = QComboBox()
        forwardModels = ['CS','FSlin', 'FSeq', 'Q']
        for forwardModel in forwardModels:
            self.forwardCombo.addItem(forwardModel)
        self.forwardCombo.setToolTip('''Choice of forward model:
        CS : Cumulative Sensitivity Function
        FSlin : Full solution with LIN conversion
        FSeq : Full solution with equivalent homogenous conductivity conversion
        Q : Full solution without use of ECa values''')

        def methodComboFunc(index):
            objs1 = [self.alphaLabel, self.alphaEdit,
                    self.betaLabel, self.betaEdit,
                    self.gammaLabel, self.gammaEdit,
                    self.nitLabel, self.nitEdit,
                    self.parallelCheck]
            objs2 = [self.annSampleLabel, self.annSampleEdit,
                     self.annNoiseLabel, self.annNoiseEdit]
            if self.methodCombo.currentText() == 'ANN':
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
            if self.methodCombo.currentText() == 'Gauss-Newton':
                self.betaEdit.setEnabled(False)
                self.lCombo.setEnabled(False)
                self.parallelCheck.setEnabled(False)
                self.nitEdit.setEnabled(False)
            else:
                self.betaEdit.setEnabled(True)
                self.lCombo.setEnabled(True)
                self.parallelCheck.setEnabled(False) # TODO
                self.nitEdit.setEnabled(True)
                
        self.methodCombo = QComboBox()
        self.methodCombo.setToolTip('''Choice of solver:
        L-BFGS-B : minimize, fast and flexible
        Gauss-Newton : fast, no support for variable depth
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
        methods = mMinimize + ['Gauss-Newton'] + mMCMC + ['ANN']
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
        if frozen != 'not':
            self.parallelCheck.setEnabled(False) # TODO on available on unfrozen version
        
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
        self.logText = QTextEdit('Hello there!')
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
                self.running = False
                self.invertBtn.setText('Invert')
                self.invertBtn.setStyleSheet('background-color:green')
                return
            outputStack.setCurrentIndex(0)
            self.logText.clear()

            # collect parameters
            depths0, conds0, fixedDepths, fixedConds = self.modelTable.getTable()
            if self.icustomModel:
                depths0 = self.problem.depths[0]
                conds0 = self.problem.models[0]
            self.problem.setInit(depths0, conds0, fixedDepths, fixedConds)
            self.writeLog('k.setInit(depths0={:s}, conds0={:s}, fixedDepths={:s}, fixedConds={:s})'.format(
                str(list(depths0)), str(list(conds0)),
                str(list(fixedDepths)), str(list(fixedConds))))
            regularization = self.lCombo.itemText(self.lCombo.currentIndex())
            alpha = float(self.alphaEdit.text()) if self.alphaEdit.text() != '' else 0.07
            forwardModel = self.forwardCombo.itemText(self.forwardCombo.currentIndex())
            method = self.methodCombo.itemText(self.methodCombo.currentIndex())
            beta = float(self.betaEdit.text()) if self.betaEdit.text() != '' else 0.0
            gamma = float(self.gammaEdit.text()) if self.gammaEdit.text() != '' else 0.0
            nit = int(self.nitEdit.text()) if self.nitEdit.text() != '' else 15
            nsample = int(self.annSampleEdit.text()) if self.annSampleEdit.text() != '' else 100
            noise = float(self.annNoiseEdit.text()) if self.annNoiseEdit.text() != '' else 0
            njobs = -1 if self.parallelCheck.isChecked() else 1
            if self.icustomModel:
                depths = np.arange(depths0.shape[1]+2)
            else:
                depths = np.r_[[0], depths0, [-np.inf]]
            self.sliceCombo.clear()
            for i in range(len(depths)-1):
                if self.icustomModel | (np.sum(~fixedConds) > 0):
                    self.sliceCombo.addItem('Layer {:d}'.format(i+1))
                else:
                    self.sliceCombo.addItem('{:.2f}m - {:.2f}m'.format(depths[i], depths[i+1]))
            self.sliceCombo.activated.connect(sliceComboFunc)
            
            # invert
            if forwardModel == 'CS (fast)':
                self.problem.invertGN(alpha=alpha, dump=logTextFunc)
                self.writeLog('k.invertGF(alpha={:s})'.format(str(alpha)))
            else:
                self.problem.invert(forwardModel=forwardModel, alpha=alpha,
                                    dump=logTextFunc, regularization=regularization,
                                    method=method, options={'maxiter':nit},
                                    beta=beta, gamma=gamma, nsample=nsample,
                                    noise=noise/100, njobs=njobs)
                self.writeLog('k.invert(forwardModel="{:s}", alpha={:s}, '
                              'regularization="{:s}", method="{:s}", '
                              'options={{"maxiter":{:d}}}, beta={:s}, gamma={:s}'
                              ', nsample={:s}, noise={:.2f}, njobs={:d})'.format(
                        forwardModel, str(alpha), regularization, method, nit,
                        str(beta), str(gamma), str(nsample), noise/100, njobs))
            
            # plot results
            if self.problem.ikill == False: # program wasn't killed
                self.mwInv.setCallback(self.problem.showResults)
                self.mwInv.replot(**showInvParams)
                self.writeLog('k.showResults', showInvParams)
                self.mwInvMap.setCallback(self.problem.showSlice)
                self.mwInvMap.replot(**showInvMapParams)
                self.writeLog('k.showSlice', showInvMapParams)
                self.mwMisfit.plot(self.problem.showMisfit)
                self.writeLog('k.showMisfit()')
                self.mwOne2One.plot(self.problem.showOne2one)
                self.writeLog('k.showOne2one()')
                if pvfound:
                    replot3D()
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
                         'cmap':'viridis_r', 'contour':False, 'dist':True}
        
        def cmapInvComboFunc(index):
            showInvParams['cmap'] = self.cmapInvCombo.itemText(index)
            self.mwInv.replot(**showInvParams)
        self.cmapInvCombo = QComboBox()
        invCmaps = ['viridis_r', 'viridis', 'Greys', 'seismic', 'rainbow', 'jet', 'jet_r', 'turbo']
        self.cmapInvCombo.addItems(invCmaps)
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
        
        # apply display settings
        def applyInvBtnFunc():
            vmin = float(self.vminInvEdit.text()) if self.vminInvEdit.text() != '' else None
            vmax = float(self.vmaxInvEdit.text()) if self.vmaxInvEdit.text() != '' else None
            showInvParams['vmin'] = vmin
            showInvParams['vmax'] = vmax
            self.mwInv.replot(**showInvParams)
        self.applyInvBtn = QPushButton('Apply')
        self.applyInvBtn.clicked.connect(applyInvBtnFunc)
        
        # contour the plot
        self.contourInvLabel = QLabel('Contour:')
        def contourInvCheckFunc(state):
            showInvParams['contour'] = state
            self.mwInv.replot(**showInvParams)
        self.contourInvCheck = QCheckBox()
        self.contourInvCheck.clicked.connect(contourInvCheckFunc)
        
        # change x axis from distance to samples
        def xlabInvComboFunc(index):
            if index == 0:
                showInvParams['dist'] = True
            else:
                showInvParams['dist'] = False
            self.mwInv.replot(**showInvParams)
        self.xlabInvCombo = QComboBox()
        self.xlabInvCombo.addItem('Distance')
        self.xlabInvCombo.addItem('Samples')
        self.xlabInvCombo.currentIndexChanged.connect(xlabInvComboFunc)
        
        # save inversion results as .csv
        def saveInvDataBtnFunc():
            fdir = QFileDialog.getExistingDirectory(invTab, 'Choose directory where to save the files')
            if fdir != '':
                self.problem.saveInvData(fdir)
                self.writeLog('k.saveInvData("{:s}"'.format(fdir))
        self.saveInvDataBtn = QPushButton('Save Results')
        self.saveInvDataBtn.clicked.connect(saveInvDataBtnFunc)
        
        
        # for the map
        showInvMapParams = {'index':0, 'islice':0, 'vmin':None, 'vmax':None, 'cmap':'viridis_r'}

        def cmapInvMapComboFunc(index):
            showInvMapParams['cmap'] = self.cmapInvMapCombo.itemText(index)
            self.mwInvMap.replot(**showInvMapParams)
        self.cmapInvMapCombo = QComboBox()
        invMapCmaps = ['viridis_r', 'viridis', 'Greys', 'seismic', 'rainbow', 'jet', 'jet_r', 'turbo']
        self.cmapInvMapCombo.addItems(invMapCmaps)
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
                self.writeLog('k.saveInvData("{:s}"'.format(fdir))
        self.saveInvMapDataBtn = QPushButton('Save Results')
        self.saveInvMapDataBtn.clicked.connect(saveInvMapDataBtnFunc)
        
        def expInvMap():
            if self.projEdit.text() == '':
                self.errorDump('First specify a coordinate system (CRS)')
                return
            fname, _ = QFileDialog.getSaveFileName(importTab,'Export raster map', self.datadir, 'TIFF (*.tif)')
            if fname != '':
                self.setProjection()
                self.problem.saveSlice(fname=fname, islice=self.sliceCombo.currentIndex(), cmap=self.cmapInvMapCombo.currentText())
                self.writeLog('k.saveSlice(fname="{:s}", islice={:d}, cmap="{:s}"'.format(
                    fname, self.sliceCombo.currentIndex(), self.cmapInvMapCombo.currentText()))
                self.infoDump('File saved in ' + fname)
        self.invMapExpBtn = QPushButton('Export GIS')
        self.invMapExpBtn.setToolTip('Export a georeferenced TIFF file to directly be imported in GIS software.\n'
                                     'Choose the correct EPSG CRS projection in the "Importing" tab!')
        self.invMapExpBtn.clicked.connect(expInvMap)
        
        
        
        # 3D options tab
        self.showInv3dParams = {'index':0, 'vmin':None, 'vmax':None,
                                'maxDepth':None, 'cmap':'viridis_r', 'elev':False,
                                'edges':False, 'pvslices':([],[],[]),
                                'pvthreshold':None, 'pvgrid':False,
                                'pvcontour':[]}
        # 3D plotter
        if pvfound:
            plotterFrame = QFrame()
            vlayout = QHBoxLayout()
            self.plInv = QtInteractor(plotterFrame)
            vlayout.addWidget(self.plInv.interactor)
            plotterFrame.setLayout(vlayout)
            self.showInv3dParams['pl'] = self.plInv
        
        def replot3D():
            self.plInv.clear()
            try:
                self.problem.show3D(**self.showInv3dParams)
                self.writeLog('k.show3D', self.showInv3dParams)
            except:
                self.errorDump('Error in 3D plotting')
        
        def surveyInv3dComboFunc(index):
            self.showInv3dParams['index'] = index
            replot3D()
        self.surveyInv3dCombo = QComboBox()
        self.surveyInv3dCombo.activated.connect(surveyInv3dComboFunc)
        
        self.vminInv3dLabel = QLabel('Vmin|Vmax')
        self.vminInv3dEdit = QLineEdit('')
        self.vminInv3dEdit.setValidator(QDoubleValidator())
        
        # self.vmaxInv3dLabel = QLabel('vmax:')
        self.vmaxInv3dEdit = QLineEdit('')
        self.vmaxInv3dEdit.setValidator(QDoubleValidator())

        # 3D specific options for pyvista
        self.pvthreshLabel = QLabel('Threshold:')
        self.pvthreshLabel.setToolTip('Value which to keep the cells.')
        self.pvthreshMin = QLineEdit('')
        self.pvthreshMin.setPlaceholderText('Min')
        self.pvthreshMin.setValidator(QDoubleValidator())
        self.pvthreshMin.setToolTip('Minimal value above which to keep the cells.')
        
        # pvthreshMaxLabel = QLabel('Max Threshold:')
        self.pvthreshMax = QLineEdit('')
        self.pvthreshMax.setPlaceholderText('Max')
        self.pvthreshMax.setValidator(QDoubleValidator())
        self.pvthreshMax.setToolTip('Maximal value below which to keep the cells.')
        
        self.pvslicesLabel = QLabel('Slices:')
        self.pvslicesLabel.setToolTip('Slice the mesh normal to X, Y and/or Z axis. '
                                 'Set multiple slices on one axis by separating values with ","')
        self.pvxslices = QLineEdit('')
        self.pvxslices.setPlaceholderText('X [m]')
        self.pvxslices.setToolTip('e.g. 4, 5 to have to slices normal'
                                  'to X in 4 and 5.')
        
        # pvyslicesLabel = QLabel('Y slices:')
        self.pvyslices = QLineEdit('')
        self.pvyslices.setPlaceholderText('Y [m]')
        self.pvyslices.setToolTip('e.g. 4, 5 to have to slices normal'
                                  'to Y in 4 and 5.')
        
        # pvzslicesLabel = QLabel('Z slices:')
        self.pvzslices = QLineEdit('')
        self.pvzslices.setPlaceholderText('Z [m]')
        self.pvzslices.setToolTip('e.g. 4, 5 to have to slices normal'
                                  'to Z in 4 and 5.')
                
        self.pvcontourLabel = QLabel('Isosurfaces:')
        self.pvcontour = QLineEdit('')
        self.pvcontour.setToolTip('Values of isosurfaces (comma separated).')
        
        def pvapplyBtnFunc():
            vmin = float(self.vminInv3dEdit.text()) if self.vminInv3dEdit.text() != '' else None
            vmax = float(self.vmaxInv3dEdit.text()) if self.vmaxInv3dEdit.text() != '' else None
            self.showInv3dParams['vmin'] = vmin
            self.showInv3dParams['vmax'] = vmax
            threshMin = float(self.pvthreshMin.text()) if self.pvthreshMin.text() != '' else None
            threshMax = float(self.pvthreshMax.text()) if self.pvthreshMax.text() != '' else None
            self.showInv3dParams['pvthreshold'] = [threshMin, threshMax]
            if self.pvxslices.text() != '':
                xslices = [float(a) for a in self.pvxslices.text().split(',')]
            else:
                xslices = []
            if self.pvyslices.text() != '':
                yslices = [float(a) for a in self.pvyslices.text().split(',')]
            else:
                yslices = []
            if self.pvzslices.text() != '':
                zslices = [float(a) for a in self.pvzslices.text().split(',')]
            else:
                zslices = []
            self.showInv3dParams['pvslices'] = (xslices, yslices, zslices)
            if self.pvcontour.text() != '':
                pvcontour = [float(a) for a in self.pvcontour.text().split(',')]
            else:
                pvcontour = []
            self.showInv3dParams['pvcontour'] = pvcontour
            replot3D()
        self.pvapplyBtn = QPushButton('Apply 3D')
        self.pvapplyBtn.setAutoDefault(True)
        self.pvapplyBtn.clicked.connect(pvapplyBtnFunc)
        self.pvapplyBtn.setToolTip('Apply 3D options')
        
        def pvgridCheckFunc(state):
            self.showInv3dParams['pvgrid'] = self.pvgridCheck.isChecked()
            replot3D()
        self.pvgridCheck = QCheckBox('Grid')
        self.pvgridCheck.stateChanged.connect(pvgridCheckFunc)
        
        def cmapInv3dComboFunc(index):
            self.showInv3dParams['cmap'] = self.cmapInv3dCombo.itemText(index)
            replot3D()
        self.cmapInv3dCombo = QComboBox()
        self.cmapInv3dCombo.addItems(invMapCmaps)
        self.cmapInv3dCombo.activated.connect(cmapInv3dComboFunc)
        
        def saveVtkBtnFunc():
            fname, _ = QFileDialog.getSaveFileName(self, 'Open File', self.datadir,
                                                   'VTK (*.vtk)')
            if fname != '':
                self.problem.saveVTK(fname, index=self.showInv3dParams['index'],
                maxDepth=self.showInv3dParams['maxDepth'],
                elev=self.showInv3dParams['elev'])
                # self.vtkWidget.screenshot(fname, transparent_background=True)
        self.saveVtkBtn = QPushButton('Save')
        self.saveVtkBtn.setAutoDefault(True)
        self.saveVtkBtn.clicked.connect(saveVtkBtnFunc)        


        # subtabs
        self.graphTabs = QTabWidget()
        self.profTab = QWidget()
        self.mapTab = QWidget()
        self.m3dTab = QWidget()
        self.graphTabs.addTab(self.profTab, 'Profile')
        self.graphTabs.addTab(self.mapTab, 'Slice')
        self.graphTabs.addTab(self.m3dTab, '3D View')
        if pvfound is False:
            self.graphTabs.setEnabled(2, False)
        
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
        profOptionsLayout.addWidget(self.xlabInvCombo)
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
        mapOptionsLayout.addWidget(self.vmaxInvMapEdit)
        mapOptionsLayout.addWidget(self.applyInvMapBtn)
        mapOptionsLayout.addWidget(self.contourInvMapLabel)
        mapOptionsLayout.addWidget(self.contourInvMapCheck)
        mapOptionsLayout.addWidget(self.cmapInvMapCombo)
        mapOptionsLayout.addWidget(self.saveInvMapDataBtn)
        mapOptionsLayout.addWidget(self.invMapExpBtn)
        mapLayout.addLayout(mapOptionsLayout)
        mapLayout.addWidget(self.mwInvMap)
        self.mapTab.setLayout(mapLayout)
        
        m3dLayout = QVBoxLayout()
        m3dOptionsLayout = QHBoxLayout()
        m3dOptionsLayout.addWidget(self.surveyInv3dCombo)
        m3dOptionsLayout.addWidget(self.vminInv3dLabel)
        m3dOptionsLayout.addWidget(self.vminInv3dEdit)
        # m3dOptionsLayout.addWidget(self.vmaxInv3dLabel)
        m3dOptionsLayout.addWidget(self.vmaxInv3dEdit)
        m3dOptionsLayout.addWidget(self.pvthreshLabel)
        m3dOptionsLayout.addWidget(self.pvthreshMin)
        m3dOptionsLayout.addWidget(self.pvthreshMax)
        m3dOptionsLayout.addWidget(self.pvslicesLabel)
        m3dOptionsLayout.addWidget(self.pvxslices)
        m3dOptionsLayout.addWidget(self.pvyslices)
        m3dOptionsLayout.addWidget(self.pvzslices)
        m3dOptionsLayout.addWidget(self.pvcontourLabel)
        m3dOptionsLayout.addWidget(self.pvcontour)
        m3dOptionsLayout.addWidget(self.pvapplyBtn)
        m3dOptionsLayout.addWidget(self.cmapInv3dCombo)
        m3dOptionsLayout.addWidget(self.pvgridCheck)
        m3dOptionsLayout.addWidget(self.saveVtkBtn)
        m3dLayout.addLayout(m3dOptionsLayout)
        m3dLayout.addWidget(plotterFrame)
        self.m3dTab.setLayout(m3dLayout)
        
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
        
        
        #%% help tab
        helpTab = QTabWidget()
        self.tabs.addTab(helpTab, 'Help')
        
        helpBasic = QTextBrowser()
        helpBasic.setReadOnly(True)
        helpBasic.setOpenExternalLinks(True)
        helpBasic.setText(
        '''
        <p> <b> Welcome to the help tab! </b> </p>
        <p> This quick start guide provides information to get started with <b>EMagPy</b> (see also <a href="https://www.sciencedirect.com/science/article/pii/S0098300420305513">McLachlan, Blanchy and Binley, 2020</a>). </p>
        <p> EMI methods use electromagnetic fields to obtain information about subsurface electrical conductivity:
           <ul>
           <li> Firstly, a primary electromagnetic field is generated by passing an electrical current through a transmitter coil (<b>Tx</b>).</li>
           <li> This electromagnetic field interacts with conductors in the subsurface to produce <b> eddy currents </b>, and consequently generate a secondary electromagnetic field.</li>
           <li> The primary and secondary fields are measured by a receiver coil (<b>Rx</b>).</li>
           <li> The interference of the primary and secondary electromagnetic fields provides information about electrical conductivity of the subsurface.</li>
          </ul>
        </p>
        <p>
        <br>

        <img width=800 src="image/fig1.png"></img>
        </p>
        <p>The depth of sensitivity of EMI measurements is dependent upon:
            <ul>
            <li> Orientation of receiver and transmitter coils.</li>
            <li> Separation distance between <b>Tx</b>  and <b>Rx</b> coils.</li>
            <li> Operating frequency of the device.</li>
            <li> Elevation of the device above the ground.</li>
            <li> Subsurface electrical conductivity distribution.</li>
            </ul>

        <p>By using measurements with different sensitivity patterns, models of conductivity can be constructed via inverse methods.
        </p>
        ''')
        helpTab.addTab(helpBasic, 'Background')


        helpFwd = QTextBrowser()
        helpFwd.setReadOnly(True)
        helpFwd.setOpenExternalLinks(True)
        helpFwd.setText(
        '''
        <p> Inverse modeling is reliant on an accurate forward model, in this tab the implemented forward models are explained.
        <p> The out-of-phase component of the primary and secondary electromagnetic fields, the quadrature (<b>Q</b>), is typically expressed as an apparent conductivity (<b>ECa</b>) value as it offers a more comprehensible value with the same units as conductivity [<b>mS/m</b>]. 
  
        <p>In <b>EMagPy</b> the forward model response (i.e. theoretical response) of any given model of conductivity is calculated using one of two options:
            <ul>
            <li> The linear cumulative sensitivity (<b>CS</b>) function <a href="http://www.geonics.com/pdfs/technicalnotes/tn6.pdf">(McNeill, 1980)</a>.</li>
            <li> A 1D Maxwell based '<i>full solution</i>' (<b>FS</b>) forward model <a href="https://doi.org/10.1016/B978-0-12-730880-7.X5001-7">(Wait, 1982)</a>.</li>
          </ul>
         
         Whereas the <b>CS</b> function calculates the response directly in terms of <b>ECa</b>, the <b>FS</b> solution produces a response in terms of <b>Q</b>. </p>  
        <p> There are several methods in which to convert <b>Q</b> values to <b>ECa</b> values, for instance:
            <ul>
            <li> The low induction number <b>LIN</b> approximation <b>CS</b> <a href="http://www.geonics.com/pdfs/technicalnotes/tn6.pdf">(McNeill, 1980)</a>.</li>
            <li> The homogenous equivalent method <a href="https://www.earthdoc.org/content/papers/10.3997/2214-4609.201602080">(Andrade, 2016)</a>.</li>
            <li> Some alternative manufacturer calibration.</li>

          </ul>
         </p>  
         <p>        

        <img width=800 src="image/fig3.png"></img>

        </p>

        <p>In the EMagPy inversion tab the <b>CS</b> forward model is selected by specifying '<i>CS</i>', given its linear nature the <b>CS</b> function is faster than the <b>FS</b> forward model. However, it is generally only suitable for low conductivities when the device is operated at ground level. </p>

        <p>Although <b>EMagPy</b> can be use with <b>Q</b> values, by specifying '<i>Q</i>' in the inversion tab, the majority of cases will likely deal with data in terms of <b>ECa</b>. </p>  In <b>EMagPy</b> the <b>FS</b> forward model with the <b>LIN</b> conversion can be implemented with '<i>FSlin</i>' whereas the <b>FS</b> forward model with the homogenous equivalent conductivity can be implemented with '<i>FSeq</i>'. </p>
        <p color='red'><b> It is essential that the theoretical response, in terms of <i>ECa</i>, is comparable to the measured data.</b><br> For instance, data expressed using ECa values obtained by some manufacturer calibration do not yield reliable inversion results if the '<i>FSlin</i>' or '<i>FSeq</i>' methods are used. </p> <p>
        ''')

        helpTab.addTab(helpFwd, 'Forward model')

        helpInput = QTextBrowser()
        helpInput.setReadOnly(True)
        helpInput.setOpenExternalLinks(True)
        helpInput.setText(
        '''
        <p>
        To operate properly, <b> EMagPy </b> must be supplied with the following aquisition setup details:
<ul>
            <li> Orientation of receiver and transmitter coils.</li>
            <ul> Horizontal Coplanar (<b>HCP</b>), 
                 Vertical Coplanar (<b>VCP</b>) and
                 Perpendicular (<b>PRP</b>).
  </ul>

            <li> Separation distance between transmitter and receiver coils in <b>metres</b>.</li>
            <li> Operating frequency of the device in <b>Hertz</b>.</li>
            <li> Elevation of the device above the ground in <b>metres</b>.</li>
  </ul>
<p> The below table provides an example of headers containing the details of a device with 3 receiver coils operated at ground level, a frequency of 9000 Hz and with both HCP and PRP coil orientations.</p>
   <p>
        <br>

        <img width=800 src="image/datalayout.png"></img>
</p>
<p> Additionally, if data from either the GF Instruments Mini-Explorer or Explorer is loaded an additional correction can be done to convert the raw EMI ECa values into LIN-ECa values. This is done because the GF Instruments contain a specific manufacturer calibration to link <b>Q</b> and <b>ECa</b>.
</p>
<p><b>Note: </b>If ERT data for calibration are available, you do not need to apply this correction as
the ERT calibration will account for it.</p>

        ''')
        helpTab.addTab(helpInput, 'Data Input')
       
        

        helpInv = QTextBrowser()
        helpInv.setReadOnly(True)
        helpInv.setOpenExternalLinks(True)
        helpInv.setText(
        '''
        <p> <b> EMagPy </b> can calibrate EMI data using inverted ERT data, check out <b>EMagPy's</b> ERT/IP counterpart  <a href="https://gitlab.com/hkex/pyr2"><b>ResIPy</b></a> for inversion of ERT data.</p>
        <p> To do this the forward model response of an inverted ERT model, in terms of ECa, can be paired with EMI measurements and fitted with a linear regression. 
        <p> EMI data collected along an ERT transect must be supplied along with information from the ERT model. The ERT data can be supplied in one of two formats:
<ul>
    <li> The x, z and resistivity information for a quadrilateral or triangular mesh inverted for a flat topography.</li>
    <li> As a series of EC profiles, with the depths of each measurement provided in the headers.
</ul>
</p>
<p>
        <img width=300 src="image/calibfile.png"></img>
        <img width=500 src="image/ec_header.png"></img>
</p>


  
        ''')
        helpTab.addTab(helpInv, 'Calibration')    
        
        helpInv = QTextBrowser()
        helpInv.setReadOnly(True)
        helpInv.setOpenExternalLinks(True)
        helpInv.setText(
        '''
        <p><b>EMagPy</b> can produce both smoothly and sharply varying models of electrical conductivity.</p>
        <p>
            The smooth inversion requires a series of fixed depths to solve the inverse problem in terms 
            of conductivity. A vertical smoothing parameter is also required to regularize the solution.
        </p>
        <p>
            The sharp model treats both model depths and model conductivties as parameters,
            however it is important that the number of layers is limited to avoid geologically
            unfeasible models.
        </p>
        <p>
            <img src='image/smooth-sharp.png' width=600></img>
        </p>
        <p><b><i>Inverse Settings</i></b></p>
        <p>The starting model parameters can be supplied in the '<i>Inversion Settings</i>' 
            tab where it can also be specified if these parameters should be fixed or not. </p>
        <p> The user can also choose to fit an L-curve which will plot a graph of model and data misfit 
        for a number of vertical smoothing values. </p>
        <p>
            <img width=250 src="image/lcurve.png"></img>

        </p>
        <p><b><i>Inversion</i></b></p>
        <p>The 'inversion tab' contains a number of options:
        <ul>
        <li> Forward model.</li>
        <ul> Specifies which forward model to use, (see Forward Model Help). </ul>
        <li> Inversion method.</li>
        <ul> Specifies the inversion method to use, (see <a href="https://www.sciencedirect.com/science/article/pii/S0098300420305513">McLachlan, Blanchy and Binley, 2020</a>). </ul>
        <li> Vertical smoothing.</li>
        <ul> This is required for smooth inversion and should be set to 0 for sharp inversions. </ul>
        <li> Lateral smoothing.</li>
        <ul> This is enables lateral smoothing to constrain 1D models to neighboring models. </ul>
        <li> Regularization.</li>
        <ul> Specifes whether to use the L1 or L2 norm (of the parameter space) to minimise misfit. </ul>
        <li> Number of iterations.</li>
        <li> Parallel computing.</li>
        <ul>If available checking this option will make use of multi-core machines (only available when running from source). </ul>
        </ul>
        </p>
        ''')
        helpTab.addTab(helpInv, 'Inversion')       
        
        
        
        #%%about tab
        tabAbout = QWidget()
        self.tabs.addTab(tabAbout, 'About')

        infoLayout = QVBoxLayout()
        aboutText = QLabel()
        aboutText.setText('''<h1>About EMagPy</h1>
            <p><b>Version: {:s}</b></p>
            <p><i>EMagPy is a free and open source software for inversion of 1D electromagnetic data</i></p>
            <p>If you encouter any issues or would like to submit a feature request, please raise an issue on our gitlab repository at:</p>
            <p><a href="https://gitlab.com/hkex/emagpy/issues">https://gitlab.com/hkex/emagpy/issues</a></p>
            <p>EMagPy uses a few Python packages: 
                <a href="https://numpy.org/">numpy</a>, 
                <a href="https://pandas.pydata.org/">pandas</a>,
                <a href="https://matplotlib.org/">matplotlib</a>,
                <a href="https://scipy.org/index.html">scipy</a>,
                <a href="https://pypi.org/project/PyQt5/">PyQt5</a>,
                <a href="https://doi.org/10.1371/journal.pone.0145180">spotpy</a>,
                <a href="http://pyproj4.github.io/pyproj/stable/">pyproj</a>,
                <a href="https://joblib.readthedocs.io/en/latest/">joblib</a> and
                <a href="https://rasterio.readthedocs.io/en/latest/index.html">rasterio (optional)</a>.
            </p>
            <p>EMagPy is the sister code of <a href="https://gitlab.com/hkex/pyr2">ResIPy</a> and has EMagPy's design has been heavily influenced by ResIPy.</p>
            <p><strong>EMagPy's core developers: Guillaume Blanchy and Paul McLachlan.<strong></p>
            <p>Contributors: Jimmy Boyd, Sina Saneiyan, Martina Tabaro</p>
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
    
    def setProjection(self):
        val = self.projEdit.text()
        try:
            if val.split(':')[0].lower() == 'epsg':
                epsg_code = [val.split(':')[1]]
            elif any(self.pcs['COORD_REF_SYS_NAME'] == val) is True:
                epsg_code = self.pcs['COORD_REF_SYS_CODE'][self.pcs['COORD_REF_SYS_NAME'] == val].values
            elif any(self.pcs['COORD_REF_SYS_NAME_rev'] == val) is True:
                epsg_code = self.pcs['COORD_REF_SYS_CODE'][self.pcs['COORD_REF_SYS_NAME_rev'] == val].values
            epsgVal = 'EPSG:'+str(epsg_code[0])
            self.problem.setProjection(targetProjection=epsgVal)
            self.writeLog('k.setProjection(targetProjection="{:s}")'.format(epsgVal))
        except:
            self.errorDump('CRS projection is not correctly defined - See "Importing" tab')
    
    def projBtnFunc(self):
        if self.projEdit.text() == '':
            self.errorDump('Define CRS first')
        else:
            # try:
            self.setProjection()
            self.problem.convertFromCoord(targetProjection=self.problem.projection)
            self.writeLog('k.convertFromCoord(targetProjection="{:s}")'.format(self.problem.projection))
            self.replot()
            # except Exception as e:
            #     self.errorDump(e)

    def replot(self):
        index = self.showParams['index']
        coil = self.showParams['coil']
        contour = self.showParams['contour']
        vmin = self.showParams['vmin']
        vmax = self.showParams['vmax']
        pts = self.showParams['pts']
        cmap = self.showParams['cmap']
        dist = self.showParams['dist']
        # if self.mapRadio.isChecked():
        if self.showCombo.currentText() == 'Map':
            self.mwRaw.replot(index=index, coil=coil, contour=contour,
                              vmin=vmin, vmax=vmax, pts=pts, cmap=cmap)
            self.writeLog('k.showMap(index={:d}, coil="{:s}", contour={:s},'
                          ' vmin={:s}, vmax={:s}, pts={:s}, cmap="{:s}")'.format(
                              index, coil, str(contour), str(vmin), str(vmax),
                              str(pts), cmap))
        else:
            self.mwRaw.replot(index=index, coil=coil, vmin=vmin, vmax=vmax,
                              dist=dist)
            if self.showCombo.currentText() == 'Dots':
                self.writeLog('k.show(index={:d}, coil="{:s}",'
                          ' vmin={:s}, vmax={:s}, dist={:s})'.format(
                              index, coil, str(vmin), str(vmax),
                              str(dist)))
            else:
                self.writeLog('k.showPseudo(index={:d}, coil="{:s}",'
                          ' vmin={:s}, vmax={:s}, dist={:s})'.format(
                              index, coil, str(vmin), str(vmax),
                              str(dist)))
                
    def processFname(self, fnames, merged=False):
        self.problem.surveys = [] # empty the list of current survey
        if len(fnames) == 1:
            fname = fnames[0]
            self.importBtn.setText(os.path.basename(fname))
            self.problem.createSurvey(fname)
            self.writeLog('k.createSurvey("{:s}")'.format(fname))
            self.gammaEdit.setVisible(False)
            self.gammaLabel.setVisible(False)
        else:
            self.importBtn.setText(os.path.basename(fnames[0]) + ' .. '
                                   + os.path.basename(fnames[-1]))
            if merged:
                self.problem.createMergedSurvey(fnames)
                self.writeLog('k.createMergedSurvey({:s})'.format(str(fnames)))
            else:
                self.gammaEdit.setVisible(True)
                self.gammaLabel.setVisible(True)
                self.problem.createTimeLapseSurvey(fnames)
                self.writeLog('k.createTimeLapseSurvey({:s})'.format(str(fnames)))
        # check if coils configuration seem to match GF instruments
        coils = ['{:s}{:.2f}'.format(a.upper(), b) for a, b in zip(self.problem.cpos, self.problem.cspacing)]
        if np.sum(np.in1d(coils, ['VCP0.32', 'HCP0.32', 'VCP1.48', 'HCP1.48'])) > 0:
            self.gfCorrectionBtn.setVisible(True)
            self.gfCalibCombo.setVisible(True)
            self.infoDump('Files well imported. GF instrument suspected. Check if you need to apply the GF correction.')
        else:
            self.gfCorrectionBtn.setVisible(False)
            self.gfCalibCombo.setVisible(False)
            self.infoDump('Files well imported')
        self.setupUI()
        
    def setupUI(self):
        # reset parameters
        self.showParams = {'index': 0, 'coil':'all', 'contour':False, 'vmin':None,
                           'vmax':None,'pts':False, 'cmap':'viridis_r', 'dist':True} 

        # draw default plot
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
        self.surveyInv3dCombo.clear()
        for survey in self.problem.surveys:
            self.surveyCombo.addItem(survey.name)
            self.surveyErrCombo.addItem(survey.name)
            self.surveyInvCombo.addItem(survey.name)
            self.surveyInvMapCombo.addItem(survey.name)
            self.surveyInv3dCombo.addItem(survey.name)
        
        # set to default values
        # self.showRadio.setChecked(True)
        self.showCombo.setCurrentIndex(0)
        self.showCombo.setEnabled(True)
        self.xlabCombo.setEnabled(True)
        self.contourCheck.setChecked(False)

        # enable widgets
        if 'latitude' in survey.df.columns:
            #try:
            #    float(survey.df['latitude'][0]) # coordinates are not string
            #except: # coordinates are string
                #self.problem.projection = 'EPSG:27700' # a default CRS if user hasn't defined anything
                #self.projBtnFunc() # automatically convert NMEA string
            self.projBtn.setEnabled(True)
            # self.projEdit.setEnabled(True)
        self.keepApplyBtn.setEnabled(True)
        self.rollingBtn.setEnabled(True)
        self.ptsKillerBtn.setEnabled(True)
        self.gridBtn.setEnabled(True)
        self.coilCombo.setEnabled(True)
        self.surveyCombo.setEnabled(True)
        # self.showRadio.setEnabled(True)
        # self.mapRadio.setEnabled(True)
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


    def writeLog(self, text, dico=None):
        if dico is None:
            self.apiLog += text + '\n'
        else:
            arg = ''
            for key in dico.keys():
                val = dico[key]
                if type(val) == str:
                    arg += key + '="{:s}", '.format(val)
                elif (type(val) == int or
                    type(val) == float or
                    type(val) == tuple or
                    type(val) == list or
                    type(val) == bool or
                    val is None):
                    arg += key + '={:s}, '.format(str(val))
                elif type(val) == type(np.ndarray):
                    arg += key + '={:s}, '.format(str(list(val)))
                else:
                    pass # ignore argument
            self.apiLog += text + '(' + arg[:-2] + ')\n'
        
    def saveLog(self):
        fname, _ = self._dialog.getSaveFileName(self)
        if fname != '':
            if fname[-3:] != '.py':
                fname = fname + '.py'
            self.apiLog = self.apiLog.replace('"None"', 'None')
            with open(fname, 'w') as f:
                f.write(self.apiLog)

    def saveData(self):
        fdir = QFileDialog.getExistingDirectory(self, 'Choose directory where to save the files')
        if fdir != '':
            self.problem.saveData(fdir)
            self.writeLog('k.saveData("{:s}"'.format(fdir))
        
        
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
