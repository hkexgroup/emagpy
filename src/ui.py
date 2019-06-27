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

from emagpy import Problem

from PyQt5.QtWidgets import (QMainWindow, QSplashScreen, QApplication, QPushButton, QWidget,
    QTabWidget, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QMessageBox,
    QFileDialog, QCheckBox, QComboBox, QTextEdit, QSlider, QHBoxLayout,
    QTableWidget, QFormLayout, QTableWidgetItem, QHeaderView, QProgressBar,
    QStackedLayout, QRadioButton, QGroupBox, QButtonGroup)#, QAction, QListWidget, QShortcut)
from PyQt5.QtGui import QIcon, QPixmap, QIntValidator, QDoubleValidator#, QKeySequence
from PyQt5.QtCore import QThread, pyqtSignal#, QProcess, QSize
from PyQt5.QtCore import Qt

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
    msg.setInformativeText('''Please see the detailed error below.<br>You can report the errors at:<p><a href='https://gitlab.com/hkex/pyr2/issues'>https://gitlab.com/hkex/pyr2/issues</a></p><br>''')
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
QT_AUTO_SCREEN_SCALE_FACTOR = True # for high dpi display

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
    def __init__(self, parent=None, navi=True, itight=False, threed=False):
        super(MatplotlibWidget, self).__init__(parent) # we can pass a figure but we can replot on it when
        # pushing on a button (I didn't find a way to do it) while with the axes, you can still clear it and
        # plot again on them
        self.itight = itight
        figure = Figure()
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
        if self.itight is True:
            self.figure.tight_layout()
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
        if self.itight is True:
            self.figure.tight_layout()
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
        
        # do the checks for wine and updates in seperate thread
#        tupdate = customThread(self.updateChecker)
#        tupdate.signal.connect(self.updateCheckerShow)
#        tupdate.start()
        
        self.setWindowTitle('EMagPy')
        self.setGeometry(100,100,1100,600)

        self.problem = None
        self.datadir = os.path.join(bundle_dir, 'emagpy', 'test')

        def errorDump(text, flag=1):
            text = str(text)
            timeStamp = time.strftime('%H:%M:%S')
            if flag == 1: # error in red
                col = 'red'
            else:
                col = 'black'
            errorLabel.setText('<i style="color:'+col+'">['+timeStamp+']: '+text+'</i>')
        errorLabel = QLabel('<i style="color:black">Error messages will be displayed here</i>')
        QApplication.processEvents()


        def infoDump(text):
            errorDump(text, flag=0)

        self.table_widget = QWidget()
        layout = QVBoxLayout()
        tabs = QTabWidget()
        
        self.problem = Problem()
        
        
        #%% tab 1 importing data
        ''' STRUCTURE OF A TAB
        - first create the tab widget and it's layout
        - then create all the widgets that will interact in the tab
        - at the end do the layouting (create sub layout if needed) and
        '''

        importTab = QTabWidget()
        tabs.addTab(importTab, 'Importing')
        

        # select type of sensors
        def sensorComboFunc(index):
            print('sensor selected is:', sensors[index])
            if index == 1 or index == 2:
                showGF(True)
            else:
                showGF(False)
        sensorCombo = QComboBox()
        sensors = ['CMD Mini-Explorer',
                   'CMD Explorer'
                   ]
        sensors = sorted(sensors)
        sensors = ['All'] + sensors
        for sensor in sensors:
            sensorCombo.addItem(sensor)
        sensorCombo.currentIndexChanged.connect(sensorComboFunc)
        
        
        # import data
        def importBtnFunc():
            fname, _ = QFileDialog.getOpenFileName(importTab, 'Select data file', self.datadir, '*.csv')
            if fname != '':
                importBtn.setText(os.path.basename(fname))
                self.problem.surveys = [] # empty the list of current survey
                self.problem.createSurvey(fname)
                infoDump(fname + ' well imported')
                setupUI()
            
        def setupUI():
            mwRaw.setCallback(self.problem.show)
            mwRaw.replot()
            
            # fill the combobox with survey and coil names
            coilCombo.disconnect()
            coilErrCombo.clear()
            coilCombo.clear()
            for coil in self.problem.coils:
                coilCombo.addItem(coil)
                coilErrCombo.addItem(coil)
            coilCombo.addItem('all')
            coilCombo.currentIndexChanged.connect(coilComboFunc)
            coilCombo.setCurrentIndex(len(self.problem.coils))
            surveyCombo.clear()
            surveyInvCombo.disconnect()
            surveyInvCombo.clear()
            surveyInvMapCombo.disconnect()
            surveyInvMapCombo.clear()
            for survey in self.problem.surveys:
                surveyCombo.addItem(survey.name)
                surveyErrCombo.addItem(survey.name)
                surveyInvCombo.addItem(survey.name)
                surveyInvMapCombo.addItem(survey.name)
            surveyInvCombo.currentIndexChanged.connect(surveyInvComboFunc)
            surveyInvMapCombo.currentIndexChanged.connect(surveyInvMapComboFunc)

            # set to default values
            showRadio.setChecked(True)
            contourCheck.setChecked(False)

            # enable widgets
            if 'Latitude' in survey.df.columns:
                projBtn.setEnabled(True)
                projEdit.setEnabled(True)
                projBtnFunc() # automatically convert NMEA string
            keepApplyBtn.setEnabled(True)
            rollingBtn.setEnabled(True)
            ptsKillerBtn.setEnabled(True)
            coilCombo.setEnabled(True)
            surveyCombo.setEnabled(True)
            showRadio.setEnabled(True)
            mapRadio.setEnabled(True)
            applyBtn.setEnabled(True)
            cmapCombo.setEnabled(True)
            contourCheck.setEnabled(True)
            ptsCheck.setEnabled(True)
            
        importBtn = QPushButton('Import Data')
        importBtn.setStyleSheet('background-color:orange')
        importBtn.clicked.connect(importBtnFunc)
        
        
        def importGFLoFunc():
            fname, _ = QFileDialog.getOpenFileName(importTab, 'Select data file', self.datadir)
            if fname != '':
                self.fnameLo = fname
                importGFLo.setText(os.path.basename(fname))
        importGFLo = QPushButton('Select Lo')
        importGFLo.clicked.connect(importGFLoFunc)
        def importGFHiFunc():
            fname, _ = QFileDialog.getOpenFileName(importTab, 'Select data file', self.datadir)
            if fname != '':
                self.fnameHi = fname
                importGFHi.setText(os.path.basename(fname))
        importGFHi = QPushButton('Select Hi')
        importGFHi.clicked.connect(importGFHiFunc)
        hxLabel = QLabel('Height above the ground [m]:')
        hxEdit = QLineEdit('0')
        hxEdit.setValidator(QDoubleValidator())
        def importGFApplyFunc():
            hx = float(hxEdit.text()) if hxEdit.text() != '' else 0
            device = sensorCombo.itemText(sensorCombo.currentIndex())
            self.problem.importGF(self.fnameLo, self.fnameHi, device, hx)
            infoDump('Surveys well imported')
            setupUI()
        importGFApply = QPushButton('Import')
        importGFApply.setStyleSheet('background-color: orange')
        importGFApply.clicked.connect(importGFApplyFunc)
        
        
        def showGF(arg):
            visibles = np.array([True, False, False, False, False, False])
            objs = [importBtn, importGFLo, importGFHi, hxLabel, hxEdit, importGFApply]
            if arg is True:
                [o.setVisible(~v) for o,v in zip(objs, visibles)]
            else:
                [o.setVisible(v) for o,v in zip(objs, visibles)]
        showGF(False)

        
        # projection (only if GPS data are available)
        projEdit = QLineEdit('27700')
        projEdit.setValidator(QDoubleValidator())
        projEdit.setEnabled(False)
        def projBtnFunc():
            val = float(projEdit.text()) if projEdit.text() != '' else None
            self.problem.convertFromNMEA(targetProjection='EPSG:{:.0f}'.format(val))
            mwRaw.replot(**showParams)
        projBtn = QPushButton('Convert NMEA')
        projBtn.clicked.connect(projBtnFunc)
        projBtn.setEnabled(False)
        
        
        # filtering options
        filtLabel = QLabel('Filter Options |')
        filtLabel.setStyleSheet('font-weight:bold')

        # vmin/vmax filtering
        vminfLabel = QLabel('Vmin:')
        vminfEdit = QLineEdit()
        vminfEdit.setValidator(QDoubleValidator())
        vmaxfLabel = QLabel('Vmax:')
        vmaxfEdit = QLineEdit()
        vmaxfEdit.setValidator(QDoubleValidator())
        def keepApplyBtnFunc():
            vmin = float(vminfEdit.text()) if vminfEdit.text() != '' else None
            vmax = float(vmaxfEdit.text()) if vmaxfEdit.text() != '' else None
            self.problem.keepBetween(vmin, vmax)
            mwRaw.replot(**showParams)
        keepApplyBtn = QPushButton('Apply')
        keepApplyBtn.clicked.connect(keepApplyBtnFunc)
        keepApplyBtn.setEnabled(False)
        keepApplyBtn.setAutoDefault(True)
        
        # rolling mean
        rollingLabel = QLabel('Window size:')
        rollingEdit = QLineEdit('3')
        rollingEdit.setValidator(QIntValidator())
        def rollingBtnFunc():
            window = int(rollingEdit.text()) if rollingEdit.text() != '' else None
            self.problem.rollingMean(window=window)
            mwRaw.replot(**showParams)
        rollingBtn = QPushButton('Rolling Mean')
        rollingBtn.clicked.connect(rollingBtnFunc)
        rollingBtn.setEnabled(False)
        rollingBtn.setAutoDefault(True)
        
        # manual point killer selection
        def ptsKillerBtnFunc():
            pass
            print('deleted 0/X points')
            # TODO delete selected
        ptsKillerBtn = QPushButton('Delete selected points')
        ptsKillerBtn.clicked.connect(ptsKillerBtnFunc)
        ptsKillerBtn.setEnabled(False)
        ptsKillerBtn.setAutoDefault(True)
        
        
        # display options
        displayLabel = QLabel('Display Options |')
        displayLabel.setStyleSheet('font-weight:bold')
        
        # survey selection (useful in case of time-lapse dataset)
        def surveyComboFunc(index):
            showParams['index'] = index
            mwRaw.replot(**showParams)
        surveyCombo = QComboBox()
        surveyCombo.currentIndexChanged.connect(surveyComboFunc)
        surveyCombo.setEnabled(False)
        
        # coil selection
        def coilComboFunc(index):
            showParams['coil'] = coilCombo.itemText(index)
            mwRaw.replot(**showParams)
        coilCombo = QComboBox()
        coilCombo.currentIndexChanged.connect(coilComboFunc)
        coilCombo.setEnabled(False)
        
        showParams = {'index': 0, 'coil':None, 'contour':False, 'vmin':None,
                      'vmax':None,'pts':False, 'cmap':'viridis_r'} 
        
        # switch between Raw data and map view
#        def showRadioFunc(state):
#            mwRaw.setCallback(self.problem.show)
#            mwRaw.replot(**showParams)
#        showRadio = QRadioButton('Raw')
#        showRadio.setChecked(True)
#        showRadio.toggled.connect(showRadioFunc)
#        showRadio.setEnabled(False)
#        def mapRadioFunc(state):
#            showMapOptions(state)
#            mwRaw.setCallback(self.problem.showMap)
#            mwRaw.replot(**showParams)
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
                mapRadio.setChecked(False)
                showMapOptions(False)
                mwRaw.setCallback(self.problem.show)
                mwRaw.replot(**showParams)
            else:
                mapRadio.setChecked(True)
        showRadio = QPushButton('Raw')
        showRadio.setCheckable(True)
        showRadio.setChecked(True)
        showRadio.toggled.connect(showRadioFunc)
        showRadio.setEnabled(False)
        showRadio.setAutoDefault(True)
        def mapRadioFunc(state):
            if state:
                showRadio.setChecked(False)
                showMapOptions(True)
                mwRaw.setCallback(self.problem.showMap)
                mwRaw.replot(**showParams)
            else:
                showRadio.setChecked(True)
        mapRadio = QPushButton('Map')
        mapRadio.setCheckable(True)
        mapRadio.setEnabled(False)
        mapRadio.setChecked(False)
        mapRadio.setAutoDefault(True)
        mapRadio.toggled.connect(mapRadioFunc)
        showGroup = QGroupBox()
        showGroupLayout = QHBoxLayout()
        showGroupLayout.addWidget(showRadio)
        showGroupLayout.addWidget(mapRadio)
        showGroup.setLayout(showGroupLayout)
        showGroup.setFlat(True)
        showGroup.setContentsMargins(0,0,0,0)
        showGroup.setStyleSheet('QGroupBox{border: 0px;'
                                'border-style:inset;}')
        
        def showMapOptions(arg):
            objs = [ptsLabel, ptsCheck, contourLabel, contourCheck]
            [o.setVisible(arg) for o in objs]
            if arg is False:
                coilCombo.addItem('all')
            else:
                n = len(self.problem.coils)
                if coilCombo.currentIndex() == n:
                    coilCombo.setCurrentIndex(n-1)
                coilCombo.removeItem(n)
            print([coilCombo.itemText(i) for i in range(coilCombo.count())])

        # apply the display vmin/vmax for colorbar or y label                
        vminEdit = QLineEdit('')
        vminEdit.setValidator(QDoubleValidator())
        vmaxEdit = QLineEdit('')
        vmaxEdit.setValidator(QDoubleValidator())        
        def applyBtnFunc():
            showParams['vmin'] = float(vminEdit.text()) if vminEdit.text() != '' else None
            showParams['vmax'] = float(vmaxEdit.text()) if vmaxEdit.text() != '' else None
            mwRaw.replot(**showParams)
        applyBtn = QPushButton('Apply')
        applyBtn.clicked.connect(applyBtnFunc)
        applyBtn.setEnabled(False)
        applyBtn.setAutoDefault(True)
        
        # select different colormap
        def cmapComboFunc(index):
            showParams['cmap'] = cmaps[index]
            mwRaw.replot(**showParams)
        cmapCombo = QComboBox()
        cmaps = ['viridis', 'viridis_r', 'seismic', 'rainbow']
        for cmap in cmaps:
            cmapCombo.addItem(cmap)
        cmapCombo.currentIndexChanged.connect(cmapComboFunc)
        cmapCombo.setEnabled(False)

        # allow map contouring using tricontourf()
        contourLabel = QLabel('Contour')
        contourLabel.setVisible(False)
        def contourCheckFunc(state):
            showParams['contour'] = state
            mwRaw.replot(**showParams)
        contourCheck = QCheckBox()
        contourCheck.setEnabled(False)
        contourCheck.clicked.connect(contourCheckFunc)
        contourCheck.setVisible(False)
        
        # show data points on the contoured map
        ptsLabel = QLabel('Points')
        ptsLabel.setVisible(False)
        def ptsCheckFunc(state):
            showParams['pts'] = state
            mwRaw.replot(**showParams)
        ptsCheck = QCheckBox()
        ptsCheck.clicked.connect(ptsCheckFunc)
        ptsCheck.setToolTip('Show measurements points')
        ptsCheck.setVisible(False)


        # display it
        mwRaw = MatplotlibWidget()
        
        
        # layout
        importLayout = QVBoxLayout()
        
        topLayout = QHBoxLayout()
        topLayout.addWidget(sensorCombo, 15)
        topLayout.addWidget(importBtn, 15)
        topLayout.addWidget(importGFLo, 10)
        topLayout.addWidget(importGFHi, 10)
        topLayout.addWidget(hxLabel, 10)
        topLayout.addWidget(hxEdit, 10)
        topLayout.addWidget(importGFApply, 10)
        topLayout.addWidget(QLabel('EPSG:'), 5)
        topLayout.addWidget(projEdit, 10)
        topLayout.addWidget(projBtn, 10)
        
        filtLayout = QHBoxLayout()
        filtLayout.addWidget(filtLabel)
        filtLayout.addWidget(vminfLabel)
        filtLayout.addWidget(vminfEdit)
        filtLayout.addWidget(vmaxfLabel)
        filtLayout.addWidget(vmaxfEdit)
        filtLayout.addWidget(keepApplyBtn)
        filtLayout.addWidget(rollingLabel)
        filtLayout.addWidget(rollingEdit)
        filtLayout.addWidget(rollingBtn)
        filtLayout.addWidget(ptsKillerBtn)
    
        midLayout = QHBoxLayout()
        midLayout.addWidget(displayLabel)
        midLayout.addWidget(surveyCombo, 7)
        midLayout.addWidget(QLabel('Select coil:'))
        midLayout.addWidget(coilCombo, 7)
        midLayout.addWidget(showGroup)
        midLayout.addWidget(QLabel('Vmin:'))
        midLayout.addWidget(vminEdit, 5)
        midLayout.addWidget(QLabel('Vmax:'))
        midLayout.addWidget(vmaxEdit, 5)
        midLayout.addWidget(applyBtn)
        midLayout.addWidget(cmapCombo)
        midLayout.addWidget(contourLabel)
        midLayout.addWidget(contourCheck)
        midLayout.addWidget(ptsLabel)
        midLayout.addWidget(ptsCheck)
        
        
        importLayout.addLayout(topLayout)
        importLayout.addLayout(filtLayout)
        importLayout.addLayout(midLayout)
        importLayout.addWidget(mwRaw)
        
        importTab.setLayout(importLayout)
        

        
#        #%% filtering data
#        filterTab = QTabWidget()
#        tabs.addTab(filterTab, 'Filtering')
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
        tabs.addTab(calibTab, 'ERT Calibration')
        
        # import ECa csv (same format, one coil per column)
        def ecaImportBtnFunc():
            fname, _ = QFileDialog.getOpenFileName(importTab, 'Select data file', self.datadir, '*.csv')
            if fname != '':
                self.fnameECa = fname
                ecaImportBtn.setText(os.path.basename(fname))
        ecaImportBtn = QPushButton('Import ECa')
        ecaImportBtn.clicked.connect(ecaImportBtnFunc)
        
        # import EC depth-specific (one depth per column) -> can be from ERT
        def ecImportBtnFunc():
            fname, _ = QFileDialog.getOpenFileName(importTab, 'Select data file', self.datadir, '*.csv')
            if fname != '':
                self.fnameEC = fname
                ecImportBtn.setText(os.path.basename(fname))
        ecImportBtn = QPushButton('Import EC profiles')
        ecImportBtn.clicked.connect(ecImportBtnFunc)
        
        # choose which forward model to use
        forwardCalibCombo = QComboBox()
        forwardCalibs = ['CS', 'FS', 'FSandrade']
        for forwardCalib in forwardCalibs:
            forwardCalibCombo.addItem(forwardCalib)
        
        # perform the fit (equations display in the console)
        def fitCalibBtnFunc():
            forwardModel = forwardCalibCombo.itemText(forwardCalibCombo.currentIndex())
            mwCalib.setCallback(self.problem.calibrate)
            mwCalib.replot(fnameECa=self.fnameECa, fnameEC=self.fnameEC,
                           forwardModel=forwardModel)
        fitCalibBtn = QPushButton('Fit calibration')
        fitCalibBtn.clicked.connect(fitCalibBtnFunc)
        
        # apply the calibration to the ECa measurements of the survey imported
        def applyCalibBtnFunc():
            infoDump('Calibration applied')
            #TODO we need a good dataset to test this !
        applyCalibBtn = QPushButton('Apply Calibration')
        applyCalibBtn.clicked.connect(applyCalibBtnFunc)
        
        
        # graph
        mwCalib = MatplotlibWidget()
        
        
        # layout
        calibLayout = QVBoxLayout()
        calibOptions = QHBoxLayout()
        calibOptions.addWidget(ecaImportBtn)
        calibOptions.addWidget(ecImportBtn)
        calibOptions.addWidget(forwardCalibCombo)
        calibOptions.addWidget(fitCalibBtn)
        calibOptions.addWidget(applyCalibBtn)
        calibLayout.addLayout(calibOptions)
        calibLayout.addWidget(mwCalib)
        
        calibTab.setLayout(calibLayout)
        
        
        #%% error model
        errTab = QTabWidget()
        tabs.addTab(errTab, 'Error Modelling')
        
        surveyErrCombo = QComboBox()
        coilErrCombo = QComboBox()
        
        def fitErrBtnFunc():
            index = surveyErrCombo.currentIndex()
            coil = coilErrCombo.itemText(coilErrCombo.currentIndex())
            mwErr.setCallback(self.problem.crossOverPoints)
            mwErr.replot(index=index, coil=coil, dump=infoDump)
            mwErrMap.setCallback(self.problem.plotCrossOverMap)
            mwErrMap.replot(index=index, coil=coil)
        fitErrBtn = QPushButton('Fit Error Model based on colocated measurements')
        fitErrBtn.clicked.connect(fitErrBtnFunc)
        
        
        # graph
        mwErr = MatplotlibWidget()
        mwErrMap = MatplotlibWidget()
        
        # layout
        errLayout = QVBoxLayout()
        errOptionLayout = QHBoxLayout()
        errOptionLayout.addWidget(surveyErrCombo)
        errOptionLayout.addWidget(coilErrCombo)
        errOptionLayout.addWidget(fitErrBtn)
        errLayout.addLayout(errOptionLayout)
        errGraphLayout = QHBoxLayout()
        errGraphLayout.addWidget(mwErrMap)
        errGraphLayout.addWidget(mwErr)
        errLayout.addLayout(errGraphLayout)
        
        errTab.setLayout(errLayout)
        
        
        
        #%% inversion settings (starting model + lcurve)
        settingsTab = QTabWidget()
        tabs.addTab(settingsTab, 'Inversion Settings')
        
        class ModelTable(QTableWidget):
            def __init__(self, nrow=3, headers=['Bottom depth of layer [m]', 'EC [mS/m]']):
                ncol = len(headers)
                super(ModelTable, self).__init__(nrow, ncol)
                self.nrow = nrow
                self.ncol = ncol
                self.headers = headers
                self.setHorizontalHeaderLabels(self.headers)
                self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
                
            def setTable(self, depths0, conds0):
                self.clear()
                for i, depth in enumerate(depths0):
                    self.setItem(i, 0, QTableWidgetItem('{:.3f}'.format(depth)))
                for j in range(i+1, self.nrow-1):
                    self.setItem(j, 0, QTableWidgetItem('0.0'))
                self.setItem(self.nrow - 1, 0, QTableWidgetItem('-'))
                self.item(self.nrow - 1, 0).setFlags(Qt.ItemIsEnabled)
                for i, cond in enumerate(conds0):
                    self.setItem(i, 1, QTableWidgetItem('{:.2f}'.format(cond)))
                for j in range(i+1, self.nrow):
                    self.setItem(j, 1, QTableWidgetItem('20.0'))

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
                    conds0[i] = float(self.item(i, 1).text())
                return depths0, conds0
            
            def addRow(self):
                depths0, conds0 = self.getTable()
                self.nrow += 1
                self.setRowCount(self.nrow)
                self.setTable(depths0, conds0)
                
            def delRow(self):
                depths0, conds0 = self.getTable()
                self.nrow -= 1
                self.setTable(depths0, conds0)
                self.setRowCount(self.nrow)
                

        
        modelLabel = QLabel('Depth of bottom layer and starting conductivity of each layer.')
        
        def addRowBtnFunc():
            modelTable.addRow()
        addRowBtn = QPushButton('Add Row')
        addRowBtn.clicked.connect(addRowBtnFunc)
        
        def delRowBtnFunc():
            modelTable.delRow()
        delRowBtn = QPushButton('Remove Row')
        delRowBtn.clicked.connect(delRowBtnFunc)
        
        modelTable = ModelTable()
        modelTable.setTable(self.problem.depths0, self.problem.conds0)
        
        
        def lcurveBtnFunc():
            mwlcurve.plot(self.problem.lcurve)
        lcurveBtn = QPushButton('Fit L-curve')
        lcurveBtn.clicked.connect(lcurveBtnFunc)
        
        # graph
        mwlcurve = MatplotlibWidget()
        
        
        # layout
        settingsLayout = QHBoxLayout()
        
        invStartLayout = QVBoxLayout()
        invStartLayout.addWidget(modelLabel)
        invStartLayout.addWidget(addRowBtn)
        invStartLayout.addWidget(delRowBtn)
        invStartLayout.addWidget(modelTable)
        settingsLayout.addLayout(invStartLayout)
        
        invlcurveLayout = QVBoxLayout()
        invlcurveLayout.addWidget(lcurveBtn)
        invlcurveLayout.addWidget(mwlcurve)
        settingsLayout.addLayout(invlcurveLayout)
        
        
        settingsTab.setLayout(settingsLayout)
        
        
        #%% invert graph
        invTab = QTabWidget()
        tabs.addTab(invTab, 'Inversion')
        
        
        forwardCombo = QComboBox()
        forwardModels = ['CS', 'CS (fast)', 'FS', 'FSandrade']
        for forwardModel in forwardModels:
            forwardCombo.addItem(forwardModel)
        
        methodCombo = QComboBox()
        methods = ['CG', 'L-BFGS-B', 'TNC', 'Nelder-Mead']
        for method in methods:
            methodCombo.addItem(method)
        
        alphaLabel = QLabel('Damping factor:')
        alphaEdit = QLineEdit('0.07')
        alphaEdit.setValidator(QDoubleValidator())
        
        lCombo = QComboBox()
        lCombo.addItem('l1')
        lCombo.addItem('l2')
        lCombo.setCurrentIndex(1)
        
        nitLabel = QLabel('Nit:')
        nitEdit = QLineEdit('15')
        nitEdit.setToolTip('Maximum Number of Iterations')
        nitEdit.setValidator(QIntValidator())
        
        
        def logTextFunc(arg):
            logText.setText(arg)
            QApplication.processEvents()
        logText = QTextEdit('hellow there !')
        logText.setReadOnly(True)
        
        def invertBtnFunc():
            outputStack.setCurrentIndex(0)
            logText.clear()
            
            # collect parameters
            depths0, conds0 = modelTable.getTable()
            self.problem.depths0 = depths0
            self.problem.conds0 = conds0
            regularization = lCombo.itemText(lCombo.currentIndex())
            alpha = float(alphaEdit.text()) if alphaEdit.text() != '' else 0.07
            forwardModel = forwardCombo.itemText(forwardCombo.currentIndex())
            method = methodCombo.itemText(methodCombo.currentIndex())
            depths = np.r_[[0], depths0, [-np.inf]]
            nit = float(nitEdit.text()) if nitEdit.text() != '' else 15
            sliceCombo.disconnect()
            sliceCombo.clear()
            for i in range(len(depths)-1):
                sliceCombo.addItem('{:.2f}m - {:.2f}m'.format(depths[i], depths[i+1]))
            sliceCombo.currentIndexChanged.connect(sliceComboFunc)
            
            # invert
            if forwardModel == 'CS (fast)':
                self.problem.invertGN(alpha=alpha, dump=logTextFunc)
            else:
                self.problem.invert(forwardModel=forwardModel, alpha=alpha,
                                    dump=logTextFunc, regularization=regularization,
                                    method=method, options={'maxiter':nit})
            
            # plot results
            mwInv.setCallback(self.problem.showResults)
            mwInv.replot(**showInvParams)
            mwInvMap.setCallback(self.problem.showSlice)
            mwInvMap.replot(**showInvMapParams)
            mwMisfit.plot(self.problem.showMisfit)
            mwOne2One.plot(self.problem.showOne2one)
            outputStack.setCurrentIndex(1)
            #TODO add kill feature
        invertBtn = QPushButton('Invert')
        invertBtn.clicked.connect(invertBtnFunc)
        
        
        # profile display
        showInvParams = {'index':0, 'vmin':None, 'vmax':None, 
                         'cmap':'viridis_r', 'contour':False}
        
        def cmapInvComboFunc(index):
            showInvParams['cmap'] = cmapInvCombo.itemText(index)
            mwInv.replot(**showInvParams)
        cmapInvCombo = QComboBox()
        cmaps = ['viridis_r', 'viridis', 'seismic', 'rainbow']
        for cmap in cmaps:
            cmapInvCombo.addItem(cmap)
        cmapInvCombo.currentIndexChanged.connect(cmapInvComboFunc)
        
        def surveyInvComboFunc(index):
            showInvParams['index'] = index
            mwInv.replot(**showParams)    
        surveyInvCombo = QComboBox()
        surveyInvCombo.currentIndexChanged.connect(surveyInvComboFunc)
        
        vminInvLabel = QLabel('vmin:')
        vminInvEdit = QLineEdit('')
        vminInvEdit.setValidator(QDoubleValidator())
        
        vmaxInvLabel = QLabel('vmax:')
        vmaxInvEdit = QLineEdit('')
        vmaxInvEdit.setValidator(QDoubleValidator())
        
        def applyInvBtnFunc():
            vmin = float(vminInvEdit.text()) if vminInvEdit.text() != '' else None
            vmax = float(vmaxInvEdit.text()) if vmaxInvEdit.text() != '' else None
            showInvParams['vmin'] = vmin
            showInvParams['vmax'] = vmax
            mwInv.replot(**showInvParams)
        applyInvBtn = QPushButton('Apply')
        applyInvBtn.clicked.connect(applyInvBtnFunc)
        
        contourInvLabel = QLabel('Contour:')
        def contourInvCheckFunc(state):
            showInvParams['contour'] = state
            mwInv.replot(**showInvParams)
        contourInvCheck = QCheckBox()
        contourInvCheck.clicked.connect(contourInvCheckFunc)
 
        
        
        # for the map
        showInvMapParams = {'index':0, 'islice':0, 'vmin':None, 'vmax':None, 'cmap':'viridis_r'}

        def cmapInvMapComboFunc(index):
            showInvMapParams['cmap'] = cmapInvMapCombo.itemText(index)
            mwInvMap.replot(**showInvMapParams)
        cmapInvMapCombo = QComboBox()
        cmaps = ['viridis_r', 'viridis', 'seismic', 'rainbow']
        for cmap in cmaps:
            cmapInvMapCombo.addItem(cmap)
        cmapInvMapCombo.currentIndexChanged.connect(cmapInvMapComboFunc)
        
        def surveyInvMapComboFunc(index):
            showInvMapParams['index'] = index
            mwInvMap.replot(**showInvMapParams)
        surveyInvMapCombo = QComboBox()
        surveyInvMapCombo.currentIndexChanged.connect(surveyInvMapComboFunc)
        
        vminInvMapLabel = QLabel('vmin:')
        vminInvMapEdit = QLineEdit('')
        vminInvMapEdit.setValidator(QDoubleValidator())
        
        vmaxInvMapLabel = QLabel('vmax:')
        vmaxInvMapEdit = QLineEdit('')
        vmaxInvMapEdit.setValidator(QDoubleValidator())
        
        def applyInvMapBtnFunc():
            vmin = float(vminInvMapEdit.text()) if vminInvMapEdit.text() != '' else None
            vmax = float(vmaxInvMapEdit.text()) if vmaxInvMapEdit.text() != '' else None
            showInvMapParams['vmin'] = vmin
            showInvMapParams['vmax'] = vmax
            mwInvMap.replot(**showInvMapParams)
        applyInvMapBtn = QPushButton('Apply')
        applyInvMapBtn.clicked.connect(applyInvMapBtnFunc)

        sliceLabel = QLabel('Layer:')
        def sliceComboFunc(index):
            showInvMapParams['islice'] = index
            mwInvMap.replot(**showInvMapParams)
        sliceCombo = QComboBox()
        sliceCombo.currentIndexChanged.connect(sliceComboFunc)
        
        contourInvMapLabel = QLabel('Contour:')
        def contourInvMapCheckFunc(state):
            showInvMapParams['contour'] = state
            mwInvMap.replot(**showInvMapParams)
        contourInvMapCheck = QCheckBox()
        contourInvMapCheck.clicked.connect(contourInvMapCheckFunc)
        
        
        graphTabs = QTabWidget()
        profTab = QTabWidget()
        mapTab = QTabWidget()
        graphTabs.addTab(profTab, 'Profile')
        graphTabs.addTab(mapTab, 'Slice')
        
        # graph or log        
        mwInv = MatplotlibWidget()
        mwInvMap = MatplotlibWidget()

        
        # layout
        invLayout = QVBoxLayout()
        
        invOptions = QHBoxLayout()
        invOptions.addWidget(forwardCombo, 15)
        invOptions.addWidget(methodCombo, 5)
        invOptions.addWidget(alphaLabel, 5)
        invOptions.addWidget(alphaEdit, 10)
        invOptions.addWidget(lCombo, 5)
        invOptions.addWidget(nitLabel, 2)
        invOptions.addWidget(nitEdit, 2)
        invOptions.addWidget(invertBtn, 25)
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
        
        outputRes.addWidget(graphTabs)
        
        profLayout = QVBoxLayout()
        profOptionsLayout = QHBoxLayout()
        profOptionsLayout.addWidget(surveyInvCombo)
        profOptionsLayout.addWidget(vminInvLabel)
        profOptionsLayout.addWidget(vminInvEdit)
        profOptionsLayout.addWidget(vmaxInvLabel)
        profOptionsLayout.addWidget(vmaxInvEdit)
        profOptionsLayout.addWidget(applyInvBtn)
        profOptionsLayout.addWidget(contourInvLabel)
        profOptionsLayout.addWidget(contourInvCheck)
        profOptionsLayout.addWidget(cmapInvCombo)
        profLayout.addLayout(profOptionsLayout)
        profLayout.addWidget(mwInv)
        profTab.setLayout(profLayout)


        mapLayout = QVBoxLayout()
        mapOptionsLayout = QHBoxLayout()
        mapOptionsLayout.addWidget(surveyInvMapCombo)
        mapOptionsLayout.addWidget(sliceLabel)
        mapOptionsLayout.addWidget(sliceCombo)
        mapOptionsLayout.addWidget(vminInvMapLabel)
        mapOptionsLayout.addWidget(vminInvMapEdit)
        mapOptionsLayout.addWidget(vmaxInvMapLabel)
        mapOptionsLayout.addWidget(vmaxInvMapEdit)
        mapOptionsLayout.addWidget(applyInvMapBtn)
        mapOptionsLayout.addWidget(cmapInvMapCombo)
        mapOptionsLayout.addWidget(contourInvMapLabel)
        mapOptionsLayout.addWidget(contourInvMapCheck)
        mapLayout.addLayout(mapOptionsLayout)
        mapLayout.addWidget(mwInvMap)
        mapTab.setLayout(mapLayout)
        
        outputLog.addWidget(logText)
        
        invLayout.addLayout(outputStack)
        
        invTab.setLayout(invLayout)
        
        
        #%% goodness of fit
        postTab = QTabWidget()
        tabs.addTab(postTab, 'Post-processing')
        
        misfitLabel = QLabel('Misfit after inversion')
        
        mwMisfit = MatplotlibWidget()
        mwOne2One = MatplotlibWidget()
        
        # layout
        postLayout = QVBoxLayout()
        postLayout.addWidget(misfitLabel)
        graphLayout = QHBoxLayout()
        graphLayout.addWidget(mwMisfit)
        graphLayout.addWidget(mwOne2One)
        postLayout.addLayout(graphLayout)
        
        postTab.setLayout(postLayout)
        
        
        #about tab
        #%% general Ctrl+Q shortcut + general tab layout

        layout.addWidget(tabs)
        layout.addWidget(errorLabel)
        self.table_widget.setLayout(layout)
        self.setCentralWidget(self.table_widget)
        self.show()


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
#    catchErrors()
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
    print('importing matplotlib')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    from matplotlib import rcParams
    rcParams.update({'font.size': 12}) # CHANGE HERE for graph font size
    progressBar.setValue(2)
    app.processEvents()

    print('importing numpy')
    import numpy as np
    progressBar.setValue(4)
    app.processEvents()

    print ('importing pandas')
    progressBar.setValue(6)
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
