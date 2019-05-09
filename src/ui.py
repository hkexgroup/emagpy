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
    def __init__(self, parent=None, navi=False, itight=False, threed=False):
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
            ax.set_aspect('equal')
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
        callback(ax=self.axis, **kwargs)
        if self.itight is True:
            self.figure.tight_layout()
        self.canvas.draw()
#
#    def setCallback(self, callback):
#        self.callback = callback
#
#    def replot(self, threed=False, **kwargs):
#        self.figure.clear()
#        if threed is False:
#            ax = self.figure.add_subplot(111)
#        else:
#            ax = self.figure.add_subplot(111, projection='3d')
#        self.axis = ax
#        self.callback(ax=ax, **kwargs)
#        ax.set_aspect('auto')
#        if self.itight is True:
#            self.figure.tight_layout()
#        self.canvas.draw()

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
        
        # import data
        def importBtnFunc():
            fname, _ = QFileDialog.getOpenFileName(importTab, 'Select data file', self.datadir, '*.csv')
            self.problem.createSurvey(fname)
            mwRaw.plot(self.problem.show)
            # fill the combobox with survey and coil names
            coilCombo.clear()
            for coil in self.problem.coils:
                coilCombo.addItem(coil)
            infoDump(fname + ' well imported')
            coilCombo.setEnabled(True)
            showRadio.setEnabled(True)
            mapRadio.setEnabled(True)
            
        
        importBtn = QPushButton('Import Data')
        importBtn.clicked.connect(importBtnFunc)
        
        # select type of sensors
        def sensorComboFunc(index):
            print('sensor selected is:', sensors[index])
        sensorCombo = QComboBox()
        sensors = ['CMD Mini-Explorer (GF-Instruments)',
                   'CMD Explorer (GF-Instruments)'
                   ]
        sensors = sorted(sensors)
        sensors = ['Other'] + sensors
        for sensor in sensors:
            sensorCombo.addItem(sensor)
        sensorCombo.currentIndexChanged.connect(sensorComboFunc)
        
        # TODO add a QCombBox for the the surveys (see self.problem.surveys)
        # for each survey, get the name using Survey.name
        
        def coilComboFunc(index):
            print('ploting raw data of', self.problem.coils[index])
            showParams['coil'] = self.problem.coils[index]
            mwRaw.plot(self.problem.show, **showParams) # add arguments for vmin/vmax
        coilCombo = QComboBox()
        coilCombo.currentIndexChanged.connect(coilComboFunc)
        coilCombo.setEnabled(False)
        
        showParams = {'coil':None, 'contour':False, 'vmin':None, 'vmax':None,
                'pts':False} 
  
        def showRadioFunc(state):
            print('show:', state)
            mwRaw.plot(self.problem.show, **showParams) # add arguments for vmin/vmax
        showRadio = QRadioButton('Raw')
        showRadio.setChecked(True)
        showRadio.toggled.connect(showRadioFunc)
        showRadio.setEnabled(False)
        def mapRadioFunc(state):
            print('map:', state)
            mwRaw.plot(self.problem.showMap, **showParams) # add arguments for vmin/vmax
        mapRadio = QRadioButton('Map')
        mapRadio.setEnabled(False)
        mapRadio.setChecked(False)
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
    
        
        
        '''
        TODO options:
        - select coils -> QComboBox (get the coil list from self.problem.coils)
            - two buttons one for show() one for showMap() all inside a QGroupButton with setExclusive(True
            see https://stackoverflow.com/questions/12472817/qt-squared-radio-button)
        - vmin/vmax as QLineEdit() with double validator and QLabel
        - apply button to apply the vmin/vmax
        - QCombox to change the colorscale (showMap only, disable for show)
        some options needs to hidden (.setVisible(False)) is show() or showMap is done
        '''
        # display it
        mwRaw = MatplotlibWidget()
        
                                   
        
        
        # layout
        importLayout = QVBoxLayout()
        
        topLayout = QHBoxLayout()
        topLayout.addWidget(importBtn)
        topLayout.addWidget(sensorCombo)
        
        midLayout = QHBoxLayout()
        midLayout.addWidget(QLabel('Plot Raw Data'))
        midLayout.addWidget(coilCombo)
      
        midLayout.addWidget(showGroup)
        
        
        importLayout.addLayout(topLayout)
        importLayout.addLayout(midLayout)
        importLayout.addWidget(mwRaw)
        
        importTab.setLayout(importLayout)
        
        
        '''
        TODO
        - tab1: data import (choice of sensors)
        - tab2: calibration + error model
        - tab3: inversion settings
            - model definition (layer, initial model)
            - inversion (smoothing, lateral constrain, full or CS, choice of method for minimize)
        - tab4: display of inverted section + export graph/data
        - tab5: goodness of fit (1:1) and 2D graph
        '''
        
        
        '''TODO add filtering tab ?
        - add filtering tab with vmin/vmax filtering
        - pointsKiller
        - regridding of spatial data
        - rolling mean
        - how replace point by :
        
        '''
        #%% calibration and error model
        calibTab = QTabWidget()
        tabs.addTab(calibTab, 'Calibration and error model')
        
        '''
        TODO as subtabs ? in a QHBoxLayout ?
        calibration:
            - QPushButton for importing calibration data (ECa)
            - QPushButton for importing calibration data (EC profiles)
            - QCombox for which forward model to use for the EC->ECa
            - QPushButton for plotting the calibration (Problem.calibrate())
            -> this plot the calibration graph
            - QPushButton to apply the calibration
        error model:
            - QPushButton to fit error model
            -> this plot the error model graph
        '''
        
        
        
        # layout
        calibLayout = QVBoxLayout()
        
        calibTab.setLayout(calibLayout)
        
        
        #%% inverted section + export
        invTab = QTabWidget()
        tabs.addTab(invTab, 'Inversion')
        
        '''TODO
        - combobox with forward choice
        - combobox with l1/l2
        - combobox with 1d, 2d, 3d or 4d inversion
        - qlineedit with damping factor
        - qpushbutton for lcurve ?
        - multi-tab or qstacked layout with:
            - .showResults() + surveyCombo vmin/vmax apply cmap exportModels
            - .lcurve() ?
            - current log while doing inversion
        '''
        
        # layout
        invLayout = QVBoxLayout()
        
        invTab.setLayout(invLayout)
        
        
        #%% goodness of fit
        postTab = QTabWidget()
        tabs.addTab(postTab, 'Post-processing')
        
        '''TODO
        - .showMisfit()
        '''
        
        # layout
        postLayout = QVBoxLayout()
        
        postTab.setLayout(postLayout)
        
        
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
    catchErrors()
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
    import pandas as pd
    progressBar.setValue(6)
    app.processEvents()

    print('importing python libraries')
    from datetime import datetime
    progressBar.setValue(8)
    app.processEvents()

    # library needed for update checker + wine checker
    print('other needed libraries')
    import platform
    OS = platform.system()
    from subprocess import PIPE, Popen
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
