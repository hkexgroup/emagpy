#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:14:25 2019

@author: jkl
"""


from PyQt5 import QtCore

from ui import App
import time

testdir = 'emagpy/test/'
sleepTime = 1 # s

#%%

def test_importing(qtbot, qapp):
    app = App() 
    qtbot.addWidget(app)
#    qtbot.mouseClick(app.importBtn, QtCore.Qt.LeftButton)
    app.processFname(testdir + 'coverCrop.csv')
    def proc():
        qapp.processEvents()
        time.sleep(sleepTime)
    qapp.processEvents()
    app.vminEdit.setText('0')
    app.vmaxEdit.setText('50')
    qtbot.mouseClick(app.keepApplyBtn, QtCore.Qt.LeftButton)
    qtbot.keyClick(app.coilCombo, QtCore.Qt.Key_Up)
    qtbot.keyClick(app.coilCombo, QtCore.Qt.Key_Up)
    qtbot.keyClick(app.coilCombo, QtCore.Qt.Key_Up)
    qtbot.keyClick(app.coilCombo, QtCore.Qt.Key_Up)
    qtbot.keyClick(app.coilCombo, QtCore.Qt.Key_Up)
    qtbot.mouseClick(app.rollingBtn, QtCore.Qt.LeftButton)
    qtbot.mouseClick(app.mapRadio, QtCore.Qt.LeftButton)
    qtbot.keyClick(app.coilCombo, QtCore.Qt.Key_Down)
    qtbot.keyClick(app.coilCombo, QtCore.Qt.Key_Down)
    qtbot.keyClick(app.coilCombo, QtCore.Qt.Key_Down)
    qtbot.keyClick(app.coilCombo, QtCore.Qt.Key_Down)
    qtbot.keyClick(app.coilCombo, QtCore.Qt.Key_Down)
    qtbot.mouseClick(app.contourCheck, QtCore.Qt.LeftButton)
    qtbot.mouseClick(app.ptsCheck, QtCore.Qt.LeftButton)
    app.vminfEdit.setText('0')
    app.vmaxfEdit.setText('30')
    qtbot.mouseClick(app.applyBtn, QtCore.Qt.LeftButton)
    proc()

    


