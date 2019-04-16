# EMagPy

Python API for inversion/modelling of electromagnetic data.

EMagPy is divided into a python API and a standalone graphical user interface (GUI).
It aims to be a powerfull but simple tool for inverting EMI data obtain from conductimeter.

Below os a draft of the API and GUI to be implemented.

API structure:
- Survey class (coil specific)
    - df: main dataframe (one column per coil, one row per location)
    - read()
    - interpolate() (kriging or gridding)
    - crossOverError() using cross-over points
- CalibrationData class (EC, ECa)
    - apply(Survey)
    - show()
- Model class (depth specific)
    - df: main dataframe (one column per layer/depths, one row per location)
    - show()
    - setDepths()
    - setEC()
- Problem class (master class)
    - surveys: list of Survey object
    - models: list of Model object
    - invert(forwardModel='CS',
             method='SCEUA/TNC/',
             constrain='quasi2D', 'quasi3D', 'none')
    - forwardEM()
    - forwardCS()
    

UI structure:
- tab1: data import (choice of sensors)
- tab2: calibration + error model + co location
- tab3: inversion settings
    - model definition (layer, initial model)
    - inversion (smoothing, lateral constrain, full or CS, choice of method for minimize)
- tab4: display of inverted section + export graph/data
- tab5: goodness of fit (1:1) and 2D graph