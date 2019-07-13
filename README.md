# EMagPy

Python API for inversion/modelling of frequency domain electromagnetic data (FDEM).

EMagPy is divided into a python API and a standalone graphical user interface (GUI).
It aims to be a powerfull but simple tool for inverting EMI data obtain from conductimeter.

# Getting started
Clone the repository:
```sh
git clone https://gitlab.com/hkex/emagpy
```
Change to the `src` directory and run `ui.py` to start the GUI.
```sh
cd emgapy/src
python ui.py # this will start the GUI
```
The python API is available by simply importing the `emagpy` module from the python shell:
```python
import emagpy
k = Problem()
k.createSurvey('./test/coverCrop.csv')
k.invert(forwardModel='CS') # specify the forward model (here the Cumulative Sensitivty of McNeil1980)
k.showResults() # display the section
k.showMisfit() # display predicted and observed apparent EC
k.showOne2one() # 1:1 line of misfit of apparent EC
```

Check out the jupyter notebook examples in `emagpy/notebooks/`.


