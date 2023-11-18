.. EmagPy documentation master file, created by
   sphinx-quickstart on Wed Aug 29 00:30:45 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

    
EMagPy python API and standalone GUI
====================================

Installation
------------

**You can find executables for Windows, Mac and Linux on:**

https://gitlab.com/hkex/emagpy


See below instruction if you want to run EMagPy from source (for development purpose) or if you want to only install the API (pypi package "emagpy") without graphical user interface.

Clone the gitlab repository::

    git clone https://gitlab.com/hkex/emagpy

To start the GUI from source, navigate through the `src` directory and run `ui.py`::

    cd emagpy/src
    python ui.py
    
From the same `src` directory you can import the module from source using python. Another solution is to install the module from pypi using pip::

    pip install emagpy


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   getting-started
   gui
   api
   gallery/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


