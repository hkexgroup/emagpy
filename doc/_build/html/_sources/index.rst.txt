.. EmagPy documentation master file, created by
   sphinx-quickstart on Wed Aug 29 00:30:45 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

    
Welcome to EMagPy's documentation!
==================================
   
.. contents:: Table of contents

Graphical User Interface
------------------------
The Graphical User Interface (GUI) is composed of differents tabs.

.. _importing:
.. figure:: image/importing.png
    :alt: importing tab
    
    Importing tab with the `coverCrop.csv` dataset. Filtering options are availables.

.. _importingMap:
.. figure:: image/importing-map.png
    :alt: importing tab with map view
    
    The dataset can also be viewed as a contour map if spatial data available.
    

In :numref:`importing` we can see the general tabbed interface with the `coverCrop.csv` dataset imported. The :numref:`importingMap` show the same dataset but plotted as a contour map.    






API documentation
-----------------
.. toctree::
   :maxdepth: 2
   :caption: Contents:
.. automodule:: emagpy.Problem
   :members:
.. automodule:: emagpy.Survey
   :members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
