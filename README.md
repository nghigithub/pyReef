
pyReef - A Python-based Carbonate Platform Model
=====

_Numerical model of carbonate platform development under changing climatic conditions, 3D stratigraphic geometries and facies evolution._

<div align="center">
    <img width=650 src="https://github.com/pyReef-model/pyReef/blob/master/Test/data/pyReef_sketch.png" alt="sketch pyReef" title="sketch of pyReef range of models."</img>
</div>
_A schematic view of 2D carbonate platform evolution model illustrating the main variables and forces simulated with_ **pyReef**: __w__ _refers to wave generation,_ __sl__ _to sea-level fluctuations,_ __c__ _to longshore and undertow currents and_ __t__ _to tectonic forces._

## Overview

**pyReef** is a stratigraphic forward model that predicts reef evolution over geological time scale. This parallel, open-source model uses [**swan**](http://swanmodel.sourceforge.net) (wave generation model) to simulate wave propagation under different climatic conditions and associated sediment transport. 

Multiple carbonate facies are simulated using a set of fuzzy logic rules from [**scikit-fuzzy**](https://github.com/scikit-fuzzy/scikit-fuzzy) which are based on wave energy, sedimentation and depth as well as tectonic and oceanic forcings (ocean temperature and salinity, eustasy). 

This work is conducted within the USyd School of Geosciences [**Geoscoastal Group**](http://sydney.edu.au/science/geosciences/research/re_geocoastal.shtml) and relies on the group extensive dataset obtained at [**One Tree Reef**](http://sydney.edu.au/science/oti/) (Great Barrier Reef).
