##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the pyReef carbonate platform modelling application.     ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
This module defines several functions used to force pyReef simulation with external
processes related to wave climate, wind field and sea level.
"""
import os
import numpy
import pandas
import mpi4py.MPI as mpi
from scipy.ndimage.filters import gaussian_filter
from scipy import interpolate
from scipy.spatial import cKDTree

class forceSim:
    """
    This class defines external forcing parameters.

    Parameters
    ----------
    string : seafile
        Path to the sea level fluctuation file (if any).

    float : sea0
        Relative sea level position in case no sea level curve is provided (default is 0.).

    string : MapRain
        Numpy array containing the rain map file names.

    float : TimeRain
        Numpy array containing the start and end times for each rain event in years.

    float : ValRain
        Value of precipitation rate for each rain event in m/a.

    bool : orographic
        Numpy boolean array defining orographic calculation if any.

    float : rbgd
        Numpy array of background precipitation.

    float : rmin
        Numpy array of minimal precipitation.

    float : rmax
        Numpy array of maximal precipitation.

    float : windx
        Numpy array of wind velocity along X.

    float : windy
        Numpy array of wind velocity along Y.

    float : tauc
        Numpy array of time conversion from cloud water to hydrometeors.

    float : tauf
        Numpy array of time for hydrometeor fallout.

    float : nm
        Numpy array of moist stability frequency.

    float : cw
        Numpy array of uplift sensitivity factor.

    float : hw
        Numpy array of depth of the moist layer.

    float : ortime
        Numpy array of rain computation time step.

    string : MapDisp
        Numpy array containing the cumulative displacement map file names.

    float : TimeDisp
        Numpy array containing the start and end times for each displacement period in years.

    float : regX
        Numpy array containing the X-coordinates of the regular input grid.

    float : regY
        Numpy array containing the Y-coordinates of the regular input grid.

    float : Tdisplay
        Display interval (in years).
    """

    def __init__(self, seafile = None, sea0 = 0., MapRain = None, TimeRain = None, ValRain = None,
                 orographic = None, rbgd = None, rmin = None, rmax = None, windx = None, windy = None,
                 tauc = None, tauf = None, nm = None, cw = None, hw = None, ortime = None, MapDisp = None,
                 TimeDisp = None, regX = None, regY = None, Tdisplay = 0.):

        self.regX = regX
        self.regY = regY
        self.xi, self.yi = numpy.meshgrid(regX, regY, indexing='xy')
        self.xyi = numpy.dstack([self.xi.flatten(), self.yi.flatten()])[0]
        self.tree = None
        self.dx = None

        self.sea0 = sea0
        self.seafile = seafile
        self.sealevel = None
        self.seatime = None
        self.seaval = None
        self.seaFunc = None

        self.next_display = None
        self.next_layer = None
        self.time_display = Tdisplay

        if self.seafile != None:
            self._build_Sea_function()

        return

    def _build_Sea_function(self):
        """
        Using Pandas library to read the sea level file and define sea level interpolation
        function based on Scipy 1D linear function.
        """

        # Read sea level file
        seadata = pandas.read_csv(self.seafile, sep=r'\s+', engine='c',
                               header=None, na_filter=False,
                               dtype=numpy.float, low_memory=False)

        self.seatime = seadata.values[:,0]
        self.seaval = seadata.values[:,1]
        self.seaFunc = interpolate.interp1d(self.seatime, self.seaval, kind='linear')

        return

    def getSea(self, time):
        """
        Computes for a given time the sea level according to input file parameters.

        Parameters
        ----------
        float : time
            Requested time for which to compute sea level elevation.
        """

        if self.seafile == None:
            self.sealevel = self.sea0
        else:
            if time < self.seatime.min():
                time = self.seatime.min()
            if time > self.seatime.max():
                time = self.seatime.max()
            self.sealevel = self.seaFunc(time)

        return
