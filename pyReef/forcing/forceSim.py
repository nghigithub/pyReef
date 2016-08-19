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

    string : tempfile
        Path to the temperature file (if any).

    float : temp0
        Sea-surface temperature value in case no temperature curve is provided (default is 25.).

    string : salfile
        Path to the salinity file (if any).

    float : sal0
        Sea-surface salinity value in case no salinity curve is provided (default is 35.5).

    float : Twave
        Wave computation interval (in years).

    float : Tdisplay
        Display interval (in years).
    """

    def __init__(self, seafile = None, sea0 = 0., tempfile = None, temp0 = 25., salfile = None, sal0 = 35.5,
                 waveNb = 0, waveTime = None, wavePerc = None, waveWu = None, waveWd = None, Twave = 0.,
                 Tdisplay = 0.):

        self.sea0 = sea0
        self.seafile = seafile
        self.sealevel = None
        self.seatime = None
        self.seaFunc = None

        self.temp0 = temp0
        self.tempfile = tempfile
        self.temptime = None
        self.tempval = None
        self.tempFunc = None

        self.sal0 = sal0
        self.salfile = salfile
        self.saltime = None
        self.salval = None
        self.salFunc = None

        self.waveNb = waveNb
        self.waveTime = waveTime
        self.wavePerc = wavePerc
        self.waveWu = waveWu
        self.waveWd = waveWd

        self.wNb = 0
        self.wPerc = None
        self.wDt = None
        self.wWindU = None
        self.wWindA = None

        self.next_display = None
        self.next_layer = None
        self.next_wave = None
        self.time_wave = Twave
        self.time_display = Tdisplay

        if self.seafile != None:
            self._build_Sea_function()

        if self.tempfile != None:
            self._build_Temperature_function()

        if self.salfile != None:
            self._build_Salinity_function()

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
        tmp = seadata.values[:,1]
        self.seaFunc = interpolate.interp1d(self.seatime, tmp, kind='linear')

        return

    def _build_Temperature_function(self):
        """
        Using Pandas library to read the sea-surface temperature file and define temperature interpolation
        function based on Scipy 1D linear function.
        """

        # Read temperature file
        tempdata = pandas.read_csv(self.tempfile, sep=r'\s+', engine='c',
                               header=None, na_filter=False,
                               dtype=numpy.float, low_memory=False)

        self.temptime = tempdata.values[:,0]
        tmp = tempdata.values[:,1]
        self.tempFunc = interpolate.interp1d(self.temptime, tmp, kind='linear')

        return

    def _build_Salinity_function(self):
        """
        Using Pandas library to read the sea-surface salinity file and define temperature interpolation
        function based on Scipy 1D linear function.
        """

        # Read salinity file
        saldata = pandas.read_csv(self.salfile, sep=r'\s+', engine='c',
                               header=None, na_filter=False,
                               dtype=numpy.float, low_memory=False)

        self.seatime = saldata.values[:,0]
        tmp = saldata.values[:,1]
        self.salFunc = interpolate.interp1d(self.saltime, tmp, kind='linear')

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


    def getTemperature(self, time):
        """
        Computes for a given time the sea-surface temperature according to input file parameters.

        Parameters
        ----------
        float : time
            Requested time for which to compute sea temperature value.
        """

        if self.tempfile == None:
            self.tempval = self.temp0
        else:
            if time < self.temptime.min():
                time = self.temptime.min()
            if time > self.temptime.max():
                time = self.temptime.max()
            self.tempval = self.tempFunc(time)

        return

    def getSalinity(self, time):
        """
        Computes for a given time the sea-surface salinity according to input file parameters.

        Parameters
        ----------
        float : time
            Requested time for which to compute sea salinity value.
        """

        if self.salfile == None:
            self.salval = self.sal0
        else:
            if time < self.saltime.min():
                time = self.saltime.min()
            if time > self.saltime.max():
                time = self.saltime.max()
            self.salval = self.salFunc(time)

        return
