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
from scipy import interpolate

class forceSim:
    """
    This class defines external forcing parameters.

    Parameters
    ----------
    class: input
        Input parameter class.

    class: sGrid
        Surface parameter class.
    """

    def __init__(self, input = None, sGrid = None):


        self.sea0 = input.seaval
        self.seafile = input.seafile
        self.sealevel = None
        self.seatime = None
        self.seaFunc = None

        self.temp0 = input.tempval
        self.tempfile = input.tempfile
        self.temptime = None
        self.tempval = None
        self.tempFunc = None

        self.sal0 = input.salval
        self.salfile = input.salfile
        self.saltime = None
        self.salval = None
        self.salFunc = None

        self.ph0 = input.phval
        self.phfile = input.phfile
        self.phtime = None
        self.phval = None
        self.phFunc = None

        self.wclim = 0
        self.wavU = None
        self.wavV = None
        self.wavPerc = None
        #self.wavH = None
        #self.wavP = None
        #self.wavL = None

        self.Map_disp = input.tectFile
        self.T_disp = input.tectTime
        self.next_disp = None

        self.next_display = None
        self.next_layer = None
        self.next_wave = None
        self.next_disp = None
        self.next_carb = None
        self.time_layer = input.laytime
        self.time_wave = input.tWave
        self.time_display = input.tDisplay

        self.sGrid = sGrid
        self.Afac = input.Afactor

        minX = sGrid.demX.min()
        maxX = sGrid.demX.max()
        minY = sGrid.demY.min()
        maxY = sGrid.demY.max()
        demRes = sGrid.demX[1]-sGrid.demX[0]
        self.demnx = int(round((maxX-minX)/demRes+1))
        self.demny = int(round((maxY-minY)/demRes+1))

        if self.Afac != 1:
            self.demX = numpy.arange(minX,maxX+demRes,demRes)
            self.demY = numpy.arange(minY,maxY+demRes,demRes)

        if self.seafile != None:
            self._build_Sea_function()

        if self.tempfile != None:
            self._build_Temperature_function()

        if self.salfile != None:
            self._build_Salinity_function()

        if self.phfile != None:
            self._build_Acidity_function()

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

    def _build_Acidity_function(self):
        """
        Using Pandas library to read the ocean acidification file and define ph interpolation
        function based on Scipy 1D linear function.
        """

        # Read temperature file
        phdata = pandas.read_csv(self.phfile, sep=r'\s+', engine='c',
                               header=None, na_filter=False,
                               dtype=numpy.float, low_memory=False)

        self.phtime = phdata.values[:,0]
        tmp = phdata.values[:,1]
        self.phFunc = interpolate.interp1d(self.phtime, tmp, kind='linear')

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

    def getph(self, time):
        """
        Computes for a given time the ocean ph according to input file parameters.

        Parameters
        ----------
        float : time
            Requested time for which to compute sea ph value.
        """

        if self.phfile == None:
            self.phval = self.ph0
        else:
            if time < self.phtime.min():
                time = self.phtime.min()
            if time > self.phtime.max():
                time = self.phtime.max()
            self.phval = self.phFunc(time)

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

    def load_Tecto_map(self, time):
        """
        Load vertical displacement map for a given period and perform interpolation from dem grid to pyReef mesh.

        Parameters
        ----------
        float : time
            Requested time interval displacement map to load.

        Return
        ----------
        variable: dispRate
            Numpy array containing the updated displacement rate for the considered domain.
        """

        events = numpy.where( (self.T_disp[:,1] - time) <= 0)[0]
        event = len(events)

        if not (time >= self.T_disp[event,0]) and not (time < self.T_disp[event,1]):
            raise ValueError('Problem finding the displacements map to load!')

        self.next_disp = self.T_disp[event,1]
        dispRate = numpy.zeros((self.sGrid.nx,self.sGrid.ny), dtype=float)

        if self.Map_disp[event] != None:
            dispMap = pandas.read_csv(str(self.Map_disp[event]), sep=r'\s+', engine='c', header=None, na_filter=False, \
                               dtype=numpy.float, low_memory=False)

            if self.Afac == 1:
                rectDisp = numpy.reshape(dispMap.values,(self.demnx,self.demny),order='F')
            else:
                rDisp = numpy.reshape(dispMap.values,(self.demnx,self.demny),order='F')
                interpolate_fct = interpolate.interp2d(self.demY,self.demX,rDisp,kind='cubic')
                rectDisp = interpolate_fct(self.sGrid.regY[1:self.ny-1],self.sGrid.regX[1:self.nx-1])

            # Define internal nodes displacements
            dispRate[1:1+rectDisp.shape[0],1:1+rectDisp.shape[1]] = rectDisp
            # Define ghosts displacements
            dispRate[0,:] = dispRate[1,:]
            dispRate[self.sGrid.nx-1,:] = dispRate[self.sGrid.nx-2,:]
            dispRate[:,0] = dispRate[:,1]
            dispRate[:,self.sGrid.ny-1] = dispRate[:,self.sGrid.ny-2]
            dispRate[0,0] = dispRate[1,1]
            dispRate[self.sGrid.nx-1,0] = dispRate[self.sGrid.nx-2,1]
            dispRate[self.sGrid.nx-1,self.ny-1] = dispRate[self.sGrid.nx-2,self.sGrid.ny-2]
            dispRate[0,self.sGrid.ny-1] = dispRate[1,self.sGrid.ny-2]

            dt = (self.T_disp[event,1] - self.T_disp[event,0])
            if dt <= 0:
                raise ValueError('Problem computing the displacements rate for event %d.'%event)
            dispRate = dispRate / dt

        return dispRate
