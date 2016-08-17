##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the pyReef carbonate platform modelling application.     ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
This module encapsulates parsing functions of pyReef XmL input file.
"""
import os
import glob
import numpy
import shutil
import xml.etree.ElementTree as ET

class xmlParser:
    """
    This class defines XmL input file variables.

    Parameters
    ----------
    string : inputfile
        The XmL input file name.
    """

    def __init__(self, inputfile = None, makeUniqueOutputDir=True):
        """
        If makeUniqueOutputDir is set, we create a uniquely-named directory for
        the output. If it's clear, we blindly accept what's in the XML file.
        """

        if inputfile==None:
            raise RuntimeError('XmL input file name must be defined to run a pyReef simulation.')
        if not os.path.isfile(inputfile):
            raise RuntimeError('The XmL input file name cannot be found in your path.')
        self.inputfile = inputfile

        self.demfile = None
        self.Afactor = 1

        self.tStart = None
        self.tEnd = None
        self.dt = None
        self.tWave = None
        self.tDisplay = None

        self.seaval = 0.
        self.seafile = None

        self.tempval = 25.
        self.tempfile = None

        self.salval = 35.5
        self.salfile = None

        self.waveBase = 10000.
        self.waveNb = 0
        self.waveWind = False
        self.waveParam = False
        self.waveTime = None
        self.wavePerc = None
        self.waveWu = None
        self.waveWd = None
        self.waveDir = None
        self.waveHs = None
        self.wavePer = None
        self.waveDs = None

        self.makeUniqueOutputDir = makeUniqueOutputDir

        self.outDir = None

        self.h5file = 'h5/surf.time'
        self.xmffile = 'xmf/surf.time'
        self.xdmffile = 'surf.series.xdmf'

        self._get_XmL_Data()

        return

    def _get_XmL_Data(self):
        """
        Main function used to parse the XmL input file.
        """

        # Load XmL input file
        tree = ET.parse(self.inputfile)
        root = tree.getroot()

        # Extract grid structure information
        grid = None
        grid = root.find('grid')
        if grid is not None:
            element = None
            element = grid.find('demfile')
            if element is not None:
                self.demfile = element.text
                if not os.path.isfile(self.demfile):
                    raise ValueError('DEM file is missing or the given path is incorrect.')
            else:
                raise ValueError('Error in the definition of the grid structure: DEM file definition is required!')
            element = None
            element = grid.find('resfactor')
            if element is not None:
                self.Afactor = float(element.text)
            else:
                self.Afactor = 1.
        else:
            raise ValueError('Error in the XmL file: grid structure definition is required!')

        # Extract time structure information
        time = None
        time = root.find('time')
        if time is not None:
            element = None
            element = time.find('start')
            if element is not None:
                self.tStart = float(element.text)
            else:
                raise ValueError('Error in the definition of the simulation time: start time declaration is required')
            element = None
            element = time.find('end')
            if element is not None:
                self.tEnd = float(element.text)
            else:
                raise ValueErself.ror('Error in the definition of the simulation time: end time declaration is required')
            if self.tStart > self.tEnd:
                raise ValueError('Error in the definition of the simulation time: start time is greater than end time!')
            element = None
            element = time.find('dt')
            if element is not None:
                self.dt = float(element.text)
            else:
                raise ValueError('Error in the definition of the simulation time: simulation time step is required')
            element = None
            element = time.find('twave')
            if element is not None:
                self.tWave = float(element.text)
            else:
                raise ValueError('Error in the definition of the simulation time: wave interval is required')
            if (self.tEnd - self.tStart) % self.tWave != 0:
                raise ValueError('Error in the definition of the simulation time: wave interval needs to be a multiple of simulation time.')
            element = None
            element = time.find('display')
            if element is not None:
                self.tDisplay = float(element.text)
            else:
                raise ValueError('Error in the definition of the simulation time: display time declaration is required')
            if (self.tEnd - self.tStart) % self.tDisplay != 0:
                raise ValueError('Error in the definition of the simulation time: display time needs to be a multiple of simulation time.')
        else:
            raise ValueError('Error in the XmL file: time structure definition is required!')

        # Extract sea-level structure information
        sea = None
        sea = root.find('sea')
        if sea is not None:
            element = None
            element = sea.find('val')
            if element is not None:
                self.seaval = float(element.text)
            else:
                self.seaval = 0.
            element = None
            element = sea.find('curve')
            if element is not None:
                self.seafile = element.text
                if not os.path.isfile(self.seafile):
                    raise ValueError('Sea level file is missing or the given path is incorrect.')
            else:
                self.seafile = None
        else:
            self.seapos = 0.
            self.seafile = None

        # Extract temperature structure information
        temp = None
        temp = root.find('temperature')
        if temp is not None:
            element = None
            element = temp.find('val')
            if element is not None:
                self.tempval = float(element.text)
            else:
                self.tempval = 25.
            element = None
            element = temp.find('curve')
            if element is not None:
                self.tempfile = element.text
                if not os.path.isfile(self.tempfile):
                    raise ValueError('Temperature file is missing or the given path is incorrect.')
            else:
                self.tempfile = None
        else:
            self.tempval = 25.
            self.tempfile = None

        # Extract salinity structure information
        sal = None
        sal = root.find('salinity')
        if sal is not None:
            element = None
            element = sal.find('val')
            if element is not None:
                self.salval = float(element.text)
            else:
                self.salval = 35.5
            element = None
            element = sal.find('curve')
            if element is not None:
                self.salfile = element.text
                if not os.path.isfile(self.salfile):
                    raise ValueError('Salinity file is missing or the given path is incorrect.')
            else:
                self.salfile = None
        else:
            self.salval = 35.5
            self.salfile = None

        # Extract wave field structure information
        wavefield = None
        wavefield = root.find('wavefield')
        if wavefield is not None:
            element = None
            element = wavefield.find('base')
            if element is not None:
                self.waveBase = float(element.text)
            else:
                self.waveBase = 10000
            element = None
            element = wavefield.find('events')
            if element is not None:
                self.waveNb = int(element.text)
            else:
                raise ValueError('The number of wave temporal events needs to be defined.')
            tmpNb = self.waveNb
            self.waveWind = numpy.empty(tmpNb,dtype=bool)
            self.waveParam = numpy.empty(tmpNb,dtype=bool)
            self.waveTime = numpy.empty((tmpNb,2))
            self.wavePerc = numpy.empty(tmpNb)
            self.waveWu = numpy.empty(tmpNb)
            self.waveWd = numpy.empty(tmpNb)
            self.waveDir = numpy.empty(tmpNb)
            self.waveHs = numpy.empty(tmpNb)
            self.wavePer = numpy.empty(tmpNb)
            self.waveDs = numpy.empty(tmpNb)
            id = 0
            for event in wavefield.iter('wave'):
                if id >= tmpNb:
                    raise ValueError('The number of wave events does not match the number of defined events.')
                element = None
                element = event.find('start')
                if element is not None:
                    self.waveTime[id,0] = float(element.text)
                else:
                    raise ValueError('Wave event %d is missing start time argument.'%id)
                element = None
                element = event.find('end')
                if element is not None:
                    self.waveTime[id,1] = float(element.text)
                else:
                    raise ValueError('Wave event %d is missing end time argument.'%id)
                if self.waveTime[id,0] >= self.waveTime[id,1]:
                    raise ValueError('Wave event %d start and end time values are not properly defined.'%id)
                element = None
                element = event.find('perc')
                if element is not None:
                    self.wavePerc[id] = float(element.text)
                    if self.wavePerc[id] < 0:
                        raise ValueError('Wave event %d percentage cannot be negative.'%id)
                else:
                    raise ValueError('Wave event %d is missing percentage argument.'%id)
                element = None
                element = event.find('windv')
                if element is not None:
                    self.waveWind[id] = True
                    self.waveParam[id] = False
                    self.waveWu[id] = float(element.text)
                    if self.waveWu[id] < 0:
                        raise ValueError('Wave event %d wind velocity cannot be negative.'%id)
                else:
                    self.waveWind[id] = False
                    self.waveWu[id] = 0.
                element = None
                element = event.find('hs')
                if element is not None:
                    self.waveWind[id] = False
                    self.waveParam[id] = True
                    self.waveHs[id] = float(element.text)
                    if self.waveHs[id] < 0:
                        raise ValueError('Wave event %d wave Hs cannot be negative.'%id)
                else:
                    self.waveParam[id] = False
                    self.waveHs[id] = 0.
                if self.waveWind[id] is False and self.waveParam[id] is False:
                    raise ValueError('Wave event %d needs to be declared with either wind or wave parameters turned on.'%id)
                if self.waveWind[id] and self.waveParam[id]:
                    raise ValueError('Wave event %d needs to be declared with one of wind or wave parameters turned off.'%id)
                element = None
                element = event.find('per')
                if element is not None:
                    self.wavePer[id] = float(element.text)
                    if self.wavePer[id] < 0:
                        raise ValueError('Wave event %d wave Per cannot be negative.'%id)
                else:
                    self.wavePer[id] = 0.
                element = None
                element = event.find('ds')
                if element is not None:
                    self.waveDs[id] = float(element.text)
                    if self.waveDs[id] < 0:
                        raise ValueError('Wave event %d wave Ds cannot be negative.'%id)
                else:
                    self.waveDs[id] = 0.
                element = None
                element = event.find('dir')
                if element is not None:
                    if self.waveWind[id]:
                        self.waveWd[id] = float(element.text)
                        if self.waveWd[id] < 0:
                            raise ValueError('Wave event %d wind direction needs to be set between 0 and 360.'%id)
                    else:
                        self.waveWd[id] = -1
                    if self.waveParam[id]:
                        self.waveDir[id] = float(element.text)
                        if self.waveDir[id] < 0:
                            raise ValueError('Wave event %d wave direction needs to be set between 0 and 360.'%id)
                    else:
                        self.waveDir[id] = -1
                else:
                    if self.waveWind[id]:
                        raise ValueError('Wave event %d is missing wind direction argument.'%id)
                    if self.waveParam[id]:
                        raise ValueError('Wave event %d is missing wave direction argument.'%id)
                id += 1
        else:
            self.waveNb = 0

        # Get output directory
        out = None
        out = root.find('outfolder')
        if out is not None:
            self.outDir = out.text
        else:
            self.outDir = os.getcwd()+'/out'

        if self.makeUniqueOutputDir:
            if os.path.exists(self.outDir):
                self.outDir += '_'+str(len(glob.glob(self.outDir+str('*')))-1)

            os.makedirs(self.outDir)
            os.makedirs(self.outDir+'/h5')
            os.makedirs(self.outDir+'/xmf')
            shutil.copy(self.inputfile,self.outDir)

        return
