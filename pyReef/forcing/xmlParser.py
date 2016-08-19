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
from decimal import Decimal

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
        self.laytime = None

        self.seaOn = False
        self.seaval = 0.
        self.seafile = None

        self.tempOn = False
        self.tempval = 25.
        self.tempfile = None

        self.salOn = False
        self.salval = 35.5
        self.salfile = None

        self.waveOn = False
        self.waveBase = 10000.
        self.waveNb = 0
        self.waveTime = None
        self.wavePerc = None
        self.waveWu = None
        self.waveWd = None
        self.wavelist = None
        self.climlist = None

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
            if Decimal(self.tEnd - self.tStart) % Decimal(self.tWave) != 0.:
                raise ValueError('Error in the definition of the simulation time: wave interval needs to be a multiple of simulation time.')
            element = None
            element = time.find('display')
            if element is not None:
                self.tDisplay = float(element.text)
            else:
                raise ValueError('Error in the definition of the simulation time: display time declaration is required')
            if Decimal(self.tEnd - self.tStart) % Decimal(self.tDisplay) != 0.:
                raise ValueError('Error in the definition of the simulation time: display time needs to be a multiple of simulation time.')
            element = None
            element = time.find('laytime')
            if element is not None:
                self.laytime = float(element.text)
            else:
                self.laytime = self.tDisplay
            if self.laytime >  self.tDisplay:
                 self.laytime = self.tDisplay
            if self.tWave >  self.tDisplay:
                  self.tWave = self.tDisplay
            if Decimal(self.tDisplay) % Decimal(self.laytime) != 0.:
                raise ValueError('Error in the XmL file: stratal layer interval needs to be an exact multiple of the display interval!')
            if Decimal(self.tDisplay) % Decimal(self.tWave) != 0.:
                raise ValueError('Error in the XmL file: wave time interval needs to be an exact multiple of the display interval!')
            if Decimal(self.tEnd-self.tStart) % Decimal(self.tDisplay) != 0.:
                raise ValueError('Error in the XmL file: display interval needs to be an exact multiple of the simulation time interval!')
        else:
            raise ValueError('Error in the XmL file: time structure definition is required!')

        # Extract sea-level structure information
        sea = None
        sea = root.find('sea')
        if sea is not None:
            self.seaOn = True
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
            self.tempOn = True
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
            self.salOn = True
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

        # Extract global wave field parameters
        wavefield = None
        wavefield = root.find('waveglobal')
        if wavefield is not None:
            self.waveOn = True
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
        else:
            self.waveNb = 0

        # Extract wave field structure information
        if self.waveNb > 0:
            tmpNb = self.waveNb
            self.waveWd = []
            self.waveWu = []
            self.wavePerc = []
            self.waveTime = numpy.empty((tmpNb,2))
            self.climNb = numpy.empty(tmpNb, dtype=int)
            w = 0
            for wavedata in root.iter('wave'):
                if w >= tmpNb:
                    raise ValueError('Wave event number above number defined in global wave structure.')
                if wavedata is not None:
                    element = None
                    element = wavedata.find('start')
                    if element is not None:
                        self.waveTime[w,0] = float(element.text)
                        if w > 0 and self.waveTime[w,0] != self.waveTime[w-1,1]:
                            raise ValueError('The start time of the wave field %d needs to match the end time of previous wave data.'%w)
                        if w == 0 and self.waveTime[w,0] != self.tStart:
                            raise ValueError('The start time of the first wave field needs to match the simulation start time.')
                    else:
                        raise ValueError('Wave event %d is missing start time argument.'%w)
                    element = None
                    element = wavedata.find('end')
                    if element is not None:
                        self.waveTime[w,1] = float(element.text)
                    else:
                        raise ValueError('Wave event %d is missing end time argument.'%w)
                    if self.waveTime[w,0] >= self.waveTime[w,1]:
                        raise ValueError('Wave event %d start and end time values are not properly defined.'%w)
                    element = None
                    element = wavedata.find('climNb')
                    if element is not None:
                        self.climNb[w] = int(element.text)
                    else:
                        raise ValueError('Wave event %d is missing climatic wave number argument.'%w)

                    if Decimal(self.waveTime[w,1]-self.waveTime[w,0]) % Decimal(self.tWave) != 0.:
                        raise ValueError('Wave event %d duration need to be a multiple of the wave interval.'%w)

                    listPerc = []
                    listWu = []
                    listWd = []
                    id = 0
                    sumPerc = 0.
                    for clim in wavedata.iter('climate'):
                        if id >= self.climNb[w]:
                            raise ValueError('The number of climatic events does not match the number of defined climates.')
                        element = None
                        element = clim.find('perc')
                        if element is not None:
                            sumPerc += float(element.text)
                            if sumPerc > 1:
                                raise ValueError('Sum of wave event %d percentage is higher than 1.'%w)
                            listPerc.append(float(element.text))
                            if listPerc[id] < 0:
                                raise ValueError('Wave event %d percentage cannot be negative.'%w)
                        else:
                            raise ValueError('Wave event %d is missing percentage argument.'%w)
                        element = None
                        element = clim.find('windv')
                        if element is not None:
                            listWu.append(float(element.text))
                            if listWu[id] < 0:
                                raise ValueError('Wave event %d wind velocity cannot be negative.'%w)
                        else:
                            raise ValueError('Wave event %d is missing wind velocity argument.'%w)
                        element = None
                        element = clim.find('dir')
                        if element is not None:
                            listWd.append(float(element.text))
                            if listWd[id] < 0:
                                raise ValueError('Wave event %d wind direction needs to be set between 0 and 360.'%w)
                            if listWd[id] > 360:
                                raise ValueError('Wave event %d wind direction needs to be set between 0 and 360.'%w)
                        else:
                            raise ValueError('Wave event %d is missing wind direction argument.'%w)
                        id += 1
                    w += 1
                    self.wavePerc.append(listPerc)
                    self.waveWu.append(listWu)
                    self.waveWd.append(listWd)
                else:
                    raise ValueError('Wave event %d is missing.'%w)

        # Construct a list of climatic events for swan model
        self.wavelist = []
        self.climlist = []
        twsteps = numpy.arange(self.tStart,self.tEnd,self.tWave)
        for t in range(len(twsteps)):
            c = -1
            # Find the wave field active during the time interval
            for k in range(self.waveNb):
                if self.waveTime[k,0] <= twsteps[t] and self.waveTime[k,1] >= twsteps[t]:
                    c = k
            # Extract the wave climate for the considered time interval
            for p in range(self.climNb[c]):
                self.wavelist.append(c)
                self.climlist.append(p)

        # Add a fake final wave field and climate
        self.wavelist.append(self.wavelist[-1])
        self.climlist.append(self.climlist[-1])

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
