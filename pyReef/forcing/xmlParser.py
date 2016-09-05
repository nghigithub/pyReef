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
        self.Wfactor = 1

        self.tStart = None
        self.tEnd = None
        self.dt = None
        self.tWave = None
        self.tSed = None
        self.tDisplay = None
        self.laytime = None

        self.faciesNb = None
        self.waterD = None
        self.density = None
        self.porosity = None
        self.efficiency = None
        self.diffusion = None
        self.faciesName = None
        self.faciesDiam = None

        self.stratlays = None
        self.stratMap = None
        self.stratVal = None
        self.thickMap = None
        self.thickVal = None

        self.seaOn = False
        self.seaval = 0.
        self.seafile = None

        self.tempOn = False
        self.tempval = 25.
        self.tempfile = None

        self.tideOn = False
        self.tideval = 0.
        self.tidefile = None

        self.phOn = False
        self.phval = 8.1
        self.phfile = None

        self.salOn = False
        self.salval = 35.5
        self.salfile = None

        self.tectNb = None
        self.tectTime = None
        self.tectFile = None

        self.waveOn = False
        self.waveBase = 10000.
        self.waveNb = 0
        self.cKom = 1.
        self.sigma = 1.
        self.waveTime = None
        self.wavePerc = None
        self.waveWh = None
        self.waveWp = None
        self.waveWs = None
        self.waveWd = None
        self.waveBrk = None
        self.bedlay = None
        self.storm = None
        self.wavelist = None
        self.climlist = None

        self.makeUniqueOutputDir = makeUniqueOutputDir
        self.outDir = None

        self.h5file = 'h5/surf.time'
        self.xmffile = 'xmf/surf.time'
        self.xdmffile = 'surf.series.xdmf'

        self.swanFile = None
        self.swanInfo = None
        self.swanBot = None
        self.swanOut = None

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
            element = grid.find('resdem')
            if element is not None:
                self.Afactor = float(element.text)
            else:
                self.Afactor = 1.
            element = None
            element = grid.find('reswave')
            if element is not None:
                self.Wfactor = float(element.text)
            else:
                self.Wfactor = 1.
            if self.Wfactor < 1.:
                self.Wfactor = 1.
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
            element = None
            element = time.find('tsed')
            if element is not None:
                self.tSed = float(element.text)*60.
            else:
                self.tSed = 3600.
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
            if Decimal(self.tDisplay) % Decimal(self.dt) != 0.:
                raise ValueError('Error in the XmL file: time step interval needs to be an exact multiple of the display interval!')
            if Decimal(self.tEnd-self.tStart) % Decimal(self.tDisplay) != 0.:
                raise ValueError('Error in the XmL file: display interval needs to be an exact multiple of the simulation time interval!')
        else:
            raise ValueError('Error in the XmL file: time structure definition is required!')

        # Extract lithofacies structure information
        litho = None
        litho = root.find('lithofacies')
        if litho is not None:
            element = None
            element = litho.find('faciesNb')
            if element is not None:
                self.faciesNb = int(element.text)
            else:
                raise ValueError('Error in the definition of the lithofacies: number of facies is required')
            element = None
            element = litho.find('waterD')
            if element is not None:
                self.waterD = float(element.text)
            else:
                self.waterD = 1010.0
            self.faciesName = numpy.empty(self.faciesNb, dtype="S10")
            self.density = numpy.zeros(self.faciesNb, dtype=float)
            self.porosity = numpy.zeros(self.faciesNb, dtype=float)
            self.efficiency = numpy.zeros(self.faciesNb, dtype=float)
            self.diffusion = numpy.zeros(self.faciesNb, dtype=float)
            self.faciesDiam = numpy.zeros(self.faciesNb, dtype=float)
            id = 0
            for facies in litho.iter('facies'):
                if id >= self.faciesNb:
                    raise ValueError('The number of facies does not match the number of defined ones.')
                element = None
                element = facies.find('name')
                if element is not None:
                    self.faciesName[id] = element.text
                else:
                    raise ValueError('Facies name %d is missing in the lithofacies structure.'%id)
                element = None
                element = facies.find('sedD')
                if element is not None:
                    self.density[id] = float(element.text)
                else:
                    self.density[id] = 2650.0
                element = None
                element = facies.find('porosity')
                if element is not None:
                    self.porosity[id] = float(element.text)
                else:
                    self.porosity[id] = 0.4
                element = None
                element = facies.find('efficiency')
                if element is not None:
                    self.efficiency[id] = float(element.text)*0.7
                else:
                    self.efficiency[id] = 0.
                element = None
                element = facies.find('diffusion')
                if element is not None:
                    self.diffusion[id] = float(element.text)
                else:
                    self.diffusion[id] = 0.1
                element = None
                element = facies.find('diam')
                if element is not None:
                    self.faciesDiam[id] = float(element.text)*0.001
                else:
                    raise ValueError('Diameter %d is missing in the lithofacies structure.'%id)
                id += 1
        else:
            raise ValueError('Error in the XmL file: lithofacies structure definition is required!')

        # Extract basement structure information
        strat = None
        strat = root.find('basement')
        if strat is not None:
            element = None
            element = strat.find('stratlayers')
            if element is not None:
                self.stratlays = int(element.text)
            else:
                raise ValueError('Error in the definition of the basement: number of layers is required')
            self.stratVal = numpy.empty((self.stratlays,self.faciesNb), dtype=float)
            self.stratMap = numpy.empty(self.stratlays, dtype=object)
            self.thickVal = numpy.empty(self.stratlays, dtype=float)
            self.thickMap = numpy.empty(self.stratlays, dtype=object)
            id = 0
            for lay in strat.iter('layer'):
                if id >= self.stratlays:
                    raise ValueError('The number of layers does not match the number of defined ones.')
                element = None
                element = lay.find('facPerc')
                if element is not None:
                    tmpPerc = numpy.fromstring(element.text, dtype=float, sep=',')
                    if len(tmpPerc) != self.faciesNb:
                        raise ValueError('The number of facies percentages defined for the layer %d does not match the number of defined lithofacies.'%id)
                    if numpy.sum(tmpPerc) != 1:
                        raise ValueError('The summation of each percentages for layer %d does not equal 1.'%id)
                    self.stratVal[id,:] = tmpPerc
                else:
                    self.stratVal[id,:] = numpy.zeros(self.faciesNb,dtype=float)
                if sum(self.stratVal[id,:]) == 0.:
                    element = None
                    element = lay.find('facmap')
                    if element is not None:
                        self.stratMap[id] = element.text
                    else:
                        raise ValueError('Error either facPerc or facmap parameters needs to be defined in layer %d structure'%id)
                else:
                    self.stratMap[id] = None

                element = None
                element = lay.find('thcst')
                if element is not None:
                    self.thickVal[id] = float(element.text)
                else:
                    self.thickVal[id] = 0.
                if self.thickVal[id] == 0.:
                    element = None
                    element = lay.find('thmap')
                    if element is not None:
                        self.thickMap[id] = element.text
                    else:
                        raise ValueError('Error either thcst or thmap parameters needs to be defined in layer %d structure'%id)
                else:
                    self.thickMap[id] = None
                id += 1
        else:
            raise ValueError('Error in the XmL file: basement structure definition is required!')

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

        # Extract ocean acidity information
        acidity = None
        acidity = root.find('acidification')
        if acidity is not None:
            self.phOn = True
            element = None
            element = acidity.find('val')
            if element is not None:
                self.phval = float(element.text)
            else:
                self.seaval = 0.
            element = None
            element = acidity.find('curve')
            if element is not None:
                self.phfile = element.text
                if not os.path.isfile(self.phfile):
                    raise ValueError('Ocean acidity file is missing or the given path is incorrect.')
            else:
                self.seafile = None
        else:
            self.phval = 0.
            self.phfile = None

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

        # Extract tidal range structure information
        tide = None
        tide = root.find('tides')
        if tide is not None:
            self.tideOn = True
            element = None
            element = tide.find('val')
            if element is not None:
                self.tideval = float(element.text)
            else:
                self.tideval = 0.0
            element = None
            element = tide.find('curve')
            if element is not None:
                self.tidefile = element.text
                if not os.path.isfile(self.tidefile):
                    raise ValueError('Tidal range file is missing or the given path is incorrect.')
            else:
                self.tidefile = None
        else:
            self.tideval = 0.
            self.tidefile = None

        # Extract Tectonic structure information
        tecto = None
        tecto = root.find('tectonic')
        if tecto is not None:
            element = None
            element = tecto.find('events')
            if element is not None:
                tmpNb = int(element.text)
            else:
                raise ValueError('The number of tectonic events needs to be defined.')
            tmpFile = numpy.empty(tmpNb,dtype=object)
            tmpTime = numpy.empty((tmpNb,2))
            id = 0
            for disp in tecto.iter('disp'):
                element = None
                element = disp.find('dstart')
                if element is not None:
                    tmpTime[id,0] = float(element.text)
                else:
                    raise ValueError('Displacement event %d is missing start time argument.'%id)
                element = None
                element = disp.find('dend')
                if element is not None:
                    tmpTime[id,1] = float(element.text)
                else:
                    raise ValueError('Displacement event %d is missing end time argument.'%id)
                if tmpTime[id,0] >= tmpTime[id,1]:
                    raise ValueError('Displacement event %d start and end time values are not properly defined.'%id)
                if id > 0:
                    if tmpTime[id,0] < tmpTime[id-1,1]:
                        raise ValueError('Displacement event %d start time needs to be >= than displacement event %d end time.'%(id,id-1))
                element = None
                element = disp.find('dfile')
                if element is not None:
                    tmpFile[id] = element.text
                    if not os.path.isfile(tmpFile[id]):
                        raise ValueError('Displacement file %s is missing or the given path is incorrect.'%(tmpFile[id]))
                else:
                    raise ValueError('Displacement event %d is missing file argument.'%id)
                id += 1
            if id != tmpNb:
                raise ValueError('Number of events %d does not match with the number of declared displacement parameters %d.' %(tmpNb,id))

            # Create continuous displacement series
            self.tectNb = tmpNb
            if tmpTime[0,0] > self.tStart:
                self.tectNb += 1
            for id in range(1,tmpNb):
                if tmpTime[id,0] > tmpTime[id-1,1]:
                    self.tectNb += 1
            if tmpTime[tmpNb-1,1] < self.tEnd:
                self.tectNb += 1
            self.tectFile = numpy.empty(self.tectNb,dtype=object)
            self.tectTime = numpy.empty((self.tectNb,2))
            id = 0
            if tmpTime[id,0] > self.tStart:
                self.tectFile[id] = None
                self.tectTime[id,0] = self.tStart
                self.tectTime[id,1] = tmpTime[0,0]
                id += 1
            self.tectFile[id] = tmpFile[0]
            self.tectTime[id,:] = tmpTime[0,:]
            id += 1
            for p in range(1,tmpNb):
                if tmpTime[p,0] > tmpTime[p-1,1]:
                    self.tectFile[id] = None
                    self.tectTime[id,0] = tmpTime[p-1,1]
                    self.tectTime[id,1] = tmpTime[p,0]
                    id += 1
                self.tectFile[id] = tmpFile[p]
                self.tectTime[id,:] = tmpTime[p,:]
                id += 1
            if tmpTime[tmpNb-1,1] < self.tEnd:
                self.tectFile[id] = None
                self.tectTime[id,0] = tmpTime[tmpNb-1,1]
                self.tectTime[id,1] = self.tEnd
        else:
            self.tectNb = 1
            self.tectTime = numpy.empty((self.tectNb,2))
            self.tectTime[0,0] = self.tEnd + 1.e5
            self.tectTime[0,1] = self.tEnd + 2.e5
            self.tectFile = numpy.empty((self.tectNb),dtype=object)
            self.tectFile = None

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
            element = None
            element = wavefield.find('cKom')
            if element is not None:
                self.cKom = float(element.text)
            element = None
            element = wavefield.find('sigma')
            if element is not None:
                self.sigma = float(element.text)
        else:
            self.waveNb = 0

        # Extract wave field structure information
        if self.waveNb > 0:
            tmpNb = self.waveNb
            self.waveWh = []
            self.waveWp = []
            self.waveWd = []
            self.waveWs = []
            self.wavePerc = []
            self.waveBrk = []
            self.bedlay = []
            self.storm = []
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
                    listWh = []
                    listWp = []
                    listWd = []
                    listWs = []
                    listStorm = []
                    listBedlay = []
                    listBreak = []
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
                        element = clim.find('hs')
                        if element is not None:
                            listWh.append(float(element.text))
                            if listWh[id] < 0:
                                raise ValueError('Wave event %d significant wave height cannot be negative.'%w)
                        else:
                            raise ValueError('Wave event %d is missing significant wave height argument.'%w)
                        element = None
                        element = clim.find('per')
                        if element is not None:
                            listWp.append(float(element.text))
                            if listWh[id] < 0:
                                raise ValueError('Wave event %d wave period cannot be negative.'%w)
                        else:
                            raise ValueError('Wave event %d is missing wave period argument.'%w)
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
                        element = None
                        element = clim.find('spread')
                        if element is not None:
                            listWs.append(float(element.text))
                            if listWs[id] < 0:
                                raise ValueError('Wave event %d wave spreading cannot be negative.'%w)
                        else:
                            listWs.append(0.)
                        element = None
                        element = clim.find('break')
                        if element is not None:
                            listBreak.append(float(element.text))
                            if listBreak[id] > 0:
                                raise ValueError('Wave event %d wave breaking depth needs to be negative.'%w)
                        else:
                            raise ValueError('Wave event %d is missing wave breaking depth argument.'%w)
                        element = None
                        element = clim.find('bedlayer')
                        if element is not None:
                            listBedlay.append(float(element.text))
                            if listBedlay[id] <= 0:
                                listBedlay[id] = 0.001
                        else:
                            raise ValueError('Wave event %d is missing bed layer thickness argument.'%w)
                        element = None
                        element = clim.find('storm')
                        if element is not None:
                            listStorm.append(int(element.text))
                            if listStorm[id] > 1:
                                listStorm[id] = 1
                            if listStorm[id] < 0:
                                listStorm[id] = 0
                        else:
                            raise ValueError('Wave event %d is missing storm argument.'%w)
                        id += 1
                    if len(listPerc) < self.climNb[w] :
                        raise ValueError('Amount of climates declared in wave event %d does not match the given climate number.'%w)

                    w += 1

                    self.wavePerc.append(listPerc)
                    self.waveWh.append(listWh)
                    self.waveWp.append(listWp)
                    self.waveWs.append(listWs)
                    self.waveWd.append(listWd)
                    self.waveBrk.append(listBreak)
                    self.bedlay.append(listBedlay)
                    self.storm.append(listStorm)
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
            os.makedirs(self.outDir+'/vtk')
            if self.waveOn:
                os.makedirs(self.outDir+'/swan')

            shutil.copy(self.inputfile,self.outDir)

        # Create swan model repository and files
        if self.waveOn:
            self.swanFile = numpy.array(self.outDir+'/swan/swan.swn')
            self.swanInfo = numpy.array(self.outDir+'/swan/swanInfo.swn')
            self.swanBot = numpy.array(self.outDir+'/swan/swan.bot')
            self.swanOut = numpy.array(self.outDir+'/swan/swan.csv')

        return
