import time
import numpy as np
import mpi4py.MPI as mpi

from pyReef import (hydrodynamic, forceSim, outputGrid, raster2surf, map2strat, xmlParser)

from pyReef.libUtils  import simswan as swan

# profiling support
import cProfile
import os
import pstats
import StringIO


class Model(object):
    """State object for the pyReef model."""

    def __init__(self):
        """
        Constructor.
        """

        # simulation state
        self.tNow = 0.
        self.tDisp = 0.
        self.waveID = 0
        self.outputStep = 0
        self.applyDisp = False
        self.simStarted = False

        self.dispRate = None

        self._rank = mpi.COMM_WORLD.rank
        self._size = mpi.COMM_WORLD.size
        self._comm = mpi.COMM_WORLD
        self.fcomm = mpi.COMM_WORLD.py2f()

    def load_xml(self, filename, verbose=False):
        """
        Load an XML configuration file.
        """

        # Only the first node should create a unique output dir
        self.input = xmlParser.xmlParser(filename, makeUniqueOutputDir=(self._rank == 0))
        self.tNow = self.input.tStart

        # Sync the chosen output dir to all nodes
        self.input.outDir = self._comm.bcast(self.input.outDir, root=0)

        # Create the spatial discretisation
        self.build_spatial_model(self.input.demfile, verbose)

    def build_spatial_model(self, filename, verbose):
        """
        Create pyReef surface grid and stratigraphic mesh.
        """

        # From DEM grid create pyReef surface grid and partition.
        self.pyGrid = raster2surf.raster2surf(inputfile=self.input.demfile, resRecFactor=self.input.Afactor)

        # Build pyReef stratigraphic mesh.
        self.pyStrat = map2strat.map2strat(self.input, self.pyGrid)

        # Initialise surface
        self.outSurf = outputGrid.outputGrid(self.pyGrid, self.input.outDir, self.input.h5file,
                                        self.input.xmffile, self.input.xdmffile)

        # Define forcing conditions: sea, temperature, salinity, wave
        self.force = forceSim.forceSim(self.input, self.pyGrid)

        # Define hydrodynamic conditions
        self.hydro = hydrodynamic.hydrodynamic(self.input.cKom, self.input.sigma, self.pyGrid.res,
                                               self.pyGrid.regX, self.pyGrid.regY)

        # Define display and wave climate next time step
        self.force.next_display = self.input.tStart
        self.force.next_layer = self.input.tStart + self.force.time_layer
        if self.input.waveOn:
            self.force.next_wave = self.input.tStart
        else:
            self.force.next_wave = self.input.tEnd + 1.e5

        # Get sea-level at start time
        if self.input.seaOn:
            self.force.getSea(self.tNow)

        # Initialise SWAN
        if self.input.waveOn:
            self.hydro.swan_init(self.input, self.pyGrid.regZ, self.waveID, self.force.sealevel)

    def run_to_time(self, tEnd, profile=False, verbose=False):
        """
        Run the simulation to a specified point in time (tEnd).

        If profile is True, dump cProfile output to /tmp.
        """

        if profile:
            pid = os.getpid()
            pr = cProfile.Profile()
            pr.enable()

        if tEnd > self.input.tEnd:
            tEnd = self.input.tEnd
            print 'Requested time is set to the simulation end time as defined in the XmL input file'

        # Define non-flow related processes times
        if not self.simStarted:
            self.tDisp = self.input.tStart
            self.force.next_display = self.input.tStart
            self.force.next_disp = self.force.T_disp[0, 0]
            self.force.next_carb = self.input.tStart
            self.exitTime = self.input.tEnd
            self.simStarted = True
            self.force.next_display = self.input.tStart

        # Perform main simulation loop
        while self.tNow < tEnd:

            # Apply displacements
            if self.applyDisp:
                if self.tDisp < self.tNow:
                    dispTH = self.dispRate * (self.tNow - self.tDisp)
                    self.pyGrid.regZ += dispTH
                    self.tDisp = self.tNow

            # Update sea-level fluctuations
            if self.input.seaOn:
                self.force.getSea(self.tNow)

            # Update sea-surface temperatures
            if self.input.tempOn:
                self.force.getTemperature(self.tNow)

            # Update sea-surface salinities
            if self.input.salOn:
                self.force.getSalinity(self.tNow)

            # Get ocean ph at start time
            if self.input.phOn:
                self.force.getph(self.tNow)

            # Update wave parameters
            if self.input.waveOn:
                if self.force.next_wave <= self.tNow:

                    # Compute wave field and associated bottom current conditions
                    self.waveID = self.hydro.swan_run(self.input, self.force, self.pyGrid.regZ, self.waveID)

                    # Update next wave time step
                    self.force.next_wave += self.force.time_wave

            # Create output
            if self.force.next_display <= self.tNow and self.force.next_display < self.input.tEnd:
                self.outSurf.write_hdf5_grid(self.pyGrid.regZ, self.force, self.tNow, self.outputStep)
                self.pyStrat.write_mesh(self.pyGrid.regZ, self.tNow, self.outputStep)
                self.force.next_display += self.force.time_display
                self.outputStep += 1

            # Add stratigraphic layer
            if self.force.next_layer <= self.tNow and self.force.next_layer < self.input.tEnd:
                self.force.next_layer += self.force.time_layer
                self.pyStrat.layID += 1

            # Update vertical displacements
            if self.force.next_disp <= self.tNow and self.force.next_disp < self.input.tEnd:
                self.dispRate = self.force.load_Tecto_map(self.tNow)
                self.applyDisp = True

            # Update carbonate parameters
            if self.force.next_carb <= self.tNow and self.force.next_carb < self.input.tEnd:
                self.force.next_carb += self.input.dt
                # if self.force.next_carb > self.force.next_display:
                #     self.force.next_carb = self.force.next_display

            # Get the maximum time before updating one of the above processes / components
            self.tNow = min([self.force.next_display, tEnd, self.force.next_disp, self.force.next_wave])

        # Finalise SWAN model run
        if self.input.waveOn and self.tNow == self.input.tEnd:
            swan.model.final(self.fcomm)

        if profile:
            pr.disable()
            s = StringIO.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.dump_stats('/tmp/profile-%d' % pid)

    def ncpus(self):
        """
        Return the number of CPUs used to generate the results.
        """

        return 1
