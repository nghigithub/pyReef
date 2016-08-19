import time
import numpy as np
import mpi4py.MPI as mpi

from pyReef import (forceSim, outputGrid, buildMesh, xmlParser)

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
        self.outputStep = 0
        self.simStarted = False

        self._rank = mpi.COMM_WORLD.rank
        self._size = mpi.COMM_WORLD.size
        self._comm = mpi.COMM_WORLD

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

        # Build pyReef surface and stratigraphic mesh
        self.pyGrid = buildMesh.construct_surface_mesh(self.input)

        # Define forcing conditions: sea, temperature, salinity, wave
        self.force = forceSim.forceSim(self.input.seafile, self.input.seaval, self.input.tempfile, self.input.tempval,
                                  self.input.salfile, self.input.salval, self.input.waveNb, self.input.waveTime,
                                  self.input.wavePerc, self.input.waveWu, self.input.waveWd, self.input.tWave,
                                  self.input.tDisplay)

        # Initialise surface
        outSurf = outputGrid.outputGrid(self.pyGrid, self.input.outDir, self.input.h5file, self.input.xmffile, self.input.xdmffile)

        # Define display and wave climate next time step
        self.force.next_display = self.input.tStart + self.force.time_display
        if self.input.waveOn:
            self.force.next_wave = self.input.tStart + self.force.time_wave
            if self.force.next_wave > self.force.next_display:
                self.force.next_wave = self.force.next_display
        else:
            self.force.next_wave = self.input.tEnd + 10.

        # Get sea-level at start time
        if self.input.seaOn:
            self.force.getSea(self.tNow)

        # Get sea-surface temperature at start time
        if self.input.tempOn:
            self.force.getTemperature(self.tNow)

        # Get sea-surface salinity at start time
        if self.input.salOn:
            self.force.getSalinity(self.tNow)

        # Output surface
        outSurf.write_hdf5_grid(self.pyGrid.regZ, self.tNow, self.outputStep)

    def run_to_time(self, tEnd, profile=False, verbose=False):
        """
        Run the simulation to a specified point in time (tEnd).

        If profile is True, dump cProfile output to /tmp.
        """

        if profile:
            pid = os.getpid()
            pr = cProfile.Profile()
            pr.enable()

        # Define non-flow related processes times
        if not self.simStarted:
            self.force.next_display = self.input.tStart
            self.exitTime = self.input.tEnd
            self.simStarted = True

        # Do work here

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
