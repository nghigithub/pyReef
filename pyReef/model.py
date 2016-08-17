import time
import numpy as np
import mpi4py.MPI as mpi

from pyReef import (buildMesh, xmlParser)

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

        self.pyGrid = buildMesh.construct_surface_mesh(self.input)



    def run_to_time(self, tEnd, profile=False, verbose=False):
        """
        Run the simulation to a specified point in time (tEnd).

        If profile is True, dump cProfile output to /tmp.
        """

        if profile:
            pid = os.getpid()
            pr = cProfile.Profile()
            pr.enable()


    def ncpus(self):
        """
        Return the number of CPUs used to generate the results.
        """

        return 1
