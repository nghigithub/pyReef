##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the pyReef carbonate platform modelling application.     ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
This file defines the functions used to build pyReef meshes and surface grids.
"""

import time
import numpy as np
import mpi4py.MPI as mpi

from pyReef import (raster2surf)


def construct_surface_mesh(input, verbose=False):
    """
    The following function is taking parsed values from the XML to:
        - build model grids & meshes,
        - define the partitioning when parallelisation is enable.
    """

    rank = mpi.COMM_WORLD.rank
    size = mpi.COMM_WORLD.size
    comm = mpi.COMM_WORLD

    # From DEM grid and create pyReef surface grid and partion.
    surface = raster2surf.raster2surf(inputfile=input.demfile, resRecFactor=input.Afactor)

    # Build pyReef stratigraphic mesh.
    #mesh = generate_mesh(surface)

    return surface
