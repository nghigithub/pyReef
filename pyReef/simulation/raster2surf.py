##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the pyReef carbonate platform modelling application.     ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
This module encapsulates functions related to the creation of regular pyReef surface
from raster type digital elevation model (DEM).
"""

import os
import numpy
import pandas
import os.path
import mpi4py.MPI as mpi

from scipy import interpolate

class raster2surf:
    """
    This class is useful for building the pyReef surface grid from a rectangular grid (DEM). This grid
    is used to generate the regular surface on which the interactions between wave processes
    and underlying carbonate stratigraphy will be computed.

    The purpose of the class is:
        1. to read and store the regular grid coordinates in numpy arrays format
        2. to define the resolution of the grid computation and adding ghost vertices
        3. to define the partition in case several processors are used

    Parameters
    ----------
    string : inputfile
        This is a string containing the path to the regular grid file.

    integer : rank
        Rank of processor.

    string : delimiter
        The delimiter between columns from the regular grid file. The regular file contains
        coordinates of each nodes and is ordered by row from SW to NE corner. The file has no
        header.
        Default: ' '

    variable: resRecFactor
        This integer gives the factor that will be used to define the resolution of the
        pyReef regular grid
                    >> pyReef resolution = DEM resolution x resRecFactor
        Default: 1
    """

    def __init__(self, inputfile=None, rank=0, delimiter=r'\s+', resRecFactor=1):

        if inputfile==None:
            raise RuntimeError('DEM input file name must be defined to construct pyReef mesh.')
        if not os.path.isfile(inputfile):
            raise RuntimeError('The DEM input file name cannot be found in your path.')
        self.inputfile = inputfile

        self.delimiter = delimiter

        self.resRecFactor = resRecFactor

        # Define class parameters
        self.nx = None
        self.ny = None
        self.res = None
        self.demX = None
        self.demY = None
        self.regX = None
        self.regY = None
        self.regZ = None
        self.res = None

        # Partition IDs
        self.partIDs = None

        # Surface creation and partition
        self._assign_surface_from_file()

    def _read_raster(self):
        """
        Using Pandas library to read the DEM file and allocating nodes and edges.
        """

        # Read DEM file
        data = pandas.read_csv(self.inputfile, sep=self.delimiter, engine='c',
                               header=None, na_filter=False,
                               dtype=numpy.float, low_memory=False)
        demX = data.values[:,0]
        demY = data.values[:,1]
        demZ = data.values[:,2]
        demRes = demX[1]-demX[0]
        minX = demX.min()
        maxX = demX.max()
        minY = demY.min()
        maxY = demY.max()
        self.demX = demX
        self.demY = demY

        # Defines pyReef surface resolution
        self.res = demRes*self.resRecFactor

        nx = int(round((maxX-minX)/demRes+1))
        ny = int(round((maxY-minY)/demRes+1))

        # Define interpolation function if required
        if self.res != demRes:
            regX = numpy.arange(minX,maxX+demRes,demRes)
            regY = numpy.arange(minY,maxY+demRes,demRes)
            regZ = numpy.reshape(demZ,(len(regX),len(regY)),order='F')
            interpolate_fct = interpolate.interp2d(regY,regX,regZ,kind='cubic')


        minX -= self.res
        maxX += self.res
        minY -= self.res
        maxY += self.res

        if self.res != demRes:
            tmpX = numpy.arange(minX+self.res,maxX,self.res)
            tmpY = numpy.arange(minY+self.res,maxY,self.res)
            tmpZ = interpolate_fct(tmpY,tmpX)
            self.nx = len(tmpX)+2
            self.ny = len(tmpY)+2
        else:
            self.nx = nx+2
            self.ny = ny+2
            tmpZ = numpy.reshape(demZ,(nx,ny),order='F')

        # Define internal nodes coordinates
        self.regX = numpy.linspace(minX,maxX,self.nx)
        self.regY = numpy.linspace(minY,maxY,self.ny)
        self.regZ = numpy.zeros((self.nx,self.ny))
        self.regZ[1:1+tmpZ.shape[0],1:1+tmpZ.shape[1]] = tmpZ

        # Define ghosts coordinates
        self.regZ[0,:] = self.regZ[1,:]
        self.regZ[self.nx-1,:] = self.regZ[self.nx-2,:]
        self.regZ[:,0] = self.regZ[:,1]
        self.regZ[:,self.ny-1] = self.regZ[:,self.ny-2]
        self.regZ[0,0] = self.regZ[1,1]
        self.regZ[self.nx-1,0] = self.regZ[self.nx-2,1]
        self.regZ[self.nx-1,self.ny-1] = self.regZ[self.nx-2,self.ny-2]
        self.regZ[0,self.ny-1] = self.regZ[1,self.ny-2]

    def _get_closest_factors(self, size):
        """
        This function finds the two closest integers which, when multiplied, equal a given number.
        This is used to defined the partition of the regular grid.

        Parameters
        ----------
        variable : size
            Integer corresponding to the number of CPUs that are used.

        Return
        ----------
        variable : nb1, nb2
            Integers which specify the number of processors along each axis.
        """
        factors =  []
        for i in range(1, size + 1):
            if size % i == 0:
                factors.append(i)
        factors = numpy.array(factors)
        if len(factors)%2 == 0:
            n1 = int( len(factors)/2 ) - 1
            n2 = n1 + 1
        else:
            n1 = int( len(factors)/2 )
            n2 = n1
        nb1 = factors[n1]
        nb2 = factors[n2]
        if nb1*nb2 != size:
            raise ValueError('Error in the decomposition grid: the number of domains \
            decomposition is not equal to the number of CPUs allocated')

        return nb1, nb2

    def _surface_partition(self, Xp=1, Yp=1):
        """
        This function defines a simple partitioning of the computational domain based on
        row and column wise decomposition. The method is relatively fast compared to other techniques
        but lack load-balancing operations.

        The purpose of the class is:
            1. to efficiently decomposed the domain from the number of processors defined along the X and Y axes
            2. to return to all processors the partition IDs for each vertice of the TIN

        Parameters
        ----------

        variable : Xp, Yp
            Integers which specify the number of processors along each axis. It is a requirement that
            the number of processors used matches the proposed decomposition:
                        >> nb CPUs = Xp x Yp
        """

        # Initialise MPI communications
        comm = mpi.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        self.partIDs = numpy.zeros((size,2),dtype=numpy.uint32)

        npX = int( len(self.regX)/size )
        for p in range(size):
            if p == 0:
                self.partIDs[p,0] = 0
            else:
                self.partIDs[p,0] = p*npX-1
            self.partIDs[p,1] = p*npX+npX
        self.partIDs[size-1,1] = len(self.regX)-1

        '''
        if Xp == 1 and Yp == 1 and size > 1:
            n1,n2 = self._get_closest_factors(size)
            if len(self.regX) > len(self.regY) :
                npX = n2
                npY = n1
            else:
                npX = n1
                npY = n2
        else:
            npX = Xp
            npY = Yp

        # Check decomposition versus CPUs number
        if size != npX*npY:
            raise ValueError('Error in the decomposition grid: the number of domains \
            decomposition is not equal to the number of CPUs allocated')

        # Define for each partition start and finish (i,j) indices
        partI = numpy.zeros((npX,2),dtype=numpy.uint32)
        partJ = numpy.zeros((npY,2),dtype=numpy.uint32)
        tmpIJ = numpy.zeros((size,4),dtype=numpy.uint32)

        # Get i extent of X partition
        nbX = int(len(self.regX)/npX)
        for p in range(npX):
            partI[p,0] = p*nbX
            partI[p,1] = partI[p,0]+nbX-1
        partI[npX-1,1] = len(self.regX)-1

        # Get j extent of Y partition
        nbY = int(len(self.regY)/npY)
        for p in range(npY):
            partJ[p,0] = p*nbY
            partJ[p,1] = partJ[p,0]+nbY-1
        partJ[npY-1,1] = len(self.regY)-1

        p = 0
        for j in range(npY):
            for i in range(npX):
                tmpIJ[p,0] = partI[i,0]
                tmpIJ[p,1] = partI[i,1]
                tmpIJ[p,2] = partJ[j,0]
                tmpIJ[p,3] = partJ[j,1]
                p = p+1

        self.partIJ = tmpIJ
        '''

    def _assign_surface_from_file(self):
        """
        Main function used to create the surface for pyReef based on
        a regular DEM file.
        """

        # Read raster file and define computational grid
        self._read_raster()

        # Define the surface partition
        self._surface_partition(Xp=1, Yp=1)

        return
