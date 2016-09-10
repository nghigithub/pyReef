##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the pyReef carbonate platform modelling application.     ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
This module encapsulates functions related to the creation of regular pyReef stratigraphy
from maps of thicknesses and sediments characteristics.
"""

import os
import numpy
import pandas
import os.path
import mpi4py.MPI as mpi

from scipy import interpolate
from pyevtk.hl import gridToVTK

class map2strat:
    """
    This class is use to build pyReef stratigraphic mesh from a set of initial layers.

    Parameters
    ----------
    class : input
        This is the class containing the parameters defined in the input file.

    class : surf
        This is the class containing the parameters defined in the surface grid.
    """

    def __init__(self, input, surf):
        """
        Constructor.
        """

        # Initialise MPI communications
        self.comm = mpi.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.layNb = int((input.tEnd - input.tStart)/input.laytime)+input.stratlays+1


        self.outTime = numpy.zeros(self.layNb,dtype=float)
        self.outTime[0] = input.tStart
        for k in range(1,self.layNb):
            self.outTime[k] = self.outTime[k-1] + input.tDisplay

        self.faciesNb = input.faciesNb
        self.layID = input.stratlays
        self.sedName = input.faciesName
        self.nx = surf.nx
        self.ny = surf.ny
        self.res = surf.res
        self.regX = surf.regX
        self.regY = surf.regY
        self.demX = surf.demX
        self.demY = surf.demY
        self.regZ = surf.regZ
        self.Afac = input.Afactor

        self.folder = input.outDir
        self.pvdfile = self.folder+'/stratal.series.pvd'
        self.vtkfile = self.folder+'/vtk/stratal.time'
        self.vtkf = 'vtk/stratal.time'

        minX = surf.demX.min()
        maxX = surf.demX.max()
        minY = surf.demY.min()
        maxY = surf.demY.max()
        demRes = surf.demX[1]-surf.demX[0]
        self.demnx = int(round((maxX-minX)/demRes+1))
        self.demny = int(round((maxY-minY)/demRes+1))

        if self.Afac != 1:
            self.demX = numpy.arange(minX,maxX+demRes,demRes)
            self.demY = numpy.arange(minY,maxY+demRes,demRes)


        self.partny = surf.ny - 2
        self.j1 = 1
        self.j2 = surf.ny-1

        self.partnx = surf.partIDs[self.rank,1]-surf.partIDs[self.rank,0]+1
        if self.rank == 0:
            self.partnx -= 1
        if self.rank == self.size-1:
            self.partnx -= 1

        if self.rank == 0:
            self.i1 = surf.partIDs[self.rank,0]+1
        else:
            self.i1 = surf.partIDs[self.rank,0]
        if self.rank == self.size-1:
            self.i2 = surf.partIDs[self.rank,1]
        else:
            self.i2 = surf.partIDs[self.rank,1]+1

        self.x = numpy.zeros((self.partnx,self.partny))
        self.y = numpy.zeros((self.partnx,self.partny))
        m = -1
        for j in range(self.j1,self.j2):
            m += 1
            n = 0
            for i in range(self.i1,self.i2):
                self.x[n,m] = surf.regX[i]
                self.y[n,m] = surf.regY[j]
                n += 1

        self.stratTH = numpy.zeros((self.partnx,self.partny,self.layNb+1),dtype=float)
        self.sedTH = numpy.zeros((self.partnx,self.partny,self.layNb+1,self.faciesNb),dtype=float)

        # Loop through the underlying layers
        for l in range(self.layID,0,-1):

            # Uniform thickness value
            if input.thickMap[l-1] == None:
                self.stratTH[:,:,l] = input.thickVal[l-1]

            # Stratal thickness map
            elif input.thickMap[l-1] != None:
                self.stratTH[:,:,l] = self._load_layer_thmap(input.thickMap[l-1])

            # Uniform facies percentages value
            if input.stratMap[l-1] == None:
                for f in range(0,self.faciesNb):
                    self.sedTH[:,:,l,f] = input.stratVal[l-1,f]*self.stratTH[:,:,l]
            # Facies percentages map
            elif input.stratMap[l-1] != None:
                perc = self._load_layer_percmap(input.stratMap[l-1],self.stratTH[:,:,l])
                self.sedTH[:,:,l,:] = perc*self.stratTH[:,:,l]

        return

    def _load_layer_thmap(self, thfile):
        """
        Load initial layer thickness map and perform interpolation from dem grid to pyReef mesh.

        Parameters
        ----------
        float : thfile
            Requested thickness map file to load.

        Return
        ----------
        variable: thick
            Numpy array containing the updated thicknesses for the considered domain.
        """

        thick = numpy.zeros((self.partnx,self.partny), dtype=float)
        thMap = pandas.read_csv(thfile, sep=r'\s+', engine='c', header=None, na_filter=False, \
                               dtype=numpy.float, low_memory=False)

        if self.Afac == 1:
            rectTH = numpy.reshape(thMap.values,(self.demnx,self.demny),order='F')
        else:
            rth = numpy.reshape(thMap.values,(self.demnx,self.demny),order='F')
            interpolate_fct = interpolate.interp2d(self.demY,self.demX,rth,kind='cubic')
            rectTH = interpolate_fct(self.regY[1:self.ny-1],self.regX[1:self.nx-1])

        # Define internal nodes thickness
        thick = rectTH[self.i1:self.i2,self.j1:self.j2]

        return thick

    def _load_layer_percmap(self, percfile, thval):
        """
        Load initial layer percentages map and perform interpolation from dem grid to pyReef mesh.

        Parameters
        ----------
        float : percfile
            Requested percentage map file to load.

        float : thval
            Thickness of the layer.

        Return
        ----------
        variable: perc
            Numpy array containing the updated percentages for the considered domain.
        """

        perc = numpy.zeros((self.partnx,self.partny,self.faciesNb), dtype=float)
        pMap = pandas.read_csv(percfile, sep=r',', engine='c', header=None, na_filter=False, \
                               dtype=numpy.float, low_memory=False)

        for f in range(self.faciesNb):

            if self.Afac == 1:
                rectPerc = numpy.reshape(pMap.values[:,f],(self.demnx,self.demny),order='F')
            else:
                rperc = numpy.reshape(pMap.values[:,f],(self.demnx,self.demny),order='F')
                interpolate_fct = interpolate.interp2d(self.demY,self.demX,rperc,kind='cubic')
                rectPerc = interpolate_fct(self.regY[1:self.ny-1],self.regX[1:self.nx-1])

            # Define internal nodes percentages
            perc[:,:,f] = rectPerc[self.i1:self.i2,self.j1:self.j2]

        # Normalise dataset
        # TODO: make it quicker
        for i in range(0,self.partnx):
            for j in range(0,self.partny):
                if thval > 0.:
                    lper = numpy.copy(perc[i,j,:])
                    lper[lper < 0.] = 0.
                    lper /= sum(lper)
                    tmp = sum(lper)
                    eps = 1.-tmp
                    if abs(eps) > 0.:
                        ids = numpy.where(numpy.logical_and( lper>abs(eps), lper<1-abs(per)))[0]
                        lper[ids[0]] += eps
                    perc[i,j,:] = lper
                else:
                    perc[i,j,:] = 0.

        return perc

    def _write_pvd(self, step):
        """
        This function writes the PVD file which is calling vtk mesh files.

        Parameters
        ----------

        variable: step
            Output visualisation step.
        """

        f= open(self.pvdfile,'w')

        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1"\n')
        f.write('       byte_order="LittleEndian" \n       compressor="vtkZLibDataCompressor">\n')
        f.write('   <Collection>\n')
        for k in range(step+1):
            time = self.outTime[k]
            for p in range(self.size):
                xfile = self.vtkf+str(k)+'.p'+str(p)+'.vts'
                f.write('       <DataSet timestep="%f" part="%d" file="%s"/>\n' %(time,p,xfile))
        f.write('   </Collection>\n')
        f.write('</VTKFile>\n')
        f.close()

        return

    def write_mesh(self, elev, time, outstep):
        """
        Create a vtk unstructured grid based on current time step stratal parameters.

        Parameters
        ----------

        variable : elev
            Elevation at current time step.

        variable : time
            Simulation current time step.

        variable : outstep
            Output time step.
        """

        vtkfile = self.vtkfile+str(outstep)+'.p'+str(self.rank)

        x = numpy.zeros((self.partnx,self.partny,self.layNb),dtype=float)
        y = numpy.zeros((self.partnx,self.partny,self.layNb),dtype=float)
        z = numpy.zeros((self.partnx,self.partny,self.layNb),dtype=float)
        h = numpy.zeros((self.partnx,self.partny,self.layNb),dtype=float)
        l = numpy.zeros((self.partnx,self.partny,self.layNb),dtype=int)
        if self.faciesNb>=1:
            sed0 = numpy.zeros((self.partnx,self.partny,self.layNb),dtype=float)
        if self.faciesNb>=2:
            sed1 = numpy.zeros((self.partnx,self.partny,self.layNb),dtype=float)
        if self.faciesNb>=3:
            sed2 = numpy.zeros((self.partnx,self.partny,self.layNb),dtype=float)
        if self.faciesNb>=4:
            sed3 = numpy.zeros((self.partnx,self.partny,self.layNb),dtype=float)
        if self.faciesNb>=5:
            sed4 = numpy.zeros((self.partnx,self.partny,self.layNb),dtype=float)
        if self.faciesNb>=6:
            sed5 = numpy.zeros((self.partnx,self.partny,self.layNb),dtype=float)

        for k in range(self.layID,-1,-1):
            m = -1

            for j in range(self.j1,self.j2):
                m += 1
                n = 0
                for i in range(self.i1,self.i2):
                    if k == self.layID:
                        z[n,m,k] = elev[i,j]
                    else:
                        z[n,m,k] = z[n,m,k+1] - self.stratTH[n,m,k+1]

                    if k == 0:
                        h[n,m,k] = self.stratTH[n,m,k+1]
                    else:
                        h[n,m,k] = self.stratTH[n,m,k]
                    l[:,:,k] = k
                    if self.faciesNb>=1:
                        if k == 0 and h[n,m,k]>0:
                            sed0[n,m,k] = self.sedTH[n,m,k+1,0]/h[n,m,k]
                        elif h[n,m,k]>0:
                            sed0[n,m,k] = self.sedTH[n,m,k,0]/h[n,m,k]
                    if self.faciesNb>=2:
                        if k == 0 and h[n,m,k]>0:
                            sed1[n,m,k] = self.sedTH[n,m,k+1,1]/h[n,m,k]
                        elif h[n,m,k]>0:
                            sed1[n,m,k] = self.sedTH[n,m,k,1]/h[n,m,k]
                    if self.faciesNb>=3:
                        if k == 0 and h[n,m,k]>0:
                            sed2[n,m,k] = self.sedTH[n,m,k+1,2]/h[n,m,k]
                        elif h[n,m,k]>0:
                            sed2[n,m,k] = self.sedTH[n,m,k,2]/h[n,m,k]
                    if self.faciesNb>=4:
                        if k == 0 and h[n,m,k]>0:
                            sed3[n,m,k] = self.sedTH[n,m,k+1,3]/h[n,m,k]
                        elif h[n,m,k]>0:
                            sed3[n,m,k] = self.sedTH[n,m,k,3]/h[n,m,k]
                    if self.faciesNb>=5:
                        if k == 0 and h[n,m,k]>0:
                            sed4[n,m,k] = self.sedTH[n,m,k+1,4]/h[n,m,k]
                        elif h[n,m,k]>0:
                            sed4[n,m,k] = self.sedTH[n,m,k,4]/h[n,m,k]
                    if self.faciesNb>=6:
                        if k == 0 and h[n,m,k]>0:
                            sed5[n,m,k] = self.sedTH[n,m,k+1,5]/h[n,m,k]
                        elif h[n,m,k]>0:
                            sed5[n,m,k] = self.sedTH[n,m,k,5]/h[n,m,k]
                    n += 1

        for k in range(0,self.layNb):
            x[:,:,k] = self.x[:,:]
            y[:,:,k] = self.y[:,:]
            if k>self.layID:
                l[:,:,k] = -1
                z[:,:,k] = z[:,:,self.layID]
                h[:,:,k] = h[:,:,self.layID]
                if self.faciesNb>=1:
                    sed0[:,:,k] = sed0[:,:,self.layID]
                if self.faciesNb>=2:
                    sed1[:,:,k] = sed1[:,:,self.layID]
                if self.faciesNb>=3:
                    sed2[:,:,k] = sed2[:,:,self.layID]
                if self.faciesNb>=4:
                    sed3[:,:,k] = sed3[:,:,self.layID]
                if self.faciesNb>=5:
                    sed4[:,:,k] = sed4[:,:,self.layID]
                if self.faciesNb>=6:
                    sed5[:,:,k] = sed5[:,:,self.layID]


        if self.faciesNb==1:
            gridToVTK(vtkfile, x, y, z, pointData = {"layer thickness" :h, "layer number" : l,
                      self.sedName[0] : sed0})
        elif self.faciesNb==2:
            gridToVTK(vtkfile, x, y, z, pointData = {"layer thickness" :h, "layer number" : l,
                      self.sedName[0] : sed0, self.sedName[1] : sed1})
        elif self.faciesNb==3:
            gridToVTK(vtkfile, x, y, z, pointData = {"layer thickness" :h, "layer number" : l,
                     self.sedName[0] : sed0, self.sedName[1] : sed1,
                     self.sedName[2] : sed2})
        elif self.faciesNb==4:
            gridToVTK(vtkfile, x, y, z, pointData = {"layer thickness" :h, "layer number" : l,
                     self.sedName[0] : sed0, self.sedName[1] : sed1,
                     self.sedName[2] : sed2, self.sedName[3] : sed3})
        elif self.faciesNb==5:
            gridToVTK(vtkfile, x, y, z, pointData = {"layer thickness" :h, "layer number" : l,
                     self.sedName[0] : sed0, self.sedName[1] : sed1,
                     self.sedName[2] : sed2, self.sedName[3] : sed3,
                     self.sedName[4] : sed4})
        elif self.faciesNb==6:
            gridToVTK(vtkfile, x, y, z, pointData = {"layer thickness" :h, "layer number" : l,
                     self.sedName[0] : sed0, self.sedName[1] : sed1,
                     self.sedName[2] : sed2, self.sedName[3] : sed3,
                     self.sedName[4] : sed4, self.sedName[5] : sed5})
        else:
            print 'Number of sediment is limited to 6.'

        if self.rank == 0:
            self._write_pvd(outstep)

        return
