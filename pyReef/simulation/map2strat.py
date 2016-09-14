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
import h5py
import numpy
import pandas
import os.path
import mpi4py.MPI as mpi

from scipy import interpolate

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
        self.xdmffile = self.folder+'/stratal.series.xdmf'
        self.h5file = 'h5/stratal.time'
        self.xmffile = 'xmf/stratal.time'

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

        self.pnx = numpy.zeros(self.size,dtype=int)
        for s in range(self.size):
            self.pnx[s] = surf.partIDs[s,1]-surf.partIDs[s,0]+1
        self.pnx[0] -= 1
        self.pnx[self.size-1] -= 1

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

    def write_mesh(self, elev, time, outstep):
        """
        Create a hdf5 grid based on current time step stratal parameters.

        Parameters
        ----------

        variable : elev
            Elevation at current time step.

        variable : time
            Simulation current time step.

        variable : outstep
            Output time step.
        """

        h5file = self.folder+'/'+self.h5file+str(outstep)+'.p'+str(self.rank)+'.hdf5'

        x = numpy.zeros((self.partnx,self.partny,self.layID+1),dtype=float)
        y = numpy.zeros((self.partnx,self.partny,self.layID+1),dtype=float)
        z = numpy.zeros((self.partnx,self.partny,self.layID+1),dtype=float)
        h = numpy.zeros((self.partnx,self.partny,self.layID+1),dtype=float)
        l = numpy.zeros((self.partnx,self.partny,self.layID+1),dtype=int)
        sed = numpy.zeros((self.partnx,self.partny,self.layID+1,self.faciesNb),dtype=float)

        for k in range(self.layID,-1,-1):
            x[:,:,k] = self.x[:,:]
            y[:,:,k] = self.y[:,:]
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
                    for s in range(self.faciesNb):
                        if k == 0 and h[n,m,k]>0:
                            sed[n,m,k,s] = self.sedTH[n,m,k+1,s]/h[n,m,k]
                        elif h[n,m,k]>0:
                            sed[n,m,k,s] = self.sedTH[n,m,k,s]/h[n,m,k]
                    n += 1

        xyz = numpy.zeros((self.partnx*self.partny*(self.layID+1),3))
        th = numpy.zeros((self.partnx*self.partny*(self.layID+1)))
        layI = numpy.zeros((self.partnx*self.partny*(self.layID+1)),dtype=int)
        sedI = numpy.zeros((self.partnx*self.partny*(self.layID+1),self.faciesNb))

        p = 0
        for k in range(0,self.layID+1):
            for j in range(0,self.partny):
                for i in range(0,self.partnx):
                    xyz[p,0] = x[i,j,k]
                    xyz[p,1] = y[i,j,k]
                    xyz[p,2] = z[i,j,k]
                    th[p] = h[i,j,k]
                    layI[p] = l[i,j,k]
                    for s in range(0,self.faciesNb):
                        sedI[p,s] = sed[i,j,k,s]
                    p += 1

        nbpoints = len(layI)
        with h5py.File(h5file, "w") as f:
            # Write node coordinates
            f.create_dataset('xyz',shape=(nbpoints,3), dtype='float32', compression='gzip')
            f["xyz"][:,:] = xyz
            # Write thicknesses
            f.create_dataset('th',shape=(nbpoints,1), dtype='float32', compression='gzip')
            f["th"][:,0] = th
            # Write layer ID
            f.create_dataset('layI',shape=(nbpoints,1), dtype='int32', compression='gzip')
            f["layI"][:,0] = layI
            # Write sediment percentages
            for s in range(0,self.faciesNb):
                f.create_dataset('sed'+str(s),shape=(nbpoints,1), dtype='float32', compression='gzip')
                f["sed"+str(s)][:,0] = sedI[:,s]

        if self.rank == 0:
            self._write_xmf(time, outstep)

        return

    def _write_xmf(self, time, step):
         """
         This function writes the XmF file which is calling each HFD5 file.

         Parameters
         ----------

        variable : time
            Simulation current time step.

         variable: step
             Output visualisation step.
         """

         xmf_file = self.folder+'/'+self.xmffile+str(step)+'.xmf'
         f= open(str(xmf_file),'w')

         f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
         f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd">\n')
         f.write('<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">\n')
         f.write(' <Domain>\n')
         f.write('    <Grid GridType="Collection" CollectionType="Spatial">\n')
         f.write('      <Time Type="Single" Value="%s"/>\n'%time)

         for p in range(self.size):
             datfile = self.h5file+str(step)+'.p'+str(p)+'.hdf5'
             f.write('      <Grid Name="Block.%s">\n' %(str(p)))
             f.write('         <Topology TopologyType="3DSMesh" Dimensions="%d %d %d"/>\n'%(self.layID+1,self.partny,self.pnx[p]))
             f.write('         <Geometry GeometryType="XYZ">\n')
             f.write('            <DataItem Dimensions="%d %d %d 3" Format="HDF" NumberType="Float" Precision="4" '%(self.pnx[p],self.partny,self.layID+1))
             f.write('>%s:/xyz</DataItem>\n'%(datfile))
             f.write('         </Geometry>\n')
             f.write('         <Attribute Type="Scalar" Center="Node" Name="Thickness">\n')
             f.write('            <DataItem Format="HDF" NumberType="Float" Precision="4" Dimensions="%d %d %d" '%(self.pnx[p],self.partny,self.layID+1))
             f.write('>%s:/th</DataItem>\n'%(datfile))
             f.write('         </Attribute>\n')
             f.write('         <Attribute Type="Scalar" Center="Node" Name="Layer ID">\n')
             f.write('            <DataItem Format="HDF" NumberType="Int" Dimensions="%d %d %d" '%(self.pnx[p],self.partny,self.layID+1))
             f.write('>%s:/layI</DataItem>\n'%(datfile))
             f.write('         </Attribute>\n')
             for s in range(self.faciesNb):
                 f.write('         <Attribute Type="Scalar" Center="Node" Name="%s">\n'%self.sedName[s])
                 f.write('            <DataItem Format="HDF" NumberType="Float" Precision="4" Dimensions="%d %d 1%d" '%(self.pnx[p],self.partny,self.layID+1))
                 f.write('>%s:/sed%d</DataItem>\n'%(datfile,s))
                 f.write('         </Attribute>\n')
             f.write('      </Grid>\n')

         f.write('    </Grid>\n')
         f.write(' </Domain>\n')
         f.write('</Xdmf>\n')
         f.close()

         self._write_xdmf(step)

         return

    def _write_xdmf(self, step):
        """
        This function writes the XDmF file which is calling the XmF file.

        Parameters
        ----------

        variable: step
            Output visualisation step.
        """

        f= open(self.xdmffile,'w')

        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd">\n')
        f.write('<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">\n')
        f.write(' <Domain>\n')
        f.write('    <Grid GridType="Collection" CollectionType="Temporal">\n')
        for p in range(step+1):
            xfile = self.xmffile+str(p)+'.xmf'
            f.write('      <xi:include href="%s" xpointer="xpointer(//Xdmf/Domain/Grid)"/>\n' %xfile)
        f.write('    </Grid>\n')
        f.write(' </Domain>\n')
        f.write('</Xdmf>\n')
        f.close()

        return
