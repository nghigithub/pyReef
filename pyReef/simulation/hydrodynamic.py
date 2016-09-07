##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the pyReef carbonate platform modelling application.     ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
This module encapsulates functions related to hydrodynamic calculation in pyReef model.
"""

import time
import math
import numpy
import mpi4py.MPI as mpi
from scipy import interpolate
from scipy.spatial import cKDTree
from collections import OrderedDict
from matplotlib import _cntr as cntr
from scipy.interpolate import interpn
from scipy.ndimage.filters import gaussian_filter
from shapely.geometry import LineString, LinearRing

from pyReef.libUtils  import simswan as swan
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter

class hydrodynamic:

    def __init__(self, CKomar, sigma, res, wfac, xv, yv, partIDs):
        """
        Constructor.
        """

        self.rank = mpi.COMM_WORLD.rank
        self.size = mpi.COMM_WORLD.size
        self.comm = mpi.COMM_WORLD
        self.fcomm = mpi.COMM_WORLD.py2f()

        self.nbtides = 1
        self.secyear = 3.154e+7
        self.tWave = None
        self.tSed = None
        self.CKomar = CKomar
        self.sigma = sigma
        self.gravity = 9.81
        self.porosity = None
        self.dh = None
        self.d50 = None
        self.efficiency = None
        self.Urms = None
        self.sl = None
        self.currentU = None
        self.currentD = None
        self.qt = None
        self.bedCx = None
        self.bedCy = None
        self.erosion = None
        self.deprate = None
        self.dSVR = None
        self.Dstar = None
        self.rho_water = None
        self.rho_sed = None
        self.relativeDens = None
        self.kinematic_viscosity = 0.801e-6

        self.xv = xv
        self.yv = yv
        self.xi, self.yi = numpy.meshgrid(xv, yv, sparse=False, indexing='ij')
        xx = numpy.ravel(self.xi,order='F')
        yy = numpy.ravel(self.yi,order='F')
        self.xyi = numpy.vstack((xx, yy)).T

        self.res = res
        self.wres = res * wfac

        self.partIDs = partIDs
        self.i1 = partIDs[self.rank,0]
        self.i2 = partIDs[self.rank,1]

        self.Wfac = wfac
        if self.Wfac > 1:
            nx = int(len(xv)/self.Wfac)
            self.swanX = numpy.linspace(xv.min(), xv.max(), num=nx)
            ny = int(len(yv)/self.Wfac)
            self.swanY = numpy.linspace(yv.min(), yv.max(), num=ny)

        self.ring = LinearRing([(-res, -res),(0,-res),(res,-res), (res,0),(res, res),(0, res),(-res,res),(-res,0)])
        self.pairs = list(self._setpairs(list(self.ring.coords)))

        self.idI = None
        self.idJ = None

        return

    def _setpairs(self, lst):

         for i in range(1, len(lst)):
             yield lst[i-1], lst[i]

         return

    def _interpolate_to_wavegrid(self, elev):
         """
         This function interpolates pyReef simulation elevation to swan grid.

         Parameters
         ----------

        variable : elev
            Elevation data from pyReef simulation grid.

         """

         if self.Wfac > 1:
            interpolate_fct = interpolate.interp2d(self.yv,self.xv,elev,kind='cubic')
            Welev = interpolate_fct(self.swanY,self.swanX)
            return Welev
         else:
            return elev

    def _interpolate_to_sedgrid(self, data):
         """
         This function interpolate swan model output to pyReef simulation grid.

         Parameters
         ----------

        variable : data
            Swan output dataset to interpolate.

         """

         if self.Wfac > 1:
            interpolate_fct = interpolate.interp2d(self.swanY,self.swanX,data,kind='cubic')
            Wdata = interpolate_fct(self.yv,self.xv)
            return Wdata
         else:
            return data

    def swan_init(self, input, z, wID, sl, tide):
         """
         This function initialise swan model.

         Parameters
         ----------

        variable : input
            Simulation input parameters time step.

         variable: z
            Elevation.

         variable: wID
             Wave number ID.

         variable: sl
            Sealevel elevation.

         variable: tide
            Tide range.
         """

         # Define variables for sediment transport
         self.d50 = input.faciesDiam[-1]
         self.rho_sed = input.density[-1]
         self.rho_water = input.waterD
         self.porosity = input.porosity[-1]
         self.diffusion = input.diffusion[-1]
         self.efficiency = input.efficiency[-1]

         self.tSed = input.tSed
         self.tWave = input.tWave
         self.relativeDens = self.rho_sed/self.rho_water
         self.Dstar = self.d50*(self.gravity*(self.relativeDens-1.)/self.kinematic_viscosity**2)**(1./3.)
         self.dSVR = ((self.relativeDens-1.)*self.gravity*self.d50)**1.2

         wl = input.wavelist[wID]
         cl = input.climlist[wID]

         elev = self._interpolate_to_wavegrid(z)
         wD = input.waveWd[wl][cl] + 2.*input.waveWs[wl][cl]*(numpy.random.rand()-0.5)
         swan.model.init(self.fcomm, input.swanFile, input.swanInfo,
                        input.swanBot, input.swanOut, elev, input.waveWh[wl][cl],
                        input.waveWp[wl][cl], wD,
                        self.wres, input.waveBase, sl-tide)

         self.idI = numpy.tile(numpy.arange(z.shape[0]),(z.shape[1],1)).T
         self.idJ = numpy.tile(numpy.arange(z.shape[1]),(z.shape[0],1))

         return

    def swan_run(self, input, force, z, wID):
         """
         This function run swan model and computes near-bed velocity
         combining cross-shore and long-shore component.

         Parameters
         ----------

         variable : input
            Simulation input parameters time step.

         variable: force
            Forcings conditions.

         variable: z
            Elevation.

         variable: WID
             Wave number ID.
         """

         self.nbtides = 1
         if force.tideval != 0:
             self.nbtides = 2

         # Initialise morphological changes
         self.sl = []
         self.dh = []

         # Initialise wave parameters
         self.Urms = []
         force.wavU = []
         force.wavV = []
         force.wavH = []
         force.Perc = []
         force.bedLay = []

         # Initialise longshore drift
         self.currentU = []
         self.currentD = []

         # Loop through the different wave climates and store swan output information
         force.wclim = input.climNb[input.wavelist[wID]]

         for clim in range(force.wclim):

            # Define next wave regime
            tw = time.clock()

            for tide in range(self.nbtides):

                # Update elevation if required
                elev = self._interpolate_to_wavegrid(z)

                # Get current climate parameters
                force.Perc.append(input.wavePerc[input.wavelist[wID]][input.climlist[wID]])
                force.bedLay.append(input.bedlay[input.wavelist[wID]][input.climlist[wID]])
                storm = input.storm[input.wavelist[wID]][input.climlist[wID]]

                if force.tideval != 0 and tide == 0:
                    wl = input.wavelist[wID]
                    cl = input.climlist[wID]
                    sl = force.sealevel - force.tideval
                    self.sl.append(sl)
                elif force.tideval != 0 and tide == 1:
                    wID += 1
                    wl = input.wavelist[wID]
                    cl = input.climlist[wID]
                    sl = force.sealevel + force.tideval
                    self.sl.append(sl)
                else:
                    wID += 1
                    wl = input.wavelist[wID]
                    cl = input.climlist[wID]
                    sl = force.sealevel
                    self.sl.append(sl)

                # Run SWAN model
                wD = input.waveWd[wl][cl] + 2.*input.waveWs[wl][cl]*(numpy.random.rand()-0.5)

                if self.rank == 0:
                    print '-----------------'
                    if force.tideval != 0 and tide == 0:
                        tw1 = input.wavelist[wID]
                        tc1 = input.climlist[wID]
                        print 'Low-tide hydrodynamics: wave field %d and climatic conditions %d' %(tw1,tc1)
                    elif force.tideval != 0 and tide == 1:
                        tw1 = input.wavelist[wID-1]
                        tc1 = input.climlist[wID-1]
                        print 'High-tide hydrodynamics: wave field %d and climatic conditions %d' %(tw1,tc1)
                    else:
                        tw1 = input.wavelist[wID-1]
                        tc1 = input.climlist[wID-1]
                        print 'Hydrodynamics: waves field %d and climatic conditions %d:' %(tw1,tc1)

                sU, sD, sH = swan.model.run(self.fcomm, elev, input.waveWh[wl][cl], input.waveWp[wl][cl], wD, self.sl[-1])

                if self.rank == 0:
                    print '   -   Wave propagation took %0.02f seconds to run.' %(time.clock()-tw)

                tw = time.clock()
                wavU = self._interpolate_to_sedgrid(sU)
                wavD = self._interpolate_to_sedgrid(sD)
                H = self._interpolate_to_sedgrid(sH)
                H = 1.5*gaussian_filter(H, 1.)
                r,c = numpy.where(z-self.sl[-1]>=0)
                wavU[r,c] = 0.
                wavD[r,c] = 0.
                H[r,c] = 0.
                self.Urms.append(wavU)

                # Define cross-shore current
                lvl = input.waveBrk[wl][cl]
                if storm == 0:
                    cU = wavU * numpy.cos(wavD)
                    cV = wavU * numpy.sin(wavD)
                else:
                    r1,c1 = numpy.where(z-sl>2*lvl)
                    wavD[r1,c1] += numpy.pi
                    cU = wavU * numpy.cos(wavD)
                    cV = wavU * numpy.sin(wavD)

                # Compute long-shore velocity field
                brk = [lvl/2.,lvl,2*lvl]
                slongV, slongD = self._compute_longshore_velocity(z-self.sl[-1], wavD, wavU, brk)
                slongV = 1.5*gaussian_filter(slongV, 1.)
                slongV[r,c] = 0.

                # Set up long-shore current
                lU = slongV * numpy.cos(slongD)
                lV = slongV * numpy.sin(slongD)
                self.currentU.append(slongV)
                self.currentD.append(slongD)

                # Store each climate induced bottom currents velocity
                totU = 1.5*gaussian_filter(cU+lU, 1.)
                totU[r,c] = 0.
                totV = 1.5*gaussian_filter(cV+lV, 1.)
                totV[r,c] = 0.

                force.wavU.append(totU)
                force.wavV.append(totV)
                force.wavH.append(H)

                if self.rank == 0:
                    print '   -   Currents model took %0.02f seconds to run.' %(time.clock()-tw)

                # Perform morphological changes
                self._bed_elevation_change(input, force, z, lvl, len(self.sl)-1)

         return wID

    def _assignBCs(self, z):
        """
        Pads the boundaries of a grid. Boundary condition pads the boundaries
        with equivalent values to the data margins, e.g. x[-1,1] = x[1,1].
        It creates a grid 2 rows and 2 columns larger than the input.
        """

        bc = numpy.zeros((z.shape[0]+2,z.shape[1]+2))
        bc[1:-1,1:-1] = z

        # Assign boundary conditions - sides
        bc[0,1:-1] = z[0,:]
        bc[-1,1:-1] = z[-1,:]
        bc[1:-1,0] = z[:,0]
        bc[1:-1,-1] = z[:,-1]

        # Assign boundary conditions - corners
        bc[0,0] = z[0,0]
        bc[0,-1] = z[0,-1]
        bc[-1,0] = z[-1,0]
        bc[-1,-1] = z[-1,0]

        return bc

    def _calcFiniteSlopes(self, z):
        """
        Calculate slope with 2nd order/centered difference method.
        """

        # Assign boundary conditions
        bc = self._assignBCs(z)

        # Compute finite differences
        sx = (bc[1:-1,2:]-bc[1:-1,:-2])/(2*self.res)
        sy = (bc[2:,1:-1]-bc[:-2,1:-1])/(2*self.res)
        smag = numpy.sqrt(sx**2 + sy**2)

        return smag

    def _erosion_component(self, input, force, z, sl, clim):
         """
         This function computes the erosion rate in combined waves and currents
         based on Soulsby - Van Rijn

         Parameters
         ----------

         variable : input
            Simulation input parameters time step.

         variable: force
            Forcings conditions.

         variable: z
            Elevation.

         variable: sl
            Sea level.

         variable: clim
            Climatic condition.
         """

         # Compute slope
         smag = self._calcFiniteSlopes(z)

         # Initialise total load and deposition rate component
         self.qt = 0.
         self.bedCx = 0.
         self.bedCy = 0.
         self.erosion = 0.

         # Loop through the different wave climates and store swan output information
         depth = sl - z

         As = numpy.zeros((z.shape[0],z.shape[1]))
         CD = numpy.zeros((z.shape[0],z.shape[1]))
         Ucr = numpy.zeros((z.shape[0],z.shape[1]))
         qt = numpy.zeros((z.shape[0],z.shape[1]))

         row,col = numpy.where(depth>0.)

         As[row,col] = 0.005*depth[row,col]*(self.d50/depth[row,col])**1.2
         As[row,col] += 0.012*self.d50*self.Dstar**(-0.6)
         As /= self.dSVR
         CD[row,col] = (0.4/(numpy.log(depth[row,col]/0.006)-1.))**2

         if self.d50 <= 0.0005:
             Ucr[row,col] = 0.19*numpy.log10(4.*depth[row,col]/self.d50)*self.d50**0.1
         else:
             Ucr[row,col] = 8.5*numpy.log10(4.*depth[row,col]/self.d50)*self.d50**0.6

         Ucr[Ucr<0] = 0.
         frac = (1.-self.porosity)*self.res**2
         tmp = 1/(1.-self.porosity)

         # Define total load from Soulsby Van Rijn formulation
         #speed = numpy.sqrt(force.wavU[clim]**2+force.wavV[clim]**2)
         speed = self.currentU[clim]
         qt[row,col] = As[row,col] * speed[row,col] * ( numpy.sqrt(speed[row,col]**2 + \
            0.018*self.Urms[clim][row,col]/CD[row,col]) - Ucr[row,col] )**2.4 * (1.-1.6*smag[row,col])
         qt[numpy.isnan(qt)] = 0.
         self.erosion = qt/frac

         return

    def _zeros_border_ids(self, data):
         """
         Set border transport indices to 0.

         Parameters
         ----------

         variable: data
         """

         data[0,:] = 0
         data[-1,:] = 0
         data[:,0] = 0
         data[:,-1] = 0

    def _sediment_distribution(self, transD):
         """
         Define intersection points based on transport direction and specify for each nodes
         the proportion of sediment which moves to its neighborhood

         Parameters
         ----------

         variable: transD
            Transport direction.
         """

         trans = transD[self.i1:self.i2,:]

         lptI0 = numpy.zeros(transD.shape,dtype=int)
         lptI0.fill(-100)
         lptI1 = numpy.zeros(transD.shape,dtype=int)
         lptI1.fill(-100)
         lptJ0 = numpy.zeros(transD.shape,dtype=int)
         lptJ0.fill(-100)
         lptJ1 = numpy.zeros(transD.shape,dtype=int)
         lptJ1.fill(-100)
         lprop0 = numpy.zeros(transD.shape)
         lprop0.fill(-100.)
         lprop1 = numpy.zeros(transD.shape)
         lprop1.fill(-100.)

         listPts = [[(0,0),(self.res*2.*numpy.cos(trans[x,y]),self.res*2.*numpy.sin(trans[x,y]))]
                    for x in range(trans.shape[0]) for y in range(trans.shape[1])]
         ls = [LineString(lP) for lP in listPts]
         pts = [l.intersection(self.ring) for l in ls]
         xy = numpy.asarray([p.coords[0] for p in pts])

         ptI0 = numpy.zeros(len(pts),dtype=int)
         ptI1 = numpy.zeros(len(pts),dtype=int)
         ptJ0 = numpy.zeros(len(pts),dtype=int)
         ptJ1 = numpy.zeros(len(pts),dtype=int)
         prop0 = numpy.zeros(len(pts))
         prop1 = numpy.zeros(len(pts))

         for p in range(len(pts)):
            for pair in self.pairs:
                if LineString([pair[0],pair[1]]).contains(pts[p]):
                    x0,y0 = LineString([pair[0],pair[1]]).coords[0]
                    x1,y1 = LineString([pair[0],pair[1]]).coords[1]
                    ptI0[p],ptJ0[p] = int(x0/self.res),int(y0/self.res)
                    ptI1[p],ptJ1[p] = int(x1/self.res),int(y1/self.res)
                    dist0 = numpy.sqrt((x0-xy[p,0])**2+(y0-xy[p,1])**2)
                    dist1 = numpy.sqrt((x1-xy[p,0])**2+(y1-xy[p,1])**2)
                    p0 = dist0/(dist0+dist1)
                    if p0 > 1:
                        p0 = 1.
                    p1 = 1-p0
                    prop0[p] = p1
                    prop1[p] = p0
                    break

         lptI0[self.i1:self.i2,:] =  ptI0.reshape(trans.shape)
         lptI1[self.i1:self.i2,:] =  ptI1.reshape(trans.shape)
         lptJ0[self.i1:self.i2,:] =  ptJ0.reshape(trans.shape)
         lptJ1[self.i1:self.i2,:] =  ptJ1.reshape(trans.shape)
         lprop0[self.i1:self.i2,:] =  prop0.reshape(trans.shape)
         lprop1[self.i1:self.i2,:] =  prop1.reshape(trans.shape)

         self.ptI0 = lptI0.flatten()
         self.comm.Allreduce(mpi.IN_PLACE, self.ptI0, op=mpi.MAX)
         self.ptI0 = self.ptI0.reshape(transD.shape)
         self._zeros_border_ids(self.ptI0)
         self.ptI0 += self.idI

         self.ptI1 = lptI1.flatten()
         self.comm.Allreduce(mpi.IN_PLACE, self.ptI1, op=mpi.MAX)
         self.ptI1 = self.ptI1.reshape(transD.shape)
         self._zeros_border_ids(self.ptI1)
         self.ptI1 += self.idI

         self.ptJ0 = lptJ0.flatten()
         self.comm.Allreduce(mpi.IN_PLACE, self.ptJ0, op=mpi.MAX)
         self.ptJ0 = self.ptJ0.reshape(transD.shape)
         self._zeros_border_ids(self.ptJ0)
         self.ptJ0 += self.idJ

         self.ptJ1 = lptJ1.flatten()
         self.comm.Allreduce(mpi.IN_PLACE, self.ptJ1, op=mpi.MAX)
         self.ptJ1 = self.ptJ1.reshape(transD.shape)
         self._zeros_border_ids(self.ptJ1)
         self.ptJ1 += self.idJ

         self.prop0 = lprop0.flatten()
         self.comm.Allreduce(mpi.IN_PLACE, self.prop0, op=mpi.MAX)
         self.prop0 = self.prop0.reshape(transD.shape)
         self._zeros_border_ids(self.prop0)

         self.prop1 = lprop1.flatten()
         self.comm.Allreduce(mpi.IN_PLACE, self.prop1, op=mpi.MAX)
         self.prop1 = self.prop1.reshape(transD.shape)
         self._zeros_border_ids(self.prop1)

         return

    def _bed_elevation_change(self, input, force, z, lvl, clim):
         """
         This function computes the bed elevation change using the second order Lax Wendroff scheme

         Parameters
         ----------

         variable : input
            Simulation input parameters time step.

         variable: force
            Forcings conditions.

         variable: z
            Elevation.

         variable: lvl
            Contour current level.

         variable: clim
            Climatic condition.
         """

         # Initialise morphological changes
         dh = numpy.zeros(z.shape)
         diff = numpy.zeros(z.shape)
         tm = time.clock()
         dt = self.tWave*force.Perc[clim]*self.secyear/self.nbtides
         speed = numpy.sqrt(force.wavU[clim]**2+force.wavV[clim]**2)

         # Erosion thickness
         tw = time.clock()
         self._erosion_component(input, force, z, self.sl[clim], clim)
         entrain = -dt*self.erosion
         if self.sigma>0:
             entrain = 1.5*gaussian_filter(entrain, self.sigma)

         # Mobile bed layer thickness limitation
         entrain[entrain>0.] = 0.
         entrain[entrain<-force.bedLay[clim]] = -force.bedLay[clim]
         r,c = numpy.where(z-self.sl[-1]>=0)
         entrain[r,c] = 0.
         z += entrain
         if self.rank == 0:
             print '     +   Erosion computation took %0.02f seconds to run.' %(time.clock()-tw)

         # Combined wave current direction
         tw = time.clock()
         transD = numpy.zeros(z.shape)
         r,c = numpy.where(numpy.logical_and(force.wavU[clim] == 0,force.wavV[clim] >= 0))
         transD[r,c] = numpy.pi/2.
         r,c = numpy.where(numpy.logical_and(force.wavU[clim] == 0,force.wavV[clim] < 0))
         transD[r,c] = 3.*numpy.pi/2.
         r,c = numpy.where(transD==0)
         transD[r,c] = numpy.arctan2(force.wavV[clim][r,c],force.wavU[clim][r,c])

         # Define sediment distribution based on transport direction
         self._sediment_distribution(transD)
         if self.rank == 0:
            print '     +   Transport streamlines took %0.02f seconds to run.' %(time.clock()-tw)

         # Setup bedload and suspended load deposition array
         # Bedload: limit transport to surrounding cells
         # Suspended load: follow streamlines until all sediments have been deposited
         tw = time.clock()
         t_stp = 0
         tmpZ = numpy.copy(z-self.sl[-1])
         sedcharge = -entrain
         sumload = numpy.sum(sedcharge)
         depo = numpy.zeros(z.shape)
         newcharge = numpy.zeros(z.shape)
         while (numpy.sum(sedcharge) > 0.001 and t_stp <= 5000):
             for i in range(z.shape[0]):
                for j in range(z.shape[1]):
                   if sedcharge[i,j] > 0. and t_stp == 0:
                        tdist0 = self.prop0[i,j]*sedcharge[i,j]
                        tdist1 = self.prop1[i,j]*sedcharge[i,j]
                        newcharge[self.ptI0[i,j],self.ptJ0[i,j]] += tdist0
                        newcharge[self.ptI1[i,j],self.ptJ1[i,j]] += tdist1
                   # Move sediments to neighbouring cells
                   elif sedcharge[i,j] > 0.:
                       depo[i,j] += (1.-self.efficiency)*sedcharge[i,j]
                       tdist0 = self.prop0[i,j]*self.efficiency*sedcharge[i,j]
                       tdist1 = self.prop1[i,j]*self.efficiency*sedcharge[i,j]
                       newcharge[self.ptI0[i,j],self.ptJ0[i,j]] += tdist0
                       newcharge[self.ptI1[i,j],self.ptJ1[i,j]] += tdist1
             sedcharge = numpy.copy(newcharge)
             newcharge.fill(0.)
             t_stp += 1
         if t_stp > 5000 and self.rank == 0:
            print '         -   Force suspended sediment deposition.'
         depo += sedcharge

         # Combine erosion/deposition and update morphology for suspended load
         if self.sigma>0:
             depo = 1.5*gaussian_filter(depo, self.sigma)
         depo[depo<0] = 0.
         sud = numpy.sum(depo)
         depo = depo*sumload/sud
         z += depo
         if self.rank == 0:
             print '     +   Deposition computation took %0.02f seconds to run and converged in %d iterations.' %(time.clock()-tw,t_stp)

         # Diffusion transport
         tw = time.clock()
         dh[1:-1,1:-1] += dt*self.diffusion*( (z[:-2,1:-1]-2*z[1:-1,1:-1]+z[2:,1:-1]) +
                          (z[1:-1,:-2]-2*z[1:-1,1:-1]+z[1:-1,2:]) )/(self.res**2*self.secyear)
         self.dh.append(dh+entrain+depo)
         z += dh
         if self.rank == 0:
             print '     +   Diffusion computation took %0.02f seconds to run.' %(time.clock()-tw)

         if self.rank == 0:
             print '   -   Morphological change took %0.02f seconds to run.' %(time.clock()-tm)
             if abs(numpy.sum(self.dh[-1])) < 0.001:
                 print '   -   Mass balance check ok.'
             else:
                 print '   -   Mass balance check: ',numpy.sum(self.dh[-1])

         return

    def _compute_longshore_velocity(self, z, waved, waveU, lvl):
         """
         This function computes velocity long-shore component.

         Parameters
         ----------

         variable: z
            Elevation.

         variable : waved
            Swan output of wave direction.

         variable : waveU
            Swan output of near-bed orbital wave velocity.

         variable: lvl
            Breaking level defined in the input.
         """

         for l in range(len(lvl)):

            # Specify parameters for contours
            c = cntr.Cntr(self.xi, self.yi, z)

            level = lvl[l]

            # trace a contour
            result = c.trace(level)

            # result is a list of arrays of vertices and path codes
            nseg = len(result) // 2
            contours, codes = result[:nseg], result[nseg:]

            # Loop through each contour
            for c in range(len(contours)):
                tmpts =  contours[c]
                closed = False
                if tmpts[0,0] == tmpts[-1,0] and tmpts[0,1] == tmpts[-1,1]:
                    closed = True
                # Remove duplicate points
                unique = OrderedDict()
                for p in zip(tmpts[:,0], tmpts[:,1]):
                    unique.setdefault(p[:2], p)
                pts = numpy.asarray(unique.values())
                if closed:
                    cpts = numpy.zeros((len(pts)+1,2))
                    cpts[0:len(pts),0:2] = pts
                    cpts[-1,0:2] = pts[0,0:2]
                else:
                    cpts = pts

                if len(cpts) > 2:
                    # Create a spline from the contour points, also note that s=0
                    # needed in order to force the spline fit to pass through all the contour points.
                    tck,u = interpolate.splprep([cpts[:,0], cpts[:,1]], s=0)

                    # Evalute positions and first derivatives
                    nsamples = len(cpts[:,0])*10
                    x, y = interpolate.splev(numpy.linspace(0,1, nsamples), tck, der=0)
                    dx,dy = interpolate.splev(numpy.linspace(0,1, nsamples), tck, der=1)
                    dydx = dy/dx
                    xy = numpy.vstack((x,y)).T

                    # Delete points not in the simulation domain
                    inIDs = numpy.where(numpy.logical_and(numpy.logical_and(xy[:,0]>=self.xv.min(),xy[:,0]<=self.xv.max()),
                                                    numpy.logical_and(xy[:,1]>=self.yv.min(),xy[:,1]<=self.yv.max())))[0]

                    # Interplate wave direction on the contour points
                    wavD = interpn((self.xv, self.yv),waved,(xy[inIDs,:2]),method='linear')

                    # Interplate wave direction on the contour points
                    wavU = interpn((self.xv, self.yv),waveU,(xy[inIDs,:2]),method='splinef2d')
                    wavU[wavU<0.] = 0.

                    # Compute wave incidence angle and longshore current direction
                    incidence = numpy.zeros(len(inIDs))
                    longDir = numpy.zeros(len(inIDs))

                    for p in range(len(inIDs)):
                        p0 = xy[inIDs[p],0]
                        p01 = xy[inIDs[p],1]
                        p1 = xy[inIDs[p],0]-self.res
                        p2 = xy[inIDs[p],0]+self.res

                        # Set wave direction
                        wdir = numpy.tan(wavD[p])
                        wdirdeg = math.degrees(wavD[p])
                        if wdirdeg > 360. or wdirdeg < 0.:
                            wdirdeg = wdirdeg%360
                        if wdirdeg>90. and wdirdeg<=270.:
                            dx_w = [p0,p2]
                            wx = p2
                            wy = p01+wdir*(p2-p0)
                            dy_w = [p01,wy]
                            ew = 1
                        else:
                            dx_w = [p1,p0]
                            wx = p1
                            wy = p01+wdir*(p1-p0)
                            dy_w = [wy,p01]
                            ew = 0

                        # Set normal direction
                        normal = numpy.tan(numpy.arctan(dydx[p])+numpy.pi/2.)
                        ny1 = p01+normal*(p1-p0)
                        ny2 = p01+normal*(p2-p0)
                        dist1 = (p1-wx)**2+(ny1-wy)**2
                        dist2 = (p2-wx)**2+(ny2-wy)**2
                        dx_n = [p0,p2]
                        dy_n = [p01,ny2]
                        en = 1
                        if dist1 < dist2:
                            dx_n = [p1,p0]
                            dy_n = [ny1,p01]
                            en = 0

                        # Set incidence angle
                        e1 = numpy.sqrt((dx_n[0]-dx_n[1])**2+(dy_n[0]-dy_n[1])**2)
                        e2 = numpy.sqrt((dx_w[0]-dx_w[1])**2+(dy_w[0]-dy_w[1])**2)
                        e3 = numpy.sqrt((dx_n[en]-dx_w[ew])**2+(dy_n[en]-dy_w[ew])**2)
                        incidence[p] = math.acos((e3**2 - e2**2 - e1**2)/(-2.0 * e1 * e2))

                        # Set tangent direction
                        tangent = dydx[p]
                        ty1 = p01+tangent*(p1-p0)
                        ty2 = p01+tangent*(p2-p0)
                        dist1 = (p1-wx)**2+(ty1-wy)**2
                        dist2 = (p2-wx)**2+(ty2-wy)**2
                        dx_t = [p0,p2]
                        dy_t = [p01,ty2]
                        et = 1
                        if dist1 > dist2:
                            dx_t = [p1,p0]
                            dy_t = [ty1,p01]
                            et = 0

                        x0 = p0 + self.res
                        y0 = p01
                        e1 = self.res
                        e2 = numpy.sqrt((dx_t[0]-dx_t[1])**2+(dy_t[0]-dy_t[1])**2)
                        e3 = numpy.sqrt((dx_t[et]-x0)**2+(dy_t[et]-y0)**2)
                        longDir[p] = math.acos((e3**2 - e2**2 - e1**2)/(-2.0 * e1 * e2))
                        if dy_t[et] < p01 and longDir[p] < 180.:
                            longDir[p] = 2.*numpy.pi-longDir[p]
                        if dy_t[et] > p01 and longDir[p] > 180.:
                            longDir[p] = 2.*numpy.pi-longDir[p]

                    # Longuet-Higgins (1970) formed an expression for the mean longshore velocity (                                   )
                    # at the breaker zone, of a planar beach, which was modified by Komar (1976) and
                    # took the form of:
                    lV = self.CKomar*wavU*numpy.cos(incidence)*numpy.sin(incidence)

                    if l == 0:
                        globXY = xy[inIDs]
                        globDir = longDir
                        globV = lV
                    else:
                        globXY = numpy.concatenate((globXY,xy[inIDs]), axis=0)
                        globDir = numpy.hstack((globDir,longDir))
                        globV = numpy.hstack((globV,lV))

         # Now map the contour values on the surface
         tree = cKDTree(globXY)
         distances, indices = tree.query(self.xyi, k=8)
         if len(globV[indices].shape) == 3:
            globV_vals = globV[indices][:,:,0]
            globD_vals = globDir[indices][:,:,0]
         else:
            globV_vals = globV[indices]
            globD_vals = globDir[indices]
         longV = numpy.average(globV_vals,weights=(1./distances), axis=1)
         onIDs = numpy.where(distances[:,0] == 0)[0]
         if len(onIDs) > 0:
            longV[onIDs] = globV[indices[onIDs,0]]
         longD = numpy.average(globD_vals,weights=(1./distances), axis=1)
         if len(onIDs) > 0:
            longD[onIDs] = globDir[indices[onIDs,0]]

         slongV = numpy.reshape(longV,(len(self.xv), len(self.yv)),order='F')
         slongD = numpy.reshape(longD,(len(self.xv), len(self.yv)),order='F')
         slongV = gaussian_filter(slongV, sigma=self.sigma)
         rows, cols = numpy.where(numpy.logical_or(z>0.,z<lvl[-1]))
         slongV[rows,cols] = 0.
         slongD[rows,cols] = 0.

         return slongV, slongD
