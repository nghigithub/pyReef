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

from pyReef.libUtils  import simswan as swan
from scipy import interpolate

class hydrodynamic:

    def __init__(self, CKomar, sigma, res, wfac, xv, yv):
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
        self.morphDT = None
        self.CKomar = CKomar
        self.sigma = sigma
        self.gravity = 9.81
        self.porosity = None
        self.dh = None
        self.d50 = None
        self.Urms = None
        self.sl = None
        self.currentU = None
        self.qt = None
        self.bedCx = None
        self.bedCy = None
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

        self.Wfac = wfac
        if self.Wfac > 1:
            nx = int(len(xv)/self.Wfac)
            self.swanX = numpy.linspace(xv.min(), xv.max(), num=nx)
            ny = int(len(yv)/self.Wfac)
            self.swanY = numpy.linspace(yv.min(), yv.max(), num=ny)

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

         # Initialise logshore drift
         self.currentU = []

         # Loop through the different wave climates and store swan output information
         force.wclim = input.climNb[input.wavelist[wID]]

         for clim in range(force.wclim):

            # Define next wave regime
            tw = time.clock()

            for tide in range(self.nbtides):

                # Update elevation if required
                elev = self._interpolate_to_wavegrid(z)

                if force.tideval != 0 and tide == 0:
                    force.Perc.append(input.wavePerc[input.wavelist[wID]][input.climlist[wID]])
                    storm = input.storm[input.wavelist[wID]][input.climlist[wID]]
                    wl = input.wavelist[wID]
                    cl = input.climlist[wID]
                    sl = force.sealevel - force.tideval
                    self.sl.append(sl)
                elif force.tideval != 0 and tide == 1:
                    force.Perc.append(input.wavePerc[input.wavelist[wID]][input.climlist[wID]])
                    storm = input.storm[input.wavelist[wID]][input.climlist[wID]]
                    wID += 1
                    wl = input.wavelist[wID]
                    cl = input.climlist[wID]
                    sl = force.sealevel + force.tideval
                    self.sl.append(sl)
                else:
                    force.Perc.append(input.wavePerc[input.wavelist[wID]][input.climlist[wID]])
                    storm = input.storm[input.wavelist[wID]][input.climlist[wID]]
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

                sU, sD, sH = swan.model.run(self.fcomm, elev, input.waveWh[wl][cl], input.waveWp[wl][cl], wD, sl)

                if self.rank == 0:
                    print '   -   Wave propagation took %0.02f seconds to run.' %(time.clock()-tw)

                tw = time.clock()
                wavU = self._interpolate_to_sedgrid(sU)
                wavD = self._interpolate_to_sedgrid(sD)
                H = self._interpolate_to_sedgrid(sH)
                r,c = numpy.where(z>=0)
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
                    r,c = numpy.where(z-sl>2*lvl)
                    wavD[r,c] += numpy.pi
                    cU = wavU * numpy.cos(wavD)
                    cV = wavU * numpy.sin(wavD)

                # Compute long-shore velocity field
                brk = [lvl/2.,lvl,2*lvl]
                slongV, slongD = self._compute_longshore_velocity(z-sl, wavD, wavU, brk)

                # Set up long-shore current
                lU = slongV * numpy.cos(slongD)
                lV = slongV * numpy.sin(slongD)
                self.currentU.append(slongV)

                # Store percentage of each climate and induced bottom currents velocity
                force.wavU.append(cU+lU)
                force.wavV.append(cV+lV)
                force.wavH.append(H)

                if self.rank == 0:
                    print '   -   Currents model took %0.02f seconds to run.' %(time.clock()-tw)

                # Perform morphological changes
                self._bed_elevation_change(input, force, z, len(self.sl)-1)

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

    def _sediment_transport_components(self, input, force, z, sl, clim):
         """
         This function computes the total load sediment transport in combined waves and currents
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
         self.deprate = 0.
         self.morphDT = 0.

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
         frac = (1.-self.porosity)*self.res
         tmp = 1/(1.-self.porosity)
         # Define total load from Soulsby Van Rijn formulation
         #speed = numpy.sqrt(force.wavU[clim]**2+force.wavV[clim]**2)
         speed = self.currentU[clim]
         qt[row,col] = As[row,col] * speed[row,col] * ( numpy.sqrt(speed[row,col]**2 + \
            0.018*self.Urms[clim][row,col]/CD[row,col]) - Ucr[row,col] )**2.4 * (1.-1.6*smag[row,col])
         qt[numpy.isnan(qt)] = 0.
         self.qt = qt

         # Compute sedimentation rate
         # d(i-1,j) - d(i,j)
         dqtx = numpy.zeros(qt.shape)
         dqtx[1:,:] = qt[:-1,:]-qt[1:,:]
         dqtx[0,:] = dqtx[1,:]
         dhx = numpy.zeros(qt.shape)
         dhx[1:,:] = z[:-1,:]-z[1:,:]
         dhx[0,:] = dhx[1,:]
         # d(i,j) - d(i,j+1)
         dqty = numpy.zeros(qt.shape)
         dqty[:,:-1] = qt[:,:-1]-qt[:,1:]
         dqty[:,-1] = dqty[:,-2]
         dhy = numpy.zeros(qt.shape)
         dhy[:,:-1] = z[:,:-1]-z[:,1:]
         dhy[:,-1] = dhy[:,-2]

         # Compute bed morphological velocity
         bedCx = dqtx/dhx*tmp
         bedCx[numpy.isnan(bedCx)] = 0.
         bedCy = dqty/dhy*tmp
         bedCy[numpy.isnan(bedCy)] = 0.
         self.bedCx = bedCx
         self.bedCy = bedCy
         self.deprate = (dqtx+dqty)/frac
         bedC2 = bedCx**2+bedCy**2
         r,c = numpy.where(bedC2>0)

         # Get CFL condition
         dtm = self.res / numpy.sqrt(2*bedC2[r,c])
         self.morphDT = dtm.min()

         return

    def _bed_elevation_change(self, input, force, z, clim):
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

         variable: clim
            Climatic condition.
         """

         # Initialise morphological changes
         self.dh = []
         tstep = 0.
         totdh = numpy.zeros(z.shape)
         diff = numpy.zeros(z.shape)
         tm = time.clock()
         tend = self.tWave*force.Perc[clim]*self.secyear/self.nbtides

         while(tstep<tend):

             self._sediment_transport_components(input, force, z, self.sl[clim], clim)
             forceDt = max(self.morphDT,self.tSed)

             tmp = -forceDt**2 / (4.*self.res)
             dh = -forceDt * self.deprate
             rCx = self.deprate * self.bedCx
             rCy = self.deprate * self.bedCy

             # rCx(i+1,j) - rCx(i-1,j)
             sum1 = numpy.zeros(rCx.shape)
             sum1[1:-1,:] = rCx[2:,:]-rCx[:-2,:]
             sum1[0,:] = sum1[1,:]
             sum1[-1,:] = sum1[-2,:]

             # rCy(i,j+1) - rCy(i,j-1)
             sum2 = numpy.zeros(rCy.shape)
             sum2[:,1:-1] = rCy[:,2:]-rCy[:,:-2]
             sum2[:,0] = sum2[:,1]
             sum2[:,-1] = sum2[:,-2]

             dh = tmp*(sum1+sum2)
             tstep += forceDt
             # Diffusion transport
             dh[1:-1,1:-1] += forceDt*self.diffusion*( (z[:-2,1:-1]-2*z[1:-1,1:-1]+z[2:,1:-1]) +
                                (z[1:-1,:-2]-2*z[1:-1,1:-1]+z[1:-1,2:]) )/(self.res**2)
             totdh += dh
             z += dh

             # Diffusion transport
             #  diff[1:-1,1:-1] = tend*self.diffusion*( (z[:-2,1:-1]-2*z[1:-1,1:-1]+z[2:,1:-1]) +
             #                     (z[1:-1,:-2]-2*z[1:-1,1:-1]+z[1:-1,2:]) )/(self.res**2)

         if self.rank == 0:
             print '   -   Morphological change took %0.02f seconds to run.' %(time.clock()-tm)

         self.dh.append(totdh)

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
