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

class hydrodynamic:

    def __init__(self, CKomar, sigma, res, xv, yv):
        """
        Constructor.
        """

        self.rank = mpi.COMM_WORLD.rank
        self.size = mpi.COMM_WORLD.size
        self.comm = mpi.COMM_WORLD
        self.fcomm = mpi.COMM_WORLD.py2f()

        self.CKomar = CKomar
        self.sigma = sigma
        self.gravity = 9.81
        self.hydroH = None

        self.xv = xv
        self.yv = yv
        self.xi, self.yi = numpy.meshgrid(xv, yv, sparse=False, indexing='ij')
        xx = numpy.ravel(self.xi,order='F')
        yy = numpy.ravel(self.yi,order='F')
        self.xyi = numpy.vstack((xx, yy)).T

        self.res = res

        return

    def swan_init(self, input, z, wID, sl):
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
         """

         wl = input.wavelist[wID]
         cl = input.climlist[wID]
         swan.model.init(self.fcomm, input.swanFile, input.swanInfo,
                        input.swanBot, input.swanOut, z, input.waveWh[wl][cl],
                        input.waveWp[wl][cl], input.waveWd[wl][cl], self.res,
                        input.waveBase, sl)

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

         force.wavU = []
         force.wavV = []
         #force.longU = []
         #force.longD = []
         force.Perc = []

         # Loop through the different wave climates and store swan output information
         force.wclim = input.climNb[input.wavelist[wID]]

         for clim in range(force.wclim):

            # Define next wave regime
            tw = time.clock()
            wID += 1
            wl = input.wavelist[wID]
            cl = input.climlist[wID]

            # Run SWAN model
            wavU, wavD, H = swan.model.run(self.fcomm, z, input.waveWh[wl][cl],
                                           input.waveWp[wl][cl], input.waveWd[wl][cl],
                                           force.sealevel)
            self.hydroH = H

            # Define cross-shore current
            if input.storm[wl][cl] == 0:
                cU = wavU * numpy.cos(wavD)
                cV = wavU * numpy.sin(wavD)
            else:
                wavD += numpy.pi
                cU = wavU * numpy.cos(wavD)
                cV = wavU * numpy.sin(wavD)

            if self.rank == 0:
                print 'Swan model of waves field %d and climatic conditions %d:' %(wl,cl)
                print 'took %0.02f seconds to run.' %(time.clock()-tw)

            # Compute long-shore velocity field
            tw = time.clock()
            lvl = input.waveBrk[wl][cl]
            brk = [lvl/2.,lvl,2*lvl]
            slongV, slongD = self._compute_longshore_velocity(z-force.sealevel, wavD, wavU, brk)

            # Long-shore current
            lU = slongV * numpy.cos(slongD)
            lV = slongV * numpy.sin(slongD)

            # Store percentage of each climate and induced bottom currents velocity
            force.wavU.append(cU+lU)
            force.wavV.append(cV+lV)
            force.Perc.append(input.wavePerc[wl][cl])

            if self.rank == 0:
                print 'Longshore model for wave field %d and climatic conditions %d:' %(wl,cl)
                print 'took %0.02f seconds to run.' %(time.clock()-tw)

         return wID

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
                    #lV = self.CKomar * numpy.sqrt(self.gravity*wavH)*numpy.cos(incidence)*numpy.sin(incidence)
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
