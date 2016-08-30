##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the pyReef carbonate platform modelling application.     ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
Here we set plotting functions used to visualise pyReef dataset.
"""
import os
import math
import h5py
import cmocean
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import cKDTree
import xml.etree.ElementTree as ETO
from scipy.interpolate import RectBivariateSpline

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

class pyReefPlot():
    """
    Class for plotting outputs from pyReef model.
    """

    def __init__(self, folder=None, ncpus=1, bbox=None, dx=None, az=315, alt=45):
        """
        Initialization function which takes the folder path to Badlands outputs
        and the number of CPUs used to run the simulation. It also takes the
        bounding box and discretization value at which one wants to interpolate
        the data.
        
        Parameters
        ----------
        variable : folder
            Folder path to Badlands outputs.
        
        variable: ncpus
            Number of CPUs used to run the simulation.
        
        variable: bbox (optional)
            Bounding box extent SW corner and NE corner.
        
        variable: dx (optional)
            Discretisation value in metres.
            
        variable : az
            Azimuth angle of the light source.The azimuth is expressed in positive
            degrees from 0 to 360, measured clockwise from north.
            The default is 315 degrees.
            
        variable : alt
            Altitude angle of the light source above the horizon. The altitude is
            expressed in positive degrees, with 0 degrees at the horizon and 90
            degrees directly overhead. The default is 45 degrees.
        """
        
        self.folder = folder
        if not os.path.isdir(folder):
            raise RuntimeError('The given folder cannot be found or the path is incomplete.')
        
        self.ncpus = ncpus
        self.dx = dx
        self.bbox = bbox
        
        self.x = None 
        self.y = None 
        self.z = None 
        self.h = None 
        self.U = None 
        self.V = None 
        self.sl = None 
        self.nx = None
        self.ny = None
        self.bc = None
        self.sx = None
        self.sy = None
        self.smag = None
        self.hillshade = None
        self.az = az
        self.alt = alt
        self.nbclim = 0
        #xi = np.linspace(0, len(self.x), num=len(self.x)) 
        #yi = np.linspace(0, len(self.y), num=len(self.y)) 
        #xm, ym = np.meshgrid(xi, yi, sparse=False, indexing='ij')
        #self.xi = xi.T
        #self.yi = yi.T
        
        return

    def loadHDF5(self, timestep=0, sl=0):
        """
        Read the HDF5 file for a given time step.
        
        Parameters
        ----------
        variable : timestep
            Time step to load.
            
        variable : Sea-level
            Sea-level elevation.
        """
        
        self.sl = sl
        for i in range(0, self.ncpus):
            dfc = h5py.File('%s/h5/xycoords.p%s.hdf5'%(self.folder, i), 'r')
            x5 = np.array((dfc['/x']))
            y5 = np.array((dfc['/y']))
            if i == 0:
                x = x5
                y = y5
            else:
                x = np.append(x, x5)
                y = np.append(y, y5)
            
            df = h5py.File('%s/h5/surf.time%s.p%s.hdf5'%(self.folder, timestep, i), 'r')
            z5 = np.array((df['/z']))
            self.nbclim = int((len(df.keys())-1)/3)
            if i == 0:
                z = z5
            else:
                z = np.append(z, z5)
                
        h = np.zeros((self.nbclim,len(z)))
        u = np.zeros((self.nbclim,len(z)))
        v = np.zeros((self.nbclim,len(z)))
        for c in range(self.nbclim):
            for i in range(0, self.ncpus):
                df = h5py.File('%s/h5/surf.time%s.p%s.hdf5'%(self.folder, timestep, i), 'r')
                st = '/wh'+str(c)
                h5 = np.array((df[st]))
                st = '/wu'+str(c)
                u5 = np.array((df[st]))
                st = '/wv'+str(c)
                v5 = np.array((df[st]))
                if i == 0:
                    hh = h5
                    uu = u5
                    vv = v5
                else:
                    uu = np.append(uu, u5)
                    vv = np.append(vv, v5)
                    hh = np.append(hh, h5)
            if self.ncpus == 1:
                u[c,:] = uu[:,0]
                v[c,:] = vv[:,0]
                h[c,:] = hh[:,0]
            else:
                u[c,:] = uu
                v[c,:] = vv
                h[c,:] = hh

        if self.dx == None:
            self.dx = x[1]-x[0]
            if self.dx==0:
                raise RuntimeError('Mesh spatial discretisation cannot be defined.')
        
        if self.bbox == None:
            self.nx = int((x.max() - x.min())/self.dx+1)
            self.ny = int((y.max() - y.min())/self.dx+1)
            self.x = np.linspace(x.min(), x.max(), self.nx)
            self.y = np.linspace(y.min(), y.max(), self.ny)
            self.bbox = np.zeros(4,dtype=float)
            self.bbox[0] = x.min()
            self.bbox[1] = y.min()
            self.bbox[2] = x.max()
            self.bbox[3] = y.max()
        else:
            if self.bbox[0] < x.min():
                self.bbox[0] = x.min()
            if self.bbox[2] > x.max():
                self.bbox[2] = x.max()
            if self.bbox[1] < y.min():
                self.bbox[1] = y.min()
            if self.bbox[3] > y.max():
                self.bbox[3] = y.max()
                    
            self.nx = int((self.bbox[2] - self.bbox[0])/self.dx+1)
            self.ny = int((self.bbox[3] - self.bbox[1])/self.dx+1)
            self.x = np.linspace(self.bbox[0], self.bbox[2], self.nx)
            self.y = np.linspace
                
        xi = np.linspace(0, len(self.x), num=len(self.x)) 
        yi = np.linspace(0, len(self.y), num=len(self.y)) 
        xm, ym = np.meshgrid(xi, yi, sparse=False, indexing='ij')
        self.xi = xi.T
        self.yi = yi.T
        
        xx, yy = np.meshgrid(self.x, self.y, sparse=False, indexing='ij')
        xyi = np.dstack([xx.flatten(), yy.flatten()])[0]
        XY = np.column_stack((x,y))
        tree = cKDTree(XY)
        distances, indices = tree.query(xyi, k=3)
        if self.ncpus == 1:
            z_vals = z[indices][:,:,0]
        else:
            z_vals = z[indices]
            
        zi = np.zeros(len(xyi))
        onID1 = np.where(distances[:,0] == 0)[0]
        if len(onID1) > 0:
            zi[onID1] = z[indices[onID1,0]]
        onID2 = np.where(distances[:,0] > 0)[0]
        if len(onID2) > 0:
            zi[onID2] = np.average(z_vals[onID2],weights=(1./distances[onID2]), axis=1)
        self.z = np.reshape(zi,(self.nx,self.ny))
        
        self.h = np.zeros((self.nbclim,self.nx,self.ny))
        self.u = np.zeros((self.nbclim,self.nx,self.ny))
        self.v = np.zeros((self.nbclim,self.nx,self.ny))
        for c in range(self.nbclim):
            h_vals = h[c,indices]
            u_vals = u[c,indices]
            v_vals = v[c,indices]
            hi = np.zeros(len(xyi))
            ui = np.zeros(len(xyi))
            vi = np.zeros(len(xyi))
            if len(onID1) > 0:
                hi[onID1] = h[c,indices[onID1,0]]
                ui[onID1] = u[c,indices[onID1,0]]
                vi[onID1] = v[c,indices[onID1,0]]
            if len(onID2) > 0:
                hi[onID2] = np.average(h_vals[onID1],weights=(1./distances[onID1]), axis=1)
                ui[onID2] = np.average(u_vals[onID1],weights=(1./distances[onID1]), axis=1)
                vi[onID2] = np.average(v_vals[onID1],weights=(1./distances[onID1]), axis=1)
            self.h[c,:,:] = np.reshape(hi,(self.nx,self.ny))
            self.u[c,:,:] = np.reshape(ui,(self.nx,self.ny))
            self.v[c,:,:] = np.reshape(vi,(self.nx,self.ny))
       
        # Data range
        self.extent = [np.amin(self.x), np.amax(self.x), np.amax(self.y), np.amin(self.y)]

        # Get hillshade and slope
        self._calcHillshade(az=self.az, alt=self.alt)
        
        return    

    def _assignBCs(self):
        """
        Pads the boundaries of a grid. Boundary condition pads the boundaries
        with equivalent values to the data margins, e.g. x[-1,1] = x[1,1].
        It creates a grid 2 rows and 2 columns larger than the input.
        """
 
        self.bc = np.zeros((self.nx + 2, self.ny + 2)) 
        self.bc[1:-1,1:-1] = self.z  

        # Assign boundary conditions - sides
        self.bc[0, 1:-1] = self.z[0, :]
        self.bc[-1, 1:-1] = self.z[-1, :]
        self.bc[1:-1, 0] = self.z[:, 0]
        self.bc[1:-1, -1] = self.z[:,-1]

        # Assign boundary conditions - corners
        self.bc[0, 0] = self.z[0, 0]
        self.bc[0, -1] = self.z[0, -1]
        self.bc[-1, 0] = self.z[-1, 0]
        self.bc[-1, -1] = self.z[-1, 0]

        return
    
    def _calcFiniteSlopes(self):
        """
        Calculate slope with 2nd order/centered difference method.
        """

        # Assign boundary conditions
        self._assignBCs()

        # Compute finite differences
        self.sx = (self.bc[1:-1, 2:] - self.bc[1:-1, :-2])/(2*self.dx)
        self.sy = (self.bc[2:,1:-1] - self.bc[:-2, 1:-1])/(2*self.dx)
        
        return

    def _calcHillshade(self, az=315, alt=45):
        """
        Creates a shaded relief from a surface raster by considering the
        illumination source angle and shadows.

        Parameters
        ----------
        variable : az
            Azimuth angle of the light source.The azimuth is expressed in positive
            degrees from 0 to 360, measured clockwise from north.
            The default is 315 degrees.
            
        variable : alt
            Altitude angle of the light source above the horizon. The altitude is
            expressed in positive degrees, with 0 degrees at the horizon and 90
            degrees directly overhead. The default is 45 degrees.
        """

        # Convert angular measurements to radians
        azRad, elevRad = (360 - az + 90)*np.pi/180, (90-alt)*np.pi/180  
        
        # Calculate slope in X and Y directions
        self._calcFiniteSlopes()  
        self.smag = np.sqrt(self.sx**2 + self.sy**2)
        
        # Angle of aspect
        aspectRad = np.arctan2(self.sy, self.sx) 
        
        # Magnitude of slope in radians
        smagRad = np.arctan(np.sqrt(self.sx**2 + self.sy**2))  

        self.hillshade = 255.0 * ((np.cos(elevRad) * np.cos(smagRad)) + 
                (np.sin(elevRad)* np.sin(smagRad) * np.cos(azRad - aspectRad)))

        return

    def bathymetry(self, color=cmocean.cm.delta, fsize=(7.5,5), fname=None, dpi=300):
        """
        Creates a shaded bathymetry from pyReef surface raster by considering the
        illumination source angle and shadows.

        Parameters
        ----------
        variable : color
            color map from cmocean.
            
        variable : size
            Plot size.
            
        variable : fname
            Save PNG filename.
            
        variable : dpi
            Figure resolution.
        """
        
        fig = plt.figure(figsize = fsize)
        ax = plt.subplot(1, 1, 1)
        
        # Elevation plot
        im = ax.imshow(self.z.T-self.sl, origin='lower', cmap=color, vmin=-40, vmax=40, aspect=1) #, extent=self.extent)
        
        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        # hillshade
        ax.imshow(self.hillshade.T, interpolation = 'bilinear', origin='lower', cmap='gray', alpha = 0.25, aspect=1)
        
        # Shoreline contour
        levels = [0]
        CS = ax.contour(self.z.T-self.sl, levels, colors='k', origin='lower', linewidths=1)
    
        # Bathymetric contours
        levels = [-60.,-50.,-40.,-30.,-20.,-10.]
        CS = ax.contour(self.z.T-self.sl, levels, colors='w', origin='lower', linewidths=0.5)  
        plt.clabel(CS, inline=1, fontsize=5)
        
        # Set title
        ax.set_title('Bathymetry')
        
        plt.show()
        
        if fname is not None:
            fig.savefig(fname, dpi=dpi)

        return

    def slope(self, color=cmocean.cm.matter, fsize=(7.5,5), fname=None, dpi=300):
        """
        Creates a shaded slope map from pyReef surface raster by considering the
        illumination source angle and shadows.

        Parameters
        ----------
        variable : color
            color map from cmocean.
            
        variable : size
            Plot size.
            
        variable : fname
            Save PNG filename.
            
        variable : dpi
            Figure resolution.
        """
        
        fig = plt.figure(figsize = fsize)
        ax = plt.subplot(1, 1, 1)
        
        # Slope map
        im = ax.imshow(self.smag.T, interpolation = 'bilinear', cmap=color, 
                       vmin=0, vmax=0.2, aspect=1, origin='lower')
        
        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        # Add hillshade surface
        ax.imshow(self.hillshade.T, interpolation = 'bilinear', origin='lower', cmap='gray', alpha = 0.25, aspect=1)
        
        # Shoreline contour
        levels = [0]
        CS = ax.contour(self.z.T-self.sl, levels, colors='k', origin='lower', linewidths=1)
        
        # Bathymetric contours
        levels = [-60.,-50.,-40.,-30.,-20.,-10.]
        CS = ax.contour(self.z.T-self.sl, levels, colors='k', origin='lower', linewidths=0.5)  
        plt.clabel(CS, inline=1, fontsize=5)
        
        # Set title
        ax.set_title('Slope magnitude')

        plt.show()
        
        if fname is not None:
            fig.savefig(fname, dpi=dpi)
            
        return
    
    def waveHeight(self, climate, lvls, color=cmocean.cm.thermal, fsize=(7.5,5), fname=None, dpi=300):
        """
        Visualise wave height distribution based on SWAN wave model.

        Parameters
        ----------
        variable : climate
            climate number to plot for the given period.
            
        variable : lvls
            wave height contour levels.
            
        variable : color
            Color map from cmocean.
            
        variable : size
            Plot size.
            
        variable : fname
            Save PNG filename.
            
        variable : dpi
            Figure resolution.
        """
        
        height = self.h[climate,:,:]
        fig = plt.figure(figsize = fsize)
        ax = plt.subplot(1, 1, 1)
        
        # Slope map
        im = ax.imshow(height.T, interpolation = 'bilinear', cmap=color, aspect=1, origin='lower')
        
        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        # Add hillshade surface
        ax.imshow(self.hillshade.T, interpolation = 'bilinear', origin='lower', cmap='gray', alpha = 0.25, aspect=1)
        
        # Shoreline contour
        levels = [0]
        CS = ax.contour(self.z.T-self.sl, levels, colors='w', origin='lower', linewidths=1)
        
        # Wave height contours
        CS = ax.contour(height.T, lvls, colors='w', origin='lower', linewidths=0.5)  
        plt.clabel(CS, inline=1, fontsize=5)
        
        # Set title
        ax.set_title('Wave height distribution')

        plt.show()
        
        if fname is not None:
            fig.savefig(fname, dpi=dpi)
            
        return  
    
    def bottomCurrents(self, climate, gauss=1, dens=0.5, color=cmocean.cm.speed, fsize=(7.5,5), fname=None, dpi=300):
        """
        Visualise wave induced bottom current streamlines based on SWAN wave model.

        Parameters
        ----------
        variable : climate
            climate number to plot for the given period.
            
        variable : gauss
            Gaussian filter for smoothing bottom current value.
            
        variable : dens
            Streamline density.
            
        variable : Color
            color map from cmocean.
            
        variable : size
            Plot size.
            
        variable : fname
            Save PNG filename.
            
        variable : dpi
            Figure resolution.
        """
        
        fig = plt.figure(figsize = fsize)
        ax = plt.subplot(1, 1, 1)
        
        U = self.u[climate,:,:].T
        V = self.v[climate,:,:].T
        
        speed = np.sqrt(U**2+V**2)
        
        # Current map
        if gauss > 0:
            svel = gaussian_filter(speed, gauss)
            im = ax.imshow(svel, interpolation = 'bilinear', cmap=color, aspect=1, origin='lower')
        else:
            im = ax.imshow(speed, interpolation = 'bilinear', cmap=color, aspect=1, origin='lower')
            
        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        # Add hillshade surface
        ax.imshow(self.hillshade.T, interpolation = 'bilinear', origin='lower', cmap='gray', alpha = 0.5, aspect=1)
        
        # Shoreline contour
        levels = [0]
        CS = ax.contour(self.z.T-self.sl, levels, colors='k', origin='lower', linewidths=1.2)
        
        # Streamlines
        lw = 2.*speed / speed.max()
        strm = ax.streamplot(self.xi, self.yi, U, V, color='k', density=dens, linewidth=lw, arrowsize=0.5) 
        
        # Set title
        ax.set_title('Bottom velocity distribution')

        plt.show()
        
        if fname is not None:
            fig.savefig(fname, dpi=dpi)
            
        return