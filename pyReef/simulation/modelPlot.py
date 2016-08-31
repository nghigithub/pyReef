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

import cmocean
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

from scipy.ndimage.filters import gaussian_filter

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

class modelPlot():
    """
    Class for plotting outputs from pyReef model.
    """

    def __init__(self, x=None, y=None):
        """
        Initialization function which takes the grid coordinates.

        Parameters
        ----------
        variable : x, y
            pyReef grid coordinates.
        """

        self.x = x
        self.y = y
        self.z = None
        self.sl = 0
        self.dx = self.x[1] - self.x[0]
        self.bc = None
        self.sx = None
        self.sy = None
        self.smag = None
        self.ny = None
        self.nx = None
        self.hillshade = None
        xi = np.linspace(0, len(self.x), num=len(self.x))
        yi = np.linspace(0, len(self.y), num=len(self.y))
        xm, ym = np.meshgrid(xi, yi, sparse=False, indexing='ij')
        self.xi = xi.T
        self.yi = yi.T

        # Data range
        self.extent = [np.amin(self.x), np.amax(self.x), np.amax(self.y), np.amin(self.y)]

        return

    def paramInit(self, z=None, sl=0, az=315, alt=45):
        """
        Initialization elevation and sea-level.

        Parameters
        ----------
        variable : z
            pyReef grid elevation.

        variable : sl
            sea-level.

        variable : az
            Azimuth angle of the light source.The azimuth is expressed in positive
            degrees from 0 to 360, measured clockwise from north.
            The default is 315 degrees.

        variable : alt
            Altitude angle of the light source above the horizon. The altitude is
            expressed in positive degrees, with 0 degrees at the horizon and 90
            degrees directly overhead. The default is 45 degrees.
        """

        self.z = z
        self.sl = sl
        self.ny, self.nx = self.z.shape

        # Get hillshade and slope
        self._calcHillshade(az=az, alt=alt)

        return

    def _assignBCs(self):
        """
        Pads the boundaries of a grid. Boundary condition pads the boundaries
        with equivalent values to the data margins, e.g. x[-1,1] = x[1,1].
        It creates a grid 2 rows and 2 columns larger than the input.
        """

        self.bc = np.zeros((self.ny + 2, self.nx + 2))
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

    def deposition(self, deprate, min=min, max=max, gauss=1, color=cmocean.cm.balance, fsize=(7.5,5), fname=None, dpi=300):
        """
        Creates a shaded deposition rate map from pyReef surface raster by considering the
        illumination source angle and shadows.

        Parameters
        ----------
        variable : deprate
            Deposition rate.

        variable : min, max
            color bar extent.

        variable : gauss
            Gaussian filter for smoothing bottom current value.

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

        # Current map
        sdeprate = deprate
        if gauss > 0:
            sdeprate = gaussian_filter(deprate, gauss)

        # Deposition rate map
        im = ax.imshow(sdeprate.T, interpolation = 'bilinear', cmap=color,
                       vmin=min, vmax=max, aspect=1, origin='lower')

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
        ax.set_title('Deposition rate')

        plt.show()

        if fname is not None:
            fig.savefig(fname, dpi=dpi)

        return

    def morphochange(self, dh, min=min, max=max, gauss=1, color=cmocean.cm.balance, fsize=(7.5,5), fname=None, dpi=300):
        """
        Creates a shaded morphological change map from pyReef surface raster by considering the
        illumination source angle and shadows.

        Parameters
        ----------
        variable : deprate
            Deposition rate.

        variable : min, max
            color bar extent.

        variable : gauss
            Gaussian filter for smoothing bottom current value.

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

        # Current map
        sdeprate = dh
        if gauss > 0:
            sdeprate = gaussian_filter(dh, gauss)

        # Deposition rate map
        im = ax.imshow(sdeprate.T, interpolation = 'bilinear', cmap=color,
                       vmin=min, vmax=max, aspect=1, origin='lower')

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
        ax.set_title('Morphological change')

        plt.show()

        if fname is not None:
            fig.savefig(fname, dpi=dpi)

        return

    def waveHeight(self, height, lvls, color=cmocean.cm.thermal, fsize=(7.5,5), fname=None, dpi=300):
        """
        Visualise wave height distribution based on SWAN wave model.

        Parameters
        ----------
        variable : color
            Color map from cmocean.

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

    def bottomCurrents(self, U, V, gauss=1, dens=0.5, color=cmocean.cm.speed, fsize=(7.5,5), fname=None, dpi=300):
        """
        Visualise wave induced bottom current streamlines based on SWAN wave model.

        Parameters
        ----------
        variable : U, V
            Component of bottom velocity along A and Y axis.

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
