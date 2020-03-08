# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        optimize.py
# Purpose:     Functionality relating to optimization of optical systems using
#              Zemax through the pyzdde interface.
# Licence:     MIT License
#              This file is subject to the terms and conditions of the MIT License.
#              For further details, please refer to LICENSE.txt
# Revision:    
#-------------------------------------------------------------------------------

import numpy as np
import pyzdde.zdde as pyz
import pyzdde.zcodes.zemaxoperands as zoper
import numpy.polynomial.legendre as leg



def leg_gauss_quad(field_data, rings=3, arms=6, obs=0.0, symm=False):
    """Calculate normalised pupil coordinates for Legendre-Gauss quadrature
       rayset selection.

    Parameters
    ----------
    field_data : list of fieldData named tuples
        The field coordinates to use for generation of the coordinates. Must
        be a list as returned from Zemax by zGetFieldTuple().  
    rings : int
        The number of rings of rays for which to compute radial pupil coordinates.
        Minimum of 3, which is the default.
    arms : int
        The number of radial arms along which to calculate the pupil coordinates.
        Default is 6. Minimum is 6. Must be even.
    obs : float
        The fractional radius of the pupil which is obscured. Defaults to 0.0
    symm : boolean
        Will reduce the number of returned points if the provided field points
        are found to lie only along the x axis or only along the y axis.
        Default is False, symmetry not to be assumed.

    Returns
    -------
    hx : array of float
        Relative field x coordinates of the ray list.
    hy : array of float
        Relative field y coordinates of the ray list.
    px : array of float
        Relative pupil x coordinates of the ray list.
    py : array of float
        Relative pupil y coodinates of the ray list.
    weights : array of float
        Legendre-Gauss weighting factors for each ray.

    Reference:
    Brian J. Baumana and Hong Xiaob,
    Gaussian quadrature for optical design with non-circular pupils and fields, 
    and broad wavelength ranges, https://doi.org/10.1117/12.872773

    Note: This function only handles circular, obscured pupils.

    """
    # Check that arms is even
    if arms < 6 or (arms//2 != arms/2.0):
        raise ValueError('Input parameter arms must be even and 6 or more.')
    if rings < 3:
        raise ValueError('Input parameter rings must be 3 or more.')
    hx = np.array([])
    hy = np.array([])
    px = np.array([])
    py = np.array([])
    weights = np.array([])  # LGQ weights
    order_tuple = tuple([0]*rings + [1])
    order_tuple1 = tuple([0]*(rings+1) + [1])
    # Get the Legendre polynomial of order equal to the number of rings
    leg_poly = leg.Legendre(order_tuple)
    # And the one after that for calculating the weights below
    leg_poly1 = leg.Legendre(order_tuple1)
    # Get the roots of the polynomial
    leg_roots = leg_poly.roots()
    # The relative radii of the rings are computed as follows (see Bauman Eq. 6)
    ring_radii = np.sqrt(obs**2.0 + (1.0+leg_roots)*(1.0-obs**2.0)/2.0)
    # Get the weights, see e.g. https://mathworld.wolfram.com/Legendre-GaussQuadrature.html
    leg_weights = 2.0 * (1.0-leg_roots**2.0)/((rings+1.0)**2.0 * (leg_poly1(leg_roots)**2.0))
    if obs > 0.0:
        pupil_weights = (1.0-obs) * leg_weights / 2.0
    else:
        pupil_weights = leg_weights / 2.0
    # Run through the field points
    theta_inc = 360.0/arms  # degrees. Angular increment between arms
    for field_point in field_data:
        # Get the coordinates and weighting factor for this field point
        xf, yf, wgt = field_point.xf, field_point.yf, field_point.wgt 
        # Generate the relative pupil coordinates for this field position
        # Takes symmetry flag into account
        if xf==0.0:  
            if symm: # Generate polar angles that are symmetric only across the y-axis
                if yf==0.0: # On axis
                    theta = 0.0  # A single arm is enough if the system is symmetric
                else:
                    theta = np.linspace(-90.0+theta_inc/2.0, 90.0-theta_inc/2.0, arms//2)
            else:
                theta = np.linspace(0.0, 360.0-theta_inc, arms)
        elif yf==0.0:  
            if symm: # Generate polar angles that are symmetric only cross the x-axis
                theta = np.linspace(theta_inc/2.0, 180.0-theta_inc/2.0, arms//2)
            else:
                theta = np.linspace(-180.0+theta_inc/2.0, 180.0-theta_inc/2.0, arms)
        theta_rad = np.deg2rad(theta)
        # Meshgrid the radii and the angles, variables fp_ are for this field point
        fp_ring_radii, fp_theta_rad = np.meshgrid(ring_radii, theta_rad)
        # pup_weights are meshgridded to make an array corresponding to the radius
        fp_pupil_weights, fp_theta_rad = np.meshgrid(pupil_weights, theta_rad)
        fp_px, fp_py = fp_ring_radii * np.cos(fp_theta_rad), fp_ring_radii * np.sin(fp_theta_rad)
        # Flatten arrays, should now all be the same length
        fp_px, fp_py, fp_pupil_weights = fp_px.flatten(), fp_py.flatten(), fp_pupil_weights.flatten()
        fp_len = np.ones(fp_px.size)
        # Replicate up the field points and field weights to the correct length
        fp_hx, fp_hy, fp_wgt = xf*fp_len, yf*fp_len, wgt*fp_pupil_weights
        # The field weigts
        hx, hy, weights = np.hstack((hx, fp_hx)), np.hstack((hy, fp_hy)), np.hstack((weights, fp_wgt))
        px, py = np.hstack((px, fp_px)), np.hstack((py, fp_py))
    # Get the maximum field radial position to perform field normalisation
    max_field_radius = np.sqrt(hx**2.0 + hy**2.0).max()
    px /= max_field_radius
    py /= max_field_radius
    return hx, hy, px, py, weights


def mf_gen_reshidko(pzln, wv_groups, n_rings, n_arms, refresh=True, pushback=True):
    """
    Generate a merit function for chromatic correction based on the second method described
    by Reshidko in
    RESHIDKO, Dmitry, MASATSUGU NAKANATO and JOSÉ SASIÁN, 
    "Ray Tracing Methods for Correcting Chromatic Aberrations in Imaging Systems",
    International Journal of Optics. 2014. https://dx.doi.org/10.1155/2014/351584

    This function will search for the DMFS operand, which marks the start of the
    default merit function, and replace all operands occurring after DMFS with
    the Reshidko 2 chromatic merit function.

    The intention with the Reshidko method is to alternate between a complete merit
    function (such as the ones produced by the Zemax MF wizard) and a chromatic
    aberration MF generated using this function. The Reshidko chromatic MF is used
    together with glass subsitution (hammer or global) optimisation in Zemax.

    Selection of glass libraries for glass subsitution is potentially very important,
    since there are typically a very large number of potential glass combinations,
    the vast majority of which will be far from optimal.

    See the ZemaxGlass python package for 

    Paramters
    ---------
    pzln : handle to the Zemax DDE server, created using pyzdde.zdde.createLink()
    wv_groups : list of lists of int
        The wavelength groups to use when setting up the merit function. The integers
        given in each group must be the Zemax wavelength numbers. The first integer in
        each group must be the reference (typically near center) wavelength number.
        The remaining wavelength numbers in the group will generate ray targets in
        the merit function
    """
    # Get a refresh of the lens into the DDE server
    if refresh:
        pzln.zGetRefresh()
    # Get the system basics
    opsys = pzln.zGetSystem()
    imsurf = opsys[0]
    i_dmfs = pzln.zDeleteDefaultMFO()  # Get rid of the current default merit function and record where it started
    # Generate the ray targets, using the first wavelength in each group as the reference

    # Get the reference ray coordinates at the image (assumed the last surface)
    for wv_group in wv_groups:  # Run through all the given wavelength groups
        for i_wv in wv_group:  # 
            pass

class MeritFunction(object):
    def __init__(self):
        pass
    