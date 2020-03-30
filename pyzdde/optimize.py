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
import pandas as pd



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
    Brian J. Bauman and Hong Xiao,
    Gaussian quadrature for optical design with non-circular pupils and fields, 
    and broad wavelength ranges, https://doi.org/10.1117/12.872773

    Notes
    ----- 
    This function only handles circular pupils with centered circular obscurations.

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
    hx /= max_field_radius
    hy /= max_field_radius
    return hx, hy, px, py, weights

def zInsertOperand(zln, i_oper, opertype='BLNK', int1=None, int2=None, data1=None, data2=None,
                     data3=None, data4=None, data5=None, data6=None, tgt=None, wgt=None):
    """Helper function for mf_gen_reshidko().
        Inserts an operand at row i_oper and returns incremented i_oper.

        See zdde.zSetOperandRow
    """
    zln.zInsertMFO(i_oper)  # Insert an operand at the requested row position
    zln.zSetOperandRow(row=i_oper, opertype=opertype, int1=int1, int2=int2, data1=data1, data2=data2,
                     data3=data3, data4=data4, data5=data5, data6=data6, tgt=tgt, wgt=wgt)
    return i_oper+1

def mf_gen_reshidko(pzln, n_rings=3, n_arms=6, obs=0.0, symm=True, operands=['TRAX', 'TRAY'], 
                    op_surfs=None, wv_groups=None, refresh=True, pushback=True, keep_dmfs=False):
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

    Selection of glass libraries for glass subsitution is very important,
    since there are typically a very large number of potential glass combinations,
    the vast majority of which will be far from optimal.

    See the ZemaxGlass python package at https://github.com/derekjgriffith/zemaxglass
    for methods of selecting glasses from multiple catalogs and writing a customised
    Zemax glass catalog (.agf file).

    Parameters
    ---------
    pzln : handle to the Zemax DDE server, created using pyzdde.zdde.createLink()
    n_rings : int
        Number of rings of rays to generate in the normalized pupil coordinate system.
        Rays are generated using Legendre-Gauss Qaudrature. Default is 3. Minimum is 3.
    n_arms : int
        Number of radial arms of rays to generate in the normalized pupil coordinate system.
        Default is 6. Minimum is 6. Ray coordinates are generated using LGQ method.
    obs : float
        Obscuration ratio of the lens system to be optimised. Default is 0.0 (no obscuration).
    symm : logical
        Set True if the system to be optimised is rotationally symmetrical. This reduces the
        number of rays to be traced and therefore also the optimization convergence time.
        Default is True.
    operands : list of str
        List of Zemax operands to use when setting up the chromatic differential merit function.
        The default is to use transverse ray aberrations with both the TRAX and TRAY operands.
        TRAX and TRAY compute transverse ray aberrations in the x and y directions, using the
        chief ray at the specified wavelength as the reference. Since the Reshidko method 2
        uses differences between operands at all wavelengths with respect to the primary
        wavelength, this is equivalent to using the REAX and REAY operands, which will
        not trace the chief ray and may therefore be more efficient.
    op_surfs : mixed list of int and str
        Surfaces at which to apply the operand. Generally, the aberration operands, such
        as TRAX and TRAY cannot be computed at any surface and hence a surface other than
        the image surface cannot be specified. However, the real ray operands such as
        REAX and REAY can be specified at any surface, in which case a corresponding surface
        number can be provided in the op_surfs list. 
        Alternatively, since surface numbers can easily change with insertions and deletions,
        a surface comment can be provided as a string. The relevant surface will then be
        sought based on the comment column in the LDE.
    wv_groups : list of list of int
        A list of wavelength groups. Each element in wv_groups is a list of Zemax
        wavelength indices. The first wavelength index in each group is the reference
        wavelength. MF operands are set up to reduce chromatic aberration differences at
        each of the other wavelengths in the group relative to the reference wavelength.
        Default is None, in which case the Zemax primary wavelength is the reference
        wavelength, forming a single group with all the other wavelengths. This feature
        is intended for optimization of systems that must work well in several
        wavelength regions, typically for different detectors.
    refresh : logical
        If set True, will get a lens DDE server refresh using ``pzln.zGetRefresh()``.
        Default is True.
    pushback : logical
        If set True, will push the lens back to the Zemax user foreground using ``pzln.zPushLens()``.
        Default is True.
    keep_dmfs : logical
        If set True, any default merit function operands, delineated by the first occurrance of
        the DMFS marker operand will be kept. If set False, all operands beyond the DMFS
        marker will be deleted i.e. the default merit function is replaced by the merit
        function defined by this method. The default is False i.e. the default merit function
        is deleted.

    Returns
    -------
    If pushback is True, returns the return code of zPushLens, otherwise None.

    Notes
    -----
    This function expects a design that is monochromatically corrected. At the very least,
    the chief ray height at the primary wavelength must be at the correct position at the
    image plane. This is used as the reference and target for all other rays at the specific
    field positions and wavelengths.

    For transverse ray aberration chromatic correction, this method sets up a series of
    operands specified by the user (operands list). 

    Each operand is computed for the entire LGQ rayset at the primary wavelength (the reference for
    each is the chief ray at the primary wavelength, which is not included in the LGQ set).
    Then the operand is set up for each ray in the LGQ rayset for all the other wavelengths.
    MF weights for all operands are set to zero. Then a series of DIFF operands
    are set up which compute the difference in the operands for each wavelength relative to
    the primary wavelength. The weights for the DIFF operands are set as the product
    of the LGQ weight and the wavelength weight. 
    """
    # Get a refresh of the lens into the DDE server
    if refresh:
        pzln.zGetRefresh()
    # Get the system basics
    sys_data = pzln.zGetSystem()
    imsurf = sys_data.numSurf
    # Deal with the operand surfaces
    if op_surfs is None:
        op_surfs = [imsurf] * len(operands)  # Default all to the image surface
    else:  # Find surfaces based on comment column in LDE
        updated_op_surfs = list(op_surfs)  # Create a copy to work on in the loop
        # Get all the surface comments
        surf_comments = [pzln.zGetSurfaceData(i_surf, 1) for i_surf in range(1, imsurf+1)]
        for i_op_surf, op_surf in enumerate(op_surfs):
            if op_surf is None:
                updated_op_surfs[i_op_surf] = imsurf  # Default None inside the list to image surface as well
                continue
            # Replace strings with surface number of matching comment, if any
            try:
                index_comment = surf_comments.index(op_surf)
                updated_op_surfs[i_op_surf] = index_comment
            except ValueError:
                pass
        op_surfs = updated_op_surfs
    # Get the wavelength data
    i_primary = pzln.zGetWave(0)[0]  # Get primary wavelength number
    n_waves = pzln.zGetWave(0)[1]  # Get the number of wavelengths
    wave_data = pzln.zGetWaveTuple()
    if wv_groups == None:  # Set up a single group
        wv_group = [i_primary]
        wv_group.extend(filter(lambda iwv: iwv!=i_primary, range(1,n_waves+1)))
        wv_groups = [wv_group]
    config_data = pzln.zGetConfig()  # Returns current config, number of configs and number of rows.
    n_config = config_data[1]
    if keep_dmfs:  # Keep the default merit function, but insert Reshidko above it
        i_dmfs = pzln.zFindDefaultMFO()
    else:
        i_dmfs = pzln.zDeleteDefaultMFO()  # Get rid of the current default merit function and record where it started
    # Generate the ray pupil coordinates 
    # First get the field data
    field_data = pzln.zGetFieldTuple()
    # Then generate the LGQ rayset
    hx, hy, px, py, lgq_weights = leg_gauss_quad(field_data, rings=n_rings, arms=n_arms, 
                                            obs=obs, symm=symm)
    # Get the reference ray coordinates at the image (assumed the last surface)
    # Run through each of the configurations and generate the same ray targets in
    # each.
    i_oper = i_dmfs + 1  # This keeps track of where we are inserting into the merit function
    for i_config in range(1, n_config+1):
        for (oper_i, operand) in enumerate(operands):
            if n_config > 1:  # Insert the CONF operand, otherwise don't bother
                i_oper = zInsertOperand(pzln, i_oper, opertype='CONF', int1=i_config)
                # Add a commented BLNK operand
                #i_oper = zInsertOperand(pzln,row=i_oper, opertype='BLNK',
                #            int1=f'Reshidko Targets for Config {i_config+1}')
            # Now run through all the rays in the rayset
            # There is a target for each ray at each wavelength
            for i_ray in range(len(hx)):
                # Run through the wavelength groups, setting up the operand for the reference wavelength,
                # followed by operands for each of the other wavelengths in the group
                for wv_group in wv_groups:
                    # Set up the operand at the reference wavelength
                    i_oper_ref = i_oper  # This will be the reference operand
                    i_oper = zInsertOperand(pzln, i_oper, opertype=operand, int1=op_surfs[oper_i], int2=wv_group[0],
                        data1=hx[i_ray], data2=hy[i_ray], data3=px[i_ray], data4=py[i_ray], tgt=0.0, wgt=0.0)
                    # Set up operand for all the other wavelengths in the group
                    for i_wave in wv_group[1:]:
                        i_oper_dif = i_oper
                        i_oper = zInsertOperand(pzln, i_oper, opertype=operand, int1=op_surfs[oper_i], int2=i_wave,
                            data1=hx[i_ray], data2=hy[i_ray], data3=px[i_ray], data4=py[i_ray], tgt=0.0, wgt=0.0)
                        # Calculate the weight for the difference
                        op_wgt = lgq_weights[i_ray] * wave_data[1][i_wave-1]
                        # Set up the difference operand
                        i_oper = zInsertOperand(pzln, i_oper, opertype='DIFF', int1=i_oper_ref, int2=i_oper_dif,
                            data1=hx[i_ray], data2=hy[i_ray], data3=px[i_ray], data4=py[i_ray], tgt=0.0, wgt=op_wgt)
    # Push the lens back to the user foreground.
    if pushback:
        return pzln.zPushLens()

class MeritFunction(object):
    def __init__(self):
        pass
    