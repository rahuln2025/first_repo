"""
Implementation of functions for the traffic model.
"""

import numpy


def rho_red_light(x, rho_max):
    """
    Computes the "red light" initial condition with shock.

    Parameters
    ----------
    x : numpy.ndarray
        Locations on the road as a 1D array of floats.
    rho_max : float
        The maximum traffic density allowed.

    Returns
    -------
    rho : numpy.ndarray
        The initial car density along the road as a 1D array of floats.
    """
    rho = rho_max * numpy.ones_like(x)
    mask = numpy.where(x < 3.0)
    rho[mask] = 0.5 * rho_max
    return rho


def flux(rho, u_max, rho_max):
    """
    Computes the traffic flux F = V * rho.

    Parameters
    ----------
    rho : numpy.ndarray
       Traffic density along the road as a 1D array of floats.
    u_max : float
        Maximum speed allowed on the road.
    rho_max : float
        Maximum car density allowed on the road.

    Returns
    -------
    F : numpy.ndarray
        The traffic flux along the road as a 1D array of floats.
    """
    F = rho * u_max * (1.0 - rho / rho_max)
    return F


def rho_green_light(x, rho_light):
    """  
    Parameters

    x: numpy.ndarray
        Locations on the road as a 1D array of floats
    rho_light : float
        Car density at the stoplight.
        
    Returns
    rho : numpy.ndarray
        The initial car density along the road as a 1D array of floats, at t = 0
        
    """
    
    rho = numpy.zeros_like(x) #_like(x) creates the shape of the array same as x array
    mask = numpy.where(x<2.0)
    rho[mask] = rho_light*x[mask]/2.0
    return rho
