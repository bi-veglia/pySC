import matplotlib.pyplot as plt
import numpy as np

from pySC.utils.at_wrapper import atpass, findorbit6, findorbit4, findspos
from pySC.utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)
import at

def SCdynamicApertureAT(RING, dE, bounds=np.array([0, 1e-3]), nturns=1000, thetas=np.linspace(0, 2 * np.pi, 16),
                      accuracy=1e-6, launchOnOrbit=False, centerOnOrbit=True, useOrbit6=False, auto=0, plot=False):
    #if auto > 0:
    #_, thetas = _autothetas(RING, dE, auto)
    bf,sf,gf = at.get_acceptance(RING,planes=['x','y'],npoints=[10,100],amplitudes=[0.5,0.5],grid_mode=0, nturns=1000,use_mp=True,dp=dE)
    # bf=boundary – (len(refpts),2) array: 1D acceptance
    # sf=survived – (n,) array: Coordinates of survived particles
    # gf= tracked – (n,) array: Coordinates of tracked particles
    if plot:
        plt.figure(6232)
        plt.plot(*gf,'.')
        plt.plot(*sf,'.')
        plt.plot(*bf)
        plt.show()
    rsat=cart2pol(*bf)
    thetas=rsat[1]
    dthetas=-np.diff(thetas)
    RMAXs=rsat[0]
    r0=RMAXs[0:(len(rsat[0]) - 1)]
    r1=RMAXs[1:(len(rsat[0]))]
    DA = np.sum(np.sin(dthetas) * r0*r1 / 2.)
    #the DA is in squared meter, RMAXs are the largest radii found at each of thetas
    return DA, RMAXs, thetas

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def SCmomentumAperture(RING, REFPTS, inibounds, nturns=1000, accuracy=1e-4, stepsize=1e-3, plot=0):
    dboundHI = np.zeros(len(REFPTS))
    dboundLO = np.zeros(len(REFPTS))
    ZCOs = findorbit6(RING, REFPTS)
    if any(~np.isfinite(ZCOs.flatten())):
        dbounds = np.array([dboundHI, dboundLO]).T
        LOGGER.info('Closed Orbit could not be determined during MA evaluation. MA set to zero.')
        return dbounds
    for i in range(len(REFPTS)):
        ord = REFPTS[i]
        local_bounds = inibounds
        SHIFTRING = RING.rotate(-ord)  # TODO +/-
        ZCO = ZCOs[:, i]
        while not check_bounds(local_bounds, SHIFTRING, ZCO, nturns):
            local_bounds = increment_bounds(local_bounds, stepsize)
            LOGGER.debug(f'ord: {ord}; Incremented: {local_bounds}')
        while abs((local_bounds[1] - local_bounds[0]) / max(local_bounds)) > accuracy:
            local_bounds = refine_bounds(local_bounds, SHIFTRING, ZCO, nturns)
            LOGGER.debug(f'ord: {ord}; Refined: {local_bounds}')
        dboundHI[i] = local_bounds[0]
        dboundLO[i] = local_bounds[1]
        LOGGER.debug(f'ord: {ord}; Found: {local_bounds}')
    dbounds = np.array([dboundHI, dboundLO]).T
    if plot:
        spos = findspos(RING)[REFPTS]
        plt.figure(81222)
        plt.clf()
        plt.plot(spos, dboundHI, 'kx-')
        plt.plot(spos, dboundLO, 'rx-')
        plt.xlabel('s [m]')
        plt.ylabel('MA')
        plt.show()
    return dbounds


def _check_bounds(RING, ZCO, nturns, theta, boundsIn):
    Zmin = ZCO[:]
    Zmin[0] = boundsIn[0] * np.cos(theta)
    Zmin[2] = boundsIn[0] * np.sin(theta)
    Zmax = ZCO[:]
    Zmax[0] = boundsIn[1] * np.cos(theta)
    Zmax[2] = boundsIn[1] * np.sin(theta)
    ROUT = atpass(RING, [Zmin, Zmax], nturns, len(RING), keep_lattice=True)  # Track
    RLAST = ROUT[:, len(ROUT) - 1:len(ROUT)]  # Get positions after last turn
    return ~np.isnan(RLAST[0, 0]) and np.isnan(RLAST[0, 1])


def _refine_bounds(RING, ZCO, nturns, theta, boundsIn):
    rmin = boundsIn[0]
    rmax = boundsIn[1]
    rmid = np.mean(boundsIn)
    Z = ZCO[:]
    Z[0] = rmid * np.cos(theta)
    Z[2] = rmid * np.sin(theta)
    ROUT = atpass(RING, Z, nturns, len(RING), keep_lattice=True)  # Track
    RLAST = ROUT[:, len(ROUT) - 1]  # Get positions after last turn
    return [rmin, rmid] if np.isnan(RLAST[0]) else [rmid, rmax]  # Midpoint is outside or inside DA


def _autothetas(RING, dE, nt):
    tin = np.linspace(0, 2 * np.pi * 3 / 4, 4)
    [DA, rs, ts] = SCdynamicAperture(RING, dE, thetas=tin, auto=0)
    a = (rs[0] + rs[2]) / 2
    b = (rs[1] + rs[3]) / 2
    e = np.sqrt(1 - b ** 2 / a ** 2)
    mu = np.arccosh(1 / e)
    nu = np.linspace(0, 2 * np.pi, nt)
    rout = np.abs(np.cosh(mu + 1j * nu))
    tout = np.angle(np.cosh(mu + 1j * nu))
    return rout, tout

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def refine_bounds(local_bounds, RING, ZCO, nturns):
    dmean = np.mean(local_bounds)
    Z0 = ZCO
    Z0[4] = Z0[4] + dmean
    ROUT = atpass(RING, Z0, nturns, len(RING))
    if np.isnan(ROUT[0]):  # Particle dead :(
        local_bounds[1] = dmean  # Set abs-upper bound to midpoint
    else:  # Particle alive :)
        local_bounds[0] = dmean  # Set abs-lower bound to midpoint
    return local_bounds


def check_bounds(local_bounds, RING, ZCO, nturns):
    Z = np.array([ZCO, ZCO])
    Z[4, :] = Z[4, :] + local_bounds[:]
    ROUT = atpass(RING, Z, nturns, len(RING))
    if np.isnan(ROUT[0, 0]) and not np.isnan(ROUT[0, 1]):
        LOGGER.warning('Closer-to-momentum particle is unstable. This shouldnt be!')
    return not np.isnan(ROUT[0, 0]) and np.isnan(ROUT[0, 1])


def increment_bounds(local_bounds, stepsize):
    local_bounds = local_bounds + np.array([-1, 1]) * mysign(local_bounds) * stepsize
    return local_bounds


def mysign(v):
    s = 1 - 2 * (v < 0)
    return s


def scale_bounds(limits, alpha):
    lower = np.mean(limits) - (np.mean(limits) - limits[0]) * alpha
    upper = np.mean(limits) - (np.mean(limits) - limits[1]) * alpha
    if np.sign(lower) != np.sign(limits[0]):
        lower = 0.0
    if np.sign(upper) != np.sign(limits[1]):
        upper = 0.0
    out = [lower, upper]
    return out
