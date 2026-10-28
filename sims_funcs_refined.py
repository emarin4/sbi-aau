# -*- coding: utf-8 -*-
"""
Functions to run the particle-spray simulations for the SBI AAU Project
"""
# Standard library
from __future__ import annotations
from typing import Any, Callable 
__docformat__ = "numpy"  # optional, descriptive only
import os, math, sys, warnings, inspect, zipfile

# Third-party
import numpy as np
import h5py
from scipy import special
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

import jax.numpy as jnp
from astropy.coordinates import SkyCoord, Galactocentric, ICRS
import astropy.units as u

# Custom packages
try:
    import agama
except ImportError:
    raise RuntimeError("agama package required for potential calculation.")

# length scale = 1 kpc, velocity = 1 km/s, mass = 1 Msun
agama.setUnits(mass=1, length=1, velocity=1)
# agama.setNumThreads(48)  # Explicitly set the number of threads

def icrs_to_aau(ra_rad, dec_rad):
    """
    define a *differentiable* coordinate transfrom from ra and dec --> AAU phi1, phi2
    Using the rotation matrix from Shipp+2019
    ra_rad: icrs ra [radians]
    dec_red: icrs dec [radians]
    """
    R = np.array(
        [
            [0.83697865, 0.29481904, -0.4610298],
            [0.51616778,-0.70514011, 0.4861566],
            [0.18176238, 0.64487142, 0.74236331],
        ]
    )

    icrs_vec = np.vstack(
        [
            np.cos(ra_rad) * np.cos(dec_rad),
            np.sin(ra_rad) * np.cos(dec_rad),
            np.sin(dec_rad),
        ]
    ).T

    stream_frame_vec = np.einsum("ij,kj->ki", R, icrs_vec)

    phi1 = np.arctan2(stream_frame_vec[:, 1], stream_frame_vec[:, 0]) * 180 / np.pi
    phi2 = np.arcsin(stream_frame_vec[:, 2]) * 180 / np.pi

    return phi1, phi2


def get_subhalo_impact_w(
    w_p_impact: np.ndarray,
    b: float,
    v_rel_para: float,
    v_rel_perp: float,
    psi: float = 0.0,
    theta: float = 90.0
) -> np.ndarray:
    """
    Computes the phase-space position of a subhalo at the moment of impact 
    using a stream-centric coordinate system.
    
    The coordinate frame is defined by:
    - e1: parallel to the stream particle's velocity (flow direction)
    - e3: perpendicular to the orbital plane (angular momentum direction)
    - e2: completes the right-handed basis (e2 = e3 × e1)
    
    Parameters
    ----------
    w_p_impact : np.ndarray, shape (6,)
        The 6D phase-space vector [x, y, z, vx, vy, vz] of the stream 
        particle at t_impact. 
        Units: [kpc, kpc, kpc, km/s, km/s, km/s]
    b : float
        Impact parameter (kpc). The perpendicular distance between the 
        subhalo and the stream particle at the moment of impact, measured
        in the plane perpendicular to the stream velocity.
    v_rel_para : float
        Relative velocity component parallel to the stream flow (km/s).
        Positive means the subhalo is moving faster along the stream direction.
    v_rel_perp : float
        Magnitude of relative velocity component perpendicular to the stream 
        flow (km/s). The direction is specified by theta.
    psi : float, optional
        Azimuthal angle specifying the subhalo's position in the plane 
        perpendicular to the stream velocity (degrees).
        - psi = 0: offset in the e2 direction (L × V direction)
        - psi = 90: offset in the e3 direction (orbital normal)
        Default: 0.0 (this is alpha from Tri's paper.)
    theta : float, optional
        Azimuthal angle specifying the direction of perpendicular relative 
        velocity in the plane perpendicular to stream flow (degrees).
        - theta = 0: perpendicular velocity in e2 direction
        - theta = 90: perpendicular velocity in e3 direction
        Default: 90.0
    
    Returns
    -------
    w_sh : np.ndarray, shape (6,)
        6D phase-space vector [x, y, z, vx, vy, vz] of the subhalo at impact.
        Units: [kpc, kpc, kpc, km/s, km/s, km/s]
    
    Notes
    -----
    The geometry assumes a local linearized encounter where the impact 
    parameter b represents the minimum separation. The basis vectors are:
    - e1 = V / |V|  (stream flow direction)
    - e3 = L / |L|  (orbital angular momentum direction)
    - e2 = e3 × e1  (completes right-handed system)
    
    Examples
    --------
    >>> # Stream particle at impact: position and velocity
    >>> w_stream = np.array([8.0, 0.0, 0.1, 0.0, 220.0, 0.0])
    >>> # Subhalo passes 0.5 kpc away with 50 km/s relative velocity
    >>> w_subhalo = get_subhalo_impact_w(
    ...     w_stream, b=0.5, v_rel_para=30.0, v_rel_perp=40.0
    ... )
    """
    #x_p = w_p_impact[:3]  # position in kpc
    #v_p = w_p_impact[3:]  # velocity in km/s
    
    x_p = np.asarray(w_p_impact[:3], dtype=float).copy()
    v_p = np.asarray(w_p_impact[3:], dtype=float).copy()
    # Convert angles from degrees to radians
    psi_rad = np.deg2rad(psi)
    theta_rad = np.deg2rad(theta)
    
    # --- 1. Construct orthonormal basis ---
    # e1: Parallel to stream velocity (flow direction)

    e1 = v_p / np.linalg.norm(v_p) 
    
    # e3: Normal to the orbital plane (angular momentum direction)
    L = np.cross(x_p, v_p)
    e3 = L / np.linalg.norm(L)     
    
    # e2: Completes right-handed basis, perpendicular to both
    e2 = np.cross(e3, e1)          
    
    # --- 2. Calculate subhalo position ---
    # Place subhalo at distance b in the perpendicular plane
    # Position offset: b * [cos(psi)*e2 + sin(psi)*e3]
    x_sh = x_p + b * (np.cos(psi_rad) * e2 + np.sin(psi_rad) * e3)
    
    # --- 3. Calculate subhalo velocity ---
    # Relative velocity has parallel and perpendicular components
    # v_sh = v_p + v_rel_para*e1 + v_rel_perp*[cos(theta)*e2 + sin(theta)*e3]
    v_parallel_vec = v_rel_para * e1
    v_perp_vec = v_rel_perp * (np.cos(theta_rad) * e2 + np.sin(theta_rad) * e3)
    v_sh = v_p + v_parallel_vec + v_perp_vec
    
    return np.concatenate([x_sh, v_sh])


def get_target_w_at_impact(
    stream_today_cart: np.ndarray, 
    stream_today_phi1: np.ndarray, 
    target_phi1: float, 
    delta_phi1: float,
    t_impact: float, 
    potential: Any
) -> np.ndarray:
    """
    Finds the phase-space position of a stream segment at the impact time
    by averaging particles in a specified phi1 range today and integrating
    backwards in time.
    
    Parameters
    ----------
    stream_today_cart : np.ndarray, shape (N, 6)
        Stream particles at present day (t=0) in Galactic Cartesian coordinates.
        Columns: [x, y, z, vx, vy, vz]
        Units: [kpc, kpc, kpc, km/s, km/s, km/s]
    stream_today_phi1 : np.ndarray, shape (N,)
        Corresponding phi1 coordinates of stream particles at present day.
        Units: degrees
    target_phi1 : float
        The phi1 coordinate (degrees) where the impact should occur in the
        present-day stream configuration. The function will find particles
        near this location and trace them back to t_impact.
    delta_phi1 : float
        The half-width (degrees) around target_phi1 to include in the average.
        Particles with |phi1 - target_phi1| < delta_phi1 are averaged.
    t_impact : float
        Time of impact (Gyr), measured relative to present day (t=0).
        Should be negative for impacts in the past (e.g., -1.0 for 1 Gyr ago).
    potential : Potential object
        Galactic potential (must have agama.orbit interface) used to 
        integrate the orbit backwards in time.
    
    Returns
    -------
    w_p_impact : np.ndarray, shape (6,)
        6D phase-space vector [x, y, z, vx, vy, vz] of the stream segment
        at the impact time t_impact.
        Units: [kpc, kpc, kpc, km/s, km/s, km/s]
    
    Notes
    -----
    The function performs two steps:
    1. Identifies stream particles within delta_phi1 of target_phi1 today
       and computes their average position/velocity
    2. Integrates this average state backwards from t=0 to t=t_impact
    
    If no particles are found within delta_phi1, the closest particle to
    target_phi1 is used instead.
    
    Examples
    --------
    >>> # Get stream position 1 Gyr ago at phi1 = -5 deg (today)
    >>> w_impact = get_target_w_at_impact(
    ...     stream_today_cart=stream_particles,
    ...     stream_today_phi1=phi1_coords,
    ...     target_phi1=-5.0,
    ...     delta_phi1=0.5,
    ...     t_impact=-1.0,  # 1 Gyr ago
    ...     potential=pot_mw
    ... )
    """
    # --- 1. Select and average particles near target_phi1 ---
    phi1_bool = np.isclose(stream_today_phi1, target_phi1, atol=delta_phi1)
    
    # If particles found, take mean; otherwise use closest particle
    if np.any(phi1_bool):
        avg_w_today = stream_today_cart[phi1_bool].mean(axis=0)
    else:
        # Fallback: use the single closest particle
        closest_idx = np.argmin(np.abs(stream_today_phi1 - target_phi1))
        avg_w_today = stream_today_cart[closest_idx]
    
    # --- 2. Integrate backwards to t_impact ---
    # Convert t_impact (Gyr) to integration time
    # For AGAMA: integrate from timestart=0 (today) to time=t_impact
    w_p_impact = agama.orbit(
        ic=avg_w_today, 
        potential=potential,  # Use the passed potential, not hardcoded potTotal
        trajsize=1,  # Only return final point
        time=t_impact,  # Target time (negative = past)
        timestart=0.0  # Start at present day
    )[1].flatten()
    
    return w_p_impact

def create_perturber_dict(
    stream_today_cart: np.ndarray,
    stream_today_phi1: np.ndarray,
    potential: Any,
    mass_perturber: float,
    scaleradius_perturber: float,
    impact_phi1: float,
    impact_time: float,
    impact_parameter: float,
    v_rel_parallel: float,
    v_rel_perpendicular: float,
    alpha_position: float = 0.0,
    alpha_velocity: float = 90.0,
    delta_phi1: float = 0.5, 
    time_window: float = 0.5

) -> Dict[str, Any]:
    """
    Create a perturber dictionary for stream-subhalo interaction simulations.
    
    This function computes the full 6D phase-space configuration of a subhalo
    at the time of impact with a stellar stream, given the impact geometry
    parameters. The resulting dictionary can be passed to stream simulation
    functions.
    
    Parameters
    ----------
    stream_today_cart : np.ndarray, shape (N, 6)
        Stream particles at present day in Galactic Cartesian coordinates.
        Units: [kpc, kpc, kpc, km/s, km/s, km/s]
    stream_today_phi1 : np.ndarray, shape (N,)
        Corresponding phi1 coordinates of stream particles at present day.
        Units: degrees
    potential : Potential object
        Galactic potential used to integrate orbits backwards in time.
    mass_perturber : float
        Mass of the perturbing subhalo. Units: Msun
    scaleradius_perturber : float
        Scale radius of the perturbing subhalo (NFW or Plummer profile).
        Units: kpc
    impact_phi1 : float
        The phi1 coordinate where the impact occurs in the present-day
        stream configuration. Units: degrees
    impact_time : float
        Time of impact relative to present day. Negative for past impacts.
        Units: Gyr
    impact_parameter : float
        Perpendicular distance between subhalo and stream at closest approach.
        Units: kpc
    v_rel_parallel : float
        Relative velocity component parallel to stream flow direction.
        Positive means subhalo overtakes stream, negative means lagging.
        Units: km/s
    v_rel_perpendicular : float
        Magnitude of relative velocity component perpendicular to stream flow.
        Must be non-negative. Units: km/s
    alpha_position : float, optional
        Azimuthal angle specifying subhalo position in the plane perpendicular
        to stream velocity. 0 corresponds to (L × V) direction. Units: degree
        Default: 0.0
    alpha_velocity : float, optional
        Azimuthal angle specifying direction of perpendicular relative velocity.
        Units: degree. Default: 90.0
    delta_phi1 : float, optional
        Half-width around impact_phi1 for averaging stream particles.
        Units: degrees. Default: 0.5
    
    Returns
    -------
    perturber_dict : dict
        Dictionary containing perturber parameters with keys:
        - 'mass': perturber mass (Msun)
        - 'scaleRadius': perturber scale radius (kpc)
        - 'w_subhalo_impact': 6D phase-space vector at impact (kpc, km/s)
        - 'time_impact': time of impact (Gyr)
        - 'time_window': time window for which the subhalo is "turned on", centered on time_impact (Gyr)
    """
    # Get stream particle position/velocity at impact time
    w_stream_impact = get_target_w_at_impact(
        stream_today_cart=stream_today_cart,
        stream_today_phi1=stream_today_phi1,
        target_phi1=impact_phi1,
        delta_phi1=delta_phi1,
        t_impact=impact_time,
        potential=potential
    )
    
    # Compute subhalo position/velocity at impact using geometry
    w_subhalo_impact = get_subhalo_impact_w(
        w_p_impact=w_stream_impact,
        b=impact_parameter,
        v_rel_para=v_rel_parallel,
        v_rel_perp=v_rel_perpendicular,
        psi=alpha_position,
        theta=alpha_velocity
    )
    
    # Create perturber dictionary
    perturber_dict = {
        'mass': mass_perturber,
        'scaleRadius': scaleradius_perturber,
        'w_subhalo_impact': w_subhalo_impact,
        'time_impact': impact_time,
        'time_window': time_window
    }
    
    return perturber_dict


def galcen_to_aau(part_xv):
    """
    Transform Galactocentric coordinates, aka what is returned by 
    nbody.fast_sims.create_particle_spray_stream:
    
    'part_xv' : np.ndarray of shape (Nparticles, Nsaves, 6) or (Nparticles, 6) if save_rate=1
    Stream particle states at each save time.
    Contains NaN values where particles weren't yet released.
    Format: [x, y, z, vx, vy, vz] in [kpc, km/s]

    and transforms into AAU coordinate frame (phi1, phi2).
    """

    c_galcen = SkyCoord(
        x= part_xv[:, 0] * u.kpc,
        y= part_xv[:, 1] * u.kpc,
        z= part_xv[:, 2] * u.kpc,
        frame=Galactocentric()
    )

    c_icrs = c_galcen.transform_to(ICRS())

    ra_rad = jnp.array(c_icrs.ra.to(u.rad).value)
    dec_rad = jnp.array(c_icrs.dec.to(u.rad).value)

    #Apply AAU rotation matrix (Shipp+2019)
    R = np.array([
        [0.83697865,  0.29481904, -0.4610298],
        [0.51616778, -0.70514011,  0.4861566],
        [0.18176238,  0.64487142,  0.74236331],
    ])

    icrs_vec = np.vstack([
        np.cos(ra_rad) * np.cos(dec_rad),
        np.sin(ra_rad) * np.cos(dec_rad),
        np.sin(dec_rad),
    ]).T

    stream_vec = np.einsum("ij,kj->ki", R, icrs_vec)

    phi1 = np.arctan2(stream_vec[:, 1], stream_vec[:, 0]) * 180 / np.pi
    phi2 = np.arcsin(stream_vec[:, 2]) * 180 / np.pi

    return phi1, phi2

AAU_ROTATION_MATRIX = np.array([
        [0.83697865,  0.29481904, -0.4610298],
        [0.51616778, -0.70514011,  0.4861566],
        [0.18176238,  0.64487142,  0.74236331],
    ])

def galcen_to_aau_full(part_xv, R=AAU_ROTATION_MATRIX):
    """
    Transform Galactocentric phase-space coordinates (as returned by
    nbody.fast_sims.create_particle_spray_stream) into the AAU stream frame.

    Parameters
    ----------
    part_xv : np.ndarray, shape (Nparticles, 6)
        [x, y, z, vx, vy, vz] in [kpc, km/s]. May contain NaN rows for
        particles not yet released.
    R : np.ndarray, shape (3, 3)
        Rotation matrix from ICRS to the AAU stream frame (Shipp+2019).

    Returns
    -------
    dict with keys in NODE_FEATURE_NAMES order — phi1, phi2, vr, pm1, pm2, dist
    (phi1, phi2 in deg; vr in km/s; pm1, pm2 in mas/yr; dist in kpc).
    """
    c_galcen = SkyCoord(
        x=part_xv[:, 0] * u.kpc, y=part_xv[:, 1] * u.kpc, z=part_xv[:, 2] * u.kpc,
        v_x=part_xv[:, 3] * u.km/u.s, v_y=part_xv[:, 4] * u.km/u.s, v_z=part_xv[:, 5] * u.km/u.s,
        frame=Galactocentric()
    )
    c_icrs = c_galcen.transform_to(ICRS())

    ra_rad = c_icrs.ra.rad
    dec_rad = c_icrs.dec.rad

    icrs_vec = np.vstack([
        np.cos(ra_rad) * np.cos(dec_rad),
        np.sin(ra_rad) * np.cos(dec_rad),
        np.sin(dec_rad),
    ]).T
    stream_vec = np.einsum("ij,kj->ki", R, icrs_vec)

    phi1 = np.degrees(np.arctan2(stream_vec[:, 1], stream_vec[:, 0]))
    phi2 = np.degrees(np.arcsin(stream_vec[:, 2]))

    dist = c_icrs.distance.kpc
    vr = c_icrs.radial_velocity.to(u.km/u.s).value
    pm_ra_cosdec = c_icrs.pm_ra_cosdec.to(u.mas/u.yr).value
    pm_dec = c_icrs.pm_dec.to(u.mas/u.yr).value

    e_ra = np.vstack([-np.sin(ra_rad), np.cos(ra_rad), np.zeros(len(ra_rad))]).T
    e_dec = np.vstack([
        -np.sin(dec_rad) * np.cos(ra_rad),
        -np.sin(dec_rad) * np.sin(ra_rad),
         np.cos(dec_rad),
    ]).T
    pm_icrs_vec = pm_ra_cosdec[:, None] * e_ra + pm_dec[:, None] * e_dec
    pm_stream_vec = np.einsum("ij,kj->ki", R, pm_icrs_vec)

    phi1_rad, phi2_rad = np.radians(phi1), np.radians(phi2)
    e_phi1 = np.vstack([-np.sin(phi1_rad), np.cos(phi1_rad), np.zeros(len(phi1))]).T
    e_phi2 = np.vstack([
        -np.sin(phi2_rad) * np.cos(phi1_rad),
        -np.sin(phi2_rad) * np.sin(phi1_rad),
         np.cos(phi2_rad),
    ]).T
    pm1 = np.sum(pm_stream_vec * e_phi1, axis=1)
    pm2 = np.sum(pm_stream_vec * e_phi2, axis=1)

    # order matches NODE_FEATURE_NAMES universally across the pipeline
    return {"phi1": phi1, "phi2": phi2, "vr": vr, "pm1": pm1, "pm2": pm2, "dist": dist}


def compute_time_window(v_rel_perp, v_rel_para, impact_parameter):
    """
    Compute the time window for the perturber impact.
    
    Parameters
    ----------
    v_rel_perp : float
        Perpendicular relative velocity (km/s).
    v_rel_para : float
        Parallel relative velocity (km/s).
    impact_parameter : float
        Impact parameter (kpc).

    Returns
    -------
    time_window : float
        Time window for the perturber impact (Gyr), clipped to [0.05, 3.0].
    """
    
    time_window = 135 * (impact_parameter / np.sqrt(v_rel_perp**2 + v_rel_para**2))
    return np.clip(time_window, 0.05, 3.0)
    

def rv_to_gsr(ra_deg, dec_deg, v_hel_kms):

    """
    Convert a heliocentric radial velocity to the Galactic Standard of Rest (GSR). Intended for observational data. 

    **For observational data**

    Removes the projection of the Sun's motion along the line of sight to the
    target, so the returned velocity is measured relative to the Galactic centre
    rather than the Sun.

    Parameters
    ----------
    ra_deg : float or array-like
        Right ascension of the target(s) in degrees (ICRS).
    dec_deg : float or array-like
        Declination of the target(s) in degrees (ICRS).
    v_hel_kms : float or array-like
        Heliocentric radial velocity in km/s.

    Returns
    -------
    v_gsr : float or np.ndarray
        GSR radial velocity in km/s.
    """

    import pandas as pd
    import astropy.coordinates as coord
    import astropy.units as u
    c = coord.SkyCoord(
        ra=ra_deg * u.deg, dec=dec_deg * u.deg,
        radial_velocity=v_hel_kms * u.km/u.s, frame='icrs'
    )
    v_sun       = coord.Galactocentric().galcen_v_sun.to_cartesian()
    gal         = c.transform_to(coord.Galactic)
    unit_vector = gal.data.to_cartesian() / gal.data.to_cartesian().norm()
    return (c.radial_velocity + v_sun.dot(unit_vector)).to(u.km/u.s).value

def compute_vgsr(xv):
    """
    Compute the GSR line-of-sight velocity for an array of particles. Intended for simulation data in Galactocentric coordinates.

    **For simulation data**

    Projects each particle's heliocentric velocity onto the unit vector pointing
    from the Sun to the particle, then adds back the Sun's own motion along that
    direction to convert to the Galactic Standard of Rest frame.

    Parameters
    ----------
    xv : np.ndarray, shape (N, 6)
        Galactocentric phase-space array with columns [x, y, z, vx, vy, vz]
        in units of kpc and km/s. Requires `sun_pos` (kpc) and `sun_vel` (km/s)
        to be defined in the enclosing scope.

    Returns
    -------
    v_gsr : np.ndarray, shape (N,)
        GSR line-of-sight velocity for each particle in km/s.
    """
    
    pos     = xv[:, :3]
    vel     = xv[:, 3:]
    pos_hel = pos - sun_pos
    dist    = np.linalg.norm(pos_hel, axis=1, keepdims=True)
    los_hat = pos_hel / dist
    vel_hel = vel - sun_vel
    return np.sum(vel_hel * los_hat, axis=1) + np.sum(sun_vel * los_hat, axis=1)

NODE_FEATURE_NAMES = ["phi1", "phi2", "vr", "pm1", "pm2", "dist"]
def write_graph_dataset(
    path: str,
    theta: np.ndarray,
    feats_list: list[np.ndarray],
    param_names: list[str],
    headers: dict | None = None,
) -> None:
    """Write a batch of simulated streams to an HDF5 graph dataset.

    Layout matches what jgnn.datasets.io.read_graph_dataset expects:
    node features (one flat dataset per name in NODE_FEATURE_NAMES,
    each galaxy's particles concatenated together, split back out via
    'ptr'), and one graph-level dataset per entry in param_names. Function 
    by Tri Nguyen. 

    Args:
        path: Destination HDF5 path.
        theta: (n_sims, len(param_names)) physical-unit parameters.
        feats_list: Length-n_sims list of (n_i, 6) node-feature arrays,
            columns NODE_FEATURE_NAMES (phi1, phi2, dist, pm1, pm2, vr).
        param_names: Names for theta's columns, in order.
        headers: Extra scalar attributes to store (e.g. round, tau).
    """
    theta = np.asarray(theta)
    n_particles = np.array([feats.shape[0] for feats in feats_list])
    ptr = np.cumsum([0] + list(n_particles))
    stacked = np.concatenate(feats_list, axis=0)

    graph_features = list(param_names) + ['num_particles', 'ptr']
    all_features = NODE_FEATURE_NAMES + graph_features

    with h5py.File(path, 'w') as f:
        for i, name in enumerate(NODE_FEATURE_NAMES):
            f.create_dataset(name, data=stacked[:, i])
        f.create_dataset('num_particles', data=n_particles)
        f.create_dataset('ptr', data=ptr)
        for i, name in enumerate(param_names):
            f.create_dataset(name, data=theta[:, i])

        f.attrs['all_features'] = all_features
        f.attrs['node_features'] = NODE_FEATURE_NAMES
        f.attrs['graph_features'] = graph_features
        for key, value in (headers or {}).items():
            f.attrs[key] = value
