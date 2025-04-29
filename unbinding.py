"""Routines directly related to particle unbinding.

Started 17 Feb 2025.
"""

import numpy as np
import monk
from pdb import set_trace
import tools

def unbind_source(
    r, v, m, u, rhalo_init, vhalo_init, hubble_z, boxsize, aexp, status=None, 
    params={}
):
    """
    Run MONK to reduce input particles to their self-bound subset.

    This is a high-level wrapper around the MONK library. Internally, it
    calls the Python Monk handler, which interfaces with the C routine.

    It is possible that this can either be abolished (call the Monk library
    directly), or that additional functionality can be added. TBD.  

    Parameters:
    -----------
    r : ndarray(float) [N, 3]
        Coordinates of all particles in the source set. Should be float64, if
        not an internal copy is created.
    v : ndarray(float) [N, 3]
        Velocities of all particles in the source set. Should be float32, if
        not an internal copy is created.
    m : ndarray(float) [N]
        Masses of all particles in the source set. Values of zero are allowed,
        which makes those particles passive (don't contribute to the
        potential, but can be part of the bound subset). Should be float32,
        if not an internal copy is created.
    u : ndarray(float) [N]
        Internal energies of all particles in the source set. If
        params['ReturnBE'] is 1, this is updated to the binding energy.
        NEED TO CHECK WHETHER IT SHOULD BE SPECIFIC OR TOTAL, and what units
        should be.
    status : ndarray(int) [N] or None, optional
        If not None (default), an N-element array that is 0 or 1
        depending on whether the i-th input particle should be considered 
        as `active' (1) or `passive' (0). Passive particles are treated 
        as mass-less in the first unbinding iteration, i.e. do not affect the
        potential. If None, all particles are assumed to be active.
        THIS MUST BE CHECKED!!!         
    rhalo_init : ndarray(float) [3]
        The initial estimate of the halo position. This will be updated
        to the final position of the subhalo.
    vhalo_init : ndarray [3]
        The initial estimate of the halo velocity. This will be updated 
        to the final velocity of the subhalo.
    hubble_z : float
        The Hubble constant in units of |vel|/|pos|, i.e. usually
        km/s/Mpc. Note that this must be the appropriate value for the 
        target snapshot redshift, *not* H_0!
    params : dict, optional
        Dict of parameters to be passed to MONK. Valid keys are:
            - FixCentre : int
              Should the halo centre be fixed (1) or free (0, default)?
            - CentreMode : int
              Should the halo be centred on the ZMF (0, default) or on the 
              subset of most-bound particles (1)? This is only relevant
              if fixCentre == 0.
            - CentreFrac : float
              If fixCentre == 0 and centreMode == 1, specifies the fraction
              of most-bound particles to use for determining the halo
              centre (in position and velocity). Otherwise, this is ignored.
              Default value is 0.1 (i.e., use most-bound 10%).
            - Monotonic : int (0 or 1)
              Set to 0 to allow 're-binding' of unbound particles. Note that
              this can, in principle, lead to runaway loops, so re-binding
              stops after 20 iterations. Default: 1, monotonic unbinding.
            - ResLimit : float
              The softening resolution limit, which sets the minimum
              distance between two particles or nodes within MONK. The default
              value is 0.0007 (0.7 kpc).
            - PotErrTol : float
              The 'node opening criterion' within MONK. Defaut: 1.0
            - Tolerance : float
              Something that is not clear. Default: 0.005
            - UseTree : int (0 or 1)
              Should MONK compute exact potentials (0) or use a tree
              (1, default)?
            - ReturnBE : int (0 or 1)
              If 1 (default), the internal energy of bound particles will
              be updated to the binding energy upon completion. If 0, no
              binding energy information is passed back from MONK.
            - Verbose : int (0 or ???)
              Set how chatty MONK should be. Default is 0, quiet.


    Returns:
    --------
    ind_bound : ndarray (int), [<=N]
        The indices (into the input particles) that are bound.
    """

    # Make sure all inputs are in the required format and convert them if not
    if r.dtype != np.float64:
        r = r.astype(np.float64)
    if v.dtype != np.float64:
        v = v.astype(np.float64)
    if m.dtype != np.float32:
        m = m.astype(np.float32)
    if u.dtype != np.float32:
        u = u.astype(np.float32)    

    # Take periodic wrapping into account (!!)
    rhalo_init_true = np.array(rhalo_init, copy=True)
    r -= rhalo_init
    rhalo_init[:] = 0
    tools.periodic_wrapping(r, boxsize)
    r *= aexp

    # Positions and velocities must be combined to 6D phase space
    pos6d = np.concatenate((r, v), axis=1)

    # Set particles to initially bound, unless specified otherwise:
    if status is None:
        status = np.zeros(n_part, dtype=np.int32) + 1
    elif status.dtype != np.int32:
        status = status.astype(np.int32)

    # Check params dict and supply defaults if need be
    defaults = [
        ('FixCentre', 0), ('CentreMode', 0), ('CentreFrac', 0.1),
        ('Monotonic', 1), ('ResLimit', 7e-4), ('PotErrTol', 1.0),
        ('UseTree', 1), ('ReturnBE', 1), ('Tolerance', 0.005),
        ('Verbose', 0), ('Bypass', 0)
    ]
    for pair in defaults:
        if pair[0] not in params:
            params[pair[0]] = pair[1]

    # TEST
    params['Verbose'] = 0
            
    # Debugging/testing option that completely bypasses MONK:
    if params['Bypass']:
        ind_bound = np.arange(n_part, dtype=int)
        return ind_bound

    # Call MONK to find bound particles:
    # (Disable maxGap as removed from code)
    ind_bound = monk.monk(
        pos6d,                  # 6D pos 
        m,                      # Mass
        u,                      # Internal energy
        status,                 # Initial binding status
        rhalo_init,             # Centre (pos)
        vhalo_init,             # Centre (vel)
        params['UseTree'],      # Mode (exact [0] or tree [1])
        params['Monotonic'],    # Monotonic unbinding [1]?
        params['Tolerance'],    # Tolerance
        10,                     # Points per leaf
        params['FixCentre'],    # fixCentre
        params['CentreMode'],   # centreMode
        params['CentreFrac'],   # centreFrac
        hubble_z,               # Hubble (H(z))
        params['ResLimit'],     # Softening limit
        params['PotErrTol'],    # Potential error tolerance
        params['Verbose'],      # Monk verbosity
        params['ReturnBE']      # Return (Binding) Energy
    )

    if params['ReturnBE']:
        return ind_bound, u
    else:
        return ind_bound
