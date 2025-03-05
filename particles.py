"""TargetParticles

Started 17 Feb 2025
"""

import numpy as np
from pdb import set_trace
import h5py as h5
from scipy.spatial import cKDTree

import tools
import unbinding

class ParticlesBase:

    def dummy(self):
        pass

class SnapshotParticles(ParticlesBase):

    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.par = snapshot.par
        self.subhaloes = snapshot.subhaloes
        self.snapshot.particles = self
        self.n_parts = None

        # Load the relevant particle properties into local attributes
        self.load_properties()

    def load_properties(self):
        """Load the relevant particle properties for this snapshot."""
        par = self.par
        snap_file = self.snapshot.snapshot_file
        verbose = self.par['Verbose']

        id_name = par['Input']['Names']['ParticleIDs']
        mass_name = par['Input']['Names']['ParticleMasses']
        pos_name = par['Input']['Names']['ParticleCoordinates']
        vel_name = par['Input']['Names']['ParticleVelocities']
        energy_name = par['Input']['Names']['ParticleInternalEnergies']

        # Load "raw" snapshot information

        ids = np.zeros(0, dtype=np.uint64)
        ptypes = np.zeros(0, dtype=np.int8)
        masses = np.zeros(0)
        coordinates = np.zeros((0, 3))
        velocities = np.zeros((0, 3), dtype=np.float32)
        internal_energies = np.zeros(0)

        # Also need to load particles' FOF IDs if we want to load full FOFs.
        if self.par['Input']['LoadFullFOF']:
            fof_ids = np.zeros(0, dtype=int)

        self.n_pt = np.zeros(6, dtype=int)
        self.n_parts = 0
        
        with h5.File(snap_file, 'r') as f:
            for ptype in par['Input']['TypeList']:
                pt_name = f'PartType{ptype}'
                if verbose:
                    print(f"Loading particle data for type {ptype}...")

                if verbose:
                    print(f"   IDs...")
                curr_ids = f[pt_name + '/' + id_name][...]
                n_pt = len(curr_ids)
                self.n_pt[ptype] = n_pt
                self.n_parts += n_pt
                ids = np.concatenate((ids, curr_ids))
                
                if verbose:
                    print(f"   Masses...")
                if ptype == 5:
                    pt_mass_name = par['Input']['Names']['BHParticleMasses']
                else:
                    pt_mass_name = mass_name
                curr_masses = f[pt_name + '/' + pt_mass_name][...]
                if len(curr_masses) != n_pt:
                    print("Inconsistent length of masses!")
                    set_trace()
                masses = np.concatenate((masses, curr_masses))

                if verbose:
                    print(f"   Coordinates...")
                curr_pos = f[pt_name + '/' + pos_name][...]
                if len(curr_pos) != n_pt:
                    print("Inconsistent length of coordinates!")
                    set_trace()
                coordinates = np.concatenate((coordinates, curr_pos))

                if verbose:
                    print(f"   Velocities...")
                curr_vel = f[pt_name + '/' + vel_name][...]
                if len(curr_vel) != n_pt:
                    print("Inconsistent length of velocities!")
                    set_trace()
                velocities = np.concatenate((velocities, curr_vel))

                # Internal energies are only relevant for gas (PT0). But we
                # still need to load dummy entries (0) for other types.
                if verbose:
                    print(f"   Internal Energies...")
                if ptype == 0:
                    curr_u = f[pt_name + '/' + energy_name][...]
                    if len(curr_u) != n_pt:
                        print("Inconsistent length of internal energies!")
                        set_trace()
                else:
                    n_curr = len(curr_ids)
                    curr_u = np.zeros(n_curr)
                internal_energies = np.concatenate((internal_energies, curr_u))

                # Record the type of each particle, if desired
                if self.par['Input']['RecordParticleType']:
                    curr_ptype = np.zeros(n_pt, dtype=np.int8) + ptype
                    ptypes = np.concatenate((ptypes, curr_ptype))

                # Load FOF indices, if desired
                if self.par['Input']['LoadFullFOF']:
                    curr_fof = f[pt_name + '/' + fof_name][...]
                    if len(curr_fof) != n_pt:
                        print("Inconsistent length of FOF IDs!")
                        set_trace()
                    fof_ids = np.concatenate((fof_ids, curr_fof))

        # Store properties as attributes
        self.ids = ids
        self.ptypes = ptypes
        self.masses = masses
        self.coordinates = coordinates
        self.velocities = velocities
        self.internal_energies = internal_energies
        if self.par['Input']['LoadFullFOF']:
            self.fof = fof_ids

        self.n_parts = np.sum(self.n_pt)
        if self.n_parts != self.ids.shape[0]:
            print("Inconsistent particle lengths!!")
            set_trace()

        # Second part: get membership info...
        self.subhalo_indices = self.subhaloes.build_particle_memberships(self)
        
    def ids_to_indices(self, ids):
        """Find the internal indices corresponding to input IDs."""
        if not hasattr(self, 'reverse_ids'):
            success = self.build_reverse_ids()

            # If we couldn't build a reverse list, we have to do this
            # by brute force. This may take a while...
            if not success:
                indices = tools.find_id_indices(ids, self.ids)
                return indices

        # If we got here, we have a reverse ID list, so we just look up
        # the answer(s).
        try:
            return self.reverse_ids[ids]
        except IndexError:
            set_trace()
        
    def build_reverse_ids(self):
        """Construct a reverse ID lookup list for quick index-finding."""
        # Check that IDs are small enough to fit into memory
        max_id = int(np.max(self.ids))
        if max_id > 1e10:
            print(f"Max ID is {max_id} -- too large for reverse list.")
            return False

        # Ok, we can go ahead
        self.reverse_ids = np.zeros(max_id + 1, dtype=int) - 1
        self.reverse_ids[self.ids] = np.arange(self.n_parts, dtype=int)
        return True
        
    def get_property(self, name, indices=None):
        """Retrieve a named particle property.

        Parameters
        ----------
        name : str
            The name of the attribute holding the property.
        indices : ndarray(int) or None, optional
            The indices of particles whose property should be retrieved. If
            None (default), all particles are loaded.

        Returns
        -------
        data : ndarray
            The requested property for the required particles.
        """
        try:
            attr = getattr(self, name)
        except AttributeError:
            print(f"Desired particle property '{name}' is not loaded!")
            set_trace()

        if indices is None:
            return attr
        else:
            return attr[indices]

    def initialise_memberships(self):
        """Initialise the subhalo membership of each particle to its central.

        The details differ depending on whether we process full FOFs or only
        particles in subhaloes, so we just delegate to an appropriate function.
        """
        if self.par['Input']['LoadFullFOF']:
            self.initialize_memberships_from_fof()
        else:
            self.initialize_memberships_from_subhaloes()

        # Initialize every particle as high origin and large radii
        self.origins = np.zeros(self.n_parts, dtype=np.int8) + 100
        self.radii = np.zeros(self.n_parts, dtype=np.float32) + np.inf
        
    def initialize_memberships_from_subhaloes(self):
        """Initialize particle membership to centrals from input subhaloes."""
        ind_in_subhalo = np.nonzero(self.subhalo_indices >= 0)[0]
        self.subhalo_indices[ind_in_subhalo] = (
            self.subhaloes.centrals[self.subhalo_indices[ind_in_subhalo]])

    def initialize_memberships_from_fof(self):
        """Initialize particle membership to centrals from FOF groups."""
        # Find the central from FOF IDs
        cen_gal_fof = self.subhaloes.fof_to_central[self.fof_ids]

        # Special treatment for particles that are in a galaxy but not in a FOF
        ind_gal_nofof = np.nonzero(
            (self.subhalo_indices >= 0) & (subhalo_indices < 0))[0]
        cen_gal_nofof = (
            self.subhaloes.centrals[self.subhalo_indices[ind_gal_nofof]])

        # Update internal subhalo_indices list (can only do it now, because we
        # needed the original indices for no-FOF-galaxies)
        self.subhalo_indices = cen_gal_fof
        self.subhalo_indices[ind_gal_nofof] = cen_gal_nofof

    def get_ids_in_sphere(self, cen, r, cen_sh):
        """Find the particle IDs within a sphere."""
        if not hasattr(self, 'tree'):
            boxsize = self.snapshot.sim.boxsize
            self.tree = cKDTree(
                self.coordinates, boxsize=boxsize, leafsize=1024)
        ind_ngbs = self.tree.query_ball_point(cen, r)
        ind_ngbs = np.array(ind_ngbs)

        ngb_cens = self.subhaloes.centrals[self.subhalo_indices[ind_ngbs]]
        subind = np.nonzero(ngb_cens == cen_sh)[0]
        return self.ids[ind_ngbs[subind]]

    def update_membership(self, galaxy_particles, galaxy):
        """Incorporate unbinding result into the particle catalogue."""

        ish = galaxy.ish
        subhaloes = galaxy.subhaloes
        boxsize = subhaloes.sim.boxsize
        
        bound_inds = galaxy.source_indices[galaxy_particles.ind_bound]
        bound_origins = galaxy_particles.origins[galaxy_particles.ind_bound]
        bound_dpos = (
            self.coordinates[bound_inds, :]
            - subhaloes.new_coordinates[ish, :]
        )
        tools.periodic_wrapping(bound_dpos, boxsize)
        bound_radii = np.linalg.norm(bound_dpos, axis=1)

        old_origins = self.origins[bound_inds]
        old_radii = self.radii[bound_inds]

        ind_better = np.nonzero(
            (bound_origins < old_origins) |
            ((bound_origins == old_origins) & (bound_radii < old_radii))
        )[0]

        # Record the result
        self.origins[bound_inds[ind_better]] = bound_origins[ind_better]
        self.radii[bound_inds[ind_better]] = bound_radii[ind_better]
        self.subhalo_indices[bound_inds[ind_better]] = ish
        
    
    def switch_memberships_to_output(self, output_shis):
        """Update particle membership to output subhalo IDs and trim."""

        # Update subhalo indices to new. This will automatically set any
        # particle in a too-small subhalo to -1 (unbound).
        self.subhalo_indices = output_shis[self.subhalo_indices]
        self.reject_unbound()

    def reject_unbound(self):
        """Trim the particle list to exclude any that are not in a subhalo."""
        n_before = len(self.subhalo_indices)
        ind_in_subhalo = np.nonzero(self.subhalo_indices >= 0)[0]
        n_now = len(ind_in_subhalo)
        print(f"About to reject unbound particles ({n_before} to {n_now}).")
        update_particle_fields(ind_in_subhalo)

    def update_particle_fields(source_indices):
        """Re-arrange the particle fields by pulling from source_indices."""

        # Record current particle numbers and check that they are ok
        n_part_before = len(self.ids)
        n_pt_before = self.n_pt
        if n_part_before != np.sum(self.n_pt):
            print("Inconsistent particle numbers!")
            set_trace()

        # Re-arrange the data fields
        self.ids = self.ids[ind_in_subhalo]
        self.ptypes = self.ptypes[ind_in_subhalo]
        self.masses = self.masses[ind_in_subhalo]
        self.coordinates = self.coordinates[ind_in_subhalo, :]
        self.velocities = self.velocities[ind_in_subhalo, :]
        self.internal_energies = self.internal_energies[ind_in_subhalo]
        self.subhalo_indices = self.subhalo_indices[ind_in_subhalo]

        # Re-calculate number of particles by type
        self.n_pt = np.bincount(self.ptypes, minlength=6)
        n_part_now = np.sum(self.n_pt)

        print(f"Updated particle fields ({n_before} --> {n_now}).")
        for pt in range(6):
            print(f"   PartType {pt}: {n_pt_before[pt]} --> {self.n_pt[pt]}")

    def calculate_radii(centres):
        """Calculate the radii of all particles from the halo centres."""

        if np.max(self.subhalo_indices) >= len(centres):
            print("Uh oh. We don't have enough centres...")
            set_trace()

        centres_by_part = centres[self.subhalo_indices, :]
        self.radii = np.linalg.norm(self.coordinates - centres_by_part)
        tools.periodic_wrapping(self.radii, boxsize=self.par['Sim']['Boxsize'])    

    def rearrange_for_output():
        """Rearrange the particles by subhalo and radius."""

        # Make sure we don't have any unbound particles
        if np.min(self.subhalo_indices) < 0:
            print("Why are there unbound particles??")
            set_trace()
        sorter = np.lexsort((self.radii, self.subhalo_indices))
        update_particle_fields(sorter)


class GalaxyParticles(ParticlesBase):

    def __init__(self, galaxy):
        self.num_part = None
        self.galaxy = galaxy
        self.snap = galaxy.snap
        self.verbose = self.snap.verbose

    def check_number_consistency(self, quant):
        """Check that the number of elements agrees with internal size."""
        if self.num_part is not None:
            if len(quant) != self.num_part:
                print(f"Requested adding {len(quant)} elements to source, but "
                      f"there are {self.num_part} particles!")
                set_trace()
        else:
            self.num_part = len(quant)

    def set_r(self, r):
        self.check_number_consistency(r)
        self.r = r.astype(np.float64)

    def set_v(self, v):
        self.check_number_consistency(v)
        self.v = v.astype(np.float64)

    def set_m(self, m):
        self.check_number_consistency(m)
        self.m = m.astype(np.float32)

    def set_m_real(self, m):
        self.check_number_consistency(m)
        self.m_real = m.astype(np.float32)
        
    def set_u(self, u):
        self.check_number_consistency(u)
        self.u = u.astype(np.float32)

    def set_origins(self, origins):
        self.check_number_consistency(origins)
        self.origins = origins
        
    def set_initial_status(self):
        self.initial_status = np.zeros(self.num_part, dtype=np.int32) + 1
    
    def unbind(self):
        """Perform the unbinding procedure on the stored particles."""

        # Set here any non-standard MONK parameters.
        # TODO: import params from parameters
        monk_params = {'Bypass': False, 'UseTree': 0, 'ReturnBE': 0}

        n_tot = len(self.m)
        n_passive = np.count_nonzero(self.m == 0)
        print(
            f"There are {n_passive} passive particles out of {n_tot}.\n\n")
        ind_bound = unbinding.unbind_source(
            self.r, self.v, self.m, self.u,
            self.galaxy.r_init, self.galaxy.v_init, self.snap.hubble_z,
            status=self.initial_status, params=monk_params
        )
        
        # Also need to record unbinding result...
        self.ind_bound = ind_bound
        
        if self.verbose:
            for iorigin in range(6):
                n_source = np.count_nonzero(self.origins == iorigin)
                n_final = np.count_nonzero(
                    self.origins[ind_bound] == iorigin)
                print(f"Origin {iorigin}: {n_source} --> {n_final}")
        
            # Check how many passive particles ended up becoming bound
            n_passive_bound = np.count_nonzero(self.m[ind_bound] == 0)
            if n_passive_bound > 0.05 * len(ind_bound):
                print(f"WARNIING: {n_passive_bound} / {len(ind_bound)} bound "
                      f"particles are passive!")

                
        # Find the coordinates of the 'most bound' (lowest PE) particle,
        # to be returned. This is easy because MONK internally moves this
        # particle to the front of the returned list.
        ind_mostbound = ind_bound[0]
        halo_centre_of_potential = self.r[ind_mostbound, :]

        # Compute the total bound mass after the end of MONK
        m_bound_after_monk = np.sum(self.m[ind_bound])
        
        return halo_centre_of_potential, m_bound_after_monk
