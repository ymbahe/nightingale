"""Galaxy population information.

Started 17 Feb 2025.
"""

import ioi
import ion
import tools
import numpy as np
from pdb import set_trace
from particles import GalaxyParticles
import h5py as h5

class GalaxyBase:

    def dummy(self):
        print("Test")

class SnapshotGalaxies(GalaxyBase):

    def __init__(self, snapshot, kind='target'):

        self.is_real = False
        if snapshot.isnap < 0:
            return
        self.is_real = True
        
        self.snap = snapshot
        self.sim = snapshot.sim
        self.par = snapshot.par
        self.kind = kind
        self.verbose = self.par['Verbose']
        
        # Actually load the subhalo data
        self.load_subhalo_properties()
        self.subhalo_ids_for_base_haloes = None # Placeholder, may remove

        self.load_subhalo_particles()
        
    def load_subhalo_properties(self):
        """Load the required properties from the input subhalo catalogue."""
        subhalo_file = self.snap.subhalo_file

        # Only need to load parent/descendant information for prior snapshots
        if self.kind == 'prior':
            with_parents = True
            with_descendants = True
        elif self.kind == 'target':
            with_parents = True
            with_descendants = True
        else:
            print(f"Unexpected subhaloes type '{self.kind}'!")
            set_trace()

        subhalo_data_names = ioi.subhalo_data_names(self.par,
            with_parents=with_parents, with_descendants=with_descendants)
        self.subhalo_data_type = None
        if self.par['Input']['FromNightingale'] and self.kind == 'prior':
            subhalo_data = ion.load_subhalo_catalogue_nightingale(
                subhalo_file, with_descendants=with_descendants)
            self.subhalo_data_type = 'Nightingale'
        elif self.par['InputHaloes']['UseSOAP']:
            subhalo_data = ioi.load_subhalo_catalogue_soap(
                subhalo_file, subhalo_data_names)
            self.subhalo_data_type = 'SOAP'
        else:
            subhalo_data = ioi.load_subhalo_catalogue_hbt(
                subhalo_file, subhalo_data_names)
            self.subhalo_data_type = 'HBT'
        for key in subhalo_data:
            setattr(self, tools.key_to_attribute_name(key), subhalo_data[key])
            
        # Build a list pointing directly to the (top-level) parent of each
        # subhalo, i.e. its central.
        self.n_input_subhaloes = len(self.galaxy_ids)

        # The function below returns the top-level parents, but also stores
        # all intermediate parents
        self.centrals = self.find_top_level_parents()

        # The rest is only relevant to target snapshots, so exit if this is
        # a prior...
        if self.kind == 'prior':
            return
        
        # Assign a fake 'FOF' index to unhosted subhaloes. But first record
        # how many real FOFs there are, i.e. where the fake ones begin
        self.maxfof = np.max(self.fof)

        ind_unhosted = np.nonzero(self.fof < 0)[0]
        print(f"There are {len(ind_unhosted)} unhosted subhaloes!")
        subind_central = np.nonzero(
            self.centrals[ind_unhosted] == ind_unhosted)[0]
        subind_satellite = np.nonzero(
            self.centrals[ind_unhosted] != ind_unhosted)[0]
        self.fof[ind_unhosted[subind_central]] = np.arange(
            self.maxfof+1, self.maxfof+1+len(subind_central))
        self.fof[ind_unhosted[subind_satellite]] = (
            self.fof[self.centrals[ind_unhosted[subind_satellite]]])

        
    def load_subhalo_particles(self):
        """Load the full particle list for subhaloes.

        Particles are loaded from either the original (`base') halo catalogue
        or from Nightingale itself, depending on parameter settings and on
        whether or not this is the target snapshot.
        """
        particle_file = self.snap.subhalo_particle_file
        if self.par['Input']['FromNightingale'] and self.snap.offset < 0:
            self.particle_ids = ion.load_subhalo_particles_nightingale(
                self.snap.nightingale_property_file,
                self.snap.nightingale_id_file
            )
            
        else:
            self.particle_ids = ioi.load_subhalo_particles_external(
                particle_file, self.subhalo_ids_for_base_haloes)

        # Load the particle IDs from the 'wait list' if we need to and
        # if this is the prior snapshot (offset == -1)
        if self.snap.offset == -1 and self.par['Input']['LoadWaitlist']:
            waitlist_file = self.snap.nightingale_waitlist_file

            # Load waitlist IDs, as 'list-of-lists' per subhalo
            self.waitlist_particle_ids = (
                ion.load_waitlist_particles_nightingale(waitlist_file))
            
    def initialize_progenitor_list(self):
        """Initialise the lookup list for progenitors of each galaxy."""
        max_desc_id = np.max(self.descendant_galaxy_ids)
        edges = np.arange(max_desc_id+1)
        self.progenitors = tools.SplitList(self.descendant_galaxy_ids, edges)
        
    def get_maximum_extent(self, ish):
        """Find the maximum extent of a subhalo"""
        if not self.par['Sources']['Neighbours']:
            return 0
        extent_factor = self.par['Sources']['ExtentFactor']
        return self.radii[ish] * extent_factor

    def get_subhalo_coordinates(self, ish):
        """Retrieve the coordinates of a subhalo."""
        return self.coordinates[ish, :]

    def get_subhalo_velocity(self, ish):
        """Retrieve the velocity of a subhalo."""
        return self.velocities[ish, :]

    def find_parent_particle_ids(self, igal):
        """Find the particle IDs in all parents of a specified galaxy."""
        
        # Initialize the properties at the level of the galaxy itself.
        ish = self.subhalo_from_galaxy(igal)
        if ish is None:
            print(f"Could not find galaxy {igal}!!")
            set_trace()
        depth = self.depth[ish]
        parents = self.parent_list[ish, :] 

        if self.verbose:
            print(f"Finding particles in parents of galaxy {igal} [SH {ish}].")
            print(f"Depth = {depth}")

        ids = np.zeros(0, dtype=np.uint64)
                
        # Loop through its parents (up to level 1 -- excluding central!) and
        # add their particles.
        n_parents = 0
        for ilevel in range(depth-1, 0, -1):
            curr_ish = parents[ilevel]

            # Sometimes, a parent may not itself be a real galaxy... Skip!
            if curr_ish < 0:
                continue
            n_parents += 1
            if self.verbose:
                print(f"   Level {ilevel}: SH={curr_ish}")

            ids_curr = self.find_subhalo_particle_ids(curr_ish)
            ids = np.concatenate((ids, ids_curr))
            print(f"   ... added {len(ids_curr)} particle IDs.")

        if self.verbose:
            print(f"   Found {len(ids)} IDs from {n_parents} parent galaxies.")
        return ids

    def subhalo_from_galaxy(self, igal):
        """Find the subhalo index of a given galaxy ID"""
        if not hasattr(self, 'galaxy_to_subhalo'):
            self.set_up_galaxy_to_subhalo_array()
        try:
            return self.galaxy_to_subhalo[igal]
        except IndexError:
            print(f"WARNING! Attempt to look up invalid galaxy ID {igal}!")
            return None
        
    def fof_from_subhalo_index(self, ish):
        """Get the FOF group for one or more subhalo indices."""
        fof = self.fof[ish]
        flag_fof = np.zeros(len(ish), dtype=np.int8)
        ind_realfof = np.nonzero(fof <= self.maxfof)[0]
        flag_fof[ind_realfof] = 1
        return fof, flag_fof
    
    def galaxy_from_subhalo(self, ish):
        """Find the galaxy ID for a given subhalo index."""
        return self.galaxy_ids[ish]

    def parent_galaxy_of_subhalo(self, ish, return_subhalo=False):
        """Find the parent galaxy (or subhalo) of a subhalo."""
        # Verify that the input is valid
        if not hasattr(ish, "__len__"):
            if ish < 0:
                print("Negative input subhalo!")
                set_trace()
        elif np.count_nonzero(ish < 0) > 0:
            print("Negative input subhalo!")
            set_trace()
            
        parent = self.parent_galaxy_ids[ish]

        # We have the *galaxy* ID, but may want its subhalo index
        if return_subhalo:
            parent = self.subhalo_from_galaxy(parent)
            if parent is None:
                print(f"WARNING/ISSUE: parent {parent} has invalid galaxy ID!")
                set_trace()
            
        return parent
    
    def set_up_galaxy_to_subhalo_array(self):
        """Build the list translating Galaxy IDs to subhalo IDs"""
        max_gal_id = np.max(self.galaxy_ids)
        n_galaxies = len(self.galaxy_ids)
        self.galaxy_to_subhalo = np.zeros(max_gal_id + 1, dtype=int) - 1
        self.galaxy_to_subhalo[self.galaxy_ids] = np.arange(n_galaxies)

    def find_galaxy_particle_ids(self, igal):
        """Find the particle IDs for a specified galaxy."""
        ish = self.subhalo_from_galaxy(igal)
        if ish is None:
            print(f"WARNING! Could not find subhalo for galaxy {igal}!!")
            return np.zeros(0, dtype=np.uint64)
        return self.find_subhalo_particle_ids(ish)

    def find_mergee_particle_ids(self, igal):
        """Find part-IDs in all galaxies merging with a specified galaxy."""
        mergees = self.find_galaxy_mergees(igal)
        ids = np.zeros(0, dtype=np.uint64) - 1
        for mergee_sh in mergees:
            curr_ids = self.find_subhalo_particle_ids(mergee_sh)
            ids = np.concatenate((ids, curr_ids))
        return ids

    def find_galaxy_mergees(self, igal):
        """Find subhaloes that will merge with a galaxy in the next snap."""
        if not hasattr(self, 'progenitors'):
            self.initialize_progenitor_list()
        try:
            mergee_subhaloes = self.progenitors(igal)
        except IndexError:
            set_trace()
        return mergee_subhaloes

    def find_subhalo_particle_ids(self, ish):
        """Find particle IDs for a specified subhalo."""
        if ish >= len(self.particle_ids):
            set_trace()
        return self.particle_ids[ish]

    def find_galaxy_waitlist_ids(self, igal):
        """Find particle IDs that are on the waitlist for a given galaxy."""
        ish = self.subhalo_from_galaxy(igal)
        if ish is None:
            print(f"WARNING! Could not find galaxy {igal}!!")
            set_trace()
        return self.find_waitlist_particle_ids(ish)

    def find_waitlist_particle_ids(self, ish):
        """Retrieve the waitlist particle IDs for a given subhalo.
        
        Note that `self.waitlist_particle_ids` is a list-of-arrays, with
        entry i containing all IDs for subhalo i.
        """

        # The waitlist IDs are grouped by *input* halo index, because they
        # are written before output subhaloes are found. If the snapshot was
        # loaded from input (HBT) this does not matter, but if it is loaded
        # from Nightingale then we have to translate the main internal subhalo
        # index (Nightingale) to input (HBT)
        if self.subhalo_data_type == 'Nightingale':
            ish_input = self.input_halo_indices[ish]
        else:
            ish_input = ish
        return self.waitlist_particle_ids[ish_input]

    def find_top_level_parents(self):
        """Find the top-level parent subhalo for all subhaloes."""

        # If we already read in a full parent list, then this is trivial:
        if hasattr(self, 'parent_list'):
            return self.parent_list[:, 0]

        # Ok, no pre-existing parent list. Initialise one and then do it
        # the hard way.
        self.parent_list = None
        
        # Can't do anything if there are no subhaloes...
        if self.n_input_subhaloes == 0:
            return np.zeros(0, dtype=np.int32)
        
        # Find maximum depth of subhaloes -- need to iterate that many times
        max_depth = np.max(self.depth)
        self.parent_list = np.zeros(
            (self.n_input_subhaloes, max_depth+1), dtype=np.int32) - 1
        
        # Initialise 'parent' subhalo list, starting with haloes themselves
        # We will then update them successively until they point to the
        # top-level parents.
        curr_parent = np.arange(self.n_input_subhaloes, dtype=np.int32)
        self.parent_list[curr_parent, self.depth] = curr_parent
        
        for iit in range(max_depth):
            gal_parent = self.parent_galaxy_ids[curr_parent]

            # We need to process those galaxies that have a valid parent.
            # This excludes ones for which we have reached the top-level
            # parent (central), but also those where an intermediate
            # parent is non-existing.
            ind_process = np.nonzero(gal_parent >= 0)[0]
            print(
                f"Iteration {iit}, processing {len(ind_process)} galaxies...")
            if len(ind_process) == 0:
                print(f"Premature end to central finding in iteration {iit}!")
                break
            curr_parent[ind_process] = (
                self.subhalo_from_galaxy(gal_parent[ind_process]))
            if curr_parent[ind_process] is None:
                print(f"WARNING! Invalid galaxy lookup in parent finding.")
                set_trace()
            self.parent_list[ind_process, self.depth[ind_process]-iit-1] = (
                curr_parent[ind_process])
            
        # Verify that all 'parents' are now centrals
        n_sat = np.count_nonzero(self.parent_galaxy_ids[curr_parent] != -1)
        if n_sat > 0:
            print(f"Uh oh. We have some satellites left!")
            set_trace()

        # Check that all centrals are sensible...
        ind_no_cen = np.nonzero(self.parent_list[:, 0] < 0)[0]
        if len(ind_no_cen) > 0:
            print(f"WARNING!! There are {len(ind_no_cen)} galaxies without "
                  f"a valid central in snapshot {self.snap.isnap}!!")
            subind_in_fof = np.nonzero(self.fof[ind_no_cen] >= 0)[0]
            if len(subind_in_fof) > 0:
                print(f"WARNING!! {len(subind_in_fof)} of them are in a valid "
                      f"FOF!")
                set_trace()

            # Galaxies that don't have a valid parent or FOF are hard to deal
            # with... assume that they are all unhosted centrals.
            subind_no_fof = np.nonzero(self.fof[ind_no_cen] < 0)[0]
            self.parent_list[ind_no_cen[subind_no_fof], 0] = (
                ind_no_cen[subind_no_fof])
                
        return self.parent_list[:, 0]

    def base_to_main_indices(self, ind_base):
        """Find the main subhalo indices for a given base subhalo indices."""
        pass

    def initialise_new_coordinates(self):
        """Initialise the array for updated coordinates.

        We don't want to overwrite the coordinates from the input catalogue,
        because this would introduce a dependence of the unbinding resul
        on the processing order of subhaloes.
        """
        self.new_coordinates = np.zeros((self.n_input_subhaloes, 3)) - 1
        self.m_bound_after_unbinding = np.zeros(self.n_input_subhaloes)

    def register_unbinding_result(self, ish, new_coordinates, bound_mass):
        """Register the updated coordinates from unbinding."""
        self.new_coordinates[ish, :] = new_coordinates
        self.m_bound_after_unbinding[ish] = bound_mass

    def register_new_velocities(self, ish, new_velocities):
        """Register the updated velocities from unbinding."""
        self.new_velocities[ish, :] = new_velocities

    def update_coordinates(self):
        """Update the subhalo coordinates with newly computed ones.

        If we didn't unbind centrals, they will not have a newly computed
        centre. Therefore we check which coordinates should be updated.
        """
        ind_update = np.nonzero(self.new_coordinates[:, 0] >= 0)[0]
        if self.verbose:
            print(f"Updating coordinates for {len(ind_update)} out of "
                  f"{self.n_input_subhaloes} subhaloes.")
        self.coordinates[ind_update, :] = self.new_coordinates[ind_update, :]

        # We don't need the new coordinates anymore, so we can free memory
        del self.new_coordinates

    def get_coordinates(self, shi):
        """Retrieve the (updated) coordinates for input subhalo[es]."""
        return self.coordinates[shi, :]

    def get_velocities(self, shi):
        """Retrieve the (updated) velocities for input subhalo[es]."""
        return self.velocities[shi, :]
        
    def build_particle_memberships(self, particles):
        """Build an input subhalo membership list for all particles."""

        # Set up a subhalo index list, initialised to -1 (not in a subhalo)
        subhalo_indices = np.zeros(particles.n_parts, dtype=np.int32) - 1

        # Go through each subhalo and mark their particles
        for ish in range(self.n_input_subhaloes):

            # Don't need to do anything if there are no particles
            nparts_expected = self.number_of_bound_particles[ish]
            if nparts_expected == 0:
                continue
            
            # Get indices of particles in current subhalo
            curr_ids = self.particle_ids[ish]
            if len(curr_ids) != self.number_of_bound_particles[ish]:
                print("Inconsistent number of IDs!!")
                set_trace()
            curr_inds = particles.ids_to_indices(curr_ids)

            # Mark relevant particles as belonging to current subhalo.
            subhalo_indices[curr_inds] = ish

        return subhalo_indices



class TargetGalaxy(GalaxyBase):

    def __init__(self, subhaloes, ish):
        self.ish = ish
        self.subhaloes = subhaloes
        self.igal = subhaloes.galaxy_from_subhalo(ish)
        self.sim = self.subhaloes.sim
        self.verbose = self.subhaloes.verbose
        self.par = self.sim.par
        self.snap = self.subhaloes.snap
        
        self.r_init = subhaloes.get_subhalo_coordinates(ish)
        self.v_init = subhaloes.get_subhalo_velocity(ish)
        
    def find_source_particles(self):
        """Find the source particles of this galaxy.

        Their indices into the full snapshot particles are found, and then
        their relevant properties extracted into a `GalaxyParticles`
        instance.
        """

        # Convenience pointer to the the full snapshot particle instance
        particles = self.snap.particles

        # Find the source indices (lots of internal heavy lifting)
        print(f"Loading sources... (type={self.subhaloes.subhalo_data_type})")
        source_inds, origins = self.find_source_indices()

        if len(source_inds) == 0: return None
        
        # Initialise the particles instance
        source = GalaxyParticles(self)

        # Load all relevant particle info into `source`
        source.set_r(particles.get_property('coordinates', source_inds))
        source.set_v(particles.get_property('velocities', source_inds))
        source.set_u(particles.get_property('internal_energies', source_inds))
        source.set_origins(origins)
        
        # For masses, we need to check whether we want to set passive ones
        # permanently to zero or only consider them as 'unbound' in the first
        # iteration.
        source_m = particles.get_property('masses', source_inds)
        source_m_real = np.array(source_m, copy=True)
        if np.min(source_m_real) == 0: set_trace()
        if self.par['Unbinding']['PassiveIsMassless']:
            source_m[np.abs(origins) > 3] = 0
        source.set_m(source_m)
        source.set_m_real(source_m_real)
        source.set_initial_status()
        
        # Attach variables of further use to self for easy re-use
        self.source = source
        self.source_indices = source_inds
        self.origins = origins

        #if np.count_nonzero(origins == 5) > 0.8 * len(origins):
        if self.ish == 1565:
            with h5.File('HaloTest_1565.hdf5', 'w') as o:
                o['Masses'] = source_m
                o['Coordinates'] = source.r
                o['Velocities'] = source.v
                o['Energies'] = source.u
                o['Origins'] = source.origins
                o['FOF'] = particles.fof[source_inds]

        # It would be a lovely idea to return the result
        return source
        
    def find_source_indices(self):
        """Gather all particle indices that belong to this galaxy's source.

        This involves loading the individual origin categories and then
        unicating them. Each individual origin-load is done through a
        separate sub-function.
        """

        snapshot = self.subhaloes.snap
        particles = snapshot.particles

        subhaloes = self.subhaloes
        prior_subhaloes = self.sim.priorSnap.subhaloes
        pre_prior_subhaloes = self.sim.prePriorSnap.subhaloes

        # Hardcode selection for now
        include_l1 = True
        include_l2 = True
        include_l3 = True
        include_l4 = True
        include_l5 = True
        include_l6 = True
        
        origins = np.zeros(0, dtype=int)
        ids = np.zeros(0, dtype=np.uint64)
        
        # Level 0: particles that were in a parent in prior snapshot
        ids_parents = prior_subhaloes.find_parent_particle_ids(self.igal)
        #origins_parents = np.zeros(len(ids_parents), dtype=np.int8)

        # Level 1: particles that were in the galaxy itself in prior
        if include_l1:
            l1_ids = prior_subhaloes.find_galaxy_particle_ids(self.igal)
            ids = np.concatenate((ids, l1_ids))
            origins = np.concatenate((origins, np.zeros(len(l1_ids)) + 1))
        
        # Level 2: particles that were in a galaxy (in the prior snapshot)
        # that merged with this galaxy by the target snapshot
        if include_l2:
            l2_ids = prior_subhaloes.find_mergee_particle_ids(self.igal)
            ids = np.concatenate((ids, l2_ids))
            origins = np.concatenate((origins, np.zeros(len(l2_ids)) + 2))

        # Level 3: particles that belong to this galaxy in the target snap,
        # according to the input subhalo catalogue
        if include_l3:
            l3_ids = subhaloes.find_subhalo_particle_ids(self.ish)
            ids = np.concatenate((ids, l3_ids))
            origins = np.concatenate((origins, np.zeros(len(l3_ids)) + 3))

        # Level 4: particles that belonged to the galaxy in the pre-prior snap
        if include_l4:
            l4_ids = pre_prior_subhaloes.find_galaxy_particle_ids(self.igal)
            ids = np.concatenate((ids, l4_ids))
            origins = np.concatenate((origins, np.zeros(len(l4_ids)) + 4))

        # Level 5: particles from the last snapshot's 'waiting list'
        if include_l5 and self.par['Input']['LoadWaitlist']:
            l5_ids = prior_subhaloes.find_galaxy_waitlist_ids(self.igal)
            ids = np.concatenate((ids, l5_ids))
            origins = np.concatenate((origins, np.zeros(len(l5_ids)) + 5))

        # Level 6: particles that are within a certain multiple of the
        # (input) subhalo half-mass radius
        if include_l6:
            r_max = subhaloes.get_maximum_extent(self.ish)
            subhalo_cen = subhaloes.get_subhalo_coordinates(self.ish)
            cen_sh = self.subhaloes.centrals[self.ish]

            l6_ids = particles.get_ids_in_sphere(subhalo_cen, r_max, cen_sh)
            ids = np.concatenate((ids, l6_ids))
            origins = np.concatenate((origins, np.zeros(len(l6_ids)) + 6))

            if len(l6_ids) > 0.8 * len(ids):
                print(f"WARNING: {len(l6_ids)} L6!")
                #set_trace()
            
        # Level 6: particles that, in the prior snapshot, belonged to a
        # galaxy that is now within the same extent as Level 5 particles
        #nearby_subhaloes = subhaloes.subhaloes_in_sphere(subhalo_cen, r_max)
        #nearby_galaxies = subhaloes.galaxy_from_subhalo(nearby_subhaloes)
        #l6_ids = prior_subhaloes.find_galaxy_particle_ids(nearby_galaxies)
        #ids = np.concatenate((ids, l6_ids))
        #origins = np.concatenate((origins, np.zeros(len(l6_ids)) + 6))

        # We now have the full list, including duplications. For bookkeeping,
        # note how long that list is
        self.n_source_tot = len(ids)

        # Now reduce the ID list to its unique subset, keeping track of the
        # best origin code for each particle
        ids, origins = self.unicate_ids(ids, origins)

        # What we really want is the indices into the particle list
        inds = self.subhaloes.snap.particles.ids_to_indices(ids)

        # Restrict the selection to only those particles in the same FOF
        # group as the target galaxy
        ind_samefof = np.nonzero(
            particles.fof[inds] == self.subhaloes.fof[self.ish])[0]
        inds = inds[ind_samefof]
        origins = origins[ind_samefof]
        ids = ids[ind_samefof]
        
        # Check which IDs are class 0. We *do not* set them to zero, but
        # to minus their original value, so that we can still set passive
        # particles to massless later.
        ind_0 = np.nonzero(np.isin(ids, ids_parents))[0]
        origins[ind_0] = -origins[ind_0]

        return inds, origins


    def unicate_ids(self, ids_full, origins_full):      # Class: Galaxy
        """Remove duplicates from the internally-loaded particle list."""

        # loc_index contains the indices of the first occurrence of
        # each particle
        ids, loc_index = np.unique(ids_full, return_index=True)
        origins = origins_full[loc_index] 
        n_source_parts = len(ids)

        # Do a consistency check to make sure we've got some left:
        if n_source_parts <= 0:
            if self.verbose:
                print(f"Galaxy {self.igal} has no source particles at all?")
            
            # This situation can actually arise when loading back from
            # Cantor, since galaxies may then have zero particles in their 
            # reference snapshot. Otherwise, this indicates a problem:
            if not par['Input']['FromCantor']:
                set_trace()

        if self.verbose:
            print("Galaxy {:d} -- {:d} unique particles (from {:d})."
                  .format(self.igal, n_source_parts, len(ids_full)))
        
        # Store unique particle number for future use:
        self.n_source_parts = n_source_parts

        return ids, origins
