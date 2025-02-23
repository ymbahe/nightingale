"""Galaxy population information.

Started 17 Feb 2025.
"""

import ioi
import tools

class GalaxyBase:

	def dummy(self):
		print("Test")

class SnapshotGalaxies(GalaxyBase):

	def __init__(self, snapshot, kind='full'):
		self.snap = snapshot
		self.par = snapshot.par
		self.kind = kind

	def load_subhalo_properties(self):
		"""Load the required properties from the input subhalo catalogue."""
		subhalo_file = self.snap.subhalo_file

		# Only need to load parent/descendant information for prior snapshots
		if self.kind == 'prior':
			with_parents = True
			with_descendants = True
		elif self.kind == 'target'
			with_parents = True
			with_descendants = False

		subhalo_data_names = ioi.subhalo_data_names(
			with_parents=with_parents, with_descendants=with_descendants)
		subhalo_data = ioi.load_subhalo_catalogue(
			subhalo_file, subhalo_data_names)

		# TO DO: attach data fields directly to self as attributes

		# Build a list pointing directly to the (top-level) parent of each
		# subhalo, i.e. its central.
		self.centrals = self.find_top_level_parents()

	def load_subhalo_particles(self):
		"""Load the full particle list for subhaloes.

		Particles are loaded from either the original (`base') halo catalogue
		or from Nightingale itself, depending on parameter settings and on
		whether or not this is the target snapshot.
		"""
		if self.par['Input']['FromNightingale'] and self.snap.offset < 0:
			self.particle_ids = ioi.load_subhalo_particles_nightingale(
				self.nightingale_property_file, self.nightingale_id_file)

		else:
			self.particle_ids = ioi.load_subhalo_particles_external(
				self.subhalo_particle_file, self.subhalo_ids_for_base_haloes)


	def initialize_progenitor_list(self):
		"""Initialise the lookup list for progenitors of each galaxy."""
		max_desc_id = np.max(self.descendant_galaxy_ids)
		edges = np.arange(max_desc_id+1, 1)
		self.progenitors = tools.SplitList(self.descendant_galaxy_ids, edges)

	def get_maximum_extent(self, ish):
		"""Find the maximum extent of a subhalo"""
		if not self.par['Sources']['Neighbours']:
			return 0
		extent_factor = self.par['Sources']['ExtentFactor']
		return self.neighbour_radii[ish] * extent_factor

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
		depth = self.depth[ish]
		ids = np.zeros(0, dtype=int)
		curr_gal = igal
		curr_ish = ish
		if self.verbose:
			print(f"Finding particles in parents of galaxy {igal} [SH {ish}].")

		# Loop through its parents (up to level 1 -- excluding central!) and
		# add their particles.
		n_parents = 0
		for ilevel in range(depth, 0, -1):
			n_parents += 1
			if self.verbose:
				print(f"   Level {ilevel}: SH={curr_ish}, GalID={curr_gal}...")
			ids_curr = self.find_galaxy_particle_ids(curr_gal)
			ids = np.concatenate((ids, ids_curr))
				print(f"   ... added {len(ids_curr)} particle IDs.")

			# Now update the galaxy and subhalo to the immediate parent
			curr_gal = self.parent_galaxy_of_subhalo(curr_ish)
			curr_ish = self.subhalo_from_galaxy(curr_gal)

		if self.verbose:
			print(f"   Found {len(ids)} IDs from {n_parents} parent galaxies.")
		return ids

	def subhalo_from_galaxy(self, igal):
		"""Find the subhalo index of a given galaxy ID"""
		if not hasattr(self, 'galaxy_to_subhalo'):
			self.set_up_galaxy_to_subhalo_array()
		return self.galaxy_to_subhalo[igal]

	def set_up_galaxy_to_subhalo_array(self):
		"""Build the list translating Galaxy IDs to subhalo IDs"""
		max_gal_id = np.max(self.galaxy_ids)
		n_galaxies = len(self.galaxy_ids)
		self.galaxy_to_subhalo = np.zeros(max_gal_id + 1, dtype=int) - 1
		self.galaxy_to_subhalo[self.galaxy_ids] = np.arange(n_galaxies)

	def find_galaxy_particle_ids(self, igal):
		"""Find the particle IDs for a specified galaxy."""
		ish = self.subhalo_from_galaxy(igal)
		return self.find_subhalo_particle_ids(ish)

	def find_mergee_particle_ids(self, igal):
		"""Find part-IDs in all galaxies merging with a specified galaxy."""
		mergees = self.find_galaxy_mergees(self, igal)
		ids = np.zeros(0, dtype=int) - 1
		for mergee_sh in mergees:
			curr_ids = self.find_subhalo_particle_ids(mergee_sh)
			ids = np.concatenate((ids, curr_ids))
		return ids

	def find_galaxy_mergees(self, igal):
		"""Find subhaloes that will merge with a galaxy in the next snap."""
		mergee_subhaloes = self.progenitors(igal)
		return mergee_subhaloes

	def find_subhalo_particle_ids(self, ish):
		"""Find particle IDs for a specified subhalo."""
		return self.particle_ids[ish]

	def find_top_level_parents(self):
		"""Find the top-level parent subhalo for all subhaloes."""
		pass

	def base_to_main_indices(self, ind_base):
		"""Find the main subhalo indices for a given base subhalo indices."""
		pass

class TargetGalaxy(GalaxyBase):

	def __init__(self, subhaloes, ish):
		self.ish = ish
		self.all = all_galaxies
		self.subhaloes = subhaloes
		self.igal = subhaloes.get_galaxy_id(ish)

	def find_source_particles(self):
		"""Find the source particles of this galaxy.

		Their indices into the full snapshot particles are found, and then
		their relevant properties extracted into a `GalaxyParticles`
		instance.
		"""

		# Convenience pointer to the the full snapshot particle instance
		particles = self.subhaloes.snap.particles

		# Find the source indices (lots of internal heavy lifting)
		source_inds, origins = self.find_source_indices()

		# Initialise the particles instance
		source = GalaxyParticles(self)

		# Load all relevant particle info into `source`
		source.set_r(particles.get_property('r', source_inds))
		source.set_v(particles.get_property('v', source_inds))
		source.set_u(particles.get_property('u', source_inds))

		# For masses, we need to check whether we want to set passive ones
		# permanently to zero or only consider them as 'unbound' in the first
		# iteration.
		source_m = particles.get_property('m', source_inds)
		if self.par['Unbinding']['PassiveIsMassless']:
			source_m[origins > 2] = 0
		source.set_m(source_m)

		# Attach variables of further use to self for easy re-use
		self.source = source
		self.source_indices = source_inds
		self.origins = origins

	def find_source_indices(self):
		"""Gather all particle indices that belong to this galaxy's source.

		This involves loading the individual origin categories and then
		unicating them. Each individual origin-load is done through a
		separate sub-function.
		"""

		snapshot = self.subhaloes.snap
		particles = snapshot.particles

		subhaloes = self.subhaloes
		prior_subhaloes = sim.priorSnap.subhaloes
		pre_prior_subhaloes = sim.prePriorSnap.subhaloes

		# Level 0: particles that were in a parent in prior snapshot
		ids = prior_subhaloes.find_parent_particle_ids(self.igal)
		origins = np.zeros(len(ids), dtype=np.int8)

		# Level 1: particles that were in the galaxy itself in prior
		l1_ids = prior_subhaloes.find_galaxy_particle_ids(self.igal)
		ids = np.concatenate((ids, l1_ids))
		origins = np.concatenate((origins, np.zeros(len(l1_ids)) + 1))

		# Level 2: particles that were in a galaxy (in the prior snapshot)
		# that merged with this galaxy by the target snapshot
		l2_ids = prior_subhaloes.find_mergee_particle_ids(self.igal)
		ids = np.concatenate((ids, l2_ids))
		origins = np.concatenate((origins, np.zeros(len(l2_ids)) + 2))

		# Level 3: particles that belong to this galaxy in the target snap,
		# according to the input subhalo catalogue
		l3_ids = subhaloes.find_subhalo_particle_ids(self.ish)
		ids = np.concatenate((ids, l3_ids))
		origins = np.concatenate((origins, np.zeros(len(l3_ids)) + 3))

		# Level 4: particles that belonged to the galaxy in the pre-prior snap
		l4_ids = pre_prior_subhaloes.find_galaxy_particle_ids(self.igal)
		ids = np.concatenate((ids, l4_ids))
		origins = np.concatenate((origins, np.zeros(len(l4_ids)) + 4))

		# Level 5: particles that are within a certain multiple of the
		# maximum (input) subhalo extent
		r_max = subhaloes.get_maximum_extent(self.ish)
		subhalo_cen = subhaloes.get_subhalo_coordinates(self.ish)
		l5_ids = particles.get_ids_in_sphere(subhalo_cen, r_max)
		ids = np.concatenate((ids, l5_ids))
		origins = np.concatenate((origins, np.zeros(len(l5_ids)) + 5))

		# Level 6: particles that, in the prior snapshot, belonged to a
		# galaxy that is now within the same extent as Level 5 particles
		nearby_subhaloes = subhaloes.subhaloes_in_sphere(subhalo_cen, r_max)
		nearby_galaxies = subhaloes.galaxy_from_subhalo(nearby_subhaloes)
		l6_ids = prior_subhaloes.find_galaxy_particle_ids(nearby_galaxies)
		ids = np.concatenate((ids, l6_ids))
		origins = np.concatenate((origins, np.zeros(len(l6_ids)) + 6))

		# We now have the full list, including duplications. For bookkeeping,
		# note how long that list is
		self.n_source_tot = len(ids)

		# Now reduce the ID list to its unique subset, keeping track of the
		# best origin code for each particle
		ids, origins = unicate_ids(ids, origins)

		# What we really want is the indices into the particle list
		inds = self.particles.get_id_indices(ids)

		return inds, origins	


	def unicate_ids(self, ids_full, origins_full):      # Class: Galaxy
	    """Remove duplicates from the internally-loaded particle list."""

	    # loc_index contains the indices of the first occurrence of
	    # each particle
	    ids, loc_index = np.unique(ids_full, return_index=True)
	    origins = origins_full[loc_index] 
	    n_source_parts = len(ids_unique)

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
	        print("Galaxy {:d} -- {:d} unique particles."
	              .format(self.igal, n_source_parts))
	    
	    # Store unique particle number for future use:
	    self.n_source_parts = n_source_parts

	    return ids, origins