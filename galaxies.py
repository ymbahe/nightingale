"""Galaxy population information.

Started 17 Feb 2025.
"""

class GalaxyBase:

	def dummy(self):
		print("Test")

class SnapshotGalaxies(GalaxyBase):

	def __init__(self, snapshot):
		self.snap = snapshot


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

		# Find the source indices (lots of internal heavy lifting)
		source_inds, origins = self.find_source_indices()

		# Initialise the particles instance
		source = GalaxyParticles(self)

		# Load all relevant particle info into `source`
		source.set_r(particles.get_property('r', source_inds))
		source.set_v(particles.get_property('v', source_inds))
		source.set_m(particles.get_property('m', source_inds))
		source.set_u(particles.get_property('u', source_inds))

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
		subhaloes = self.subhaloes
		prior_subhaloes = snapshot.prior_subhaloes
		pre_prior_subhaloes = snapshot.pre_prior_subhaloes
		particles = snapshot.particles

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