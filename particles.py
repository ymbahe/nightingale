"""TargetParticles

Started 17 Feb 2025
"""

import unbinding

class ParticlesBase:

	def dummy(self):
		pass

class SnapshotParticles(ParticlesBase):

	def __init__(self, snapshot):
		self.snapshot = snapshot
		self.par = snapshot.par

class GalaxyParticles(ParticlesBase):

	def __init__(self, galaxy):
		pass

	def unbind(self):
		"""Perform the unbinding procedure on the stored particles."""

		# Set here any non-standard MONK parameters.
		# TODO: import params from parameters
		monk_params = {}

		ind_bound, binding_energies = unbinding.unbind_source(
			self.r, self.v, self.m, self.u, self.initial_status,
			self.galaxy.r_init, self.galaxy.v_init, self.snap.hubble_z,
			monk_params
		)

		# Also need to record unbinding result...
		self.ind_bound = ind_bound

		# Find the coordinates of the most bound particle, to be returned
		ind_mostbound = np.argmin(binding_energies[ind_bound])
		halo_centre_of_potential = self.r[ind_bound[ind_mostbound], :]

		return halo_centre_of_potential

