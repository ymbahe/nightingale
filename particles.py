"""TargetParticles

Started 17 Feb 2025
"""

import numpy as np
from pdb import set_trace

import unbinding

class ParticlesBase:

	def dummy(self):
		pass

class SnapshotParticles(ParticlesBase):

	def __init__(self, snapshot):
		self.snapshot = snapshot
		self.par = snapshot.par

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

class GalaxyParticles(ParticlesBase):

	def __init__(self, galaxy):
		self.num_part = None

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

	def set_u(self, u):
		self.check_number_consistency(u)
		self.u = u.astype(np.float32)

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

