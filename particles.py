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
		self.subhaloes = snapshot.subhaloes

		# Load the relevant particle properties into local attributes
		self.load_properties()

	def load_properties(self):
		"""Load the relevant particle properties for this snapshot."""
		snap_file = self.snapshot.snapshot_file
		membership_file = self.snapshot.subhalo_membership_file
		verbose = self.par['Verbose']

		id_name = par['Input']['Names']['ParticleIDs']
		mass_name = par['Input']['Names']['ParticleMasses']
		pos_name = par['Input']['Names']['ParticleCoordinates']
		vel_name = par['Input']['Names']['ParticleVelocities']
		energy_name = par['Input']['Names']['ParticleInternalEnergies']

		# Load "raw" snapshot information

		ids = np.zeros(0, dtype=int)
		ptypes = np.zeros(0, dtype=np.int8)
		masses = np.zeros(0)
		coordinates = np.zeros((0, 3))
		velocities = np.zeros((0, 3), dtype=np.float32)
		internal_energies = np.zeros(0)

		# Also need to load particles' FOF IDs if we want to load full FOFs.
		if self.par['Input']['LoadFullFOF']:
			fof_ids = np.zeros(0, dtype=int)

		self.n_pt = np.zeros(6, dtype=int)

		with h5.File(snap_file, 'r') as f:
			for ptype in [0, 1, 4, 5]:
				pt_name = f'PartType{ptype}'
				if verbose:
					print(f"Loading particle data for type {ptype}...")

				if verbose:
					print(f"   IDs...")
				curr_ids = f[pt_name + '/' + id_name][...]
				ids = np.concatenate((ids, curr_ids))
				n_pt = len(ids)
				self.n_pt[ptype] = n_pt

				if verbose:
					print(f"   Masses...")
				curr_masses = f[pt_name + '/' + mass_name][...]
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

		# Second part: load membership info...
		membership_name = par['Input']['Names']['MembershipName']
		subhalo_indices = np.zeros(0, dtype=int)
		with h5.File(membership_file, 'r') as f:
			for ptype in [0, 1, 4, 5]:
				pt_name = f'PartType{ptype}'
				if verbose:
					print(f"Loading particle memberships for type {ptype}...")
				curr_bsi = f[pt_name + '/' + membership_name][...]
				if len(curr_bsi) != self.n_pt[ptype]:
					print("Inconsistent length of memberships!")
					set_trace()
				subhalo_indices = np.concatenate((subhalo_indices, curr_bsi))

		# Convert 'base' to 'main' subhalo indices (i.e. HBTplus --> SOAP)
		subhalo_indices = subhaloes.base_to_main_indices(subhalo_indices)

		# Store properties as attributes
		self.ids = ids
		self.ptypes = ptypes
		self.masses = masses
		self.coordinates = coordinates
		self.velocities = velocities
		self.internal_energies = internal_energies
		self.subhalo_indices = subhalo_indices
		if self.par['Input']['LoadFullFOF']:
			self.fof = fof_ids

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

	def initialize_memberships_from_subhaloes():
		"""Initialize particle membership to centrals from input subhaloes."""
		self.subhalo_indices = self.subhalo.centrals[self.subhalo_indices]

	def initialize_memberships_from_fof():
		"""Initialize particle membership to centrals from FOF groups."""
		# Find the central from FOF IDs
		cen_gal_fof = self.subhalo.fof_to_central[self.fof_ids]

		# Special treatment for particles that are in a galaxy but not in a FOF
		ind_gal_nofof = np.nonzero(
			(self.subhalo_indices >= 0) & (subhalo_indices < 0))[0]
		cen_gal_nofof = (
			self.subhalo.centrals[self.subhalo_indices[ind_gal_nofof]])

		# Update internal subhalo_indices list (can only do it now, because we
		# needed the original indices for no-FOF-galaxies)
		self.subhalo_indices = cen_gal_fof
		self.subhalo_indices[ind_gal_nofof] = cen_gal_nofof


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

