"""Snapshot class to hold basic snapshot information.

Started 17 Feb 2025.
"""

import numpy as np
import ioi
import io

class Snapshot:
	"""Hold and process snapshot-level information."""	

	def __init__(self, sim, offset=0):
		self.sim = sim
		self.par = sim.par

		self.isnap = par['Sim']['Snapshot'] + offset
		self.offset = offset

		self.redshift = sim.redshifts[self.isnap]
		self.aexp = sim.aexps[self.isnap]

		# Compute Hubble parameter for this snapshot.
		self.hubble_z = cosmo.compute_hubble_z(self.redshift)
		if self.par['Check']['NoHubble']:
			self.hubble_z = 0

		# Set input file names
		self.set_input_file_names()

	def set_input_file_names(self):
		"""Form all input file names relevant to this snapshot."""
		self.snapshot_file = ioi.form_snapshot_file(self.par, self.isnap)
		self.subhalo_file = ioi.form_subhalo_file(self.par, self.isnap)
		self.subhalo_particle_file = ioi.form_subhalo_particle_file(
			self.par, self.isnap)
		self.nightingale_property_file = io.form_nightingale_property_file(
			self.par, self.isnap)
		self.nightingale_id_file = io.form_nightingale_id_file(
			self.par, self.isnap)

