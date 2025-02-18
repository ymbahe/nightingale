"""Snapshot class to hold basic snapshot information.

Started 17 Feb 2025.
"""

import numpy as np

class Snapshot:
	"""Hold and process snapshot-level information."""	

	def __init__(self, sim, offset=0):
		self.sim = sim
		self.par = sim.par

		self.isnap = par['Sim']['Snapshot'] + offset
		self.snapshot_file = ioi.form_snapshot_file(par, self.isnap)

	def 